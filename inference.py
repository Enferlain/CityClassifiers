# Version: 1.6.0
# Desc: Consistent fit-pad preprocessing, explicit config path for all pipelines,
#       loads attention params from config, cleaner formatting, tiling default false for classifier.

import os
import json
import torch
import torchvision.transforms.functional as F # Keep for potential five_crop use elsewhere
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModel
from PIL import Image
import math # For isnan checks maybe

# Ensure model and utility functions are imported correctly
try:
    from model import PredictorModel
    from utils import get_embed_params
except ImportError as e:
    print(f"Error importing PredictorModel or get_embed_params: {e}")
    print("Ensure model.py and utils.py are in the same directory or accessible.")
    raise

# --- Preprocessing Function ---
# (Ensure this is IDENTICAL to the one in generate_embeddings.py)
def preprocess_fit_pad(img_pil, target_size=512, fill_color=(0, 0, 0)):
    """Resizes image to fit, pads to target size."""
    original_width, original_height = img_pil.size
    target_w, target_h = target_size, target_size
    scale = min(target_w / original_width, target_h / original_height)
    new_w = int(original_width * scale)
    new_h = int(original_height * scale)
    if new_w == 0 or new_h == 0: return None # Avoid error on zero-size image
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_padded = Image.new(img_pil.mode, (target_w, target_h), fill_color)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    img_padded.paste(img_resized, (pad_left, pad_top))
    return img_padded
# --- End Preprocessing ---

# --- Utility: Load Config ---
def _load_config_helper(config_path):
    """Loads full config from JSON file."""
    if not config_path or not os.path.isfile(config_path):
        print(f"DEBUG: Config file not found or not provided: {config_path}")
        return None
    try:
        with open(config_path) as f: config_data = json.load(f)
        print(f"DEBUG: Loaded config from {config_path}")
        return config_data
    except Exception as e:
        print(f"Error reading or processing config file {config_path}: {e}")
        return None
# --- End Utility: Load Config ---

# --- Utility: Load Model State Dict ---
def _load_model_helper(model_path, expected_features, expected_outputs=None):
    """Loads state dict, verifies input features, checks output size."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        sd = load_file(model_path)
        first_layer_key = "up.0.weight" # Assuming first layer of MLP head

        # Verify input feature dimension
        if first_layer_key not in sd:
             # Check if attention exists - the first layer might be attention now
             first_attn_key = "attention.in_proj_weight" # Common key for MHA
             if first_attn_key in sd:
                  # Input to attention should match feature dimension
                  found_features = sd[first_attn_key].shape[1] # Q,K,V weights stacked, dim 1 is embed_dim
                  if found_features != expected_features:
                       print(f"Warning: Model {model_path} attention input dim mismatch!")
                       print(f"  Expected: {expected_features}, Found: {found_features}")
                  # If attention exists, the MLP check below is less critical? Or check MLP input dim.
                  # Let's check the first linear layer *after* attention if possible
                  first_mlp_key = "up.0.weight"
                  if first_mlp_key in sd and sd[first_mlp_key].shape[1] != expected_features:
                      print(f"Warning: Model {model_path} MLP input dim mismatch!")
                      print(f"  Expected: {expected_features}, Found: {sd[first_mlp_key].shape[1]}")

             else:
                  print(f"Warning: Model {model_path} first layer key '{first_layer_key}' not found and no obvious attention key found.")
                  print("  Cannot verify input feature dimension from state dict.")
        elif sd[first_layer_key].shape[1] != expected_features:
             print(f"Warning: Model {model_path} first layer input dimension mismatch!")
             print(f"  Expected input features: {expected_features}")
             print(f"  Found input features: {sd[first_layer_key].shape[1]}")
             print("  Proceeding, but model instantiation might fail.")

        # Verify output dimension
        final_bias_key = "down.10.bias" # Check key based on latest model.py structure
        actual_outputs = None
        if final_bias_key in sd: actual_outputs = sd[final_bias_key].shape[0]
        else: raise KeyError(f"Could not determine outputs from key '{final_bias_key}' in {model_path}.")

        if expected_outputs is not None and actual_outputs != expected_outputs:
            print(f"Warning: Output size mismatch! Expected {expected_outputs}, found {actual_outputs} in {model_path}.")
            # If flexible, return actual_outputs? For now, just warn.

        return sd, actual_outputs
    except Exception as e:
        print(f"Error loading model state dict from {model_path}: {e}")
        raise
# --- End Utility: Load Model State Dict ---


# ================================================
#        Base Pipeline Class
# ================================================
class BasePipeline:
    """Base class for inference pipelines."""
    def __init__(self, model_path: str, config_path: str = None, device: str = "cpu", clip_dtype: torch.dtype = torch.float32):
        self.device = device
        self.clip_dtype = clip_dtype # Dtype for vision model computation
        self.model_path = model_path

        # --- Config Loading ---
        if config_path is None or not os.path.isfile(config_path):
             print("DEBUG: Config path not provided or invalid, attempting to infer...")
             config_path = self._infer_config_path(model_path)
             if config_path is None:
                  raise FileNotFoundError(f"Could not find or infer config file for model {model_path}")
        self.config_path = config_path
        self.config = _load_config_helper(self.config_path)
        if self.config is None: raise ValueError("Failed to load configuration.")
        # --- End Config Loading ---

        # --- Vision Model Setup ---
        self.base_vision_model_name = self.config.get("model", {}).get("base_vision_model")
        if not self.base_vision_model_name:
            # Fallback if missing - should be set by write_config
            print("Warning: 'base_vision_model' not found in config. Defaulting to CLIP.")
            self.base_vision_model_name = "openai/clip-vit-large-patch14-336"

        self.proc = None
        self.clip_model = None
        self.proc_size = 512 # Default, updated in _init_vision_model
        self._init_vision_model()
        # --- End Vision Model Setup ---

        # --- Predictor Head Setup (Loaded in subclasses) ---
        self.model = None
        # --- End Predictor Head Setup ---

    def _infer_config_path(self, model_path: str) -> str | None:
        """Tries to infer the base config path from a checkpoint path."""
        model_base_name = os.path.basename(model_path)
        # Remove .safetensors suffix first
        if model_base_name.endswith(".safetensors"):
             model_base_name = model_base_name[:-len(".safetensors")]

        # Remove common suffixes iteratively
        suffixes_to_remove = ["_best_val", "_efinal"]
        for suffix in suffixes_to_remove:
             if model_base_name.endswith(suffix):
                  model_base_name = model_base_name[:-len(suffix)]
                  break # Stop after removing one suffix type

        # Remove step suffixes like _s28K, _s1M etc.
        if "_s" in model_base_name:
             parts = model_base_name.split("_s")
             # Check if the part after _s looks like a step count (e.g., digits + K/M)
             if len(parts) > 1 and parts[-1] and (parts[-1][0].isdigit() or parts[-1][-1] in ['K', 'M']):
                  model_base_name = parts[0]

        inferred_path = os.path.join(os.path.dirname(model_path), f"{model_base_name}.config.json")
        if os.path.isfile(inferred_path):
            print(f"DEBUG: Inferred config path: {inferred_path}")
            return inferred_path
        else:
            print(f"DEBUG: Could not infer config path from {model_path}, tried {inferred_path}")
            return None

    def _init_vision_model(self):
        """Initializes the vision model and processor, disabling auto-preprocessing."""
        model_name = self.base_vision_model_name
        print(f"Initializing Vision Model: {model_name} on {self.device} with dtype {self.clip_dtype}")
        try:
            # Load processor and model
            self.proc = AutoProcessor.from_pretrained(model_name)
            self.clip_model = AutoModel.from_pretrained(
                model_name, torch_dtype=self.clip_dtype, trust_remote_code=True
            ).to(self.device).eval()
            print(f"  Loaded model: {self.clip_model.__class__.__name__}")
            print(f"  Loaded processor: {self.proc.__class__.__name__}")

            # Disable processor's built-in resize/crop/rescale
            if hasattr(self.proc, 'image_processor'):
                 image_processor = self.proc.image_processor
                 print(f"  DEBUG Inference: Original image processor config: {image_processor}")
                 if hasattr(image_processor, 'do_resize'): image_processor.do_resize = False
                 if hasattr(image_processor, 'do_center_crop'): image_processor.do_center_crop = False
                 if hasattr(image_processor, 'do_rescale'): image_processor.do_rescale = False # Let normalization handle scaling
                 if hasattr(image_processor, 'do_normalize'): image_processor.do_normalize = True # Keep normalization
                 print(f"  DEBUG Inference: Modified image processor config: {image_processor}")
            else:
                 print("  Warning: Cannot access image_processor to disable auto-preprocessing.")

            # Determine model's expected input size
            self.proc_size = 512 # Default
            if hasattr(self.proc, 'image_processor') and hasattr(self.proc.image_processor, 'size'):
                 proc_sz = self.proc.image_processor.size
                 if isinstance(proc_sz, dict): self.proc_size = int(proc_sz.get("height", 512)) # Use height or crop size
                 elif isinstance(proc_sz, int): self.proc_size = int(proc_sz)
            # print(f"  Determined processor target size: {self.proc_size}") # Already printed below essentially

        except Exception as e:
            print(f"Error initializing vision model {model_name}: {e}")
            raise

    def _preprocess_images(self, img_list: list[Image.Image]) -> list[Image.Image] | None:
        """Applies fit-and-pad preprocessing to a list of PIL images."""
        processed_imgs = []
        for img_pil in img_list:
            try:
                 img_padded = preprocess_fit_pad(img_pil, target_size=self.proc_size)
                 if img_padded: processed_imgs.append(img_padded)
                 else: print("Warning: Preprocessing returned None for an image.")
            except Exception as e_pad:
                 print(f"Error during fit-and-pad in inference: {e_pad}. Skipping image.")
        return processed_imgs if processed_imgs else None

    def get_clip_emb(self, img_list: list[Image.Image]) -> torch.Tensor | None:
        """Applies fit-and-pad, uses processor for normalization, gets embeddings, and L2 normalizes."""
        if not isinstance(img_list, list): img_list = [img_list]

        # Preprocess: Fit and Pad
        processed_imgs = self._preprocess_images(img_list)
        if not processed_imgs:
            print("Error: No images left after preprocessing.")
            return None

        # Process with Hugging Face Processor (Normalization + Tensor Conversion)
        try:
            inputs = self.proc(images=processed_imgs, return_tensors="pt")
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None: raise ValueError("Processor did not return 'pixel_values'.")
            # Move to target device and set dtype for vision model computation
            pixel_values = pixel_values.to(device=self.device, dtype=self.clip_dtype)
        except Exception as e:
            print(f"Error processing padded images with HF processor: {e}")
            return None

        # Get Embeddings from Vision Model
        with torch.no_grad():
            try:
                if hasattr(self.clip_model, 'get_image_features'):
                     emb = self.clip_model.get_image_features(pixel_values=pixel_values)
                else: # Fallback
                     print("Warning: Vision model lacks get_image_features. Using forward pass.")
                     outputs = self.clip_model(pixel_values=pixel_values)
                     if hasattr(outputs, 'image_embeds'): emb = outputs.image_embeds
                     elif hasattr(outputs, 'pooler_output'): emb = outputs.pooler_output
                     else: emb = outputs.last_hidden_state.mean(dim=1)
            except Exception as e_fwd:
                 print(f"Error during vision model forward pass: {e_fwd}")
                 return None

        # Post-process: Detach, CPU, FP32 for MLP head, Normalize
        final_emb = emb.detach().to(device='cpu', dtype=torch.float32)
        if final_emb.numel() == 0: return None # Handle empty tensor case

        try:
            if final_emb.ndim >= 1: # Ensure tensor is not empty/scalar
                norm_dim = 1 if final_emb.ndim == 2 else 0
                emb_norm = torch.linalg.norm(final_emb, dim=norm_dim, keepdim=True)
                # Prevent division by zero
                safe_norm = emb_norm.clamp(min=1e-8)
                final_emb = final_emb / safe_norm
        except Exception as e_norm:
             print(f"Error during L2 normalization: {e_norm}")
             return None # Or return unnormalized? Normalization failure is concerning.

        return final_emb

    def get_clip_emb_tiled(self, raw_pil_image: Image.Image, tiling: bool = False) -> torch.Tensor | None:
        """Generates embeddings, potentially using 5-crop tiling."""
        # For fit-pad models, tiling is discouraged, default should be False in pipeline __call__.
        # If tiling is forced True, this will process 5 padded versions of the *same* image.
        img_list = []
        if tiling:
            print("DEBUG: Tiling requested. Generating 5 padded views.")
            # Apply fit-pad first to get the base 512x512 padded image
            base_padded_img = self._preprocess_images([raw_pil_image])
            if not base_padded_img: return None # Preprocessing failed
            # Create 5 identical copies (no actual cropping needed/useful here)
            img_list = [base_padded_img[0]] * 5
        else:
            # Preprocess the single image (fit-pad happens in get_clip_emb)
             img_list = [raw_pil_image]

        # Get embeddings for the single image or the 5 identical padded images
        return self.get_clip_emb(img_list)

    def _load_predictor_head(self, required_outputs: int | None = None):
        """Loads the MLP head model state dict using config."""
        model_conf = self.config.get("model", {})
        embed_ver = model_conf.get("embed_ver", "CLIP") # Get from loaded config
        model_params_conf = self.config.get("model_params", {})
        expected_features = model_params_conf.get("features")
        expected_hidden = model_params_conf.get("hidden")
        config_outputs = model_params_conf.get("outputs")

        # Load parameters needed by PredictorModel constructor
        try:
             # Use config values directly if available, otherwise fallback to get_embed_params
             if not expected_features: expected_features = get_embed_params(embed_ver)["features"]
             if not expected_hidden: expected_hidden = get_embed_params(embed_ver)["hidden"]
             # Get attention params from model config section
             num_attn_heads = model_conf.get("num_attn_heads", 8) # Default if missing
             attn_dropout = model_conf.get("attn_dropout", 0.1) # Default if missing
        except Exception as e:
             print(f"Error getting model parameters for embed_ver '{embed_ver}': {e}")
             raise ValueError("Could not determine model parameters from config or defaults.")

        # Determine expected output count
        outputs_to_check = required_outputs if required_outputs is not None else config_outputs
        if outputs_to_check is not None: outputs_to_check = int(outputs_to_check)

        # Load state dict and get actual output count from file
        sd, outputs_in_file = _load_model_helper(self.model_path, expected_features, outputs_to_check)

        # Use required_outputs if provided, else use outputs from file/config
        final_outputs = required_outputs if required_outputs is not None else \
                        (outputs_to_check if outputs_to_check is not None else outputs_in_file)

        if final_outputs is None or final_outputs <= 0:
             raise ValueError(f"Could not determine valid number of outputs for model {self.model_path}")

        # Instantiate model
        try:
            model = PredictorModel(
                features=expected_features,
                outputs=final_outputs,
                hidden=expected_hidden,
                num_attn_heads=num_attn_heads,
                attn_dropout=attn_dropout
            )
            print(f"DEBUG: Instantiating PredictorModel(features={expected_features}, hidden={expected_hidden}, outputs={final_outputs}, heads={num_attn_heads}, attn_drop={attn_dropout})")
        except Exception as e:
            raise TypeError(f"Failed to instantiate PredictorModel: {e}") from e

        # Load state dict
        model.eval()
        try: model.load_state_dict(sd, strict=True)
        except RuntimeError as e: print(f"ERROR: State dict mismatch in {self.model_path}."); raise e

        model.to(self.device) # Move predictor head to device
        print(f"Successfully loaded predictor head '{os.path.basename(self.model_path)}' with {model.outputs} outputs.")
        self.model = model # Assign to instance variable

    def get_model_pred_generic(self, emb: torch.Tensor) -> torch.Tensor:
        """Runs prediction using the loaded predictor head."""
        if self.model is None: raise ValueError("Predictor head model not loaded.")
        if emb is None: raise ValueError("Input embedding is None.")
        # Model head expects FP32 on its device
        with torch.no_grad():
             # Move embedding to the correct device right before prediction
             pred = self.model(emb.to(self.device, dtype=torch.float32))
        # Return predictions on CPU
        return pred.detach().cpu()

# ================================================
#        Aesthetics Pipeline (Scorer)
# ================================================
class CityAestheticsPipeline(BasePipeline):
    """Pipeline for single-output score prediction models."""
    # v1.1: Added explicit config_path handling
    def __init__(self, model_path: str, config_path: str = None, device: str = "cpu", clip_dtype: torch.dtype = torch.float32):
        # Base class handles config loading (inferring if config_path is None)
        super().__init__(model_path=model_path, config_path=config_path, device=device, clip_dtype=clip_dtype)
        # Load the predictor head, requiring 1 output
        self._load_predictor_head(required_outputs=1)
        print("CityAestheticsPipeline: Init OK.")

    def __call__(self, raw_pil_image: Image.Image) -> float:
        """Processes a single image and returns a score."""
        # Get non-tiled embedding (fit-pad happens in get_clip_emb)
        emb = self.get_clip_emb_tiled(raw_pil_image, tiling=False)
        if emb is None: return 0.0 # Return default score on error? Or raise?

        pred = self.get_model_pred_generic(emb)
        # Squeeze potential batch dim and convert to float
        return float(pred.squeeze().item())

# ================================================
#        Classifier Pipeline (Single Model)
# ================================================
class CityClassifierPipeline(BasePipeline):
    """Pipeline for multi-class classification models."""
    # v1.2: Added explicit config_path, default tiling=False
    def __init__(self, model_path: str, config_path: str = None, device: str = "cpu", clip_dtype: torch.dtype = torch.float32):
        # Base class handles config loading
        super().__init__(model_path=model_path, config_path=config_path, device=device, clip_dtype=clip_dtype)

        # Load labels from config
        self.labels = self.config.get("labels", {})

        # Load the predictor head, output count inferred from config/state_dict
        self._load_predictor_head(required_outputs=None)
        self.num_labels = self.model.outputs # Get actual number of outputs

        # Populate default labels if needed (based on actual outputs)
        if not self.labels:
             self.labels = {str(i): str(i) for i in range(self.num_labels)}
             print(f"DEBUG: Using default numeric labels (0 to {self.num_labels-1})")
        elif len(self.labels) != self.num_labels:
             print(f"Warning: Config labels count ({len(self.labels)}) != model outputs ({self.num_labels}). Check config.")
             # Optionally reconcile or prioritize model outputs? For now, just warn.

        print(f"CityClassifierPipeline: Init OK (Labels: {self.labels})")

    # v1.6: Default tiling=False, passes args to format_pred
    def __call__(self, raw_pil_image: Image.Image, default: bool = True, tiling: bool = False, tile_strat: str = "mean") -> dict:
        """Processes image, returns dict of label probabilities."""
        # Get embedding (tiling flag passed, but default is False)
        emb = self.get_clip_emb_tiled(raw_pil_image, tiling=tiling)
        if emb is None: return {"error": "Failed to get embedding"}

        # Get predictions (shape [num_tiles, num_classes])
        pred = self.get_model_pred_generic(emb)

        # Format predictions (handles tiling combination if needed)
        num_tiles_pred = pred.shape[0] if pred.ndim == 2 else 1
        formatted_output = self.format_pred(
            pred,
            labels=self.labels,
            drop_default=(not default),
            tile_strategy=tile_strat if tiling and num_tiles_pred > 1 else "raw"
        )
        return formatted_output

    # v1.1: Corrected median/max/min, final formatting
    def format_pred(self, pred: torch.Tensor, labels: dict, drop_default: bool = False, tile_strategy: str = "mean") -> dict:
        """Formats raw predictions into a dictionary, applying tile strategy."""
        num_classes = pred.shape[-1]
        num_tiles = pred.shape[0] if pred.ndim == 2 else 1

        if num_tiles > 1 and tile_strategy != "raw":
            # Combine tile predictions into a 1D tensor of probabilities
            combined_pred = torch.zeros(num_classes, device=pred.device) # Keep on device for calculation
            for k in range(num_classes):
                tile_scores = pred[:, k] # Shape [num_tiles]
                val = 0.0 # Default value
                try:
                    if   tile_strategy == "mean":   val = torch.mean(tile_scores).item()
                    elif tile_strategy == "median":
                        median_value_tensor = torch.median(tile_scores)
                        val = median_value_tensor.item()
                    elif tile_strategy == "max":    val = torch.max(tile_scores).item()    # max on 1D returns 0-dim tensor
                    elif tile_strategy == "min":    val = torch.min(tile_scores).item()    # min on 1D returns 0-dim tensor
                    else: raise NotImplementedError(f"Invalid strategy '{tile_strategy}'")
                except Exception as e_comb:
                    print(f"Error calculating tile strategy '{tile_strategy}' for class {k}: {e_comb}")
                    # Decide how to handle error - skip class, use mean, return NaN? Using 0.0 for now.
                    val = 0.0
                combined_pred[k] = val # Assign Python float
            pred_to_format = combined_pred.cpu() # Move final combined tensor to CPU
        else:
            # Single tile/embedding or raw output requested
            pred_to_format = pred[0].cpu() # Take first (only) prediction, move to CPU

        # Format into dictionary
        out = {}
        for k in range(num_classes):
            label_index_str = str(k)
            if k == 0 and drop_default: continue
            key = labels.get(label_index_str, label_index_str)
            # Ensure value is float
            value = pred_to_format[k].item() if isinstance(pred_to_format[k], torch.Tensor) else pred_to_format[k]
            out[key] = float(value)
        return out

# ================================================
#        Multi-Model Classifier Pipeline
# ================================================
class CityClassifierMultiModelPipeline(BasePipeline):
    """Pipeline for running multiple classification models on one image."""
    # v1.1: Uses BasePipeline __init__ logic implicitly, needs own _load_predictor_head loop
    def __init__(self, model_paths: list[str], config_paths: list[str] = None, device: str = "cpu", clip_dtype: torch.dtype = torch.float32):

        # Init common things (vision model, processor) using the *first* model's config
        # This assumes all models in the multi list use the SAME vision backbone and embedding type.
        first_config_path = None
        if config_paths and config_paths[0]: first_config_path = config_paths[0]
        super().__init__(model_path=model_paths[0], config_path=first_config_path, device=device, clip_dtype=clip_dtype)

        # Store paths
        self.model_paths = model_paths
        self.config_paths = config_paths if config_paths else [None] * len(model_paths)
        if len(self.model_paths) != len(self.config_paths):
             raise ValueError("Mismatch between number of model paths and config paths.")

        # Load individual predictor heads
        self.models = {}
        self.labels = {}
        print(f"Initializing MultiModel Classifier Pipeline for {len(self.model_paths)} models...")
        for i, m_path in enumerate(self.model_paths):
            if not os.path.isfile(m_path): print(f"Warning: Model path not found: {m_path}"); continue
            name = os.path.splitext(os.path.basename(m_path))[0]
            c_path = self.config_paths[i]

            # Infer config if needed (using BasePipeline's helper)
            if c_path is None or not os.path.isfile(c_path):
                 inferred_c_path = self._infer_config_path(m_path)
                 if inferred_c_path: c_path = inferred_c_path
                 else: print(f"Warning: Could not load or infer config for model: {name}"); continue

            try:
                 # Load this specific model's config for its parameters
                 current_config = _load_config_helper(c_path)
                 if not current_config: raise ValueError("Failed to load model config.")

                 # Use _load_predictor_head logic, but don't assign to self.model
                 model_conf = current_config.get("model", {})
                 embed_ver = model_conf.get("embed_ver", "CLIP")
                 model_params_conf = current_config.get("model_params", {})
                 expected_features = model_params_conf.get("features")
                 expected_hidden = model_params_conf.get("hidden")
                 config_outputs = model_params_conf.get("outputs")
                 if not expected_features: expected_features = get_embed_params(embed_ver)["features"]
                 if not expected_hidden: expected_hidden = get_embed_params(embed_ver)["hidden"]
                 num_attn_heads = model_conf.get("num_attn_heads", 8)
                 attn_dropout = model_conf.get("attn_dropout", 0.1)

                 sd, outputs_in_file = _load_model_helper(m_path, expected_features, config_outputs)
                 final_outputs = config_outputs if config_outputs is not None else outputs_in_file

                 current_model = PredictorModel(features=expected_features, outputs=final_outputs, hidden=expected_hidden, num_attn_heads=num_attn_heads, attn_dropout=attn_dropout)
                 current_model.load_state_dict(sd, strict=True)
                 current_model.to(self.device).eval()

                 self.models[name] = current_model

                 # Store labels
                 current_labels_from_config = current_config.get("labels", {})
                 if not current_labels_from_config: self.labels[name] = {str(j): str(j) for j in range(current_model.outputs)}
                 else: self.labels[name] = current_labels_from_config
                 print(f"  Loaded model: {name} (Outputs: {current_model.outputs}, Labels: {self.labels[name]})")

            except Exception as e:
                print(f"Error loading model/config for {name} from {m_path}: {e}")

        if not self.models: raise ValueError("No valid models loaded for MultiModel pipeline.")
        print("CityClassifier MultiModel: Pipeline init ok")


    # v1.6: Default tiling=False
    def __call__(self, raw_pil_image: Image.Image, default: bool = True, tiling: bool = False, tile_strat: str = "mean") -> list[dict]:
        """Processes image with all models, returns list of results."""
        # Get embedding once (fit-pad happens in get_clip_emb)
        emb = self.get_clip_emb_tiled(raw_pil_image, tiling=tiling)
        if emb is None: return [{"error": "Failed to get embedding"}] * len(self.models)

        out_list = []
        # Loop through each loaded model head
        for name, model_head in self.models.items():
             # Get predictions using the shared embedding
             # Need generic predictor logic here, moving emb inside loop temporarily
             # pred = self.get_model_pred_generic(model_head, emb) # Incorrectly uses self.model
             with torch.no_grad():
                  pred = model_head(emb.to(self.device, dtype=torch.float32)).detach().cpu()

             # Format the prediction (applies tile combination strategy)
             # Use the _format_single_pred helper method
             num_tiles_pred = pred.shape[0] if pred.ndim == 2 else 1
             formatted_pred = self._format_single_pred(
                 pred,
                 labels=self.labels[name],
                 drop_default=(not default),
                 tile_strategy=tile_strat if tiling and num_tiles_pred > 1 else "raw",
                 num_classes=model_head.outputs
             )
             out_list.append(formatted_pred)

        return out_list

    # v1.1: Corrected median/max/min, final formatting
    def _format_single_pred(self, pred: torch.Tensor, labels: dict, drop_default: bool, tile_strategy: str, num_classes: int) -> dict:
        """Helper to format predictions for one model in the multi-pipeline."""
        # (Identical logic to CityClassifierPipeline.format_pred)
        num_tiles = pred.shape[0] if pred.ndim == 2 else 1
        if num_tiles > 1 and tile_strategy != "raw":
            combined_pred = torch.zeros(num_classes, device=pred.device)
            for k in range(num_classes):
                tile_scores = pred[:, k]
                val = 0.0
                try:
                    if   tile_strategy == "mean":   val = torch.mean(tile_scores).item()
                    elif tile_strategy == "median":
                        median_value_tensor = torch.median(tile_scores)
                        val = median_value_tensor.item()
                    elif tile_strategy == "max":    val = torch.max(tile_scores).item()
                    elif tile_strategy == "min":    val = torch.min(tile_scores).item()
                    else: raise NotImplementedError(f"Invalid strategy '{tile_strategy}'")
                except Exception as e_comb:
                    print(f"Error calculating tile strategy '{tile_strategy}' for class {k}: {e_comb}")
                    val = 0.0
                combined_pred[k] = val
            pred_to_format = combined_pred.cpu()
        else:
            pred_to_format = pred[0].cpu()

        out = {}
        for k in range(num_classes):
            label_index_str = str(k)
            if k == 0 and drop_default: continue
            key = labels.get(label_index_str, label_index_str)
            value = pred_to_format[k].item() if isinstance(pred_to_format[k], torch.Tensor) else pred_to_format[k]
            out[key] = float(value)
        return out

# --- get_model_path (Utility) ---
# v1.1: Corrected path joining for local dir
def get_model_path(name: str, repo: str, token: str | bool | None = None, extension: str = "safetensors", local_dir: str = "models", use_hf: bool = True) -> str:
    """Gets model path, checking local folder first, then Hugging Face Hub."""
    fname = f"{name}.{extension}"
    # Look relative to the inference.py script's directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    local_path = os.path.join(script_dir, local_dir, fname)

    if os.path.isfile(local_path):
        print(f"Using local model: '{local_path}'")
        return local_path

    if not use_hf:
        raise FileNotFoundError(f"Local model '{local_path}' not found and Hugging Face download disabled.")

    # HF download logic
    hf_token_arg = None
    if isinstance(token, str) and token: hf_token_arg = token
    elif token is True: hf_token_arg = True # Use env var or cached login
    # token=False or None means no token explicitly passed

    print(f"Local model not found. Downloading from HF Hub: '{repo}/{fname}'")
    try:
        downloaded_path = hf_hub_download(repo_id=repo, filename=fname, token=hf_token_arg)
        return str(downloaded_path) # Ensure it's a string path
    except Exception as e:
        print(f"Error downloading model '{fname}' from repo '{repo}': {e}")
        raise

# --- End get_model_path ---