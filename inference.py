# Version: 1.7.0 (Adds dynamic preprocessing selection)

import os
import json
import torch
import torchvision.transforms.functional as TF # This might already be there for tiling?
import torch.nn.functional as F # <<< ADD THIS LINE or make sure it exists!
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

try:
    import timm
    import timm.data
    TIMM_AVAILABLE = True
except ImportError:
    print("Warning: timm library not found. Cannot use TIMM models for inference.")
    TIMM_AVAILABLE = False


# --- v4.0: Aims for target_patches, ensures dims multiple of patch_size (floor) ---
# Copied from generate_embeddings.py - simplified slightly for inference use
def preprocess_naflex_resize(img_pil, target_patches=1024, patch_size=16):
    """
    Resizes image preserving aspect ratio to have close to target_patches,
    ensuring dimensions are multiples of patch_size by flooring.
    (Inference version - doesn't return patch count)
    """
    original_width, original_height = img_pil.size
    if original_width <= 0 or original_height <= 0: return None

    aspect_ratio = original_width / original_height

    ideal_patch_w_f = math.sqrt(target_patches * aspect_ratio)
    ideal_patch_h_f = math.sqrt(target_patches / aspect_ratio)

    ideal_width_f = ideal_patch_w_f * patch_size
    ideal_height_f = ideal_patch_h_f * patch_size

    new_width = math.floor(ideal_width_f / patch_size) * patch_size
    new_height = math.floor(ideal_height_f / patch_size) * patch_size

    if new_width == 0: new_width = patch_size
    if new_height == 0: new_height = patch_size

    # Calculate resulting patches (for debugging/verification)
    num_patches_w = new_width // patch_size
    num_patches_h = new_height // patch_size
    total_patches = num_patches_w * num_patches_h
    # print(f"  DEBUG Inf NaflexResize(v4): Original: {original_width}x{original_height}, TargetPatches: {target_patches}, New: {new_width}x{new_height}, Patches: {total_patches} ({num_patches_w}x{num_patches_h})")

    if total_patches > target_patches: # Should not happen with floor, but good sanity check
        print(f"  ERROR Inf: Calculated patches ({total_patches}) exceed target ({target_patches})!")
        return None # Return None if calculation seems wrong

    img_resized = img_pil.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)
    return img_resized

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

        # --- Verify input feature dimension (keep existing logic) ---
        # Check attention first, then MLP input
        first_layer_key_mlp = "initial_proj.weight" # New first linear layer
        first_attn_key = "attention.in_proj_weight" # Attention key
        found_features_from_key = None

        if first_attn_key in sd:
             # Input to attention should match feature dimension
             found_features_from_key = sd[first_attn_key].shape[1]
             if found_features_from_key != expected_features:
                  print(f"Warning: Model {model_path} attention input dim mismatch! Expected: {expected_features}, Found: {found_features_from_key}")
             # Check the MLP input dimension too for safety
             if first_layer_key_mlp in sd and sd[first_layer_key_mlp].shape[1] != expected_features:
                  print(f"Warning: Model {model_path} initial_proj input dim mismatch! Expected: {expected_features}, Found: {sd[first_layer_key_mlp].shape[1]}")

        elif first_layer_key_mlp in sd:
             # No attention, check input to initial_proj
             found_features_from_key = sd[first_layer_key_mlp].shape[1]
             if found_features_from_key != expected_features:
                  print(f"Warning: Model {model_path} initial_proj input dim mismatch! Expected: {expected_features}, Found: {found_features_from_key}")
        else:
             print(f"Warning: Cannot find expected first layer keys ('{first_attn_key}' or '{first_layer_key_mlp}') in {model_path}. Cannot verify input features.")
        # --- End Input Verification ---

        # --- Verify output dimension (Use NEW key name) ---
        # v1.7.14: Look for 'final_layer.bias' from PredictorModel v2
        final_bias_key = "final_layer.bias"
        actual_outputs = None
        if final_bias_key in sd:
             actual_outputs = sd[final_bias_key].shape[0]
        else:
             # Add fallback for old key? Or just error? Let's error for now.
             print(f"ERROR: Could not find final layer bias key '{final_bias_key}' in state dict {model_path}.")
             print(f"       Available keys start with: {[k for k in sd.keys()[:5]]}...") # Print first few keys for debug
             raise KeyError(f"Could not determine outputs from key '{final_bias_key}' in {model_path}.")

        if expected_outputs is not None and actual_outputs != expected_outputs:
            print(f"Warning: Output size mismatch! Expected {expected_outputs}, found {actual_outputs} in {model_path}.")
        # --- End Output Verification ---

        return sd, actual_outputs # Return state dict and actual output count
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

        # --- Store Embed Version ---
        self.embed_ver = self.config.get("model", {}).get("embed_ver", "unknown")
        print(f"DEBUG BasePipeline: Found embed_ver: {self.embed_ver}")
        # --- End Store Embed Version ---

        # --- Vision Model Setup ---
        self.base_vision_model_name = self.config.get("model", {}).get("base_vision_model")
        if not self.base_vision_model_name:
            # Fallback if missing (shouldn't happen with new config)
            raise ValueError("Missing 'base_vision_model' in config file.")

        # <<< Initialize vision model attributes >>>
        self.vision_model_type = "unknown" # To store 'hf' or 'timm'
        self.vision_model = None           # Stores the actual model (HF AutoModel or TIMM model)
        self.hf_processor = None           # Stores HF processor (if HF model)
        self.timm_transforms = None        # Stores TIMM transforms (if TIMM model)
        self.proc_size = 512               # Default/HF processor size
        # <<< End Initialize >>>

        self._init_vision_model() # Call the updated init function
        # --- End Vision Model Setup ---

        # <<< START NEW: Select Preprocessing Function >>>
        self.preprocess_func = None
        if "NaflexResize" in self.embed_ver or "naflex" in self.embed_ver.lower(): # Check if it's a NaFlex embedding
            print("DEBUG BasePipeline: Selecting NaFlex resize preprocessing.")
            self.preprocess_func = preprocess_naflex_resize
        elif "FitPad" in self.embed_ver: # Check if it's FitPad
            print("DEBUG BasePipeline: Selecting FitPad preprocessing.")
            self.preprocess_func = preprocess_fit_pad
        # Add elif for CenterCrop if needed
        # elif "CenterCrop" in self.embed_ver:
        #     print("DEBUG BasePipeline: Selecting CenterCrop preprocessing.")
        #     self.preprocess_func = preprocess_center_crop # Assuming this exists
        else:
            # Default fallback (or raise error)
            print(f"DEBUG BasePipeline: Unknown embed_ver '{self.embed_ver}'. Defaulting to FitPad preprocessing.")
            self.preprocess_func = preprocess_fit_pad
        # <<< END NEW >>>

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
        """Initializes the vision model and processor/transforms based on config."""
        model_name = self.base_vision_model_name
        print(f"Initializing Vision Model: {model_name} on {self.device} with dtype {self.clip_dtype}")

        # --- Check if TIMM Model ---
        if model_name.startswith("timm/") and TIMM_AVAILABLE:
            self.vision_model_type = "timm"
            print("  Detected TIMM model type.")
            try:
                timm_model_name = model_name.split('/', 1)[1] # Get actual model name for timm
                # Load TIMM model without classifier head
                self.vision_model = timm.create_model(
                    timm_model_name,
                    pretrained=True,
                    num_classes=0 # <<< Crucial: Get pooled features
                ).to(self.device).eval()

                # Apply clip_dtype AFTER loading pretrained weights (usually FP32)
                self.vision_model = self.vision_model.to(dtype=self.clip_dtype)

                # Get TIMM transforms
                data_config = timm.data.resolve_model_data_config(self.vision_model)
                self.timm_transforms = timm.data.create_transform(**data_config, is_training=False)
                print(f"  Loaded TIMM model: {timm_model_name}")
                print(f"  Loaded TIMM transforms: {self.timm_transforms}")

                # Store input size if needed (less critical for TIMM as transforms handle it)
                timm_input_size = data_config.get('input_size')
                if timm_input_size: self.proc_size = timm_input_size[-1]

            except Exception as e:
                print(f"Error initializing TIMM vision model {timm_model_name}: {e}")
                raise

        # --- Else: Assume Hugging Face Model ---
        else:
            if model_name.startswith("timm/"):
                 print("Warning: TIMM model specified but 'timm' library not available or failed import.")
            self.vision_model_type = "hf"
            print("  Detected Hugging Face model type.")
            try:
                # Load HF processor and model
                self.hf_processor = AutoProcessor.from_pretrained(model_name)
                self.vision_model = AutoModel.from_pretrained(
                    model_name, torch_dtype=self.clip_dtype, trust_remote_code=True
                ).to(self.device).eval()
                print(f"  Loaded HF model: {self.vision_model.__class__.__name__}")
                print(f"  Loaded HF processor: {self.hf_processor.__class__.__name__}")

                # Disable processor's built-in transforms (Keep As Is)
                if hasattr(self.hf_processor, 'image_processor'):
                     image_processor = self.hf_processor.image_processor
                     print(f"  DEBUG Inference: Original image processor config: {image_processor}")
                     if hasattr(image_processor, 'do_center_crop'): image_processor.do_center_crop = False
                     if hasattr(image_processor, 'do_normalize'): image_processor.do_normalize = True
                     print(f"  DEBUG Inference: Modified image processor config: {image_processor}")
                else:
                     print("  Warning: Cannot access image_processor to disable auto-preprocessing.")

                # Determine processor size (Keep As Is)
                self.proc_size = 512
                if hasattr(self.hf_processor, 'image_processor') and hasattr(self.hf_processor.image_processor, 'size'):
                    proc_sz = self.hf_processor.image_processor.size
                    if isinstance(proc_sz, dict): self.proc_size = int(proc_sz.get("height", 512))
                    elif isinstance(proc_sz, int): self.proc_size = int(proc_sz)
                print(f"  Determined processor target size (for FitPad/CenterCrop): {self.proc_size}")

            except Exception as e:
                print(f"Error initializing Hugging Face vision model {model_name}: {e}")
                raise


    # --- Step 3: Modify _preprocess_images ---
    def _preprocess_images(self, img_list: list[Image.Image]) -> list[Image.Image] | None:
        """Applies the selected preprocessing function to a list of PIL images."""
        if not self.preprocess_func:
             print("ERROR: Preprocessing function not selected in pipeline init.")
             return None

        processed_imgs = []
        for img_pil in img_list:
            try:
                img_processed = None
                # Check which function we stored and call it appropriately
                if self.preprocess_func == preprocess_fit_pad:
                    # FitPad needs target_size from the processor/model
                    img_processed = self.preprocess_func(img_pil, target_size=self.proc_size)
                elif self.preprocess_func == preprocess_naflex_resize:
                    # NaFlex resize (v4) uses defaults for target_patches/patch_size
                    img_processed = self.preprocess_func(img_pil)
                # Add elif for other functions if needed
                # elif self.preprocess_func == preprocess_center_crop:
                #     img_processed = self.preprocess_func(img_pil, target_size=self.proc_size)
                else:
                    # Fallback for safety, though should be set in init
                    print(f"Warning: Unknown preprocess_func {self.preprocess_func.__name__}, trying basic call.")
                    img_processed = self.preprocess_func(img_pil) # Attempt basic call

                if img_processed:
                     processed_imgs.append(img_processed)
                else:
                     print(f"Warning: Preprocessing function {self.preprocess_func.__name__} returned None for an image.")
            except Exception as e_prep:
                 # Include function name in error
                 print(f"Error during {getattr(self.preprocess_func, '__name__', 'unknown preprocessing')} in inference: {e_prep}. Skipping image.")
                 import traceback
                 traceback.print_exc() # Add traceback here for debugging
        return processed_imgs if processed_imgs else None
    # --- End Modify _preprocess_images ---

    # Version 1.8.0: Adds TIMM DINOv2 support, keeps HF logic untouched
    def get_clip_emb(self, img_list: list[Image.Image]) -> torch.Tensor | None:
        """
        Generates embeddings for a list of PIL images using the loaded vision model.
        Handles HF (SigLIP/NaFlex) and TIMM (DINOv2) models.
        """
        if not isinstance(img_list, list): img_list = [img_list]
        if not img_list:
            print("Error: Empty image list provided to get_clip_emb.")
            return None

        final_emb = None # Initialize

        # --- TIMM Model Inference ---
        if self.vision_model_type == "timm":
            if self.vision_model is None or self.timm_transforms is None:
                print("Error: TIMM model or transforms not initialized.")
                return None
            try:
                # Apply TIMM transforms and create batch tensor
                # Handle multiple images in the list
                processed_tensors = [self.timm_transforms(img) for img in img_list]
                input_batch = torch.stack(processed_tensors).to(device=self.device, dtype=self.clip_dtype)

                with torch.no_grad():
                    # Get pooled features directly (since num_classes=0)
                    emb = self.vision_model(input_batch) # Shape: [Batch, Features]

                    # <<< IMPORTANT: L2 Normalize TIMM DINOv2 Embeddings >>>
                    emb_normalized = F.normalize(emb.float(), p=2, dim=-1)

                # Move to CPU, ensure FP32 for predictor head
                final_emb = emb_normalized.detach().to(device='cpu', dtype=torch.float32)

            except Exception as e:
                 print(f"Error during TIMM model inference: {e}")
                 import traceback
                 traceback.print_exc()
                 return None

        # --- Hugging Face Model Inference (Keep Existing Logic) ---
        elif self.vision_model_type == "hf":
            if self.vision_model is None or self.hf_processor is None:
                 print("Error: Hugging Face model or processor not initialized.")
                 return None

            # Check if NaFlex mode based on embed_ver
            is_naflex_mode = "Naflex" in self.embed_ver
            is_siglip2_model = "Siglip2Model" in self.vision_model.__class__.__name__

            model_call_kwargs = {}
            pixel_values = None
            attention_mask = None
            spatial_shapes = None

            try:
                # NaFlex Mode: Use Processor Logic @ 1024
                if is_siglip2_model and is_naflex_mode:
                    inputs = self.hf_processor(images=img_list, return_tensors="pt", max_num_patches=1024)
                    pixel_values = inputs.get("pixel_values")
                    attention_mask = inputs.get("pixel_attention_mask")
                    spatial_shapes = inputs.get("spatial_shapes")
                    if pixel_values is None or attention_mask is None or spatial_shapes is None:
                         raise ValueError("Missing required tensors from HF NaFlex processor.")
                    if pixel_values.shape[1] != 1024 or attention_mask.shape[1] != 1024:
                         raise ValueError(f"HF NaFlex Processor output seq len != 1024 ({pixel_values.shape[1]})")

                    model_call_kwargs = {
                        "pixel_values": pixel_values.to(device=self.device, dtype=self.clip_dtype),
                        "attention_mask": attention_mask.to(device=self.device),
                        "spatial_shapes": torch.tensor(spatial_shapes, dtype=torch.long).to(device=self.device)
                    }

                # FitPad / CenterCrop / Other HF Modes: Manual Preprocessing
                else:
                    processed_imgs = self._preprocess_images(img_list)
                    if not processed_imgs:
                        print("Error: No images left after HF manual preprocessing.")
                        return None
                    inputs = self.hf_processor(images=processed_imgs, return_tensors="pt")
                    pixel_values_from_proc = inputs.get("pixel_values")
                    if pixel_values_from_proc is None: raise ValueError("HF Processor didn't return 'pixel_values'.")

                    pixel_values = pixel_values_from_proc.to(device=self.device, dtype=self.clip_dtype)
                    model_call_kwargs = {"pixel_values": pixel_values}
                    attention_mask_from_processor = inputs.get("pixel_attention_mask")
                    if attention_mask_from_processor is not None:
                        model_call_kwargs["attention_mask"] = attention_mask_from_processor.to(device=self.device)

            except Exception as e:
                print(f"Error processing images with HF processor in inference: {e}")
                import traceback; traceback.print_exc(); return None

            # Get Embeddings from HF Vision Model
            emb = None
            with torch.no_grad():
                try:
                    # <<< Use self.vision_model here (was self.clip_model) >>>
                    vision_model_component = getattr(self.vision_model, 'vision_model', None)
                    if vision_model_component:
                        vision_outputs = vision_model_component(**model_call_kwargs)
                        emb = vision_outputs.pooler_output
                    elif hasattr(self.vision_model, 'get_image_features'):
                        kwargs_for_get = {"pixel_values": model_call_kwargs["pixel_values"]}
                        # Only pass other args if they exist in the call kwargs
                        if "attention_mask" in model_call_kwargs: kwargs_for_get["attention_mask"] = model_call_kwargs["attention_mask"]
                        if "spatial_shapes" in model_call_kwargs: kwargs_for_get["spatial_shapes"] = model_call_kwargs["spatial_shapes"]
                        emb = self.vision_model.get_image_features(**kwargs_for_get)
                    else:
                        # This was the error source before! Now handled by TIMM check above.
                        # If we reach here with an HF model, it *should* have one of these.
                        raise AttributeError("HF Model has neither 'vision_model' nor 'get_image_features'.")

                    if emb is None: raise ValueError("Failed to get embedding from HF model call.")

                except Exception as e_fwd:
                    print(f"Error during HF vision model forward pass in inference: {e_fwd}")
                    import traceback; traceback.print_exc(); return None

                # Post-processing (Normalization already handled by HF models usually)
                # <<< Normalization is handled inside HF get_image_features or vision_model >>>
                # Just ensure correct device and dtype for output
                final_emb = emb.detach().to(device='cpu', dtype=torch.float32)
                # Optional extra normalization just in case? Usually not needed for HF.
                # norm = torch.linalg.norm(final_emb, dim=-1, keepdim=True).clamp(min=1e-8)
                # final_emb = final_emb / norm

        # --- Unknown Model Type ---
        else:
            print(f"Error: Unknown vision_model_type '{self.vision_model_type}' in get_clip_emb.")
            return None

        # Return final embedding (FP32 on CPU)
        return final_emb

    # Version 1.7.4: Corrected five_crop argument name
    def get_clip_emb_tiled(self, raw_pil_image: Image.Image, tiling: bool = False) -> torch.Tensor | None:
        """Generates embeddings, potentially using 5-crop tiling."""
        img_list = []
        # Check if embed_ver indicates NaFlex mode
        is_naflex_mode = "Naflex" in self.embed_ver

        if tiling:
            # If tiling is requested BUT it's NaFlex mode, process single view instead
            if is_naflex_mode:
                 print("DEBUG: Tiling requested with NaFlex, processing single view instead.")
                 img_list = [raw_pil_image]
            else:
                 # Original FitPad tiling logic
                 # We need _preprocess_images for the base padded view if using 5-crop
                 base_padded_img_list = self._preprocess_images([raw_pil_image]) # Returns a list
                 if not base_padded_img_list:
                     print("Error: Preprocessing failed for tiling base image.")
                     return None
                 base_padded_img_tensor = TF.to_tensor(base_padded_img_list[0])

                 # Use TF.five_crop - needs tensor input, returns tuple of tensors
                 # Size for five_crop should be the CLIP input size (self.proc_size)
                 crop_size = [self.proc_size, self.proc_size] # e.g., [512, 512]
                 try:
                      # Ensure image is large enough for the crop size
                      if base_padded_img_tensor.shape[1] < crop_size[0] or base_padded_img_tensor.shape[2] < crop_size[1]:
                            print(f"Warning: Image ({base_padded_img_tensor.shape}) smaller than crop size ({crop_size}). Using original padded image.")
                            img_list = base_padded_img_list
                      else:
                            # <<< CORRECTED ARGUMENT NAME HERE >>>
                            tiled_crops_tensors = TF.five_crop(base_padded_img_tensor, size=crop_size)
                            # Convert tensors back to PIL images for get_clip_emb processing
                            img_list = [TF.to_pil_image(crop) for crop in tiled_crops_tensors]
                            # print(f"DEBUG: Generated {len(img_list)} tiled crops.")
                 except Exception as e_tile:
                      print(f"Error during five_crop tiling: {e_tile}. Falling back to single view.")
                      img_list = base_padded_img_list # Fallback

        else:
             # Process single image (preprocessing happens inside get_clip_emb)
             img_list = [raw_pil_image]

        # Ensure img_list contains PIL images
        if not all(isinstance(img, Image.Image) for img in img_list):
             print("Error: img_list does not contain PIL images before get_clip_emb.")
             return None

        return self.get_clip_emb(img_list) # Calls the updated get_clip_emb

    # --- v1.7.12: Loads PredictorModel v2 using enhanced config ---
    def _load_predictor_head(self, required_outputs: int | None = None):
        """Loads the MLP head model state dict using enhanced config from self.config."""
        print("DEBUG _load_predictor_head: Loading predictor head...")
        if not self.config:
             raise ValueError("Pipeline configuration not loaded.")

        # --- Get Parameters from the NEW 'predictor_params' section ---
        pred_params_conf = self.config.get("predictor_params", {})
        if not pred_params_conf:
             # Try falling back to old 'model_params' for compatibility? Or raise error?
             # Let's try a fallback for now, but warn heavily.
             print("Warning: 'predictor_params' section not found in config. Trying old 'model_params'.")
             pred_params_conf = self.config.get("model_params", {})
             if not pred_params_conf:
                  raise ValueError("Could not find 'predictor_params' or 'model_params' in config.")

        # Load core parameters (use get with defaults matching PredictorModel v2 init)
        expected_features = pred_params_conf.get("features")
        # If features still missing, try inferring from embed_ver as last resort
        if expected_features is None:
             try: expected_features = get_embed_params(self.embed_ver)["features"]
             except Exception as e: raise ValueError(f"Could not determine features from config or embed_ver '{self.embed_ver}': {e}")

        hidden_dim = pred_params_conf.get("hidden_dim")
        if hidden_dim is None: # Fallback to old 'hidden' name?
            hidden_dim = pred_params_conf.get("hidden") # Try old name
            if hidden_dim is None: # Fallback to embed_ver default
                 try: hidden_dim = get_embed_params(self.embed_ver)["hidden"]
                 except Exception: hidden_dim = 1280 # Hard fallback
                 print(f"Warning: 'hidden_dim' not found, using default/inferred: {hidden_dim}")

        # Determine number of output classes
        # Use required_outputs if explicitly provided (e.g., by AestheticsPipeline)
        num_classes = required_outputs
        if num_classes is None: # Otherwise, get from config
             num_classes = pred_params_conf.get("num_classes")
             if num_classes is None: # Fallback to old 'outputs' name?
                  num_classes = pred_params_conf.get("outputs")
                  if num_classes is None: # Final fallback - try to infer from state dict
                       print("Warning: 'num_classes'/'outputs' not found in config. Inferring from state dict (might be unreliable).")
                       # Need to load state dict temporarily to infer - less ideal
                       try:
                            sd_temp, outputs_in_file = _load_model_helper(self.model_path, expected_features, None)
                            num_classes = outputs_in_file
                            del sd_temp
                       except Exception as e_sd:
                            raise ValueError(f"Cannot determine num_classes from config or state dict: {e_sd}")
                  else: print("Warning: Using 'outputs' key for num_classes (old config format?).")
             num_classes = int(num_classes) # Ensure integer

        # Load enhanced architecture parameters
        use_attention = pred_params_conf.get("use_attention", True) # Default True if missing
        num_attn_heads = pred_params_conf.get("num_attn_heads", 8)
        attn_dropout = pred_params_conf.get("attn_dropout", 0.1)
        num_res_blocks = pred_params_conf.get("num_res_blocks", 1)
        dropout_rate = pred_params_conf.get("dropout_rate", 0.1)
        output_mode = pred_params_conf.get("output_mode", 'linear') # Default linear if missing

        # --- Load State Dict ---
        # We need the *actual* number of classes in the file for verification
        # Pass None as expected_outputs to _load_model_helper, it will return the count from the file
        sd, outputs_in_file = _load_model_helper(self.model_path, expected_features, None)

        # --- Consistency Check ---
        # If required_outputs was specified, make sure it matches the file
        if required_outputs is not None and required_outputs != outputs_in_file:
             print(f"ERROR: Mismatch! Required outputs ({required_outputs}) != Outputs in state dict ({outputs_in_file}).")
             raise ValueError("State dict output count mismatch.")
        # If num_classes was determined from config, check against file
        elif required_outputs is None and num_classes != outputs_in_file:
             print(f"Warning: Configured num_classes ({num_classes}) != Outputs in state dict ({outputs_in_file}). Using state dict value.")
             num_classes = outputs_in_file # Prioritize state dict if different from config

        # --- Instantiate PredictorModel v2 ---
        # Use all the parameters loaded from config
        try:
            model = PredictorModel(
                features=expected_features,
                hidden_dim=hidden_dim,
                num_classes=num_classes, # Use verified number of classes
                use_attention=use_attention,
                num_attn_heads=num_attn_heads,
                attn_dropout=attn_dropout,
                num_res_blocks=num_res_blocks,
                dropout_rate=dropout_rate,
                output_mode=output_mode
            )
            # Initialization log now happens inside PredictorModel.__init__
        except Exception as e:
            raise TypeError(f"Failed to instantiate PredictorModel v2: {e}") from e

        # --- Load State Dict into Model ---
        model.eval()
        try:
             # Use strict=False temporarily ONLY if migrating from old models
             # Otherwise keep strict=True
             missing_keys, unexpected_keys = model.load_state_dict(sd, strict=True)
             if unexpected_keys: print(f"Warning: Unexpected keys found in state dict: {unexpected_keys}")
             if missing_keys: print(f"Warning: Missing keys in state dict: {missing_keys}")
        except RuntimeError as e:
            print(f"ERROR: State dict mismatch loading into PredictorModel v2 for {self.model_path}.")
            raise e

        model.to(self.device) # Move predictor head to device
        print(f"Successfully loaded predictor head '{os.path.basename(self.model_path)}' with {model.num_classes} outputs (output_mode='{model.output_mode}').")
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
        self.num_labels = self.model.num_classes

        # Populate default labels if needed (based on actual outputs)
        if not self.labels:
             self.labels = {str(i): str(i) for i in range(self.num_labels)}
             print(f"DEBUG: Using default numeric labels (0 to {self.num_labels-1})")
        elif len(self.labels) != self.num_labels:
             print(f"Warning: Config labels count ({len(self.labels)}) != model outputs ({self.num_labels}). Check config.")
             # Optionally reconcile or prioritize model outputs? For now, just warn.

        print(f"CityClassifierPipeline: Init OK (Labels: {self.labels}, Num Outputs: {self.num_labels})")

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

    # v1.3: Handle num_classes=1 case
    def format_pred(self, pred: torch.Tensor, labels: dict, drop_default: bool = False, tile_strategy: str = "mean") -> dict:
        """Formats raw predictions into a dictionary, applying tile strategy."""
        # Expects pred to be [Tiles, Classes] or [Classes] after tiling combination/no tiling

        num_classes_model = pred.shape[-1] if pred.ndim > 0 else 1 # Get num outputs from tensor
        num_tiles = pred.shape[0] if pred.ndim >= 2 else 1

        if num_tiles > 1 and tile_strategy != "raw":
            # Combine tile predictions
            combined_pred = torch.zeros(num_classes_model, device='cpu') # Calc on CPU
            for k in range(num_classes_model):
                tile_scores = pred[:, k].cpu() # Move tiles to CPU
                val = 0.0
                try:
                    if   tile_strategy == "mean":   val = torch.mean(tile_scores).item()
                    elif tile_strategy == "median": val = torch.median(tile_scores).item() # Get median value
                    elif tile_strategy == "max":    val = torch.max(tile_scores).item()
                    elif tile_strategy == "min":    val = torch.min(tile_scores).item()
                    else: raise NotImplementedError(f"Invalid strategy '{tile_strategy}'")
                except Exception as e_comb:
                    print(f"Error calculating tile strategy '{tile_strategy}' for class {k}: {e_comb}")
                    val = 0.0
                combined_pred[k] = val
            pred_to_format = combined_pred # Shape [num_classes_model] on CPU
        else:
            # Single tile/embedding or raw output requested
            # Ensure it's at least 1D before potential squeeze, move to CPU
            pred_to_format = pred.cpu().squeeze(0) if pred.ndim > 1 else pred.cpu() # Shape [num_classes_model] or [] if scalar output initially?

        # --- Handle Output Formatting Based on num_classes_model ---
        out = {}
        if num_classes_model == 1:
             # Single output (likely BCE or Scorer)
             # Get the scalar value
             scalar_value = pred_to_format.item() if pred_to_format.ndim == 0 else pred_to_format[0].item()

             # Decide which label this corresponds to. Conventionally, single output
             # represents the probability/score of the "positive" class (index 1).
             # Need to handle potential sigmoid/tanh scaling based on output_mode?
             # Assuming linear output for BCE here, needs sigmoid for probability.
             # If model output_mode was 'sigmoid' or 'tanh_scaled', value is already 0-1.
             # If model output_mode was 'linear', apply sigmoid here for probability display.
             # Let's assume the loaded pipeline knows the model's output_mode? Access self.model.output_mode?
             final_score = scalar_value
             if hasattr(self, 'model') and self.model.output_mode == 'linear':
                  # print("DEBUG format_pred: Applying sigmoid to linear output for probability display.")
                  final_score = torch.sigmoid(torch.tensor(scalar_value)).item()
             elif hasattr(self, 'model') and self.model.output_mode == 'tanh_scaled':
                  final_score = scalar_value # Already scaled 0-1
             # Else assume it's already a probability if mode was sigmoid

             # Get label name for the positive class (index '1')
             # Use self.labels if available, fallback to '1'
             positive_label_key = '1'
             positive_label_name = self.labels.get(positive_label_key, positive_label_key)

             # Get label name for the negative class (index '0')
             negative_label_key = '0'
             negative_label_name = self.labels.get(negative_label_key, negative_label_key)

             # Output both probabilities (P(1) and P(0) = 1 - P(1))
             if not (drop_default and negative_label_name == self.labels.get('0')): # Check if default should be dropped
                 out[negative_label_name] = float(1.0 - final_score)
             out[positive_label_name] = float(final_score)

        elif num_classes_model > 1:
             # Multi-output (likely CE/Focal)
             # Apply softmax here if output was linear for probability display?
             probabilities = pred_to_format # Assume already probabilities if output_mode was softmax
             if hasattr(self, 'model') and self.model.output_mode == 'linear':
                  # print("DEBUG format_pred: Applying softmax to linear output for probability display.")
                  probabilities = F.softmax(pred_to_format, dim=-1)

             for k in range(num_classes_model):
                 label_index_str = str(k)
                 # Handle dropping default class (index 0)
                 if k == 0 and drop_default: continue
                 # Get label name from loaded labels (might differ from self.labels if multi-model)
                 key = labels.get(label_index_str, label_index_str)
                 value = probabilities[k].item() # Get probability for class k
                 out[key] = float(value)
        else:
             print(f"ERROR: Invalid num_classes_model ({num_classes_model}) in format_pred.")

        return out

# ================================================
#        Multi-Model Classifier Pipeline
# ================================================
class CityClassifierMultiModelPipeline(BasePipeline):
    """Pipeline for running multiple classification models on one image."""
    # v1.2: Updated to load PredictorModel v2 params
    def __init__(self, model_paths: list[str], config_paths: list[str] = None, device: str = "cpu", clip_dtype: torch.dtype = torch.float32):
        # Init common things (vision model, processor) using the *first* model's config
        # Assumes all models use the SAME vision backbone.
        first_config_path = None
        if config_paths and config_paths[0]: first_config_path = config_paths[0]
        # BasePipeline.__init__ handles loading first config, vision model, selecting preprocess_func
        super().__init__(model_path=model_paths[0], config_path=first_config_path, device=device, clip_dtype=clip_dtype)

        # Store paths
        self.model_paths = model_paths
        self.config_paths = config_paths if config_paths else [None] * len(model_paths)
        if len(self.model_paths) != len(self.config_paths):
             raise ValueError("Mismatch between number of model paths and config paths.")

        # Load individual predictor heads
        self.models = {} # Stores name -> loaded PredictorModel instance
        self.labels = {} # Stores name -> labels dict
        print(f"Initializing MultiModel Classifier Pipeline for {len(self.model_paths)} models...")

        for i, m_path in enumerate(self.model_paths):
            if not os.path.isfile(m_path):
                 print(f"Warning: Model path not found, skipping: {m_path}")
                 continue
            name = os.path.splitext(os.path.basename(m_path))[0]
            c_path = self.config_paths[i]

            # Infer config if needed (using BasePipeline's helper)
            if c_path is None or not os.path.isfile(c_path):
                 inferred_c_path = self._infer_config_path(m_path)
                 if inferred_c_path:
                     c_path = inferred_c_path
                     print(f"DEBUG MultiModel: Inferred config path {c_path} for {name}")
                 else:
                     print(f"Warning: Could not load or infer config for model: {name}. Skipping.")
                     continue

            try:
                 # --- Load THIS model's config ---
                 current_config = _load_config_helper(c_path)
                 if not current_config:
                     raise ValueError(f"Failed to load model config from {c_path}")

                 # --- Get Parameters using logic similar to _load_predictor_head ---
                 current_embed_ver = current_config.get("model", {}).get("embed_ver", self.embed_ver) # Use specific or default

                 pred_params_conf = current_config.get("predictor_params", {})
                 if not pred_params_conf: # Fallback for older configs
                     print(f"Warning: 'predictor_params' not found in {c_path}. Trying 'model_params'.")
                     pred_params_conf = current_config.get("model_params", {})
                     if not pred_params_conf: raise ValueError("Missing predictor/model params.")

                 # Load features, hidden_dim (with fallbacks)
                 expected_features = pred_params_conf.get("features")
                 if expected_features is None: expected_features = get_embed_params(current_embed_ver)["features"]

                 hidden_dim = pred_params_conf.get("hidden_dim", pred_params_conf.get("hidden")) # Try new then old name
                 if hidden_dim is None: hidden_dim = get_embed_params(current_embed_ver)["hidden"]

                 # Load num_classes (infer from state dict as fallback)
                 num_classes = pred_params_conf.get("num_classes", pred_params_conf.get("outputs"))
                 if num_classes is None: # Infer if missing
                      sd_temp, outputs_in_file = _load_model_helper(m_path, expected_features, None)
                      num_classes = outputs_in_file
                      del sd_temp
                 num_classes = int(num_classes)

                 # Load other PredictorModel v2 params (with defaults)
                 use_attention = pred_params_conf.get("use_attention", True)
                 num_attn_heads = pred_params_conf.get("num_attn_heads", 8)
                 attn_dropout = pred_params_conf.get("attn_dropout", 0.1)
                 num_res_blocks = pred_params_conf.get("num_res_blocks", 1)
                 dropout_rate = pred_params_conf.get("dropout_rate", 0.1)
                 output_mode = pred_params_conf.get("output_mode", 'linear')
                 # --- End Get Parameters ---

                 # --- Load State Dict & Verify Outputs ---
                 sd, outputs_in_file = _load_model_helper(m_path, expected_features, None)
                 if num_classes != outputs_in_file:
                      print(f"Warning: Config num_classes ({num_classes}) != state dict outputs ({outputs_in_file}) for {name}. Using state dict value.")
                      num_classes = outputs_in_file
                 # --- End Load State Dict ---

                 # --- Instantiate PredictorModel v2 ---
                 current_model = PredictorModel(
                     features=expected_features,
                     hidden_dim=hidden_dim,
                     num_classes=num_classes,
                     use_attention=use_attention,
                     num_attn_heads=num_attn_heads,
                     attn_dropout=attn_dropout,
                     num_res_blocks=num_res_blocks,
                     dropout_rate=dropout_rate,
                     output_mode=output_mode
                 )
                 # --- End Instantiate ---

                 # --- Load Weights ---
                 current_model.load_state_dict(sd, strict=True) # Keep strict=True
                 current_model.to(self.device).eval()
                 # --- End Load Weights ---

                 self.models[name] = current_model

                 # Store labels from this model's config
                 current_labels_from_config = current_config.get("labels", {})
                 model_outputs = current_model.num_classes
                 if not current_labels_from_config: self.labels[name] = {str(j): str(j) for j in range(model_outputs)}
                 elif len(current_labels_from_config) != model_outputs:
                      print(f"  Warning: Label count mismatch in {c_path} for {name}. Using defaults.")
                      self.labels[name] = {str(j): str(j) for j in range(model_outputs)}
                 else: self.labels[name] = current_labels_from_config
                 print(f"  Loaded model: {name} (Outputs: {model_outputs})") # Use variable here

            except Exception as e:
                print(f"Error loading model/config for {name} from {m_path}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for multi-model issues

        if not self.models:
            raise ValueError("No valid models loaded for MultiModel pipeline.")
        print(f"CityClassifier MultiModel: Pipeline init ok ({len(self.models)} models loaded)")

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