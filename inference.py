# Version: 1.7.0 (Adds dynamic preprocessing selection)

import os
import json
import traceback

import torch
import torchvision.transforms.functional as TF # This might already be there for tiling?
import torch.nn.functional as F # <<< ADD THIS LINE or make sure it exists!
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModel
from PIL import Image
import math # For isnan checks maybe

# --- Import Models ---
try:
    from model import PredictorModel # For old pipelines
    from head_model import HeadModel # <<< ADD THIS IMPORT >>>
    from utils import get_embed_params # Keep utils import
except ImportError as e:
    print(f"Error importing model classes or get_embed_params: {e}")
    raise

try:
    import timm
    import timm.data
    TIMM_AVAILABLE = True
except ImportError:
    print("Warning: timm library not found. Cannot use TIMM models for inference.")
    TIMM_AVAILABLE = False

# Define target_len, maybe make it configurable later
TARGET_LEN_INFERENCE = 4096


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
    # v1.1.0: Read base config keys directly
    def __init__(self, model_path: str, config_path: str = None, device: str = "cpu", clip_dtype: torch.dtype = torch.float32):
        self.device = device
        self.clip_dtype = clip_dtype
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
        print(f"DEBUG: Config keys loaded: {list(self.config.keys())}") # Add print

        # --- Store Embed Version (Read directly) ---
        # <<< FIX 1: Read 'embed_ver' from top level >>>
        self.embed_ver = self.config.get("embed_ver") # Can be None if not present
        if self.embed_ver is None: self.embed_ver = "unknown" # Default if None or missing
        print(f"DEBUG BasePipeline: Found embed_ver: {self.embed_ver}")

        # --- Vision Model Setup (Read directly) ---
        # <<< FIX 2: Read 'base_vision_model' from top level >>>
        self.base_vision_model_name = self.config.get("base_vision_model")
        if not self.base_vision_model_name:
            # Now this error is correct if the key is truly missing from the top level
            raise ValueError("Missing 'base_vision_model' in config file.")
        print(f"DEBUG BasePipeline: Found base_vision_model: {self.base_vision_model_name}")

        # --- Initialize vision model attributes ---
        self.vision_model_type = "unknown"; self.vision_model = None; self.hf_processor = None;
        self.timm_transforms = None; self.proc_size = 512;
        self._init_vision_model() # Call the init function

        # --- Preprocessing function selection ---
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

        # --- Hugging Face Model Path ---
        else:
            if model_name.startswith("timm/"): print("Warning: TIMM model specified but 'timm' library not available.")
            self.vision_model_type = "hf"
            print("  Detected Hugging Face model type.")
            try:
                self.hf_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                self.vision_model = AutoModel.from_pretrained(
                    model_name, torch_dtype=self.clip_dtype, trust_remote_code=True
                ).to(self.device).eval()
                print(f"  Loaded HF model: {self.vision_model.__class__.__name__}")
                print(f"  Loaded HF processor: {self.hf_processor.__class__.__name__}")

                # Disable processor's built-in transforms (remains same)
                if hasattr(self.hf_processor, 'image_processor'):
                     image_processor = self.hf_processor.image_processor
                     # This check might be the one failing to disable for BitImageProcessor
                     print(f"  DEBUG Inference: Original processor config: {image_processor}")
                     if hasattr(image_processor, 'do_center_crop'): image_processor.do_center_crop = False
                     else: print("  Warning: Cannot access 'do_center_crop' on image_processor.")
                     if hasattr(image_processor, 'do_normalize'): image_processor.do_normalize = True # Keep normalization ON
                     else: print("  Warning: Cannot access 'do_normalize' on image_processor.")
                     print(f"  DEBUG Inference: Attempted modified processor config: {image_processor}")
                else:
                     print("  Warning: Cannot access image_processor to disable auto-preprocessing.")


                # --- Determine Processor Size (Attempt automatic first) ---
                self.proc_size = 512 # Start with default
                if hasattr(self.hf_processor, 'image_processor') and hasattr(self.hf_processor.image_processor, 'size'):
                    proc_sz = self.hf_processor.image_processor.size
                    if isinstance(proc_sz, dict): self.proc_size = int(proc_sz.get("height", 512))
                    elif isinstance(proc_sz, int): self.proc_size = int(proc_sz)
                # Also check model config directly for size (like in generate_embeddings)
                model_config = getattr(self.vision_model, 'config', None)
                if model_config and hasattr(model_config, 'image_size'):
                     config_size = int(model_config.image_size)
                     if config_size != self.proc_size:
                          print(f"  DEBUG: Overriding proc size ({self.proc_size}) with model config size ({config_size}).")
                          self.proc_size = config_size

                # <<< ADD EXPLICIT OVERRIDE FOR DINOv2 GIANT >>>
                if "dinov2" in model_name.lower() and "giant" in model_name.lower():
                    if self.proc_size != 518:
                        print(f"  INFO: Explicitly setting processor target size to 518 for DINOv2 Giant (was {self.proc_size}).")
                        self.proc_size = 518
                # <<< END OVERRIDE >>>

                print(f"  Determined processor target size (for FitPad/CenterCrop): {self.proc_size}") # This log should now show 518

            except Exception as e:
                print(f"Error initializing Hugging Face vision model {model_name}: {e}")
                raise


    # Version 1.8.4: Correct target_size determination for FitPad
    def _preprocess_images(self, img_list: list[Image.Image]) -> list[Image.Image] | None:
        """Applies the selected manual preprocessing function to a list of PIL images."""
        if not self.preprocess_func: print("ERROR: Preprocessing function not selected."); return None

        processed_imgs = []
        for img_pil in img_list:
            try:
                img_processed = None
                # --- Determine Correct Target Size ---
                target_s = self.proc_size # Default size (e.g., 512 from HF SigLIP)
                # Specific overrides based on model/embed_ver for manual modes
                # Check for DINOv2 Giant specifically
                if "dinov2" in self.base_vision_model_name.lower() and "giant" in self.base_vision_model_name.lower():
                    target_s = 518 # DINOv2 Giant uses 518
                    # print(f"DEBUG _preprocess_images: Using target_size {target_s} for DINOv2 Giant.")
                # Add other overrides if needed (e.g., checking self.embed_ver)
                # elif 'dinov2' in self.base_vision_model_name and '_224' in self.embed_ver: target_s = 224

                # --- Apply Selected Preprocessing ---
                if self.preprocess_func == preprocess_fit_pad:
                    img_processed = self.preprocess_func(img_pil, target_size=target_s)
                elif self.preprocess_func == preprocess_naflex_resize: # NaFlex doesn't use target_s
                    img_processed = self.preprocess_func(img_pil)
                # Add elif for CenterCrop if needed
                # elif self.preprocess_func == preprocess_center_crop:
                #    img_processed = self.preprocess_func(img_pil, target_size=target_s)
                else: # Fallback
                    img_processed = self.preprocess_func(img_pil) # Attempt basic call

                if img_processed: processed_imgs.append(img_processed)
                else: print(f"Warning: Preprocessing returned None.")
            except Exception as e_prep:
                 print(f"Error during {getattr(self.preprocess_func, '__name__', 'unknown preprocessing')} in inference: {e_prep}. Skipping image.")
                 import traceback; traceback.print_exc()
        return processed_imgs if processed_imgs else None

    # Version 1.8.6: Restore TIMM path, correctly handle FB DINOv2, preserve SigLIP
    def get_clip_emb(self, img_list: list[Image.Image]) -> torch.Tensor | None:
        """
        Generates embeddings for a list of PIL images using the loaded vision model.
        Correctly handles TIMM-native, FB DINOv2 (Manual), SigLIP NaFlex, SigLIP Manual.
        Input img_list expected to be RAW PIL images for NaFlex/TIMM-native.
        Input img_list expected to be PREPROCESSED PIL images for Manual modes if called after _preprocess_images.
        """
        if not isinstance(img_list, list): img_list = [img_list]
        if not img_list: print("Error: Empty image list provided."); return None

        final_emb = None
        do_l2_normalize = False

        try:
            # --- Path 1: TIMM Model using TIMM Transforms ---
            if self.vision_model_type == "timm":
                if self.vision_model is None or self.timm_transforms is None: raise ValueError("TIMM model or transforms not initialized.")
                # Expects RAW PIL images in img_list here
                processed_tensors = [self.timm_transforms(img) for img in img_list]
                input_batch = torch.stack(processed_tensors).to(device=self.device, dtype=self.clip_dtype)

                with torch.no_grad():
                    emb = self.vision_model(input_batch) # Pooled features
                # Check if base model name indicates DINOv2 for normalization
                do_l2_normalize = "dinov2" in self.base_vision_model_name.lower()

            # --- Path 2: Hugging Face Models (SigLIP or FB DINOv2) ---
            elif self.vision_model_type == "hf":
                if self.vision_model is None or self.hf_processor is None: raise ValueError("HF model or processor not initialized.")
                is_naflex_mode = "Naflex" in self.embed_ver
                is_siglip_model = "Siglip" in self.vision_model.__class__.__name__
                is_dinov2_model = "Dinov2" in self.vision_model.__class__.__name__ or "dinov2" in self.base_vision_model_name.lower()

                # --- SubPath 2a: SigLIP NaFlex (Processor handles all) ---
                if is_siglip_model and is_naflex_mode:
                    # Expects RAW PIL images in img_list
                    inputs = self.hf_processor(images=img_list, return_tensors="pt", max_num_patches=1024)
                    pixel_values = inputs.get("pixel_values"); attention_mask = inputs.get("pixel_attention_mask"); spatial_shapes = inputs.get("spatial_shapes")
                    if pixel_values is None or attention_mask is None or spatial_shapes is None: raise ValueError("Missing tensors from HF NaFlex processor.")

                    model_call_kwargs = {
                        "pixel_values": pixel_values.to(device=self.device, dtype=self.clip_dtype),
                        "attention_mask": attention_mask.to(device=self.device),
                        "spatial_shapes": torch.tensor(spatial_shapes, dtype=torch.long).to(device=self.device)
                    }
                    # Call SigLIP model
                    with torch.no_grad():
                        vision_model_component = getattr(self.vision_model, 'vision_model', None)
                        if vision_model_component: emb = vision_model_component(**model_call_kwargs).pooler_output
                        elif hasattr(self.vision_model, 'get_image_features'):
                            kwargs_for_get = {k: v for k, v in model_call_kwargs.items() if k in ['pixel_values', 'attention_mask', 'spatial_shapes']}
                            emb = self.vision_model.get_image_features(**kwargs_for_get)
                        else: raise AttributeError("SigLIP Model missing expected methods.")
                    do_l2_normalize = False # SigLIP internal norm

                # --- SubPath 2b: Manual Preprocessing (FB DINOv2 or SigLIP FitPad/CenterCrop) ---
                elif not is_naflex_mode:
                    # Expects PREPROCESSED PIL images in img_list (processed by _preprocess_images)
                    # Use processor ONLY for ToTensor + Normalize
                    inputs = self.hf_processor(images=img_list, return_tensors="pt")
                    pixel_values = inputs.get("pixel_values")
                    if pixel_values is None: raise ValueError("HF Processor didn't return 'pixel_values'.")
                    model_call_kwargs = {"pixel_values": pixel_values.to(device=self.device, dtype=self.clip_dtype)}
                    attention_mask = inputs.get("pixel_attention_mask")
                    if attention_mask is not None: model_call_kwargs["attention_mask"] = attention_mask.to(device=self.device)

                    # Call appropriate model
                    with torch.no_grad():
                        if is_dinov2_model:
                            outputs = self.vision_model(**model_call_kwargs)
                            emb = outputs.last_hidden_state[:, 0] # CLS token
                        elif is_siglip_model:
                             vision_model_component = getattr(self.vision_model, 'vision_model', None)
                             if vision_model_component: emb = vision_model_component(**model_call_kwargs).pooler_output
                             elif hasattr(self.vision_model, 'get_image_features'):
                                  kwargs_for_get = {k: v for k, v in model_call_kwargs.items() if k in ['pixel_values', 'attention_mask', 'spatial_shapes']}
                                  emb = self.vision_model.get_image_features(**kwargs_for_get)
                             else: raise AttributeError("SigLIP Model missing expected methods.")
                        else: # Should not happen if model loaded correctly
                            raise TypeError(f"Model type mismatch in HF manual preproc path: {self.vision_model.__class__.__name__}")

                    # Set normalization flag based on model type
                    do_l2_normalize = is_dinov2_model

                else: # Should not happen: is_naflex_mode is True but model isn't SigLIP?
                    raise ValueError(f"Unsupported NaFlex mode for model type: {self.vision_model.__class__.__name__}")

            else: # Unknown vision_model_type
                raise ValueError(f"Unknown vision_model_type '{self.vision_model_type}'")

            # --- Check if emb was obtained ---
            if emb is None: raise ValueError("Failed to get embedding from model call.")

            # --- Final Normalization & Conversion ---
            if do_l2_normalize:
                norm = torch.linalg.norm(emb.float(), dim=-1, keepdim=True).clamp(min=1e-8)
                emb = emb / norm.to(emb.dtype)
            final_emb = emb.detach().to(device='cpu', dtype=torch.float32)

        except Exception as e:
            print(f"Error during get_clip_emb (v1.8.6): {e}")
            import traceback; traceback.print_exc(); return None

        return final_emb

    # --- v1.8.6: Adjusted comments to reflect where preprocessing happens ---
    def get_clip_emb_tiled(self, raw_pil_image: Image.Image, tiling: bool = False) -> torch.Tensor | None:
        """Generates embeddings, potentially using 5-crop tiling."""
        img_list = []
        is_naflex_mode = "Naflex" in self.embed_ver

        # --- Prepare img_list (PIL images) ---
        if is_naflex_mode:
             # NaFlex always uses single view RAW image
             if tiling: print("DEBUG: Tiling requested with NaFlex, processing single view instead.")
             img_list = [raw_pil_image]
        elif tiling:
             # Manual Modes (FitPad/CenterCrop) Tiling:
             # Apply manual preprocessing FIRST to get base image for cropping
             base_processed_img_list = self._preprocess_images([raw_pil_image]) # Uses correct size (e.g., 518)
             if not base_processed_img_list: print("Error: Preprocessing failed for tiling base."); return None
             base_processed_img = base_processed_img_list[0]

             # Determine crop size (should match final input size)
             crop_size = [self.proc_size, self.proc_size] # self.proc_size should be correct now (e.g., 518)

             # Perform Tiling on the *Preprocessed* Base Image
             base_processed_img_tensor = TF.to_tensor(base_processed_img)
             try:
                  if base_processed_img_tensor.shape[1] < crop_size[0] or base_processed_img_tensor.shape[2] < crop_size[1]:
                        print(f"Warning: Preprocessed image ({base_processed_img_tensor.shape}) smaller than crop size ({crop_size}). Using single view.")
                        img_list = [base_processed_img] # Pass the single PREPROCESSED image
                  else:
                        tiled_crops_tensors = TF.five_crop(base_processed_img_tensor, size=crop_size)
                        # Pass PREPROCESSED crops (as PIL) to get_clip_emb
                        # get_clip_emb for manual modes expects preprocessed PIL
                        img_list = [TF.to_pil_image(crop) for crop in tiled_crops_tensors]
             except Exception as e_tile:
                  print(f"Error during five_crop tiling: {e_tile}. Falling back to single view.")
                  img_list = [base_processed_img] # Fallback to single PREPROCESSED image
        else:
             # Manual Modes (FitPad/CenterCrop) Non-Tiled:
             # Apply manual preprocessing first.
             processed_single_view = self._preprocess_images([raw_pil_image])
             if not processed_single_view: print("Error: Preprocessing failed for single view."); return None
             img_list = processed_single_view # List containing one PREPROCESSED PIL image

        if not all(isinstance(img, Image.Image) for img in img_list): print("Error: img_list invalid."); return None

        # Call get_clip_emb. It expects RAW images for NaFlex/TIMM-native,
        # and PREPROCESSED images for manual modes (FitPad/CenterCrop).
        return self.get_clip_emb(img_list)

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

    # <<< --- NEW METHOD to Load HeadModel --- >>>
    # v1.9.1: Load HeadModel params from top-level config keys
    def _load_head_model(self, head_model_path: str):
        """Loads the HeadModel state dict using config (reading top-level keys)."""
        print(f"DEBUG _load_head_model: Loading head from {os.path.basename(head_model_path)}...")
        if not self.config: raise ValueError("Pipeline configuration not loaded.")

        # --- Get Parameters directly from self.config ---
        # No longer expect head_params_conf dictionary

        # Load necessary params (use self.config.get)
        expected_features = self.config.get("head_features") # <<< Read from top level
        if expected_features is None:
            if self.vision_model and hasattr(self.vision_model, 'config') and hasattr(self.vision_model.config,
                                                                                      'hidden_size'):
                expected_features = self.vision_model.config.hidden_size
                print(f"DEBUG: Inferred head input features from vision model config: {expected_features}")
            else:
                try:
                    expected_features = get_embed_params(self.embed_ver)["features"]
                except Exception as e:
                    raise ValueError(
                        f"Could not determine features for HeadModel from config, vision model, or embed_ver '{self.embed_ver}': {e}")

            # Determine num_classes (remains the same logic, reads from top level)
        num_classes = self.config.get("num_classes", self.config.get("num_labels"))
        if num_classes is None:
             # Try inferring from state dict as last resort (less ideal)
             print("Warning: 'num_classes'/'num_labels' not found in config. Inferring from state dict.")
             try:
                 # NOTE: The key finding logic needs to be robust here, assuming sequential head
                 sd_temp = load_file(head_model_path)
                 head_keys = [k for k in sd_temp if k.startswith("head.")]
                 final_layer_key = None
                 if head_keys:
                     try:
                         max_idx = max(int(k.split('.')[1]) for k in head_keys if k.split('.')[1].isdigit())
                         potential_key = f"head.{max_idx}.bias"
                         if potential_key in sd_temp: final_layer_key = potential_key
                     except:
                         pass
                 if final_layer_key is None: raise KeyError("Could not determine final layer bias key")
                 outputs_in_file = sd_temp[final_layer_key].shape[0]
                 del sd_temp
                 num_classes = outputs_in_file
             except Exception as e_sd:
                 raise ValueError(f"Cannot determine num_classes from config or state dict: {e_sd}")
             num_classes = int(num_classes)

        # --- Load other HeadModel parameters directly from self.config ---
        pooling_strategy = self.config.get('pooling_strategy', 'attn') # <<< Read from top level
        hidden_dim = self.config.get('head_hidden_dim', 1024)        # <<< Read from top level
        num_res_blocks = self.config.get('head_num_res_blocks', 3)     # <<< Read from top level
        dropout_rate = self.config.get('head_dropout_rate', 0.2)      # <<< Read from top level
        output_mode = self.config.get('head_output_mode', 'linear')   # <<< Read from top level
        # Get attn pool params even if pooling is none, for robust init
        attn_pool_heads = self.config.get('attn_pool_heads', 16)       # <<< Read from top level (might be missing if not used)
        attn_pool_dropout = self.config.get('attn_pool_dropout', 0.2) # <<< Read from top level (might be missing if not used)

        # --- Load State Dict ---
        # Adapt _load_model_helper if needed, or load directly here
        if not os.path.isfile(head_model_path):
            raise FileNotFoundError(f"HeadModel file not found: {head_model_path}")
        try:
             sd = load_file(head_model_path)
             # Re-check outputs_in_file here too for safety
             head_keys = [k for k in sd if k.startswith("head.")]
             final_layer_key = None
             if head_keys:
                 try:
                      max_idx = max(int(k.split('.')[1]) for k in head_keys if k.split('.')[1].isdigit())
                      potential_key = f"head.{max_idx}.bias"
                      if potential_key in sd: final_layer_key = potential_key
                 except: pass
             if final_layer_key is None: raise KeyError("Could not determine final layer bias key")
             outputs_in_file = sd[final_layer_key].shape[0]

             if num_classes != outputs_in_file:
                  print(f"Warning: Config num_classes ({num_classes}) != state dict outputs ({outputs_in_file}). Using state dict value.")
                  num_classes = outputs_in_file
        except Exception as e: raise ValueError(f"Error re-checking state dict outputs: {e}")

        # --- Consistency Check ---
        if num_classes != outputs_in_file:
             print(f"Warning: Configured num_classes ({num_classes}) != Outputs in state dict ({outputs_in_file}). Using state dict value.")
             num_classes = outputs_in_file # Prioritize state dict

        # --- Instantiate HeadModel ---
        try:
            model = HeadModel(
                features=expected_features, # Input feature dim
                num_classes=num_classes,    # Verified number of classes
                pooling_strategy=pooling_strategy, # From config ('none' expected)
                hidden_dim=hidden_dim,
                num_res_blocks=num_res_blocks,
                dropout_rate=dropout_rate,
                output_mode=output_mode,
                attn_pool_heads=attn_pool_heads,
                attn_pool_dropout=attn_pool_dropout
            )
        except Exception as e:
            raise TypeError(f"Failed to instantiate HeadModel: {e}") from e

        # --- Load State Dict into Model ---
        model.eval()
        try:
             # Use strict=True unless migrating requires False
             missing_keys, unexpected_keys = model.load_state_dict(sd, strict=True)
             if unexpected_keys: print(f"Warning: Unexpected keys found loading HeadModel state dict: {unexpected_keys}")
             if missing_keys: print(f"Warning: Missing keys loading HeadModel state dict: {missing_keys}")
        except RuntimeError as e:
            print(f"ERROR: State dict mismatch loading into HeadModel for {head_model_path}.")
            raise e

        model.to(self.device)
        print(f"Successfully loaded HeadModel '{os.path.basename(head_model_path)}' with {model.num_classes} outputs (pooling='{model.pooling_strategy}', output_mode='{model.output_mode}').")
        self.model = model # Assign to instance variable (overwrites PredictorModel if loaded before)

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


# ================================================
#        Head Sequence Pipeline
# ================================================
class HeadSequencePipeline(BasePipeline):
    """Pipeline for HeadModels trained on pre-computed sequences with padding/masking."""
    # v1.1.0: Updated for padding/masking inference
    def __init__(self, head_model_path: str, config_path: str = None, device: str = "cpu", clip_dtype: torch.dtype = torch.float32):
        super().__init__(model_path=head_model_path, config_path=config_path, device=device, clip_dtype=clip_dtype)

        self.labels = self.config.get("labels", {})
        if not self.labels:
             num_classes = self.config.get("num_classes", self.config.get("num_labels", 0))
             self.labels = {str(i): str(i) for i in range(num_classes)}

        # <<< --- Get target_len used during training (if saved, else use default) --- >>>
        # Check if 'fixed_len' or similar was saved in config, otherwise use default
        self.target_len = self.config.get("fixed_len", TARGET_LEN_INFERENCE)
        print(f"HeadSequencePipeline: Using target_len={self.target_len} for padding/masking.")

        # Load the HeadModel (expects pooling_strategy='attn' in config now)
        self._load_head_model(head_model_path)
        if not hasattr(self.model, 'pooling_strategy') or self.model.pooling_strategy != 'attn':
             print(f"Warning: Loaded HeadModel pooling is '{getattr(self.model, 'pooling_strategy', 'N/A')}', but padding/masking pipeline expects 'attn'.")

        self.num_labels = self.model.num_classes

        print(f"HeadSequencePipeline (Padding/Masking Mode): Init OK (Labels: {self.labels}, TargetLen: {self.target_len}, Outputs: {self.num_labels})")

    # --- Helper: Extract and Normalize Sequence ---
    # v1.1.0: Renamed and simplified: only extracts and normalizes
    @torch.no_grad()
    def _extract_and_normalize_sequence(self, pil_image: Image.Image) -> torch.Tensor | None:
        """Extracts sequence using vision model and normalizes it."""
        if self.vision_model is None or self.hf_processor is None:
             print("Error: Vision model/processor not initialized."); return None

        # 1. Preprocess Image (Handle >4096 patches resize)
        img_to_process = pil_image
        try:
            original_width, original_height = pil_image.size
            if original_width <= 0 or original_height <= 0: raise ValueError("Invalid image dimensions")

            TARGET_MAX_PATCHES = self.target_len # Use pipeline's target_len
            PATCH_SIZE = 14

            patches_w_initial = math.floor(original_width / PATCH_SIZE)
            patches_h_initial = math.floor(original_height / PATCH_SIZE)
            total_patches_initial = patches_w_initial * patches_h_initial

            if total_patches_initial > TARGET_MAX_PATCHES:
                # Replicate the resizing logic from generate_features
                scale_factor = math.sqrt(TARGET_MAX_PATCHES / total_patches_initial)
                resize_needed = True
                max_iterations = 10
                iterations = 0
                while iterations < max_iterations:
                    target_w = int(original_width * scale_factor + 0.5)
                    target_h = int(original_height * scale_factor + 0.5)
                    if target_w < 1: target_w = 1;
                    if target_h < 1: target_h = 1;
                    new_patches_w = math.floor(target_w / PATCH_SIZE)
                    new_patches_h = math.floor(target_h / PATCH_SIZE)
                    if new_patches_w * new_patches_h <= TARGET_MAX_PATCHES:
                        # print(f"  - Resizing ({original_width}x{original_height}, {total_patches_initial}p) -> ({target_w}x{target_h}, {new_patches_w * new_patches_h}p)")
                        img_to_process = pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
                        resize_needed = False
                        break
                    scale_factor *= 0.995
                    iterations += 1
                if resize_needed: print(
                    f"Warning: Could not find suitable resize. Using original."); img_to_process = pil_image;

        except Exception as e_resize: print(f"Error during inference resize check: {e_resize}"); return None

        # 2. Extract Sequence (last_hidden_state)
        feature_sequence = None
        try:
            with torch.amp.autocast(device_type=self.device, enabled=(self.clip_dtype != torch.float32), dtype=self.clip_dtype):
                 inputs = self.hf_processor(images=[img_to_process], return_tensors="pt").to(self.device)
                 outputs = self.vision_model(**inputs)
                 if hasattr(outputs, 'last_hidden_state'):
                     feature_sequence = outputs.last_hidden_state # Shape [1, N, F]
                 else: raise ValueError("Model output missing 'last_hidden_state'")
        except Exception as e_extract: print(f"Error extracting features: {e_extract}"); return None

        if feature_sequence is None or feature_sequence.ndim != 3 or feature_sequence.shape[0] != 1:
             print(f"Error: Invalid feature sequence shape {feature_sequence.shape if feature_sequence is not None else 'None'}."); return None

        # 3. Normalize Sequence
        try:
             feature_sequence = F.normalize(feature_sequence, p=2, dim=-1) # Normalize [1, N, F] along F dim
             if torch.isnan(feature_sequence).any(): raise ValueError("NaN detected after normalization")
        except Exception as e_norm: print(f"Error during sequence normalization: {e_norm}"); return None

        # <<< --- Return the normalized sequence on GPU --- >>>
        # Keep on GPU, convert to Float32 for subsequent processing
        return feature_sequence.squeeze(0).float() # Shape [N, F], Float32

    # --- Main Call Method ---
    # v1.1.0: Implements padding/masking before calling HeadModel
    def __call__(self, raw_pil_image: Image.Image) -> dict:
        """Processes image, extracts+normalizes sequence, pads/masks, runs HeadModel."""

        # 1. Extract and Normalize Sequence
        # Returns shape [N, F] on GPU, Float32
        norm_sequence = self._extract_and_normalize_sequence(raw_pil_image)

        if norm_sequence is None:
            return {"error": "Failed to extract or normalize sequence"}

        # 2. Pad/Truncate Sequence and Create Mask
        try:
            current_len = norm_sequence.shape[0]
            features_dim = norm_sequence.shape[1]
            # Create tensors on the same device as the sequence
            final_sequence_gpu = torch.zeros((self.target_len, features_dim), dtype=torch.float32, device=norm_sequence.device)
            attention_mask_gpu = torch.zeros(self.target_len, dtype=torch.bool, device=norm_sequence.device)

            if current_len == 0:
                 print("Warning: Empty normalized sequence. Using zero padding.")
            elif current_len > self.target_len:
                 # This shouldn't happen if generation capped length, but handle defensively
                 print(f"Warning: Inference sequence length {current_len} > target {self.target_len}. Truncating.")
                 final_sequence_gpu = norm_sequence[:self.target_len, :]
                 attention_mask_gpu[:] = True # All real tokens
            else: # Pad
                 final_sequence_gpu[:current_len, :] = norm_sequence
                 attention_mask_gpu[:current_len] = True # Mark real tokens

            # Add batch dimension for the model
            final_sequence_batch = final_sequence_gpu.unsqueeze(0) # Shape [1, target_len, F]
            attention_mask_batch = attention_mask_gpu.unsqueeze(0) # Shape [1, target_len]

        except Exception as e_pad:
             print(f"Error during padding/masking: {e_pad}"); return {"error": "Padding/masking failed"}

        # 3. Run Head Model Prediction
        # self.model is the loaded HeadModel, expects Float32 input [B, Seq, F] and mask [B, Seq]
        try:
            with torch.no_grad():
                 # Pass sequence AND mask
                 pred = self.model(final_sequence_batch, attention_mask=attention_mask_batch)
        except Exception as e_pred:
             print(f"Error during HeadModel prediction: {e_pred}"); traceback.print_exc(); return {"error": "Prediction failed"}

        # 4. Format Output (remains the same as before)
        pred_cpu = pred.detach().cpu()
        formatted_output = {}
        try:
            if self.num_labels == 1:
                scalar_value = pred_cpu.item()
                final_score = scalar_value
                if self.model.output_mode == 'linear': final_score = torch.sigmoid(torch.tensor(scalar_value)).item()
                pos_label_name = self.labels.get('1', '1')
                neg_label_name = self.labels.get('0', '0')
                formatted_output[neg_label_name] = float(1.0 - final_score)
                formatted_output[pos_label_name] = float(final_score)
            elif self.num_labels > 1:
                probabilities = pred_cpu.squeeze(0)  # Shape [C]
                if self.model.output_mode == 'linear': probabilities = F.softmax(probabilities, dim=-1)
                for k in range(self.num_labels):
                    label_index_str = str(k)
                    key = self.labels.get(label_index_str, label_index_str)
                    value = probabilities[k].item()
                    formatted_output[key] = float(value)
            else:
                formatted_output = {"error": f"Invalid num_labels: {self.num_labels}"}
        except Exception as e_format:
            print(f"Error formatting prediction: {e_format}"); return {"error": "Formatting failed"}

        return formatted_output


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