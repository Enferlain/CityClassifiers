# generate_embeddings.py
# Version: 4.2.0 (Added TIMM DINOv2 support)

import os
import torch
# <<< ADD TIMM IMPORT >>>
try:
    import timm
    import timm.data
    TIMM_AVAILABLE = True
except ImportError:
    print("Warning: timm library not found. TIMM models cannot be used.")
    TIMM_AVAILABLE = False
# <<< END ADD >>>
import torchvision.transforms.functional as TF # Renamed from F to avoid conflict
import torch.nn.functional as torch_func
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, AutoModel, Dinov2Model
import argparse
import math
import shutil
import traceback


# --- Constants ---
TARGET_DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_PRECISION_MAP = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
# Defaulting to standard SigLIP here, will be overridden by args
DEFAULT_MODEL_ID = "google/siglip-so400m-patch14-384" # Changed default for clarity

# --- Argument Parsing ---
# v4.2.0: Added dinov2_large_timm_fitpad choice
def parse_gen_args():
    parser = argparse.ArgumentParser(description="Generate vision model embeddings for image folders.")
    parser.add_argument('--image_dir', required=True, help="Directory containing subfolders (e.g., '0', '1') with images.")
    parser.add_argument('--output_dir_root', required=True, help="Root directory to save embeddings (e.g., 'data').")
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_ID, help=f"Vision model name from Hugging Face or TIMM. Default: {DEFAULT_MODEL_ID}")
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'bf16', 'fp16'], help="Precision for model computation (default: fp32).")
    parser.add_argument('--preprocess_mode', type=str, default='fit_pad',
                        choices=[
                            'fit_pad',              # Manual FitPad (SigLIP HF)
                            'center_crop',          # Manual CenterCrop (SigLIP HF)
                            'avg_crop',             # Manual AvgCrop (NYI)
                            'naflex_resize',        # NaFlex HF (Proc Logic @ 1024)
                            'dinov2_large_timm_fitpad',   # DINOv2 TIMM (TIMM Transforms)
                            'dinov2_giant_fb'      # <<< ADDED: DINOv2 Facebook (Manual FitPad)
                        ],
                        help="Image preprocessing method before embedding.")
    parser.add_argument('--resize_factor_avg_crop', type=float, default=2.0,
                        help="Factor to multiply model's native size by for resizing ONLY for avg_crop mode (default: 2.0).")
    parser.add_argument('--output_dir_suffix', type=str, default="", help="Optional suffix for the output directory name.")

    args = parser.parse_args()

    # --- Validation ---
    if args.preprocess_mode == 'dinov2_large_fb_fitpad' and not TIMM_AVAILABLE:
         parser.error("preprocess_mode 'dinov2_large_fb_fitpad' requires 'timm'.")
    if args.preprocess_mode == 'dinov2_large_fb_fitpad' and not args.model_name.startswith('timm/'):
         print(f"Warning: Mode 'dinov2_large_fb_fitpad' expects a TIMM model (timm/...), got '{args.model_name}'.")
    if args.preprocess_mode == 'dinov2_giant_fb_fitpad' and not args.model_name.startswith('facebook/dinov2'):
         print(f"Warning: Mode 'dinov2_giant_fb_fitpad' expects a Facebook DINOv2 model (facebook/dinov2-...), got '{args.model_name}'.")

    return args

# --- Preprocessing Functions ---

def preprocess_fit_pad(img_pil, target_size=512, fill_color=(0, 0, 0)):
    """Resizes image to fit, pads to target size."""
    original_width, original_height = img_pil.size
    if original_width <= 0 or original_height <= 0: return None
    target_w, target_h = target_size, target_size
    scale = min(target_w / original_width, target_h / original_height)
    new_w = int(original_width * scale)
    new_h = int(original_height * scale)
    if new_w == 0: new_w = 1 # Prevent zero size
    if new_h == 0: new_h = 1
    try:
        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_padded = Image.new(img_pil.mode, (target_w, target_h), fill_color)
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        img_padded.paste(img_resized, (pad_left, pad_top))
        return img_padded
    except Exception as e:
        print(f"Error during preprocess_fit_pad: {e}")
        return None

def preprocess_center_crop(img_pil, target_size=512):
    """Resizes shortest side to target_size, then center crops."""
    original_width, original_height = img_pil.size
    if original_width <= 0 or original_height <= 0: return None
    try:
        short, long = (original_width, original_height) if original_width <= original_height else (original_height, original_width)
        if short == 0: return None # Avoid division by zero
        scale = target_size / short
        new_w = int(original_width * scale)
        new_h = int(original_height * scale)
        if new_w == 0: new_w = 1
        if new_h == 0: new_h = 1

        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        crop_w, crop_h = target_size, target_size
        left = (new_w - crop_w) // 2
        top = (new_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h

        img_cropped = img_resized.crop((left, top, right, bottom))
        return img_cropped
    except Exception as e:
        print(f"Error during preprocess_center_crop: {e}")
        return None

# v4.1: Targets patches, floors dims, ensures multiple-of-16, returns (img, patch_count)
def preprocess_naflex_resize(img_pil, target_patches=1024, patch_size=16):
    """
    Resizes image preserving aspect ratio for target_patches, ensuring dimensions
    are multiples of patch_size (flooring). Returns tuple: (resized_image, total_patches).
    """
    original_width, original_height = img_pil.size
    if original_width <= 0 or original_height <= 0: return None, 0
    try:
        aspect_ratio = original_width / original_height
        ideal_patch_w_f = math.sqrt(target_patches * aspect_ratio)
        ideal_patch_h_f = math.sqrt(target_patches / aspect_ratio)
        ideal_width_f = ideal_patch_w_f * patch_size
        ideal_height_f = ideal_patch_h_f * patch_size
        new_width = math.floor(ideal_width_f / patch_size) * patch_size
        new_height = math.floor(ideal_height_f / patch_size) * patch_size
        if new_width == 0: new_width = patch_size
        if new_height == 0: new_height = patch_size
        num_patches_w = new_width // patch_size
        num_patches_h = new_height // patch_size
        total_patches = num_patches_w * num_patches_h

        print(f"  DEBUG Preprocess NaflexResize(v4.1): Original: {original_width}x{original_height}, TargetPatches: {target_patches}, New: {new_width}x{new_height}, Patches: {total_patches} ({num_patches_w}x{num_patches_h})")

        if total_patches > target_patches * 1.05: # Allow small overshoot? Or strict?
            print(f"  ERROR: Calculated patches ({total_patches}) exceed target ({target_patches})!")
            return None, 0

        img_resized = img_pil.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)
        return img_resized, total_patches
    except Exception as e:
        print(f"Error during preprocess_naflex_resize: {e}")
        return None, 0


# --- Model Initialization ---
def init_vision_model(model_name, device, dtype):
    """Initializes vision model and processor, disabling auto-preprocessing."""
    print(f"Initializing Vision Model: {model_name} on {device} with dtype {dtype}")
    try:
        # use_fast=False recommended to silence warning if using older models/processors
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True).to(device).eval()

        print(f"  Loaded model class: {model.__class__.__name__}")
        print(f"  Loaded processor class: {processor.__class__.__name__}")

        # --- Disable Processor Auto-Preprocessing ---
        if hasattr(processor, 'image_processor'):
             image_processor = processor.image_processor
             print(f"  DEBUG: Original processor config: {image_processor}")
#             if hasattr(image_processor, 'do_resize'): image_processor.do_resize = False
             if hasattr(image_processor, 'do_center_crop'): image_processor.do_center_crop = False
#            if hasattr(image_processor, 'do_rescale'): image_processor.do_rescale = False
             if hasattr(image_processor, 'do_normalize'): image_processor.do_normalize = True
             print(f"  DEBUG: Modified processor config: {image_processor}")
        else: print("  Warning: Cannot access image_processor to disable auto-preprocessing.")

        # Determine model's standard input size
        image_size = 384  # Default guess
        try:
            model_config = getattr(model, 'config', None)
            if model_config:
                # Check standard vision_config first (like SigLIP)
                vision_config = getattr(model_config, 'vision_config', None)
                if vision_config and hasattr(vision_config, 'image_size'):
                    image_size = int(vision_config.image_size)
                    print(f"  DEBUG: Found size via model.config.vision_config.image_size: {image_size}")
                # Check DINOv2 specific config location
                elif hasattr(model_config, 'image_size'):
                    image_size = int(model_config.image_size)
                    print(f"  DEBUG: Found size via model.config.image_size: {image_size}")
                # Fallback to processor config (might not reflect model needs accurately)
                elif hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'size'):
                    proc_sz_config = processor.image_processor.size
                    if isinstance(proc_sz_config, dict):
                        image_size = int(proc_sz_config.get("height", image_size))
                    elif isinstance(proc_sz_config, int):
                        image_size = int(proc_sz_config)
                    print(f"  DEBUG: Found size via processor.image_processor.size: {image_size}")
                else:
                    print(f"  DEBUG: Could not find specific image_size config. Using default: {image_size}")
            else:
                print(f"  DEBUG: Model has no 'config' attribute. Using default size: {image_size}")

            print(f"  Determined Model Standard Input Size (for non-Naflex modes): {image_size}x{image_size}")
        except Exception as e_size:
            print(f"  Warning: Could not determine model standard input size: {e_size}. Using default {image_size}.")

        return processor, model, image_size

    except Exception as e:
        print(f"Error initializing vision model {model_name}: {e}")
        raise


# --- Unified Embedding Function ---
# Version 4.2.2: Added Facebook DINOv2 Manual Preprocessing support
@torch.no_grad()
def get_embedding(
    raw_img_pil: Image.Image,
    preprocess_mode: str,
    model_info: dict,
    device,
    dtype,
    model_image_size: int, # Size needed for manual preprocess modes
    # resize_factor_avg_crop: float = 2.0, # Removed avg_crop for now
) -> np.ndarray | None:
    """
    Generates embedding for a single raw PIL image using the specified mode
    and the provided model_info bundle.
    Returns normalized float32 numpy embedding or None on error.
    """
    img_name = getattr(raw_img_pil, 'filename', 'UNKNOWN')
    model_type = model_info.get("type", "hf") # Default to Hugging Face

    try:
        emb = None # Initialize embedding variable
        do_l2_normalize = False # Flag to normalize at the end

        # --- TIMM Model Handling (e.g., DINOv2 TIMM) ---
        if model_type == "timm" and preprocess_mode == 'dinov2_large_timm_fitpad':
            timm_model = model_info.get("model")
            timm_transforms = model_info.get("transforms")
            if timm_model is None or timm_transforms is None:
                 raise ValueError("TIMM model/transforms missing for dinov2_large_timm_fitpad.")

            input_tensor = timm_transforms(raw_img_pil).unsqueeze(0).to(device=device, dtype=dtype)
            output = timm_model(input_tensor) # Pooled features [1, D]
            emb = output
            do_l2_normalize = True # Normalize DINOv2 output

        # --- Hugging Face Model Handling (SigLIP or Facebook DINOv2) ---
        elif model_type == "hf":
            processor = model_info.get("processor")
            model = model_info.get("model")
            if processor is None or model is None:
                 raise ValueError("HF processor or model missing.")

            model_call_kwargs = {}
            is_siglip2_model = "Siglip2Model" in model.__class__.__name__
            is_dinov2_model = "Dinov2Model" in model.__class__.__name__

            # --- HF NaFlex Mode (Processor Logic @ 1024) ---
            if is_siglip2_model and preprocess_mode == 'naflex_resize':
                # print(f"DEBUG get_embedding [HF/NaFlex]: Using Processor logic (target 1024).")
                try:
                    inputs = processor(images=[raw_img_pil], return_tensors="pt", max_num_patches=1024)
                except Exception as e_proc: # ... (error handling) ...
                    return None

                pixel_values = inputs.get("pixel_values")
                attention_mask = inputs.get("pixel_attention_mask")
                spatial_shapes = inputs.get("spatial_shapes")

                if pixel_values is None or attention_mask is None or spatial_shapes is None:
                    raise ValueError("Missing required tensors from HF NaFlex processor output.")
                if pixel_values.shape[1] != 1024 or attention_mask.shape[1] != 1024:
                    raise ValueError(f"HF NaFlex Processor output seq len != 1024 ({pixel_values.shape[1]})")

                model_call_kwargs = {
                    "pixel_values": pixel_values.to(device=device, dtype=dtype),
                    "attention_mask": attention_mask.to(device=device),
                    "spatial_shapes": torch.tensor(spatial_shapes, dtype=torch.long).to(device=device)
                }

                do_l2_normalize = False

                # --- HF DINOv2 Mode (Manual FitPad + CLS Token) ---
            elif is_dinov2_model and preprocess_mode == 'dinov2_giant_fb_fitpad':
                # print(f"DEBUG get_embedding [HF/DINOv2]: Using manual FitPad.")
                # Use model_image_size determined during init (should be 518)
                processed_img_pil = preprocess_fit_pad(raw_img_pil, target_size=model_image_size)
                if processed_img_pil is None: return None

                inputs = processor(images=[processed_img_pil], return_tensors="pt")
                pixel_values = inputs.get("pixel_values")
                if pixel_values is None: raise ValueError("HF DINOv2 Processor didn't return 'pixel_values'.")
                # DINOv2 base forward usually doesn't need mask/shapes for fixed size input
                model_call_kwargs = {"pixel_values": pixel_values.to(device=device, dtype=dtype)}
                do_l2_normalize = True  # Normalize DINOv2 CLS token

            # --- HF Manual Modes (FitPad / CenterCrop) ---
            elif preprocess_mode in ['fit_pad', 'center_crop']:
                # Apply manual preprocessing (works for both SigLIP and FB DINOv2)
                processed_img_pil = None
                if preprocess_mode == 'fit_pad':
                    processed_img_pil = preprocess_fit_pad(raw_img_pil, target_size=model_image_size)
                elif preprocess_mode == 'center_crop':
                    processed_img_pil = preprocess_center_crop(raw_img_pil, target_size=model_image_size)
                if processed_img_pil is None: return None # Preprocessing failed

                # Use processor ONLY for ToTensor + Normalize
                inputs = processor(images=[processed_img_pil], return_tensors="pt")
                pixel_values = inputs.get("pixel_values")
                if pixel_values is None: raise ValueError("Processor didn't return 'pixel_values'.")

                # Set model call args (usually just pixel_values for these models/modes)
                model_call_kwargs = {"pixel_values": pixel_values.to(device=device, dtype=dtype)}
                attention_mask = inputs.get("pixel_attention_mask") # Include mask if processor provides it
                if attention_mask is not None: model_call_kwargs["attention_mask"] = attention_mask.to(device=device)

                # <<< Set normalize flag ONLY for DINOv2 in these modes >>>
                if is_dinov2_model:
                    do_l2_normalize = True
                else: # SigLIP FitPad/CenterCrop doesn't need external normalization
                    do_l2_normalize = False

            # --- Invalid HF Mode ---
            else:
                print(f"ERROR get_embedding [HF]: Invalid/unsupported preprocess_mode '{preprocess_mode}' for detected HF model type.")
                return None

            # --- HF Model Call ---
            # print(f"DEBUG get_embedding [HF]: Calling HF model ({model.__class__.__name__})...")
            # DINOv2Model forward directly returns BaseModelOutput /w last_hidden_state
            # Siglip2Model forward needs specific component call or get_image_features
            if is_dinov2_model:
                outputs = model(**model_call_kwargs)
                last_hidden_state = outputs.last_hidden_state
                if last_hidden_state is None: raise ValueError("DINOv2 model did not return last_hidden_state.")
                emb = last_hidden_state[:, 0]  # Extract CLS token
            elif is_siglip2_model:
                vision_model_component = getattr(model, 'vision_model', None)
                if vision_model_component:
                    vision_outputs = vision_model_component(**model_call_kwargs)
                    emb = vision_outputs.pooler_output  # Use pooled output
                elif hasattr(model, 'get_image_features'):
                    emb = model.get_image_features(
                        **model_call_kwargs)  # Use get_image_features if vision_model missing
                else:
                    raise AttributeError("SigLIP Model has neither 'vision_model' nor 'get_image_features'.")
            else:  # Other HF model types?
                print(f"Warning: Unknown HF model type {model.__class__.__name__}. Attempting direct call.")
                outputs = model(**model_call_kwargs)
                emb = getattr(outputs, 'pooler_output', getattr(outputs, 'last_hidden_state', None))
                
                assert isinstance(emb, torch.Tensor), f"Expected emb to be a Tensor, but got {type(emb)}"
                
                if emb is not None and emb.ndim == 3:  # If we got LHS, try taking first token
                    print("Warning: Got LHS from unknown model, taking CLS token [:, 0].")
                    emb = emb[:, 0]

            if emb is None: raise ValueError("Failed to get embedding from HF model call.")

        # --- Invalid Mode ---
        else:
            print(
                f"ERROR get_embedding: Unknown model_type '{model_type}' or unhandled preprocess_mode '{preprocess_mode}'.")
            return None

        # --- Final Normalization & Conversion ---
        if emb is None: print(f"ERROR: Embedding is None before final conversion for {img_name}."); return None
        if not isinstance(emb, torch.Tensor): raise TypeError(f"Embedding is not a Tensor ({type(emb)}).")

        if do_l2_normalize:
            # print(f"DEBUG get_embedding: L2 Normalizing final embedding.")
            norm = torch.linalg.norm(emb.float(), dim=-1, keepdim=True).clamp(min=1e-8)
            normalized_emb = emb / norm.to(emb.dtype)
            emb = normalized_emb

        embedding_result_np = emb.cpu().to(torch.float32).numpy().squeeze()
        return embedding_result_np

    except Exception as e:
        print(f"\nError during get_embedding (v4.2.0) for {img_name} (Mode: {preprocess_mode}, Type: {model_type}):")
        traceback.print_exc()
        return None
# --- End Unified Embedding Function ---


# --- Main Execution Logic ---
if __name__ == "__main__":
    args = parse_gen_args()

    # --- Setup Compute Precision ---
    COMPUTE_DTYPE = DEFAULT_PRECISION_MAP.get(args.precision, torch.float32)
    if COMPUTE_DTYPE != torch.float32 and TARGET_DEV == 'cpu':
        print(f"Warning: {args.precision} requested on CPU. Using float32.")
        COMPUTE_DTYPE = torch.float32
    if COMPUTE_DTYPE == torch.bfloat16 and (TARGET_DEV != 'cuda' or not torch.cuda.is_bf16_supported()):
         print("Warning: bf16 not supported. Using float32.")
         COMPUTE_DTYPE = torch.float32

    # --- Initialize Model & Preprocessing Info ---
    model_info = {} # Bundle to hold model, processor, transforms, type
    model_image_size = 0 # Needed for manual preprocessing

    # --- TIMM Model Handling ---
    if args.model_name.startswith("timm/") and TIMM_AVAILABLE:
         print("Detected TIMM model name.")
         if args.preprocess_mode != 'dinov2_large_timm_fitpad':
             exit("Error: TIMM model requires 'dinov2_large_timm_fitpad' preprocess_mode.")
         try:
              print(f"Initializing TIMM Model: {args.model_name}...")
              timm_model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
              data_config = timm.data.resolve_model_data_config(timm_model)
              timm_input_size = data_config.get('input_size')
              if timm_input_size: model_image_size = timm_input_size[-1]
              else: model_image_size = 224 # Fallback
              print(f"  TIMM Input Size: {model_image_size}")
              timm_transforms = timm.data.create_transform(**data_config, is_training=False)
              timm_model = timm_model.to(device=TARGET_DEV, dtype=COMPUTE_DTYPE).eval()

              model_info = {"type": "timm", "model": timm_model, "transforms": timm_transforms}
         except Exception as e: # ... (error handling) ...
              exit(1)

    # --- Hugging Face Model Handling (SigLIP or FB DINOv2) ---
    else:
         print("Assuming Hugging Face model name.")
         try:
              # init_vision_model loads processor/model and determines image_size
              processor, vision_model, hf_model_image_size = init_vision_model(args.model_name, TARGET_DEV, COMPUTE_DTYPE)
              model_info = {"type": "hf", "processor": processor, "model": vision_model}
              model_image_size = hf_model_image_size # Use size from HF config/processor

              # Add specific check for DINOv2 FB size
              if args.preprocess_mode == 'dinov2_giant_fb_fitpad' and model_image_size != 518:
                   print(f"Warning: FB DINOv2 mode selected, but detected image size is {model_image_size} (expected 518). Using detected size.")
                   # Or force it? Let's use detected for now.
                   # model_image_size = 518

         except Exception as e:
             print(f"Error initializing Hugging Face model/processor: {e}")
             exit(1)

    # --- Determine Output Directory ---
    model_name_safe = args.model_name.split('/')[-1].replace('-', '_')
    # Updated Naming Map
    mode_suffix_map = {
        'fit_pad': "FitPad",              # HF SigLIP Manual FitPad
        'center_crop': "CenterCrop",      # HF SigLIP Manual CenterCrop
        'avg_crop': "AvgCrop",            # NYI
        'naflex_resize': "Naflex_Proc1024", # HF SigLIP NaFlex Processor@1024
        'dinov2_large_timm_fitpad': "FitPad",   # TIMM DINOv2 (uses FitPad via transforms)
        # No need for dinov2_giant_fb_fitpad here, handled by model type prefix now
    }
    mode_suffix = mode_suffix_map.get(args.preprocess_mode, "UnknownMode")

    # Add prefix based on model source AND type for clarity
    model_prefix = ""
    if model_info["type"] == "timm": model_prefix = "timm_"
    elif model_info["type"] == "hf" and "dinov2" in args.model_name.lower(): model_prefix = "fb_" # Prefix for Facebook DINOv2

    output_subdir_name = f"{model_prefix}{model_name_safe}_{mode_suffix}{args.output_dir_suffix}"
    final_output_dir = os.path.join(args.output_dir_root, output_subdir_name)

    print(f"\nSelected Model: {args.model_name} ({model_info['type']} type)")
    print(f"Selected Preprocessing Mode: {args.preprocess_mode}")
    if args.preprocess_mode == 'naflex_resize': print("  (Using HF Processor logic with target max_num_patches=1024)")
    if args.preprocess_mode == 'dinov2_large_timm_fitpad': print(f"  (Using TIMM transforms with input size {model_image_size})")
    if args.preprocess_mode == 'dinov2_giant_fb_fitpad': print(f"  (Using Manual FitPad to size {model_image_size})")
    print(f"Embeddings will be saved in: {final_output_dir}")

    # --- Process Source Folders ---
    source_subfolders = sorted([d for d in os.listdir(args.image_dir) if os.path.isdir(os.path.join(args.image_dir, d))])
    if not source_subfolders: exit(f"Error: No subfolders found in image directory: {args.image_dir}")
    print(f"Found source subfolders: {source_subfolders}")

    for src_folder in source_subfolders:
        current_image_dir = os.path.join(args.image_dir, src_folder)
        target_label = src_folder
        current_output_subdir = os.path.join(final_output_dir, target_label)
        os.makedirs(current_output_subdir, exist_ok=True)
        print(f"\nProcessing: {current_image_dir} -> {current_output_subdir}")

        image_files = [f for f in os.listdir(current_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not image_files: print("  No image files found."); continue

        for fname in tqdm(image_files, desc=f"Folder '{src_folder}'"):
            embedding_result = None
            try:
                img_path = os.path.join(current_image_dir, fname)
                try:
                    raw_img_pil = Image.open(img_path).convert("RGB")
                except Exception as img_e:
                    print(f"\nError opening image {fname}: {img_e}. Skipping.")
                    continue

                # --- Call unified embedding function ---
                embedding_result = get_embedding(
                    raw_img_pil=raw_img_pil,
                    preprocess_mode=args.preprocess_mode,
                    model_info=model_info, # Pass the bundle
                    device=TARGET_DEV,
                    dtype=COMPUTE_DTYPE,
                    model_image_size=model_image_size, # Pass size needed for manual modes
                    # resize_factor_avg_crop=args.resize_factor_avg_crop # Removed avg_crop
                )

                # --- Save result ---
                if embedding_result is not None:
                    base_fname = os.path.splitext(fname)[0]
                    output_path = os.path.join(current_output_subdir, f"{base_fname}.npy")
                    np.save(output_path, embedding_result)
                else:
                    print(f"Info: Skipping save for {fname} as embedding generation returned None.")

            except Exception as e: # Catch any unexpected errors in the loop
                print(f"\nCAUGHT UNEXPECTED EXCEPTION while processing image {fname}. Details below. Skipping.")
                print(f"  EXCEPTION TYPE: {type(e)}")
                print(f"  EXCEPTION VALUE: {e}")
                traceback.print_exc()
                continue
        # --- End Image Loop ---
    # --- End Folder Loop ---

    print("\nEmbedding generation complete!")