# generate_embeddings.py
# Version: 4.1.0 (Unified get_embedding function, robust handling)

import os
import torch
import torchvision.transforms.functional as F # Needed for AvgCrop's five_crop
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, AutoModel
import argparse
import math
import shutil # Keep for potential future use
import traceback # For detailed error printing

# --- Constants ---
TARGET_DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_PRECISION_MAP = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
# Defaulting to standard SigLIP here, will be overridden by args
DEFAULT_MODEL_ID = "google/siglip-so400m-patch14-384" # Changed default for clarity

# --- Argument Parsing ---
def parse_gen_args():
    parser = argparse.ArgumentParser(description="Generate vision model embeddings for image folders.")
    parser.add_argument('--image_dir', required=True, help="Directory containing subfolders (e.g., '0', '1') with images.")
    parser.add_argument('--output_dir_root', required=True, help="Root directory to save embeddings (e.g., 'data').")
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_ID, help=f"Vision model name from Hugging Face. Default: {DEFAULT_MODEL_ID}")
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'bf16', 'fp16'], help="Precision for model computation (default: fp32).")
    parser.add_argument('--preprocess_mode', type=str, default='fit_pad',
                        choices=['fit_pad', 'center_crop', 'avg_crop', 'naflex_resize'],
                        help="Image preprocessing method before embedding (default: fit_pad).")
    parser.add_argument('--resize_factor_avg_crop', type=float, default=2.0,
                        help="Factor to multiply model's native size by for resizing ONLY for avg_crop mode (default: 2.0).")
    parser.add_argument('--output_dir_suffix', type=str, default="", help="Optional suffix for the output directory name.")
    return parser.parse_args()

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
             if hasattr(image_processor, 'do_resize'): image_processor.do_resize = False
             if hasattr(image_processor, 'do_center_crop'): image_processor.do_center_crop = False
             if hasattr(image_processor, 'do_rescale'): image_processor.do_rescale = False
             if hasattr(image_processor, 'do_normalize'): image_processor.do_normalize = True
             print(f"  DEBUG: Modified processor config: {image_processor}")
        else: print("  Warning: Cannot access image_processor to disable auto-preprocessing.")

        # Determine model's standard input size (for non-naflex modes)
        image_size = 384 # Default guess
        try:
             if hasattr(model, 'config') and hasattr(model.config, 'vision_config') and hasattr(model.config.vision_config, 'image_size'):
                  image_size = int(model.config.vision_config.image_size)
             elif hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'size'):
                  # Use .get('height') for dict sizes, handle int directly
                  proc_sz_config = processor.image_processor.size
                  if isinstance(proc_sz_config, dict): image_size = int(proc_sz_config.get("height", image_size))
                  elif isinstance(proc_sz_config, int): image_size = int(proc_sz_config)
             print(f"  Determined Model Standard Input Size (for non-Naflex modes): {image_size}x{image_size}")
        except Exception as e_size:
             print(f"  Warning: Could not determine model standard input size: {e_size}. Using default {image_size}.")

        return processor, model, image_size
    except Exception as e:
        print(f"Error initializing vision model {model_name}: {e}")
        raise


# --- Unified Embedding Function ---
@torch.no_grad()
def get_embedding(
    raw_img_pil: Image.Image,
    preprocess_mode: str,
    processor,
    model, # The overall model object (e.g., SiglipModel or Siglip2Model)
    device,
    dtype, # Compute dtype
    model_image_size: int, # Standard size for non-Naflex
    resize_factor_avg_crop: float = 2.0
) -> np.ndarray | None:
    """
    Generates embedding for a single raw PIL image using the specified mode.
    Handles FitPad, CenterCrop, NaFlexResize, and AvgCrop.
    Returns normalized float32 numpy embedding or None on error.
    """
    img_name = getattr(raw_img_pil, 'filename', 'UNKNOWN') # For error logging

    try:
        # --- Determine Model Type ---
        is_siglip2_model = "Siglip2Model" in model.__class__.__name__
        print(f"DEBUG get_embedding: Model Type: {'Siglip2' if is_siglip2_model else 'Standard Siglip/Other'}")

        # --- Handle AvgCrop Separately ---
        if preprocess_mode == 'avg_crop':
            print(f"DEBUG get_embedding: Using AvgCrop logic for {img_name}")
            # This needs separate implementation as it involves multiple model calls
            # Paste or implement the get_avg_crop_embedding logic here
            # For now, just return None to indicate it needs implementation
            print("ERROR: AvgCrop embedding generation not fully implemented in unified function yet.")
            return None # Placeholder

        # --- Preprocessing for other modes ---
        processed_img_pil = None
        actual_num_patches = None # Only set by NaFlex

        if preprocess_mode == 'fit_pad':
            processed_img_pil = preprocess_fit_pad(raw_img_pil, target_size=model_image_size)
        elif preprocess_mode == 'center_crop':
            processed_img_pil = preprocess_center_crop(raw_img_pil, target_size=model_image_size)
        elif preprocess_mode == 'naflex_resize':
            # This function now returns (image, patch_count)
            resize_output, patches = preprocess_naflex_resize(raw_img_pil)
            if resize_output is not None:
                processed_img_pil = resize_output
                actual_num_patches = patches
        else: # Should not happen if args are validated, but good practice
            print(f"ERROR get_embedding: Invalid preprocess_mode '{preprocess_mode}' for non-AvgCrop.")
            return None

        if processed_img_pil is None:
            print(f"Warning get_embedding: Preprocessing returned None for {img_name}. Skipping.")
            return None

        # --- Calculate patch count if not already known (for non-Naflex) ---
        if actual_num_patches is None:
            w, h = processed_img_pil.size; p = 16 # Assume patch size 16
            num_patches_w = w // p; num_patches_h = h // p
            actual_num_patches = num_patches_w * num_patches_h
            print(f"DEBUG get_embedding: Calculated num_patches={actual_num_patches} for {preprocess_mode}")
        # Ensure actual_num_patches has a valid integer value
        if not isinstance(actual_num_patches, int) or actual_num_patches <= 0:
             print(f"ERROR get_embedding: Invalid calculated patch count ({actual_num_patches}) for {img_name}.")
             return None

        # --- Processor Call ---
        print(f"DEBUG get_embedding: Calling processor...")
        inputs = processor(images=[processed_img_pil], return_tensors="pt")
        print("DEBUG get_embedding: Processor call finished.")

        # --- Extract Tensors & Prepare Model Args ---
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None: raise ValueError("Processor didn't return 'pixel_values'.")
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        model_call_kwargs = {"pixel_values": pixel_values}

        if is_siglip2_model:
            # Siglip2 needs attention_mask and spatial_shapes
            # Let's try letting the processor handle the mask size now based on input?
            attention_mask = inputs.get("pixel_attention_mask")
            spatial_shapes = inputs.get("spatial_shapes")

            if attention_mask is None: raise ValueError("Processor didn't return 'pixel_attention_mask' for Siglip2.")
            if spatial_shapes is None: raise ValueError("Processor didn't return 'spatial_shapes' for Siglip2.")

            # Check if mask size matches calculated patches (it should now if processor works)
            if attention_mask.shape[-1] != actual_num_patches:
                print(f"WARNING get_embedding: Processor mask size {attention_mask.shape[-1]} != calculated patches {actual_num_patches}. Using calculated size for model call.")
                # Override with manually created mask if size mismatch persists
                attention_mask = torch.ones((1, actual_num_patches), dtype=torch.long, device=device)
            else:
                print(f"DEBUG get_embedding: Using processor-generated mask size {attention_mask.shape[-1]}.")
                attention_mask = attention_mask.to(device=device)

            spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long).to(device=device)
            model_call_kwargs["attention_mask"] = attention_mask
            model_call_kwargs["spatial_shapes"] = spatial_shapes
        else: # Standard Siglip
             # Try passing the mask if the processor provides one (e.g., for FitPad=1024)
             attention_mask = inputs.get("pixel_attention_mask")
             if attention_mask is not None:
                  print(f"DEBUG get_embedding: Passing processor mask size {attention_mask.shape[-1]} to Standard Siglip.")
                  model_call_kwargs["attention_mask"] = attention_mask.to(device=device)
             else:
                   print("DEBUG get_embedding: No mask from processor for Standard Siglip.")

        # --- Model Call ---
        emb = None
        vision_model_component = getattr(model, 'vision_model', None)

        if vision_model_component:
            print(f"DEBUG get_embedding: Calling vision_model ({vision_model_component.__class__.__name__}) with keys: {list(model_call_kwargs.keys())}")
            vision_outputs = vision_model_component(**model_call_kwargs)
            emb = vision_outputs.pooler_output
        elif hasattr(model, 'get_image_features'): # Fallback
             print(f"DEBUG get_embedding: Falling back to get_image_features.")
             # get_image_features usually only takes pixel_values
             emb = model.get_image_features(pixel_values=model_call_kwargs["pixel_values"])
        else:
            raise AttributeError("Model has neither 'vision_model' nor 'get_image_features'.")

        if emb is None: raise ValueError("Failed to get embedding from model call.")

        # --- Normalize & Convert ---
        # Use float() for norm calculation for stability, then divide original tensor
        norm = torch.linalg.norm(emb.detach().float(), dim=-1, keepdim=True).clamp(min=1e-8)
        normalized_embedding_tensor = emb / norm.to(emb.dtype) # Divide original tensor by norm cast back to its dtype
        embedding_result_np = normalized_embedding_tensor.cpu().to(torch.float32).numpy().squeeze()
        return embedding_result_np

    except Exception as e:
        print(f"\nError during get_embedding for {img_name} (Mode: {preprocess_mode}):")
        traceback.print_exc()
        return None


# --- Main Execution Logic ---
if __name__ == "__main__":
    args = parse_gen_args()

    # --- Setup Compute Precision ---
    COMPUTE_DTYPE = DEFAULT_PRECISION_MAP.get(args.precision, torch.float32)
    # ... (Hardware checks for bf16/fp16) ...
    if COMPUTE_DTYPE != torch.float32 and TARGET_DEV == 'cpu':
        print(f"Warning: {args.precision} requested on CPU. Using float32.")
        COMPUTE_DTYPE = torch.float32
    if COMPUTE_DTYPE == torch.bfloat16 and (TARGET_DEV != 'cuda' or not torch.cuda.is_bf16_supported()):
         print("Warning: bf16 not supported. Using float32.")
         COMPUTE_DTYPE = torch.float32

    # --- Initialize Model ---
    processor, vision_model, model_image_size = init_vision_model(args.model_name, TARGET_DEV, COMPUTE_DTYPE)

    # --- Determine Output Directory ---
    model_name_safe = args.model_name.split('/')[-1].replace('-', '_')
    mode_suffix_map = {'fit_pad': "FitPad", 'center_crop': "CenterCrop",
                       'avg_crop': "AvgCrop", 'naflex_resize': "NaflexResize"}
    mode_suffix = mode_suffix_map.get(args.preprocess_mode, "UnknownMode")
    output_subdir_name = f"{model_name_safe}_{mode_suffix}{args.output_dir_suffix}"
    final_output_dir = os.path.join(args.output_dir_root, output_subdir_name)
    print(f"\nSelected Preprocessing Mode: {args.preprocess_mode}")
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
                # --- Load Image ---
                try:
                    raw_img_pil = Image.open(img_path).convert("RGB")
                except Exception as img_e:
                    print(f"\nError opening image {fname}: {img_e}. Skipping.")
                    continue # Skip this image

                # --- Call unified embedding function ---
                embedding_result = get_embedding(
                    raw_img_pil=raw_img_pil,
                    preprocess_mode=args.preprocess_mode,
                    processor=processor,
                    model=vision_model,
                    device=TARGET_DEV,
                    dtype=COMPUTE_DTYPE,
                    model_image_size=model_image_size,
                    resize_factor_avg_crop=args.resize_factor_avg_crop
                )

                # --- Save result ---
                if embedding_result is not None:
                    base_fname = os.path.splitext(fname)[0]
                    output_path = os.path.join(current_output_subdir, f"{base_fname}.npy")
                    np.save(output_path, embedding_result)
                else:
                    # get_embedding function now handles internal errors and prints warnings
                    print(f"Info: Skipping save for {fname} as embedding generation returned None.")

            except Exception as e: # Catch any unexpected errors in the loop
                print(f"\nCAUGHT UNEXPECTED EXCEPTION while processing image {fname}. Details below. Skipping.")
                print(f"  EXCEPTION TYPE: {type(e)}")
                print(f"  EXCEPTION VALUE: {e}")
                traceback.print_exc()
                continue # Skip to next image
        # --- End Image Loop ---
    # --- End Folder Loop ---

    print("\nEmbedding generation complete!")