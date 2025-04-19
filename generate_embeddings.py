# generate_embeddings.py
# Version: 4.1.0 (Unified get_embedding function, robust handling)

import os
import torch
import torchvision.transforms.functional as F # Needed for AvgCrop's five_crop
import torch.nn.functional as torch_func
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
#             if hasattr(image_processor, 'do_resize'): image_processor.do_resize = False
             if hasattr(image_processor, 'do_center_crop'): image_processor.do_center_crop = False
#            if hasattr(image_processor, 'do_rescale'): image_processor.do_rescale = False
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
# Version 4.1.5: Permanent NaFlex uses Processor Logic @ 1024. Removed manual resize/pad.
@torch.no_grad()
def get_embedding(
    raw_img_pil: Image.Image,
    preprocess_mode: str,
    processor,
    model,
    device,
    dtype,
    model_image_size: int,
    resize_factor_avg_crop: float = 2.0,
) -> np.ndarray | None:
    """
    Generates embedding for a single raw PIL image using the specified mode.
    - NaFlex mode now uses processor's internal logic targeting max 1024 patches.
    - FitPad/CenterCrop use manual resize then processor call.
    Returns normalized float32 numpy embedding or None on error.
    """
    img_name = getattr(raw_img_pil, 'filename', 'UNKNOWN')

    try:
        is_siglip2_model = "Siglip2Model" in model.__class__.__name__
        model_call_kwargs = {} # Initialize outside conditional blocks

        # --- NaFlex Handling (Processor Logic @ 1024) ---
        if is_siglip2_model and preprocess_mode == 'naflex_resize':
            # print(f"DEBUG get_embedding: Using Processor NaFlex logic (target max 1024 patches) for {img_name}.")
            try:
                inputs = processor(
                    images=[raw_img_pil],
                    return_tensors="pt",
                    max_num_patches=1024 # <<< Override target length
                )
            except Exception as e_proc:
                print(f"Error calling processor directly for NaFlex: {e_proc}")
                return None

            # Extract outputs directly from processor
            pixel_values = inputs.get("pixel_values")
            attention_mask = inputs.get("pixel_attention_mask")
            spatial_shapes = inputs.get("spatial_shapes")

            if pixel_values is None: raise ValueError("Processor didn't return 'pixel_values'.")
            if attention_mask is None: raise ValueError("Processor didn't return 'pixel_attention_mask'.")
            if spatial_shapes is None: raise ValueError("Processor didn't return 'spatial_shapes'.")

            # Quick check on sequence length - should now be consistently 1024
            if pixel_values.shape[1] != 1024 or attention_mask.shape[1] != 1024:
                 print(f"ERROR: NaFlex Processor output sequence length is not 1024! Got {pixel_values.shape[1]}.")
                 return None

            # Prepare args for model call
            model_call_kwargs = {
                "pixel_values": pixel_values.to(device=device, dtype=dtype),
                "attention_mask": attention_mask.to(device=device),
                "spatial_shapes": torch.tensor(spatial_shapes, dtype=torch.long).to(device=device)
            }

        # --- FitPad / CenterCrop Handling ---
        elif preprocess_mode in ['fit_pad', 'center_crop']:
            processed_img_pil = None
            if preprocess_mode == 'fit_pad':
                # print(f"DEBUG get_embedding: Using manual FitPad preprocess for {img_name}.")
                processed_img_pil = preprocess_fit_pad(raw_img_pil, target_size=model_image_size)
            elif preprocess_mode == 'center_crop':
                # print(f"DEBUG get_embedding: Using manual CenterCrop preprocess for {img_name}.")
                processed_img_pil = preprocess_center_crop(raw_img_pil, target_size=model_image_size)

            if processed_img_pil is None:
                print(f"Warning get_embedding: Manual preprocessing returned None for {img_name}. Skipping.")
                return None

            # Call processor with the *already resized* image.
            # Processor's internal resize won't trigger if size matches.
            inputs = processor(images=[processed_img_pil], return_tensors="pt")
            pixel_values_from_proc = inputs.get("pixel_values")
            if pixel_values_from_proc is None: raise ValueError("Processor didn't return 'pixel_values'.")

            pixel_values = pixel_values_from_proc.to(device=device, dtype=dtype)
            model_call_kwargs = {"pixel_values": pixel_values}
            # Pass mask if available (e.g., for standard SigLIP if processor adds one)
            attention_mask_from_processor = inputs.get("pixel_attention_mask")
            if attention_mask_from_processor is not None:
                  model_call_kwargs["attention_mask"] = attention_mask_from_processor.to(device=device)

        # --- AvgCrop Handling (Still needs implementation) ---
        elif preprocess_mode == 'avg_crop':
             print("ERROR: AvgCrop embedding generation not fully implemented.")
             return None
        # --- Unknown Mode ---
        else:
            print(f"ERROR get_embedding: Invalid/unhandled preprocess_mode '{preprocess_mode}'.")
            return None

        # --- Model Call (Common Logic) ---
        emb = None
        vision_model_component = getattr(model, 'vision_model', None)

        if vision_model_component:
            # print(f"DEBUG get_embedding: Calling vision_model for {img_name}...")
            vision_outputs = vision_model_component(**model_call_kwargs)
            emb = vision_outputs.pooler_output
        elif hasattr(model, 'get_image_features'):
             # print(f"DEBUG get_embedding: Calling get_image_features for {img_name}...")
             emb = model.get_image_features(pixel_values=model_call_kwargs["pixel_values"])
        else:
            raise AttributeError("Model has neither 'vision_model' nor 'get_image_features'.")

        if emb is None: raise ValueError("Failed to get embedding from model call.")

        # --- Normalize & Convert (Common Logic) ---
        norm = torch.linalg.norm(emb.detach().float(), dim=-1, keepdim=True).clamp(min=1e-8)
        normalized_embedding_tensor = emb / norm.to(emb.dtype)
        embedding_result_np = normalized_embedding_tensor.cpu().to(torch.float32).numpy().squeeze()
        return embedding_result_np

    except Exception as e:
        print(f"\nError during get_embedding (v4.1.5) for {img_name} (Mode: {preprocess_mode}):")
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

    # --- Initialize Model & Processor ---
    # Calls the updated init_vision_model which keeps processor defaults
    processor, vision_model, model_image_size = init_vision_model(args.model_name, TARGET_DEV, COMPUTE_DTYPE)

    # --- Determine Output Directory ---
    model_name_safe = args.model_name.split('/')[-1].replace('-', '_')
    # <<< UPDATED: Permanent Naming for Processor Logic @ 1024 >>>
    proc_suffix = "_Proc1024" if args.preprocess_mode == 'naflex_resize' else ""
    mode_suffix_map = {'fit_pad': "FitPad", 'center_crop': "CenterCrop",
                       'avg_crop': "AvgCrop", 'naflex_resize': f"Naflex{proc_suffix}"} # e.g., Naflex_Proc1024
    mode_suffix = mode_suffix_map.get(args.preprocess_mode, "UnknownMode")
    output_subdir_name = f"{model_name_safe}_{mode_suffix}{args.output_dir_suffix}"
    final_output_dir = os.path.join(args.output_dir_root, output_subdir_name)
    print(f"\nSelected Preprocessing Mode: {args.preprocess_mode}")
    if args.preprocess_mode == 'naflex_resize':
         print("  (Using Processor logic with target max_num_patches=1024)") # Updated message
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
                # target_length=1024 is now the default in get_embedding v4.1.1
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