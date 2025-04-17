# Version: 4.0.0
# Desc: Added selectable preprocessing modes (fit_pad, center_crop, avg_crop),
#       removed test.npy creation, refactored preprocessing.

import os
import torch
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, AutoModel
import argparse
import math
import shutil # Added for potential use later if needed

# --- Constants ---
TARGET_DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_PRECISION_MAP = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
DEFAULT_MODEL_ID = "google/siglip2-so400m-patch16-512" # Defaulting to SigLIP now

# --- Argument Parsing ---
def parse_gen_args():
    parser = argparse.ArgumentParser(description="Generate vision model embeddings for image folders.")
    parser.add_argument('--image_dir', required=True, help="Directory containing subfolders (e.g., '0', '1') with images.")
    parser.add_argument('--output_dir_root', required=True, help="Root directory to save embeddings (e.g., 'data').")
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_ID, help=f"Vision model name from Hugging Face. Default: {DEFAULT_MODEL_ID}")
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'bf16', 'fp16'], help="Precision for model computation (default: fp32).")
    # --- NEW: Preprocessing Mode ---
    parser.add_argument('--preprocess_mode', type=str, default='fit_pad', choices=['fit_pad', 'center_crop', 'avg_crop', 'naflex_resize'],
                        help="Image preprocessing method before embedding (default: fit_pad).")
    # --- Modified: Resize Factor only for avg_crop ---
    parser.add_argument('--resize_factor_avg_crop', type=float, default=2.0,
                        help="Factor to multiply model's native size by for resizing ONLY for avg_crop mode (default: 2.0). Ignored otherwise.")
    parser.add_argument('--output_dir_suffix', type=str, default="", help="Optional suffix for the output directory name.")
    return parser.parse_args()

# --- Preprocessing Functions ---

# --- v4.0: Aims for target_patches, ensures dims multiple of patch_size (floor) ---
def preprocess_naflex_resize(img_pil, target_patches=1024, patch_size=16):
    """
    Resizes image preserving aspect ratio to have close to target_patches,
    ensuring dimensions are multiples of patch_size by flooring.
    """
    original_width, original_height = img_pil.size
    if original_width <= 0 or original_height <= 0: return None

    aspect_ratio = original_width / original_height

    # 1. Calculate ideal float dimensions for target_patches
    # target_patches ≈ (new_width / patch_size) * (new_height / patch_size)
    # target_patches ≈ (sqrt(target_patches * aspect_ratio) * patch_size / patch_size) * (sqrt(target_patches / aspect_ratio) * patch_size / patch_size)
    # Let's derive ideal float patch counts first
    ideal_patch_w_f = math.sqrt(target_patches * aspect_ratio)
    ideal_patch_h_f = math.sqrt(target_patches / aspect_ratio)

    # Ideal float dimensions
    ideal_width_f = ideal_patch_w_f * patch_size
    ideal_height_f = ideal_patch_h_f * patch_size

    # 2. Adjust dimensions DOWN to the nearest multiple of patch_size (use floor)
    # This ensures we never exceed target_patches
    new_width = math.floor(ideal_width_f / patch_size) * patch_size
    new_height = math.floor(ideal_height_f / patch_size) * patch_size

    # Ensure we don't get zero dimensions (if target_patches is too small)
    if new_width == 0: new_width = patch_size
    if new_height == 0: new_height = patch_size

    # Calculate resulting patches
    num_patches_w = new_width // patch_size
    num_patches_h = new_height // patch_size
    total_patches = num_patches_w * num_patches_h

    print(
        f"  DEBUG NaflexResize(v4.1): Original: {original_width}x{original_height}, TargetPatches: {target_patches}, New: {new_width}x{new_height}, Patches: {total_patches} ({num_patches_w}x{num_patches_h})")

    # Check if patches exceed target (shouldn't happen with floor)
    if total_patches > target_patches:
        print(f"  ERROR: Calculated patches ({total_patches}) exceed target ({target_patches})! Check logic.")
        # Optionally return None or raise error here if needed

    img_resized = img_pil.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)

    # <<< RETURN TOTAL PATCHES TOO >>>
    return img_resized, total_patches

def preprocess_fit_pad(img_pil, target_size=512, fill_color=(0, 0, 0)):
    """Resizes image to fit, pads to target size."""
    original_width, original_height = img_pil.size
    if original_width == 0 or original_height == 0: return None
    target_w, target_h = target_size, target_size
    scale = min(target_w / original_width, target_h / original_height)
    new_w = int(original_width * scale); new_h = int(original_height * scale)
    if new_w == 0 or new_h == 0: return None
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_padded = Image.new(img_pil.mode, (target_w, target_h), fill_color)
    pad_left = (target_w - new_w) // 2; pad_top = (target_h - new_h) // 2
    img_padded.paste(img_resized, (pad_left, pad_top))
    return img_padded

def preprocess_center_crop(img_pil, target_size=512):
    """Resizes shortest side to target_size, then center crops."""
    original_width, original_height = img_pil.size
    if original_width == 0 or original_height == 0: return None

    # Resize shortest side to target_size
    short, long = (original_width, original_height) if original_width <= original_height else (original_height, original_width)
    scale = target_size / short
    new_w, new_h = int(original_width * scale), int(original_height * scale)
    if new_w == 0 or new_h == 0: return None

    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center crop
    crop_w, crop_h = target_size, target_size
    left = (new_w - crop_w) // 2
    top = (new_h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    img_cropped = img_resized.crop((left, top, right, bottom))
    return img_cropped

# --- Model Initialization ---
def init_vision_model(model_name, device, dtype):
    """Initializes vision model and processor, disabling auto-preprocessing."""
    print(f"Initializing Vision Model: {model_name} on {device} with dtype {dtype}")
    try:
        # Use explicit cache dir to potentially avoid issues if default is problematic
        # cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        # processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        # model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True, cache_dir=cache_dir).to(device).eval()

        processor = AutoProcessor.from_pretrained(model_name)
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
             if hasattr(image_processor, 'do_normalize'): image_processor.do_normalize = True # Keep normalization
             print(f"  DEBUG: Modified processor config: {image_processor}")
        else: print("  Warning: Cannot access image_processor to disable auto-preprocessing.")
        # --- End Disable ---

        # Determine model's expected input size
        image_size = 384 # Default guess
        try:
             # Try getting from model config first, often more reliable
             if hasattr(model, 'config') and hasattr(model.config, 'image_size'):
                  image_size = int(model.config.image_size)
             # Fallback to processor config
             elif hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'size'):
                  proc_sz = processor.image_processor.size
                  if isinstance(proc_sz, dict): image_size = int(proc_sz.get("height", 384))
                  elif isinstance(proc_sz, int): image_size = int(proc_sz)
             print(f"  Determined Model Input Target Size: {image_size}x{image_size}")
        except Exception as e_size:
             print(f"  Warning: Could not reliably determine model input size: {e_size}. Using default {image_size}.")

        return processor, model, image_size
    except Exception as e:
        print(f"Error initializing vision model {model_name}: {e}")
        raise

# --- Embedding Functions ---

@torch.no_grad()
def get_single_view_embedding(processed_img_pil, actual_num_patches, processor, model, device,
                              dtype):  # <<< Added actual_num_patches arg
    """Gets embedding for a *single, already preprocessed* image."""
    # <<< No need to check processed_img_pil is None here, happens before call >>>
    # if processed_img_pil is None: return None
    try:
        # --- v4.1: Pass actual_num_patches to processor ---
        processor_kwargs = {"max_num_patches": actual_num_patches}
        inputs = processor(images=[processed_img_pil], return_tensors="pt", **processor_kwargs)
        # --- End v4.1 Change ---

        # Extract inputs (same as before)
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs.get("pixel_attention_mask")
        spatial_shapes = inputs.get("spatial_shapes")

        if pixel_values is None: raise ValueError("Processor didn't return pixel_values.")

        # Move tensors (same as before)
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        if attention_mask is not None:
            # <<< Ensure mask shape matches actual_num_patches >>>
            if attention_mask.shape[-1] != actual_num_patches:
                print(
                    f"WARNING: Attention mask shape {attention_mask.shape} != actual_num_patches {actual_num_patches}. Check processor call.")
            attention_mask = attention_mask.to(device=device)
        if spatial_shapes is not None:
            spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long).to(device=device)

        # Get features using the correct arguments (same as before)
        if hasattr(model, 'get_image_features'):
            embedding_tensor = model.get_image_features(
                pixel_values=pixel_values,
                pixel_attention_mask=attention_mask,
                spatial_shapes=spatial_shapes
            )
        else:
            print("Warning: Model lacks get_image_features. Using forward pass.")
            # Note: The direct forward call might also need attention_mask and spatial_shapes depending on the model.
            # This fallback might break for Siglip2.
            outputs = model(pixel_values=pixel_values) # Simplified fallback
            if hasattr(outputs, 'image_embeds'): embedding_tensor = outputs.image_embeds
            elif hasattr(outputs, 'pooler_output'): embedding_tensor = outputs.pooler_output
            else: embedding_tensor = outputs.last_hidden_state.mean(dim=1) # Generic fallback

        # Normalize & Convert (same as before)
        norm = torch.linalg.norm(embedding_tensor, dim=-1, keepdim=True).clamp(min=1e-8)
        normalized_embedding_tensor = embedding_tensor / norm
        embedding = normalized_embedding_tensor.cpu().to(torch.float32).numpy().squeeze()
        return embedding

    except Exception as e:
        img_name = getattr(processed_img_pil, 'filename', 'UNKNOWN_PREPROCESSED')
        # Print traceback for better debugging
        import traceback
        print(f"Error during single view embedding generation for {img_name}:")
        traceback.print_exc() # Print the full traceback to see where the error occurs
        return None

@torch.no_grad()
def get_avg_crop_embedding(raw_img_pil, processor, model, device, dtype, target_size, resize_factor):
    """Gets 5-crop averaged embedding."""
    try:
        crop_size = target_size
        resize_target_size = int(crop_size * resize_factor)
        if resize_target_size < crop_size: resize_target_size = crop_size # Sanity check

        # Manual Resize
        if min(raw_img_pil.size) != resize_target_size:
             scale = resize_target_size / min(raw_img_pil.size)
             new_size = (int(round(raw_img_pil.width * scale)), int(round(raw_img_pil.height * scale)))
             if min(new_size) < crop_size: # Upscale if needed to make cropping possible
                  scale = crop_size / min(raw_img_pil.size)
                  new_size = (int(math.ceil(raw_img_pil.width * scale)), int(math.ceil(raw_img_pil.height * scale)))
             img_resized = raw_img_pil.resize(new_size, Image.Resampling.LANCZOS)
        else:
             img_resized = raw_img_pil

        # Manual Five Crop
        crops = F.five_crop(img_resized, (crop_size, crop_size))

        # Get embedding for each crop
        all_embeddings = []
        for i, crop in enumerate(crops):
             # Reuse single view function, passing the crop
             embedding = get_single_view_embedding(crop, processor, model, device, dtype)
             if embedding is not None:
                  all_embeddings.append(embedding)
             else:
                  print(f"Warning: Failed to get embedding for crop {i+1}.")

        if not all_embeddings:
            print("Error: No crops processed successfully for avg_crop.")
            return None

        # Average and Normalize the final average embedding
        avg_embedding = np.mean(np.array(all_embeddings), axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm == 0: print("Warning: Avg embedding is zero vector."); return avg_embedding
        avg_embedding = avg_embedding / norm
        return avg_embedding

    except Exception as e:
        img_name = getattr(raw_img_pil, 'filename', 'UNKNOWN')
        print(f"Error during avg_crop processing for {img_name}: {e}")
        return None

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
    # --- End Precision Setup ---

    # --- Initialize Model ---
    # init_vision_model now disables processor auto-preprocessing
    processor, vision_model, model_image_size = init_vision_model(args.model_name, TARGET_DEV, COMPUTE_DTYPE)
    # --- End Model Init ---

    # --- Determine Output Directory ---
    model_name_safe = args.model_name.split('/')[-1].replace('-', '_') # Make safer for dir names
    # Add preprocessing mode to directory name
    mode_suffix_map = {'fit_pad': "FitPad", 'center_crop': "CenterCrop",
                       'avg_crop': "AvgCrop", 'naflex_resize': "NaflexResize"} # <-- Add naflex_resize
    mode_suffix = mode_suffix_map.get(args.preprocess_mode, "UnknownMode")
    output_subdir_name = f"{model_name_safe}_{mode_suffix}{args.output_dir_suffix}"
    final_output_dir = os.path.join(args.output_dir_root, output_subdir_name)
    print(f"\nSelected Preprocessing Mode: {args.preprocess_mode}")
    print(f"Embeddings will be saved in: {final_output_dir}")
    # --- End Output Dir ---

    # --- Process Source Folders ---
    source_subfolders = sorted([d for d in os.listdir(args.image_dir) if os.path.isdir(os.path.join(args.image_dir, d))])
    if not source_subfolders: exit(f"Error: No subfolders found in image directory: {args.image_dir}")
    print(f"Found source subfolders: {source_subfolders}")

    for src_folder in source_subfolders:
        current_image_dir = os.path.join(args.image_dir, src_folder)
        target_label = src_folder # Assuming folder name is label (0, 1, etc.)
        current_output_subdir = os.path.join(final_output_dir, target_label)
        os.makedirs(current_output_subdir, exist_ok=True)
        print(f"\nProcessing: {current_image_dir} -> {current_output_subdir}")

        image_files = [f for f in os.listdir(current_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not image_files: print("  No image files found."); continue

        for fname in tqdm(image_files, desc=f"Folder '{src_folder}'"):
            embedding_result = None
            try:
                img_path = os.path.join(current_image_dir, fname)
                raw_img_pil = Image.open(img_path).convert("RGB")

                # --- Apply Chosen Preprocessing and Get Embedding ---
                processed_img_for_embedding = None
                actual_num_patches = None # <<< Initialize patch count variable
                if args.preprocess_mode == 'fit_pad':
                    processed_img = preprocess_fit_pad(raw_img_pil, target_size=model_image_size)
                    embedding_result = get_single_view_embedding(processed_img, processor, vision_model, TARGET_DEV, COMPUTE_DTYPE)
                elif args.preprocess_mode == 'center_crop':
                    processed_img = preprocess_center_crop(raw_img_pil, target_size=model_image_size)
                    embedding_result = get_single_view_embedding(processed_img, processor, vision_model, TARGET_DEV, COMPUTE_DTYPE)
                elif args.preprocess_mode == 'naflex_resize':
                    # <<< v4.1: Get image AND patch count >>>
                    resize_output = preprocess_naflex_resize(raw_img_pil)  # Get the tuple first
                    # print(
                    #    f"DEBUG: Output of preprocess_naflex_resize: {type(resize_output)}, Value: {resize_output}")  # Check the output type
                    if isinstance(resize_output, tuple) and len(resize_output) == 2:
                        processed_img_for_embedding, actual_num_patches = resize_output  # Unpack the tuple
                        # print(
                        #    f"DEBUG: Unpacked - Type of processed_img_for_embedding: {type(processed_img_for_embedding)}")
                        # print(f"DEBUG: Unpacked - Value of actual_num_patches: {actual_num_patches}")
                    else:
                        print("ERROR: preprocess_naflex_resize did not return a tuple of (image, patches)!")
                        processed_img_for_embedding = None
                        actual_num_patches = None

                if args.preprocess_mode == 'avg_crop':
                    embedding_result = get_avg_crop_embedding(raw_img_pil, processor, vision_model, TARGET_DEV, COMPUTE_DTYPE,
                                                             target_size=model_image_size, resize_factor=args.resize_factor_avg_crop)
                elif processed_img_for_embedding is not None and actual_num_patches is not None:  # Make sure both exist for naflex
                    # <<< v4.1: Pass actual_num_patches if available (needed for naflex) >>>
                    num_patches_for_processor = actual_num_patches  # Use the unpacked value directly
                    print(
                        f"DEBUG: Calling get_single_view_embedding with image type {type(processed_img_for_embedding)} and patch count {num_patches_for_processor}")
                    embedding_result = get_single_view_embedding(
                        processed_img_for_embedding,  # Should be PIL image
                        num_patches_for_processor,  # Should be int (972)
                        processor,
                        vision_model,
                        TARGET_DEV,
                        COMPUTE_DTYPE
                    )
                    # --- End v4.1 Change ---
                else:
                    # Handle cases where preprocessing failed or mode is invalid
                    print(f"Warning: Preprocessing failed or invalid mode for {fname}. Skipping embedding.")
                    embedding_result = None
                # --- End Embedding Logic ---

                # Save the result
                if embedding_result is not None:
                    base_fname = os.path.splitext(fname)[0]
                    output_path = os.path.join(current_output_subdir, f"{base_fname}.npy")
                    np.save(output_path, embedding_result)
                else:
                    print(f"Warning: Skipping save for {fname} due to embedding error.")
            except Exception as e:
                print(f"\nError processing image {fname}: {e}. Skipping.")
                continue
        # --- End Image Loop ---
    # --- End Folder Loop ---

    # --- REMOVED test.npy CREATION ---

    print("\nEmbedding generation complete!")