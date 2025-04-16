# Version 3.1.0: Use AutoClasses and get_image_features based on SigLIP 2 model card

import os
import torch
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
# MOD v3.1.0: Import AutoClasses
from transformers import AutoProcessor, AutoModel
import argparse
import math

# --- Default Configuration ---
DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14-336" # Still default to CLIP for backward compatibility
TARGET_DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_PRECISION = torch.float32

# --- Argument Parsing ---
# (Keep parse_gen_args function exactly the same as v3.0.0)
def parse_gen_args():
    parser = argparse.ArgumentParser(description="Generate vision model embeddings for image folders (5-crop avg).")
    parser.add_argument('--image_dir', required=True,
                        help="Directory containing subfolders (e.g., '0', '1' or 'good', 'bad') with images.")
    parser.add_argument('--output_dir_root', required=True,
                        help="Root directory to save embeddings (e.g., 'data'). Subfolder based on model name will be created.")
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_ID,
                        help=f"Name of the vision model from Hugging Face (e.g., openai/clip..., google/siglip...). Default: {DEFAULT_MODEL_ID}")
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'bf16', 'fp16'],
                        help="Precision for model computation (fp32, bf16, fp16). Default: fp32")
    parser.add_argument('--resize_factor', type=float, default=2.0,
                        help="Factor to multiply model's native image size by for resizing before cropping (e.g., 2.0 for 5-crop). Default: 2.0")
    parser.add_argument('--output_dir_suffix', type=str, default="",
                        help="Optional suffix to append to the automatically generated output directory name.")
    return parser.parse_args()

# --- Model Initialization ---
# MOD v3.1.3: Added logic to disable processor resize/crop for fit-and-pad
def init_vision_model(model_name, device, dtype):
    """
    Initializes vision model and processor using AutoClasses.
    Modifies the processor to disable automatic resizing and cropping,
    as preprocessing (fit-and-pad) will be done manually before calling the processor.
    """
    print(f"Initializing Vision Model using AutoClasses: {model_name} on {device} with dtype {dtype}")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        # Add trust_remote_code=True for broader compatibility
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()
        print(f"Loaded model class: {model.__class__.__name__}")
        print(f"Loaded processor class: {processor.__class__.__name__}")

        # --- DISABLE PROCESSOR RESIZE/CROP ---
        # Access the image processor component within the AutoProcessor
        # Handle potential variations in attribute naming
        if hasattr(processor, 'image_processor'):
             image_processor = processor.image_processor
             print(f"DEBUG GenerateEmbed: Original image processor config: {image_processor}") # Debug print
             # Common attributes to disable preprocessing steps
             if hasattr(image_processor, 'do_resize'): image_processor.do_resize = False
             if hasattr(image_processor, 'do_center_crop'): image_processor.do_center_crop = False
             # Less common, but check just in case
             if hasattr(image_processor, 'do_rescale'): image_processor.do_rescale = False
             if hasattr(image_processor, 'do_normalize'): image_processor.do_normalize = True # Keep True! Normalization needed.
             print(f"DEBUG GenerateEmbed: Modified image processor config: {image_processor}") # Verify changes
        else:
             print("Warning: Could not access 'image_processor' attribute on the loaded processor. Cannot disable resize/crop.")
        # --- END DISABLE ---

        # Determine image_size (the target size the model expects)
        # (Keep the logic from previous versions to determine this, default if needed)
        image_size = 512 # Assuming SigLIP 512 model for now
        if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'size'):
             proc_sz = processor.image_processor.size
             if isinstance(proc_sz, dict):
                 image_size = int(proc_sz.get("shortest_edge", proc_sz.get("height", proc_sz.get("crop_size", 512))))
             elif isinstance(proc_sz, int):
                 image_size = int(proc_sz)
        elif hasattr(processor, 'size'): # Older style
             proc_sz = processor.size
             if isinstance(proc_sz, dict): image_size = int(proc_sz.get("shortest_edge", proc_sz.get("height", 512)))
             elif isinstance(proc_sz, int): image_size = int(proc_sz)

        print(f"Model Input Target Size: {image_size}x{image_size}")

        return processor, model, image_size # Return image_size too
    except Exception as e:
        print(f"Error initializing vision model {model_name}: {e}")
        raise

# --- Embedding Generation for 5 Crops ---
@torch.no_grad()
# MOD v3.1.0: Renamed function slightly for clarity
def get_5crop_avg_embedding_from_features(img_pil, processor, model, device, dtype, resize_target_size, crop_size):
    """
    Processes a single PIL image: resize, 5-crop, get embeddings via get_image_features, average.
    Returns a single numpy array (averaged embedding) or None on error.
    """
    try:
        # 1. Resize (same as before)
        if min(img_pil.size) != resize_target_size:
             scale = resize_target_size / min(img_pil.size)
             new_size = (int(round(img_pil.width * scale)), int(round(img_pil.height * scale)))
             if min(new_size) < crop_size:
                  scale = crop_size / min(img_pil.size)
                  new_size = (int(math.ceil(img_pil.width * scale)), int(math.ceil(img_pil.height * scale)))
             img_resized = img_pil.resize(new_size, Image.Resampling.LANCZOS)
        else:
             img_resized = img_pil

        # 2. Five Crop (same as before)
        crop_size_int = int(crop_size)
        crops = F.five_crop(img_resized, (crop_size_int, crop_size_int))

        # 3. Process each crop
        all_embeddings = []
        for crop in crops:
            try:
                inputs = processor(images=[crop], return_tensors="pt")["pixel_values"].to(device=device, dtype=dtype)

                # MOD v3.1.0: Use get_image_features instead of forward pass + output parsing
                # Note: Ensure model is the base AutoModel, not just vision tower if different classes were used
                if hasattr(model, 'get_image_features'):
                    embedding_tensor = model.get_image_features(pixel_values=inputs)
                    embedding = embedding_tensor.cpu().to(torch.float32).numpy().squeeze()
                    # Optional: Normalize L2 norm? Check if SigLIP features are pre-normalized
                    # embedding = embedding / np.linalg.norm(embedding)
                    all_embeddings.append(embedding)
                else:
                    # Fallback if loaded model doesn't have get_image_features (e.g., loaded only VisionTower)
                    print("Warning: model does not have get_image_features. Falling back to forward pass (may be incorrect).")
                    outputs = model(pixel_values=inputs)
                    if hasattr(outputs, 'pooler_output'):
                         embedding = outputs.pooler_output.cpu().to(torch.float32).numpy().squeeze()
                         all_embeddings.append(embedding)
                    elif hasattr(outputs, 'image_embeds'): # Should not happen with AutoModel usually
                         embedding = outputs.image_embeds.cpu().to(torch.float32).numpy().squeeze()
                         all_embeddings.append(embedding)
                    else:
                         print("Error: Cannot extract features. Skipping crop.")
                         continue # Skip crop

            except Exception as e:
                print(f"Error processing crop: {e}. Skipping crop.")
                continue

        if not all_embeddings:
            print("No crops processed successfully for this image.")
            return None

        # 4. Average Embeddings (same as before)
        avg_embedding = np.mean(np.array(all_embeddings), axis=0)

        # MOD v3.1.1: Add L2 normalization to the averaged embedding
        norm = np.linalg.norm(avg_embedding)
        if norm == 0:
            # Handle zero vector case to avoid division by zero
            print(f"Warning: Averaged embedding resulted in a zero vector for image {img_pil.filename if hasattr(img_pil, 'filename') else 'UNKNOWN'}. Returning zero vector.")
            # Return avg_embedding (which is zeros) or handle as error?
        else:
            avg_embedding = avg_embedding / norm
            # print(f"DEBUG: Normalized avg embedding norm: {np.linalg.norm(avg_embedding):.4f}") # Should be ~1.0

        return avg_embedding

    except Exception as e:
        print(f"Error during 5-crop processing for image: {e}")
        return None

# v3.1.2: New function for single global embedding
@torch.no_grad()
def get_single_embedding_from_features(img_pil, processor, model, device, dtype):
    """
    Processes a single PIL image: uses processor directly, get embedding via get_image_features, normalize.
    Returns a single numpy array (normalized embedding) or None on error.
    """
    try:
        # 1. Process image using the processor directly
        #    The processor should handle resizing/cropping/padding as needed for the model.
        inputs = processor(images=[img_pil], return_tensors="pt").to(device=device)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
             raise ValueError("Processor did not return 'pixel_values'. Check processor output.")
        # Ensure pixel_values have the correct dtype for the model
        pixel_values = pixel_values.to(dtype=dtype)

        # 2. Get image features
        if hasattr(model, 'get_image_features'):
            embedding_tensor = model.get_image_features(pixel_values=pixel_values)
            # Note: SigLIP features might already be normalized, but let's ensure it.
            # Normalize the single embedding vector
            norm = torch.linalg.norm(embedding_tensor, dim=-1, keepdim=True)
            normalized_embedding_tensor = embedding_tensor / (norm + 1e-8) # Add epsilon for safety

            embedding = normalized_embedding_tensor.cpu().to(torch.float32).numpy().squeeze() # Squeeze batch dim
            # print(f"DEBUG: Single embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}") # Optional debug
            return embedding
        else:
            print("Error: model does not have get_image_features. Cannot extract single embedding.")
            return None

    except Exception as e:
        img_name = getattr(img_pil, 'filename', 'UNKNOWN')
        print(f"Error during single embedding processing for image {img_name}: {e}")
        return None

def preprocess_fit_pad(img_pil, target_size=512, fill_color=(0, 0, 0)):
    """
    Resizes an image to fit within target_size maintaining aspect ratio,
    then pads with fill_color to reach target_size.
    """
    original_width, original_height = img_pil.size
    target_w, target_h = target_size, target_size

    # Calculate scaling factor to fit largest dimension
    scale = min(target_w / original_width, target_h / original_height)

    # Calculate new size
    new_w = int(original_width * scale)
    new_h = int(original_height * scale)

    # Resize image
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create new black canvas
    img_padded = Image.new(img_pil.mode, (target_w, target_h), fill_color)

    # Calculate padding offset (integer division)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    # Paste resized image onto canvas
    img_padded.paste(img_resized, (pad_left, pad_top))

    return img_padded


# --- Main Processing Logic ---
if __name__ == "__main__":
    args = parse_gen_args()

    # Determine Precision (same as v3.0.0)
    precision_map = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
    COMPUTE_DTYPE = precision_map.get(args.precision, torch.float32)
    # (Hardware/CPU checks remain the same)
    if COMPUTE_DTYPE != torch.float32 and TARGET_DEV == 'cpu':
        print(f"Warning: {args.precision} requested but device is CPU. Using float32.")
        COMPUTE_DTYPE = torch.float32
    if COMPUTE_DTYPE == torch.bfloat16 and (TARGET_DEV != 'cuda' or not torch.cuda.is_bf16_supported()):
         print("Warning: bf16 requested but not supported by hardware/device. Using float32.")
         COMPUTE_DTYPE = torch.float32
    if COMPUTE_DTYPE == torch.float16 and TARGET_DEV == 'cpu':
         print("Warning: fp16 on CPU selected. This might be slow or unstable. Consider fp32.")

    # Initialize Model using new function
    processor, vision_model, model_image_size = init_vision_model(args.model_name, TARGET_DEV, COMPUTE_DTYPE)

    # Determine Crop and Resize Sizes (same as v3.0.0)
    CROP_SIZE = model_image_size
    RESIZE_TARGET = int(CROP_SIZE * args.resize_factor)
    if RESIZE_TARGET < CROP_SIZE:
        print(f"Warning: Calculated resize target ({RESIZE_TARGET}) is smaller than CROP_SIZE ({CROP_SIZE}). Adjusting resize target to {CROP_SIZE}.")
        RESIZE_TARGET = CROP_SIZE
    print(f"Using Crop Size: {CROP_SIZE}, Resize Target (before crop): {RESIZE_TARGET}")

    # Inside generate_embeddings.py main block (around line 140)
    model_name_safe = args.model_name.split('/')[-1]
    # New suffix for these embeddings
    output_subdir_name = f"{model_name_safe}_FitPad{args.output_dir_suffix}"  # <<< CHANGED
    final_output_dir = os.path.join(args.output_dir_root, output_subdir_name)
    print(f"Embeddings will be saved in: {final_output_dir}")

    # Process Source Subfolders (same as v3.0.0)
    source_subfolders = sorted([d for d in os.listdir(args.image_dir) if os.path.isdir(os.path.join(args.image_dir, d))])
    if not source_subfolders: exit(f"Error: No subfolders found in image directory: {args.image_dir}")
    print(f"Found source subfolders (assuming class labels): {source_subfolders}")

    for src_folder in source_subfolders:
        current_image_dir = os.path.join(args.image_dir, src_folder)
        target_label = src_folder
        current_output_subdir = os.path.join(final_output_dir, target_label)
        os.makedirs(current_output_subdir, exist_ok=True)
        print(f"\nProcessing images in: {current_image_dir}")
        print(f"Saving embeddings to: {current_output_subdir}")

        image_files = [f for f in os.listdir(current_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not image_files: print("No image files found."); continue

        for fname in tqdm(image_files, desc=f"Folder '{src_folder}'"):
            try:
                img_path = os.path.join(current_image_dir, fname)
                img_pil = Image.open(img_path).convert("RGB")

                # --- Apply Fit and Pad ---
                img_processed_pil = preprocess_fit_pad(img_pil, target_size=model_image_size)
                # --- End Apply Fit and Pad ---

                # Get embedding from the preprocessed image
                # (Using the existing single embedding function, which now receives a 512x512 image)
                embedding = get_single_embedding_from_features(
                    img_processed_pil, processor, vision_model, TARGET_DEV, COMPUTE_DTYPE
                )
                # Note: get_single_embedding_from_features calls processor() internally,
                # which will now only do normalization thanks to the modifications in init_vision_model.

                # Save the result
                if embedding is not None:
                    base_fname = os.path.splitext(fname)[0]
                    output_path = os.path.join(current_output_subdir, f"{base_fname}.npy")
                    np.save(output_path, embedding)
                else:
                    print(f"Skipping save for {fname} due to processing error.")
            except Exception as e:
                print(f"\nError processing image {fname}: {e}. Skipping.")
                continue

    # Create dummy test.npy (same as v3.0.0)
    created_test_file = False
    for src_folder in source_subfolders:
        class_output_dir = os.path.join(final_output_dir, src_folder)
        try:
            npy_files = [f for f in os.listdir(class_output_dir) if f.endswith('.npy')]
            if npy_files:
                src_test_emb = os.path.join(class_output_dir, npy_files[0])
                dst_test_emb = os.path.join(final_output_dir, "test.npy")
                import shutil
                shutil.copy2(src_test_emb, dst_test_emb)
                print(f"\nCreated test.npy in {final_output_dir} using embedding from {npy_files[0]}")
                created_test_file = True
                break
        except Exception as e:
            print(f"\nWarning: Error finding/creating test.npy in {class_output_dir}: {e}")
            continue
    if not created_test_file: print(f"\nWarning: Could not create test.npy in {final_output_dir}.")

    print("\nEmbedding generation complete!")