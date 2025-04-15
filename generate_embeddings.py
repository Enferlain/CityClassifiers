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
# MOD v3.1.0: Updated function to use AutoClasses
def init_vision_model(model_name, device, dtype):
    print(f"Initializing Vision Model using AutoClasses: {model_name} on {device} with dtype {dtype}")
    try:
        # MOD v3.1.0: Use AutoProcessor and AutoModel
        processor = AutoProcessor.from_pretrained(model_name)
        # MOD v3.1.0: Add trust_remote_code=True, often needed for AutoModel with newer architectures
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device).eval()
        print(f"Loaded model class: {model.__class__.__name__}")
        print(f"Loaded processor class: {processor.__class__.__name__}")

        # Determine image_size from processor config reliably
        if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'size'):
             # Handle newer processors where size might be nested
             proc_sz = processor.image_processor.size
             if isinstance(proc_sz, dict):
                 # Common keys are 'shortest_edge', 'height', 'width', 'crop_size'
                 image_size = proc_sz.get("shortest_edge", proc_sz.get("height", proc_sz.get("crop_size", 224)))
             elif isinstance(proc_sz, int):
                 image_size = proc_sz
             else: image_size = 224
        elif hasattr(processor, 'size'): # Older style
             proc_sz = processor.size
             if isinstance(proc_sz, dict): image_size = proc_sz.get("shortest_edge", proc_sz.get("height", 224))
             elif isinstance(proc_sz, int): image_size = proc_sz
             else: image_size = 224
        else:
            print("Warning: Could not determine processor image size automatically. Defaulting to 224.")
            image_size = 224
        # Ensure image_size is int
        image_size = int(image_size)

        print(f"Determined Model Input Size: {image_size}x{image_size}")

        return processor, model, image_size
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


    # Determine Output Directory (same as v3.0.0)
    model_name_safe = args.model_name.split('/')[-1]
    output_subdir_name = f"{model_name_safe}{args.output_dir_suffix}"
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

                # MOD v3.1.0: Call the renamed embedding function
                avg_embedding = get_5crop_avg_embedding_from_features(
                    img_pil, processor, vision_model, TARGET_DEV, COMPUTE_DTYPE, RESIZE_TARGET, CROP_SIZE
                )

                if avg_embedding is not None:
                    base_fname = os.path.splitext(fname)[0]
                    output_path = os.path.join(current_output_subdir, f"{base_fname}.npy")
                    np.save(output_path, avg_embedding)
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