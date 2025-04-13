# generate_clip_embeddings_5crop_avg.py
# Version 2.0.0: Implements averaged 5-crop strategy

import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import argparse

# --- Configuration ---
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
TARGET_DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_DTYPE = torch.float16 if TARGET_DEV == "cuda" else torch.float32


# --- Argument Parsing ---
def parse_gen_args():
    parser = argparse.ArgumentParser(description="Generate averaged 5-crop CLIP embeddings for image folders.")
    parser.add_argument('--image_dir', required=True,
                        help="Directory containing subfolders (e.g., 'good', 'bad') with images.")
    parser.add_argument('--output_dir', required=True,
                        help="Root directory to save embeddings (e.g., 'data/CLIP-Anatomy').")
    # Removed batch_size, process one image at a time due to 5 crops
    # parser.add_argument('--batch_size', type=int, default=1, help="Batch size for CLIP processing (per crop).")
    parser.add_argument('--resize_shortest', type=int, default=672,
                        help="Resize shortest edge to this size before cropping.")
    return parser.parse_args()


# --- CLIP Initialization ---
def init_clip(device, dtype):
    print(f"Initializing CLIP: {CLIP_MODEL_ID} on {device} with dtype {dtype}")
    try:
        processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
        # Get the target input size expected by the processor (e.g., 336)
        clip_input_size = processor.size.get("shortest_edge", 224)  # Default needed? Check processor config
        # Handle cases where size might be under 'height'/'width' or just 'size'
        if isinstance(clip_input_size, dict):  # e.g. {'height': 336, 'width': 336}
            clip_input_size = clip_input_size.get('height', 224)
        elif isinstance(clip_input_size, (tuple, list)):  # e.g. [336, 336]
            clip_input_size = clip_input_size[0]

        print(f"CLIP processor expects input size: {clip_input_size}x{clip_input_size}")

        model = CLIPVisionModelWithProjection.from_pretrained(
            CLIP_MODEL_ID,
            torch_dtype=dtype,
        ).to(device).eval()
        return processor, model, clip_input_size
    except Exception as e:
        print(f"Error initializing CLIP: {e}")
        exit(1)


# --- Embedding Generation for 5 Crops ---
@torch.no_grad()
def get_5crop_avg_embedding(img_pil, processor, model, device, dtype, resize_shortest_edge, crop_size):
    """
    Processes a single PIL image: resize, 5-crop, get embeddings, average.
    Returns a single numpy array (averaged embedding) or None on error.
    """
    try:
        # 1. Resize shortest edge
        scale = resize_shortest_edge / min(img_pil.size)
        new_size = (int(round(img_pil.width * scale)), int(round(img_pil.height * scale)))
        img_resized = img_pil.resize(new_size, Image.Resampling.LANCZOS)  # Use LANCZOS for better quality downscale

        # 2. Five Crop - takes size tuple (h, w)
        crops = TF.five_crop(img_resized, (crop_size, crop_size))  # e.g., 336x336 crops

        # 3. Process each crop
        all_embeddings = []
        for crop in crops:
            try:
                # Processor expects list of images, feed one crop at a time
                inputs = processor(images=[crop], return_tensors="pt")["pixel_values"].to(device=device, dtype=dtype)
                outputs = model(pixel_values=inputs)
                embedding = outputs.image_embeds.cpu().to(torch.float32).numpy().squeeze()  # Get single embedding
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing crop: {e}. Skipping crop.")
                continue  # Skip this crop if it fails

        # 4. Average Embeddings
        if not all_embeddings:  # Check if any crops were processed successfully
            print("No crops processed successfully for this image.")
            return None

        avg_embedding = np.mean(np.array(all_embeddings), axis=0)
        return avg_embedding

    except Exception as e:
        print(f"Error during 5-crop processing for image: {e}")
        return None  # Return None to indicate failure


# --- Main Processing Logic ---
if __name__ == "__main__":
    args = parse_gen_args()

    # Determine crop size based on CLIP input size
    try:
        temp_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
        clip_input_size = temp_processor.size.get("shortest_edge", 224)
        if isinstance(clip_input_size, dict):
            clip_input_size = clip_input_size.get('height', 224)
        elif isinstance(clip_input_size, (tuple, list)):
            clip_input_size = clip_input_size[0]
        del temp_processor  # Free memory
        CROP_SIZE = clip_input_size  # Crop size should match CLIP input size
        print(f"Determined CROP_SIZE: {CROP_SIZE}")
    except Exception as e:
        print(f"Could not determine CLIP input size automatically, defaulting crop size to 336. Error: {e}")
        CROP_SIZE = 336  # Fallback default for ViT-L/14-336

    # Check if resize_shortest makes sense
    if args.resize_shortest < CROP_SIZE:
        print(
            f"Warning: resize_shortest ({args.resize_shortest}) is smaller than CROP_SIZE ({CROP_SIZE}). Adjusting resize_shortest to {CROP_SIZE}.")
        args.resize_shortest = CROP_SIZE

    processor, clip_model, _ = init_clip(TARGET_DEV, CLIP_DTYPE)  # Don't need size from here anymore

    source_folders = ["bad", "good"]  # Subfolders within args.image_dir
    target_labels = ["0", "1"]  # Corresponding output folders in args.output_dir

    for src_folder, target_label in zip(source_folders, target_labels):
        current_image_dir = os.path.join(args.image_dir, src_folder)
        current_output_dir = os.path.join(args.output_dir, target_label)

        if not os.path.isdir(current_image_dir):
            print(f"Warning: Source image directory not found: {current_image_dir}. Skipping.")
            continue

        os.makedirs(current_output_dir, exist_ok=True)
        print(f"\nProcessing images in: {current_image_dir}")
        print(f"Saving averaged embeddings to: {current_output_dir}")

        image_files = [f for f in os.listdir(current_image_dir) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        if not image_files:
            print("No image files found.")
            continue

        # Process one image at a time now
        for fname in tqdm(image_files, desc=f"Folder '{src_folder}'"):
            try:
                img_path = os.path.join(current_image_dir, fname)
                img_pil = Image.open(img_path).convert("RGB")

                avg_embedding = get_5crop_avg_embedding(
                    img_pil,
                    processor,
                    clip_model,
                    TARGET_DEV,
                    CLIP_DTYPE,
                    args.resize_shortest,
                    CROP_SIZE
                )

                if avg_embedding is not None:
                    base_fname = os.path.splitext(fname)[0]
                    output_path = os.path.join(current_output_dir, f"{base_fname}.npy")
                    np.save(output_path, avg_embedding)
                else:
                    print(f"Skipping save for {fname} due to processing error.")

            except Exception as e:
                print(f"\nError processing image {fname}: {e}. Skipping.")
                continue

    # --- Create dummy test.npy ---
    # Simple approach: find one good embedding and copy it
    test_output_dir = args.output_dir
    good_emb_dir = os.path.join(args.output_dir, "1")
    try:
        good_files = [f for f in os.listdir(good_emb_dir) if f.endswith('.npy')]
        if good_files:
            src_test_emb = os.path.join(good_emb_dir, good_files[0])
            dst_test_emb = os.path.join(test_output_dir, "test.npy")
            import shutil

            shutil.copy2(src_test_emb, dst_test_emb)
            print(f"\nCreated test.npy using {good_files[0]}")
        else:
            print("\nWarning: Could not create test.npy, no embeddings found in 'good' directory.")
    except Exception as e:
        print(f"\nWarning: Error creating test.npy: {e}")
    # --- End dummy test.npy ---

    print("\nAveraged 5-crop embedding generation complete!")