# v1.1: Improved config handling, argument clarity, pipeline call

import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import shutil # Use shutil.copy2 for better metadata copying
import json # Needed for reading config to get labels

# Make sure inference classes are imported correctly
try:
    from inference import CityAestheticsPipeline, CityClassifierPipeline, _load_config_helper # Import helper
except ImportError:
    print("Error: Could not import pipeline classes or helpers from inference.py.")
    print("Ensure inference.py is in the same directory or accessible in PYTHONPATH.")
    exit(1)

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]

def parse_args():
    parser = argparse.ArgumentParser(description="Test model by running it on an entire folder")
    parser.add_argument('--src', required=True, help="Folder with images to score/classify")
    parser.add_argument('--dst', default="output_passed", help="Folder to copy images that pass the threshold")
    parser.add_argument('--model', required=True, help="Path to the .safetensors model checkpoint file")
    # NEW: Optional explicit config path
    parser.add_argument('--config', default=None, help="(Optional) Path to the model's .config.json file. If None, attempts to infer from model path.")
    parser.add_argument("--arch", choices=["score", "class"], required=True, help="Model type (score or class)") # Made required
    # Args for filtering/copying
    parser.add_argument('--min_score', type=int, default=0, help="Lower limit (inclusive) for score/probability percentage")
    parser.add_argument('--max_score', type=int, default=100, help="Upper limit (inclusive) for score/probability percentage")
    # Changed --label to be clearer for classifiers
    parser.add_argument('--target_label_name', type=str, default=None,
                        help="For 'class' arch: The *name* of the target label (e.g., 'Good Anatomy') whose probability score to filter/display. Required if arch is class.")
    parser.add_argument('--copy_passed', action=argparse.BooleanOptionalAction, default=False, help="Copy images meeting the score threshold to the dst folder")
    parser.add_argument('--keep_structure', action=argparse.BooleanOptionalAction, default=False, help="When copying, keep original subfolder structure within dst")
    # Added Tiling Args
    parser.add_argument('--use_tiling', action=argparse.BooleanOptionalAction, default=True, help="Enable tiling for classifier inference (default: True)")
    parser.add_argument('--tile_strategy', choices=["mean", "median", "max", "min"], default="mean", help="Tiling combination strategy (default: mean)")

    args = parser.parse_args()

    # Validation for classifier target label
    if args.arch == "class" and args.target_label_name is None:
        parser.error("--target_label_name is required when --arch is class.")

    return args

# v1.2: Pass tiling args correctly using keywords
def process_file(pipeline, config_labels, args, src_path, dst_path):
    """Processes a single image file."""
    try:
        img = Image.open(src_path)
    except Exception as e:
        print(f"\nError opening image {src_path}: {e}. Skipping.")
        return

    try:
        # Pass tiling arguments as KEYWORDS using values from args object
        # v1.3: Corrected keyword for first argument
        if args.arch == "class":
             pred_dict = pipeline(
                 raw_pil_image=img,             # <<< CHANGED raw= to raw_pil_image=
                 tiling=args.use_tiling,
                 tile_strat=args.tile_strategy
             )
        else: # Score pipeline __call__ takes only one positional arg
             pred_dict = pipeline(img)          # Keep positional for Aesthetics pipeline

        score_to_display = -1 # Default if error

        if args.arch == "score":
            # For score, pred_dict is just the float score
            score_to_display = int(pred_dict * 100)
        elif args.arch == "class":
            # For class, pred_dict is a dictionary {'Label Name': probability}
            # Find the probability for the target label name
            target_prob = pred_dict.get(args.target_label_name)

            if target_prob is not None:
                score_to_display = int(target_prob * 100)
            else:
                print(f"\nWarning: Target label '{args.target_label_name}' not found in prediction keys: {list(pred_dict.keys())} for image {os.path.basename(src_path)}")
                # Try finding by index as fallback (less reliable)
                label_index = None
                for idx, name in config_labels.items(): # Use labels from loaded config
                     if name == args.target_label_name:
                          label_index = idx
                          break
                if label_index is not None:
                     fallback_prob = pred_dict.get(label_index) # Try getting by index string
                     if fallback_prob is not None:
                          print(f"  Using fallback score based on index '{label_index}'.")
                          score_to_display = int(fallback_prob * 100)
                     else: print("  Fallback score by index also failed.")
                else: print("  Could not find corresponding index for target label name.")

    except Exception as e:
        print(f"\nError during prediction for {src_path}: {e}")
        score_to_display = -1 # Indicate error

    # --- Display and Copy Logic ---
    if score_to_display != -1:
        tqdm.write(f" {score_to_display:>3}% [{os.path.basename(src_path)}]" + (f" ({args.target_label_name})" if args.arch == 'class' else ""))
        if args.min_score <= score_to_display <= args.max_score:
            if dst_path:
                try:
                    # Ensure destination directory exists
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path) # copy2 preserves more metadata
                except Exception as e_copy:
                    print(f"\nError copying {src_path} to {dst_path}: {e_copy}")
    else:
        # Only print error cases if prediction failed
        tqdm.write(f" ERR [{os.path.basename(src_path)}]")


def process_folder(pipeline, config_labels, args):
    """Walks through source folder and processes image files."""
    print(f"Processing images in: {args.src}")
    if args.copy_passed:
        print(f"Copying images with score [{args.min_score}-{args.max_score}]% to: {args.dst}")
    if args.target_label_name:
         print(f"Filtering based on probability of label: '{args.target_label_name}'")

    processed_count = 0
    copied_count = 0
    error_count = 0

    # Use tqdm for overall progress tracking
    file_list = []
    for path, _, files in os.walk(args.src):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                file_list.append(os.path.join(path, fname))

    if not file_list:
         print("No image files found in source directory.")
         return

    with tqdm(total=len(file_list), desc="Processing folder") as pbar:
        for src_path in file_list:
            dst_path = None
            if args.copy_passed:
                dst_dir = args.dst
                if args.keep_structure:
                    src_rel = os.path.relpath(os.path.dirname(src_path), args.src)
                    if src_rel != ".":
                        dst_dir = os.path.join(args.dst, src_rel)
                dst_path = os.path.join(dst_dir, os.path.basename(src_path))

            try:
                # Pass necessary args to process_file
                process_file(pipeline, config_labels, args, src_path, dst_path)
                processed_count += 1
                # Check if file was copied for counting (crude check based on dst_path existing and score range)
                # A more robust way would be to have process_file return a status
                if dst_path and os.path.exists(dst_path): # Assuming process_file copies if criteria met
                     # This might slightly overcount if copy fails but file exists later
                     copied_count += 1 # Simplistic count, accuracy depends on process_file behavior
            except Exception as e_proc:
                print(f"\nUnhandled error processing {src_path}: {e_proc}")
                error_count += 1
            finally:
                pbar.update(1)

    print(f"\nProcessing complete.")
    print(f"  Processed: {processed_count} images")
    if args.copy_passed: print(f"  Copied (estimate): {copied_count} images")
    if error_count > 0: print(f"  Errors: {error_count} images")


if __name__ == "__main__":
    args = parse_args()

    # --- Config Handling ---
    # Use provided config path OR infer from model path
    config_path = args.config
    if config_path is None:
        # Infer config path by removing potential suffixes like _best_val, _sXXXK, _efinal
        model_base_name = os.path.basename(args.model)
        # Simple suffix removal (can be made more robust)
        suffixes_to_remove = ["_best_val", "_efinal"]
        for suffix in suffixes_to_remove:
             if model_base_name.endswith(f"{suffix}.safetensors"):
                  model_base_name = model_base_name[:-len(f"{suffix}.safetensors")]
                  break
        # Remove step suffixes like _s28K
        if "_s" in model_base_name:
             model_base_name = model_base_name.split("_s")[0]

        inferred_config_path = os.path.join(os.path.dirname(args.model), f"{model_base_name}.config.json")
        if os.path.isfile(inferred_config_path):
            print(f"DEBUG: Using inferred config path: {inferred_config_path}")
            config_path = inferred_config_path
        else:
            print(f"Error: Could not infer config path from '{args.model}'. Please provide --config.")
            exit(1)
    elif not os.path.isfile(config_path):
         print(f"Error: Provided config path not found: {config_path}")
         exit(1)

    # Load the specified or inferred config
    config_data = _load_config_helper(config_path)
    if config_data is None:
        print(f"Error: Failed to load config data from {config_path}")
        exit(1)
    config_labels = config_data.get("labels", {}) # Get labels for classifier display fallback
    # --- End Config Handling ---

    os.makedirs(args.dst, exist_ok=True)
    print(f"Using model: {os.path.basename(args.model)}")
    print(f"Model architecture: {args.arch}")

    # --- Pipeline Setup ---
    pipeline_args = {}
    if torch.cuda.is_available():
        pipeline_args["device"] = "cuda"
        # Note: clip_dtype applies to the vision model, not the predictor head
        # Predictor head usually runs in fp32
        pipeline_args["clip_dtype"] = torch.float16 # Or float32 if preferred/needed

    pipeline = None
    try:
        if args.arch == "score":
            # Pass explicit config_path to ensure it uses the correct one
            pipeline = CityAestheticsPipeline(args.model, config_path=config_path, **pipeline_args)
        elif args.arch == "class":
            # Pass explicit config_path here too
            pipeline = CityClassifierPipeline(args.model, config_path=config_path, **pipeline_args)
        else:
            # This case should not be reachable due to argparse choices
             raise ValueError(f"Unknown model architecture '{args.arch}'")
    except Exception as e_pipe:
         print(f"\nError initializing pipeline: {e_pipe}")
         print("Ensure the model file, config file, and architecture type match.")
         exit(1)
    # --- End Pipeline Setup ---

    # Run processing
    process_folder(pipeline, config_labels, args)