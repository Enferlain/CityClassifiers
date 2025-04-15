# test_single_image.py
import json
import os
import torch
import argparse
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

# Assuming model.py is in the same directory or accessible
try:
    from model import PredictorModel
    from utils import get_embed_params # Import necessary function
except ImportError as e:
    print(f"Error importing PredictorModel or get_embed_params: {e}")
    print("Ensure model.py and utils.py are in the same directory or your PYTHONPATH is set correctly.")
    raise

# --- Argument Parsing ---
def parse_test_args():
    parser = argparse.ArgumentParser(description="Test a single image with an anatomy classifier.")
    parser.add_argument('--model_path', required=True, help="Path to the trained .safetensors model file.")
    parser.add_argument('--image_path', required=True, help="Path to the input image file.")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run on (cuda or cpu).")
    parser.add_argument('--precision', default='fp32', choices=['fp32', 'bf16', 'fp16'], help='Precision for CLIP model (fp32, bf16, fp16).')
    # Optional: Config path if needed, otherwise infer
    parser.add_argument('--config_path', default=None, help="(Optional) Path to the model's .config.json file.")
    return parser.parse_args()

# --- CLIP Initialization ---
def init_clip(device, dtype):
    print(f"Initializing CLIP...")
    try:
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14-336",
            torch_dtype=dtype,
        ).to(device).eval()
        print("CLIP Initialized.")
        return processor, model
    except Exception as e:
        print(f"Error initializing CLIP: {e}")
        raise

# --- Predictor Model Loading ---
def load_predictor_model(model_path, config_path, device):
    print(f"Loading predictor model from: {model_path}")
    # Try to load associated config if path not given
    if config_path is None:
        base_name = os.path.splitext(model_path)[0]
        inferred_config_path = f"{base_name}.config.json"
        if os.path.isfile(inferred_config_path):
            print(f"Found associated config: {inferred_config_path}")
            config_path = inferred_config_path
        else:
            print("Warning: No config file found or provided. Will attempt to infer model outputs.")

    # Load labels and model params from config (using a simplified helper here)
    outputs_from_config = None
    labels = { '0': '0', '1': '1' } # Default labels
    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                config_data = json.load(f)
            loaded_labels = config_data.get("labels")
            if loaded_labels:
                 labels = loaded_labels # Use labels from config if available
            model_params = config_data.get("model_params")
            if model_params and "outputs" in model_params:
                outputs_from_config = int(model_params["outputs"])
                print(f"Outputs from config: {outputs_from_config}")
        except Exception as e:
            print(f"Warning: Error loading config {config_path}: {e}")

    # Load state dict and determine outputs
    try:
        sd = load_file(model_path)
        # Determine outputs from state dict as fallback/check
        final_bias_key_new = "down.10.bias" # v1.1 arch
        final_bias_key_old = "down.5.bias"  # Original arch
        outputs_in_file = None
        if final_bias_key_new in sd: outputs_in_file = sd[final_bias_key_new].shape[0]
        elif final_bias_key_old in sd: outputs_in_file = sd[final_bias_key_old].shape[0]

        if outputs_in_file is None:
             raise KeyError("Could not determine model outputs from state_dict.")

        # Use config outputs if available, otherwise use file outputs
        final_outputs = outputs_from_config if outputs_from_config is not None else outputs_in_file
        if outputs_from_config is not None and outputs_from_config != outputs_in_file:
            print(f"Warning: Config outputs ({outputs_from_config}) != file outputs ({outputs_in_file}). Using config value.")

        if final_outputs <= 1:
             raise ValueError(f"Loaded model appears to be a scorer (outputs={final_outputs}), not a classifier.")

        # Instantiate model
        model_init_params = get_embed_params("CLIP") # Assuming CLIP base
        model_init_params["outputs"] = final_outputs
        predictor_model = PredictorModel(**model_init_params)

        # Load weights
        predictor_model.load_state_dict(sd, strict=True)
        predictor_model.to(device)
        predictor_model.eval()
        print(f"Predictor model loaded with {final_outputs} outputs.")
        return predictor_model, labels
    except Exception as e:
        print(f"Error loading predictor model: {e}")
        raise

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_test_args()

    # --- Setup Device and Precision ---
    device = torch.device(args.device)
    precision_map = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
    clip_dtype = precision_map.get(args.precision, torch.float32)
    if clip_dtype != torch.float32 and device.type == 'cpu':
        print(f"Warning: {args.precision} requested but device is CPU. Using float32 for CLIP.")
        clip_dtype = torch.float32
    if clip_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
         print("Warning: bf16 requested but not supported by hardware. Using float32 for CLIP.")
         clip_dtype = torch.float32


    # --- Initialize Models ---
    clip_processor, clip_model = init_clip(device, clip_dtype)
    predictor_model, labels = load_predictor_model(args.model_path, args.config_path, device)

    # --- Load and Process Image ---
    print(f"Processing image: {args.image_path}")
    try:
        pil_image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image_path}")
        exit(1)
    except Exception as e:
        print(f"Error opening image {args.image_path}: {e}")
        exit(1)

    # --- Get CLIP Embedding (No Tiling) ---
    inputs = clip_processor(images=pil_image, return_tensors="pt").to(device=device, dtype=clip_dtype)
    with torch.no_grad():
        try:
             clip_embeddings = clip_model(**inputs).image_embeds.to(dtype=torch.float32) # Use float32 for predictor
        except Exception as e:
             print(f"Error during CLIP embedding extraction: {e}")
             exit(1)


    # --- Get Classifier Prediction ---
    with torch.no_grad():
        # Predictor model expects float32 input
        predictions = predictor_model(clip_embeddings.to(device)) # Ensure embedding is on predictor device
        # Apply softmax if not already done by the model (PredictorModel does it internally for outputs > 1)
        # probabilities = torch.softmax(predictions, dim=-1) # Usually done internally by model
        probabilities = predictions # Assume model output is already probabilities (post-softmax)

    # --- Display Results ---
    print("\n--- Results ---")
    print(f"Image: {os.path.basename(args.image_path)}")
    probabilities_cpu = probabilities.squeeze().cpu().numpy() # Remove batch dim, move to CPU

    if len(probabilities_cpu.shape) == 0: # Handle case where model output was scalar
        print("Error: Model output seems scalar, expected probabilities per class.")
    else:
        num_classes = len(probabilities_cpu)
        for i in range(num_classes):
            class_name = labels.get(str(i), f"Class {i}")
            probability = probabilities_cpu[i]
            print(f"  {class_name}: {probability:.4f} ({probability*100:.2f}%)")

    print("-------------\n")