# test_single_image.py
# Version 1.2: Aligned with utils v2.2.1 & inference v1.5.1

import os
import torch
import argparse
import json # Added json import
from PIL import Image
from safetensors.torch import load_file
# Use AutoClasses
from transformers import AutoProcessor, AutoModel

# Ensure model.py and utils.py are accessible
try:
    from model import PredictorModel
    from utils import get_embed_params
except ImportError as e:
    print(f"Error importing PredictorModel/utils: {e}")
    print("Ensure model.py and utils.py are in the same directory or your PYTHONPATH is set correctly.")
    raise

# --- Argument Parsing ---
def parse_test_args():
    parser = argparse.ArgumentParser(description="Test a single image with an anatomy classifier.")
    parser.add_argument('--model_path', required=True, help="Path to the trained predictor .safetensors model file.")
    parser.add_argument('--image_path', required=True, help="Path to the input image file.")
    parser.add_argument('--config_path', default=None, help="(Optional) Path to the model's .config.json file.")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run on (cuda or cpu).")
    parser.add_argument('--precision', default='fp32', choices=['fp32', 'bf16', 'fp16'], help='Precision for VISION model (fp32, bf16, fp16).')
    return parser.parse_args()

# --- Utility function to load config ---
# (Copied from inference.py v1.5 for consistency)
def _load_config_helper(config_path):
    """Loads full config from JSON file."""
    if not config_path or not os.path.isfile(config_path):
        print(f"DEBUG: Config file not found or not provided: {config_path}")
        return None # Return None if no config found
    try:
        with open(config_path) as f: config_data = json.load(f)
        print(f"DEBUG: Loaded config from {config_path}")
        return config_data # Return the whole dictionary
    except Exception as e:
        print(f"Error reading or processing config file {config_path}: {e}")
        return None

# --- Vision Model Initialization ---
def init_vision_model(model_name, device, dtype):
    """Initializes vision model using AutoClasses."""
    print(f"Initializing Vision Model using AutoClasses: {model_name} on {device} with dtype {dtype}")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        # Add trust_remote_code=True for broader compatibility
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()
        print(f"Vision Model Initialized (Processor: {processor.__class__.__name__}, Model: {model.__class__.__name__}).")
        return processor, model
    except Exception as e: print(f"Error initializing vision model {model_name}: {e}"); raise

# --- Predictor Model Loading ---
def load_predictor_model(model_path, config_path, device):
    """Loads the trained PredictorModel, using config for parameters."""
    print(f"Loading predictor model from: {model_path}")

    # --- Load Config ---
    # Infer config path if not provided
    if config_path is None:
        inferred_config_path = f"{os.path.splitext(model_path)[0]}.config.json"
        if os.path.isfile(inferred_config_path):
            print(f"DEBUG: Using inferred config path: {inferred_config_path}")
            config_path = inferred_config_path
        else:
            print("Warning: No config file found or provided. Critical info might be missing.")

    # Load the config data
    config_data = _load_config_helper(config_path)
    if config_data is None:
        print("ERROR: Cannot proceed without model configuration (.config.json).")
        print("       Please ensure the config file exists and matches the model path.")
        raise FileNotFoundError("Missing required .config.json file")

    # --- Get Parameters from Config ---
    labels = config_data.get("labels", {'0': '0', '1': '1'}) # Default if missing
    model_params_config = config_data.get("model_params", {})
    embed_ver = config_data.get("embed_ver", "CLIP") # Get embed version, default CLIP
    final_outputs = model_params_config.get("outputs")

    if final_outputs is None:
        print("Warning: 'outputs' not found in config's model_params. Trying to infer from state_dict (less reliable).")
        # Attempt to infer from state dict as last resort (using simplified logic)
        try:
            sd_temp = load_file(model_path)
            final_bias_key_new = "down.10.bias"; final_bias_key_old = "down.5.bias"
            outputs_in_file_tensor = sd_temp.get(final_bias_key_new, sd_temp.get(final_bias_key_old, None))
            if outputs_in_file_tensor is not None: final_outputs = outputs_in_file_tensor.shape[0]
            else: raise KeyError("Could not infer outputs from state_dict.")
            print(f"Inferred outputs={final_outputs} from state_dict.")
            del sd_temp # Free memory
        except Exception as e:
            print(f"ERROR: Could not determine model outputs from config or state_dict: {e}")
            raise ValueError("Cannot determine model outputs.") from e
    else:
        final_outputs = int(final_outputs)

    if final_outputs <= 1 and config_data.get("arch") == "class":
         print("Warning: Config arch is 'class' but loaded outputs <= 1. Check config/model.")
         # Proceeding, but classification output formatting might fail later

    # --- Instantiate PredictorModel ---
    try:
        # Get features/hidden size based on embed_ver from config
        model_init_params = get_embed_params(embed_ver)
        model_init_params["outputs"] = final_outputs
        predictor_model = PredictorModel(**model_init_params)
        print(f"DEBUG: Instantiating PredictorModel with features={model_init_params['features']}, outputs={final_outputs}.")
    except Exception as e:
        print(f"Error instantiating PredictorModel based on config embed_ver '{embed_ver}': {e}")
        raise TypeError(f"Failed to instantiate PredictorModel") from e

    # --- Load State Dict ---
    try:
        sd = load_file(model_path)
        predictor_model.load_state_dict(sd, strict=True)
        predictor_model.to(device); predictor_model.eval()
        print(f"Predictor model loaded successfully.")

        # Ensure labels dictionary matches output count
        if len(labels) != final_outputs:
             print(f"Warning: Loaded labels ({len(labels)}) != model outputs ({final_outputs}). Using default numeric labels.")
             labels = {str(i): str(i) for i in range(final_outputs)}

        return predictor_model, labels, config_data # Return model, labels, and full config

    except Exception as e:
        print(f"Error loading state_dict into PredictorModel: {e}")
        raise


# --- Main Execution ---
if __name__ == "__main__":
    args = parse_test_args()

    # --- Setup Device and Precision ---
    device = torch.device(args.device)
    precision_map = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
    vision_dtype = precision_map.get(args.precision, torch.float32) # Precision for vision model
    # (Hardware/CPU checks for vision_dtype)
    if vision_dtype != torch.float32 and device.type == 'cpu':
        print(f"Warning: {args.precision} requested but device is CPU. Using float32 for vision model.")
        vision_dtype = torch.float32
    if vision_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
         print("Warning: bf16 requested but not supported. Using float32 for vision model.")
         vision_dtype = torch.float32

    # --- Load Predictor & Config ---
    # This now requires the config file to exist to get base_vision_model etc.
    predictor_model, labels, config_data = load_predictor_model(args.model_path, args.config_path, device)

    # --- Initialize Correct Vision Model ---
    base_vision_model_name = config_data.get("base_vision_model")
    if not base_vision_model_name:
        print("ERROR: 'base_vision_model' not found in config file. Cannot initialize vision model.")
        exit(1)
    processor, vision_model = init_vision_model(base_vision_model_name, device, vision_dtype)

    # --- Load and Process Image ---
    print(f"Processing image: {args.image_path}")
    try: pil_image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError: print(f"Error: Image file not found: {args.image_path}"); exit(1)
    except Exception as e: print(f"Error opening image {args.image_path}: {e}"); exit(1)

    # --- Get Embedding ---
    # Note: No tiling here, processing whole image via processor's default handling
    inputs = processor(images=pil_image, return_tensors="pt").to(device=device)
    with torch.no_grad():
        try:
            if hasattr(vision_model, 'get_image_features'):
                 embeddings = vision_model.get_image_features(**inputs).to(dtype=torch.float32) # Use float32 for predictor
            else:
                 # Basic fallback for models loaded differently (e.g., just vision tower)
                 outputs = vision_model(pixel_values=inputs.get("pixel_values").to(vision_dtype)) # Pass pixel_values with correct dtype
                 if hasattr(outputs, 'image_embeds'): embeddings = outputs.image_embeds.to(dtype=torch.float32)
                 elif hasattr(outputs, 'pooler_output'): embeddings = outputs.pooler_output.to(dtype=torch.float32)
                 else: raise AttributeError("Cannot get embeddings from vision model.")
        except Exception as e: print(f"Error during embedding extraction: {e}"); exit(1)

    # --- Get Classifier Prediction ---
    with torch.no_grad():
        # Predictor expects float32 on its device
        predictions = predictor_model(embeddings.to(device))
        # PredictorModel's forward applies Softmax for outputs>1
        probabilities = predictions

    # --- Display Results ---
    print("\n--- Results ---")
    print(f"Image: {os.path.basename(args.image_path)}")
    probabilities_cpu = probabilities.squeeze().cpu().numpy() # Remove batch dim, move to CPU
    print(f"Raw Probabilities (Bad=0, Good=1): {probabilities_cpu}")  # <-- ADD THIS LINE

    # Check if output is scalar (shouldn't be for classifier) or array
    if probabilities_cpu.ndim == 0: # Is scalar
         # This likely means it loaded a scorer (outputs=1) despite expecting classifier
         print(f"  Score (0-1): {probabilities_cpu:.4f}")
         print(f"  (Model seems to be a scorer, expected a classifier based on recent runs)")
    elif probabilities_cpu.ndim == 1: # Is array (expected for classifier)
        num_classes = len(probabilities_cpu)
        # Use labels loaded from config
        for i in range(num_classes):
            class_name = labels.get(str(i), f"Class {i}") # Use loaded labels
            probability = probabilities_cpu[i]
            print(f"  {class_name}: {probability:.4f} ({probability*100:.2f}%)")
    else:
         print(f"Error: Unexpected prediction output shape: {probabilities_cpu.shape}")

    print("-------------\n")