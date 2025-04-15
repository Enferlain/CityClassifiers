# Version 1.3: Fixed imports, MultiModel pipeline logic, format_pred call, removed duplicate function.

import os
import json
import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as F # Correct functional import
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from PIL import Image # Make sure PIL is imported if used (e.g. for resizing)

# Ensure model and utility functions are imported correctly
try:
    from model import PredictorModel
    # Ensure get_embed_params is imported ONCE and used
    from utils import get_embed_params
except ImportError as e:
    print(f"Error importing PredictorModel or get_embed_params: {e}")
    print("Ensure model.py and utils.py are in the same directory or your PYTHONPATH is set correctly.")
    raise

# --- Utility function to load config (shared) ---
def _load_config_helper(config_path):
    """Loads labels and model params from a JSON config file."""
    if not config_path or not os.path.isfile(config_path):
        print(f"DEBUG: Config file not found or not provided: {config_path}")
        return ({}, None)
    try:
        with open(config_path) as f:
            data = json.load(f)
        labels = data.get("labels", {})
        model_params_config = data.get("model_params", {})
        print(f"DEBUG: Loaded config from {config_path}")
        if "outputs" in model_params_config:
             model_params_config["outputs"] = int(model_params_config["outputs"])
        return (labels, model_params_config if model_params_config else None)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from config file {config_path}: {e}")
        return ({}, None)
    except Exception as e:
        print(f"Error reading or processing config file {config_path}: {e}")
        return ({}, None)

# --- Utility function to load model state dict (shared) ---
def _load_model_helper(model_path, expected_outputs=None):
    """Loads state dict and optionally verifies expected outputs."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        sd = load_file(model_path)
        first_layer_key = "up.0.weight"
        if first_layer_key not in sd or tuple(sd[first_layer_key].shape) != (1024, 768):
             raise ValueError(f"Model {model_path} structure mismatch. Key: {first_layer_key}")

        final_bias_key_new = "down.10.bias" # v1.1 arch
        final_bias_key_old = "down.5.bias"  # Original arch
        actual_outputs = None
        if final_bias_key_new in sd: actual_outputs = sd[final_bias_key_new].shape[0]
        elif final_bias_key_old in sd: actual_outputs = sd[final_bias_key_old].shape[0]
        else: raise KeyError(f"Could not determine outputs from {model_path}.")

        if expected_outputs is not None and actual_outputs != expected_outputs:
            print(f"Warning: Output size mismatch! Expected {expected_outputs}, found {actual_outputs} in {model_path}.")
            # Decide whether to trust config or file if mismatch occurs
            # Sticking with expected_outputs if provided for now.

        return sd, actual_outputs
    except Exception as e:
        print(f"Error loading model state dict from {model_path}: {e}")
        raise

# --- Base Pipeline Class (Common Methods) ---
# Added a base class to avoid repetition
class BasePipeline:
    clip_ver = "openai/clip-vit-large-patch14-336"

    def __init__(self, device="cpu", clip_dtype=torch.float32):
        self.device = device
        self.clip_dtype = clip_dtype
        self.proc = None
        self.clip = None
        self.proc_size = 224 # Default, will be updated
        self._init_clip()

    def _init_clip(self):
        self.proc = CLIPImageProcessor.from_pretrained(self.clip_ver)
        proc_sz = self.proc.size
        if isinstance(proc_sz, dict): self.proc_size = proc_sz.get("shortest_edge", proc_sz.get("height", 224))
        elif isinstance(proc_sz, int): self.proc_size = proc_sz
        else: self.proc_size = 224
        print(f"DEBUG: Determined CLIP processor size: {self.proc_size}")
        self.clip = CLIPVisionModelWithProjection.from_pretrained(
            self.clip_ver, torch_dtype=self.clip_dtype
        ).to(self.device).eval()

    def get_clip_emb(self, img_list):
        """Processes a list of PIL images and returns embeddings."""
        if not isinstance(img_list, list): img_list = [img_list]
        try:
            processed = self.proc(images=img_list, return_tensors="pt")["pixel_values"] \
                .to(self.clip_dtype).to(self.device)
        except Exception as e:
            print(f"Error during CLIP processing: {e}")
            raise
        with torch.no_grad():
            emb = self.clip(pixel_values=processed)["image_embeds"]
        return emb.detach().to(device='cpu', dtype=torch.float32) # Return float32 on CPU

    def get_clip_emb_tiled(self, raw_pil_image, tiling=False):
        """Gets CLIP embeddings for a single PIL image, applying tiling if requested."""
        target_size = self.proc_size
        img_list = []
        if tiling and min(raw_pil_image.size) > target_size * 2:
            resize_target = target_size * 2
            if max(raw_pil_image.size) > resize_target * 2:
                 scale = resize_target / min(raw_pil_image.size)
                 new_size = (int(round(raw_pil_image.width * scale)), int(round(raw_pil_image.height * scale)))
                 print(f"DEBUG: Resizing image from {raw_pil_image.size} to {new_size} before tiling.")
                 raw_resized = raw_pil_image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                 raw_resized = raw_pil_image
            try:
                 crops = F.five_crop(raw_resized, (target_size, target_size)) # Use correct functional call `F`
                 img_list.extend(crops)
                 print(f"DEBUG: Using {len(img_list)} tiles.")
            except Exception as e:
                 print(f"Error during five_crop: {e}. Falling back to single image.")
                 img_list = [raw_pil_image]
        else:
            img_list = [raw_pil_image]

        # Use the common embedding getter
        return self.get_clip_emb(img_list)

    def _load_model_generic(self, path, config_args=None, required_outputs=None):
        """Generic model loader using helpers."""
        # Determine expected outputs: use required if provided, else from config, else None
        expected_outputs = required_outputs if required_outputs is not None else (config_args.get("outputs") if config_args else None)

        sd, outputs_in_file = _load_model_helper(path, expected_outputs=expected_outputs)

        # Use expected_outputs if provided, otherwise trust file
        final_outputs = expected_outputs if expected_outputs is not None else outputs_in_file
        if final_outputs is None or final_outputs <= 0:
             raise ValueError(f"Could not determine number of outputs for model {path}")

        try:
            model_params = get_embed_params("CLIP")
            model_params["outputs"] = final_outputs
            model = PredictorModel(**model_params)
            print(f"DEBUG: Instantiating PredictorModel with {final_outputs} outputs.")
        except Exception as e:
             raise TypeError(f"Failed to instantiate PredictorModel: {e}") from e

        model.eval()
        try:
            missing_keys, unexpected_keys = model.load_state_dict(sd, strict=True)
            if missing_keys: print(f"Warning: Missing keys: {missing_keys}")
            if unexpected_keys: print(f"Warning: Unexpected keys: {unexpected_keys}")
        except RuntimeError as e:
             print(f"ERROR: State dict does not match model architecture in {path}.")
             raise e

        model.to(self.device)
        print(f"Successfully loaded model '{os.path.basename(path)}' with {model.outputs} outputs.")
        return model

    def get_model_pred_generic(self, model, emb):
        """Generic prediction method."""
        with torch.no_grad():
            pred = model(emb.to(self.device))
        return pred.detach().cpu()


# --- Aesthetics Pipelines ---
class CityAestheticsPipeline(BasePipeline):
    """Pipeline for single [image=>score] model."""
    def __init__(self, model_path, device="cpu", clip_dtype=torch.float32):
        super().__init__(device=device, clip_dtype=clip_dtype) # Call BasePipeline init
        self.model = self._load_model_generic(model_path, required_outputs=1)
        print("CityAesthetics: Pipeline init ok")

    def __call__(self, raw):
        emb = self.get_clip_emb_tiled(raw, tiling=False) # Tiling usually not needed/desired for aesthetic score
        pred = self.get_model_pred_generic(self.model, emb)
        return float(pred.squeeze()) # Return single float score


class CityAestheticsMultiModelPipeline(BasePipeline):
    """Pipeline for multiple [image=>score] models."""
    def __init__(self, model_paths, device="cpu", clip_dtype=torch.float32):
        super().__init__(device=device, clip_dtype=clip_dtype) # Call BasePipeline init
        self.models = {}
        print(f"Initializing MultiModel Aesthetics Pipeline for {len(model_paths)} models...")
        for path in model_paths:
            if not os.path.isfile(path):
                 print(f"Warning: Model path not found, skipping: {path}")
                 continue
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                 # Each model must be a scorer (outputs=1)
                 self.models[name] = self._load_model_generic(path, required_outputs=1)
                 print(f"  Loaded model: {name}")
            except Exception as e:
                 print(f"Error loading model {name} from {path}: {e}")
        if not self.models: raise ValueError("No valid models were loaded.")
        print("CityAesthetics MultiModel: Pipeline init ok")

    def __call__(self, raw):
        emb = self.get_clip_emb_tiled(raw, tiling=False)
        out = {}
        for name, model in self.models.items():
            pred = self.get_model_pred_generic(model, emb)
            out[name] = float(pred.squeeze()) # Ensure float output
        return out


# --- Classifier Pipelines ---
class CityClassifierPipeline(BasePipeline):
    """Pipeline for single [image=>label] classification model."""
    def __init__(self, model_path, config_path=None, device="cpu", clip_dtype=torch.float32):
        super().__init__(device=device, clip_dtype=clip_dtype) # Call BasePipeline init

        if config_path is None or not os.path.isfile(config_path):
             base_name = os.path.splitext(model_path)[0]
             inferred_config_path = f"{base_name}.config.json"
             if os.path.isfile(inferred_config_path):
                  print(f"DEBUG: Using inferred config path: {inferred_config_path}")
                  config_path = inferred_config_path
             else:
                  print(f"DEBUG: No config file provided or found for model: {model_path}")

        self.labels, model_args_from_config = _load_config_helper(config_path)
        self.model = self._load_model_generic(model_path, config_args=model_args_from_config, required_outputs=None) # Let loader determine outputs
        self.num_labels = self.model.outputs

        if not self.labels: # Use default numeric labels if config didn't provide them
             self.labels = {str(i): str(i) for i in range(self.num_labels)}
             print(f"DEBUG: Using default numeric labels (0 to {self.num_labels-1})")

        print(f"CityClassifier: Pipeline init ok (Labels: {self.labels})")

    def __call__(self, raw, default=True, tiling=True, tile_strat="mean"):
        # Use the base class tiled embedding getter
        emb = self.get_clip_emb_tiled(raw, tiling=tiling) # shape: [num_tiles, embed_dim]
        # Use base class prediction getter
        pred = self.get_model_pred_generic(self.model, emb) # shape: [num_tiles, num_classes]
        # Format the predictions
        return self.format_pred(
            pred,
            labels=self.labels,
            drop_default=(not default), # Pass boolean flag
            tile_strategy=tile_strat if tiling and emb.shape[0] > 1 else "raw",
        )

    def format_pred(self, pred, labels, drop_default=False, tile_strategy="mean"):
        """Formats predictions, handling tiling strategies."""
        num_classes = pred.shape[-1]
        num_tiles = pred.shape[0]

        if num_tiles > 1 and tile_strategy != "raw":
            combined_pred = torch.zeros(num_classes)
            for k in range(num_classes):
                tile_scores = pred[:, k]
                if   tile_strategy == "mean":   val = torch.mean(tile_scores)
                elif tile_strategy == "median": val = torch.median(tile_scores).values
                elif tile_strategy == "max":    val = torch.max(tile_scores)
                elif tile_strategy == "min":    val = torch.min(tile_scores)
                else: raise NotImplementedError(f"Invalid combine strategy '{tile_strategy}'!")
                combined_pred[k] = float(val)
            pred_to_format = combined_pred # Use combined scores
        else:
            pred_to_format = pred[0] # Use the first (or only) tile's scores

        out = {}
        for k in range(num_classes):
            if k == 0 and drop_default: continue
            key = labels.get(str(k), str(k))
            out[key] = float(pred_to_format[k])

        return out


class CityClassifierMultiModelPipeline(BasePipeline):
    """Pipeline for multiple [image=>label] classification models."""
    def __init__(self, model_paths, config_paths=[], device="cpu", clip_dtype=torch.float32):
        super().__init__(device=device, clip_dtype=clip_dtype) # Call BasePipeline init
        self.models = {}
        self.labels = {} # Store labels per model

        if not config_paths: # If no configs provided, try to infer them
             config_paths = [None] * len(model_paths)
             print("DEBUG: No config paths provided for MultiModel Classifier, will try to infer.")

        if len(model_paths) != len(config_paths):
             raise ValueError("Number of model paths and config paths must match!")

        print(f"Initializing MultiModel Classifier Pipeline for {len(model_paths)} models...")
        for i, m_path in enumerate(model_paths):
            if not os.path.isfile(m_path):
                 print(f"Warning: Model path not found, skipping: {m_path}")
                 continue

            name = os.path.splitext(os.path.basename(m_path))[0]
            c_path = config_paths[i]

            # Try to infer config if needed
            if c_path is None or not os.path.isfile(c_path):
                 base_name = os.path.splitext(m_path)[0]
                 inferred_c_path = f"{base_name}.config.json"
                 if os.path.isfile(inferred_c_path):
                      print(f"DEBUG: Using inferred config path for {name}: {inferred_c_path}")
                      c_path = inferred_c_path
                 else:
                      print(f"DEBUG: No config file provided or found for model: {name}")

            # Load config and model for this specific model
            try:
                 current_labels, model_args_from_config = _load_config_helper(c_path)
                 current_model = self._load_model_generic(m_path, config_args=model_args_from_config, required_outputs=None)

                 self.models[name] = current_model
                 # Store labels, using defaults if needed
                 if not current_labels:
                      self.labels[name] = {str(j): str(j) for j in range(current_model.outputs)}
                 else:
                      self.labels[name] = current_labels
                 print(f"  Loaded model: {name} (Outputs: {current_model.outputs}, Labels: {self.labels[name]})")

            except Exception as e:
                 print(f"Error loading model or config for {name} from {m_path}: {e}")

        if not self.models: raise ValueError("No valid models were loaded.")
        print("CityClassifier MultiModel: Pipeline init ok")

    def __call__(self, raw, default=True, tiling=True, tile_strat="mean"):
        # Get embeddings once (potentially tiled)
        # Use base class tiled embedding getter
        emb = self.get_clip_emb_tiled(raw, tiling=tiling) # shape: [num_tiles, embed_dim]

        out_list = [] # Gradio interface expects a list of outputs
        for name, model in self.models.items():
             # Get predictions for this model
             pred = self.get_model_pred_generic(model, emb) # shape: [num_tiles, num_classes]

             # Format predictions using the single classifier's formatter
             # Need an instance of the single pipeline to call its method, or duplicate logic.
             # Let's reuse the logic by calling a static method or duplicating it.
             # For simplicity here, let's duplicate the formatting logic:
             formatted_pred = self._format_single_pred(
                 pred,
                 labels=self.labels[name], # Use labels specific to this model
                 drop_default=(not default),
                 tile_strategy=tile_strat if tiling and emb.shape[0] > 1 else "raw",
                 num_classes = model.outputs # Pass num_classes explicitly
             )
             out_list.append(formatted_pred) # Append dict to list

        # Return the list of dictionaries
        # Note: The original GRADIO HOTFIX might have been needed if Gradio expects
        # a specific structure. Check Gradio interface definition. If it needs N outputs,
        # returning a list of N dicts is usually correct.
        return out_list

    # Duplicated formatting logic from CityClassifierPipeline for use here
    # Alternatively, could make format_pred a static method or utility function
    def _format_single_pred(self, pred, labels, drop_default, tile_strategy, num_classes):
        """Formats predictions for a single model, handling tiling."""
        num_tiles = pred.shape[0]
        if num_tiles > 1 and tile_strategy != "raw":
            combined_pred = torch.zeros(num_classes)
            for k in range(num_classes):
                tile_scores = pred[:, k]
                if   tile_strategy == "mean":   val = torch.mean(tile_scores)
                elif tile_strategy == "median": val = torch.median(tile_scores).values
                elif tile_strategy == "max":    val = torch.max(tile_scores)
                elif tile_strategy == "min":    val = torch.min(tile_scores)
                else: raise NotImplementedError(f"Invalid combine strategy '{tile_strategy}'!")
                combined_pred[k] = float(val)
            pred_to_format = combined_pred
        else:
            pred_to_format = pred[0]

        out = {}
        for k in range(num_classes):
            if k == 0 and drop_default: continue
            key = labels.get(str(k), str(k))
            out[key] = float(pred_to_format[k])
        return out

# --- get_model_path (Utility) ---
def get_model_path(name, repo, token=None, extension="safetensors", local_dir="models", use_hf=True):
    """Returns local model path or falls back to HF hub if required."""
    fname = f"{name}.{extension}"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    local_path = os.path.join(script_dir, local_dir, fname)

    if os.path.isfile(local_path):
        print(f"Using local model: '{local_path}'")
        return local_path
    if not use_hf: raise OSError(f"Local model '{local_path}' not found and HF download disabled.")

    hf_token_arg = None
    if isinstance(token, str) and token: hf_token_arg = token
    elif token is True: hf_token_arg = True
    elif token is False: hf_token_arg = False

    print(f"Local model not found. Downloading from HF Hub: '{repo}/{fname}'")
    try:
        downloaded_path = hf_hub_download(repo_id=repo, filename=fname, token=hf_token_arg)
        return str(downloaded_path)
    except Exception as e:
        print(f"Error downloading model from Hugging Face Hub: {e}")
        raise

# --- Removed duplicate get_embed_params ---