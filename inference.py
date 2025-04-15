# Version 1.5: Fixed _load_model_generic call signature, mutable default, clarity.

import os
import json
import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as F # Correct functional import
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModel # Using AutoClasses now
from PIL import Image

# Ensure model and utility functions are imported correctly
try:
    from model import PredictorModel
    from utils import get_embed_params
except ImportError as e:
    print(f"Error importing PredictorModel or get_embed_params: {e}")
    print("Ensure model.py and utils.py are in the same directory or your PYTHONPATH is set correctly.")
    raise

# --- Utility function to load config (shared) ---
def _load_config_helper(config_path):
    """Loads full config from JSON file."""
    if not config_path or not os.path.isfile(config_path):
        print(f"DEBUG: Config file not found or not provided: {config_path}")
        return None
    try:
        with open(config_path) as f: config_data = json.load(f)
        print(f"DEBUG: Loaded config from {config_path}")
        return config_data
    except Exception as e:
        print(f"Error reading or processing config file {config_path}: {e}")
        return None

# MOD v1.5.1: Accept expected_features argument
def _load_model_helper(model_path, expected_features, expected_outputs=None):
    """Loads state dict, verifies first layer features, and optionally checks outputs."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        sd = load_file(model_path)
        first_layer_key = "up.0.weight"

        # MOD v1.5.1: Check against passed expected_features
        if first_layer_key not in sd or tuple(sd[first_layer_key].shape)[1] != expected_features:
             # Changed from Warning to potentially stricter Error or more informative Warning
             print(f"Warning: Model {model_path} first layer input dimension mismatch!")
             print(f"  Expected input features: {expected_features}")
             if first_layer_key in sd: print(f"  Found input features: {tuple(sd[first_layer_key].shape)[1]}")
             else: print(f"  Key '{first_layer_key}' not found in state dict.")
             print("  Proceeding, but model instantiation might fail if architecture is wrong.")
             # Optionally raise error: raise ValueError(f"...")

        final_bias_key_new = "down.10.bias"; final_bias_key_old = "down.5.bias"
        actual_outputs = None
        if final_bias_key_new in sd: actual_outputs = sd[final_bias_key_new].shape[0]
        elif final_bias_key_old in sd: actual_outputs = sd[final_bias_key_old].shape[0]
        else: raise KeyError(f"Could not determine outputs from {model_path}.")
        if expected_outputs is not None and actual_outputs != expected_outputs:
            print(f"Warning: Output size mismatch! Expected {expected_outputs}, found {actual_outputs} in {model_path}.")

        return sd, actual_outputs
    except Exception as e:
        print(f"Error loading model state dict from {model_path}: {e}")
        raise


# --- Base Pipeline Class (Common Methods) ---
class BasePipeline:
    def __init__(self, config, device="cpu", clip_dtype=torch.float32):
        self.device = device
        self.clip_dtype = clip_dtype
        self.config = config if config is not None else {} # Ensure config is a dict

        self.base_vision_model_name = self.config.get("base_vision_model")
        if not self.base_vision_model_name:
            print("Warning: base_vision_model not found in config. Defaulting to OpenAI CLIP.")
            self.base_vision_model_name = "openai/clip-vit-large-patch14-336"

        self.proc = None
        self.clip_model = None # Renamed from 'clip' for clarity
        self.proc_size = 224
        self._init_vision_model()

    def _init_vision_model(self):
        model_name = self.base_vision_model_name
        print(f"Initializing Vision Model using AutoClasses: {model_name} on {self.device} with dtype {self.clip_dtype}")
        try:
            self.proc = AutoProcessor.from_pretrained(model_name)
            self.clip_model = AutoModel.from_pretrained(
                model_name, torch_dtype=self.clip_dtype, trust_remote_code=True
            ).to(self.device).eval()
            print(f"Loaded model class: {self.clip_model.__class__.__name__}")
            print(f"Loaded processor class: {self.proc.__class__.__name__}")

            # Determine proc_size
            if hasattr(self.proc, 'image_processor') and hasattr(self.proc.image_processor, 'size'):
                 proc_sz = self.proc.image_processor.size
                 if isinstance(proc_sz, dict): self.proc_size = int(proc_sz.get("shortest_edge", proc_sz.get("height", proc_sz.get("crop_size", 224))))
                 elif isinstance(proc_sz, int): self.proc_size = int(proc_sz)
                 else: self.proc_size = 224
            elif hasattr(self.proc, 'size'):
                 proc_sz = self.proc.size
                 if isinstance(proc_sz, dict): self.proc_size = int(proc_sz.get("shortest_edge", proc_sz.get("height", 224)))
                 elif isinstance(proc_sz, int): self.proc_size = int(proc_sz)
                 else: self.proc_size = 224
            else: self.proc_size = 224
            print(f"Determined processor input size: {self.proc_size}")
        except Exception as e:
            print(f"Error initializing vision model {model_name}: {e}")
            raise

    def get_clip_emb(self, img_list):
        if not isinstance(img_list, list): img_list = [img_list]
        try:
            inputs = self.proc(images=img_list, return_tensors="pt").to(device=self.device)
        except Exception as e: print(f"Error processing images: {e}"); raise
        with torch.no_grad():
            if hasattr(self.clip_model, 'get_image_features'):
                 emb = self.clip_model.get_image_features(**inputs)
            else:
                 print("Warning: Loaded vision model has no get_image_features. Attempting standard forward.")
                 outputs = self.clip_model(**inputs)
                 if hasattr(outputs, 'image_embeds'): emb = outputs.image_embeds
                 elif hasattr(outputs, 'pooler_output'): emb = outputs.pooler_output
                 else: emb = outputs.last_hidden_state.mean(dim=1)
        return emb.detach().to(device='cpu', dtype=torch.float32)

    def get_clip_emb_tiled(self, raw_pil_image, tiling=False):
        target_size = self.proc_size
        img_list = []
        if tiling and min(raw_pil_image.size) > target_size * 2:
            resize_target = target_size * 2
            if max(raw_pil_image.size) > resize_target * 2:
                 scale = resize_target / min(raw_pil_image.size)
                 new_size = (int(round(raw_pil_image.width * scale)), int(round(raw_pil_image.height * scale)))
                 print(f"DEBUG: Resizing image from {raw_pil_image.size} to {new_size} before tiling.")
                 raw_resized = raw_pil_image.resize(new_size, Image.Resampling.LANCZOS)
            else: raw_resized = raw_pil_image
            try:
                 crops = F.five_crop(raw_resized, (target_size, target_size))
                 img_list.extend(crops); print(f"DEBUG: Using {len(img_list)} tiles.")
            except Exception as e:
                 print(f"Error during five_crop: {e}. Falling back."); img_list = [raw_pil_image]
        else: img_list = [raw_pil_image]
        return self.get_clip_emb(img_list)

    # _load_model_generic needs access to embed_ver from config
    def _load_model_generic(self, path, config_args=None, required_outputs=None):
        """Generic model loader using helpers, gets embed_ver from config_args."""
        model_config = config_args
        if model_config is None:
             model_config_path = f"{os.path.splitext(path)[0]}.config.json"
             model_config = _load_config_helper(model_config_path)
             if model_config is None: model_config = {}

        config_embed_ver = model_config.get("embed_ver", "CLIP")
        config_model_params = model_config.get("model_params", {})
        config_outputs = config_model_params.get("outputs")
        if config_outputs is not None: config_outputs = int(config_outputs)

        # MOD v1.5.1: Get expected features based on embed_ver from config
        try:
            expected_features = get_embed_params(config_embed_ver)["features"]
        except ValueError as e:
             print(f"Error getting params for embed_ver '{config_embed_ver}': {e}. Assuming default CLIP features (768).")
             expected_features = 768 # Fallback, though should ideally error out
        except KeyError:
             print(f"Error: 'features' key missing from get_embed_params for '{config_embed_ver}'. Assuming default CLIP features (768).")
             expected_features = 768

        expected_outputs = required_outputs if required_outputs is not None else config_outputs

        # MOD v1.5.1: Pass expected_features to the helper function
        sd, outputs_in_file = _load_model_helper(path, expected_features=expected_features, expected_outputs=expected_outputs)

        final_outputs = expected_outputs if expected_outputs is not None else outputs_in_file
        if final_outputs is None or final_outputs <= 0:
             raise ValueError(f"Could not determine valid number of outputs for model {path}")

        try:
            # Instantiate model using the correct features
            model_init_params = get_embed_params(config_embed_ver)
            model_init_params["outputs"] = final_outputs
            model = PredictorModel(**model_init_params)
            print(f"DEBUG: Instantiating PredictorModel with features={model_init_params['features']}, outputs={final_outputs}.")
        except Exception as e: raise TypeError(f"Failed to instantiate PredictorModel: {e}") from e

        model.eval()
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as e: print(f"ERROR: State dict does not match model architecture in {path}."); raise e

        model.to(self.device)
        print(f"Successfully loaded model '{os.path.basename(path)}' with {model.outputs} outputs.")
        return model

    def get_model_pred_generic(self, model, emb):
        """Generic prediction method."""
        with torch.no_grad(): pred = model(emb.to(self.device))
        return pred.detach().cpu()


# --- Aesthetics Pipelines ---
class CityAestheticsPipeline(BasePipeline):
    def __init__(self, model_path, device="cpu", clip_dtype=torch.float32):
        config_path = f"{os.path.splitext(model_path)[0]}.config.json"
        config = _load_config_helper(config_path)
        super().__init__(config=config, device=device, clip_dtype=clip_dtype)
        # Pass config_args explicitly
        self.model = self._load_model_generic(model_path, config_args=self.config, required_outputs=1)
        print("CityAesthetics: Pipeline init ok")

    def __call__(self, raw):
        emb = self.get_clip_emb_tiled(raw, tiling=False)
        pred = self.get_model_pred_generic(self.model, emb)
        return float(pred.squeeze())


class CityAestheticsMultiModelPipeline(BasePipeline):
     def __init__(self, model_paths, device="cpu", clip_dtype=torch.float32):
         super().__init__(config={}, device=device, clip_dtype=clip_dtype) # Init base with empty config
         self.models = {}
         print(f"Initializing MultiModel Aesthetics Pipeline for {len(model_paths)} models...")
         for path in model_paths:
             if not os.path.isfile(path): print(f"Warning: Model path not found: {path}"); continue
             name = os.path.splitext(os.path.basename(path))[0]
             try:
                 # Load config for this model to pass args (even if empty)
                 config_path = f"{os.path.splitext(path)[0]}.config.json"
                 model_config = _load_config_helper(config_path)
                 # Pass config_args, require outputs=1
                 self.models[name] = self._load_model_generic(path, config_args=model_config, required_outputs=1)
                 print(f"  Loaded model: {name}")
             except Exception as e: print(f"Error loading model {name} from {path}: {e}")
         if not self.models: raise ValueError("No valid models loaded.")
         print("CityAesthetics MultiModel: Pipeline init ok")

     def __call__(self, raw):
        emb = self.get_clip_emb_tiled(raw, tiling=False)
        out = {}
        for name, model in self.models.items():
            pred = self.get_model_pred_generic(model, emb)
            out[name] = float(pred.squeeze())
        return out


# --- Classifier Pipelines ---
class CityClassifierPipeline(BasePipeline):
    def __init__(self, model_path, config_path=None, device="cpu", clip_dtype=torch.float32):
        # Find and load config first
        if config_path is None or not os.path.isfile(config_path):
             inferred_config_path = f"{os.path.splitext(model_path)[0]}.config.json"
             if os.path.isfile(inferred_config_path): config_path = inferred_config_path
             else: print(f"DEBUG: No config file provided or found for model: {model_path}")
        loaded_config = _load_config_helper(config_path)

        # Init BasePipeline with loaded config (or empty dict)
        super().__init__(config=loaded_config, device=device, clip_dtype=clip_dtype)

        # Get labels from potentially loaded config
        self.labels = self.config.get("labels", {})

        # Load model, passing its config args, no required_outputs needed here
        self.model = self._load_model_generic(model_path, config_args=self.config, required_outputs=None)
        self.num_labels = self.model.outputs

        if not self.labels: # Populate default labels if needed
             self.labels = {str(i): str(i) for i in range(self.num_labels)}
             print(f"DEBUG: Using default numeric labels (0 to {self.num_labels-1})")

        print(f"CityClassifier: Pipeline init ok (Labels: {self.labels})")

    def __call__(self, raw, default=True, tiling=True, tile_strat="mean"):
        emb = self.get_clip_emb_tiled(raw, tiling=tiling)
        pred = self.get_model_pred_generic(self.model, emb)
        return self.format_pred(pred, labels=self.labels, drop_default=(not default), tile_strategy=tile_strat if tiling and emb.shape[0] > 1 else "raw")

    def format_pred(self, pred, labels, drop_default=False, tile_strategy="mean"):
        # (format_pred logic remains the same as v1.3)
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
            pred_to_format = combined_pred
        else: pred_to_format = pred[0]
        out = {}; k = 0
        for k in range(num_classes):
            if k == 0 and drop_default: continue
            key = labels.get(str(k), str(k))
            out[key] = float(pred_to_format[k])
        return out

# Fixed mutable default for config_paths=[]
class CityClassifierMultiModelPipeline(BasePipeline):
    def __init__(self, model_paths, config_paths=None, device="cpu", clip_dtype=torch.float32): # Changed default to None
        # Init base pipeline with placeholder config
        super().__init__(config={}, device=device, clip_dtype=clip_dtype)
        self.models = {}
        self.labels = {}

        # Handle None default for config_paths
        if config_paths is None: config_paths = [None] * len(model_paths)
        if len(model_paths) != len(config_paths): raise ValueError("Mismatch model/config paths")

        print(f"Initializing MultiModel Classifier Pipeline for {len(model_paths)} models...")
        for i, m_path in enumerate(model_paths):
            if not os.path.isfile(m_path): print(f"Warning: Model path not found: {m_path}"); continue
            name = os.path.splitext(os.path.basename(m_path))[0]
            c_path = config_paths[i]

            # Infer config path if needed
            if c_path is None or not os.path.isfile(c_path):
                 inferred_c_path = f"{os.path.splitext(m_path)[0]}.config.json"
                 if os.path.isfile(inferred_c_path): c_path = inferred_c_path
                 else: print(f"DEBUG: No config file for model: {name}")

            try:
                 current_config = _load_config_helper(c_path)
                 # Load model using generic loader, passing its config
                 current_model = self._load_model_generic(m_path, config_args=current_config, required_outputs=None)

                 self.models[name] = current_model
                 current_labels_from_config = current_config.get("labels", {}) if current_config else {}
                 if not current_labels_from_config: self.labels[name] = {str(j): str(j) for j in range(current_model.outputs)}
                 else: self.labels[name] = current_labels_from_config
                 print(f"  Loaded model: {name} (Outputs: {current_model.outputs}, Labels: {self.labels[name]})")
            except Exception as e: print(f"Error loading model/config for {name} from {m_path}: {e}")

        if not self.models: raise ValueError("No valid models loaded.")
        print("CityClassifier MultiModel: Pipeline init ok")

    def __call__(self, raw, default=True, tiling=True, tile_strat="mean"):
        emb = self.get_clip_emb_tiled(raw, tiling=tiling)
        out_list = []
        for name, model in self.models.items():
             pred = self.get_model_pred_generic(model, emb)
             formatted_pred = self._format_single_pred(pred, labels=self.labels[name], drop_default=(not default),
                 tile_strategy=tile_strat if tiling and emb.shape[0] > 1 else "raw", num_classes=model.outputs)
             out_list.append(formatted_pred)
        return out_list

    # (Keep _format_single_pred helper method as in v1.4)
    def _format_single_pred(self, pred, labels, drop_default, tile_strategy, num_classes):
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
        else: pred_to_format = pred[0]
        out = {}; k = 0
        for k in range(num_classes):
            if k == 0 and drop_default: continue
            key = labels.get(str(k), str(k))
            out[key] = float(pred_to_format[k])
        return out

# --- get_model_path (Utility - unchanged from v1.4) ---
def get_model_path(name, repo, token=None, extension="safetensors", local_dir="models", use_hf=True):
    fname = f"{name}.{extension}"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    local_path = os.path.join(script_dir, local_dir, fname)
    if os.path.isfile(local_path): print(f"Using local model: '{local_path}'"); return local_path
    if not use_hf: raise OSError(f"Local model '{local_path}' not found and HF download disabled.")
    hf_token_arg = None
    if isinstance(token, str) and token: hf_token_arg = token
    elif token is True: hf_token_arg = True
    elif token is False: hf_token_arg = False
    print(f"Local model not found. Downloading from HF Hub: '{repo}/{fname}'")
    try:
        downloaded_path = hf_hub_download(repo_id=repo, filename=fname, token=hf_token_arg)
        return str(downloaded_path)
    except Exception as e: print(f"Error downloading model from Hugging Face Hub: {e}"); raise