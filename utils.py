# Version: 2.4.0 (Refactored for end2end mode)
import math
import os
import json
import traceback

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import shutil # Added for removing old best checkpoints
from tqdm import tqdm
from safetensors.torch import save_file, load_file # Added load_file
from losses import GHMC_Loss, FocalLoss # Keep loss imports here

# --- Constants ---
LOSS_MEMORY = 500 # How many recent steps' training loss to average for display
SAVE_FOLDER = "models" # Folder to save models and configs


# --- Parameter Dictionary ---
# Version 2.4.0: Added DINOv2 and final NaFlex types
def get_embed_params(ver):
    """Returns features and hidden dim based on embedding version string."""
    # Add new embedding types here as needed
    # Key should match the 'embed_ver' field used in YAML configs
    embed_configs = {
        # --- SigLIP ---
        "CLIP": {"features": 768, "hidden": 1024}, # Legacy CLIP Example
        "siglip2_so400m_patch16_512_FitPad": {"features": 1152, "hidden": 1280},
        "siglip2_so400m_patch16_512_CenterCrop": {"features": 1152, "hidden": 1280},
        # Final NaFlex version using Processor Logic @ 1024
        "siglip2_so400m_patch16_naflex_Naflex_Proc1024": {"features": 1152, "hidden": 1280},
        # (Can comment out or remove older/unused NaFlex keys if desired)
        # "siglip2_so400m_patch16_naflex_NaflexResize": {"features": 1152, "hidden": 1280},
        # "siglip2_so400m_patch16_naflex_NaflexResize_Pad1024": {"features": 1152, "hidden": 1280},
        # "siglip2_so400m_patch16_naflex_AvgCrop": {"features": 1152, "hidden": 1280},
        # "siglip2_so400m_patch16_naflex_FitPad": {"features": 1152, "hidden": 1280},
        # "siglip2_so400m_patch16_naflex_CenterCrop": {"features": 1152, "hidden": 1280},

        # --- DINOv2 ---
        # Assumes FitPad preprocessing was used for generation
        "fb_dinov2_giant_FitPad": {"features": 1536, "hidden": 1280}, # ViT-Giant
        "timm_vit_large_patch14_dinov2.lvd142m_FitPad": {"features": 1024, "hidden": 1280}, # ViT-Large

        # --- Other ---
        "META": {"features": 1024, "hidden": 1280}, # Legacy MetaCLIP Example
        # <<< Add AIMv2 placeholder? Or let it be handled by E2E path? >>>
        # Let's omit AIMv2 here since it's handled by the E2E path.
    }
    if ver in embed_configs:
        return embed_configs[ver]
    else:
        # Attempt to infer from HF model name if ver contains '/'? More complex.
        # For now, raise error for unknown explicitly defined versions.
        raise ValueError(f"Unknown/undefined embedding version key '{ver}' in get_embed_params. Please add it or check YAML config.")
# --- End Parameter Dictionary ---


# --- Argument Parsing and Config Loading ---
# Version 2.6.0: Handles is_end_to_end flag and associated parameters
def parse_and_load_args():
    """
    Parses command line args and merges with YAML config.
    Handles conditional loading of parameters based on 'is_end_to_end'.
    Command line args take precedence over YAML.
    """
    parser = argparse.ArgumentParser(description="Train aesthetic predictor/classifier or end-to-end model")
    # --- Command Line Arguments ---
    parser.add_argument("--config", required=True, help="Training config YAML file")
    parser.add_argument('--resume', help="Checkpoint (.safetensors model file) to resume from")
    # Training Duration Overrides
    parser.add_argument('--max_train_epochs', type=int, default=None, help="Override: Train for a specific number of epochs.")
    parser.add_argument('--max_train_steps', type=int, default=None, help="Override: Train for a specific number of steps (takes priority).")
    # Other Overrides (Keep these common ones)
    parser.add_argument('--precision', type=str, default=None, choices=['fp32', 'fp16', 'bf16'], help='Override training precision.')
    parser.add_argument("--nsave", type=int, default=None, help="Override save frequency (steps).")
    parser.add_argument("--val_split_count", type=int, default=None, help="Override validation split count.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--preload_data", action=argparse.BooleanOptionalAction, default=None, help="Override data preloading (only embedding mode).")
    parser.add_argument("--data_root", type=str, default=None, help="Override data root directory.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Override WandB project name.")
    parser.add_argument('--optimizer', type=str, default=None, help='Override optimizer choice.')
    parser.add_argument('--loss_function', type=str, default=None, help="Override loss function choice.")
    parser.add_argument('--lr', type=float, default=None, help="Override learning rate.")
    parser.add_argument('--betas', type=float, nargs='+', default=None, help="Override optimizer betas.") # Keep common overrides
    parser.add_argument('--weight_decay', type=float, default=None, help='Override optimizer weight decay.')
    parser.add_argument('--eps', type=float, default=None, help='Override optimizer epsilon.')
    parser.add_argument('--batch', type=int, default=None, help='Override batch size.')
    parser.add_argument('--freeze_base_model', action=argparse.BooleanOptionalAction, default=None, help="Override E2E base model freezing.")

    cmd_args = parser.parse_args()

    # Load YAML config
    if not os.path.isfile(cmd_args.config): parser.error(f"Config file '{cmd_args.config}' not found.")
    try:
        with open(cmd_args.config) as f: conf = yaml.safe_load(f)
    except Exception as e: parser.error(f"Error loading YAML config '{cmd_args.config}': {e}")

    args = argparse.Namespace() # Final args object

    # Helper function to get value prioritizing CMD > YAML > Default
    def get_arg_value(arg_name, cmd_value, yaml_conf_section, yaml_key=None, default=None, expected_type=None):
        if yaml_key is None: yaml_key = arg_name
        yaml_value = yaml_conf_section.get(yaml_key, default)
        final_value = cmd_value if cmd_value is not None else yaml_value
        if final_value is not None and expected_type is not None:
            try:
                if expected_type == bool: # Handle BooleanOptionalAction properly
                     if isinstance(final_value, str): # Convert string from YAML/CMD
                          if final_value.lower() in ['true', '1', 'yes']: final_value = True
                          elif final_value.lower() in ['false', '0', 'no']: final_value = False
                          else: raise ValueError(f"Invalid boolean string '{final_value}'")
                     final_value = bool(final_value)
                elif expected_type == int: final_value = int(final_value)
                elif expected_type == float: final_value = float(final_value)
                elif expected_type == tuple and isinstance(final_value, list): final_value = tuple(final_value)
            except (ValueError, TypeError):
                print(f"Warning: Arg '{arg_name}' value '{final_value}' invalid for {expected_type}. Using default: {default}")
                final_value = default
        return final_value if final_value is not None else default

    # --- Populate Args (Prioritize CMD > YAML > Default) ---
    # Config path / Resume
    args.config = cmd_args.config
    args.resume = cmd_args.resume
    # Basic Model Info
    model_conf = conf.get("model", {})
    args.base = model_conf.get("base", "unknown_model")
    args.rev = model_conf.get("rev", "v0.0")
    args.arch = model_conf.get("arch", "class")
    args.name = f"{args.base}-{args.rev}"

    # <<< Determine Mode: End-to-End or Embedding-based >>>
    args.is_end_to_end = model_conf.get("is_end_to_end", False) # Default to False

    if args.is_end_to_end:
        print("DEBUG: Parsing args for End-to-End model.")
        # --- End-to-End Specific Parameters ---
        args.base_vision_model = model_conf.get("base_vision_model")
        if not args.base_vision_model:
             parser.error("Missing 'model.base_vision_model' in config for end-to-end model.")

        # Load EarlyExtract specific args (assuming they are directly under 'model' or a dedicated section)
        # Let's check directly under 'model' first for simplicity
        args.extract_layer = model_conf.get("extract_layer", -1)
        args.pooling_strategy = model_conf.get("pooling_strategy", 'attn')
        args.freeze_base_model = get_arg_value('freeze_base_model', cmd_args.freeze_base_model, model_conf, default=True, expected_type=bool)

        # Load Head specific args (check 'head_params' or fallback to 'predictor_params')
        head_conf = conf.get("head_params", conf.get("predictor_params", {}))
        args.head_hidden_dim = head_conf.get("hidden_dim", 1024) # Default for head maybe 1024?
        args.head_num_res_blocks = head_conf.get("num_res_blocks", 2)
        args.head_dropout_rate = head_conf.get("dropout_rate", 0.2)
        args.head_output_mode = head_conf.get("output_mode") # REQUIRED for head
        if args.head_output_mode is None:
            parser.error("Missing 'output_mode' in 'head_params' (or 'predictor_params') section for end-to-end model.")

        # Load Attn Pool specific args (check 'attn_pool_params' or directly under 'model')
        if args.pooling_strategy == 'attn':
            attn_pool_conf = conf.get("attn_pool_params", model_conf)
            args.attn_pool_heads = attn_pool_conf.get("attn_pool_heads", 8)
            args.attn_pool_dropout = attn_pool_conf.get("attn_pool_dropout", 0.1)

        # Set embedding-specific args to None
        args.embed_ver = None
        args.features = None
        args.hidden_dim = None
        args.use_attention = None # Not applicable for E2E head structure shown
        args.num_attn_heads = None
        args.attn_dropout = None
        args.output_mode = None # Use head_output_mode instead

    else: # Embedding Path
        print("DEBUG: Parsing args for Embedding-based model.")
        # --- Embedding-based Specific Parameters ---
        args.embed_ver = model_conf.get("embed_ver")
        if not args.embed_ver:
             parser.error("Missing 'model.embed_ver' in config for embedding-based model.")
        args.base_vision_model = model_conf.get("base_vision_model", None) # Optional here

        # Get defaults from embed_ver
        try: embed_params = get_embed_params(args.embed_ver)
        except ValueError as e: parser.error(f"Error getting embed params: {e}")
        args.features = embed_params.get('features')
        default_hidden = embed_params.get('hidden', 1280)

        # Load PredictorModel specific args
        predictor_conf = conf.get("predictor_params", {})
        args.hidden_dim = predictor_conf.get("hidden_dim", default_hidden)
        args.use_attention = predictor_conf.get("use_attention", True)
        args.num_attn_heads = predictor_conf.get("num_attn_heads", 8)
        args.attn_dropout = predictor_conf.get("attn_dropout", 0.1)
        args.num_res_blocks = predictor_conf.get("num_res_blocks", 1)
        args.dropout_rate = predictor_conf.get("dropout_rate", 0.1)
        args.output_mode = predictor_conf.get("output_mode") # REQUIRED for predictor
        if args.output_mode is None:
            parser.error("Missing 'output_mode' in 'predictor_params' section for embedding-based model.")

        # Set E2E specific args to None
        args.extract_layer = None
        args.pooling_strategy = None
        args.freeze_base_model = None
        args.head_hidden_dim = None
        args.head_num_res_blocks = None
        args.head_dropout_rate = None
        args.head_output_mode = None
        args.attn_pool_heads = None
        args.attn_pool_dropout = None
        # For embedding mode, data preload CAN be used
        args.preload_data = get_arg_value('preload_data', cmd_args.preload_data, conf.get("train", {}), default=True, expected_type=bool)

    # --- Training Params (Merged - applies to both modes) ---
    train_conf = conf.get("train", {})
    args.lr = get_arg_value('lr', cmd_args.lr, train_conf, default=1e-4, expected_type=float)
    args.batch = get_arg_value('batch', cmd_args.batch, train_conf, default=4, expected_type=int)
    args.loss_function = get_arg_value('loss_function', cmd_args.loss_function, train_conf) # Default set later
    args.optimizer = get_arg_value('optimizer', cmd_args.optimizer, train_conf, default='AdamW')
    # Common overrides
    args.betas = get_arg_value('betas', cmd_args.betas, train_conf, expected_type=tuple)
    args.eps = get_arg_value('eps', cmd_args.eps, train_conf, expected_type=float)
    args.weight_decay = get_arg_value('weight_decay', cmd_args.weight_decay, train_conf, expected_type=float)
    # Training duration (values will be calculated later in train.py)
    args.max_train_epochs = get_arg_value('max_train_epochs', cmd_args.max_train_epochs, train_conf, default=None, expected_type=int)
    args.max_train_steps = get_arg_value('max_train_steps', cmd_args.max_train_steps, train_conf, default=None, expected_type=int)

    # Other Params (applies to both modes where relevant)
    args.precision = get_arg_value('precision', cmd_args.precision, train_conf, default='fp32')
    args.nsave = get_arg_value('nsave', cmd_args.nsave, train_conf, default=10000, expected_type=int)
    args.val_split_count = get_arg_value('val_split_count', cmd_args.val_split_count, train_conf, default=0, expected_type=int)
    args.seed = get_arg_value('seed', cmd_args.seed, train_conf, default=218, expected_type=int)
    args.num_workers = get_arg_value('num_workers', cmd_args.num_workers, train_conf, default=0, expected_type=int)
    # args.preload_data handled conditionally above
    args.data_root = get_arg_value('data_root', cmd_args.data_root, conf, default="data") # Get from top level or default
    args.wandb_project = get_arg_value('wandb_project', cmd_args.wandb_project, conf, default="city-classifiers") # Get from top level or default

    # --- Copy ALL OTHER keys from train_conf directly into args ---
    # Handles optimizer/scheduler specific args etc.
    # <<< Define handled keys carefully based on above >>>
    handled_keys_train = ['lr', 'batch', 'loss_function', 'optimizer', 'betas', 'eps', 'weight_decay',
                           'max_train_epochs', 'max_train_steps', 'precision', 'nsave', 'val_split_count',
                           'seed', 'num_workers', 'preload_data']
    handled_keys_model = ['base', 'rev', 'arch', 'embed_ver', 'base_vision_model', 'is_end_to_end',
                           'extract_layer', 'pooling_strategy', 'freeze_base_model'] # Basic model keys
    handled_keys_pred = ['features', 'hidden_dim', 'use_attention', 'num_attn_heads', 'attn_dropout',
                           'num_res_blocks', 'dropout_rate', 'output_mode'] # Predictor keys
    handled_keys_head = ['head_hidden_dim', 'head_num_res_blocks', 'head_dropout_rate', 'head_output_mode'] # Head keys
    handled_keys_attn = ['attn_pool_heads', 'attn_pool_dropout'] # Attn Pool keys
    handled_keys_other = ['config', 'resume', 'data_root', 'wandb_project'] # Other top-level keys

    all_handled_keys = set(handled_keys_train + handled_keys_model + handled_keys_pred + handled_keys_head + handled_keys_attn + handled_keys_other)

    # Copy from train_conf
    for key, value in train_conf.items():
         if key not in all_handled_keys and not hasattr(args, key):
              if isinstance(value, list): value = tuple(value)
              setattr(args, key, value)
    # Copy from predictor_params (if not handled)
    predictor_conf = conf.get("predictor_params", {})
    for key, value in predictor_conf.items():
         if key not in all_handled_keys and not hasattr(args, key):
              if isinstance(value, list): value = tuple(value)
              setattr(args, key, value)
    # Copy from head_params (if not handled)
    head_conf = conf.get("head_params", {})
    for key, value in head_conf.items():
         if key not in all_handled_keys and not hasattr(args, key):
              if isinstance(value, list): value = tuple(value)
              setattr(args, key, value)
    # --- End Copy ---

    # --- Labels/Weights ---
    labels_conf = conf.get("labels", {})
    args.labels = None; args.weights = None; args.num_labels = 1 if args.arch == 'score' else 0
    if args.arch == "class":
        if labels_conf:
            args.labels = {str(k): v.get("name", str(k)) for k, v in labels_conf.items() if str(k).isdigit()} # Ensure keys are digits
            if args.labels:
                 try: args.num_labels = max(int(k) for k in args.labels.keys()) + 1
                 except ValueError: args.num_labels = 0 # Handle empty or non-digit keys
            else: args.num_labels = 0 # If no valid digit keys found

            if args.num_labels > 0:
                weights = [1.0] * args.num_labels
                for k_str, label_conf in labels_conf.items():
                    if str(k_str).isdigit(): # Process only valid digit keys
                        try: weights[int(k_str)] = float(label_conf.get("loss", 1.0))
                        except (ValueError, IndexError): pass # Ignore bad index/value
                args.weights = weights
        else: args.num_labels = 2 # Default to 2 classes if none specified

    # --- Final Validation ---
    if args.arch == "score" and args.loss_function not in [None, 'l1', 'mse']:
         args.loss_function = 'l1' # Default for score
    valid_class_losses = [None, 'crossentropy', 'focal', 'bce', 'nll', 'ghm']
    if args.arch == "class" and args.loss_function not in valid_class_losses:
         args.loss_function = 'focal' # Default for class

    # Validate output modes were set correctly earlier
    if args.is_end_to_end and args.head_output_mode is None:
        parser.error("Internal Error: head_output_mode is None for end-to-end path.")
    if not args.is_end_to_end and args.output_mode is None:
        parser.error("Internal Error: output_mode is None for embedding path.")

    if args.max_train_epochs is None and args.max_train_steps is None:
         print("Warning: Neither max_train_epochs nor max_train_steps specified. Defaulting to max_train_steps = 100000.")
         args.max_train_steps = 100000

    # Prevent preload_data for E2E mode
    if args.is_end_to_end and getattr(args, 'preload_data', False):
        print("Warning: 'preload_data' is True but model is end-to-end. Setting preload_data=False.")
        args.preload_data = False

    # Defer final print until after calculations in train.py

    return args
# --- End Argument Parsing ---

# --- write_config Function ---
# Version 2.6.0: Include is_end_to_end and related params
def write_config(args):
    """Writes the final training configuration, including E2E params."""
    conf = { "model": {}, "train": {} } # Initialize sections

    # Basic Model Info
    conf["model"]["base"] = getattr(args, 'base', '?')
    conf["model"]["rev"] = getattr(args, 'rev', '?')
    conf["model"]["arch"] = getattr(args, 'arch', '?')
    conf["model"]["is_end_to_end"] = getattr(args, 'is_end_to_end', False)

    # Conditional Params
    if args.is_end_to_end:
        conf["model"]["base_vision_model"] = getattr(args, 'base_vision_model', '?')
        conf["model"]["extract_layer"] = getattr(args, 'extract_layer', '?')
        conf["model"]["pooling_strategy"] = getattr(args, 'pooling_strategy', '?')
        conf["model"]["freeze_base_model"] = getattr(args, 'freeze_base_model', '?')
        # Head Params (store under a separate key for clarity)
        conf["head_params"] = {
            "hidden_dim": getattr(args, 'head_hidden_dim', '?'),
            "num_classes": getattr(args, 'num_classes', '?'), # Use final calculated num_classes
            "num_res_blocks": getattr(args, 'head_num_res_blocks', '?'),
            "dropout_rate": getattr(args, 'head_dropout_rate', '?'),
            "output_mode": getattr(args, 'head_output_mode', '?')
        }
        # Attn Pool Params
        if args.pooling_strategy == 'attn':
            conf["attn_pool_params"] = {
                "attn_pool_heads": getattr(args, 'attn_pool_heads', '?'),
                "attn_pool_dropout": getattr(args, 'attn_pool_dropout', '?')
            }
    else: # Embedding Path
        conf["model"]["embed_ver"] = getattr(args, 'embed_ver', '?')
        conf["model"]["base_vision_model"] = getattr(args, 'base_vision_model', None) # Optional
        # Predictor Params
        conf["predictor_params"] = {
            "features": getattr(args, 'features', '?'),
            "hidden_dim": getattr(args, 'hidden_dim', '?'),
            "num_classes": getattr(args, 'num_classes', '?'), # Use final calculated num_classes
            "use_attention": getattr(args, 'use_attention', '?'),
            "num_attn_heads": getattr(args, 'num_attn_heads', '?'),
            "attn_dropout": getattr(args, 'attn_dropout', '?'),
            "num_res_blocks": getattr(args, 'num_res_blocks', '?'),
            "dropout_rate": getattr(args, 'dropout_rate', '?'),
            "output_mode": getattr(args, 'output_mode', '?')
        }

    # Training Params
    conf["train"]["max_train_epochs"] = getattr(args, 'num_train_epochs', '?') # Use calculated value
    conf["train"]["max_train_steps"] = getattr(args, 'max_train_steps', '?') # Use calculated value
    conf["train"]["lr"] = getattr(args, 'lr', '?')
    conf["train"]["batch"] = getattr(args, 'batch', '?')
    conf["train"]["optimizer"] = getattr(args, 'optimizer', '?')
    conf["train"]["loss_function"] = getattr(args, 'loss_function', '?')
    conf["train"]["precision"] = getattr(args, 'precision', '?')
    conf["train"]["val_split_count"] = getattr(args, 'val_split_count', '?')
    # Copy other train args dynamically (optimizer/scheduler specific)
    known_train_keys = {'max_train_epochs', 'max_train_steps', 'lr', 'batch', 'optimizer', 'loss_function', 'precision', 'val_split_count', 'num_train_epochs'} # Need to track calculated ones
    for key, value in vars(args).items():
         # Include keys starting with 'scheduler_' or common optimizer args,
         # but exclude keys already handled in other sections or base train keys.
         if key not in known_train_keys and \
            key not in conf["model"] and \
            key not in conf.get("predictor_params", {}) and \
            key not in conf.get("head_params", {}) and \
            key not in conf.get("attn_pool_params", {}) and \
            key not in ['config', 'resume', 'labels', 'weights', 'num_labels', 'data_root', 'wandb_project', 'name', 'arch', 'is_end_to_end', 'features', 'hidden_dim', 'output_mode', 'head_output_mode'] and \
            not key.startswith('_') and value is not None: # Exclude None values
             conf["train"][key] = value


    # Other top-level config
    if args.arch == "class": conf["labels"] = getattr(args, 'labels', None)
    conf["data_root"] = getattr(args, 'data_root', '?')
    conf["wandb_project"] = getattr(args, 'wandb_project', '?')

    # --- Clean up None values recursively ---
    def remove_none_values(d):
        if isinstance(d, dict):
            return {k: remove_none_values(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list):
            return [remove_none_values(i) for i in d if i is not None]
        else:
            return d
    conf = remove_none_values(conf)

    # Save config
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    config_path = f"{SAVE_FOLDER}/{args.name}.config.json"
    try:
        with open(config_path, "w") as f:
            f.write(json.dumps(conf, indent=2, default=repr))
        print(f"Saved final effective training config to {config_path}")
    except Exception as e:
        print(f"Error saving config file {config_path}: {e}")
# --- End write_config ---


# --- ModelWrapper Class ---
# Version 2.7.0: Updates evaluate_on_validation_set for E2E compatibility
class ModelWrapper:
    def __init__(self, name, model, optimizer, criterion, scheduler=None, device="cpu",
                 stdout=True, scaler=None, wandb_run=None, num_labels=1, log_file_path=None):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        self.wandb_run = wandb_run
        self.stdout = stdout
        self.num_labels = num_labels # Set during init

        self.losses = [] # Buffer for recent training losses
        self.current_epoch = 0
        self.current_global_step = 0
        self.best_val_loss = float('inf')
        self.current_best_val_model_path = None

        self._find_initial_best_model()

        # Setup logging
        if log_file_path is None: log_file_path = f"{SAVE_FOLDER}/{self.name}.csv"
        self.log_file_path = log_file_path
        file_mode = "a" if os.path.exists(self.log_file_path) else "w"
        self.csvlog = None
        try:
            self.csvlog = open(self.log_file_path, file_mode)
            if file_mode == "w":
                self.csvlog.write("step,train_loss_avg,eval_loss,learning_rate\n")
                self.csvlog.flush()
        except IOError as e: print(f"Warning: Could not open CSV log file {self.log_file_path}: {e}")
        print(f"ModelWrapper initialized. Logging to {self.log_file_path} (mode: {file_mode})")

    def _find_initial_best_model(self):
        """Checks save folder for existing _best_val file on init."""
        try:
            os.makedirs(SAVE_FOLDER, exist_ok=True)
            potential_best = None
            latest_mtime = 0
            for fname in os.listdir(SAVE_FOLDER):
                 if fname.startswith(self.name) and fname.endswith("_best_val.safetensors"):
                      fpath = os.path.join(SAVE_FOLDER, fname)
                      try:
                           mtime = os.path.getmtime(fpath)
                           if mtime > latest_mtime:
                                latest_mtime = mtime
                                potential_best = fpath
                      except OSError: continue
            if potential_best:
                 self.current_best_val_model_path = potential_best
                 print(f"Found existing best model checkpoint: {os.path.basename(potential_best)}")
                 # TODO: Optionally load the best_val_loss associated with this checkpoint if state is saved
            else:
                 print("No existing best model checkpoint found.")
        except Exception as e:
             print(f"Warning: Error finding initial best model: {e}")

    def update_step(self, global_step):
        """Update the internal global step counter."""
        self.current_global_step = global_step

    def get_current_step(self):
        """Get the current global step counter."""
        return self.current_global_step

    def log_step(self, loss):
        """Log training loss for averaging."""
        if not math.isnan(loss):
             self.losses.append(loss)
             if len(self.losses) > LOSS_MEMORY:
                  self.losses.pop(0)

    # v2.8.2: Added tqdm to validation loop
    @torch.no_grad()
    def evaluate_on_validation_set(self, val_loader, args):
        """
        Performs evaluation on the provided validation DataLoader.
        Handles both End-to-End (list batches) and Embedding-based (dict batch) models.
        """
        if val_loader is None:
             print("Validation loader not provided, skipping evaluation.")
             return float('nan')

        self.model.eval()
        is_e2e = getattr(args, 'is_end_to_end', False)

        # --- Set Optimizer Eval Mode ---
        original_optimizer_mode_is_training = False
        needs_optim_switch = (hasattr(self.optimizer, 'eval') and callable(self.optimizer.eval) and
                              hasattr(self.optimizer, 'train') and callable(self.optimizer.train) and
                              hasattr(self.optimizer, 'state') and any(s for s in self.optimizer.state.values()))
        if needs_optim_switch:
            is_training = getattr(self.optimizer, 'train_mode', True) # Assume training if flag unknown
            if is_training:
                try: self.optimizer.eval(); original_optimizer_mode_is_training = True
                except Exception as e: print(f"Warning: Error calling optimizer.eval(): {e}"); needs_optim_switch = False
        # --- End Optimizer Eval Mode ---

        total_loss = 0.0
        total_samples = 0
        # Determine AMP settings based on scaler state
        autocast_enabled = self.scaler is not None and self.scaler.is_enabled()
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 # Default AMP dtype

        # <<< Wrap val_loader with tqdm >>>
        val_iterator = tqdm(val_loader, desc="Validation", leave=False, dynamic_ncols=True)

        # --- Evaluation Loop ---
        for batch_data in val_iterator:  # <<< Iterate over tqdm iterator >>>
            if batch_data is None or not batch_data: continue

            if isinstance(batch_data, dict):
                batch_data_list = [batch_data]
            elif isinstance(batch_data, list):
                batch_data_list = batch_data
            else:
                continue  # Skip unexpected types

            # --- Inner Loop: Iterate through Mini-Batches ---
            for sub_batch in batch_data_list:
                try:
                    # --- Get Sub-Batch Data ---
                    target_val = sub_batch.get("label" if is_e2e else "val")
                    if target_val is None: continue

                    model_input_dict = {}
                    emb_input = None
                    current_sub_batch_size = 0
                    if is_e2e:
                        pixel_values = sub_batch.get("pixel_values")
                        if pixel_values is None: continue
                        model_input_dict["pixel_values"] = pixel_values.to(self.device)
                        current_sub_batch_size = pixel_values.size(0)
                    else:
                        emb_input = sub_batch.get("emb")
                        if emb_input is None: continue
                        emb_input = emb_input.to(self.device)
                        current_sub_batch_size = emb_input.size(0)

                    # --- Prepare Target ---
                    try:
                        if self.num_labels == 1:
                            target = target_val.to(self.device, dtype=torch.float32)
                        else:
                            target = target_val.to(self.device, dtype=torch.long)
                        if target.shape[0] != current_sub_batch_size: target = target.view(current_sub_batch_size,
                                                                                           -1).squeeze()
                        if target.shape[0] != current_sub_batch_size: raise ValueError("Target shape mismatch")
                    except Exception as e_target:
                        print(f"Error val target: {e_target}"); continue

                    # --- Prediction and Loss ---
                    loss = torch.tensor(float('nan'), device=self.device)
                    with torch.amp.autocast(device_type=self.device, enabled=autocast_enabled, dtype=amp_dtype):
                        if is_e2e:
                            y_pred = self.model(**model_input_dict)
                        else:
                            y_pred = self.model(emb_input)

                        # ... (Prepare y_pred_for_loss) ...
                        y_pred_for_loss = y_pred
                        if isinstance(self.criterion,
                                      (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)) and self.num_labels == 1:
                            if y_pred.ndim > 1 and y_pred.shape[1] == 1: y_pred_for_loss = y_pred.squeeze(-1)

                        # <<< Ensure calculation happens on CPU if possible to reduce VRAM peak? >>>
                        # <<< Or keep on GPU for speed? Let's keep on GPU for now. >>>
                        y_pred_final = y_pred_for_loss.to(torch.float32)
                        target_for_loss = target.to(y_pred_final.device)

                        # ... (Calculate loss based on criterion) ...
                        if isinstance(self.criterion, nn.NLLLoss):
                            loss_input = F.log_softmax(y_pred_final, dim=-1)  # NLLLoss expects log-probs
                            loss = self.criterion(loss_input, target_for_loss.long())
                        elif isinstance(self.criterion, (nn.CrossEntropyLoss, FocalLoss, GHMC_Loss)):
                            loss = self.criterion(y_pred_final, target_for_loss.long())  # These take logits
                        elif isinstance(self.criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)):
                            loss = self.criterion(y_pred_final, target_for_loss.float())  # These take logits/values
                        else:
                            loss = torch.tensor(float('nan'), device=self.device)

                    # --- Accumulate Loss ---
                    if not math.isnan(loss.item()):
                        total_loss += loss.item() * current_sub_batch_size
                        total_samples += current_sub_batch_size
                    else:
                        pass  # Avoid printing NaN warning every time

                    # <<< Explicitly delete tensors inside loop? Maybe helps? >>>
                    del target_val, model_input_dict, emb_input, target, y_pred, y_pred_for_loss, loss
                    # <<< Maybe even clear cache? >>>
                    # if torch.cuda.is_available(): torch.cuda.empty_cache() # Might slow things down!

                except Exception as e_val_sub:
                    print(f"Error processing validation sub-batch: {e_val_sub}")
                    traceback.print_exc()
                    continue
            # --- End Inner Mini-Batch Loop ---

            # <<< Update validation tqdm postfix >>>
            if total_samples > 0:
                val_iterator.set_postfix({"AvgLoss": f"{(total_loss / total_samples):.4e}"})

        # --- End Evaluation Loop ---
        val_iterator.close()  # Close the tqdm bar

        # --- Restore Modes ---
        self.model.train()
        if original_optimizer_mode_is_training and needs_optim_switch:
            if hasattr(self.optimizer, 'train') and callable(self.optimizer.train):
                try: self.optimizer.train()
                except Exception as e: print(f"Warning: Error calling optimizer.train(): {e}")

        # --- Calculate Average Loss ---
        if total_samples == 0:
            print("Warning: No valid samples processed during validation.")
            return float('nan')
        avg_loss = total_loss / total_samples
        print(f"Validation finished. Average Loss: {avg_loss:.4e} ({total_samples} samples)")
        return avg_loss

    # v2.8.1: Removed tqdm.write to allow TQDM postfix to handle updates
    def log_main(self, step, train_loss_batch, eval_loss):
        """Logs metrics to CSV and Wandb. Updates internal state."""
        self.update_step(step)
        self.log_step(train_loss_batch) # Add current step's avg loss to buffer

        lr = float(self.optimizer.param_groups[0]['lr']) if self.optimizer.param_groups else 0.0
        # Calculate long-term average loss from buffer for logging consistency
        train_loss_avg = sum(self.losses) / len(self.losses) if self.losses else float('nan')

        # <<< REMOVE Stdout via tqdm.write() >>>
        # if self.stdout:
        #     eval_loss_str = f"{eval_loss:.4e}" if not math.isnan(eval_loss) else "N/A"
        #     train_avg_str = f"{train_loss_avg:.4e}" if not math.isnan(train_loss_avg) else "N/A"
        #     tqdm.write(f"Step: {str(step):<8} | Loss(avg): {train_avg_str} | Eval Loss: {eval_loss_str} | LR: {lr:.3e}")

        # Wandb Logging (Keep)
        if self.wandb_run:
            log_data = {"train/loss_batch": train_loss_batch, "train/loss_avg": train_loss_avg, "train/learning_rate": lr}
            # Use the potentially updated eval_loss passed to this function
            if not math.isnan(eval_loss): log_data["eval/loss"] = eval_loss
            try:
                self.wandb_run.log(log_data, step=step)
            except Exception as e_wandb:
                print(f"Warning: Failed to log to WandB at step {step}: {e_wandb}")

        # CSV Logging (Keep)
        if self.csvlog:
            try:
                eval_loss_csv = eval_loss if not math.isnan(eval_loss) else ''
                train_avg_csv = train_loss_avg if not math.isnan(train_loss_avg) else ''
                self.csvlog.write(f"{step},{train_avg_csv},{eval_loss_csv},{lr}\n")
                self.csvlog.flush()
            except Exception as e_csv: print(f"Warning: Error writing CSV log: {e_csv}")

        # Update internal best loss tracking (Keep)
        if not math.isnan(eval_loss) and eval_loss < self.best_val_loss:
             self.best_val_loss = eval_loss
             # print(f"DEBUG Wrapper: Updated best_val_loss to {self.best_val_loss}") # Optional debug print

    # --- Model Saving ---
    # Version 2.8.6: Reverted to saving full state_dict always
    def save_model(self, step=None, epoch=None, suffix="", save_aux=False, args=None): # Still need args to know context
        """
        Saves model checkpoint and optionally training state.
        Always saves the full model state dictionary.
        Manages the single latest _best_val checkpoint.
        """
        if args is None: print("Error: 'args' object not provided to save_model."); return

        # <<< Always get the full state dict >>>
        state_dict_to_save = self.model.state_dict()
        save_full_model = True # Always true now

        # Determine context for logging message
        is_e2e = getattr(args, 'is_end_to_end', False)
        is_frozen = getattr(args, 'freeze_base_model', True) # Default True if missing

        # --- Filenames ---
        current_global_step = step if step is not None else self.current_global_step
        current_epoch_to_save = epoch if epoch is not None else self.current_epoch
        step_str = ""
        if current_global_step is not None:
             if current_global_step >= 1_000_000: step_str = f"_s{round(current_global_step / 1_000_000, 1)}M"
             elif current_global_step >= 1_000: step_str = f"_s{round(current_global_step / 1_000)}K"
             else: step_str = f"_s{current_global_step}"
        epoch_str = "_efinal" if isinstance(epoch, str) and epoch.lower() == "final" else ""
        base_output_name = f"./{SAVE_FOLDER}/{self.name}{step_str}{epoch_str}{suffix}"
        model_output_path = f"{base_output_name}.safetensors"
        optim_output_path = f"{base_output_name}.optim"
        sched_output_path = f"{base_output_name}.sched"
        scaler_output_path = f"{base_output_name}.scaler"
        state_output_path = f"{base_output_name}.state"

        is_best = "_best_val" in suffix

        # Log message indicating context
        context_str = "Unknown"
        if is_e2e: context_str = "E2E (Base Frozen)" if is_frozen else "E2E (Base Unfrozen)"
        else: context_str = "Embedding-Based"
        print(f"\nSaving checkpoint: {os.path.basename(base_output_name)} (Epoch: {current_epoch_to_save}, Step: {current_global_step}) - [Full Model - Context: {context_str}]")

        # <<< Remove pre-save checks/debug saves >>>

        try:
            os.makedirs(SAVE_FOLDER, exist_ok=True)
            # --- Delete OLD Best ---
            if is_best:
                old_best_model_path = self.current_best_val_model_path
                if old_best_model_path and os.path.normpath(old_best_model_path) != os.path.normpath(model_output_path):
                    old_base = os.path.splitext(old_best_model_path)[0]
                    files_to_remove = [old_best_model_path] + [old_base + ext for ext in
                                                               ['.optim', '.sched', '.scaler', '.state']]
                    print(f"  New best found. Removing previous best files starting with: {os.path.basename(old_base)}")
                    removed_count = 0
                    for f_path in files_to_remove:
                        if os.path.exists(f_path):
                            try:
                                os.remove(f_path); removed_count += 1
                            except OSError as e:
                                print(f"  Warning: Could not remove '{os.path.basename(f_path)}': {e}")
                    if removed_count > 0: print(f"  Removed {removed_count} previous best file(s).")
                    self.current_best_val_model_path = None

            # --- Save CURRENT Checkpoint Files ---
            save_file(self.model.state_dict(), model_output_path)

            print(f"Checkpoint files based on ({os.path.basename(model_output_path)}) saved successfully. Actual size: {os.path.getsize(model_output_path)/(1024*1024):.2f} MB") # Print actual size

            if save_aux:
                print("  Saving auxiliary files (optim, sched, scaler, state)...")
                torch.save(self.optimizer.state_dict(), optim_output_path)
                if self.scheduler is not None:
                    try:
                        torch.save(self.scheduler.state_dict(), sched_output_path)
                    except Exception as e_sched:
                        print(f"  Warning: Failed to save scheduler state: {e_sched}")
                if self.scaler is not None and self.scaler.is_enabled():
                    torch.save(self.scaler.state_dict(), scaler_output_path)
                train_state = {
                    'epoch': current_epoch_to_save, 'global_step': current_global_step,
                    'best_val_loss': self.best_val_loss
                }
                torch.save(train_state, state_output_path)
            else:
                # Clean up orphaned aux files *for this specific checkpoint name*
                # This prevents leaving old .optim files if save_aux becomes False later
                print(
                    "  save_aux is False. Skipping save/Ensuring removal of auxiliary files for this specific checkpoint name.")
                for ext in ['.optim', '.sched', '.scaler', '.state']:
                    path_to_check = base_output_name + ext
                    if os.path.exists(path_to_check):
                        try:
                            os.remove(path_to_check)
                            print(f"  Removed existing aux file: {os.path.basename(path_to_check)}")
                        except OSError as e:
                            print(
                                f"  Warning: Failed to remove existing aux file {os.path.basename(path_to_check)}: {e}")

                print(
                    f"Checkpoint files based on ({os.path.basename(model_output_path)}) saved successfully. Actual size: {os.path.getsize(model_output_path) / (1024 * 1024):.2f} MB")

                if is_best:
                    self.current_best_val_model_path = model_output_path
                    print(f"  Marked {os.path.basename(model_output_path)} as new best model path.")

        except Exception as e:
            print(f"Error saving checkpoint {base_output_name}: {e}")
            import traceback
            traceback.print_exc()

    def close(self):
        """Closes the CSV log file."""
        if self.csvlog:
            try: self.csvlog.close()
            except Exception as e: print(f"Warning: Error closing CSV log file: {e}")
            finally: self.csvlog = None
        print("ModelWrapper resources closed.")
# --- End ModelWrapper Class ---

# --- Helper function to load states ---
# (Moved out of ModelWrapper, now used by load_checkpoint in train.py)
def load_optimizer_state(optimizer, path, device):
     if os.path.exists(path):
         try:
             optimizer.load_state_dict(torch.load(path, map_location=device))
             print(f"Optimizer state loaded from {os.path.basename(path)}.")
             return True
         except Exception as e:
             print(f"Warning: Could not load optimizer state from {os.path.basename(path)}: {e}")
     else:
         print(f"Warning: Optimizer state file not found at {path}. Starting fresh.")
     return False

def load_scheduler_state(scheduler, path, device):
     if scheduler is not None and os.path.exists(path):
         try:
             scheduler.load_state_dict(torch.load(path, map_location=device))
             print(f"Scheduler state loaded from {os.path.basename(path)}.")
             return True
         except Exception as e:
             print(f"Warning: Could not load scheduler state from {os.path.basename(path)}: {e}")
     return False

def load_scaler_state(scaler, path, device):
      if scaler is not None and scaler.is_enabled() and os.path.exists(path):
          try:
              scaler.load_state_dict(torch.load(path, map_location=device))
              print(f"GradScaler state loaded from {os.path.basename(path)}.")
              return True
          except Exception as e:
              print(f"Warning: Could not load GradScaler state from {os.path.basename(path)}: {e}")
      return False
# --- End Helper Functions ---