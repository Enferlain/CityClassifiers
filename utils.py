# Version: 2.3.0 (Refactored ModelWrapper, Added Loss Arg, Cleaned Saving)
import math
import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import shutil # Added for removing old best checkpoints
from tqdm import tqdm
from safetensors.torch import save_file, load_file # Added load_file
from losses import GHMC_Loss, FocalLoss

# --- Constants ---
LOSS_MEMORY = 500 # How many recent steps' training loss to average for display
LOG_EVERY_N = 100 # How often to log metrics and check validation
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
        "hf_dinov2_giant_FitPad": {"features": 1536, "hidden": 1280}, # ViT-Giant
        "vit_large_patch14_dinov2.lvd142m_FitPad": {"features": 1024, "hidden": 1280}, # ViT-Large

        # --- Other ---
        "META": {"features": 1024, "hidden": 1280}, # Legacy MetaCLIP Example
    }
    if ver in embed_configs:
        return embed_configs[ver]
    else:
        # Attempt to infer from HF model name if ver contains '/'? More complex.
        # For now, raise error for unknown explicitly defined versions.
        raise ValueError(f"Unknown/undefined embedding version key '{ver}' in get_embed_params. Please add it or check YAML config.")
# --- End Parameter Dictionary ---

# --- Argument Parsing and Config Loading ---
# Version 2.5.0: Simplified optimizer arg handling, rely on dynamic loader, added epoch args
def parse_and_load_args():
    """
    Parses command line args and merges with YAML config. Epoch-based aware.
    Command line args take precedence over YAML.
    """
    parser = argparse.ArgumentParser(description="Train aesthetic predictor/classifier")
    # --- Command Line Arguments ---
    parser.add_argument("--config", required=True, help="Training config YAML file")
    parser.add_argument('--resume', help="Checkpoint (.safetensors model file) to resume from")
    # Training Duration Overrides
    parser.add_argument('--max_train_epochs', type=int, default=None, help="Override: Train for a specific number of epochs.")
    parser.add_argument('--max_train_steps', type=int, default=None, help="Override: Train for a specific number of steps (takes priority).")
    # Other Overrides
    parser.add_argument('--precision', type=str, default=None, choices=['fp32', 'fp16', 'bf16'], help='Override training precision.')
    parser.add_argument("--nsave", type=int, default=None, help="Override save frequency (steps).")
    parser.add_argument("--val_split_count", type=int, default=None, help="Override validation split count.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--preload_data", action=argparse.BooleanOptionalAction, default=None, help="Override data preloading.")
    parser.add_argument("--data_root", type=str, default=None, help="Override data root directory.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Override WandB project name.")
    parser.add_argument('--optimizer', type=str, default=None, help='Override optimizer choice.')
    parser.add_argument('--loss_function', type=str, default=None, help="Override loss function choice.")
    parser.add_argument('--lr', type=float, default=None, help="Override learning rate.")
    parser.add_argument('--betas', type=float, nargs='+', default=None, help="Override optimizer betas.") # Keep common overrides
    parser.add_argument('--weight_decay', type=float, default=None, help='Override optimizer weight decay.')
    parser.add_argument('--eps', type=float, default=None, help='Override optimizer epsilon.')
    parser.add_argument('--batch', type=int, default=None, help='Override batch size.')

    cmd_args = parser.parse_args()

    # Load YAML config
    if not os.path.isfile(cmd_args.config): parser.error(f"Config file '{cmd_args.config}' not found.")
    try:
        with open(cmd_args.config) as f: conf = yaml.safe_load(f)
    except Exception as e: parser.error(f"Error loading YAML config '{cmd_args.config}': {e}")

    args = argparse.Namespace() # Final args object

    # Helper function remains the same
    def get_arg_value(arg_name, cmd_value, yaml_conf_section, yaml_key=None, default=None, expected_type=None):
        if yaml_key is None: yaml_key = arg_name
        yaml_value = yaml_conf_section.get(yaml_key, default)
        final_value = cmd_value if cmd_value is not None else yaml_value
        if final_value is not None and expected_type is not None:
            try:
                if expected_type == bool: final_value = bool(final_value)
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
    # Model Info
    model_conf = conf.get("model", {})
    args.base = model_conf.get("base", "unknown_model"); args.rev = model_conf.get("rev", "v0.0")
    args.arch = model_conf.get("arch", "class"); args.embed_ver = model_conf.get("embed_ver", "unknown")
    args.name = f"{args.base}-{args.rev}"
    args.base_vision_model = model_conf.get("base_vision_model")
    # Predictor Params (Directly from YAML)
    predictor_conf = conf.get("predictor_params", {})
    args.use_attention = predictor_conf.get("use_attention", True); args.num_attn_heads = predictor_conf.get("num_attn_heads", 8)
    args.attn_dropout = predictor_conf.get("attn_dropout", 0.1); args.hidden_dim = predictor_conf.get("hidden_dim", None)
    args.num_res_blocks = predictor_conf.get("num_res_blocks", 1); args.dropout_rate = predictor_conf.get("dropout_rate", 0.1)
    args.output_mode = predictor_conf.get("output_mode", None) # MUST be set in YAML

    # Training Params (Merged)
    train_conf = conf.get("train", {})
    args.lr = get_arg_value('lr', cmd_args.lr, train_conf, default=1e-4, expected_type=float)
    args.batch = get_arg_value('batch', cmd_args.batch, train_conf, default=4, expected_type=int)
    args.loss_function = get_arg_value('loss_function', cmd_args.loss_function, train_conf)
    args.optimizer = get_arg_value('optimizer', cmd_args.optimizer, train_conf, default='AdamW')
    # Common overrides
    args.betas = get_arg_value('betas', cmd_args.betas, train_conf, expected_type=tuple)
    args.eps = get_arg_value('eps', cmd_args.eps, train_conf, expected_type=float)
    args.weight_decay = get_arg_value('weight_decay', cmd_args.weight_decay, train_conf, expected_type=float)
    # Training duration (values will be calculated later in train.py)
    args.max_train_epochs = get_arg_value('max_train_epochs', cmd_args.max_train_epochs, train_conf, default=None, expected_type=int)
    args.max_train_steps = get_arg_value('max_train_steps', cmd_args.max_train_steps, train_conf, default=None, expected_type=int)

    # Other Params
    args.precision = get_arg_value('precision', cmd_args.precision, train_conf, default='fp32')
    args.nsave = get_arg_value('nsave', cmd_args.nsave, train_conf, default=10000, expected_type=int)
    args.val_split_count = get_arg_value('val_split_count', cmd_args.val_split_count, train_conf, default=0, expected_type=int)
    args.seed = get_arg_value('seed', cmd_args.seed, train_conf, default=218, expected_type=int)
    args.num_workers = get_arg_value('num_workers', cmd_args.num_workers, train_conf, default=0, expected_type=int)
    args.preload_data = get_arg_value('preload_data', cmd_args.preload_data, train_conf, default=True, expected_type=bool)
    args.data_root = get_arg_value('data_root', cmd_args.data_root, conf, default="data") # Get from top level or default
    args.wandb_project = get_arg_value('wandb_project', cmd_args.wandb_project, conf, default="city-classifiers") # Get from top level or default

    # --- Copy ALL OTHER keys from train_conf directly into args ---
    # This passes through optimizer/scheduler specific args from YAML
    handled_keys = ['lr', 'max_train_steps', 'max_train_epochs', 'batch', 'loss_function', 'optimizer', 'betas', 'eps', 'weight_decay', 'precision', 'nsave', 'val_split_count', 'seed', 'num_workers', 'preload_data']
    for key, value in train_conf.items():
         if key not in handled_keys and not hasattr(args, key):
              # Attempt type inference for common scheduler args if needed
              if isinstance(value, list): value = tuple(value) # Convert lists to tuples for consistency?
              setattr(args, key, value)
    # --- End Copy ---

    # --- Set Defaults based on embed_ver and arch ---
    try: embed_params = get_embed_params(args.embed_ver)
    except ValueError as e: print(f"Error: {e}"); embed_params = {}
    if args.hidden_dim is None: args.hidden_dim = embed_params.get('hidden', 1280)
    args.features = embed_params.get('features')
    if args.features is None: exit(f"ERROR: Could not determine embedding features for '{args.embed_ver}'.")

    # Labels/Weights
    labels_conf = conf.get("labels", {})
    args.labels = None; args.weights = None; args.num_labels = 1 if args.arch == 'score' else 0
    if args.arch == "class":
        if labels_conf:
            args.labels = {str(k): v.get("name", str(k)) for k, v in labels_conf.items()}
            try: args.num_labels = max(int(k) for k in labels_conf.keys()) + 1
            except: args.num_labels = 0 # Handle empty labels case
            if args.num_labels > 0:
                weights = [1.0] * args.num_labels
                for k_str, label_conf in labels_conf.items():
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

    if args.output_mode is None and args.arch != 'score':
         exit(f"ERROR: predictor_params.output_mode must be set in YAML for arch '{args.arch}'.")
    if args.max_train_epochs is None and args.max_train_steps is None:
         print("Warning: Neither max_train_epochs nor max_train_steps specified. Defaulting to max_train_steps = 100000.")
         args.max_train_steps = 100000

    # Defer final print until after calculations in train.py

    return args
# --- End Argument Parsing ---

# --- Updated write_config Function ---
# Version 2.5.0: Saves calculated steps/epochs and scheduler args
def write_config(args):
    """Writes the final training configuration, including calculated steps/epochs and scheduler args."""
    try: embed_params = get_embed_params(args.embed_ver)
    except ValueError as e: print(f"Error: {e}"); embed_params = {}

    conf = {
        "model": {
            "base": getattr(args, 'base', '?'), "rev": getattr(args, 'rev', '?'),
            "arch": getattr(args, 'arch', '?'), "embed_ver": getattr(args, 'embed_ver', '?'),
            "base_vision_model": getattr(args, 'base_vision_model', None),
        },
        "predictor_params": {
            "features": getattr(args, 'features', '?'),
            "hidden_dim": getattr(args, 'hidden_dim', '?'),
            "num_classes": getattr(args, 'num_classes', '?'),
            "use_attention": getattr(args, 'use_attention', '?'),
            "num_attn_heads": getattr(args, 'num_attn_heads', '?'),
            "attn_dropout": getattr(args, 'attn_dropout', '?'),
            "num_res_blocks": getattr(args, 'num_res_blocks', '?'),
            "dropout_rate": getattr(args, 'dropout_rate', '?'),
            "output_mode": getattr(args, 'output_mode', '?')
        },
        "train": {
            # Save calculated values if they exist
            "max_train_epochs": getattr(args, 'num_train_epochs', '?'),
            "max_train_steps": getattr(args, 'max_train_steps', '?'),
            # Save other core params
            "lr": getattr(args, 'lr', '?'), "batch": getattr(args, 'batch', '?'),
            "optimizer": getattr(args, 'optimizer', '?'), "loss_function": getattr(args, 'loss_function', '?'),
            "precision": getattr(args, 'precision', '?'), "val_split_count": getattr(args, 'val_split_count', '?'),
            # Save optimizer specific args found on args object
            **{k: v for k, v in vars(args).items() if k in [
                'betas', 'eps', 'weight_decay', 'weight_decouple', 'rectify', # Add others if needed
                'gamma', 'r_sf', 'wlpow_sf', 'state_precision', 'adaptive_clip', 'adaptive_clip_eps',
                 # ... copy relevant keys from the list in parse_and_load_args generic loop ...
                'focal_loss_gamma', 'ghm_bins', 'ghm_momentum' # Add loss-specific args
                ] and hasattr(args, k) and getattr(args, k) is not None},
            # Save scheduler specific args found on args object
            "scheduler_name": getattr(args, 'scheduler_name', None),
             **{k: v for k, v in vars(args).items() if k.startswith('scheduler_') and hasattr(args, k) and getattr(args, k) is not None},
        },
        "labels": getattr(args, 'labels', None) if getattr(args, 'arch', 'class') == "class" else None,
        "data_root": getattr(args, 'data_root', '?'),
        "wandb_project": getattr(args, 'wandb_project', '?'),
    }
    # Remove None values for cleaner output
    conf['train'] = {k: v for k, v in conf['train'].items() if v is not None}
    if conf['labels'] is None: del conf['labels']

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
# Version 2.6.0: Saves/loads epoch and global step state
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
        self.num_labels = num_labels

        self.losses = []
        self.current_epoch = 0 # Tracks CURRENT epoch index (0-based) being processed or just completed
        self.current_global_step = 0 # Tracks total steps completed
        self.best_val_loss = float('inf')
        self.current_best_val_model_path = None # Path to the *.safetensors file

        # Initialize or find existing best model path
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
            lowest_loss = float('inf') # Ignored for now, just find latest file
            latest_mtime = 0

            for fname in os.listdir(SAVE_FOLDER):
                 if fname.startswith(self.name) and fname.endswith("_best_val.safetensors"):
                      fpath = os.path.join(SAVE_FOLDER, fname)
                      try:
                           mtime = os.path.getmtime(fpath)
                           if mtime > latest_mtime:
                                latest_mtime = mtime
                                potential_best = fpath
                      except OSError: continue # Skip if file disappears

            if potential_best:
                 self.current_best_val_model_path = potential_best
                 print(f"Found existing best model checkpoint: {os.path.basename(potential_best)}")
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

    # v2.3.1: Simplified + correct device/dtype/shape handling
    def evaluate_on_validation_set(self, val_loader):
        """Performs evaluation on the provided validation DataLoader."""
        if val_loader is None:
             print("Validation loader not provided, skipping evaluation.")
             return float('nan')

        self.model.eval()  # Set model to evaluation mode
        # --- Set Optimizer Eval Mode (if applicable) ---
        original_optimizer_mode_is_training = False
        # Check if optimizer has state and distinct eval/train modes
        needs_optim_switch = (hasattr(self.optimizer, 'eval') and callable(self.optimizer.eval) and
                              hasattr(self.optimizer, 'train') and callable(self.optimizer.train) and
                              hasattr(self.optimizer, 'state') and any(s for s in self.optimizer.state.values()))
        if needs_optim_switch:
            # Try to determine current mode and switch to eval
            is_training = False
            if hasattr(self.optimizer, 'train_mode'): # Explicit flag
                is_training = self.optimizer.train_mode
            elif hasattr(self.optimizer, 'param_groups') and any(pg.get('step', 0) > 0 for pg in self.optimizer.param_groups): # Heuristic: check steps
                is_training = True

            if is_training:
                try:
                    self.optimizer.eval()
                    original_optimizer_mode_is_training = True # Mark that we need to switch back
                except Exception as e:
                    print(f"Warning: Error calling optimizer.eval(): {e}")
                    needs_optim_switch = False # Don't try to switch back if eval failed
        # --- End Optimizer Eval Mode ---

        total_loss = 0.0
        total_samples = 0
        autocast_enabled = self.scaler is not None and self.scaler.is_enabled()
        amp_dtype = torch.float32
        if autocast_enabled:
            # Determine AMP dtype
            if hasattr(self.scaler, 'get_amp_dtype'): current_amp_dtype = self.scaler.get_amp_dtype()
            elif self.device == 'cuda' and torch.cuda.is_bf16_supported(): current_amp_dtype = torch.bfloat16
            else: current_amp_dtype = torch.float16 # Default AMP dtype
            amp_dtype = current_amp_dtype # Use determined dtype

        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue # Skip bad batches from dataloader

                emb = batch.get("emb")
                val = batch.get("val")
                if emb is None or val is None:
                    print("Warning: Validation batch missing 'emb' or 'val'. Skipping.")
                    continue

                emb = emb.to(self.device) # Move embeddings to device

                # --- Prepare Target Tensor (val) ---
                try:
                    if self.num_labels == 1: # Scorer mode
                        val = val.to(device=self.device, dtype=torch.float32) # Needs Float on device
                        if val.ndim == 2 and val.shape[1] == 1: val = val.squeeze(1) # [B, 1] -> [B]
                        elif val.ndim != 1: val = val.squeeze() # Try general squeeze if needed
                        if val.ndim != 1: raise ValueError(f"Scorer target shape {val.shape}, expected 1D [B].")
                    elif self.num_labels > 1: # Classifier mode
                        if val.dtype == torch.float and val.ndim == 2: val = torch.argmax(val, dim=1) # Handle one-hot float input
                        val = val.squeeze().to(device=self.device, dtype=torch.long) # Ensure Long, on device, shape [B]
                        if val.ndim != 1: raise ValueError(f"Classifier target shape {val.shape}, expected 1D [B].")
                    else: # Should not happen
                        raise ValueError(f"Invalid num_labels ({self.num_labels})")
                except Exception as e:
                    print(f"Error processing val tensor in validation: {e}")
                    print(f"  Original shape: {batch.get('val').shape}, dtype: {batch.get('val').dtype}")
                    continue # Skip batch if target prep fails
                # --- End Target Prep ---

                batch_size = emb.size(0)
                if emb.shape[0] != val.shape[0]:
                     print(f"Error: Validation batch size mismatch emb ({emb.shape[0]}) vs val ({val.shape[0]}). Skipping.")
                     continue

                # --- Prediction and Loss ---
                loss = torch.tensor(0.0, device=self.device)  # Initialize loss
                try:  # Add try/except around prediction and loss calc
                    with torch.amp.autocast(device_type=self.device, enabled=autocast_enabled, dtype=amp_dtype):
                        y_pred = self.model(emb)  # Model forward pass

                        # Prepare prediction shape if needed (ensure y_pred is defined)
                        y_pred_for_loss = y_pred
                        if isinstance(self.criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)):
                            # These expect [B] input if num_classes=1
                            if y_pred.ndim == 2 and y_pred.shape[1] == 1:
                                y_pred_for_loss = y_pred.squeeze(1)
                        # CE/Focal/NLL expect [B, C] input, so no squeeze needed here for y_pred

                    # Ensure prediction is Float32 for stability/activation
                    y_pred_final = y_pred_for_loss.to(torch.float32)

                    # --- Calculate loss with Correct Target Dtypes ---
                    if isinstance(self.criterion, (torch.nn.CrossEntropyLoss, FocalLoss)):
                        # Expects logits [B, C] and Long target [B]
                        # Ensure val is Long
                        loss = self.criterion(y_pred_final, val.long())

                    elif isinstance(self.criterion, nn.NLLLoss):
                        # Expects LogSoftmax input [B, C] and Long target [B]
                        print("DEBUG eval: Applying LogSoftmax before NLLLoss.")
                        log_probs = F.log_softmax(y_pred_final, dim=-1)
                        # Ensure val is Long
                        loss = self.criterion(log_probs, val.long())

                    elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
                        # Expects logits [B] and Float target [B]
                        # Ensure val is Float and shape [B]
                        if val.ndim != 1: val = val.squeeze()  # Make sure val is [B]
                        loss = self.criterion(y_pred_final, val.float())

                    elif isinstance(self.criterion, (nn.L1Loss, nn.MSELoss)):
                        # Expects prediction [B] and Float target [B]
                        # Ensure val is Float and shape [B]
                        if val.ndim != 1: val = val.squeeze()  # Make sure val is [B]
                        loss = self.criterion(y_pred_final, val.float())

                        # <<< ADD THIS GHM BLOCK >>>
                    elif isinstance(self.criterion, GHMC_Loss):
                        # GHMC Loss expects logits [B, C] and Long target [B] (like CE/Focal)
                        print("DEBUG eval: Calculating GHMC_Loss.")
                        # Ensure val is Long
                        loss = self.criterion(y_pred_final, val.long())
                    # <<< END GHM BLOCK >>>

                    else:  # Fallback for truly unknown
                        print(
                            f"Warning: Unknown criterion type {type(self.criterion)} in validation loop. Loss not calculated.")
                        loss = torch.tensor(float('nan'), device=self.device)  # Set loss to NaN if unknown
                    # --- End Loss Calculation ---

                except Exception as e_loss:
                    print(f"Error during validation prediction/loss calculation: {e_loss}")
                    print(
                        f"  Input emb shape: {emb.shape}, Target val shape: {val.shape}, Target val dtype: {val.dtype}")
                    # Handle error, maybe set loss to NaN or skip accumulation?
                    loss = torch.tensor(float('nan'), device=self.device)  # Set loss to NaN on error
                # --- End Prediction and Loss ---

                # --- Accumulate Loss ---
                if not math.isnan(loss.item()):
                    total_loss += loss.item() * batch_size # Use item's batch size
                    total_samples += batch_size
                else:
                    print("Warning: NaN encountered in validation loss calculation.")
                # --- End Accumulate Loss ---

            # End batch loop
        # End torch.no_grad()

        # --- Restore Modes ---
        self.model.train()
        if original_optimizer_mode_is_training and hasattr(self.optimizer, 'train') and callable(self.optimizer.train):
            self.optimizer.train()

        if total_samples == 0: print("Warning: No valid samples in validation."); return float('nan')
        return total_loss / total_samples

    def log_main(self, step, train_loss_batch, eval_loss):
        """Logs metrics to stdout, CSV, and Wandb."""
        self.update_step(step) # Update internal step counter
        self.log_step(train_loss_batch) # Add loss to buffer

        lr = float(self.optimizer.param_groups[0]['lr']) if self.optimizer.param_groups else 0.0
        train_loss_avg = sum(self.losses) / len(self.losses) if self.losses else float('nan')

        # Stdout
        if self.stdout:
            eval_loss_str = f"{eval_loss:.4e}" if not math.isnan(eval_loss) else "N/A"
            train_avg_str = f"{train_loss_avg:.4e}" if not math.isnan(train_loss_avg) else "N/A"
            tqdm.write(f"{str(step):<10} Loss(avg): {train_avg_str} | Eval Loss: {eval_loss_str} | LR: {lr:.4e}")

        # Wandb
        if self.wandb_run:
            log_data = {"train/loss_batch": train_loss_batch, "train/loss_avg": train_loss_avg, "train/learning_rate": lr}
            if not math.isnan(eval_loss): log_data["eval/loss"] = eval_loss
            self.wandb_run.log(log_data, step=step)

        # CSV
        if self.csvlog:
            try:
                eval_loss_csv = eval_loss if not math.isnan(eval_loss) else ''
                train_avg_csv = train_loss_avg if not math.isnan(train_loss_avg) else ''
                self.csvlog.write(f"{step},{train_avg_csv},{eval_loss_csv},{lr}\n")
                self.csvlog.flush()
            except Exception as e_csv: print(f"Warning: Error writing CSV log: {e_csv}")

        # --- Update Best Loss Tracking ---
        # Note: Saving is now handled ONLY by the explicit save_model call from train_loop
        if not math.isnan(eval_loss) and eval_loss < self.best_val_loss:
             self.best_val_loss = eval_loss
             # We don't save here anymore, just update the best loss value

    # --- Model Saving ---
    # Version 2.6.2: Default save_aux=False, improved deletion logic
    def save_model(self, step=None, epoch=None, suffix="", save_aux=False): # <<< Changed default to False
        """
        Saves model checkpoint and optionally training state.
        Manages the single latest _best_val checkpoint.
        Auxiliary files (optim, sched, scaler, state) saved ONLY if save_aux is True.
        """
        current_global_step = step if step is not None else self.current_global_step
        current_epoch_to_save = epoch if epoch is not None else self.current_epoch

        # --- Determine Filename Components ---
        step_str = ""
        if current_global_step >= 1_000_000: step_str = f"_s{round(current_global_step / 1_000_000, 1)}M"
        elif current_global_step >= 1_000: step_str = f"_s{round(current_global_step / 1_000)}K"
        else: step_str = f"_s{current_global_step}"
        epoch_str = f"_e{current_epoch_to_save}" if isinstance(epoch, str) else "" # Only add if epoch is string like "final"
        base_output_name = f"./{SAVE_FOLDER}/{self.name}{epoch_str}{step_str}{suffix}"

        model_output_path = f"{base_output_name}.safetensors"
        optim_output_path = f"{base_output_name}.optim"
        sched_output_path = f"{base_output_name}.sched"
        scaler_output_path = f"{base_output_name}.scaler"
        state_output_path = f"{base_output_name}.state"

        is_best = "_best_val" in suffix
        print(f"\nSaving checkpoint: {os.path.basename(base_output_name)} (Epoch: {current_epoch_to_save}, Global Step: {current_global_step})")

        try:
            # --- Delete OLD Best Checkpoint & Aux FIRST (if this is a new best) ---
            if is_best:
                old_best_model_path = self.current_best_val_model_path
                # Check if old best exists and is different from the new one we're saving
                if old_best_model_path and os.path.normpath(old_best_model_path) != os.path.normpath(model_output_path):
                     old_base = os.path.splitext(old_best_model_path)[0]
                     files_to_remove = [old_best_model_path] + [old_base + ext for ext in ['.optim', '.sched', '.scaler', '.state']]
                     print(f"  New best found. Removing previous best files starting with: {os.path.basename(old_base)}")
                     removed_count = 0
                     for f_path in files_to_remove:
                         if os.path.exists(f_path):
                             try:
                                 os.remove(f_path)
                                 removed_count += 1
                             except OSError as e: print(f"  Warning: Could not remove '{os.path.basename(f_path)}': {e}")
                     if removed_count > 0: print(f"  Removed {removed_count} previous best file(s).")
                     self.current_best_val_model_path = None # Clear old path
                # else: print("  New best model has same name or no previous best known.")

            # --- Save CURRENT Checkpoint Files ---
            # Always save the model weights
            save_file(self.model.state_dict(), model_output_path)

            # Save Aux Files ONLY if requested
            if save_aux:
                print("  Saving auxiliary files (optim, sched, scaler, state)...")
                torch.save(self.optimizer.state_dict(), optim_output_path)
                if self.scheduler is not None:
                     try: torch.save(self.scheduler.state_dict(), sched_output_path)
                     except Exception as e_sched: print(f"  Warning: Failed to save scheduler state: {e_sched}")
                if self.scaler is not None and self.scaler.is_enabled():
                     torch.save(self.scaler.state_dict(), scaler_output_path)
                # Save Epoch (completed) and Global Step (reached)
                train_state = {'epoch': current_epoch_to_save, 'global_step': current_global_step}
                torch.save(train_state, state_output_path)
            else:
                 # If not saving aux, ensure any potentially orphaned aux files for THIS name are removed
                 # (e.g., if a periodic save happened before, but now only model is saved for best_val)
                 if is_best: # Only cleanup aux for best_val if save_aux is False
                      for ext in ['.optim', '.sched', '.scaler', '.state']:
                           if os.path.exists(base_output_name + ext):
                                try: os.remove(base_output_name + ext)
                                except OSError: pass # Ignore errors here

            print(f"Checkpoint base ({os.path.basename(model_output_path)}) saved successfully.")

            # --- Update Path if This Was the Best Model ---
            if is_best:
                self.current_best_val_model_path = model_output_path
                print(f"  Marked {os.path.basename(model_output_path)} as new best model path.")

        except Exception as e:
            print(f"Error saving checkpoint {base_output_name}: {e}")
            import traceback
            traceback.print_exc()
    # --- End Model Saving ---

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