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


# Version 3.1.0: Always load base_vision_model and embed_ver for inference needs.
def parse_and_load_args(config_path: str):
    """
    Loads configuration SOLELY from a YAML file.
    Ensures base_vision_model is always loaded for inference compatibility.
    """
    if not os.path.isfile(config_path):
         raise FileNotFoundError(f"Config file '{config_path}' not found.")
    try:
        with open(config_path) as f:
            conf = yaml.safe_load(f)
    except Exception as e:
         raise ValueError(f"Error loading YAML config '{config_path}': {e}") from e

    args = argparse.Namespace()
    args.config_path = config_path # Store config path for reference

    # --- Helper to get required value from nested dict ---
    def get_required_config(key_path, config_dict):
        keys = key_path.split('.')
        value = config_dict
        try:
            for key in keys:
                value = value[key]
            if value is None: raise KeyError # Treat None as missing for required keys
            return value
        except (KeyError, TypeError):
             raise ValueError(f"Missing required configuration key: '{key_path}' in {config_path}")

    # --- Helper to get optional value ---
    def get_optional_config(key_path, config_dict, default=None):
         keys = key_path.split('.')
         value = config_dict
         try:
             for key in keys: value = value[key]
             # Return default if YAML value is explicitly None, otherwise return YAML value
             return default if value is None else value
         except (KeyError, TypeError):
             return default # Return default if key path doesn't exist

    # --- Load Common / Top-Level Settings (no changes here) ---
    args.data_root = get_optional_config("data_root", conf, default="data")
    args.wandb_project = get_optional_config("wandb_project", conf, default="city-classifiers")
    args.resume = get_optional_config("resume", conf, default=None)

    # --- Load Model Info (ALWAYS Load These Keys) ---
    args.base = get_required_config("model.base", conf)
    args.rev = get_required_config("model.rev", conf)
    args.arch = get_optional_config("model.arch", conf, default="class")
    args.name = f"{args.base}-{args.rev}"
    # <<< --- ALWAYS LOAD these for inference compatibility --- >>>
    args.base_vision_model = get_required_config("model.base_vision_model", conf)
    args.embed_ver = get_optional_config("model.embed_ver", conf) # Optional, but useful context
    # <<< --- END ALWAYS LOAD --- >>>

    # --- Load Data Mode ---
    args.data_mode = get_required_config("data.mode", conf) # e.g., 'embeddings', 'features', 'images'

    # --- Mode-Specific Loading ---
    # (These sections now only load params SPECIFIC to that mode's training)
    if args.data_mode == 'embeddings':
        print("DEBUG: Loading config for EMBEDDINGS mode...")
        args.is_end_to_end = False
        if not args.embed_ver:  # embed_ver is essential for this mode
            raise ValueError("Config Error: 'model.embed_ver' must be specified for embeddings mode.")
        try:
            embed_params = get_embed_params(args.embed_ver)
        except ValueError as e:
            raise ValueError(f"Config Error: {e}") from e
        args.features = embed_params['features']  # Input features for PredictorModel
        default_hidden = embed_params.get('hidden', 1280)
        args.preload_data = get_optional_config("train.preload_data", conf, default=True)

        # Load PredictorModel params (these define the trainable head for this mode)
        predictor_conf = get_optional_config("predictor_params", conf, default={})
        args.hidden_dim = predictor_conf.get("hidden_dim", default_hidden)
        args.use_attention = predictor_conf.get("use_attention", True)
        args.num_attn_heads = predictor_conf.get("num_attn_heads", 8)
        args.attn_dropout = predictor_conf.get("attn_dropout", 0.1)
        args.num_res_blocks = predictor_conf.get("num_res_blocks", 1)
        args.dropout_rate = predictor_conf.get("dropout_rate", 0.1)
        # Output mode is REQUIRED for the predictor model
        args.output_mode = get_required_config("predictor_params.output_mode", conf)

    elif args.data_mode == 'features':
        print("DEBUG: Loading config for FEATURES mode...")
        args.is_end_to_end = False
        args.feature_dir_name = get_required_config("data.feature_dir_name", conf)
        args.preload_data = False  # Cannot easily preload large features

        # Load HeadModel params (these define the trainable head for this mode)
        head_conf = get_optional_config("head_params", conf, default={})
        args.head_features = get_required_config("head_params.features", conf)  # Input features REQUIRED
        args.head_hidden_dim = head_conf.get("hidden_dim", 1024)
        args.pooling_strategy = head_conf.get("pooling_strategy", 'attn')  # Pooling used *during training*
        args.head_num_res_blocks = head_conf.get("num_res_blocks", 3)
        args.head_dropout_rate = head_conf.get("dropout_rate", 0.2)
        # Output mode is REQUIRED for the head model
        args.head_output_mode = get_required_config("head_params.output_mode", conf)
        # Load Attn Pool specific args only if needed by the strategy
        if args.pooling_strategy == 'attn':
            attn_conf = get_optional_config("attn_pool_params", conf, default={})
            args.attn_pool_heads = attn_conf.get("attn_pool_heads", 16)
            args.attn_pool_dropout = attn_conf.get("attn_pool_dropout", 0.2)

    elif args.data_mode == 'images':
        print("DEBUG: Loading config for E2E IMAGES mode...")
        args.is_end_to_end = True
        args.preload_data = False  # Cannot preload images easily
        # base_vision_model already loaded

        # Load EarlyExtract specific args (if section exists)
        e2e_conf = get_optional_config("e2e_params", conf, default={})
        args.extract_layer = e2e_conf.get("extract_layer", -1)
        args.pooling_strategy = e2e_conf.get("pooling_strategy", 'attn')  # Pooling after base model
        args.freeze_base_model = e2e_conf.get("freeze_base_model", True)

        # Load Head specific args (defines the trainable head added on top)
        head_conf = get_optional_config("head_params", conf, default={})
        # Head features determined by base model output
        args.head_hidden_dim = head_conf.get("hidden_dim", 1024)
        args.head_num_res_blocks = head_conf.get("num_res_blocks", 2)
        args.head_dropout_rate = head_conf.get("dropout_rate", 0.2)
        # Output mode is REQUIRED for the head model
        args.head_output_mode = get_required_config("head_params.output_mode", conf)
        # Load Attn Pool specific args only if needed by the strategy
        if args.pooling_strategy == 'attn':
            attn_conf = get_optional_config("attn_pool_params", conf, default={})
            args.attn_pool_heads = attn_conf.get("attn_pool_heads", 8)
            args.attn_pool_dropout = attn_conf.get("attn_pool_dropout", 0.1)

    else:
        raise ValueError(
            f"Invalid data.mode '{args.data_mode}' in config. Must be 'embeddings', 'features', or 'images'.")

    # --- Load Training Params (Common to all modes) ---
    train_conf = get_required_config("train", conf) # Train section is required
    args.lr = get_optional_config("lr", train_conf, default=1e-4)
    args.batch = get_optional_config("batch", train_conf, default=4)
    args.loss_function = get_optional_config("loss_function", train_conf) # Default set later based on arch
    args.optimizer = get_optional_config("optimizer", train_conf, default='AdamW')
    args.betas = get_optional_config("betas", train_conf)
    args.eps = get_optional_config("eps", train_conf)
    args.weight_decay = get_optional_config("weight_decay", train_conf)
    args.max_train_epochs = get_optional_config("max_train_epochs", train_conf)
    args.max_train_steps = get_optional_config("max_train_steps", train_conf)
    args.precision = get_optional_config("precision", train_conf, default='fp32')
    args.nsave = get_optional_config("nsave", train_conf, default=0)
    args.val_split_count = get_optional_config("val_split_count", conf.get("data",{}), default=0)
    args.seed = get_optional_config("seed", train_conf, default=42)
    args.num_workers = get_optional_config("num_workers", train_conf, default=0)
    args.save_full_model = get_optional_config("save_full_model", train_conf, default=False)
    args.log_every_n = get_optional_config("log_every_n", train_conf, default=50)
    args.validate_every_n = get_optional_config("validate_every_n", train_conf, default=0)
    # Copy ALL OTHER keys from train_conf
    known_train_keys = {'lr', 'batch', 'loss_function', 'optimizer', 'betas', 'eps',
                        'weight_decay', 'max_train_epochs', 'max_train_steps', 'precision',
                        'nsave', 'seed', 'num_workers', 'preload_data', 'save_full_model',
                        'log_every_n', 'validate_every_n'}
    for key, value in train_conf.items():
         if isinstance(value, list): value = tuple(value)
         if key not in known_train_keys and not hasattr(args, key) and value is not None:
              setattr(args, key, value)


    # --- Load Labels/Weights (Only relevant for arch: class) ---
    if args.arch == "class":
        labels_conf = get_optional_config("labels", conf, default={})
        args.labels = None; args.weights = None; args.num_labels = 0
        if labels_conf:
            # Filter for digit keys AFTER loading the whole section
            valid_labels = {str(k): v for k, v in labels_conf.items() if str(k).isdigit()}
            if valid_labels:
                args.labels = {k: v.get("name", k) for k, v in valid_labels.items()}
                try: args.num_labels = max(int(k) for k in args.labels.keys()) + 1
                except ValueError: args.num_labels = 0
                if args.num_labels > 0:
                    weights = [1.0] * args.num_labels
                    for k_str, label_conf in valid_labels.items():
                        try: weights[int(k_str)] = float(label_conf.get("loss", 1.0))
                        except (ValueError, IndexError, TypeError): pass
                    args.weights = weights
        # If no valid labels found or specified, num_labels remains 0 or 1 (handled later)
        if args.num_labels == 0: args.num_labels = 2 # Default to 2 classes if none specified
    else: # arch: score
         args.labels = None; args.weights = None; args.num_labels = 1


    # --- Final Validation / Defaults ---
    if args.arch == "score" and args.loss_function is None: args.loss_function = 'l1'
    if args.arch == "class" and args.loss_function is None: args.loss_function = 'focal'
    if args.max_train_epochs is None and args.max_train_steps is None: args.max_train_steps = 10000 # Default steps


    # Validate output modes were set for relevant modes
    if args.data_mode == 'embeddings' and not hasattr(args, 'output_mode'): raise ValueError("Missing predictor_params.output_mode for embeddings mode.")
    if args.data_mode == 'features' and not hasattr(args, 'head_output_mode'): raise ValueError("Missing head_params.output_mode for features mode.")
    if args.data_mode == 'images' and not hasattr(args, 'head_output_mode'): raise ValueError("Missing head_params.output_mode for images (E2E) mode.")

    return args
# --- End Argument Parsing ---


# --- write_config Function (Needs update to reflect new structure) ---
def write_config(args):
    """Writes the final training configuration based on args Namespace."""
    # Convert Namespace to dict, remove sensitive/unneeded info if any
    conf_to_save = vars(args).copy()
    # Remove things we don't need to save? e.g., config_path itself?
    conf_to_save.pop('config_path', None)
    # Maybe structure it back into nested dicts for readability? (Optional)

    os.makedirs(SAVE_FOLDER, exist_ok=True)
    # Use args.name which should be defined
    config_path_out = f"{SAVE_FOLDER}/{getattr(args, 'name', 'config')}.config.json"
    try:
        with open(config_path_out, "w") as f:
            # Use repr for non-serializable items like functions or types if any crept in
            f.write(json.dumps(conf_to_save, indent=2, default=repr))
        print(f"Saved final effective training config to {config_path_out}")
    except Exception as e:
        print(f"Error saving config file {config_path_out}: {e}")


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


# ================================================
#        Standalone Validation Functions
# ================================================

@torch.no_grad()
def run_validation_embeddings(model, val_loader, criterion, device, scaler):
    """Runs validation loop for embedding-based models (stacked batches)."""
    if val_loader is None: return float('nan')
    model.eval()
    autocast_enabled = scaler is not None and scaler.is_enabled()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    total_loss = 0.0
    total_samples = 0

    val_iterator = tqdm(val_loader, desc="Validation (Embeddings)", leave=False, dynamic_ncols=True)
    for batch_data in val_iterator:
        if batch_data is None or not batch_data: continue
        try:
            emb_input = batch_data.get("emb")
            target_val = batch_data.get("val")
            if emb_input is None or target_val is None: continue

            emb_input = emb_input.to(device)
            target_val = target_val.to(device) # Type adjusted based on loss later
            current_batch_size = emb_input.size(0)

            # Assume model output count is accessible, e.g., model.num_classes or determined by criterion
            num_classes = getattr(model, 'num_classes', 1) # Get from model if possible

            # Prepare target
            if num_classes == 1: target = target_val.to(dtype=torch.float32).view(current_batch_size, -1).squeeze(-1)
            else: target = target_val.to(dtype=torch.long).view(current_batch_size)

            with torch.amp.autocast(device_type=device, enabled=autocast_enabled, dtype=amp_dtype):
                y_pred = model(emb_input) # Expects [B, Emb] -> [B, Classes] or [B]

                y_pred_for_loss = y_pred
                if isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)) and num_classes == 1:
                     if y_pred.ndim > 1 and y_pred.shape[-1] == 1: y_pred_for_loss = y_pred.squeeze(-1)

                y_pred_final = y_pred_for_loss.to(torch.float32)
                target_for_loss = target.to(y_pred_final.device)

                # Calculate loss
                if isinstance(criterion, nn.NLLLoss): loss = criterion(F.log_softmax(y_pred_final, dim=-1), target_for_loss.long())
                elif isinstance(criterion, (nn.CrossEntropyLoss, FocalLoss, GHMC_Loss)): loss = criterion(y_pred_final, target_for_loss.long())
                elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)): loss = criterion(y_pred_final, target_for_loss.float())
                else: loss = torch.tensor(float('nan'), device=device)

            if not math.isnan(loss.item()):
                total_loss += loss.item() * current_batch_size
                total_samples += current_batch_size

            if total_samples > 0: val_iterator.set_postfix({"AvgLoss": f"{(total_loss / total_samples):.4e}"})

        except Exception as e_val:
            print(f"Error during embedding validation step: {e_val}")
            traceback.print_exc(); continue # Skip batch on error
        finally:
             # Minimal cleanup for embedding mode
             try: del emb_input, target_val, target, y_pred, y_pred_for_loss, loss
             except NameError: pass

    val_iterator.close()
    model.train() # Set model back to training mode
    if total_samples == 0: return float('nan')
    avg_loss = total_loss / total_samples
    print(f"Validation (Embeddings) finished. Avg Loss: {avg_loss:.4e} ({total_samples} samples)")
    return avg_loss


@torch.no_grad()
def run_validation_sequences(model, val_loader, criterion, device, scaler):
    """Runs validation loop for precomputed feature sequences (bucketed batches)."""
    if val_loader is None: return float('nan')
    model.eval()
    autocast_enabled = scaler is not None and scaler.is_enabled()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    total_loss = 0.0
    total_samples = 0

    val_iterator = tqdm(val_loader, desc="Validation (Sequences)", leave=False, dynamic_ncols=True)
    for batch_data in val_iterator:
        if batch_data is None or not batch_data: continue
        try:
            # Expect stacked batches from BucketBatchSampler + collate_sequences
            sequence_batch = batch_data.get('sequence')
            label_batch = batch_data.get('label')
            if sequence_batch is None or label_batch is None: continue

            sequence_batch = sequence_batch.to(device) # Should be float16 from dataset
            label_batch = label_batch.to(device) # Should be long from dataset
            current_batch_size = sequence_batch.size(0)

            # Assume model output count is accessible
            num_classes = getattr(model, 'num_classes', 1)

            # Prepare target
            if num_classes == 1: target = label_batch.to(dtype=torch.float32).view(current_batch_size, -1).squeeze(-1)
            else: target = label_batch.to(dtype=torch.long).view(current_batch_size)

            with torch.amp.autocast(device_type=device, enabled=autocast_enabled, dtype=amp_dtype):
                # Pass sequence batch [B, NumPatches, Features] to HeadModel
                y_pred = model(sequence_batch)

                y_pred_for_loss = y_pred
                if isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)) and num_classes == 1:
                    if y_pred.ndim > 1 and y_pred.shape[-1] == 1: y_pred_for_loss = y_pred.squeeze(-1)

                y_pred_final = y_pred_for_loss.to(torch.float32)
                target_for_loss = target.to(y_pred_final.device)

                # Calculate loss
                if isinstance(criterion, nn.NLLLoss): loss = criterion(F.log_softmax(y_pred_final, dim=-1), target_for_loss.long())
                elif isinstance(criterion, (nn.CrossEntropyLoss, FocalLoss, GHMC_Loss)): loss = criterion(y_pred_final, target_for_loss.long())
                elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)): loss = criterion(y_pred_final, target_for_loss.float())
                else: loss = torch.tensor(float('nan'), device=device)

            if not math.isnan(loss.item()):
                total_loss += loss.item() * current_batch_size
                total_samples += current_batch_size

            if total_samples > 0: val_iterator.set_postfix({"AvgLoss": f"{(total_loss / total_samples):.4e}"})

        except Exception as e_val:
            print(f"Error during sequence validation step: {e_val}")
            traceback.print_exc(); continue
        finally:
             try: del sequence_batch, label_batch, target, y_pred, y_pred_for_loss, loss
             except NameError: pass

    val_iterator.close()
    model.train()
    if total_samples == 0: return float('nan')
    avg_loss = total_loss / total_samples
    print(f"Validation (Sequences) finished. Avg Loss: {avg_loss:.4e} ({total_samples} samples)")
    return avg_loss


# --- ModelWrapper Class (Simplified) ---
# Version 3.0.0: Removed validation loop and CSV logging
class ModelWrapper:
    def __init__(self, name, model, optimizer, criterion, scheduler=None, device="cpu",
                 stdout=True, scaler=None, wandb_run=None, num_labels=None): # Remove log_file_path, require num_labels?
        self.name = name
        self.model = model
        # <<< Store essentials needed by training loop and saving >>>
        self.optimizer = optimizer
        self.criterion = criterion # Keep criterion if needed elsewhere? Maybe not.
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        self.wandb_run = wandb_run
        self.stdout = stdout # Keep stdout flag? Maybe for other prints?

        # <<< Store num_labels if provided, useful for consistency >>>
        self.num_labels = num_labels if num_labels is not None else getattr(model, 'num_classes', '?')

        # State tracking
        self.losses = [] # Buffer for recent training losses
        self.current_epoch = 0
        self.current_global_step = 0
        self.best_val_loss = float('inf') # Still track best loss if validation run separately
        self.current_best_val_model_path = None

        self._find_initial_best_model()
        # <<< Removed CSV logger setup >>>
        print(f"ModelWrapper initialized for '{name}'.")


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
        """Add training loss to buffer for averaging."""
        # This buffer is used by log_main to calculate train_loss_avg
        if not math.isnan(loss):
             self.losses.append(loss)
             if len(self.losses) > LOSS_MEMORY:
                  self.losses.pop(0)


    # Version 3.0.0: Removed CSV logging, stdout logging
    def log_main(self, step, train_loss_batch, eval_loss=None): # eval_loss is now optional input
        """Logs metrics to Wandb ONLY. Updates internal best loss."""
        self.update_step(step)
        # Add the latest batch loss (or maybe avg loss over log_every_n steps?)
        self.log_step(train_loss_batch) # Assumes train_loss_batch is avg over log period

        # Calculate long-term average loss from buffer for potential WandB logging
        train_loss_avg = sum(self.losses) / len(self.losses) if self.losses else float('nan')
        lr = float(self.optimizer.param_groups[0]['lr']) if self.optimizer.param_groups else 0.0

        # <<< REMOVED Stdout via tqdm.write() >>>

        # Wandb Logging (Primary Log Target)
        if self.wandb_run:
            log_data = {
                "train/loss_step": train_loss_batch, # Log the loss passed in (avg over log_every_n)
                "train/loss_avg_buffer": train_loss_avg, # Log the longer-term buffer average
                "train/learning_rate": lr
            }
            # Log eval_loss ONLY if a valid value is passed in
            if eval_loss is not None and not math.isnan(eval_loss):
                log_data["eval/loss"] = eval_loss
            try:
                self.wandb_run.log(log_data, step=step)
            except Exception as e_wandb:
                print(f"Warning: Failed to log to WandB at step {step}: {e_wandb}")

        # <<< REMOVED CSV Logging >>>

        # Update internal best loss tracking if eval_loss provided
        if eval_loss is not None and not math.isnan(eval_loss):
            if eval_loss < self.best_val_loss:
                 print(f"  ---> New best validation loss recorded: {eval_loss:.4e} (previous: {self.best_val_loss:.4e})")
                 self.best_val_loss = eval_loss
                 # Saving happens in train loop based on this updated value

    # --- Model Saving ---
    # Version 2.9.1: Optional full model save based on args.save_full_model
    def save_model(self, step=None, epoch=None, suffix="", save_aux=False, args=None):
        """
        Saves model checkpoint and optionally training state.
        Saves full model or only trainable parts based on args.save_full_model.
        Manages the single latest _best_val checkpoint.
        """
        if args is None:
            print("Warning: 'args' object not provided to save_model. Defaulting to saving trainable parts only.")
            should_save_full = False
        else:
            should_save_full = getattr(args, 'save_full_model', False)

        state_dict_to_save = {} # Initialize dictionary to save

        if should_save_full:
            print("DEBUG save_model: Configured to save FULL model state_dict.")
            try:
                state_dict_to_save = self.model.state_dict()
            except Exception as e:
                print(f"Error getting full model state_dict: {e}"); return
        else:
            print("DEBUG save_model: Configured to save TRAINABLE parts only. Manually collecting parameters...")
            # Manual Parameter Collection (Head + Pooler if applicable)
            collected_count = 0
            # Check for standard E2E structure (head + optional pooler)
            is_e2e = getattr(args, 'is_end_to_end', False)
            if is_e2e and hasattr(self.model, 'head') and isinstance(self.model.head, nn.Module):
                print("  - Collecting head parameters/buffers...")
                for name, param in self.model.head.named_parameters():
                    state_dict_to_save[f"head.{name}"] = param.detach().clone().cpu()
                    collected_count += 1
                for name, buf in self.model.head.named_buffers():
                     state_dict_to_save[f"head.{name}"] = buf.detach().clone().cpu()
                     collected_count += 1

                # Collect Pooler state dict (only if strategy is attn and pooler exists)
                pool_strat = getattr(args, 'pooling_strategy', None)
                if pool_strat == 'attn' and hasattr(self.model, 'pooler') and self.model.pooler is not None:
                    print("  - Collecting pooler parameters/buffers...")
                    for name, param in self.model.pooler.named_parameters():
                        state_dict_to_save[f"pooler.{name}"] = param.detach().clone().cpu()
                        collected_count += 1
                    for name, buf in self.model.pooler.named_buffers():
                         state_dict_to_save[f"pooler.{name}"] = buf.detach().clone().cpu()
                         collected_count += 1
            # Handle case where model might BE the head (embedding models)
            elif not is_e2e and isinstance(self.model, nn.Module):
                 print("  - Model seems to be head-only (Embedding Mode). Collecting all its parameters/buffers...")
                 for name, param in self.model.named_parameters():
                     state_dict_to_save[name] = param.detach().clone().cpu()
                     collected_count += 1
                 for name, buf in self.model.named_buffers():
                     state_dict_to_save[name] = buf.detach().clone().cpu()
                     collected_count += 1
            else:
                 print("Warning: Could not determine model structure for partial save.")

            # Fallback if nothing was collected but partial save was intended
            if collected_count == 0 and not should_save_full:
                print("Error: No parameters collected for partial save! Saving full model as fallback.")
                try:
                    state_dict_to_save = self.model.state_dict(); should_save_full = True
                except Exception as e:
                    print(f"Error getting full model state_dict during fallback: {e}"); return

            elif collected_count > 0:
                print(f"  Manually collected {collected_count} parameters/buffers for partial save.")
            # If should_save_full was true from the start, state_dict_to_save already has the full dict

        # Final check if dictionary is populated
        if not state_dict_to_save:
             print("Error: state_dict_to_save is empty or invalid after collection/fallback."); return

        # --- Filenames ---
        current_global_step = step if step is not None else self.current_global_step
        current_epoch_to_save = epoch if epoch is not None else self.current_epoch
        step_str = ""; epoch_str = ""
        if current_global_step is not None:
             if current_global_step >= 1_000_000: step_str = f"_s{round(current_global_step / 1_000_000, 1)}M"
             elif current_global_step >= 1_000: step_str = f"_s{round(current_global_step / 1_000)}K"
             else: step_str = f"_s{current_global_step}"
        if isinstance(epoch, str) and epoch.lower() == "final": epoch_str = "_efinal"
        base_output_name = f"./{SAVE_FOLDER}/{self.name}{step_str}{epoch_str}{suffix}"
        model_output_path = f"{base_output_name}.safetensors"
        # <<< Define ALL aux paths >>>
        optim_output_path = f"{base_output_name}.optim"
        sched_output_path = f"{base_output_name}.sched"
        scaler_output_path = f"{base_output_name}.scaler"
        state_output_path = f"{base_output_name}.state"

        is_best = "_best_val" in suffix

        save_type_str = 'Full Model' if should_save_full else 'Trainable Parts Only'
        print(f"\nSaving checkpoint: {os.path.basename(base_output_name)} ... [{save_type_str}]")
        # Estimate size based on the actual dictionary being saved
        estimated_size_mb = sum(p.numel() * p.element_size() for p in state_dict_to_save.values() if hasattr(p, 'numel')) / (1024 * 1024)
        print(f"DEBUG: Keys in FINAL state_dict: {len(state_dict_to_save)}")
        print(f"DEBUG: Estimated size of FINAL state_dict: {estimated_size_mb:.2f} MB")

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
                            try: os.remove(f_path); removed_count += 1
                            except OSError as e: print(f"  Warning: Could not remove '{os.path.basename(f_path)}': {e}")
                    if removed_count > 0: print(f"  Removed {removed_count} previous best file(s).")
                    self.current_best_val_model_path = None # Clear regardless of success

            # --- Save CURRENT Checkpoint Files ---
            # <<< Save the correct dictionary >>>
            save_file(state_dict_to_save, model_output_path)
            actual_size_mb = os.path.getsize(model_output_path) / (1024 * 1024)
            # <<< Print only ONCE after saving >>>
            print(f"  Model saved successfully. Actual size: {actual_size_mb:.2f} MB")

            if save_aux:
                print("  Saving auxiliary files (optim, sched, scaler, state)...")
                torch.save(self.optimizer.state_dict(), optim_output_path)
                if self.scheduler is not None:
                    try: torch.save(self.scheduler.state_dict(), sched_output_path)
                    except Exception as e_sched: print(f"  Warning: Failed to save scheduler state: {e_sched}")
                if self.scaler is not None and self.scaler.is_enabled():
                    torch.save(self.scaler.state_dict(), scaler_output_path)
                train_state = {
                    'epoch': current_epoch_to_save, 'global_step': current_global_step,
                    'best_val_loss': self.best_val_loss
                }
                torch.save(train_state, state_output_path)
            else:
                # Clean up orphaned aux files for this specific checkpoint name
                # print("  save_aux is False. Skipping save/Ensuring removal of aux files...") # Optional
                for ext in ['.optim', '.sched', '.scaler', '.state']:
                    path_to_check = base_output_name + ext
                    if os.path.exists(path_to_check):
                        try: os.remove(path_to_check) # ; print(f" Removed aux file: {os.path.basename(path_to_check)}") # Optional print
                        except OSError as e: pass # print(f" Warning: Failed to remove aux file {os.path.basename(path_to_check)}: {e}") # Optional print

            # <<< Removed duplicate print >>>

            # Update best model path tracking
            if is_best:
                self.current_best_val_model_path = model_output_path
                print(f"  Marked {os.path.basename(model_output_path)} as new best model path.")

            print(f"  Checkpoint saving process complete for base: {os.path.basename(base_output_name)}")

        except Exception as e:
            print(f"Error saving checkpoint {base_output_name}: {e}")
            import traceback; traceback.print_exc()

    def close(self):
        """Placeholder for closing resources if any."""
        # <<< Removed CSV log closing >>>
        print("ModelWrapper closed.")
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