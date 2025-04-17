# Version: 2.3.0 (Refactored ModelWrapper, Added Loss Arg, Cleaned Saving)
import math
import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from safetensors.torch import save_file, load_file # Added load_file
import shutil # Added for removing old best checkpoints

# --- Constants ---
LOSS_MEMORY = 500 # How many recent steps' training loss to average for display
LOG_EVERY_N = 100 # How often to log metrics and check validation
SAVE_FOLDER = "models" # Folder to save models and configs

# --- Focal Loss Definition ---
# (Keep the FocalLoss class definition here as before)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean': return torch.mean(focal_loss)
        elif self.reduction == 'sum': return torch.sum(focal_loss)
        else: return focal_loss
# --- End Focal Loss ---


# --- Parameter Dictionary ---
def get_embed_params(ver):
    """Returns features and hidden dim based on embedding version string."""
    # Add new embedding types here as needed
    embed_configs = {
        "CLIP": {"features": 768, "hidden": 1024},
        "siglip2_so400m_patch16_512": {"features": 1152, "hidden": 1280},
        "siglip2_so400m_patch16_512_AvgCrop": {"features": 1152, "hidden": 1280},
        "siglip2_so400m_patch16_512_FitPad": {"features": 1152, "hidden": 1280},
        "siglip2_so400m_patch16_512_CenterCrop": {"features": 1152, "hidden": 1280},
        "siglip2_so400m_patch16_naflex_NaflexResize": {"features": 1152, "hidden": 1280},
        "siglip2_so400m_patch16_naflex_AvgCrop": {"features": 1152, "hidden": 1280},
        "siglip2_so400m_patch16_naflex_FitPad": {"features": 1152, "hidden": 1280},
        "siglip2_so400m_patch16_naflex_CenterCrop": {"features": 1152, "hidden": 1280},
        "META": {"features": 1024, "hidden": 1280},
    }
    if ver in embed_configs:
        return embed_configs[ver]
    else:
        # Attempt to infer from HF model name if ver contains '/'? More complex.
        # For now, raise error for unknown explicitly defined versions.
        raise ValueError(f"Unknown embedding version '{ver}'. Please add it to get_embed_params in utils.py.")
# --- End Parameter Dictionary ---

# --- Argument Parsing and Config Loading ---
# Version 2.3.1: Added smarter base_vision_model lookup
def parse_and_load_args():
    """Parses command line args and merges with YAML config."""
    parser = argparse.ArgumentParser(description="Train aesthetic predictor/classifier")
    # Basic args
    parser.add_argument("--config", required=True, help="Training config YAML file")
    parser.add_argument('--resume', help="Checkpoint (.safetensors model file) to resume from")
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'], help='Training precision (fp32, fp16, bf16).')
    parser.add_argument("--nsave", type=int, default=10000, help="Save model every N steps (0 to disable periodic saves).")
    parser.add_argument("--val_split_count", type=int, default=0, help="Samples per class for validation (0 to disable).")
    parser.add_argument("--seed", type=int, default=218, help="Random seed for dataset splitting.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--preload_data", action=argparse.BooleanOptionalAction, default=True, help="Preload dataset embeddings to RAM.")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory for datasets.")
    parser.add_argument("--wandb_project", type=str, default="city-classifiers", help="Weights & Biases project name.")

    # Optimizer Choice
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['AdamW', 'FMARSCropV3ExMachina', 'ADOPT', 'ADOPTScheduleFree', 'ADOPTAOScheduleFree'],
                        help='Optimizer to use.')
    # Loss Function Choice
    parser.add_argument('--loss_function', type=str, default=None,
                        choices=['crossentropy', 'focal', 'l1', 'mse'],
                        help="Loss function ('crossentropy'/'focal' for class, 'l1'/'mse' for score). Default based on arch.")

    # Optimizer Hyperparameters (allow override via command line, primarily set in YAML)
    parser.add_argument('--lr', type=float, default=None, help="Learning rate (override YAML).")
    parser.add_argument('--betas', type=float, nargs='+', default=None, help="Optimizer betas (override YAML).")
    parser.add_argument('--eps', type=float, default=None, help='Optimizer epsilon (override YAML).')
    parser.add_argument('--weight_decay', type=float, default=None, help='Optimizer weight decay (override YAML).')

    # Parse known args first
    args = parser.parse_args()

    # Load YAML config
    if not os.path.isfile(args.config):
        parser.error(f"Can't find config file '{args.config}'")
    with open(args.config) as f:
        conf = yaml.safe_load(f)

    # --- Merge YAML config into args ---
    # Model params
    model_conf = conf.get("model", {})
    args.base = model_conf.get("base", "unknown_model")
    args.rev = model_conf.get("rev", "v0.0")
    args.arch = model_conf.get("arch", "class")
    args.embed_ver = model_conf.get("embed_ver", "CLIP")
    args.name = f"{args.base}-{args.rev}"
    args.num_attn_heads = model_conf.get("num_attn_heads", 8)
    args.attn_dropout = model_conf.get("attn_dropout", 0.1)

    # --- Vision model lookup ---
    # Try getting base_vision_model explicitly from YAML first
    args.base_vision_model = model_conf.get("base_vision_model")

    # If not found in YAML, try inferring from embed_ver using the LONGEST matching key
    if not args.base_vision_model:
        # Dictionary mapping BASE embed versions (or common prefixes) to HF model names
        # Keys should ideally be distinct prefixes
        default_vision_models = {
            "CLIP": "openai/clip-vit-large-patch14-336",
            "siglip2_so400m_patch16_512": "google/siglip2-so400m-patch16-512",
            # Ensure this key is distinct enough or checked correctly
            "siglip-naflex": "google/siglip2-so400m-patch16-naflex",
            "META": "facebook/metaclip-h14-fullcc2.5b"
            # Add other BASE identifiers here
        }

        found_model = None
        l_embed_ver = args.embed_ver.lower() if args.embed_ver else ""
        best_match_key = None

        # Find the longest matching prefix key (This logic should work with better keys)
        for key in default_vision_models.keys():
            if l_embed_ver.startswith(key.lower()):
                if best_match_key is None or len(key) > len(best_match_key):
                    best_match_key = key

        if best_match_key:
            found_model = default_vision_models[best_match_key]
            print(
                f"DEBUG: Inferred base_vision_model '{found_model}' from embed_ver '{args.embed_ver}' using longest matching base key '{best_match_key}'")
        else:
            print(
                f"Warning: Could not automatically determine base_vision_model for embed_ver '{args.embed_ver}'. Please specify it in the YAML config.")

        args.base_vision_model = found_model

        # Add a warning if we still couldn't find it
        if not args.base_vision_model:
            print(f"Warning: Could not automatically determine base_vision_model for embed_ver '{args.embed_ver}'. Please specify it in the YAML config.")

    # Training params (YAML overrides defaults, command line overrides YAML if provided)
    train_conf = conf.get("train", {})
    args.lr = args.lr if args.lr is not None else float(train_conf.get("lr", 1e-4))
    args.steps = int(train_conf.get("steps", 100000))
    args.batch = int(train_conf.get("batch", 4))
    args.loss_function = args.loss_function if args.loss_function is not None else train_conf.get("loss_function")
    args.optimizer = args.optimizer if args.optimizer != parser.get_default("optimizer") else train_conf.get("optimizer", args.optimizer)
    args.betas = args.betas if args.betas is not None else tuple(map(float, train_conf.get("betas", []))) or None
    args.eps = args.eps if args.eps is not None else train_conf.get("eps")
    args.weight_decay = args.weight_decay if args.weight_decay is not None else train_conf.get("weight_decay")
    args.cosine = bool(train_conf.get("cosine", True))
    args.warmup_steps = int(train_conf.get("warmup_steps", 0))

    # Populate other optimizer hyperparams from YAML if not set via command line
    optimizer_specific_args = ['gamma', 'r_sf', 'wlpow_sf', 'state_precision', 'weight_decouple', 'stable_weight_decay', 'adaptive_clip', 'adaptive_clip_eps', 'adaptive_clip_type', 'debias_beta2', 'use_beta2_warmup', 'beta2_warmup_initial', 'beta2_warmup_steps', 'mars_gamma', 'use_muon_pp', 'fisher', 'update_strategy', 'stable_update', 'atan2_denom', 'use_orthograd', 'use_spam_clipping', 'spam_clipping_threshold', 'spam_clipping_start_step', 'spam_clipping_type']
    for arg_name in optimizer_specific_args:
         if not hasattr(args, arg_name) or getattr(args, arg_name) is None:
              setattr(args, arg_name, train_conf.get(arg_name))

    # Set coded defaults for optimizer hyperparams if still None
    if args.betas is None: args.betas = (0.9, 0.999)
    if args.eps is None: args.eps = 1e-8 if args.optimizer.lower() == 'adamw' else 1e-6
    if args.weight_decay is None: args.weight_decay = 0.0
    if args.gamma is None and 'mars' in args.optimizer.lower(): args.gamma = 0.005
    if args.r_sf is None and 'schedulefree' in args.optimizer.lower(): args.r_sf = 0.0
    if args.wlpow_sf is None and 'schedulefree' in args.optimizer.lower(): args.wlpow_sf = 2.0

    # Labels/Weights for classifier
    labels_conf = conf.get("labels", {})
    if args.arch == "class":
        if labels_conf:
            args.labels = {str(k): v.get("name", str(k)) for k, v in labels_conf.items()}
            try: args.num_labels = max(int(k) for k in labels_conf.keys()) + 1
            except: args.num_labels = 0
            weights = [1.0] * args.num_labels
            for k_str, label_conf in labels_conf.items():
                try: weights[int(k_str)] = float(label_conf.get("loss", 1.0))
                except: pass
            args.weights = weights
        else:
             args.num_labels = model_conf.get("outputs", 2)
             args.labels = None; args.weights = None
    else: # Score
        args.num_labels = 1; args.labels = None; args.weights = None

    # Validate inputs
    assert args.arch in ["score", "class"], f"Unknown arch '{args.arch}'"
    if args.arch == "score" and args.loss_function not in [None, 'l1', 'mse']:
         print(f"Warning: Loss '{args.loss_function}' specified for score arch. Using default L1.")
         args.loss_function = 'l1'
    if args.arch == "class" and args.loss_function not in [None, 'crossentropy', 'focal']:
         print(f"Warning: Loss '{args.loss_function}' specified for class arch. Using default CrossEntropy.")
         args.loss_function = 'crossentropy'

    return args

def write_config(args):
    """Writes the final training configuration to a JSON file."""
    # Determine embed params based on the version string
    try: embed_params = get_embed_params(args.embed_ver)
    except ValueError as e: print(f"Error: {e}"); embed_params = {"features": 0, "hidden": 0} # Handle error case

    # Consolidate config dictionary
    conf = {
        "model": {
            "base": args.base,
            "rev": args.rev,
            "arch": args.arch,
            "embed_ver": args.embed_ver,
            "base_vision_model": args.base_vision_model,
            "num_attn_heads": getattr(args, 'num_attn_heads', None), # Store attention params if used
            "attn_dropout": getattr(args, 'attn_dropout', None),
        },
        "train": {
            "lr": args.lr, "steps": args.steps, "batch": args.batch,
            "optimizer": args.optimizer, "loss_function": args.loss_function,
            "precision": args.precision, "val_split_count": args.val_split_count,
            # Store relevant optimizer hyperparams
            "betas": args.betas, "eps": args.eps, "weight_decay": args.weight_decay,
            "cosine": args.cosine, "warmup_steps": args.warmup_steps,
            # Add others as needed from args, checking if they exist first
            **{k: v for k, v in vars(args).items() if k in ['gamma', 'r_sf', 'wlpow_sf', 'state_precision', 'weight_decouple', 'stable_weight_decay', 'adaptive_clip', 'adaptive_clip_eps', 'adaptive_clip_type', 'debias_beta2', 'use_beta2_warmup', 'beta2_warmup_initial', 'beta2_warmup_steps', 'mars_gamma', 'use_muon_pp', 'fisher', 'update_strategy', 'stable_update', 'atan2_denom', 'use_orthograd', 'use_spam_clipping', 'spam_clipping_threshold', 'spam_clipping_start_step', 'spam_clipping_type'] and v is not None}
        },
        "labels": args.labels if args.arch == "class" else None,
        "model_params": { # Params passed to PredictorModel constructor
            "features": embed_params["features"],
            "hidden": embed_params["hidden"],
            "outputs": args.num_labels
        }
    }
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    config_path = f"{SAVE_FOLDER}/{args.name}.config.json"
    try:
        with open(config_path, "w") as f:
            f.write(json.dumps(conf, indent=2, default=lambda x: repr(x))) # Use repr for non-serializable
        print(f"Saved training config to {config_path}")
    except Exception as e:
        print(f"Error saving config file {config_path}: {e}")

# --- End Argument Parsing and Config Loading ---


# v2.5.0: Correctly handles single latest _sXXX_best_val file, optional aux saving.
class ModelWrapper:
    def __init__(self, name, model, optimizer, criterion, scheduler=None, device="cpu",
                 stdout=True, scaler=None, wandb_run=None, num_labels=1):
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
        self.current_step = 0
        self.best_val_loss = float('inf')
        # Store the path to the *last known* best checkpoint file ending with _best_val.safetensors
        self.current_best_val_model_path = None
        # Try to find existing best model on init/resume? More complex, skip for now.

        os.makedirs(SAVE_FOLDER, exist_ok=True)
        self.log_file_path = f"{SAVE_FOLDER}/{self.name}.csv"
        file_mode = "a" if os.path.exists(self.log_file_path) else "w"
        try:
            self.csvlog = open(self.log_file_path, file_mode)
            if file_mode == "w":
                self.csvlog.write("step,train_loss_avg,eval_loss,learning_rate\n")
                self.csvlog.flush() # Ensure header is written immediately
        except IOError as e:
            print(f"Warning: Could not open CSV log file {self.log_file_path}: {e}")
            self.csvlog = None
        self.stdout = stdout
        print(f"ModelWrapper initialized. Logging to {self.log_file_path} (mode: {file_mode})")

    def update_step(self, step):
        """Update the internal step counter."""
        self.current_step = step

    def get_current_step(self):
        """Get the current step counter."""
        return self.current_step

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
                with torch.amp.autocast(device_type=self.device, enabled=autocast_enabled, dtype=amp_dtype):
                    y_pred = self.model(emb) # Model forward pass

                    # Prepare prediction shape for loss function
                    if self.num_labels == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
                         y_pred_for_loss = y_pred.squeeze(1) # [B, 1] -> [B] for L1/MSE
                    else:
                         y_pred_for_loss = y_pred # Shape [B, C] for CE/Focal

                    # Ensure prediction is Float32 for stable loss calculation
                    y_pred_final = y_pred_for_loss.to(torch.float32)

                    # Calculate loss, ensuring target dtype matches loss function expectation
                    if isinstance(self.criterion, (torch.nn.CrossEntropyLoss, FocalLoss)):
                        loss = self.criterion(y_pred_final, val) # val is Long
                    else: # L1/MSE etc.
                        loss = self.criterion(y_pred_final, val.to(y_pred_final.dtype)) # val needs to be Float
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
    # v2.5.1: Final corrected saving logic
    def save_model(self, step=None, epoch=None, suffix="", save_aux=False):
        """
        Saves model checkpoint. If suffix contains '_best_val', it manages the single
        latest best checkpoint with that naming convention.
        Auxiliary files (optim, etc.) are only saved if save_aux is True.
        """
        current_step_num = step if step is not None else self.current_step

        # --- Determine Filename Components ---
        step_str = ""
        is_best = "_best_val" in suffix # Check if this save is intended as a best model save

        if epoch is None: # Use step count for filename base
            if current_step_num >= 1_000_000: step_str = f"_s{round(current_step_num / 1_000_000, 1)}M"
            elif current_step_num >= 1_000: step_str = f"_s{round(current_step_num / 1_000)}K"
            else: step_str = f"_s{current_step_num}"
        else: # Use epoch string if provided (e.g., "final")
            step_str = f"_e{epoch}"

        # Construct the full path for the file(s) being saved in THIS call
        # Includes the step and the full suffix (which might contain _best_val)
        base_output_name = f"./{SAVE_FOLDER}/{self.name}{step_str}{suffix}"
        model_output_path = f"{base_output_name}.safetensors"
        # Define aux paths even if not saved, for potential deletion
        optim_output_path = f"{base_output_name}.optim"
        sched_output_path = f"{base_output_name}.sched"
        scaler_output_path = f"{base_output_name}.scaler"
        # --- End Filename Components ---

        print(f"\nSaving checkpoint: {base_output_name} (Step: {current_step_num})")
        try:
            # --- Delete OLD Best Checkpoint FIRST (if this is a new best) ---
            if is_best:
                # Check if we have a record of the PREVIOUS best model's path
                if self.current_best_val_model_path and os.path.exists(self.current_best_val_model_path):
                     # Check if the file we are about to save is DIFFERENT from the stored previous best
                     if os.path.abspath(model_output_path) != os.path.abspath(self.current_best_val_model_path):
                          try:
                               print(f"  Removing previous best model checkpoint: {os.path.basename(self.current_best_val_model_path)}")
                               os.remove(self.current_best_val_model_path)
                               # Remove corresponding aux files ONLY if save_aux was likely True when THEY were saved
                               # This is tricky. Simpler: don't save/delete aux for _best_val saves at all.
                          except OSError as e:
                               print(f"  Warning: Could not remove previous best checkpoint file: {e}")
                     else:
                          # If the filename is the same (e.g. re-saving best at same step), don't delete.
                           print(f"  New best model has same name as previous, not removing.")
            # --- End Delete OLD ---

            # --- Save CURRENT Checkpoint Files ---
            # Save Model (safetensors)
            save_file(self.model.state_dict(), model_output_path)

            # Optionally Save Aux Files
            if save_aux:
                print("  Saving auxiliary files (optim, sched, scaler)...")
                torch.save(self.optimizer.state_dict(), optim_output_path)
                if self.scheduler is not None: torch.save(self.scheduler.state_dict(), sched_output_path)
                if self.scaler is not None and self.scaler.is_enabled(): torch.save(self.scaler.state_dict(), scaler_output_path)
            # --- End Save CURRENT ---

            print(f"Checkpoint base ({os.path.basename(model_output_path)}) saved successfully.")

            # --- Update Path if This Was the Best Model ---
            if is_best:
                self.current_best_val_model_path = model_output_path # Store the path we just saved
                print(f"  Marked {os.path.basename(model_output_path)} as new best model path.")
            # --- End Update Path ---

        except Exception as e:
            print(f"Error saving checkpoint {base_output_name}: {e}")
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