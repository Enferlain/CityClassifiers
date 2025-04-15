# Version: 2.2.0
# Desc: Integrate wandb logging and more args
import math
import os
import json
import yaml
import torch
import argparse  # Make sure argparse is imported
from tqdm import tqdm
from safetensors.torch import save_file

# --- Keep existing constants ---
LOSS_MEMORY = 500
LOG_EVERY_N = 100
SAVE_FOLDER = "models"


# --- Keep existing get_embed_params ---
# (This might become less relevant if using models that take images directly,
# but keep it for now for the current model structure)
def get_embed_params(ver):
    if ver == "CLIP":
        # OpenAI CLIP-L/14 @ 336px
        return {"features": 768, "hidden": 1024}
    elif ver == "CLIP-Anatomy": # Keep this if still needed for old data
         # Same as CLIP for now unless specified otherwise
         return {"features": 768, "hidden": 1024}
    elif ver == "SIGLIP2-SO400M-512": # <<< New Version
        # google/siglip2-so400m-patch16-512
        return {"features": 1152, "hidden": 1280} # Use 1280 hidden? Or keep 1024? Let's try 1280
    elif ver == "META":
        # open_clip
        #  metaclip_fullcc | ViT-H-14-quickgelu
        print("META ver. was only meant for testing!")
        return {"features": 1024, "hidden": 1280}
    else:
        raise ValueError(f"Unknown model '{ver}'")


# --- Modify parse_args ---
# v2.1.0: Added val_split_count
def parse_args():
    parser = argparse.ArgumentParser(description="Train aesthetic predictor/classifier")
    parser.add_argument("--config", required=True, help="Training config YAML file")
    parser.add_argument('--resume', help="Checkpoint (.safetensors model file) to resume from")
    parser.add_argument('--images', action=argparse.BooleanOptionalAction, default=False,
                        help="Live process images instead of using pre-computed embeddings")
    parser.add_argument("--nsave", type=int, default=10000,
                        help="Save model every N steps (set to 0 or negative to disable)")
    # --- NEW ARG for Validation Split ---
    parser.add_argument("--val_split_count", type=int, default=0,
                        help="Number of samples *per class* to hold out for validation (0 to disable)")
    # --- END NEW ARG ---

    # --- Optimizer Args (Keep as is) ---
    parser.add_argument('--optimizer', type=str, default='AdamW', # Keep default here
                        choices=['AdamW', 'FMARSCropV3ExMachina', 'ADOPT',
                                 'ADOPTScheduleFree', 'ADOPTAOScheduleFree'],
                        help='Optimizer to use.')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'],  # ...
                        help='Training precision (fp32, fp16, bf16).')
    # ... (rest of optimizer args) ...
    parser.add_argument('--betas', type=float, nargs='+', default=None,  # ...
                        help='Optimizer beta parameters (e.g., 0.9 0.999 or 0.9 0.999 0.99)')
    parser.add_argument('--eps', type=float, default=None, help='Optimizer epsilon term.')  # ...
    parser.add_argument('--weight_decay', type=float, default=None, help='Optimizer weight decay.')  # ...
    parser.add_argument('--gamma', type=float, default=None,
                        help='Gamma for MARS correction (FMARSCrop/ADOPTMARS).')  # ...
    parser.add_argument('--r_sf', type=float, default=None,
                        help='ScheduleFree r parameter (polynomial weight power).')  # ...
    parser.add_argument('--wlpow_sf', type=float, default=None, help='ScheduleFree weight_lr_power parameter.')  # ...
    parser.add_argument('--state_precision', type=str, default='parameter',  # ...
                        choices=['parameter', 'q8bit', 'q4bit', 'qfp8'],
                        help='Precision for optimizer state (ADOPTAOScheduleFree).')

    args = parser.parse_args()
    if not os.path.isfile(args.config):
        parser.error(f"Can't find config file '{args.config}'")

    args = get_training_args(args, parser.get_default("optimizer"))

    # --- Set defaults for optimizer params if not provided by YAML or command line ---
    # We set defaults here *after* loading YAML so command line overrides YAML,
    # and YAML overrides these defaults.

    # General defaults
    if args.betas is None:
        # Default depends on optimizer, handle this later or set a common default
        args.betas = (0.9, 0.999)  # AdamW default, adjust if needed
        if args.optimizer.lower() == 'fmarscropv3exmachina':
            args.betas = (0.99, 0.9999, 0.999)  # FMARSCrop default
        elif args.optimizer.lower() in ['adoptschedulefree', 'adoptaoschedulefree', 'adoptmarsschedulefree']:
            args.betas = (0.9, 0.9999)  # ADOPT default
    if args.eps is None:
        args.eps = 1e-8 if args.optimizer.lower() == 'adamw' else 1e-6  # Different defaults common
    if args.weight_decay is None:
        args.weight_decay = 0.0

    # FMARSCrop / ADOPTMARS defaults
    if args.gamma is None and 'mars' in args.optimizer.lower():
        args.gamma = 0.005  # Example default for FMARSCropV3ExMachina

    # ScheduleFree defaults
    if args.r_sf is None and 'schedulefree' in args.optimizer.lower():
        args.r_sf = 0.0
    if args.wlpow_sf is None and 'schedulefree' in args.optimizer.lower():
        args.wlpow_sf = 2.0

    # --- End Defaults ---

    return args

# --- Modify get_training_args ---
# Version 2.1.2: Actually load optimizer name from YAML config
def get_training_args(args, default_optimizer_name):
    """Loads YAML config and merges with argparse args."""
    with open(args.config) as f:
        conf = yaml.safe_load(f)

    # --- Load Training Params from YAML ---
    train_conf = conf.get("train", {})
    args.lr = float(train_conf.get("lr", getattr(args, 'lr', 1e-4)))  # Use getattr for robustness
    args.steps = int(train_conf.get("steps", getattr(args, 'steps', 100000)))
    args.batch = int(train_conf.get("batch", getattr(args, 'batch', 1)))
    # args.val_split_count is usually command-line, but could be added here if needed

    # --- Updated logic to load optimizer name ---
    # If the current args.optimizer is still the default one (meaning it wasn't set via command line),
    # try to load it from the YAML file.
    if args.optimizer == default_optimizer_name:
         # Use YAML value if present, otherwise keep the default from args
         args.optimizer = train_conf.get("optimizer", args.optimizer)

    if args.betas is None: args.betas = tuple(map(float, train_conf.get("betas", []))) or None
    if args.eps is None: args.eps = train_conf.get("eps", None)
    if args.weight_decay is None: args.weight_decay = train_conf.get("weight_decay", None)
    if args.gamma is None: args.gamma = train_conf.get("gamma", None)
    if args.r_sf is None: args.r_sf = train_conf.get("r_sf", None)
    if args.wlpow_sf is None: args.wlpow_sf = train_conf.get("wlpow_sf", None)
    args.cosine = train_conf.get("cosine", getattr(args, 'cosine', True))
    args.warmup_steps = int(train_conf.get("warmup_steps", getattr(args, 'warmup_steps', 5000)))

    # --- Model Params ---
    assert "model" in conf.keys(), "Model config not optional!"
    model_conf = conf["model"] # Get the whole model dict
    args.base = model_conf.get("base", "unknown")
    args.rev = model_conf.get("rev", "v0.0")
    args.arch = model_conf.get("arch", None)
    # Use 'embed_ver' for the type of embedding (CLIP, SIGLIP2...), keep 'clip' if needed for backward compat?
    # Let's rename 'clip' -> 'embed_ver' conceptually
    args.embed_ver = model_conf.get("embed_ver", model_conf.get("clip", "CLIP")) # Read 'embed_ver', fallback to 'clip', then 'CLIP'
    args.name = f"{args.base}-{args.rev}"

    # --- ADD: Store the actual vision model name ---
    # Default mapping from embed_ver to HF model name
    default_vision_models = {
        "CLIP": "openai/clip-vit-large-patch14-336",
        "CLIP-Anatomy": "openai/clip-vit-large-patch14-336", # Assuming same base
        "SIGLIP2-SO400M-512": "google/siglip2-so400m-patch16-512",
        "META": "openai/clip-vit-large-patch14-336" # Default if META specified? Needs clarification
    }
    # Allow overriding in config: model: { embed_ver: ..., base_vision_model: ... }
    args.base_vision_model = model_conf.get("base_vision_model", default_vision_models.get(args.embed_ver, None))
    if args.base_vision_model is None:
         print(f"Warning: Could not determine base_vision_model for embed_ver '{args.embed_ver}'. Inference scripts might fail.")

    assert args.arch in ["score", "class"], f"Unknown arch '{args.arch}'"
    # v2.2.1: Use args.embed_ver in assertion check
    allowed_versions = list(default_vision_models.keys()) # Get allowed from our map
    assert args.embed_ver in allowed_versions, f"Unknown embed version '{args.embed_ver}'" # <<< Use args.embed_ver

    # --- Load Labels/Weights from YAML ---
    labels = conf.get("labels", {})
    if args.arch == "class":
        if labels:
            args.labels = {str(k): v.get("name", str(k)) for k, v in labels.items()}
            try: args.num_labels = max(int(k) for k in labels.keys()) + 1
            except: args.num_labels = 0
            weights = [1.0] * args.num_labels
            for k_str, label_conf in labels.items():
                try:
                    k = int(k_str); weights[k] = float(label_conf.get("loss", 1.0))
                except: pass # Ignore errors
            args.weights = weights
        else: # Need num_labels even if no labels section (e.g. if dataset handles it)
             args.num_labels = model_conf.get("outputs", 2) # Default to 2 outputs for class if not specified
             args.labels = None
             args.weights = None
    else: # Score
        args.num_labels = 1
        args.labels = None
        args.weights = None

    return args

def write_config(args):
    # Determine embed params based on the version string
    embed_params = get_embed_params(args.embed_ver)

    conf = {
        "name": args.base,
        "rev": args.rev,
        "arch": args.arch,
        "labels": args.labels,
        "embed_ver": args.embed_ver,             # <<< Store embedding version used
        "base_vision_model": args.base_vision_model, # <<< Store base vision model used
        "train_args": {
            # ... (store all relevant train args as before) ...
            "lr": args.lr, "steps": args.steps, "batch": args.batch, "cosine": args.cosine, "warmup_steps": args.warmup_steps,
            "optimizer": args.optimizer, "precision": args.precision, "val_split_count": args.val_split_count, "betas": args.betas,
            "eps": args.eps, "weight_decay": args.weight_decay, "gamma": args.gamma, "r_sf": args.r_sf, "wlpow_sf": args.wlpow_sf,
            "state_precision": args.state_precision
        },
        "model_params": {
            "features": embed_params["features"], # <<< Store correct feature dim
            "hidden": embed_params["hidden"],     # <<< Store hidden dim used
            "outputs": args.num_labels
        }
    }
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    config_path = f"{SAVE_FOLDER}/{args.name}.config.json"
    try:
        with open(config_path, "w") as f:
            f.write(json.dumps(conf, indent=2))
        print(f"Saved training config to {config_path}")
    except Exception as e:
        print(f"Error saving config file {config_path}: {e}")


# --- Updated ModelWrapper Class ---
# v2.1.0: Reworked evaluation logic for validation loader
class ModelWrapper:
    # Removed dataset from init args, removed eval_src/eval_dst
    def __init__(self, name, model, optimizer, criterion, scheduler=None, device="cpu",
                 stdout=True, scaler=None, wandb_run=None, num_labels=1):
        self.name = name
        self.device = device
        self.losses = []  # Stores recent training losses for averaging

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scaler = scaler
        self.wandb_run = wandb_run
        self.num_labels = num_labels  # Store num_labels if needed for logging predictions

        os.makedirs(SAVE_FOLDER, exist_ok=True)
        # --- Log file handling ---
        self.log_file_path = f"{SAVE_FOLDER}/{self.name}.csv"
        # Check if resuming only if log file exists (less reliable than checking optim state)
        file_mode = "a" if os.path.exists(self.log_file_path) else "w"
        try:
            self.csvlog = open(self.log_file_path, file_mode)
            if file_mode == "w":
                self.csvlog.write("step,train_loss_avg,eval_loss,learning_rate\n")
        except IOError as e:
            print(f"Warning: Could not open CSV log file {self.log_file_path}: {e}")
            self.csvlog = None
        self.stdout = stdout
        print(f"ModelWrapper initialized. Logging to {self.log_file_path} (mode: {file_mode})")

    # log_step should also probably receive the step, even if not used directly now
    def log_step(self, loss, step): # Added step back
        if not math.isnan(loss):
             self.losses.append(loss)
             if len(self.losses) > LOSS_MEMORY: # Use LOSS_MEMORY directly now
                  self.losses.pop(0) # Remove oldest element efficiently

    # New method for evaluation using a DataLoader
    def evaluate_on_validation_set(self, val_loader):
        """Performs evaluation on the provided validation DataLoader."""
        if val_loader is None:
            return float('nan')

        self.model.eval()  # Set model to evaluation mode
        # --- Set Optimizer Eval Mode (if applicable) ---
        original_optimizer_mode_is_training = False
        needs_optim_switch = (hasattr(self.optimizer, 'eval') and callable(self.optimizer.eval) and
                              hasattr(self.optimizer, 'train') and callable(self.optimizer.train) and
                              hasattr(self.optimizer, 'state') and any(
                    s for s in self.optimizer.state.values()))  # Check if state exists
        if needs_optim_switch:
            if hasattr(self.optimizer, 'train_mode'):
                original_optimizer_mode_is_training = self.optimizer.train_mode
            else:  # Heuristic if train_mode attribute doesn't exist
                # Check if any param group has a non-zero step if available
                if hasattr(self.optimizer, 'param_groups') and any(
                        pg.get('step', 0) > 0 for pg in self.optimizer.param_groups):
                    original_optimizer_mode_is_training = True  # Assume training if steps > 0
            if original_optimizer_mode_is_training:
                try:
                    self.optimizer.eval()
                except Exception as e:
                    print(f"Warning: Error calling optimizer.eval(): {e}")
                    needs_optim_switch = False  # Disable switch if call fails

        total_loss = 0.0
        total_samples = 0
        # Determine autocast dtype based on scaler state (similar to eval_model logic)
        autocast_enabled = self.scaler is not None and self.scaler.is_enabled()
        if autocast_enabled:
            if hasattr(self.scaler, 'get_amp_dtype'):  # More robust way if scaler provides it
                current_amp_dtype = self.scaler.get_amp_dtype()
            # Fallback heuristic
            elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                current_amp_dtype = torch.bfloat16
            else:
                current_amp_dtype = torch.float16
        else:
            current_amp_dtype = torch.float32

        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue  # Skip None batches from collate_ignore_none

                # Ensure batch items are on the correct device and correct type
                emb = batch.get("emb").to(self.device)
                val = batch.get("val")
                if val is None: continue  # Skip if target is missing

                # Move target 'val' to device and ensure correct dtype
                if self.criterion.__class__.__name__ == 'L1Loss':  # Score mode
                    val = val.to(self.device, dtype=torch.float32)
                elif self.criterion.__class__.__name__ == 'CrossEntropyLoss':  # Class mode
                    # Check if val is already one-hot float or class indices long
                    if val.dtype == torch.float and val.ndim == 2:  # Assume one-hot float
                        val = val.to(self.device, dtype=torch.float32)
                    elif val.dtype == torch.long and val.ndim == 1:  # Class indices
                        val = val.to(self.device, dtype=torch.long)
                    elif val.ndim == 2 and val.shape[1] == 1:  # Indices with extra dim
                        val = val.squeeze(-1).to(self.device, dtype=torch.long)
                    else:  # Fallback/error case? Assume class indices needed
                        print(
                            f"Warning: Unexpected validation target shape/type: {val.shape}, {val.dtype}. Attempting Long conversion.")
                        try:
                            val = val.squeeze().to(self.device, dtype=torch.long)
                        except Exception as e:
                            print(f"  Conversion failed: {e}. Skipping batch.")
                            continue

                batch_size = emb.size(0)

                with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=current_amp_dtype):
                    y_pred = self.model(emb)
                    loss = self.criterion(y_pred.to(torch.float32), val)  # Ensure compatible types for loss

                if not math.isnan(loss.item()):
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                else:
                    print("Warning: NaN encountered in validation loss calculation. Skipping batch contribution.")

        # --- Restore Modes ---
        self.model.train()  # Set model back to training mode
        if needs_optim_switch and original_optimizer_mode_is_training:
            try:
                self.optimizer.train()
            except Exception as e:
                print(f"Warning: Error calling optimizer.train(): {e}")

        if total_samples == 0:
            return float('nan')  # Avoid division by zero if no valid samples seen
        return total_loss / total_samples

    def log_main(self, step, train_loss_batch, eval_loss): # Renamed train_loss for clarity
        lr = float(self.optimizer.param_groups[0]['lr']) if self.optimizer.param_groups else 0.0

        # --- Calculate moving average of training loss ---
        # Ensure self.losses is not empty before calculating average
        if self.losses: # Check if the list has items
             # Calculate average directly from the current buffer
             train_loss_avg = sum(self.losses) / len(self.losses)
        else:
             # If buffer is empty (e.g., first log step), use current batch loss or NaN
             train_loss_avg = train_loss_batch # Or use float('nan') if you prefer N/A on first step
             if math.isnan(train_loss_avg): train_loss_avg = float('nan') # Ensure NaN propagates

        # --- Stdout Logging ---
        if self.stdout:
            eval_loss_str = f"{eval_loss:.4e}" if not math.isnan(eval_loss) else "N/A"
            train_avg_str = f"{train_loss_avg:.4e}" if not math.isnan(train_loss_avg) else "N/A"
            # Use train_loss_batch for instant loss, train_loss_avg for average
            # Let's keep showing the average as Loss(avg)
            tqdm.write(f"{str(step):<10} Loss(avg): {train_avg_str} | Eval Loss: {eval_loss_str} | LR: {lr:.4e}")
        # --- End Stdout Logging ---

        # --- Wandb Logging ---
        if self.wandb_run:
            log_data = {
                "train/loss_batch": train_loss_batch, # Log current batch loss
                "train/loss_avg": train_loss_avg, # Log the calculated average
                "train/learning_rate": lr,
            }
            if not math.isnan(eval_loss):
                 log_data["eval/loss"] = eval_loss
            self.wandb_run.log(log_data, step=step)
        # --- End Wandb Logging ---

        # --- CSV Logging ---
        if self.csvlog:
            try:
                eval_loss_csv = eval_loss if not math.isnan(eval_loss) else ''
                train_avg_csv = train_loss_avg if not math.isnan(train_loss_avg) else ''
                # Add current batch loss to CSV? Optional. Let's stick to avg for now.
                self.csvlog.write(f"{step},{train_avg_csv},{eval_loss_csv},{lr}\n")
                self.csvlog.flush()
            except IOError as e:
                print(f"Warning: Could not write to CSV log file: {e}")
            except Exception as e_csv:
                 print(f"Warning: Error writing data to CSV log at step {step}: {e_csv}")
        # --- End CSV Logging ---

    # Modified save_model to accept optional suffix
    def save_model(self, step=None, epoch=None, suffix=""):
        current_step_num = step if step is not None else (len(self.losses) if self.losses else 0)

        if epoch is None and current_step_num >= 10 ** 6:
            epoch_str = f"_s{round(current_step_num / 10 ** 6, 2)}M"
        elif epoch is None and current_step_num >= 10 ** 3:
            epoch_str = f"_s{round(current_step_num / 10 ** 3)}K"
        elif epoch is not None:
            epoch_str = f"_e{epoch}"
        else:  # Handle step 0 or low steps
            epoch_str = f"_s{current_step_num}"

        # Add suffix to filename if provided
        output_name = f"./{SAVE_FOLDER}/{self.name}{epoch_str}{suffix}"
        print(f"\nSaving checkpoint: {output_name} (Step: {current_step_num})")

        try:
            save_file(self.model.state_dict(), f"{output_name}.safetensors")
            torch.save(self.optimizer.state_dict(), f"{output_name}.optim.pth")
            if self.scheduler is not None:
                torch.save(self.scheduler.state_dict(), f"{output_name}.sched.pth")
            if self.scaler is not None and self.scaler.is_enabled():
                torch.save(self.scaler.state_dict(), f"{output_name}.scaler.pth")
            print("Checkpoint saved successfully.")
        except Exception as e:
            print(f"Error saving checkpoint {output_name}: {e}")

    def close(self):
        if self.csvlog:
            try:
                self.csvlog.close()
            except Exception as e:
                print(f"Warning: Error closing CSV log file: {e}")
            finally:
                self.csvlog = None
# --- End ModelWrapper Modifications ---
# Optional: Close wandb run if passed and managed here?
# if self.wandb_run:
#     self.wandb_run.finish()
#     self.wandb_run = None
# Usually wandb.finish() is called at the end of the main script.

# --- End ModelWrapper Modifications ---
