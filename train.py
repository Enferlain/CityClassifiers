# Version: 2.2.0 (Refactored for Clarity and Loss Selection)
import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import wandb
import math
import torch.nn as nn               # <--- Added nn import
import torch.nn.functional as F     # <--- Added functional import

# --- Local Imports ---
# Assuming utils.py is in the same directory or accessible
from utils import (
    ModelWrapper, get_embed_params, parse_and_load_args, write_config, # <<< CHANGED HERE
    LOG_EVERY_N, FocalLoss
)
from dataset import EmbeddingDataset
from model import PredictorModel

# --- Optimizer Imports ---
# Add the directory containing 'train.py' to the path for optimizer imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
optimizer_dir_path = os.path.join(current_script_dir, 'optimizer')
if os.path.isdir(optimizer_dir_path):
    print(f"Adding optimizer directory to sys.path: {current_script_dir}")
    if current_script_dir not in sys.path:
        sys.path.insert(0, current_script_dir)
else:
    print(f"Warning: Optimizer directory not found at {optimizer_dir_path}")

try:
    from optimizer.fmarscrop import FMARSCropV3ExMachina
    from optimizer.adopt import ADOPT
    from optimizer.schedulefree import (
        ScheduleFreeWrapper, ADOPTScheduleFree, ADOPTAOScheduleFree
    )
    print("Imported custom optimizers.")
    optimizers_available = True
except ImportError as e:
    print(f"Warning: Custom optimizer import failed ({e}). Check dependencies. Only AdamW/standard torch optims available.")
    FMARSCropV3ExMachina = None
    ADOPT = None
    ScheduleFreeWrapper = None
    ADOPTScheduleFree = None
    ADOPTAOScheduleFree = None
    optimizers_available = False
# --- End Optimizer Imports ---

# --- Global Settings ---
TARGET_DEV = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TF32 support enabled for CUDA operations.")
else:
    print("CUDA not available, TF32 settings not applied.")
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
# --- End Global Settings ---

# ================================================
#        Setup Functions
# ================================================

def setup_precision(args):
    """Sets up training precision based on args."""
    precision_arg = getattr(args, 'precision', 'fp32').lower()
    enabled_amp = False
    amp_dtype = torch.float32

    if precision_arg == 'fp16':
        if TARGET_DEV == 'cuda':
            amp_dtype = torch.float16
            enabled_amp = True
            print("Using fp16 mixed precision.")
        else:
            print("Warning: fp16 requested but CUDA is not available. Falling back to fp32.")
    elif precision_arg == 'bf16':
        if TARGET_DEV == 'cuda' and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            enabled_amp = True
            print("Using bf16 mixed precision.")
        else:
            if TARGET_DEV != 'cuda': print("Warning: bf16 requested but CUDA is not available. Falling back to fp32.")
            else: print("Warning: bf16 requested but not supported by hardware. Falling back to fp32.")
    else:
        print("Using fp32 precision.")
        if precision_arg not in ['fp32']:
             print(f"Warning: Unknown precision '{precision_arg}' specified. Using fp32.")

    return amp_dtype, enabled_amp

def setup_wandb(args):
    """Initializes Weights & Biases if available and enabled."""
    if not hasattr(wandb, 'init'): # Check if wandb was imported successfully
         print("Wandb library not available.")
         return None
    try:
        wandb_run = wandb.init(
            project=getattr(args, 'wandb_project', 'city-classifiers'),
            name=args.name,
            config=vars(args)
        )
        print("Weights & Biases initialized successfully.")
        return wandb_run
    except Exception as e:
        print(f"Could not initialize Weights & Biases: {e}. Training without wandb logging.")
        return None

def setup_dataloaders(args):
    """Sets up the dataset and dataloaders."""
    data_root_path = getattr(args, 'data_root', 'data')
    dataset_version = args.embed_ver
    print(f"Using dataset version (folder name): {dataset_version}")

    try:
        dataset = EmbeddingDataset(
            dataset_version,
            root=data_root_path,
            mode=args.arch,
            preload=getattr(args, 'preload_data', True),
            validation_split_count=args.val_split_count,
            seed=getattr(args, 'seed', 218)
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print(f"Check if dataset folder exists: {os.path.join(data_root_path, dataset_version)}")
        exit(1)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=getattr(args, 'num_workers', 0),
        collate_fn=getattr(dataset, 'collate_ignore_none', None)
    )

    val_loader = dataset.get_validation_loader(
        batch_size=args.batch * 2,
        num_workers=getattr(args, 'num_workers', 0)
    )
    if val_loader:
        print(f"Created validation loader with {len(val_loader.dataset)} samples.")
    else:
        print("No validation split requested or possible, skipping validation during training.")

    return dataset, train_loader, val_loader

def setup_model_criterion(args, dataset):
    """Sets up the model and criterion based on args."""
    criterion = None
    model = None
    loss_function_name = getattr(args, 'loss_function', None) # Get loss func name from args/config

    embed_params = get_embed_params(args.embed_ver)

    if args.arch == "score":
        num_outputs = 1
        # Select scorer loss function
        if loss_function_name == 'l1':
            print("Using L1Loss for scorer.")
            criterion = nn.L1Loss(reduction='mean')
        elif loss_function_name == 'mse':
            print("Using MSELoss (L2) for scorer.")
            criterion = nn.MSELoss(reduction='mean')
        else: # Default for scorer
            print("Defaulting to L1Loss for scorer.")
            criterion = nn.L1Loss(reduction='mean')

    elif args.arch == "class":
        num_outputs = getattr(args, 'num_labels', None) or dataset.num_labels
        if getattr(args, 'num_labels', None) and args.num_labels != dataset.num_labels:
             print(f"Warning: Label count mismatch! Config/Args: {args.num_labels}, Dataset: {dataset.num_labels}. Using dataset value.")
             args.num_labels = dataset.num_labels # Update args to match dataset

        # Select classifier loss function
        if loss_function_name == 'focal':
            print("Using Focal Loss (gamma=2.0) for classifier.")
            criterion = FocalLoss(gamma=getattr(args, 'focal_loss_gamma', 2.0))
            # Note: Basic FocalLoss here ignores class weights from args.weights
        elif loss_function_name == 'crossentropy':
            print("Using CrossEntropyLoss for classifier.")
            weights = None
            if args.weights and len(args.weights) == num_outputs:
                weights = torch.tensor(args.weights, device=TARGET_DEV, dtype=torch.float32)
                print(f"Class weights: {args.weights}")
            elif args.weights:
                print(f"Warning: Mismatch weights ({len(args.weights)}) vs labels ({num_outputs}). Ignoring weights.")
            criterion = nn.CrossEntropyLoss(weight=weights)
        else: # Default for classifier
            print("Defaulting to CrossEntropyLoss for classifier.")
            criterion = nn.CrossEntropyLoss() # No weights by default

    else:
        raise ValueError(f"Unknown model architecture '{args.arch}'")

    # Instantiate the model
    # Pass attention params from args if they exist, otherwise defaults in PredictorModel apply
    model_init_kwargs = embed_params.copy()
    model_init_kwargs["outputs"] = num_outputs
    if hasattr(args, 'num_attn_heads'): model_init_kwargs['num_attn_heads'] = args.num_attn_heads
    if hasattr(args, 'attn_dropout'): model_init_kwargs['attn_dropout'] = args.attn_dropout

    model = PredictorModel(**model_init_kwargs)
    model.to(TARGET_DEV)

    return model, criterion

def setup_optimizer_scheduler(args, model):
    """Sets up the optimizer and scheduler based on args."""
    optimizer = None
    scheduler = None
    is_schedule_free = False
    optimizer_name = getattr(args, 'optimizer', 'AdamW').lower()
    print(f"Using optimizer: {optimizer_name}")

    # --- Optimizer Selection Logic ---
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(args.lr), betas=tuple(getattr(args, 'betas', (0.9, 0.999))),
            weight_decay=float(getattr(args, 'weight_decay', 0.0)), eps=float(getattr(args, 'eps', 1e-8)),
        )
    elif optimizer_name == 'fmarscropv3exmachina' and FMARSCropV3ExMachina is not None:
        print("Instantiating FMARSCropV3ExMachina...")
        # +++ START ADDED BLOCK +++
        try:
            # Gather necessary args from the 'args' object (set by config/defaults)
            fmarscrop_args = {
                'lr': float(args.lr),
                'betas': tuple(getattr(args, 'betas', (0.99, 0.95))),  # Default specific to FMARS
                'eps': float(getattr(args, 'eps', 1e-6)),
                'weight_decay': float(getattr(args, 'weight_decay', 1e-3)),
                'gamma': float(getattr(args, 'gamma', 0.005)),
                # Include other relevant defaults if needed by constructor
                'eps2': float(getattr(args, 'eps2', 1e-2)),
                'eps_floor': getattr(args, 'eps_floor', None),
                'centralization': float(getattr(args, 'centralization', 0.0)),
                'moment_centralization': float(getattr(args, 'moment_centralization', 0.0)),
                'diff_mult': float(getattr(args, 'diff_mult', 1.0)),
                'momentum_lambda': float(getattr(args, 'momentum_lambda', 2.0)),
                'adaptive_clip': float(getattr(args, 'adaptive_clip', 1.0)),
                'adaptive_clip_eps': float(getattr(args, 'adaptive_clip_eps', 1e-3)),
                'adaptive_clip_type': getattr(args, 'adaptive_clip_type', 'global'),
                'stable_weight_decay': bool(getattr(args, 'stable_weight_decay', False)),
                'debias_beta1': bool(getattr(args, 'debias_beta1', False)),
                'debias_beta2': bool(getattr(args, 'debias_beta2', False)),
                'use_muon_pp': bool(getattr(args, 'use_muon_pp', False)),
                'update_strategy': getattr(args, 'update_strategy', 'cautious'),
                'stable_update': bool(getattr(args, 'stable_update', False)),
                'atan2_denom': bool(getattr(args, 'atan2_denom', False)),
                'use_orthograd': bool(getattr(args, 'use_orthograd', False)),
            }
            # Filter out None values ONLY if the constructor *cannot* handle them
            # fmarscrop_args = {k: v for k, v in fmarscrop_args.items() if v is not None}
            optimizer = FMARSCropV3ExMachina(model.parameters(), **fmarscrop_args)
        except (TypeError, AttributeError) as e:  # Catch errors if args mismatch
            print(f"ERROR: Failed to instantiate FMARSCropV3ExMachina. Check arguments.")
            print(f"  Error details: {e}")
            print("  Falling back to AdamW.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))  # Fallback

    elif optimizer_name == 'adoptaoschedulefree' and ADOPTAOScheduleFree is not None:
        # Gather args specifically for ADOPTAOScheduleFree
        adopt_args = {k: getattr(args, k) for k in [
            'lr', 'betas', 'weight_decay', 'eps', 'eps2', 'eps_floor',
            'weight_decouple', 'stable_weight_decay',
            'adaptive_clip', 'adaptive_clip_eps', 'adaptive_clip_type', # <--- Includes adaptive_clip_eps
            'debias_beta2', 'use_beta2_warmup', 'beta2_warmup_initial', 'beta2_warmup_steps',
            'mars_gamma', 'use_muon_pp', 'r', 'weight_lr_power',
            'fisher', 'update_strategy', 'stable_update', 'atan2_denom', 'use_orthograd',
            'use_spam_clipping', 'spam_clipping_threshold', 'spam_clipping_start_step', 'spam_clipping_type',
            'state_precision'] if hasattr(args, k)}

        # --- Add/Ensure this conversion exists ---
        if 'adaptive_clip_eps' in adopt_args and adopt_args['adaptive_clip_eps'] is not None:
             try: adopt_args['adaptive_clip_eps'] = float(adopt_args['adaptive_clip_eps'])
             except ValueError: print(f"Warning: Could not convert adaptive_clip_eps '{adopt_args['adaptive_clip_eps']}' to float. Using default?") # Handle

        # Convert types as needed
        adopt_args['lr'] = float(adopt_args['lr'])
        adopt_args['betas'] = tuple(adopt_args['betas'])
        adopt_args['weight_decay'] = float(adopt_args.get('weight_decay', 0.0))
        adopt_args['eps'] = float(adopt_args.get('eps', 1e-6))
        adopt_args['r'] = float(adopt_args.get('r', 0.0)) # Use get for non-essential args
        adopt_args['weight_lr_power'] = float(adopt_args.get('weight_lr_power', 2.0))
        # Remove None values if constructor doesn't handle them
        adopt_args = {k: v for k, v in adopt_args.items() if v is not None}

        optimizer = ADOPTAOScheduleFree(model.parameters(), **adopt_args)
        is_schedule_free = True
    # Add elif blocks for other custom optimizers (ADOPT, ScheduleFreeWrapper) if needed...
    else:
        if optimizer_name != 'adamw':
            print(f"Warning: Optimizer '{optimizer_name}' not found or not supported. Falling back to AdamW.")
        # Fallback to AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(args.lr), betas=tuple(getattr(args, 'betas', (0.9, 0.999))),
            weight_decay=float(getattr(args, 'weight_decay', 0.0)), eps=float(getattr(args, 'eps', 1e-8)),
        )
    # --- End Optimizer Selection ---

    # --- Scheduler Setup ---
    if not is_schedule_free:
        if getattr(args, 'cosine', True): # Default to cosine if not schedule free
            print("Using CosineAnnealingLR Scheduler.")
            t_max_steps = max(1, int(args.steps / args.batch))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_steps)
        elif getattr(args, 'warmup_steps', 0) > 0:
            print("Using LinearLR Warmup Scheduler.")
            warmup_iters = max(1, int(args.warmup_steps / args.batch))
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
        else:
             print("No scheduler specified or needed (not schedule-free, no cosine, no warmup).")
    else:
        print("Using a schedule-free optimizer, no scheduler will be used.")
    # --- End Scheduler Setup ---

    return optimizer, scheduler, is_schedule_free

def load_checkpoint(args, model, optimizer, scheduler, scaler):
    """Loads state from checkpoint if args.resume is provided."""
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        if not os.path.isfile(args.resume):
            print(f"Error: Resume file not found: {args.resume}"); exit(1)

        try:
             model.load_state_dict(load_file(args.resume))
        except Exception as e:
             print(f"Error loading model state_dict: {e}"); exit(1)

        optim_path = f"{os.path.splitext(args.resume)[0]}.optim.pth"
        sched_path = f"{os.path.splitext(args.resume)[0]}.sched.pth"
        scaler_path = f"{os.path.splitext(args.resume)[0]}.scaler.pth"

        if os.path.exists(optim_path):
            try: optimizer.load_state_dict(torch.load(optim_path, map_location=TARGET_DEV)); print("Optimizer state loaded.")
            except Exception as e: print(f"Warning: Could not load optimizer state: {e}")
        else: print("Warning: Optimizer state file not found. Starting fresh.")

        if scheduler is not None and os.path.exists(sched_path):
             try: scheduler.load_state_dict(torch.load(sched_path, map_location=TARGET_DEV)); print("Scheduler state loaded.")
             except Exception as e: print(f"Warning: Could not load scheduler state: {e}")

        if scaler.is_enabled() and os.path.exists(scaler_path):
             try: scaler.load_state_dict(torch.load(scaler_path, map_location=TARGET_DEV)); print("GradScaler state loaded.")
             except Exception as e: print(f"Warning: Could not load GradScaler state: {e}")

        # Try to determine start step from filename
        try:
            step_str = args.resume.split('_s')[-1].split('.')[0].split('_')[0] # More robust split
            scale = 1
            if 'K' in step_str: scale = 1000; step_str = step_str.replace('K', '')
            elif 'M' in step_str: scale = 1000000; step_str = step_str.replace('M', '')
            start_step = int(float(step_str) * scale)
            print(f"Attempting to resume progress from step ~{start_step}")
        except:
            print("Could not determine step number from checkpoint filename.")
            start_step = 0 # Default to 0 if parse fails
    return start_step

# ================================================
#        Main Training Loop Function
# ================================================

def train_loop(args, model, criterion, optimizer, scheduler, scaler,
               train_loader, val_loader, wrapper, start_step, enabled_amp, amp_dtype):
    """Runs the main training loop."""

    if hasattr(optimizer, 'train') and callable(optimizer.train):
        print("Setting optimizer to train mode.")
        optimizer.train()
    model.train()

    progress = tqdm(total=args.steps, initial=start_step, desc="Training")
    if start_step > 0:
        progress.n = start_step
        progress.last_print_n = start_step

    current_step = start_step
    best_eval_loss = float('inf')

    while current_step < args.steps:
        for batch in train_loader:
            if current_step >= args.steps: break
            if batch is None:
                print(f"Warning: Skipping step {current_step} due to invalid batch.")
                progress.update(args.batch)
                continue

            emb = batch.get("emb").to(TARGET_DEV)
            val = batch.get("val")
            if val is None:
                print(f"Error: 'val' is None in batch at step {current_step}. Skipping.")
                progress.update(args.batch)
                continue

            # --- Prepare Target Tensor ---
            try:
                if args.arch == "score":
                    # Scorer needs Float target, shape [B]
                    val = val.to(TARGET_DEV, dtype=torch.float32).squeeze()
                    if val.ndim != 1: raise ValueError(f"Scorer target shape {val.shape} != 1D")
                elif args.arch == "class":
                    # Classifier needs Long target, shape [B]
                    val = val.squeeze().to(dtype=torch.long, device=TARGET_DEV) # Squeeze first, then convert
                    if val.ndim != 1: raise ValueError(f"Classifier target shape {val.shape} != 1D")
            except Exception as e:
                print(f"Error processing target tensor 'val' at step {current_step}: {e}")
                print(f"  Original val shape: {batch.get('val').shape}, dtype: {batch.get('val').dtype}")
                progress.update(args.batch); continue
            # --- End Target Prep ---

            if emb.shape[0] != val.shape[0]:
                print(f"Error: Batch size mismatch emb ({emb.shape[0]}) vs val ({val.shape[0]}). Skipping.")
                progress.update(args.batch); continue

            # --- Forward/Backward Pass ---
            with torch.amp.autocast(device_type=TARGET_DEV, enabled=enabled_amp, dtype=amp_dtype):
                y_pred = model(emb) # Prediction on device

                # Prepare prediction for loss (e.g., squeeze for scorer)
                if args.arch == "score" and y_pred.ndim == 2 and y_pred.shape[1] == 1:
                     y_pred_for_loss = y_pred.squeeze(1)
                else:
                     y_pred_for_loss = y_pred

                # Calculate loss - ensure prediction is float32 for stability, target 'val' has correct type
                loss = criterion(y_pred_for_loss.to(torch.float32), val)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected at step {current_step}. Skipping step.")
                progress.update(args.batch)
                # Optionally: Try skipping optimizer step but still increment step?
                # current_step += args.batch
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # Optional: Gradient Clipping here if needed
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()
            # --- End Forward/Backward ---

            current_step += args.batch # Use actual batch size processed?
            progress.update(args.batch) # Update progress bar by configured batch size

            # --- Logging and Saving ---
            if current_step % LOG_EVERY_N == 0 and current_step > 0:
                eval_loss_val = float('nan')
                if val_loader:
                    eval_loss_val = wrapper.evaluate_on_validation_set(val_loader) # This handles model.eval()/train()
                    model.train() # Re-ensure train mode just in case

                if math.isnan(eval_loss_val): print(f"Step {current_step}: Eval loss is NaN.")
                wrapper.log_main(step=current_step, train_loss_batch=loss.item(), eval_loss=eval_loss_val)

                if not math.isnan(eval_loss_val) and eval_loss_val < best_eval_loss:
                    best_eval_loss = eval_loss_val
                    print(f"\nNew best validation loss: {best_eval_loss:.4e}. Saving best model...")
                    wrapper.save_model(step=current_step, suffix="_best_val")

            if args.nsave > 0 and (current_step // args.batch) % (args.nsave // args.batch) == 0:
                 if current_step > start_step: # Avoid saving at step 0 if not resuming
                      wrapper.save_model(step=current_step)
            # --- End Logging and Saving ---

            if current_step >= args.steps: break # Break inner loop if steps reached

        if current_step >= args.steps: break # Break outer loop

    progress.close()
    print(f"\nTraining loop finished at step {current_step}.")

# ================================================
#        Main Execution Block
# ================================================

def main():
    """Main function to run the training process."""
    args = parse_and_load_args() # <<< CHANGED HERE
    print(f"Target device: {TARGET_DEV}")

    amp_dtype, enabled_amp = setup_precision(args)
    wandb_run = setup_wandb(args)
    dataset, train_loader, val_loader = setup_dataloaders(args)
    model, criterion = setup_model_criterion(args, dataset)
    optimizer, scheduler, is_schedule_free = setup_optimizer_scheduler(args, model)

    # Needs to be created after potential precision changes
    scaler = torch.amp.GradScaler(device=TARGET_DEV, enabled=(enabled_amp and TARGET_DEV == 'cuda'))
    print(f"GradScaler enabled: {scaler.is_enabled()}")

    start_step = load_checkpoint(args, model, optimizer, scheduler, scaler)

    # Config needs to be written after all args might have been updated (e.g., num_labels)
    write_config(args)

    wrapper = ModelWrapper(
        name=args.name,
        model=model,
        device=TARGET_DEV,
        num_labels=getattr(args, 'num_labels', 1), # Use updated num_labels
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        wandb_run=wandb_run
    )

    try:
        train_loop(args, model, criterion, optimizer, scheduler, scaler,
                   train_loader, val_loader, wrapper, start_step, enabled_amp, amp_dtype)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
         print(f"\nAn error occurred during training: {e}")
         import traceback
         traceback.print_exc() # Print full traceback for debugging
    finally:
        # --- Final Save and Cleanup ---
        print(f"Saving final model...")
        # Determine final step count accurately
        final_step = wrapper.get_current_step() if hasattr(wrapper, 'get_current_step') else start_step # Assuming wrapper tracks steps, else use last known
        wrapper.save_model(epoch="final", step=final_step)
        wrapper.close() # Close log file

        if wandb_run:
            print("Finishing Weights & Biases run...")
            wandb_run.finish()

        print("Training script finished.")

if __name__ == "__main__":
    main()