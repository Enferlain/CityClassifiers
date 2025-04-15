# Version: 2.1.1
# Desc: Enabled TF32, fixed potential undefined 'val' variable.
import os
import sys

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import wandb
import math # Added for isnan check

# Add the directory containing 'train.py' to the path
# This ensures Python looks for the 'optimizer' folder relative to 'train.py'
current_script_dir = os.path.dirname(os.path.abspath(__file__))
optimizer_dir_path = os.path.join(current_script_dir, 'optimizer') # Construct full path

# Check if the optimizer directory exists before adding to path
if os.path.isdir(optimizer_dir_path):
     print(f"Adding optimizer directory to sys.path: {current_script_dir}") # Use current_script_dir for clarity
     # Add the *parent* directory of 'optimizer' so 'import optimizer' works
     if current_script_dir not in sys.path:
         sys.path.insert(0, current_script_dir)
else:
     print(f"Warning: Optimizer directory not found at {optimizer_dir_path}")

# --- Remove try/except temporarily to see the full error ---

from optimizer.fmarscrop import FMARSCropV3ExMachina # Example FMARSCrop
from optimizer.adopt import ADOPT # Example ADOPT
from optimizer.schedulefree import ( # Example ScheduleFree
    ScheduleFreeWrapper,
    ADOPTScheduleFree,
    ADOPTAOScheduleFree
)
print("Imported ScheduleFree variants")
# <-- Add imports for any other specific optimizers you want to use
optimizers_available = True
# except ImportError as e:
#     print(f"Warning: Custom optimizer import failed ({e}). Check dependencies within optimizer files. Only AdamW available.")
#     FMARSCropV3ExMachina = None
#     ADOPT = None
#     ScheduleFreeWrapper = None
#     ADOPTScheduleFree = None
#     ADOPTAOScheduleFree = None
#     optimizers_available = False
# --- End temporary modification ---

from dataset import EmbeddingDataset # ImageDataset might need similar update if used
from utils import ModelWrapper, get_embed_params, parse_args, write_config, LOG_EVERY_N # Import LOG_EVERY_N
from model import PredictorModel

# --- Enable TF32 potentially speeds up fp32 operations on Ampere+ ---
# Check if CUDA is available before trying to set backend flags
if torch.cuda.is_available():
    # These lines enable TensorFloat-32 for matrix multiplications and convolutions on supported hardware.
    # It can provide a significant speedup for fp32 operations with minimal precision loss.
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for matrix multiplications
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cuDNN convolution operations
    print("TF32 support enabled for CUDA operations.")
else:
    print("CUDA not available, TF32 settings not applied.")
# --- End TF32 ---


torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark mode (good if input sizes don't vary much)
torch.manual_seed(0)  # Set random seed for reproducibility

TARGET_DEV = "cuda" if torch.cuda.is_available() else "cpu"  # Use CPU if CUDA not found
print(f"Using target device: {TARGET_DEV}")

if __name__ == "__main__":
    args = parse_args()  # !! REMINDER: Add args for optimizer choice, hyperparameters, and --precision in optimizer_utils.py !!

    # --- Configure Precision ---
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
            precision_arg = 'fp32'
    elif precision_arg == 'bf16':
        if TARGET_DEV == 'cuda' and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            enabled_amp = True
            print("Using bf16 mixed precision.")
        else:
            if TARGET_DEV != 'cuda':
                print("Warning: bf16 requested but CUDA is not available. Falling back to fp32.")
            else:
                print("Warning: bf16 requested but not supported by hardware. Falling back to fp32.")
            precision_arg = 'fp32'  # Fallback
    else:
        print("Using fp32 precision.")
        if precision_arg != 'fp32':
            print(f"Warning: Unknown precision '{precision_arg}' specified. Using fp32.")
    # --- End Precision Config ---

    # --- Initialize Wandb (Keep as before) ---
    try:
        wandb.init(
            project="city-classifiers",  # Or your preferred project name
            name=args.name,  # Use the model name for the run name
            config=vars(args)  # Log all command-line args/config from optimizer_utils.py
        )
        print("Weights & Biases initialized successfully.")
    except Exception as e:
        print(f"Could not initialize Weights & Biases: {e}. Training without wandb logging.")
        wandb = None  # Set wandb to None if initialization fails
    # --- Wandb Initialized ---

    # v2.3.0: Use args.embed_ver instead of args.clip
    data_root_path = getattr(args, 'data_root', 'data')

    # --- Determine dataset version ---
    # The dataset folder name is directly determined by embed_ver in the config now
    dataset_version = args.embed_ver  # <<< Use the correct attribute directly

    # We can remove the special check for "CCAnatomy-AnatomyFlaws" base name,
    # as the config YAML should specify the correct embed_ver (like "CLIP-Anatomy" or "SIGLIP2-SO400M-512")
    # if args.base == "CCAnatomy-AnatomyFlaws":
    #     print(f"Anatomy config base name detected ({args.base}), using embed_ver: {args.embed_ver}")
    # --- End dataset version determination ---
    print(f"Using dataset version (folder name): {dataset_version}")

    try:
        # Pass the validation split count to the dataset constructor
        dataset = EmbeddingDataset(
            dataset_version,  # Pass the correctly determined version
            root=data_root_path,
            mode=args.arch,
            preload=True,  # Consider preload=False if RAM is an issue
            validation_split_count=args.val_split_count,
            seed=218  # Assuming seed is okay as default
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        # Consider adding more detail, e.g., which dataset_version folder failed
        print(f"  Check if dataset folder exists: {os.path.join(data_root_path, dataset_version)}")
        exit(1)

    # Training DataLoader uses the main dataset instance (which now only has training shards)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=0, # Set num_workers > 0 if not using preload=True or ImageDataset
        collate_fn=dataset.collate_ignore_none if hasattr(dataset, 'collate_ignore_none') else None # Use the collate_fn from dataset if available
    )

    # Create Validation DataLoader
    val_loader = dataset.get_validation_loader(
        batch_size=args.batch * 2, # Can often use larger batch for validation
        num_workers=0 # Use 0 if validation data is preloaded
    )
    if val_loader:
        print(f"Created validation loader with {len(val_loader.dataset)} samples.")
    else:
        print("No validation split requested or possible, skipping validation during training.")

    # --- Model Definition (No Change) ---
    if args.arch == "score":
        # Make sure criterion handles potential NaN outputs if model or target is bad
        criterion = torch.nn.L1Loss(reduction='mean') # Specify reduction for clarity
        model = PredictorModel(
            outputs=1,
            **get_embed_params(args.embed_ver)
        )
    elif args.arch == "class":
        args.num_labels = args.num_labels or dataset.num_labels
        if args.num_labels != dataset.num_labels:
             print(f"Warning: Label count mismatch! Config/Args: {args.num_labels}, Dataset: {dataset.num_labels}. Using dataset value.")
             args.num_labels = dataset.num_labels # Prioritize dataset derived value

        weights = None
        if args.weights and len(args.weights) == args.num_labels:
            weights = torch.tensor(args.weights, device=TARGET_DEV, dtype=torch.float32)
            print(f"Class weights: '{args.weights}'")
        elif args.weights:
            print(f"Warning: Mismatch between number of weights ({len(args.weights)}) and number of labels ({args.num_labels}). Ignoring weights.")

        criterion = torch.nn.CrossEntropyLoss(weight=weights) #label_smoothing=0.1) # Add label_smoothing=0.1
        model = PredictorModel(
            outputs=args.num_labels,
            **get_embed_params(args.embed_ver)
        )
    else:
        raise ValueError(f"Unknown model architecture '{args.arch}'")

    model.to(TARGET_DEV)

    # --- Optimizer Selection (Keep as before) ---
    optimizer = None
    is_schedule_free = False
    optimizer_name = getattr(args, 'optimizer', 'AdamW').lower()
    print(f"Using optimizer: {optimizer_name}")
    # (Optimizer selection logic remains the same as previous version)
    # Example for AdamW:
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            betas=tuple(getattr(args, 'betas', (0.9, 0.999))),  # Ensure betas is a tuple
            weight_decay=float(getattr(args, 'weight_decay', 0.0)),
            eps=float(getattr(args, 'eps', 1e-8)),
        )

        # --- ADD THIS BLOCK ---
    elif optimizer_name == 'fmarscropv3exmachina' and 'FMARSCropV3ExMachina' in globals() and FMARSCropV3ExMachina is not None:
        print("Instantiating FMARSCropV3ExMachina...")
        try:
            # Gather necessary args from the 'args' object (set by config/defaults)
            fmarscrop_args = {
                'lr': float(args.lr),
                'betas': tuple(getattr(args, 'betas', (0.99, 0.95))),  # Default specific to FMARS
                'eps': float(getattr(args, 'eps', 1e-6)),
                'weight_decay': float(getattr(args, 'weight_decay', 1e-3)),
                'gamma': float(getattr(args, 'gamma', 0.005)),
                # Include other relevant defaults from FMARSCropV3ExMachina.__init__ if needed
                'eps2': float(getattr(args, 'eps2', 1e-2)),
                'eps_floor': getattr(args, 'eps_floor', None),
                'centralization': float(getattr(args, 'centralization', 0.0)),
                'moment_centralization': float(getattr(args, 'moment_centralization', 0.0)),
                'diff_mult': float(getattr(args, 'diff_mult', 1.0)),  # Default might vary between FMARS versions
                'momentum_lambda': float(getattr(args, 'momentum_lambda', 2.0)),  # Default for V3Ex
                # 'clip': float(getattr(args, 'clip', 1.0)),  # Default for V3Ex
                'adaptive_clip': float(getattr(args, 'adaptive_clip', 1.0)),  # Default for V3Ex
                'adaptive_clip_eps': float(getattr(args, 'adaptive_clip_eps', 1e-3)),  # Default for V3Ex
                'adaptive_clip_type': getattr(args, 'adaptive_clip_type', 'global'),  # Default for V3Ex
                'stable_weight_decay': bool(getattr(args, 'stable_weight_decay', False)),  # Default for V3Ex
                'debias_beta1': bool(getattr(args, 'debias_beta1', False)),  # Default for V3Ex
                'debias_beta2': bool(getattr(args, 'debias_beta2', False)),  # Default for V3Ex
                'use_muon_pp': bool(getattr(args, 'use_muon_pp', False)),  # Default for V3Ex
                'update_strategy': getattr(args, 'update_strategy', 'cautious'),  # Default for V3Ex
                'stable_update': bool(getattr(args, 'stable_update', False)),  # Default for V3Ex
                'atan2_denom': bool(getattr(args, 'atan2_denom', False)),  # Default for V3Ex
                'use_orthograd': bool(getattr(args, 'use_orthograd', False)),  # Default for V3Ex
            }
            # Filter out None values if the optimizer __init__ doesn't handle them
            # fmarscrop_args = {k: v for k, v in fmarscrop_args.items() if v is not None} # Optional filtering

            optimizer = FMARSCropV3ExMachina(model.parameters(), **fmarscrop_args)
        except TypeError as e:
            print(
                f"ERROR: Failed to instantiate FMARSCropV3ExMachina. Check arguments in config/defaults vs optimizer __init__.")
            print(f"  Error details: {e}")
            print("  Falling back to AdamW.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))  # Fallback
    # --- END ADD BLOCK ---

    elif optimizer_name == 'adoptaoschedulefree' and 'ADOPTAOScheduleFree' in globals() and ADOPTAOScheduleFree is not None:
        optimizer = ADOPTAOScheduleFree(
            model.parameters(),
            lr=float(args.lr),
            betas=tuple(getattr(args, 'betas', (0.9, 0.9999))),  # Ensure betas is tuple
            weight_decay=float(getattr(args, 'weight_decay', 0.0)),
            eps=float(getattr(args, 'eps', 1e-6)),
            r=float(getattr(args, 'r_sf', 0.0)),
            weight_lr_power=float(getattr(args, 'wlpow_sf', 2.0)),
            state_precision=getattr(args, 'state_precision', 'parameter'),
            # ... other ADOPTAOScheduleFree params ...
        )
        is_schedule_free = True
    # ... Fallback to AdamW ...
    else:
        if optimizer_name != 'adamw':
            print(f"Warning: Optimizer '{optimizer_name}' not found or not supported. Falling back to AdamW.")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            betas=tuple(getattr(args, 'betas', (0.9, 0.999))),
            weight_decay=float(getattr(args, 'weight_decay', 0.0)),
            eps=float(getattr(args, 'eps', 1e-8)),
        )

    # --- Scheduler Initialization (Conditional - Keep as before) ---
    scheduler = None
    if not is_schedule_free:
        if args.cosine:
            print("Using CosineAnnealingLR")
            # Ensure T_max is at least 1
            t_max_steps = max(1, int(args.steps / args.batch))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max_steps,
            )
        else:
            print("Using LinearLR Warmup")
            # Ensure total_iters is at least 1
            warmup_iters = max(1, int(getattr(args, 'warmup_steps', 5000) / args.batch))
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_iters,
            )
    else:
        print("Using a schedule-free optimizer, no scheduler will be used.")

    # --- Initialize GradScaler ---
    # Enabled=False just makes its methods no-ops if not using AMP / on CPU
    scaler = torch.cuda.amp.GradScaler(enabled=(enabled_amp and TARGET_DEV == 'cuda'))
    print(f"GradScaler enabled: {scaler.is_enabled()}")
    # --- End GradScaler ---

    # --- Resume Logic (Modified Scaler Part) ---
    start_step = 0  # Keep track of starting step for resuming progress bar
    if args.resume:
        print(f"Resuming from {args.resume}")
        if not os.path.isfile(args.resume):
            print(f"Error: Resume file not found: {args.resume}")
            exit(1)

        model.load_state_dict(load_file(args.resume))

        optim_path = f"{os.path.splitext(args.resume)[0]}.optim.pth"
        if os.path.exists(optim_path):
            optimizer.load_state_dict(torch.load(optim_path, map_location=TARGET_DEV))  # Load to correct device
            print("Optimizer state loaded.")
            # Find the step number from the checkpoint filename if possible
            try:
                step_str = args.resume.split('_s')[-1].split('K')[0].split('M')[0]
                scale = 1000 if 'K' in args.resume else (1000000 if 'M' in args.resume else 1)
                start_step = int(float(step_str) * scale)
                print(f"Attempting to resume progress from step ~{start_step}")
            except:
                print("Could not determine step number from checkpoint filename.")

            # Optional: try to load scheduler state if it exists
            sched_path = f"{os.path.splitext(args.resume)[0]}.sched.pth"
            if scheduler is not None and os.path.exists(sched_path):
                scheduler.load_state_dict(torch.load(sched_path, map_location=TARGET_DEV))
                print("Scheduler state loaded.")
            # Optional: try to load scaler state
            scaler_path = f"{os.path.splitext(args.resume)[0]}.scaler.pth"
            if scaler.is_enabled() and os.path.exists(scaler_path):
                scaler.load_state_dict(torch.load(scaler_path, map_location=TARGET_DEV))
                print("GradScaler state loaded.")

        else:
            print(f"Warning: Optimizer state file not found at {optim_path}. Starting with fresh optimizer state.")
    # --- End Resume Logic ---

    # --- Config and Wrapper (Pass potentially None scheduler) ---
    write_config(args)
    # !! REMINDER: ModelWrapper needs modification for wandb logging & optimizer.eval/train !!
    wrapper = ModelWrapper(
        name=args.name,
        model=model,
        device=TARGET_DEV,
        # dataset=dataset,
        num_labels=dataset.num_labels if args.arch == 'class' else 1,
        # Pass num_labels explicitly if needed by wrapper logging
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,  # Pass scheduler (might be None)
        scaler=scaler,  # Pass scaler
        wandb_run=wandb  # Pass wandb run object
    )

    # --- Set Optimizer Train Mode (Keep as before) ---
    if hasattr(optimizer, 'train') and callable(optimizer.train):
        print("Setting optimizer to train mode.")
        optimizer.train()

    model.train() # Set model to train mode initially
    progress = tqdm(total=args.steps, initial=start_step)
    if start_step > 0:
        progress.n = start_step
        progress.last_print_n = start_step

    current_step = start_step
    best_eval_loss = float('inf') # Keep track of best validation loss

    while current_step < args.steps:
        for batch in loader:
            if current_step >= args.steps: break
            if batch is None: # Handle None batch from collate_ignore_none
                 print(f"Warning: Skipping step {current_step} due to invalid batch (likely data loading error).")
                 # Optionally increment step counter even if skipping? Depends on desired behavior.
                 # current_step += args.batch # If you want skipped batches to count towards total steps
                 progress.update(args.batch) # Update progress bar anyway
                 continue # Skip this iteration

            # Ensure batch items are on the correct device
            emb = batch.get("emb").to(TARGET_DEV)
            val = batch.get("val") # Keep on CPU initially if processing needed

            if val is None:
                print(f"Error: 'val' is None in batch at step {current_step}. Skipping step.")
                progress.update(args.batch)
                continue

            # Move target value 'val' to device and ensure correct dtype
            if args.arch == "score":
                val = val.to(TARGET_DEV, dtype=torch.float32)
            # v2.2.3: Correct target tensor preparation for CrossEntropyLoss
            elif args.arch == "class":
                # Target for CrossEntropyLoss needs to be 1D LongTensor of class indices
                if val is None:  # Check if val was loaded correctly
                    print(f"Error: Target 'val' is None in batch at step {current_step}. Skipping.")
                    progress.update(args.batch)
                    continue  # Skip this batch

                try:
                    # Dataset returns shape [B, 1], squeeze to [B] and convert to long
                    val = val.squeeze(-1).to(dtype=torch.long, device=TARGET_DEV)
                    # Shape check after processing
                    if val.ndim != 1 or val.shape[0] != emb.shape[0]:
                        raise ValueError(f"Processed target shape {val.shape} is not 1D or batch size mismatch.")
                except Exception as e:
                    print(f"Error processing target tensor 'val' at step {current_step}: {e}")
                    print(f"  Original val shape: {batch.get('val').shape}, dtype: {batch.get('val').dtype}")
                    progress.update(args.batch)
                    continue  # Skip this batch

            # Check shapes
            if emb.shape[0] != val.shape[0]:
                print(f"Error: Mismatch batch size between embeddings ({emb.shape[0]}) and targets ({val.shape[0]}) at step {current_step}. Skipping.")
                progress.update(args.batch)
                continue

            # Forward/Backward Pass with AMP
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled(), dtype=amp_dtype):
                y_pred = model(emb)
                # Check for NaNs in prediction
                if torch.isnan(y_pred).any():
                    print(f"Warning: NaN detected in model prediction at step {current_step}. Skipping step.")
                    progress.update(args.batch) # Still update progress
                    # Consider stopping training or reducing LR if NaNs persist
                    continue
                loss = criterion(y_pred.to(torch.float32), val) # Cast prediction to fp32 for stable loss calculation
                # Check for NaNs/inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf detected in loss at step {current_step}. Skipping backward/step.")
                    progress.update(args.batch)
                    # Skip optimizer step if loss is invalid
                    current_step += args.batch # Increment step here since we skip the logging section maybe? Or log with NaN loss?
                    continue # Skip the rest of the loop for this batch

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # Optional: Gradient Clipping (before scaler.step)
            # scaler.unscale_(optimizer) # Unscale first if needed by clip_grad_norm_
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Example
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            current_step += args.batch
            progress.update(args.batch)

            # --- Evaluation and Logging Step ---
            if current_step % LOG_EVERY_N == 0 and current_step > 0:
                eval_loss_val = float('nan') # Default to NaN if no validation
                if val_loader:
                    # Perform evaluation on the validation set
                    eval_loss_val = wrapper.evaluate_on_validation_set(val_loader)
                    model.train() # Ensure model is back in train mode after eval

                # Log metrics (pass eval loss to log_main)
                # Check if eval_loss_val is NaN before logging
                if math.isnan(eval_loss_val):
                     print(f"Step {current_step}: Eval loss is NaN (validation loader might be empty or evaluation failed).")

                # v2.2.1: Use correct arg name train_loss_batch when calling log_main
                wrapper.log_main(
                    step=current_step,
                    train_loss_batch=loss.item(),  # Log current batch train loss <--- NEW ARG NAME
                    eval_loss=eval_loss_val
                )

                # Checkpoint saving based on validation loss (optional)
                if not math.isnan(eval_loss_val) and eval_loss_val < best_eval_loss:
                     best_eval_loss = eval_loss_val
                     print(f"\nNew best validation loss: {best_eval_loss:.4e}. Saving best model...")
                     wrapper.save_model(step=current_step, suffix="_best_val") # Add suffix

            # --- Periodic Saving (Keep original logic too) ---
            if args.nsave > 0 and (current_step // args.batch) % (args.nsave // args.batch) == 0:
                wrapper.save_model(step=current_step)

            if current_step >= args.steps: break # Inner loop break

        # End of epoch/loader iteration
        if current_step >= args.steps: break # Outer loop break

    progress.close()

    # --- Final Save and Cleanup ---
    print(f"\nTraining finished at step {current_step}. Saving final model...")
    wrapper.save_model(epoch="final", step=current_step) # Use 'final' epoch string
    # Save final best val model again? Usually covered by periodic save or last step.
    wrapper.close()

    if wandb:
        print("Finishing Weights & Biases run...")
        wandb.finish()

    print("Training script finished.")