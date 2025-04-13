# Version: 2.1.1
# Desc: Enabled TF32, fixed potential undefined 'val' variable.
import os
import sys

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import wandb

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
# try:
from optimizer.fmarscrop import FMARSCropV3ExMachina # Example FMARSCrop
print("Imported FMARSCropV3ExMachina")
from optimizer.adopt import ADOPT # Example ADOPT
print("Imported ADOPT")
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

from dataset import EmbeddingDataset, ImageDataset
# !! REMINDER: optimizer_utils.py needs updates for args and ModelWrapper !!
from utils import ModelWrapper, get_embed_params, parse_args, write_config
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

    # --- Dataset and DataLoader ---
    data_root_path = getattr(args, 'data_root', 'data')  # Use arg if exists, else default

    # --- Determine dataset version ---
    # Default to args.clip, but override if using anatomy config?
    dataset_version = args.clip  # Default from YAML model->clip
    if args.base == "CCAnatomy-AnatomyFlaws":  # Check base name from config
        dataset_version = "CLIP-Anatomy"  # Force version for this config
        print(f"Anatomy config detected, forcing dataset version to: {dataset_version}")
    # --- End dataset version determination ---

    try:
        if args.images:
            dataset = ImageDataset(dataset_version, root=data_root_path, mode=args.arch)
        else:
            dataset = EmbeddingDataset(dataset_version, root=data_root_path, mode=args.arch, preload=True)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        exit(1)  # Exit if dataset fails

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        pin_memory=False,  # Usually False is fine, sometimes True helps if data loading is bottleneck
        num_workers=0,  # Set num_workers > 0 for background data loading if not using --images
        # Make sure dataset __getitem__ is safe for multiprocessing if num_workers > 0
    )

    # --- Model Definition (No Change) ---
    if args.arch == "score":
        criterion = torch.nn.L1Loss()
        model = PredictorModel(
            outputs=1,
            **get_embed_params(args.clip)
        )
    elif args.arch == "class":
        args.num_labels = args.num_labels or dataset.num_labels
        assert args.num_labels == dataset.num_labels, "Label count mismatch!"
        weights = None
        if args.weights:
            weights = torch.tensor(args.weights, device=TARGET_DEV)
            print(f"Class weights: '{args.weights}'")
        criterion = torch.nn.CrossEntropyLoss(weight=weights)  # Pass weights to criterion
        model = PredictorModel(
            outputs=args.num_labels,
            **get_embed_params(args.clip)
        )
    else:
        raise ValueError(f"Unknown model architecture '{args.arch}'")

    model.to(TARGET_DEV)  # Move model to device

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
    # ... Add elif branches for FMARSCrop, ADOPT, ScheduleFree variants ...
    elif optimizer_name == 'adoptaoschedulefree' and ADOPTAOScheduleFree:
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
        dataset=dataset,
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

    model.train()
    progress = tqdm(total=args.steps, initial=start_step)  # Set initial step for progress bar
    # Set progress bar step to prevent issues if start_step > 0
    if start_step > 0:
        progress.n = start_step
        progress.last_print_n = start_step  # Avoid immediate print if resuming near end

    current_step = start_step  # Use current_step for logic
    while current_step < args.steps:
        for batch in loader:
            if current_step >= args.steps: break  # Check before processing batch

            emb = batch.get("emb").to(TARGET_DEV)  # Assume input embeddings are fp32

            # --- Initialize val before conditional assignment ---
            val = None
            # --- Label Preparation ---
            if args.arch == "score":
                val = batch.get("val").to(TARGET_DEV, dtype=torch.float32)  # Ensure target is fp32
            elif args.arch == "class":
                try:
                    # More robust handling for batch labels
                    raw_labels = batch.get("raw")
                    if isinstance(raw_labels, torch.Tensor):
                        raw_labels = raw_labels.tolist()  # Convert tensor to list if needed

                    val_template = torch.zeros(args.num_labels, device=TARGET_DEV)
                    current_vals = []
                    for raw_label in raw_labels:
                        val_i = val_template.clone()
                        # Ensure raw_label is a valid index
                        label_idx = int(raw_label)
                        if 0 <= label_idx < args.num_labels:
                            val_i[label_idx] = 1.0
                        else:
                            print(f"Warning: Invalid label index {label_idx} encountered. Skipping.")
                            # Decide how to handle invalid labels (skip batch? use default?)
                            # For now, let's just create a zero vector which might cause issues depending on criterion
                            val_i = torch.zeros_like(val_template)  # Or handle differently

                        current_vals.append(val_i)
                    if not current_vals:  # Handle case where all labels were invalid
                        print("Warning: No valid labels found in batch. Skipping step.")
                        continue  # Skip to next iteration
                    val = torch.stack(current_vals).to(dtype=torch.float32)  # Ensure target is fp32
                except Exception as e:
                    print(f"Error processing batch labels: {e}. Skipping step.")
                    continue  # Skip to next iteration

            # Check if val was successfully assigned
            if val is None:
                print(f"Error: 'val' could not be determined for arch '{args.arch}'. Skipping step.")
                continue  # Skip to next iteration
            # --- End Label Prep ---

            # --- Modified Forward/Backward Pass with AMP ---
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled(), dtype=amp_dtype):
                y_pred = model(emb)  # forward
                # Ensure y_pred and val have compatible shapes and types for criterion
                loss = criterion(y_pred.to(torch.float32), val)  # Cast y_pred to fp32 for loss calculation if needed

            # Backward pass using scaler
            optimizer.zero_grad(set_to_none=True)  # Use set_to_none=True for potential speedup
            scaler.scale(loss).backward()

            # Optimizer step using scaler
            scaler.step(optimizer)

            # Update scaler for next iteration
            scaler.update()
            # --- End Modified Forward/Backward ---

            # --- Step Scheduler (Conditional - Keep as before) ---
            if scheduler is not None:
                scheduler.step()

            # --- Eval/Save/Log ---
            current_step += args.batch  # Increment step counter
            progress.update(args.batch)
            wrapper.log_step(loss.data.item(), current_step)  # Pass current_step

            # Saving logic based on current_step
            # Ensure nsave is positive before checking modulo
            # Check if current_step is a multiple of nsave, accounting for batch size
            if args.nsave > 0 and (current_step // args.batch) % (args.nsave // args.batch) == 0:
                wrapper.save_model(step=current_step)  # Pass current_step for filename

        # End of epoch/loader iteration (inner loop)
        if current_step >= args.steps: break  # Ensure outer loop breaks if steps reached mid-epoch

    # End of training loop (outer loop)
    progress.close()

    # --- Final Save and Cleanup ---
    print(f"\nTraining finished at step {current_step}. Saving final model...")
    wrapper.save_model(epoch="final", step=current_step)  # Use 'final' epoch string, pass final step
    wrapper.close()

    # --- Finish Wandb Run ---
    if wandb:
        print("Finishing Weights & Biases run...")
        wandb.finish()

    print("Training script finished.")