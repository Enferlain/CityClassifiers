# Version: 2.0.0 
# Desc: Integrate wandb logging and flexible optimizer selection 
#       (AdamW, FMARSCrop, ADOPT, ScheduleFree)

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import wandb  # <-- Added wandb import

# --- Assuming new optimizers are in an 'optimizer' directory ---
# --- Adjust path if necessary ---
# --- You MUST install pytorch-optimizer: pip install pytorch-optimizer ---
# --- You MIGHT need torchao/triton for ADOPTAOScheduleFree: pip install torchao triton ---
try:
    from optimizer.fmarscrop import FMARSCropV3ExMachina  # Example FMARSCrop
    from optimizer.adopt import ADOPT  # Example ADOPT
    from optimizer.schedulefree import (  # Example ScheduleFree
        ScheduleFreeWrapper,
        ADOPTScheduleFree,
        ADOPTAOScheduleFree
    )
# <-- Add imports for any other specific optimizers you want to use
except ImportError:
    print("Warning: Custom optimizer files not found. Only AdamW will be available.")
    FMARSCropV3ExMachina = None
    ADOPT = None
    ScheduleFreeWrapper = None
    ADOPTScheduleFree = None
    ADOPTAOScheduleFree = None
# --- End Optimizer Imports ---

from dataset import EmbeddingDataset, ImageDataset
from utils import ModelWrapper, get_embed_params, parse_args, write_config
from model import PredictorModel

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

TARGET_DEV = "cuda"

if __name__ == "__main__":
    args = parse_args()  # !! IMPORTANT: Add args for optimizer choice and its hyperparameters in utils.py !!
    # e.g., --optimizer [AdamW|FMARSCropV3ExMachina|ADOPTAOScheduleFree]
    # e.g., --gamma, --state_precision, --sf_momentum, etc.

    # --- Initialize Wandb ---
    try:
        wandb.init(
            project="city-classifiers",  # Or your preferred project name
            name=args.name,  # Use the model name for the run name
            config=vars(args)  # Log all command-line args/config from utils.py
        )
        print("Weights & Biases initialized successfully.")
    except Exception as e:
        print(f"Could not initialize Weights & Biases: {e}. Training without wandb logging.")
        wandb = None  # Set wandb to None if initialization fails
    # --- Wandb Initialized ---

    if args.images:
        dataset = ImageDataset(args.clip, mode=args.arch)
    else:
        dataset = EmbeddingDataset(args.clip, mode=args.arch, preload=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
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
        criterion = torch.nn.CrossEntropyLoss(weights)
        model = PredictorModel(
            outputs=args.num_labels,
            **get_embed_params(args.clip)
        )
    else:
        raise ValueError(f"Unknown model architecture '{args.arch}'")

    model.to(TARGET_DEV)  # Move model to device early

    # --- Optimizer Selection ---
    optimizer = None
    is_schedule_free = False  # Flag to track if scheduler is needed

    # !! IMPORTANT: Add 'optimizer' argument in parse_args (utils.py) !!
    optimizer_name = getattr(args, 'optimizer', 'AdamW').lower()  # Default to AdamW if arg missing

    print(f"Using optimizer: {optimizer_name}")

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            # Add betas, weight_decay, eps from args if needed
            betas=getattr(args, 'betas', (0.9, 0.999)),
            weight_decay=float(getattr(args, 'weight_decay', 0.0)),
            eps=float(getattr(args, 'eps', 1e-8)),
        )
    elif optimizer_name == 'fmarscropv3exmachina' and FMARSCropV3ExMachina:
        optimizer = FMARSCropV3ExMachina(
            model.parameters(),
            lr=float(args.lr),
            # !! Add args for FMARSCrop specific params in utils.py !!
            betas=getattr(args, 'betas', (0.99, 0.9999, 0.999)),
            weight_decay=float(getattr(args, 'weight_decay', 0.0)),
            eps=float(getattr(args, 'eps', 1e-6)),
            gamma=float(getattr(args, 'gamma', 0.005)),
            # ... other FMARSCrop params ...
        )
    elif optimizer_name == 'adopt' and ADOPT:
        optimizer = ADOPT(
            model.parameters(),
            lr=float(args.lr),
            # !! Add args for ADOPT specific params in utils.py !!
            betas=getattr(args, 'betas', (0.9, 0.9999)),
            weight_decay=float(getattr(args, 'weight_decay', 0.0)),
            eps=float(getattr(args, 'eps', 1e-6)),
            # ... other ADOPT params ...
        )
    elif optimizer_name == 'adoptschedulefree' and ADOPTScheduleFree:
        optimizer = ADOPTScheduleFree(
            model.parameters(),
            lr=float(args.lr),
            # !! Add args for ADOPTScheduleFree specific params in utils.py !!
            betas=getattr(args, 'betas', (0.9, 0.9999)),
            weight_decay=float(getattr(args, 'weight_decay', 0.0)),
            eps=float(getattr(args, 'eps', 1e-6)),
            r=float(getattr(args, 'r_sf', 0.0)),
            weight_lr_power=float(getattr(args, 'wlpow_sf', 2.0)),
            # ... other ADOPTScheduleFree params ...
        )
        is_schedule_free = True  # Mark as schedule-free
    elif optimizer_name == 'adoptaoschedulefree' and ADOPTAOScheduleFree:
        optimizer = ADOPTAOScheduleFree(
            model.parameters(),
            lr=float(args.lr),
            # !! Add args for ADOPTAOScheduleFree specific params in utils.py !!
            betas=getattr(args, 'betas', (0.9, 0.9999)),
            weight_decay=float(getattr(args, 'weight_decay', 0.0)),
            eps=float(getattr(args, 'eps', 1e-6)),
            r=float(getattr(args, 'r_sf', 0.0)),
            weight_lr_power=float(getattr(args, 'wlpow_sf', 2.0)),
            state_precision=getattr(args, 'state_precision', 'parameter'),  # Add arg for this!
            # ... other ADOPTAOScheduleFree params ...
        )
        is_schedule_free = True  # Mark as schedule-free
    # --- Add elif branches for other optimizers ---
    else:
        if optimizer_name != 'adamw':
            print(f"Warning: Optimizer '{optimizer_name}' not found or not supported. Falling back to AdamW.")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            betas=getattr(args, 'betas', (0.9, 0.999)),
            weight_decay=float(getattr(args, 'weight_decay', 0.0)),
            eps=float(getattr(args, 'eps', 1e-8)),
        )

    # --- Scheduler Initialization (Conditional) ---
    scheduler = None
    if not is_schedule_free:  # Only initialize scheduler if not using a schedule-free optimizer
        if args.cosine:
            print("Using CosineAnnealingLR")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(args.steps / args.batch),
            )
        else:
            # Assuming LinearLR is default if cosine is false
            print("Using LinearLR")
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,  # Make these configurable via args?
                end_factor=1.0,
                total_iters=int(getattr(args, 'warmup_steps', 5000) / args.batch),  # Add warmup_steps arg?
            )
    else:
        print("Using a schedule-free optimizer, no scheduler will be used.")

    # --- Resume Logic (Modified Scheduler Part) ---
    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(load_file(args.resume))
        # model.to(TARGET_DEV) # Model already moved

        optim_path = f"{os.path.splitext(args.resume)[0]}.optim.pth"
        if os.path.exists(optim_path):
            optimizer.load_state_dict(torch.load(optim_path))
            # Restore LR only if scheduler exists and was likely saved
            if scheduler is not None and hasattr(scheduler, 'base_lrs'):
                # This might need adjustment depending on how state is saved/loaded
                # It assumes the base LR was intended to be restored.
                try:
                    optimizer.param_groups[0]['lr'] = scheduler.base_lrs[0]
                    print(f"Optimizer state loaded. Reset LR based on scheduler base_lrs.")
                except Exception as e:
                    print(f"Could not reset LR from scheduler state: {e}")
            else:
                print("Optimizer state loaded. LR not reset from scheduler (scheduler absent or state mismatch).")

        else:
            print(f"Warning: Optimizer state file not found at {optim_path}. Starting with fresh optimizer state.")

    # --- Config and Wrapper (Pass potentially None scheduler) ---
    write_config(args)  # model config file
    wrapper = ModelWrapper(  # model wrapper for saving/eval/etc
        name=args.name,
        model=model,
        device=TARGET_DEV,
        dataset=dataset,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,  # Pass scheduler (might be None)
        # Add wandb object if ModelWrapper handles logging
        # wandb_run = wandb
    )

    # --- Optional: Wandb Watch Model ---
    # if wandb:
    #    wandb.watch(model, log="all", log_freq=100) # Log gradients, params every 100 steps

    # --- Set Optimizer Train Mode (for ScheduleFree) ---
    if hasattr(optimizer, 'train') and callable(optimizer.train):
        print("Setting optimizer to train mode.")
        optimizer.train()

    model.train()  # Ensure model is in train mode
    progress = tqdm(total=args.steps)
    while progress.n < args.steps:
        for batch in loader:
            emb = batch.get("emb").to(TARGET_DEV)

            if args.arch == "score":
                val = batch.get("val").to(TARGET_DEV)
            elif args.arch == "class":
                # Maybe pre-allocate this outside the loop?
                val_template = torch.zeros(args.num_labels, device=TARGET_DEV)
                current_vals = []
                for raw_label in batch.get("raw"):  # Assuming "raw" might contain labels for the batch
                    val_i = val_template.clone()
                    val_i[raw_label] = 1.0
                    current_vals.append(val_i)
                val = torch.stack(current_vals)  # Create batch of one-hot vectors

            # Original logic (seems to assume batch size 1 or duplicates label?)
            # val = torch.zeros(args.num_labels, device=TARGET_DEV)
            # val[batch.get("raw")] = 1.0 # expand classes - Error if batch.get("raw") is a tensor/list
            # val = val.unsqueeze(0).repeat(emb.shape[0], 1) # Repeats the *same* label for the whole batch

            with torch.cuda.amp.autocast():
                y_pred = model(emb)  # forward
                loss = criterion(y_pred, val)  # loss

            # backward
            optimizer.zero_grad()
            loss.backward()

            # Optional: Gradient Clipping (standard torch method)
            # if getattr(args, 'grad_clip_norm', None):
            #    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            optimizer.step()

            # --- Step Scheduler (Conditional) ---
            if scheduler is not None:
                scheduler.step()

            # --- Eval/Save/Log ---
            # !! IMPORTANT: Logging via wandb.log() should happen INSIDE ModelWrapper !!
            # !! You need to modify ModelWrapper in utils.py to call wandb.log() !!
            progress.update(args.batch)  # Use actual batch size used by loader
            wrapper.log_step(loss.data.item(), progress.n)
            # wrapper.log_point(loss.data.item(), batch.get("index")) # Maybe log less often?
            if args.nsave > 0 and (progress.n // args.batch) % (args.nsave // args.batch) == 0 and progress.n > 0:
                # Save based on number of batches processed, avoid saving at step 0
                wrapper.save_model(step=progress.n)
            if progress.n >= args.steps:
                break
    progress.close()

    # --- Final Save and Cleanup ---
    wrapper.save_model(epoch="")  # final save
    # wrapper.enum_point() # outliers - Maybe make this optional via arg?
    wrapper.close()  # Close CSV logger if used

    # --- Finish Wandb Run ---
    if wandb:
        wandb.finish()
