# Script for training on pre-computed feature sequences with bucketing
import argparse
import inspect
import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import wandb
import math
import torch.nn as nn
import torch.nn.functional as F
import traceback # Keep traceback
import collections # Keep collections
from transformers import AutoProcessor # <<< Added AutoProcessor import
from head_model import HeadModel
from losses import GHMC_Loss, FocalLoss # Keep these

try:
    from sageattention import sageattn
    F.scaled_dot_product_attention = sageattn
    print("!!! Successfully applied SageAttention monkey-patch !!!")
except ImportError:
    print("SageAttention not found or failed to import, using default F.scaled_dot_product_attention.")
except Exception as e:
    print(f"Error applying SageAttention monkey-patch: {e}")

# --- Local Imports ---
# Assuming utils.py is in the same directory or accessible
from utils import (
    ModelWrapper, parse_and_load_args, write_config,
    load_optimizer_state, load_scheduler_state, load_scaler_state,
    run_validation_sequences, SAVE_FOLDER  # Add load helpers if needed here
)

# <<< Import Feature Sequence Dataset stuff >>>
try:
    from sequence_dataset import FeatureSequenceDataset, BucketBatchSampler, collate_sequences
    FEATURE_DATASET_AVAILABLE = True
except ImportError:
    print("Error: sequence_dataset.py not found or failed to import required classes.")
    FEATURE_DATASET_AVAILABLE = False
    # exit(1) # Or handle gracefully later

# --- Attempt to import our custom optimizers ---
try:
    # Make sure the optimizer directory is in the path if needed
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    optimizer_dir_path = os.path.join(current_script_dir, 'optimizer')
    if os.path.isdir(optimizer_dir_path) and current_script_dir not in sys.path:
         sys.path.insert(0, current_script_dir)
    from optimizer import OPTIMIZERS, SCHEDULERS
    print(f"Successfully imported custom optimizers: {list(OPTIMIZERS.keys())}")
    print(f"Successfully imported custom schedulers: {list(SCHEDULERS.keys())}")
    custom_modules_available = True
except ImportError as e:
    print(f"Warning: Custom optimizer/scheduler import failed ({e}). Check optimizer/__init__.py. Standard torch modules only.")
    OPTIMIZERS = {}
    SCHEDULERS = {}
    custom_modules_available = False

# --- Global Settings ---
# LOG_EVERY_N = 1 # How often to log metrics and check validation
# VALIDATE_EVERY_N = 50 # Run validation less often
TARGET_DEV = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TF32 support enabled for CUDA operations.")
else:
    print("CUDA not available, TF32 settings not applied.")
torch.backends.cudnn.benchmark = True
# Seed setting moved to main after args parsing
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
        # Ensure args.name is set
        if not hasattr(args, 'name') or not args.name:
             args.name = f"{getattr(args, 'base', 'model')}-{getattr(args, 'rev', 'rev')}"

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
    """Sets up FeatureSequenceDataset and DataLoader with BucketBatchSampler."""
    # --- Configuration Checks ---
    if not hasattr(args, 'data_mode') or args.data_mode != 'features':
         exit(f"Error: Incorrect data_mode ('{getattr(args, 'data_mode', None)}') found in args. Expected 'features' for train_features.py.")
    if not hasattr(args, 'feature_dir_name'):
         exit("Error: Missing 'feature_dir_name' argument/config key specifying the feature directory.")
    if not hasattr(args, 'data_root'): args.data_root = "data"

    feature_root_dir = os.path.join(args.data_root, args.feature_dir_name)
    print(f"Setting up FeatureSequenceDataset from: {feature_root_dir}")

    # --- Instantiate Dataset ---
    try:
        dataset = FeatureSequenceDataset(
            feature_root_dir=feature_root_dir,
            validation_split_count=args.val_split_count,
            seed=args.seed
        )
        args.num_labels = dataset.num_labels # Update args
        print(f"DEBUG: Updated args.num_labels from FeatureSequenceDataset: {args.num_labels}")
    except Exception as e: print(f"Error creating FeatureSequenceDataset: {e}"); exit(1)
    if len(dataset) == 0: exit("Error: Training dataset is empty.")

    # --- Create Samplers ---
    train_drop_last = getattr(args, 'train_drop_last', True)
    train_sampler = BucketBatchSampler(
        buckets=dataset.buckets,
        batch_size=args.batch,
        drop_last=train_drop_last,
        seed=args.seed,
        shuffle_buckets = True,  # <<< Change this to False >>>
        shuffle_within_bucket = True  # Keep shuffling within bucket
    )

    val_loader = None
    if dataset.val_indices:
        val_buckets = collections.defaultdict(list)
        for idx in dataset.val_indices:
             try: meta = dataset.metadata[idx]; val_buckets[meta['seq_len']].append(idx)
             except IndexError: print(f"Warning: Invalid validation index {idx} during bucket creation.")

        if any(val_buckets.values()):
            val_sampler = BucketBatchSampler(
                buckets=val_buckets,
                batch_size=args.batch,
                drop_last=False,
                seed=args.seed,
                shuffle_buckets=False,
                shuffle_within_bucket=False
            )
            val_loader = DataLoader(
                dataset, # Pass main dataset, sampler selects val indices
                batch_sampler=val_sampler,
                num_workers=args.num_workers,
                collate_fn=collate_sequences,
                persistent_workers=True if args.num_workers > 0 else False,
                prefetch_factor=getattr(args, 'prefetch_factor', 2) if args.num_workers > 0 else None

            )
            print(
                f"Created validation loader with {len(val_sampler)} batches ({len(dataset.val_indices)} samples). Persistent Workers: {args.num_workers > 0}")
        else: print("Validation split requested, but no validation samples found in buckets.")
    else: print("No validation split requested or validation data available.")

    # --- Create Training Loader ---
    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
        persistent_workers = True if args.num_workers > 0 else False,
        prefetch_factor = getattr(args, 'prefetch_factor', 2) if args.num_workers > 0 else None
    )
    print(f"Created training loader with {len(train_sampler)} batches ({len(dataset.train_indices)} samples). Persistent Workers: {args.num_workers > 0}")

    args.steps_per_epoch = len(train_sampler) # Update steps per epoch
    print(f"DEBUG: Updated args.steps_per_epoch based on BucketBatchSampler: {args.steps_per_epoch}")

    return dataset, train_loader, val_loader

def setup_model_criterion(args, dataset):
    """Sets up the HeadModel and criterion for feature sequence training."""
    print("DEBUG setup_model_criterion: Setting up HeadModel and criterion...")

    # --- Common Setup ---
    loss_function_name = getattr(args, 'loss_function')
    # num_labels should have been updated by setup_dataloaders based on the dataset
    num_classes = getattr(args, 'num_labels', None)
    if num_classes is None:
        exit("Error: num_labels not set by dataset loader. Cannot determine model output size.")
    if args.arch == 'score' and num_classes != 1:
        print(f"Warning: arch is 'score' but dataset seems to have {num_classes} labels? Forcing num_classes=1.")
        num_classes = 1
    if args.arch == 'class' and num_classes == 0:
         exit("Error: arch is 'class' but dataset found 0 labels.")
    if args.arch == 'class' and num_classes == 1:
         print("Warning: arch is 'class' but dataset only has 1 label (e.g., only '0' folder). Consider adding more classes or using 'score' arch.")
         # Defaulting to 2 classes for binary classification if only one label found might be confusing.
         # Let's proceed with num_classes=1 for BCE loss, but CE/Focal/NLL/GHM will fail.
         # num_classes = 2 # Force to 2? Or let it fail later? Let's keep 1 for now.

    args.num_classes = num_classes # Store the final determined number of classes back to args

    criterion = None
    class_weights_tensor = None
    if args.arch == "class" and hasattr(args, 'weights') and args.weights:
         if len(args.weights) == num_classes:
              try:
                  class_weights_tensor = torch.tensor(args.weights, device=TARGET_DEV, dtype=torch.float32)
                  print(f"DEBUG: Using class weights: {args.weights}")
              except Exception as e:
                   print(f"Warning: Failed to create tensor from class weights: {e}. Ignoring weights.")
         else:
              print(f"Warning: Length of weights in config ({len(args.weights)}) != num_classes ({num_classes}). Ignoring weights.")


    # --- Determine Criterion ---
    # This logic remains mostly the same, but uses the final num_classes
    # It also validates against the model's output_mode required by the loss
    head_output_mode = getattr(args, 'head_output_mode', 'linear').lower() # Get head output mode


    if loss_function_name == 'l1':
        print("DEBUG: Setting criterion to L1Loss.")
        criterion = nn.L1Loss(reduction='mean')
        if num_classes != 1: print(f"Warning: L1Loss typically used with num_classes=1, but found {num_classes}.")
        if head_output_mode not in ['tanh_scaled', 'sigmoid', 'linear']: print(f"Warning: L1Loss typically paired with scaled/linear output, got '{head_output_mode}'.")

    elif loss_function_name == 'mse':
        print("DEBUG: Setting criterion to MSELoss.")
        criterion = nn.MSELoss(reduction='mean')
        if num_classes != 1: print(f"Warning: MSELoss typically used with num_classes=1, but found {num_classes}.")
        if head_output_mode not in ['tanh_scaled', 'linear', 'sigmoid']: print(f"Warning: MSELoss typically paired with linear/scaled output, got '{head_output_mode}'.")

    elif loss_function_name == 'focal':
        print("DEBUG: Setting criterion to FocalLoss.")
        if num_classes <= 1: exit(f"Error: FocalLoss requires num_classes > 1, but found {num_classes}.")
        criterion = FocalLoss(gamma=getattr(args, 'focal_loss_gamma', 2.0))
        if head_output_mode != 'linear':
             print(f"Warning: FocalLoss usually expects output_mode='linear', but got '{head_output_mode}'. Training with non-logit input.")
             # Allow non-linear input, but loss calculation might be suboptimal.

    elif loss_function_name == 'crossentropy':
        print("DEBUG: Setting criterion to CrossEntropyLoss.")
        if num_classes <= 1: exit(f"Error: CrossEntropyLoss requires num_classes > 1, but found {num_classes}.")
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        if head_output_mode != 'linear':
             print(f"Warning: CrossEntropyLoss usually expects output_mode='linear', but got '{head_output_mode}'. Training with non-logit input.")

    elif loss_function_name == 'bce':
        print("DEBUG: Setting criterion to BCEWithLogitsLoss.")
        # BCEWithLogitsLoss *can* handle multi-label if target is multi-label float,
        # but for single-label classification, num_classes=1 is standard.
        if num_classes != 1:
            print(f"Warning: BCEWithLogitsLoss selected, but num_classes={num_classes}. Ensure targets are correctly formatted (e.g., multi-label floats) or consider using CrossEntropyLoss.")
        criterion = nn.BCEWithLogitsLoss(weight=class_weights_tensor if num_classes > 1 else None) # Weight applied per element for multi-label
        if head_output_mode != 'linear':
             print(f"Warning: BCEWithLogitsLoss expects output_mode='linear', but got '{head_output_mode}'. Training with non-logit input.")

    elif loss_function_name == 'nll':
        print("DEBUG: Setting criterion to NLLLoss (expects LogSoftmax input).")
        if num_classes <= 1: exit(f"Error: NLLLoss requires num_classes > 1, but found {num_classes}.")
        criterion = nn.NLLLoss(weight=class_weights_tensor)
        if head_output_mode != 'linear':
             # NLL *requires* log-probabilities. If the model isn't outputting logits, this won't work correctly.
             exit(f"Error: loss_function='nll' selected, but model output_mode is '{head_output_mode}'. Must be 'linear' for NLLLoss (LogSoftmax applied in train loop).")

    elif loss_function_name == 'ghm':
        print("DEBUG: Setting criterion to GHMC_Loss.")
        if num_classes <= 1: exit(f"Error: GHMC_Loss currently requires num_classes > 1, but found {num_classes}.")
        ghm_bins = getattr(args, 'ghm_bins', 10)
        ghm_momentum = getattr(args, 'ghm_momentum', 0.75)
        criterion = GHMC_Loss(bins=ghm_bins, momentum=ghm_momentum, reduction='mean')
        if head_output_mode != 'linear':
             exit(f"Error: GHMC_Loss requires output_mode='linear', but got '{head_output_mode}'.")
        if class_weights_tensor is not None: print("Warning: Class weights specified but GHMC_Loss does not use them directly.")

    else:
        raise ValueError(f"Unknown loss_function '{loss_function_name}' specified in config.")

    if criterion is None:
        exit("Error: Criterion setup failed.")

    # --- Instantiate the HeadModel ---
    print(f"DEBUG: Instantiating HeadModel (Classes: {num_classes}, Output Mode: {head_output_mode})")
    try:
        # Get necessary args (feature dim should be known for the precomputed features)
        # We need to add 'head_features' to args/config, e.g., 1024 for AIMv2
        head_features = getattr(args, 'head_features', None)
        if head_features is None: exit("Error: 'head_features' (input dimension for head) not specified in args/config.")

        model = HeadModel(
            features=args.head_features,
            num_classes=num_classes,
            pooling_strategy=getattr(args, 'pooling_strategy', 'attn'),
            hidden_dim=getattr(args, 'head_hidden_dim', 1024),
            num_res_blocks=getattr(args, 'head_num_res_blocks', 3),
            dropout_rate=getattr(args, 'head_dropout_rate', 0.2),
            output_mode=getattr(args, 'head_output_mode', 'linear'), # Get head output mode
            attn_pool_heads=getattr(args, 'attn_pool_heads', 16),
            attn_pool_dropout=getattr(args, 'attn_pool_dropout', 0.2)
        )
    except Exception as e:
        print(f"Error details during HeadModel instantiation: {e}")
        traceback.print_exc()
        exit(f"Error instantiating HeadModel.")

    model.to(TARGET_DEV)
    print("HeadModel and criterion setup complete.")
    return model, criterion

# Version 2.5.0: Conditional parameter selection for optimizer
def setup_optimizer_scheduler(args, model):
    """
    Sets up the optimizer and scheduler based on args.
    Dynamically loads custom optimizers/schedulers from OPTIMIZERS/SCHEDULERS dicts.
    Uses inspect to gather relevant arguments from args object.
    Selects parameters to optimize based on end-to-end mode and freezing settings.
    """
    optimizer = None
    scheduler = None
    is_schedule_free = False
    optimizer_name = getattr(args, 'optimizer', 'AdamW').lower()

    print(f"Attempting to setup optimizer: {optimizer_name}")

    # --- Determine Parameters to Optimize ---
    params_to_optimize = None
    if getattr(args, 'is_end_to_end', False) and getattr(args, 'freeze_base_model', True):
        print("DEBUG Optimizer: Base model is frozen, optimizing HEAD parameters only.")
        # Access parameters of the head module (assuming it's named 'head')
        if hasattr(model, 'head') and isinstance(model.head, nn.Module):
             params_to_optimize = model.head.parameters()
             # Double-check if any parameters were found in the head
             if not list(model.head.parameters()):
                 print("Warning: model.head found, but it has no parameters to optimize!")
                 # Fallback to optimizing everything? Or error? Let's warn and optimize all for now.
                 params_to_optimize = model.parameters()
        else:
             # Error if we expect a frozen base but can't find the head
             exit("Error: Cannot find 'model.head' submodule to optimize when base model is frozen.")
    else:
        # Optimize all parameters if not E2E, or if E2E but base is not frozen
        if getattr(args, 'is_end_to_end', False):
             print("DEBUG Optimizer: End-to-end model, base is NOT frozen. Optimizing ALL trainable model parameters.")
        else:
             print("DEBUG Optimizer: Embedding-based model. Optimizing ALL trainable model parameters.")
        params_to_optimize = model.parameters()

    # Check if any parameters were selected
    # Convert generator to list to check emptiness
    param_list_for_check = list(params_to_optimize)
    if not param_list_for_check:
         exit("Error: No parameters selected for optimization!")
    # Need to use the original generator for the optimizer constructor if not checking length
    # Re-assign if we consumed the generator by converting to list
    params_to_optimize = iter(param_list_for_check) # Or get it again if model structure is simple
    # Safer: Just get the generator again if needed
    if getattr(args, 'is_end_to_end', False) and getattr(args, 'freeze_base_model', True):
         params_to_optimize = model.head.parameters()
    else:
         params_to_optimize = model.parameters()
    # --- End Parameter Determination ---


    # --- Dynamic Optimizer Loading ---
    if optimizer_name in OPTIMIZERS:
        optimizer_class = OPTIMIZERS[optimizer_name]
        print(f"Found optimizer class: {optimizer_class.__name__}")

        try:
            # Inspect the optimizer's __init__ signature
            sig = inspect.signature(optimizer_class.__init__)
            available_params = sig.parameters.keys()
            # Gather potential kwargs from args (same logic as before)
            potential_kwargs = {}
            args_dict = vars(args)
            for param_name in available_params:
                if param_name in ['self', 'params', 'model', 'args', 'kwargs']: continue
                if param_name in args_dict and args_dict[param_name] is not None:
                    # (Type conversion logic remains the same as before)
                    expected_type = sig.parameters[param_name].annotation
                    default_value = sig.parameters[param_name].default
                    value = args_dict[param_name]
                    try:
                        if expected_type == inspect.Parameter.empty: potential_kwargs[param_name] = value
                        elif expected_type == bool: potential_kwargs[param_name] = bool(value)
                        elif expected_type == int: potential_kwargs[param_name] = int(value)
                        elif expected_type == float: potential_kwargs[param_name] = float(value)
                        elif expected_type == str: potential_kwargs[param_name] = str(value)
                        elif expected_type == tuple or expected_type == list or \
                             (hasattr(expected_type, '__origin__') and expected_type.__origin__ in [tuple, list]):
                            if isinstance(value, (list, tuple)):
                                inner_type = float
                                if hasattr(expected_type, '__args__') and expected_type.__args__: inner_type = expected_type.__args__[0]
                                converted_list = [inner_type(v) for v in value]
                                potential_kwargs[param_name] = tuple(converted_list) if expected_type == tuple or expected_type.__origin__ == tuple else converted_list
                            else: print(f"Warning: Arg {param_name} expects {expected_type} but got {type(value)}. Skipping.")
                        else: potential_kwargs[param_name] = value
                    except (ValueError, TypeError) as e_type:
                         print(f"Warning: Could not convert arg '{param_name}' (value: {value}) to expected type {expected_type}. Error: {e_type}. Using default or skipping.")
                         if default_value != inspect.Parameter.empty: potential_kwargs[param_name] = default_value

            print(f"  Instantiating {optimizer_class.__name__} with args: {potential_kwargs}")
            # <<< Pass the correctly selected parameters >>>
            optimizer = optimizer_class(params_to_optimize, **potential_kwargs)

            # Check if it's a schedule-free optimizer
            if "schedulefree" in optimizer_name:
                is_schedule_free = True
                print("  Detected schedule-free optimizer.")

        except Exception as e:
            print(f"ERROR: Failed to instantiate optimizer '{optimizer_name}' dynamically: {e}")
            import traceback; traceback.print_exc()
            print("  Falling back to AdamW.")
            optimizer_name = 'adamw'
            optimizer = None
            is_schedule_free = False

    # --- Fallback / Default Optimizer ---
    if optimizer is None:
        if optimizer_name != 'adamw':
             print(f"Warning: Optimizer '{optimizer_name}' not found or failed. Falling back to AdamW.")
        optimizer_name = 'adamw'
        # Use defaults from args for AdamW
        adamw_kwargs = {
             'lr': float(getattr(args, 'lr', 1e-4)),
             'betas': tuple(getattr(args, 'betas', (0.9, 0.999))),
             'weight_decay': float(getattr(args, 'weight_decay', 0.0)),
             'eps': float(getattr(args, 'eps', 1e-8)),
        }
        print(f"Instantiating torch.optim.AdamW with args: {adamw_kwargs}")
        # <<< Pass the correctly selected parameters >>>
        optimizer = torch.optim.AdamW(params_to_optimize, **adamw_kwargs)
        is_schedule_free = False
    # --- End Fallback ---

    # --- Scheduler Setup ---
    if not is_schedule_free:
        # (Scheduler loading logic remains unchanged - operates on the created `optimizer`)
        scheduler_name = getattr(args, 'scheduler_name', 'CosineAnnealingLR').lower()
        print(f"Attempting to setup scheduler: {scheduler_name}")
        scheduler_class = None
        if scheduler_name in SCHEDULERS:
            scheduler_class = SCHEDULERS[scheduler_name]
            print(f"Found custom scheduler class: {scheduler_class.__name__}")
            try:
                # (Inspect and gather kwargs logic remains the same)
                sig = inspect.signature(scheduler_class.__init__)
                available_params = sig.parameters.keys()
                scheduler_kwargs = {}
                args_dict = vars(args)
                for param_name in available_params:
                    if param_name in ['self', 'optimizer', 'last_epoch', 'args', 'kwargs']: continue
                    arg_key = f"scheduler_{param_name}"
                    if arg_key in args_dict and args_dict[arg_key] is not None:
                        expected_type = sig.parameters[param_name].annotation
                        default_value = sig.parameters[param_name].default
                        value = args_dict[arg_key]
                        try:
                            # (Type conversion logic remains the same)
                            if expected_type == inspect.Parameter.empty: scheduler_kwargs[param_name] = value
                            elif expected_type == bool: scheduler_kwargs[param_name] = bool(value)
                            elif expected_type == int: scheduler_kwargs[param_name] = int(value)
                            elif expected_type == float: scheduler_kwargs[param_name] = float(value)
                            elif expected_type == str: scheduler_kwargs[param_name] = str(value)
                            else: scheduler_kwargs[param_name] = value
                        except (ValueError, TypeError) as e_type:
                             print(f"Warning: Could not convert scheduler arg '{param_name}' (value: {value}) to {expected_type}. Error: {e_type}. Using default or skipping.")
                             if default_value != inspect.Parameter.empty: scheduler_kwargs[param_name] = default_value

                print(f"  Instantiating {scheduler_class.__name__} with args: {scheduler_kwargs}")
                scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            except Exception as e:
                 print(f"ERROR: Failed to instantiate custom scheduler '{scheduler_name}': {e}")
                 print("  Falling back to standard PyTorch schedulers.")
                 scheduler = None # Reset to trigger fallback

        # --- Fallback to Standard PyTorch Schedulers ---
        if scheduler is None:
            scheduler_type = None
            # Check args.cosine (legacy) or specific names
            is_cosine = getattr(args, 'cosine', None) # Check if cosine flag exists
            has_warmup = getattr(args, 'warmup_steps', 0) > 0

            if is_cosine is True or scheduler_name in ['cosineannealinglr', 'cosine']:
                 scheduler_type = 'cosine'
            elif has_warmup and scheduler_name == 'linearlr': # Only use linear if warmup steps > 0
                 scheduler_type = 'warmup'
            elif scheduler_name in ['none', None]:
                 scheduler_type = None # Explicitly no scheduler
            elif scheduler_name not in SCHEDULERS: # If name specified but not found
                 print(f"Warning: Scheduler '{scheduler_name}' not found in custom SCHEDULERS or standard options. No scheduler used.")

            # Instantiate standard scheduler
            if scheduler_type == 'cosine':
                print("Using standard torch.optim.lr_scheduler.CosineAnnealingLR.")
                t_max_steps = getattr(args, 'scheduler_t_max', args.max_train_steps) # Default T_max to total steps
                eta_min = getattr(args, 'scheduler_eta_min', 0)
                print(f"  Setting T_max = {t_max_steps}, eta_min = {eta_min}")
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(t_max_steps), eta_min=float(eta_min))
            elif scheduler_type == 'warmup':
                print("Using standard torch.optim.lr_scheduler.LinearLR for warmup.")
                warmup_iters = int(args.warmup_steps)
                print(f"  Setting total_iters = {warmup_iters} for LinearLR.")
                # Typical usage: Warm up from low LR to the optimizer's initial LR
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
            elif scheduler is None: # Handles cases where no match found or name was 'none'
                 print("No matching standard or custom scheduler specified. Proceeding without scheduler.")

    else: # Handle Schedule-Free Case
        print("Using a schedule-free optimizer, no scheduler will be used.")
    # --- End Scheduler Setup ---

    print("Optimizer and Scheduler setup complete.")
    return optimizer, scheduler, is_schedule_free

# ================================================
#        Checkpoint Loading (Epoch-Aware)
# ================================================
# Version 2.0.0: Loads epoch and global step
def load_checkpoint(args, model, optimizer, scheduler, scaler):
    """Loads state from checkpoint if args.resume is provided."""
    start_epoch = 0
    initial_global_step = 0 # Tracks overall steps completed

    if args.resume:
        print(f"Resuming from {args.resume}")
        if not os.path.isfile(args.resume):
             print(f"Error: Resume file not found: {args.resume}"); exit(1)

        # --- Load Base Model ---
        # Use strict=False for potentially changing model structures? Or handle errors better?
        # Let's try strict=True first and see if errors occur when resuming across E2E/Embedding changes.
        try:
             print("Loading model state_dict...")
             model.load_state_dict(load_file(args.resume), strict=True)
             print("Model state loaded successfully.")
        except Exception as e:
            print(f"Error loading model state_dict from {args.resume}: {e}")
            print("Attempting to load with strict=False (may indicate model structure mismatch)...")
            try:
                 model.load_state_dict(load_file(args.resume), strict=False)
                 print("Model state loaded with strict=False. Check for missing/unexpected keys.")
            except Exception as e_nonstrict:
                 print(f"Error loading model state_dict even with strict=False: {e_nonstrict}")
                 exit(1)

        # --- Load Aux Files ---
        base_path = os.path.splitext(args.resume)[0]
        optim_path = f"{base_path}.optim"
        sched_path = f"{base_path}.sched"
        scaler_path = f"{base_path}.scaler"
        state_path = f"{base_path}.state" # File storing epoch/global_step

        # Use helper functions for loading optimizer, scheduler, scaler
        load_optimizer_state(optimizer, optim_path, TARGET_DEV)
        load_scheduler_state(scheduler, sched_path, TARGET_DEV)
        load_scaler_state(scaler, scaler_path, TARGET_DEV)

        # --- Load Training State (Epoch/Step) ---
        if os.path.exists(state_path):
             try:
                  train_state = torch.load(state_path, map_location='cpu')
                  start_epoch = train_state.get('epoch', 0) # Epoch just completed
                  initial_global_step = train_state.get('global_step', 0) # Global step reached
                  print(f"Loaded training state: Resuming from start of Epoch {start_epoch + 1}, Global Step {initial_global_step}")
             except Exception as e:
                  print(f"Warning: Could not load training state file '{state_path}': {e}. Resuming from epoch 0, step 0.")
                  start_epoch, initial_global_step = 0, 0
        else:
             print(f"Warning: Training state file '{state_path}' not found. Resuming from epoch 0, step 0.")
             start_epoch, initial_global_step = 0, 0
    else:
        print("Starting new training run.")

    return start_epoch, initial_global_step
# ================================================

# ================================================
#        Main Training Loop (for Feature Sequences)
# ================================================
# Version 1.0.0 (Feature Sequence Mode)
def train_loop(args, model, criterion, optimizer, scheduler, scaler,
               train_loader, val_loader, wrapper, start_epoch, initial_global_step,
               enabled_amp, amp_dtype, is_schedule_free):
    """
    Runs the main training loop for pre-computed feature sequences with bucketing.
    Expects batches with stacked 'sequence' and 'label' tensors.
    """
    # --- Initial Setup ---
    if not hasattr(args, 'num_train_epochs') or not hasattr(args, 'steps_per_epoch') or not hasattr(args, 'max_train_steps'):
         print("ERROR: num_train_epochs, steps_per_epoch, or max_train_steps missing from args.")
         return

    model.train() # Ensure model (HeadModel) is in train mode
    if hasattr(optimizer, 'train') and callable(optimizer.train):
        try: optimizer.train()
        except Exception as e: print(f"Warning: Error calling optimizer.train(): {e}")

    total_steps_to_run = args.max_train_steps
    total_epochs_to_run = args.num_train_epochs
    # steps_per_epoch should be set correctly by setup_dataloaders using sampler length
    steps_per_epoch = getattr(args, 'steps_per_epoch', len(train_loader)) # Fallback

    log_every_n = getattr(args, 'log_every_n', 10) # Logging frequency
    validate_every_n = getattr(args, 'validate_every_n', steps_per_epoch) # Validation frequency

    print(f"Starting Feature Sequence training loop. Target Epochs: {total_epochs_to_run}, Target Steps: {total_steps_to_run}")
    print(f"Steps per epoch (batches): {steps_per_epoch}. Logging every {log_every_n}. Validating every {validate_every_n}.")

    global_step = initial_global_step
    best_eval_loss = float('inf') # Track best loss locally
    last_eval_loss_val = float('nan') # Track last validation result

    progress_bar = tqdm(initial=initial_global_step, total=total_steps_to_run, desc="Training Head", unit="step", dynamic_ncols=True)

    # Use wrapper's loss buffer if needed, or simple step loss
    accumulated_loss_for_log = torch.tensor(0.0, device=TARGET_DEV)
    accumulation_steps_for_log = 0

    # --- Outer Epoch Loop ---
    try:
        for epoch in range(start_epoch, total_epochs_to_run):
            if global_step >= total_steps_to_run: break

            # Set epoch for sampler (important for shuffling)
            if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
                 train_loader.batch_sampler.set_epoch(epoch)
                 print(f"\n--- Starting Epoch {epoch + 1}/{total_epochs_to_run} (Sampler Epoch Set) ---")
            else:
                 print(f"\n--- Starting Epoch {epoch + 1}/{total_epochs_to_run} ---")

            wrapper.current_epoch = epoch # Update wrapper's epoch tracking

            model.train() # Ensure model is in train mode
            if hasattr(optimizer, 'train') and callable(optimizer.train): optimizer.train()

            # --- Step Loop (Iterates through batches yielded by sampler) ---
            for i, batch_data in enumerate(train_loader):
                if global_step >= total_steps_to_run: break

                # Check for bad batches (e.g., from collate errors)
                if batch_data is None or not batch_data:
                     print(f"Warning: Skipping step {global_step} due to invalid batch_data.")
                     # Need to decide if a step counter increments here? Let's not increment global_step yet.
                     continue # Skip to next batch

                # --- Skipping Logic (If resuming) ---
                # Note: This might skip partial epochs slightly incorrectly if steps_per_epoch varies?
                # Safer to rely on global_step comparison.
                if global_step < initial_global_step:
                    global_step += 1
                    progress_bar.update(1)
                    continue
                # --- End Skipping Logic ---

                optimizer_stepped = False
                optimizer.zero_grad(set_to_none=True)

                # --- Process the Stacked Batch ---
                avg_loss_value_log = float('nan') # Default for this step
                loss = torch.tensor(float('nan'), device=TARGET_DEV)
                try:
                    # Extract data from batch dictionary
                    sequence_batch = batch_data.get('sequence')
                    label_batch = batch_data.get('label')

                    if sequence_batch is None or label_batch is None:
                         print(f"Warning: Batch missing 'sequence' or 'label' at step {global_step}. Skipping.")
                         global_step += 1; progress_bar.update(1); continue # Increment step as we consumed batch data

                    # Move data to device
                    sequence_batch = sequence_batch.to(TARGET_DEV) # Should be float16 from dataset
                    label_batch = label_batch.to(TARGET_DEV) # Should be long from dataset

                    current_batch_size = sequence_batch.size(0)

                    # Prepare target tensor shape/type based on criterion
                    target = label_batch # Assume label_batch is already correct shape [B]
                    if args.num_classes == 1: # Regression/BCE
                         target = target.to(dtype=torch.float32).view(current_batch_size, -1).squeeze(-1)
                    else: # Classification
                         target = target.to(dtype=torch.long).view(current_batch_size)

                    # --- Forward/Backward Pass ---
                    with torch.amp.autocast(device_type=TARGET_DEV, enabled=enabled_amp, dtype=amp_dtype):
                        # Pass sequence batch to HeadModel
                        y_pred = model(sequence_batch) # Input: [B, NumPatches, Features] -> Output: [B, Classes] or [B]

                        # Prepare prediction shape for loss
                        y_pred_for_loss = y_pred
                        if isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)) and args.num_classes == 1:
                            if y_pred.ndim > 1 and y_pred.shape[-1] == 1: y_pred_for_loss = y_pred.squeeze(-1) # Ensure [B]

                        # Calculate loss
                        loss_input = y_pred_for_loss.to(torch.float32) # Loss often expects float32 input
                        target_for_loss = target.to(loss_input.device)

                        if isinstance(criterion, nn.NLLLoss): loss_input = F.log_softmax(loss_input, dim=-1)

                        loss = criterion(loss_input, target_for_loss.long() if isinstance(criterion, (nn.CrossEntropyLoss, FocalLoss, nn.NLLLoss, GHMC_Loss)) else target_for_loss.float())

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/Inf loss detected at step {global_step}. Skipping backward/step.")
                        loss = None
                    else:
                        # Accumulate loss for logging average
                        accumulated_loss_for_log += loss.item() # Log item value
                        accumulation_steps_for_log += 1

                        # --- Backward Pass & Optimizer Step ---
                        scaler.scale(loss).backward() # Single backward for the whole batch
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer_stepped = True

                except Exception as e:
                    print(f"\nError processing batch step {global_step}: {e}")
                    import traceback; traceback.print_exc()
                    # Don't step optimizer if error occurred, but still increment step? Yes.
                    optimizer.zero_grad(set_to_none=True) # Zero grads to prevent issues on next step

                # --- Scheduler Step ---
                if scheduler is not None and not is_schedule_free and optimizer_stepped:
                    try: scheduler.step()
                    except Exception as e_sched: print(f"Warning: Error during scheduler step: {e_sched}")

                    # --- Accumulate Loss for Logging ---
                    if loss is not None and not math.isnan(loss.item()):  # Check loss was valid
                        accumulated_loss_for_log += loss.item()
                        accumulation_steps_for_log += 1

                    # --- Update Counters ---
                    global_step += 1
                    progress_bar.update(1)
                    wrapper.update_step(global_step)

                    # --- Frequent Logging & Postfix Update ---
                    if global_step % log_every_n == 0 and global_step > 0:
                        avg_loss_value_log = float('nan')
                        if accumulation_steps_for_log > 0:
                            avg_loss_value_log = accumulated_loss_for_log / accumulation_steps_for_log
                        accumulated_loss_for_log = torch.tensor(0.0, device=TARGET_DEV)  # Reset accumulator
                        accumulation_steps_for_log = 0

                        # <<< Pass last_eval_loss_val to log_main >>>
                        # log_main will log it to WandB if it's not NaN
                        wrapper.log_main(step=global_step, train_loss_batch=avg_loss_value_log,
                                         eval_loss=last_eval_loss_val)

                        # Update Progress Bar Postfix
                        lr = optimizer.param_groups[0]['lr']
                        postfix_dict = collections.OrderedDict()
                        postfix_dict["Epoch"] = epoch + 1
                        postfix_dict["AvgLoss"] = f"{avg_loss_value_log:.3e}" if not math.isnan(
                            avg_loss_value_log) else "N/A"
                        if not math.isnan(last_eval_loss_val): postfix_dict["LastEval"] = f"{last_eval_loss_val:.3e}"
                        postfix_dict["LR"] = f"{lr:.1e}"
                        progress_bar.set_postfix(ordered_dict=postfix_dict, refresh=False)

                    # --- Validation & Best Model Saving ---
                    if global_step % validate_every_n == 0 and global_step > 0:
                        print(f"\n--- Running Validation @ Step {global_step} ---")
                        eval_loss_val = float('nan')  # Reset for this run
                        if val_loader:
                            eval_loss_val = run_validation_sequences(model=model, val_loader=val_loader,
                                                                     criterion=criterion, device=TARGET_DEV,
                                                                     scaler=scaler)
                            last_eval_loss_val = eval_loss_val  # Update last known eval loss *after* running
                            model.train()  # Ensure model is back in train mode
                        else:
                            print("--- Validation skipped (no val_loader) ---")

                        # Optional: Log eval loss *again* specifically if needed for clarity in WandB
                        if wrapper.wandb_run and not math.isnan(eval_loss_val):
                            try:
                                wrapper.wandb_run.log({"eval/loss_on_val_step": eval_loss_val}, step=global_step)
                            except Exception as e:
                                print(f"Wandb eval log error: {e}")

                        if math.isnan(eval_loss_val):
                            print(f"Warning: Eval loss is NaN at Step {global_step}.")
                        else:
                            print(f"--- Validation Complete @ Step {global_step}: Eval Loss = {eval_loss_val:.4e} ---")

                        # Update postfix again with the new eval loss
                        # <<< Fetch avg loss again for potentially updated value >>>
                        avg_loss_postfix = avg_loss_value_log if not math.isnan(avg_loss_value_log) else "?"
                        lr = optimizer.param_groups[0]['lr']
                        postfix_dict = collections.OrderedDict()
                        postfix_dict["Epoch"] = epoch + 1
                        postfix_dict["AvgLoss"] = f"{avg_loss_postfix:.3e}"
                        if not math.isnan(eval_loss_val): postfix_dict[
                            "EvalLoss"] = f"{eval_loss_val:.3e}"  # Show *current* eval
                        postfix_dict["LR"] = f"{lr:.1e}"
                        progress_bar.set_postfix(ordered_dict=postfix_dict, refresh=False)  # Update postfix

                        # <<< Check / Save best model using the NEW eval_loss_val >>>
                        if not math.isnan(eval_loss_val) and eval_loss_val < best_eval_loss:
                            print(
                                f"New best val loss: {eval_loss_val:.4e} (was {best_eval_loss:.4e}). Saving best model...")
                            best_eval_loss = eval_loss_val  # <<< Update the local best loss tracker >>>
                            # <<< Tell wrapper about the new best loss BEFORE saving state >>>
                            wrapper.best_val_loss = best_eval_loss
                            # <<< Save with aux states - including updated best loss in .state file >>>
                            wrapper.save_model(step=global_step, epoch=epoch, suffix="_best_val", save_aux=False,
                                               args=args)
                        elif not math.isnan(eval_loss_val):
                            print(f"Validation loss {eval_loss_val:.4e} did not improve on best {best_eval_loss:.4e}")

                    # --- Periodic saving ---
                    if args.nsave > 0 and global_step % args.nsave == 0:
                        if global_step > initial_global_step:
                            print(f"\nSaving periodic checkpoint @ step {global_step}...")
                            wrapper.save_model(step=global_step, epoch=epoch, save_aux=False,
                                               args=args)  # Keep aux=False

                    if global_step >= total_steps_to_run: break
                # --- End Batch Loop ---
                if global_step >= total_steps_to_run: break

        # --- End Outer Epoch Loop ---

    except KeyboardInterrupt: print("\nTraining interrupted by user.")
    finally: progress_bar.close(); print(f"\nTraining loop finished. Reached Global Step: {global_step}")

# ================================================
#        Main Execution Block
# ================================================
# Version 1.0.0 (for train_features.py)
def main():
    # 1. Load Config / Args
    # <<< Add argparse for --config argument >>>
    parser = argparse.ArgumentParser(description="Train HeadModel on precomputed features.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    # Add --resume here too if needed as override? Or handle via config only?
    # Let's keep resume simple, require it in YAML if needed.
    # parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from (overrides config).')
    script_args = parser.parse_args()
    # <<< End add >>>

    # 1. Load Config / Args using the provided path
    # <<< Pass the config path from script_args >>>
    args = parse_and_load_args(config_path=script_args.config)
    # <<< Optionally merge resume override if we added it >>>
    # if script_args.resume: args.resume = script_args.resume

    print(f"Target device: {TARGET_DEV}")


    # 2. Setup Seed, Precision, WandB
    seed = getattr(args, 'seed', 0); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    print(f"Set random seed to: {seed}")
    amp_dtype, enabled_amp = setup_precision(args)
    wandb_run = setup_wandb(args)

    # 3. Setup Dataloaders (uses FeatureSequenceDataset, BucketBatchSampler)
    dataset, train_loader, val_loader = setup_dataloaders(args)

    # 4. Calculate final training duration (after sampler length is known)
    # args.steps_per_epoch is updated inside setup_dataloaders now
    if args.steps_per_epoch == 0: exit("Error: steps_per_epoch is zero (sampler length is zero).")
    if args.max_train_steps is not None and args.max_train_steps > 0:
        args.num_train_epochs = math.ceil(args.max_train_steps / args.steps_per_epoch)
    elif args.max_train_epochs is not None and args.max_train_epochs > 0:
        args.num_train_epochs = args.max_train_epochs
        args.max_train_steps = args.num_train_epochs * args.steps_per_epoch
    else: # Default if neither specified
        args.max_train_steps = 10000
        args.num_train_epochs = math.ceil(args.max_train_steps / args.steps_per_epoch)

    # 5. Setup Model (HeadModel) & Criterion
    model, criterion = setup_model_criterion(args, dataset)

    # 6. Setup Optimizer, Scheduler (passing HeadModel)
    optimizer, scheduler, is_schedule_free = setup_optimizer_scheduler(args, model)
    scaler = torch.amp.GradScaler(device=TARGET_DEV, enabled=(enabled_amp and TARGET_DEV == 'cuda'))

    # 7. Load Checkpoint (loading HeadModel state)
    start_epoch, initial_global_step = load_checkpoint(args, model, optimizer, scheduler, scaler)

    # 8. Write Final Config
    print("\n--- Final Calculated Args ---")
    for k, v in sorted(vars(args).items()): print(f"  {k}: {v}")
    print("--------------------------\n")
    write_config(args) # From utils.py

    # 9. Instantiate Wrapper (passing HeadModel)
    wrapper = ModelWrapper(
        name=args.name, model=model, optimizer=optimizer,
        criterion=criterion, # Pass criterion for reference if needed? Maybe not.
        scheduler=scheduler, device=TARGET_DEV, scaler=scaler,
        wandb_run=wandb_run, num_labels=args.num_labels
    )
    # Restore best loss if loaded from checkpoint state
    if initial_global_step > 0 and os.path.exists(f"./{SAVE_FOLDER}/{args.name}_s{initial_global_step}.state"): # Check state file from step
         try:
              train_state = torch.load(f"./{SAVE_FOLDER}/{args.name}_s{initial_global_step}.state", map_location='cpu')
              wrapper.best_val_loss = train_state.get('best_val_loss', float('inf'))
              print(f"Restored best_val_loss from checkpoint state: {wrapper.best_val_loss}")
         except Exception as e_state: print(f"Could not load best_val_loss from state: {e_state}")


    # 10. Run Training Loop (Feature Sequence version)
    try:
        train_loop(args, model, criterion, optimizer, scheduler, scaler,
                   train_loader, val_loader, wrapper,
                   start_epoch, initial_global_step,
                   enabled_amp, amp_dtype, is_schedule_free)
    finally:
        # 11. Final Save & Cleanup
        print(f"Saving final model...")
        final_step = wrapper.get_current_step()
        wrapper.save_model(step=final_step, epoch="final", suffix="_final", save_aux=False, args=args)
        wrapper.close()
        if wandb_run: wandb_run.finish()
        print("Training script finished.")

if __name__ == "__main__":
    main()