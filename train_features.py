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
    from sequence_dataset import (FeatureSequenceDataset, collate_sequences, ValidationSubDatasetFeaturesPadded,
                                  ValidationSubDatasetFeaturesPadded)
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
    """Sets up FeatureSequenceDataset and DataLoader.
       v1.1.0: Uses standard DataLoader with shuffle=True, pooling done in dataset.
    """
    feature_root_dir = os.path.join(args.data_root, args.feature_dir_name)
    print(f"Setting up FeatureSequenceDataset (pooling in __getitem__) from: {feature_root_dir}")

    # --- Instantiate Dataset ---
    try:
        # Pass preload args if needed (dataset handles it internally)
        dataset = FeatureSequenceDataset(
            feature_root_dir=feature_root_dir,
            validation_split_count=args.val_split_count,
            seed=args.seed,
            preload=getattr(args, 'preload_data', False), # Check if preload is in args
            preload_limit_gb=getattr(args, 'preload_limit_gb', 30.0) # Get limit from args if exists
        )
        args.num_labels = dataset.num_labels
        print(f"DEBUG: Updated args.num_labels from FeatureSequenceDataset: {args.num_labels}")
    except Exception as e: print(f"Error creating FeatureSequenceDataset: {e}"); exit(1)
    if len(dataset.train_indices) == 0: exit("Error: Training dataset partition is empty.") # Check train_indices specifically

    # --- REMOVE BucketBatchSampler Creation ---
    # train_sampler = BucketBatchSampler(...) # Remove this

    # --- Setup Validation Loader (Uses standard DataLoader now via get_validation_loader) ---
    val_loader = dataset.get_validation_loader(
         batch_size=args.batch, # Or a different validation batch size if needed
         num_workers=args.num_workers
    )
    # get_validation_loader now handles creating the sub-dataset and standard DataLoader

    # --- Create Standard Training Loader ---
    print(f"DEBUG: Creating standard DataLoader for training...")
    train_loader = DataLoader(
        dataset, # Use the main dataset instance
        # <<< --- MODIFICATIONS --- >>>
        batch_size=args.batch,    # Use regular batch_size argument
        shuffle=True,           # Shuffle the training data each epoch
        # batch_sampler=train_sampler, # REMOVE batch_sampler argument
        # <<< --- END MODIFICATIONS --- >>>
        num_workers=args.num_workers,
        collate_fn=collate_sequences, # Keep using our collate function (handles None)
        persistent_workers = True if args.num_workers > 0 else False,
        prefetch_factor = getattr(args, 'prefetch_factor', 2) if args.num_workers > 0 else None,
        drop_last=getattr(args, 'train_drop_last', True) # Keep drop_last if needed
    )
    print(f"Created training loader with shuffle=True ({len(dataset.train_indices)} samples).") # Use dataset len

    # --- Update steps_per_epoch Calculation ---
    # Use the length of the training partition of the dataset
    num_train_samples = len(dataset.train_indices)
    if num_train_samples == 0: exit("Error: No training samples available after split.")
    # Calculate steps based on standard DataLoader logic
    steps_per_epoch = num_train_samples // args.batch
    if not getattr(args, 'train_drop_last', True) and num_train_samples % args.batch != 0:
        steps_per_epoch += 1 # Add step for the last partial batch if not dropping
    args.steps_per_epoch = steps_per_epoch
    print(f"DEBUG: Updated args.steps_per_epoch based on standard DataLoader: {args.steps_per_epoch}")

    return dataset, train_loader, val_loader

# <<< REWRITTEN setup_model_criterion v1.1.0 >>>
def setup_model_criterion(args, dataset):
    """
    Sets up the HeadModel and criterion based on args config,
    validating against the dataset's discovered labels.
    """
    print("DEBUG setup_model_criterion v1.1.0: Setting up HeadModel and criterion...")

    # --- 1. Determine Intended num_classes from Config ---
    config_loss_function = getattr(args, 'loss_function', None)
    config_arch = getattr(args, 'arch', 'class') # Default to class if missing
    intended_num_classes = None
    loss_requires_num_classes_1 = config_loss_function in ['bce', 'l1', 'mse']
    loss_requires_num_classes_multi = config_loss_function in ['crossentropy', 'focal', 'nll', 'ghm']

    if config_arch == 'score':
        intended_num_classes = 1
        if config_loss_function is None: config_loss_function = 'l1' # Default for score
        elif not loss_requires_num_classes_1:
             print(f"Warning: arch='score' usually implies num_classes=1, but loss_function='{config_loss_function}' suggests multi-class. Prioritizing loss function needs.")
             # This case is ambiguous, let's rely on loss function requirement
             if loss_requires_num_classes_multi: intended_num_classes = 2 # Assume 2 if loss needs it
             else: print("Warning: Ambiguous config for arch='score' and loss. Assuming num_classes=1.")

    elif config_arch == 'class':
        if loss_requires_num_classes_1:
            intended_num_classes = 1
        elif loss_requires_num_classes_multi:
            intended_num_classes = 2 # Default to 2 for multi-class losses if dataset doesn't specify more
        elif config_loss_function is None:
            # Default loss for class arch if none specified (e.g., focal)
            config_loss_function = 'focal'
            intended_num_classes = 2 # Default to 2 if loss defaults to multi-class type
            print(f"DEBUG: No loss_function specified for arch='class', defaulting to '{config_loss_function}' (implies num_classes=2+).")
        else: # Loss specified but doesn't fit category? Should have been caught by parse_args
             exit(f"Error: Unknown loss function '{config_loss_function}' behavior for arch='class'.")
    else:
        exit(f"Error: Unknown model architecture '{config_arch}'.")

    if intended_num_classes is None: # Safety check
         exit("Error: Could not determine intended number of classes from configuration.")

    print(f"DEBUG: Intended num_classes based on config (arch='{config_arch}', loss='{config_loss_function}'): {intended_num_classes}")

    # --- 2. Get num_labels from Dataset ---
    num_labels_from_dataset = getattr(dataset, 'num_labels', 0)
    print(f"DEBUG: Labels found by dataset: {num_labels_from_dataset}")

    # --- 3. Validate Intended vs. Dataset Labels ---
    final_num_classes = intended_num_classes

    if num_labels_from_dataset == 0:
        # This shouldn't happen if dataset init works, but handle it.
        exit("Error: Dataset reported finding 0 labels/classes. Check data directory structure.")

    if intended_num_classes == 1:
        if num_labels_from_dataset > 2:
            # Cannot do binary loss if dataset has 3+ classes (e.g., folders 0, 1, 2)
            exit(f"Config Error: Intended num_classes=1 (e.g., for BCE loss), but dataset found {num_labels_from_dataset} classes. Ambiguous target for binary loss.")
        elif num_labels_from_dataset == 1:
             # Only one class found (e.g., only folder '0'), cannot train binary.
             exit(f"Config Error: Intended num_classes=1 (e.g., for BCE loss), but dataset only found 1 class (folder '{getattr(dataset, 'idx_to_label', {}).get(0, 'N/A')}'). Need at least two classes (0 and 1 folders) for binary classification training.")
        # If intended=1 and found=2, it's okay, we proceed with num_classes=1.
        elif num_labels_from_dataset == 2:
            print("DEBUG: Config intends num_classes=1 (binary loss), dataset found 2 classes (0/1 folders). Proceeding with num_classes=1.")
            final_num_classes = 1
        # Else (intended=1, found=0) - already handled above

    elif intended_num_classes >= 2: # Intending multi-class style (CE/Focal etc.)
         if num_labels_from_dataset == 1:
              exit(f"Config Error: Intended num_classes={intended_num_classes} (e.g., for CE/Focal loss), but dataset only found 1 class. Cannot train.")
         elif num_labels_from_dataset != intended_num_classes:
              # If intending 2+, but found a different number (e.g., 3 folders 0/1/2)
              # Use the number found by the dataset as the source of truth for multi-class.
              print(f"Warning: Intended num_classes={intended_num_classes} based on loss type default, but dataset found {num_labels_from_dataset} classes. Using {num_labels_from_dataset} classes.")
              final_num_classes = num_labels_from_dataset
         # Else (intended >= 2 and found == intended) -> OK

    # Store the final validated number back to args for other parts of the script
    args.num_classes = final_num_classes
    print(f"DEBUG: Final validated num_classes to be used: {args.num_classes}")

    # --- 4. Instantiate Criterion (Using Validated Classes) ---
    criterion = None
    class_weights_tensor = None
    # Get weights from args if they exist and match final_num_classes
    config_weights = getattr(args, 'weights', None)
    if config_weights and len(config_weights) == final_num_classes:
         try:
              class_weights_tensor = torch.tensor(config_weights, device=TARGET_DEV, dtype=torch.float32)
              print(f"DEBUG: Using class weights: {config_weights}")
         except Exception as e: print(f"Warning: Failed to make tensor from weights: {e}.")
    elif config_weights: print(f"Warning: Config weights len ({len(config_weights)}) != final num_classes ({final_num_classes}). Ignoring weights.")

    # Re-check loss compatibility with final_num_classes
    if config_loss_function == 'l1':
        if final_num_classes != 1: exit(f"Config Error: loss_function='l1' requires final_num_classes=1, but got {final_num_classes}.")
        criterion = nn.L1Loss(reduction='mean')
    elif config_loss_function == 'mse':
        if final_num_classes != 1: exit(f"Config Error: loss_function='mse' requires final_num_classes=1, but got {final_num_classes}.")
        criterion = nn.MSELoss(reduction='mean')
    elif config_loss_function == 'focal':
        if final_num_classes <= 1: exit(f"Config Error: loss_function='focal' requires final_num_classes > 1, but got {final_num_classes}.")
        criterion = FocalLoss(gamma=getattr(args, 'focal_loss_gamma', 2.0))
    elif config_loss_function == 'crossentropy':
        if final_num_classes <= 1: exit(f"Config Error: loss_function='crossentropy' requires final_num_classes > 1, but got {final_num_classes}.")
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif config_loss_function == 'bce':
        if final_num_classes != 1: exit(f"Config Error: loss_function='bce' requires final_num_classes=1, but got {final_num_classes}.")
        # BCEWithLogitsLoss weights are applied per *sample* if needed, not per class usually. Pass None.
        criterion = nn.BCEWithLogitsLoss(weight=None) # Use pos_weight for weighting positive class if needed
    elif config_loss_function == 'nll':
        if final_num_classes <= 1: exit(f"Config Error: loss_function='nll' requires final_num_classes > 1, but got {final_num_classes}.")
        criterion = nn.NLLLoss(weight=class_weights_tensor)
        # Check model output mode for NLL
        head_output_mode_nll = getattr(args, 'head_output_mode', 'linear').lower()
        if head_output_mode_nll != 'linear': exit(f"Config Error: loss_function='nll' requires head_output_mode='linear', but got '{head_output_mode_nll}'.")
    elif config_loss_function == 'ghm':
        if final_num_classes <= 1: exit(f"Config Error: loss_function='ghm' requires final_num_classes > 1, but got {final_num_classes}.")
        # Check model output mode for GHM
        head_output_mode_ghm = getattr(args, 'head_output_mode', 'linear').lower()
        if head_output_mode_ghm != 'linear': exit(f"Config Error: loss_function='ghm' requires head_output_mode='linear', but got '{head_output_mode_ghm}'.")
        criterion = GHMC_Loss(bins=getattr(args, 'ghm_bins', 10), momentum=getattr(args, 'ghm_momentum', 0.75))
    else: # Should be caught earlier, but safety net
        exit(f"Internal Error: Reached criterion setup with unknown loss function '{config_loss_function}'.")

    print(f"DEBUG: Criterion set to: {type(criterion).__name__}")

    # --- 5. Instantiate the HeadModel (Using Validated Classes) ---
    print(f"DEBUG: Instantiating HeadModel...")
    try:
        head_features = getattr(args, 'head_features', None)
        if head_features is None: exit("Error: 'head_features' not specified.")

        # Load other params from args (these should be correctly populated by parse_and_load_args)
        hidden_dim = getattr(args, 'head_hidden_dim', 1024)
        pooling_strategy = getattr(args, 'pooling_strategy', 'attn') # Needed for HeadModel internal logic
        num_res_blocks = getattr(args, 'head_num_res_blocks', 3)
        dropout_rate = getattr(args, 'head_dropout_rate', 0.2)
        output_mode = getattr(args, 'head_output_mode', 'linear') # Already checked compatibility with NLL/GHM
        attn_pool_heads = getattr(args, 'attn_pool_heads', 16)
        attn_pool_dropout = getattr(args, 'attn_pool_dropout', 0.2)

        model = HeadModel(
            features=head_features,
            num_classes=args.num_classes, # Use the FINAL validated number
            pooling_strategy=pooling_strategy, # Pass strategy for info, though pooling done elsewhere now
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            dropout_rate=dropout_rate,
            output_mode=output_mode,
            attn_pool_heads=attn_pool_heads,
            attn_pool_dropout=attn_pool_dropout
        )
    except Exception as e:
        print(f"Error details during HeadModel instantiation: {e}")
        traceback.print_exc()
        exit(f"Error instantiating HeadModel.")

    model.to(TARGET_DEV)
    print(f"HeadModel (Output Classes: {args.num_classes}) and criterion setup complete.")
    return model, criterion

# Version 2.5.1: Improved float conversion for optimizer/scheduler args
def setup_optimizer_scheduler(args, model):
    """
    Sets up the optimizer and scheduler based on args.
    Handles float conversion for scientific notation strings.
    """
    optimizer = None
    scheduler = None
    is_schedule_free = False
    optimizer_name = getattr(args, 'optimizer', 'AdamW').lower()

    print(f"Attempting to setup optimizer: {optimizer_name}")

    # --- Determine Parameters to Optimize (Logic Remains Same) ---
    params_to_optimize = model.parameters() # Simplified for feature training (always optimize head)
    # Check if any parameters were selected
    param_list_for_check = list(params_to_optimize)
    if not param_list_for_check: exit("Error: No parameters selected for optimization!")
    params_to_optimize = model.parameters() # Get generator again
    # --- End Parameter Determination ---

    # --- Dynamic Optimizer Loading ---
    if optimizer_name in OPTIMIZERS:
        optimizer_class = OPTIMIZERS[optimizer_name]
        print(f"Found optimizer class: {optimizer_class.__name__}")
        try:
            sig = inspect.signature(optimizer_class.__init__)
            available_params = sig.parameters.keys()
            potential_kwargs = {}
            args_dict = vars(args)
            for param_name in available_params:
                if param_name in ['self', 'params', 'model', 'args', 'kwargs']: continue
                if param_name in args_dict and args_dict[param_name] is not None:
                    expected_type = sig.parameters[param_name].annotation
                    default_value = sig.parameters[param_name].default
                    value = args_dict[param_name]
                    try:
                        # <<< START TYPE CONVERSION FIX >>>
                        converted_value = None
                        if expected_type == inspect.Parameter.empty:
                            # No type hint, try common types (float first for LR etc.)
                            try: converted_value = float(value)
                            except (ValueError, TypeError):
                                 try: converted_value = int(value)
                                 except (ValueError, TypeError):
                                      if isinstance(value, str) and value.lower() in ['true', 'false']:
                                           converted_value = value.lower() == 'true'
                                      else: converted_value = value # Use as is if no conversion works
                        elif expected_type == bool:
                            converted_value = str(value).lower() == 'true' if isinstance(value, str) else bool(value)
                        elif expected_type == int: converted_value = int(value)
                        elif expected_type == float: converted_value = float(value) # Handles '1e-4' correctly
                        elif expected_type == str: converted_value = str(value)
                        elif expected_type == tuple or expected_type == list or \
                             (hasattr(expected_type, '__origin__') and expected_type.__origin__ in [tuple, list]):
                            if isinstance(value, (list, tuple)):
                                inner_type = float # Default inner type
                                if hasattr(expected_type, '__args__') and expected_type.__args__: inner_type = expected_type.__args__[0]
                                converted_list = [inner_type(v) for v in value]
                                converted_value = tuple(converted_list) if expected_type == tuple or expected_type.__origin__ == tuple else converted_list
                            else: print(f"Warning: Arg {param_name} expects {expected_type} but got {type(value)}. Skipping.")
                        else: # Fallback for other types or Any
                            converted_value = value
                        # <<< END TYPE CONVERSION FIX >>>

                        if converted_value is not None:
                            potential_kwargs[param_name] = converted_value

                    except (ValueError, TypeError) as e_type:
                         print(f"Warning: Could not convert arg '{param_name}' (value: {value}) to expected type {expected_type}. Error: {e_type}. Using default or skipping.")
                         if default_value != inspect.Parameter.empty: potential_kwargs[param_name] = default_value

            print(f"  Instantiating {optimizer_class.__name__} with args: {potential_kwargs}")
            optimizer = optimizer_class(params_to_optimize, **potential_kwargs)
            if "schedulefree" in optimizer_name: is_schedule_free = True; print("  Detected schedule-free optimizer.")

        except Exception as e:
            print(f"ERROR: Failed to instantiate optimizer '{optimizer_name}' dynamically: {e}"); traceback.print_exc()
            print("  Falling back to AdamW."); optimizer_name = 'adamw'; optimizer = None; is_schedule_free = False

    # --- Fallback / Default Optimizer ---
    if optimizer is None:
        if optimizer_name != 'adamw': print(f"Warning: Optimizer '{optimizer_name}' failed. Falling back to AdamW.")
        optimizer_name = 'adamw'
        adamw_kwargs = { # Ensure these are floats
             'lr': float(getattr(args, 'lr', 1e-4)),
             'betas': tuple(getattr(args, 'betas', (0.9, 0.999))), # Assumes betas are already tuple/list
             'weight_decay': float(getattr(args, 'weight_decay', 0.0)),
             'eps': float(getattr(args, 'eps', 1e-8)),
        }
        print(f"Instantiating torch.optim.AdamW with args: {adamw_kwargs}")
        optimizer = torch.optim.AdamW(params_to_optimize, **adamw_kwargs); is_schedule_free = False
    # --- End Fallback ---

    # --- Scheduler Setup (Ensure float conversion here too if needed) ---
    if not is_schedule_free:
        scheduler_name = getattr(args, 'scheduler_name', 'CosineAnnealingLR').lower()
        print(f"Attempting to setup scheduler: {scheduler_name}")
        scheduler_class = None
        if scheduler_name in SCHEDULERS:
            scheduler_class = SCHEDULERS[scheduler_name]
            print(f"Found custom scheduler class: {scheduler_class.__name__}")
            try:
                sig = inspect.signature(scheduler_class.__init__); available_params = sig.parameters.keys()
                scheduler_kwargs = {}; args_dict = vars(args)
                for param_name in available_params:
                    if param_name in ['self', 'optimizer', 'last_epoch', 'args', 'kwargs']: continue
                    arg_key = f"scheduler_{param_name}" # Look for prefixed args
                    if arg_key not in args_dict and param_name in args_dict: arg_key = param_name # Fallback to non-prefixed
                    if arg_key in args_dict and args_dict[arg_key] is not None:
                        value = args_dict[arg_key]
                        # <<< Add simplified float conversion here too >>>
                        try:
                            # Try float conversion first for common scheduler args
                            scheduler_kwargs[param_name] = float(value)
                        except (ValueError, TypeError):
                             try: # Try int
                                  scheduler_kwargs[param_name] = int(value)
                             except (ValueError, TypeError):
                                  try: # Try bool
                                       if isinstance(value, str): scheduler_kwargs[param_name] = value.lower() == 'true'
                                       else: scheduler_kwargs[param_name] = bool(value)
                                  except: scheduler_kwargs[param_name] = value # Fallback
                print(f"  Instantiating {scheduler_class.__name__} with args: {scheduler_kwargs}")
                scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            except Exception as e: print(f"ERROR: Failed custom scheduler '{scheduler_name}': {e}"); scheduler = None
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
# Version 1.2.0 logging changes
def train_loop(args, model, criterion, optimizer, scheduler, scaler,
               train_loader, val_loader, wrapper, start_epoch, initial_global_step,
               enabled_amp, amp_dtype, is_schedule_free):
    """
    Runs the main training loop for sequence models with gradient accumulation
    and detailed logging metrics.
    Logs: loss/current, loss/epoch, loss/current_val_loss, loss/average_val_loss
    """
    # --- Initial Setup ---
    if not all(hasattr(args, attr) for attr in ['num_train_epochs', 'steps_per_epoch', 'max_train_steps', 'num_labels']):
         print("ERROR: Missing required args attributes (num_train_epochs, steps_per_epoch, max_train_steps, num_labels).")
         return

    model.train()
    if hasattr(optimizer, 'train') and callable(optimizer.train):
        try: optimizer.train()
        except Exception as e: print(f"Warning: Error calling optimizer.train(): {e}")

    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    effective_batch_size = args.batch * gradient_accumulation_steps
    print(f"Using Gradient Accumulation: {gradient_accumulation_steps} steps (Micro Bsz: {args.batch}, Eff Bsz: {effective_batch_size})")

    total_steps_to_run = args.max_train_steps
    total_epochs_to_run = args.num_train_epochs
    # steps_per_epoch should be MICRO-batches per epoch if using standard loader,
    # or GLOBAL steps per epoch if using calculated value (ensure consistency from main())
    steps_per_epoch = getattr(args, 'steps_per_epoch', len(train_loader))

    log_every_n = getattr(args, 'log_every_n', 100) # Logging frequency (GLOBAL steps)
    validate_every_n = getattr(args, 'validate_every_n', 0) # Validation frequency (GLOBAL steps, 0=disabled)
    if validate_every_n <= 0: # Default to once per epoch if disabled or invalid
         global_steps_per_epoch = math.ceil(steps_per_epoch / gradient_accumulation_steps) if gradient_accumulation_steps > 0 else steps_per_epoch
         validate_every_n = global_steps_per_epoch if global_steps_per_epoch > 0 else 1
         print(f"Validation frequency not set or invalid, defaulting to once per epoch ({validate_every_n} steps).")
    if log_every_n <= 0: log_every_n = 1 # Ensure logging happens

    print(f"Starting Training Loop. Target Epochs: {total_epochs_to_run}, Target Steps: {total_steps_to_run}")
    print(f"Micro-batches per epoch: {steps_per_epoch}. Logging every {log_every_n} steps. Validating every {validate_every_n} steps.")

    # --- State Variables ---
    global_step = initial_global_step
    best_eval_loss = wrapper.best_val_loss if wrapper.best_val_loss is not None and not math.isnan(wrapper.best_val_loss) else float('inf')
    last_eval_loss_val = float('nan') # Holds the most recent validation result for stepped graph
    current_train_loss_avg = float('nan') # Holds the most recent logged train loss avg

    # Accumulators for logging metrics
    current_window_loss_sum = torch.tensor(0.0, device=TARGET_DEV) # For loss/current
    current_window_steps = 0                                  # For loss/current
    epoch_loss_sum = torch.tensor(0.0, device=TARGET_DEV)         # For loss/epoch
    epoch_steps = 0                                           # For loss/epoch
    all_validation_losses = []                                # For loss/average_val_loss

    progress_bar = tqdm(initial=initial_global_step, total=total_steps_to_run, desc=f"Training (Eff Bsz {effective_batch_size})", unit="it", dynamic_ncols=True)
    # --- End State Variables ---

    # --- Outer Epoch Loop ---
    try:
        for epoch in range(start_epoch, total_epochs_to_run):
            if global_step >= total_steps_to_run: break

            # Reset Epoch Accumulators
            epoch_loss_sum.zero_()
            epoch_steps = 0

            # Set sampler epoch if applicable (might not be used with standard loader)
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                 print(f"\n--- Starting Epoch {epoch + 1}/{total_epochs_to_run} (Sampler Epoch Set) ---")
                 train_loader.sampler.set_epoch(epoch)
            elif hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
                 print(f"\n--- Starting Epoch {epoch + 1}/{total_epochs_to_run} (Batch Sampler Epoch Set) ---")
                 train_loader.batch_sampler.set_epoch(epoch)
            else:
                 print(f"\n--- Starting Epoch {epoch + 1}/{total_epochs_to_run} ---")

            wrapper.current_epoch = epoch
            model.train()
            if hasattr(optimizer, 'train') and callable(optimizer.train): optimizer.train()

            # Zero gradients ONCE before starting accumulation cycle for the epoch
            # We zero again after optimizer step
            optimizer.zero_grad(set_to_none=True)

            # --- Inner Step Loop (Iterates through MICRO-batches) ---
            # Use enumerate(train_loader) - assumes standard loader now
            for i, batch_data in enumerate(train_loader):
                # Check global step limit before processing
                if global_step >= total_steps_to_run: break

                # Handle bad batches
                if batch_data is None or not batch_data:
                     print(f"Warning: Skipping micro-batch {i} due to invalid batch_data.")
                     continue

                # --- Process Micro-Batch ---
                loss_this_step = torch.tensor(float('nan'), device=TARGET_DEV)
                try:
                    # Get sequence, mask, label (logic for padding/masking setup)
                    sequence_batch = batch_data.get('sequence')
                    mask_batch = batch_data.get('mask')
                    label_batch = batch_data.get('label')
                    if sequence_batch is None or mask_batch is None or label_batch is None: continue

                    # Data Integrity Check
                    if not torch.isfinite(sequence_batch).all():  # Check sequence_batch
                        # <<< Correct variable name in error message >>>
                        num_bad_elements = (~torch.isfinite(sequence_batch)).sum().item()
                        print(
                            f"\n!!! WARNING: Non-finite values detected in input sequence_batch at micro-batch {i}! Num bad: {num_bad_elements}. Skipping batch.")
                        continue
                    # <<< End Check >>>

                    # Move to device
                    sequence_batch = sequence_batch.to(TARGET_DEV)
                    mask_batch = mask_batch.to(TARGET_DEV)
                    label_batch = label_batch.to(TARGET_DEV)
                    current_micro_batch_size = sequence_batch.size(0)

                    # Prepare target tensor
                    target = label_batch
                    if args.num_classes == 1: target = target.to(dtype=torch.float32).view(current_micro_batch_size, -1).squeeze(-1)
                    else: target = target.to(dtype=torch.long).view(current_micro_batch_size)

                    # --- Forward/Backward Pass ---
                    with torch.amp.autocast(device_type=TARGET_DEV, enabled=enabled_amp, dtype=amp_dtype):
                        # Pass sequence AND mask to model (ensure HeadModel uses mask)
                        y_pred = model(sequence_batch, attention_mask=mask_batch)
                        # Prepare prediction shape for loss
                        y_pred_for_loss = y_pred # ... (squeeze if needed based on loss/num_classes) ...
                        if isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)) and args.num_classes == 1:
                            if y_pred.ndim == 2 and y_pred.shape[1] == 1: y_pred_for_loss = y_pred.squeeze(-1)

                        # Calculate loss
                        loss_input = y_pred_for_loss.to(torch.float32)
                        target_for_loss = target.to(loss_input.device)
                        if isinstance(criterion, nn.NLLLoss): loss_input = F.log_softmax(loss_input, dim=-1)
                        loss = criterion(loss_input, target_for_loss.long() if isinstance(criterion, (nn.CrossEntropyLoss, FocalLoss, nn.NLLLoss, GHMC_Loss)) else target_for_loss.float())

                        # Store the *un-normalized* loss for logging
                        loss_this_step = loss.detach().item() # Get scalar value

                        # Normalize loss for accumulation
                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/Inf loss detected at micro-batch {i}. Skipping backward.")
                        loss = None # Prevent backward pass
                    else:
                        # Accumulate gradients
                        scaler.scale(loss).backward()

                except Exception as e:
                    print(f"\nError processing micro-batch {i} (Global Step approx {global_step}): {e}")
                    traceback.print_exc()
                    loss = None # Prevent accumulation if error occurred

                # Accumulate Losses for Logging (if loss was valid)
                if loss is not None and not math.isnan(loss_this_step):
                     current_window_loss_sum += loss_this_step
                     current_window_steps += 1
                     epoch_loss_sum += loss_this_step
                     epoch_steps += 1

                # --- Optimizer Step Check ---
                # Check using micro-batch index 'i' and grad acc steps
                if (i + 1) % gradient_accumulation_steps == 0:
                    # --- Skipping Logic for Resume ---
                    if global_step < initial_global_step:
                        if gradient_accumulation_steps > 1: optimizer.zero_grad(set_to_none=True)
                        global_step += 1; progress_bar.update(1); continue

                    # --- Perform Optimizer Step ---
                    optimizer_stepped = False
                    try:
                        # <<< --- ADD GRADIENT CLIPPING --- >>>
                        # Unscale the gradients before clipping
                        # scaler.unscale_(optimizer)

                        # Clip the norm of the gradients
                        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Example: max_norm=1.0

                        # Log grad_norm if desired (helps monitor clipping)
                        # if wrapper.wandb_run and global_step % log_every_n == 0: # Log at the same frequency as loss
                        #      try: wrapper.wandb_run.log({"train/grad_norm": grad_norm.item()}, step=global_step)
                        #      except Exception as e_gn: print(f"Wandb grad_norm log error: {e_gn}")

                        # Scaler step will skip update if gradients are invalid (NaN/Inf)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer_stepped = True
                    except Exception as e_step:
                         print(f"\nError during optimizer step {global_step}: {e_step}")
                         traceback.print_exc()
                         optimizer_stepped = False
                    # <<< Zero gradients AFTER stepping >>>
                    optimizer.zero_grad(set_to_none=True)

                    # Scheduler Step (AFTER optimizer step)
                    if scheduler is not None and not is_schedule_free and optimizer_stepped:
                        try: scheduler.step()
                        except Exception as e_sched: print(f"Warning: Error during scheduler step: {e_sched}")

                    # Increment global_step only AFTER a potential optimizer step
                    if optimizer_stepped:
                         global_step += 1
                         progress_bar.update(1)
                         wrapper.update_step(global_step)

                         # --- Logging (Every LOG_EVERY_N global steps) ---
                         if global_step % log_every_n == 0 and global_step > 0:
                             # Calculate loss/current
                             if current_window_steps > 0:
                                 current_train_loss_avg = (current_window_loss_sum / current_window_steps).item()
                             else: current_train_loss_avg = float('nan')
                             # Reset CURRENT window accumulators
                             current_window_loss_sum.zero_()
                             current_window_steps = 0

                             # Log to wrapper
                             wrapper.log_main(
                                 step=global_step,
                                 current_train_loss=current_train_loss_avg,
                                 current_val_loss=last_eval_loss_val # Pass last known validation loss
                             )
                             # Update Postfix
                             lr = optimizer.param_groups[0]['lr']
                             postfix_dict = collections.OrderedDict(); postfix_dict["Epoch"] = epoch + 1
                             postfix_dict["Loss(curr)"] = f"{current_train_loss_avg:.3e}" if not math.isnan(current_train_loss_avg) else "N/A"
                             if not math.isnan(last_eval_loss_val): postfix_dict["Loss(val)"] = f"{last_eval_loss_val:.3e}"
                             postfix_dict["LR"] = f"{lr:.1e}"; progress_bar.set_postfix(ordered_dict=postfix_dict, refresh=False)
                         # --- End Logging ---

                         # --- Validation (Every VALIDATE_EVERY_N global steps) ---
                         if validate_every_n > 0 and global_step % validate_every_n == 0 and global_step > 0:
                             print(f"\n--- Running Validation @ Step {global_step} ---")
                             eval_loss_val = run_validation_sequences(model, val_loader, criterion, TARGET_DEV, scaler, args.num_labels)
                             model.train() # Ensure back in train mode

                             if not math.isnan(eval_loss_val):
                                 last_eval_loss_val = eval_loss_val # Update last known
                                 all_validation_losses.append(eval_loss_val) # Store for average
                                 print(f"--- Validation Complete: Eval Loss = {eval_loss_val:.4e} ---")

                                 # Calculate and Log Average Validation Loss
                                 avg_val_loss = sum(all_validation_losses) / len(all_validation_losses)
                                 print(f"--- Average Validation Loss So Far: {avg_val_loss:.4e} ---")
                                 if wrapper.wandb_run:
                                      try:
                                           wrapper.wandb_run.log({
                                                # Log the specific result for this run
                                                "loss/eval_run_result": eval_loss_val,
                                                # Log the running average
                                                "loss/average_val_loss": avg_val_loss
                                                }, step=global_step)
                                      except Exception as e: print(f"Wandb val log error: {e}")

                                 # Update progress bar postfix
                                 lr = optimizer.param_groups[0]['lr']
                                 postfix_dict = collections.OrderedDict(); postfix_dict["Epoch"] = epoch + 1
                                 # Use the value calculated during logging step if available, else NaN
                                 postfix_dict["Loss(curr)"] = f"{current_train_loss_avg:.3e}" if not math.isnan(current_train_loss_avg) else "N/A"
                                 postfix_dict["Loss(val)"] = f"{eval_loss_val:.3e}"
                                 postfix_dict["Loss(avg_val)"] = f"{avg_val_loss:.3e}"
                                 postfix_dict["LR"] = f"{lr:.1e}"; progress_bar.set_postfix(ordered_dict=postfix_dict, refresh=False)

                                 # Check / Save best model
                                 if eval_loss_val < best_eval_loss:
                                      best_eval_loss = eval_loss_val
                                      wrapper.best_val_loss = best_eval_loss
                                      print(f"New best val loss: {best_eval_loss:.4e}. Saving best model...")
                                      wrapper.save_model(step=global_step, epoch=epoch, suffix="_best_val", save_aux=False, args=args)
                                 else:
                                      print(f"Validation loss {eval_loss_val:.4e} did not improve on best {best_eval_loss:.4e}")
                             else: # Handle NaN validation loss
                                 print(f"Warning: Eval loss is NaN at Step {global_step}. Cannot update averages or save best.")
                         # --- End Validation ---

                         # --- Periodic saving ---
                         if args.nsave > 0 and global_step % args.nsave == 0:
                             if global_step > initial_global_step:
                                  print(f"\nSaving periodic checkpoint @ step {global_step}...")
                                  wrapper.save_model(step=global_step, epoch=epoch, save_aux=False, args=args)

                         # Check global step limit AFTER potentially saving
                         if global_step >= total_steps_to_run: break
                # --- End Optimizer Step Check ---

                # Check global step limit again before next micro-batch
                if global_step >= total_steps_to_run: break
            # --- End Inner Step (micro-batch) Loop ---

            # --- Log Epoch Average Loss ---
            if epoch_steps > 0:
                epoch_avg_loss = (epoch_loss_sum / epoch_steps).item()
                print(f"--- Epoch {epoch + 1} Finished. Average Train Loss: {epoch_avg_loss:.4e} ---")
                if wrapper.wandb_run:
                    try: wrapper.wandb_run.log({"loss/epoch": epoch_avg_loss}, step=global_step)
                    except Exception as e: print(f"Wandb epoch log error: {e}")
            # --- End Epoch Logging ---

            if global_step >= total_steps_to_run: break
        # --- End Outer Epoch Loop ---

    except KeyboardInterrupt: print("\nTraining interrupted by user.")
    finally:
        progress_bar.close()
        print(f"\nTraining loop finished. Reached Global Step: {global_step}")


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

    # 4. Calculate final training duration (Corrected for Grad Acc)
    if args.steps_per_epoch == 0: exit("Error: steps_per_epoch is zero.")

    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    # <<< Calculate GLOBAL steps per epoch >>>
    global_steps_per_epoch = math.ceil(args.steps_per_epoch / gradient_accumulation_steps)
    if global_steps_per_epoch == 0: global_steps_per_epoch = 1 # Avoid division by zero if epoch is too short
    print(f"DEBUG Main: Micro-batches/epoch = {args.steps_per_epoch}, Grad Acc = {gradient_accumulation_steps}, Global Steps/epoch = {global_steps_per_epoch}")

    # <<< Use global_steps_per_epoch for calculations >>>
    if args.max_train_steps is not None and args.max_train_steps > 0:
        # Calculate epochs needed based on GLOBAL steps
        args.num_train_epochs = math.ceil(args.max_train_steps / global_steps_per_epoch)
        print(f"DEBUG Main: Calculated num_train_epochs = {args.num_train_epochs} from max_train_steps={args.max_train_steps}")
    elif args.max_train_epochs is not None and args.max_train_epochs > 0:
        args.num_train_epochs = args.max_train_epochs
        # Calculate total GLOBAL steps based on epochs
        args.max_train_steps = args.num_train_epochs * global_steps_per_epoch
        print(f"DEBUG Main: Calculated max_train_steps = {args.max_train_steps} from max_train_epochs={args.num_train_epochs}")
    else: # Default if neither specified
        args.max_train_steps = 10000 # Default GLOBAL steps
        args.num_train_epochs = math.ceil(args.max_train_steps / global_steps_per_epoch)
        print(f"DEBUG Main: Defaulting to max_train_steps={args.max_train_steps}, calculated num_train_epochs={args.num_train_epochs}")

    # 5. Setup Model & Criterion
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