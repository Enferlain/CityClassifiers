# Version: 2.2.0 (Refactored for Clarity and Loss Selection)
import inspect
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

from losses import GHMC_Loss
# --- Local Imports ---
# Assuming utils.py is in the same directory or accessible
from utils import (
    ModelWrapper, get_embed_params, parse_and_load_args, write_config, # <<< CHANGED HERE
    LOG_EVERY_N, FocalLoss
)
from dataset import EmbeddingDataset
from model import PredictorModel

# --- Attempt to import our custom optimizers ---
try:
    # Make sure the optimizer directory is in the path if needed
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming 'optimizer' folder is directly inside the folder containing train.py
    optimizer_dir_path = os.path.join(current_script_dir, 'optimizer')
    if os.path.isdir(optimizer_dir_path) and current_script_dir not in sys.path:
         # Add the directory containing train.py, which should allow 'import optimizer'
         sys.path.insert(0, current_script_dir)

         # <<< Import BOTH dictionaries >>>
    from optimizer import OPTIMIZERS, SCHEDULERS

    print(f"Successfully imported custom optimizers: {list(OPTIMIZERS.keys())}")
    print(f"Successfully imported custom schedulers: {list(SCHEDULERS.keys())}")
    custom_modules_available = True
except ImportError as e:
    print(
        f"Warning: Custom optimizer/scheduler import failed ({e}). Check optimizer/__init__.py. Standard torch modules only.")
    OPTIMIZERS = {}
    SCHEDULERS = {}  # Define as empty dicts if import fails
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
    """Sets up the model and criterion based on explicit args using PredictorModel v2."""
    print("DEBUG setup_model_criterion: Setting up model and criterion...")

    # --- Get Model Parameters from Args ---
    embed_params = get_embed_params(args.embed_ver)
    features = embed_params['features']
    hidden_dim = getattr(args, 'hidden_dim', embed_params.get('hidden', 1280))
    arch = args.arch
    loss_function_name = getattr(args, 'loss_function') # Assume it's set by parse_and_load_args
    use_attention = getattr(args, 'use_attention', True)
    num_attn_heads = getattr(args, 'num_attn_heads', 8)
    attn_dropout = getattr(args, 'attn_dropout', 0.1)
    num_res_blocks = getattr(args, 'num_res_blocks', 1)
    dropout_rate = getattr(args, 'dropout_rate', 0.1)
    # --- Read EXPLICIT output_mode from args (set via YAML/cmd line) ---
    output_mode = getattr(args, 'output_mode', None)
    if output_mode is None: raise ValueError("predictor_params.output_mode must be explicitly set.")
    output_mode = output_mode.lower()
    args.output_mode = output_mode

    num_classes = 1
    criterion = None
    class_weights = None # Logic to load weights remains same

    # --- Determine Criterion (Allowing non-standard pairings) ---
    if loss_function_name == 'l1':
        print("DEBUG: Setting criterion to L1Loss.")
        criterion = nn.L1Loss(reduction='mean')
        num_classes = 1
        # Suggest 'tanh_scaled' or 'sigmoid' but don't enforce
        if output_mode not in ['tanh_scaled', 'sigmoid', 'linear']: print(f"Warning: L1Loss typically paired with scaled output, got '{output_mode}'.")

    elif loss_function_name == 'mse':
        print("DEBUG: Setting criterion to MSELoss.")
        criterion = nn.MSELoss(reduction='mean')
        num_classes = 1
        # Suggest 'tanh_scaled' or 'linear' but don't enforce
        if output_mode not in ['tanh_scaled', 'linear', 'sigmoid']: print(f"Warning: MSELoss typically paired with linear/scaled output, got '{output_mode}'.")

    elif loss_function_name == 'focal':
        print("DEBUG: Setting criterion to FocalLoss.")
        criterion = FocalLoss(gamma=getattr(args, 'focal_loss_gamma', 2.0))
        num_classes = getattr(args, 'num_labels', 2)
        # Warn if output is not linear, but allow it
        if output_mode != 'linear': print(f"Warning: FocalLoss usually expects output_mode='linear', but got '{output_mode}'. Training with non-logit input.")

    elif loss_function_name == 'crossentropy':
        print("DEBUG: Setting criterion to CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        num_classes = getattr(args, 'num_labels', 2)
        # Warn if output is not linear, but allow it
        if output_mode != 'linear': print(f"Warning: CrossEntropyLoss usually expects output_mode='linear', but got '{output_mode}'. Training with non-logit input.")

    elif loss_function_name == 'bce':
        print("DEBUG: Setting criterion to BCEWithLogitsLoss.")
        criterion = nn.BCEWithLogitsLoss()
        num_classes = getattr(args, 'num_classes', 1)
        # Warn if output is not linear, but allow it
        if output_mode != 'linear': print(f"Warning: BCEWithLogitsLoss usually expects output_mode='linear', but got '{output_mode}'. Training with non-logit input.")
        if num_classes != 1: print(f"Warning: BCEWithLogitsLoss used with num_classes={num_classes}. Ensure targets are correct.")

    elif loss_function_name == 'nll':
        print("DEBUG: Setting criterion to NLLLoss (expects LogSoftmax input).")
        # --- FIX: Handle Weights Correctly ---
        num_classes = getattr(args, 'num_labels', 2) # NLL needs num_classes >= 2
        if args.weights and len(args.weights) == num_classes:
             class_weights = torch.tensor(args.weights, device=TARGET_DEV, dtype=torch.float32) # Create tensor correctly
             print(f"DEBUG: Using NLLLoss with weights: {args.weights}")
        elif args.weights:
             print(f"Warning: NLLLoss weight mismatch. Ignoring weights.")
        # --- END FIX ---
        criterion = nn.NLLLoss(weight=class_weights)

        # --- FIX: NLL needs Logits from model, LogSoftmax applied before loss ---
        # Force output_mode to linear
        if output_mode != 'linear':
             print(f"Warning: loss_function='nll' selected, but output_mode was '{output_mode}'. Forcing output_mode='linear'. NLLLoss expects LogSoftmax input, which will be applied in train loop.")
             output_mode = 'linear'
             args.output_mode = 'linear' # Update args for model instantiation
        # --- END FIX ---

        args.loss_function = 'nll' # Update args for config saving

    # <<< ADD GHM BLOCK >>>
    elif loss_function_name == 'ghm':
        print("DEBUG: Setting criterion to GHMC_Loss.")
        # Get optional GHM parameters from config/args
        ghm_bins = getattr(args, 'ghm_bins', 10)
        ghm_momentum = getattr(args, 'ghm_momentum', 0.75)
        criterion = GHMC_Loss(bins=ghm_bins, momentum=ghm_momentum, reduction='mean')
        num_classes = getattr(args, 'num_labels', 2) # GHM expects >= 2 classes currently
        if num_classes < 2:
             raise ValueError("GHMC_Loss currently requires num_classes >= 2.")
        # Validation: GHM requires linear output (logits)
        if output_mode != 'linear':
             raise ValueError(f"GHMC_Loss requires output_mode='linear', but got '{output_mode}' in config.")
        args.loss_function = 'ghm' # Update args
        # Handle weights? GHMC doesn't directly take class weights, it harmonizes gradients.
        if args.weights: print("Warning: Class weights specified but GHMC_Loss does not use them directly.")
    # <<< END GHM BLOCK >>>

    else:
        raise ValueError(f"Unknown loss_function '{loss_function_name}' specified in config.")

    # --- Final check on num_classes consistency ---
    config_num_classes = getattr(args, 'num_classes', None)
    if config_num_classes is not None and config_num_classes != num_classes:
        print(f"Warning: predictor_params.num_classes ({config_num_classes}) seems inconsistent with loss function expectation ({num_classes}). Using {num_classes}.")
    # Use the num_classes determined by the loss function logic
    args.num_classes = num_classes

    # --- Instantiate the Enhanced Model ---
    print(f"DEBUG: Instantiating PredictorModel v2 with num_classes={num_classes}, output_mode='{output_mode}'")
    model = PredictorModel(
        features=features,
        hidden_dim=hidden_dim,
        num_classes=num_classes, # Use determined num_classes
        use_attention=use_attention,
        num_attn_heads=num_attn_heads,
        attn_dropout=attn_dropout,
        num_res_blocks=num_res_blocks,
        dropout_rate=dropout_rate,
        output_mode=output_mode # Pass explicit output_mode from config
    )
    model.to(TARGET_DEV)

    # --- Store parameters back to args ---
    args.features = features
    args.hidden_dim = hidden_dim
    # args.num_classes already updated
    args.use_attention = use_attention
    args.num_res_blocks = num_res_blocks
    args.dropout_rate = dropout_rate
    # args.output_mode already updated

    return model, criterion

# Version 2.4.0: Dynamic Optimizer & Scheduler Loading
def setup_optimizer_scheduler(args, model):
    """
    Sets up the optimizer and scheduler based on args.
    Dynamically loads custom optimizers/schedulers from OPTIMIZERS/SCHEDULERS dicts.
    Uses inspect to gather relevant arguments from args object.
    """
    optimizer = None
    scheduler = None
    is_schedule_free = False
    optimizer_name = getattr(args, 'optimizer', 'AdamW').lower()

    print(f"Attempting to setup optimizer: {optimizer_name}")

    # --- Dynamic Optimizer Loading ---
    if optimizer_name in OPTIMIZERS:
        optimizer_class = OPTIMIZERS[optimizer_name]
        print(f"Found optimizer class: {optimizer_class.__name__}")

        try:
            # Inspect the optimizer's __init__ signature
            sig = inspect.signature(optimizer_class.__init__)
            available_params = sig.parameters.keys()
            # print(f"  Optimizer signature params: {list(available_params)}")

            # Gather potential kwargs from args, converting types where necessary
            potential_kwargs = {}
            args_dict = vars(args)

            for param_name in available_params:
                if param_name in ['self', 'params', 'model', 'args', 'kwargs']: # Skip standard/generic args
                    continue

                if param_name in args_dict and args_dict[param_name] is not None:
                    # Get the expected type from the signature if available
                    expected_type = sig.parameters[param_name].annotation
                    default_value = sig.parameters[param_name].default

                    value = args_dict[param_name]

                    # --- Type Conversion Logic ---
                    try:
                        if expected_type == inspect.Parameter.empty: # No type hint, use value as is
                             potential_kwargs[param_name] = value
                        elif expected_type == bool:
                             potential_kwargs[param_name] = bool(value)
                        elif expected_type == int:
                             potential_kwargs[param_name] = int(value)
                        elif expected_type == float:
                             potential_kwargs[param_name] = float(value)
                        elif expected_type == str:
                             potential_kwargs[param_name] = str(value)
                        elif expected_type == tuple or expected_type == list or \
                             (hasattr(expected_type, '__origin__') and expected_type.__origin__ in [tuple, list]):
                            # Handle tuples/lists, attempt conversion if value is list-like
                            if isinstance(value, (list, tuple)):
                                # Try converting elements if type hint suggests (e.g., Tuple[float, float])
                                inner_type = float # Default assumption
                                if hasattr(expected_type, '__args__') and expected_type.__args__:
                                    inner_type = expected_type.__args__[0]
                                converted_list = [inner_type(v) for v in value]
                                potential_kwargs[param_name] = tuple(converted_list) if expected_type == tuple else converted_list
                            else:
                                print(f"Warning: Arg {param_name} expects {expected_type} but got {type(value)}. Skipping.")
                        else:
                            # Use value directly if type doesn't match known basic types
                            potential_kwargs[param_name] = value
                    except (ValueError, TypeError) as e_type:
                         print(f"Warning: Could not convert arg '{param_name}' (value: {value}) to expected type {expected_type}. Error: {e_type}. Using default or skipping.")
                         if default_value != inspect.Parameter.empty:
                              potential_kwargs[param_name] = default_value # Use default if conversion fails
                    # --- End Type Conversion Logic ---

            print(f"  Instantiating {optimizer_class.__name__} with args: {potential_kwargs}")
            optimizer = optimizer_class(model.parameters(), **potential_kwargs)

            # Check if it's a schedule-free optimizer (simple name check for now)
            if "schedulefree" in optimizer_name:
                is_schedule_free = True

        except Exception as e:
            print(f"ERROR: Failed to instantiate optimizer '{optimizer_name}' dynamically: {e}")
            print("  Falling back to AdamW.")
            optimizer_name = 'adamw' # Force fallback
            optimizer = None # Reset optimizer variable
            is_schedule_free = False

    # --- Fallback / Default Optimizer ---
    if optimizer is None:
        if optimizer_name != 'adamw':
             print(f"Warning: Optimizer '{optimizer_name}' not found in OPTIMIZERS or failed to instantiate. Falling back to AdamW.")
        optimizer_name = 'adamw'
        # Use defaults from args for AdamW, converting types explicitly
        adamw_kwargs = {
             'lr': float(getattr(args, 'lr', 1e-4)),
             'betas': tuple(getattr(args, 'betas', (0.9, 0.999))),
             'weight_decay': float(getattr(args, 'weight_decay', 0.0)),
             'eps': float(getattr(args, 'eps', 1e-8)),
        }
        print(f"Instantiating torch.optim.AdamW with args: {adamw_kwargs}")
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
        is_schedule_free = False # AdamW needs a scheduler
    # --- End Fallback ---

    # --- Scheduler Setup ---
    # --- Dynamic Scheduler Setup ---
    if not is_schedule_free:
        # Get desired scheduler name from config, default to standard CosineAnnealingLR if not specified
        # Use 'scheduler_name' in YAML to avoid conflict with 'cosine' boolean flag
        scheduler_name = getattr(args, 'scheduler_name', 'CosineAnnealingLR').lower()
        print(f"Attempting to setup scheduler: {scheduler_name}")

        scheduler_class = None
        # Check if it's one of our custom schedulers
        if scheduler_name in SCHEDULERS:
            scheduler_class = SCHEDULERS[scheduler_name]
            print(f"Found custom scheduler class: {scheduler_class.__name__}")
            try:
                # Inspect the scheduler's __init__ signature
                sig = inspect.signature(scheduler_class.__init__)
                available_params = sig.parameters.keys()
                scheduler_kwargs = {}
                args_dict = vars(args)

                # Gather relevant arguments, looking for "scheduler_{param_name}" in args
                for param_name in available_params:
                    if param_name in ['self', 'optimizer', 'last_epoch', 'args', 'kwargs']: continue # Skip standard/generic

                    arg_key = f"scheduler_{param_name}" # Look for prefixed arg name
                    if arg_key in args_dict and args_dict[arg_key] is not None:
                        expected_type = sig.parameters[param_name].annotation
                        default_value = sig.parameters[param_name].default
                        value = args_dict[arg_key]

                        # --- Type Conversion Logic (similar to optimizer) ---
                        try:
                            if expected_type == inspect.Parameter.empty: scheduler_kwargs[param_name] = value
                            elif expected_type == bool: scheduler_kwargs[param_name] = bool(value)
                            elif expected_type == int: scheduler_kwargs[param_name] = int(value)
                            elif expected_type == float: scheduler_kwargs[param_name] = float(value)
                            elif expected_type == str: scheduler_kwargs[param_name] = str(value)
                            # Add tuple/list handling if needed for schedulers
                            else: scheduler_kwargs[param_name] = value
                        except (ValueError, TypeError) as e_type:
                             print(f"Warning: Could not convert scheduler arg '{param_name}' (value: {value}) to {expected_type}. Error: {e_type}. Using default or skipping.")
                             if default_value != inspect.Parameter.empty: scheduler_kwargs[param_name] = default_value
                        # --- End Type Conversion ---

                print(f"  Instantiating {scheduler_class.__name__} with args: {scheduler_kwargs}")
                # Instantiate with optimizer and gathered kwargs
                scheduler = scheduler_class(optimizer, **scheduler_kwargs)

            except Exception as e:
                 print(f"ERROR: Failed to instantiate custom scheduler '{scheduler_name}': {e}")
                 print("  Falling back to standard PyTorch schedulers.")
                 scheduler = None # Reset to trigger fallback

        # --- Fallback to Standard PyTorch Schedulers ---
        if scheduler is None: # If custom failed or wasn't specified
            # Use existing logic based on 'cosine' flag or 'warmup_steps'
            scheduler_type = None
            # Check args.cosine ONLY if scheduler_name wasn't explicitly something else
            if getattr(args, 'cosine', True) and scheduler_name in ['cosineannealinglr', 'cosine']:
                 scheduler_type = 'cosine'
            elif getattr(args, 'warmup_steps', 0) > 0 and scheduler_name == 'linearlr': # Check warmup AND if linear was maybe intended
                 scheduler_type = 'warmup'
            elif scheduler_name not in ['cosineannealinglr', 'linearlr', 'none', None]:
                 print(f"Warning: Scheduler '{scheduler_name}' not found in custom SCHEDULERS and doesn't match standard options. No scheduler used.")

            if scheduler_type == 'cosine':
                print("Using standard torch.optim.lr_scheduler.CosineAnnealingLR.")
                t_max_steps = getattr(args, 'scheduler_t_max', args.steps) # Allow T_max override
                eta_min = getattr(args, 'scheduler_eta_min', 0) # Allow eta_min override
                print(f"  Setting T_max = {t_max_steps}, eta_min = {eta_min}")
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(t_max_steps), eta_min=float(eta_min))
            elif scheduler_type == 'warmup':
                print("Using standard torch.optim.lr_scheduler.LinearLR.")
                warmup_iters = int(args.warmup_steps)
                print(f"  Setting total_iters = {warmup_iters} for LinearLR.")
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
            elif scheduler is None: # Handles cases where no match found or name was 'none'
                 print("No matching standard or custom scheduler specified.")

    # --- Handle Schedule-Free Case ---
    else:
        print("Using a schedule-free optimizer, no scheduler will be used.")
    # --- End Scheduler Setup ---

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
        try: model.load_state_dict(load_file(args.resume))
        except Exception as e: print(f"Error loading model state_dict: {e}"); exit(1)

        # --- Load Aux Files ---
        base_path = os.path.splitext(args.resume)[0]
        optim_path = f"{base_path}.optim"
        sched_path = f"{base_path}.sched"
        scaler_path = f"{base_path}.scaler"
        state_path = f"{base_path}.state" # File storing epoch/global_step

        if os.path.exists(optim_path):
             try: optimizer.load_state_dict(torch.load(optim_path, map_location=TARGET_DEV)); print("Optimizer state loaded.")
             except Exception as e: print(f"Warning: Could not load optimizer state: {e}")
        else: print("Warning: Optimizer state file (.optim) not found.")

        if scheduler is not None and os.path.exists(sched_path):
             try: scheduler.load_state_dict(torch.load(sched_path, map_location=TARGET_DEV)); print("Scheduler state loaded.")
             except Exception as e: print(f"Warning: Could not load scheduler state: {e}")
        else: print("Warning: Scheduler state file (.sched) not found or scheduler is None.")

        if scaler.is_enabled() and os.path.exists(scaler_path):
             try: scaler.load_state_dict(torch.load(scaler_path, map_location=TARGET_DEV)); print("GradScaler state loaded.")
             except Exception as e: print(f"Warning: Could not load GradScaler state: {e}")
        else: print("Warning: GradScaler state file (.scaler) not found or scaler disabled.")

        # --- Load Training State (Epoch/Step) ---
        if os.path.exists(state_path):
             try:
                  train_state = torch.load(state_path, map_location='cpu')
                  # Resume from the *next* epoch and step
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
#        Main Training Loop (Epoch-Based)
# ================================================
# Version 3.3.0 (Proper Epoch/Iterator Handling)
def train_loop(args, model, criterion, optimizer, scheduler, scaler,
               train_loader, val_loader, wrapper, start_epoch, initial_global_step,
               enabled_amp, amp_dtype):
    """
    Runs the main training loop (Epoch-based outer, Step-based inner/progress)
    with proper DataLoader iterator handling to avoid restarts within an epoch.
    Includes deferred loss processing.
    """

    # --- Initial Setup ---
    if not hasattr(args, 'num_train_epochs') or not hasattr(args, 'steps_per_epoch'):
         print("ERROR: num_train_epochs or steps_per_epoch missing from args.")
         return
    if not hasattr(args, 'max_train_steps'):
         print("ERROR: max_train_steps missing from args.")
         return

    if hasattr(optimizer, 'train') and callable(optimizer.train): optimizer.train()
    model.train()

    total_steps_to_run = args.max_train_steps
    total_epochs_to_run = args.num_train_epochs
    steps_per_epoch = args.steps_per_epoch

    print(f"Starting training loop. Target Epochs: {total_epochs_to_run}, Target Steps: {total_steps_to_run}")
    print(f"Steps per epoch: {steps_per_epoch}")

    global_step = initial_global_step
    best_eval_loss = float('inf')

    progress_bar = tqdm(initial=initial_global_step, total=total_steps_to_run, desc="Overall Training", unit="step", dynamic_ncols=True)

    accumulated_loss = torch.tensor(0.0, device=TARGET_DEV)
    accumulation_steps = 0

    # <<< Create the iterator ONCE before the epoch loop >>>
    print("Initializing DataLoader iterator...")
    train_iterator = iter(train_loader)

    # --- Outer Epoch Loop ---
    try:
        for epoch in range(start_epoch, total_epochs_to_run):
            if global_step >= total_steps_to_run: break # Check limit before starting epoch

            # Print epoch message less frequently
            if epoch == start_epoch or (epoch + 1) % 50 == 0 or epoch == total_epochs_to_run - 1:
                 print(f"\n--- Starting Epoch {epoch + 1} / {total_epochs_to_run} (Global Step: {global_step}) ---")
            wrapper.current_epoch = epoch

            model.train()
            if hasattr(optimizer, 'train'): optimizer.train()

            step_in_epoch = 0
            # --- Inner Step Loop (Fixed number of steps per epoch) ---
            while step_in_epoch < steps_per_epoch:
                if global_step >= total_steps_to_run: break # Check limit before getting batch

                # --- Get Batch using Persistent Iterator ---
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    # <<< Handle DataLoader Exhaustion: Restart Iterator >>>
                    # print("\n--- DataLoader exhausted, restarting iterator ---")
                    train_iterator = iter(train_loader)
                    try:
                         batch = next(train_iterator)
                    except StopIteration:
                         # Should not happen if dataset is not empty, but handle anyway
                         print("ERROR: DataLoader provided no batches even after restart. Stopping.")
                         global_step = total_steps_to_run # Force exit
                         break
                except Exception as e_iter:
                     print(f"\nError getting batch from iterator at step {global_step}: {e_iter}. Stopping.")
                     global_step = total_steps_to_run # Force exit
                     break
                # --- End Get Batch ---

                # --- Skipping Logic (Based on Global Step for Resuming) ---
                # This needs to be inside the step loop now
                if global_step < initial_global_step:
                    global_step += 1
                    progress_bar.update(1)
                    # Do *not* increment step_in_epoch here, as this step wasn't "part" of the current epoch's work
                    continue # Skip processing
                # --- End Skipping Logic ---

                # --- Handle Bad Batches ---
                if batch is None:
                    print(f"Warning: Skipping step {global_step} due to invalid batch.")
                    global_step += 1
                    step_in_epoch += 1 # Count this as a step within the epoch
                    progress_bar.update(1)
                    continue

                # --- Get Batch Data (remains same) ---
                emb = batch.get("emb")
                target_val_from_batch = batch.get("val")
                val = None
                if emb is None or target_val_from_batch is None:
                     print(f"Error: Batch missing 'emb' or 'val' at step {global_step}. Skipping.")
                     global_step += 1; step_in_epoch += 1; progress_bar.update(1); continue
                emb = emb.to(TARGET_DEV)

                try: # Prepare val target tensor
                     if args.arch == "score": val = target_val_from_batch.to(TARGET_DEV, dtype=torch.float32).squeeze()
                     elif args.arch == "class":
                         if isinstance(criterion, nn.BCEWithLogitsLoss): val = target_val_from_batch.to(dtype=torch.float32, device=TARGET_DEV).squeeze()
                         else: val = target_val_from_batch.squeeze().to(dtype=torch.long, device=TARGET_DEV)
                     if val.ndim == 0: val = val.unsqueeze(0)
                except Exception as e:
                     print(f"Error processing target 'val' at step {global_step}: {e}")
                     global_step += 1; step_in_epoch += 1; progress_bar.update(1); continue
                # --- End Batch Data ---

                # --- Forward/Backward Pass (remains same, uses accumulated loss) ---
                loss = torch.tensor(0.0, device=TARGET_DEV)
                optimizer_stepped = False
                y_pred_for_loss = None
                try:
                    if not model.training: model.train()
                    with torch.amp.autocast(device_type=TARGET_DEV, enabled=enabled_amp, dtype=amp_dtype):
                        y_pred = model(emb)
                        y_pred_for_loss = y_pred
                        if args.arch == "score" or isinstance(criterion, nn.BCEWithLogitsLoss):
                            if y_pred.ndim == 2 and y_pred.shape[1] == 1: y_pred_for_loss = y_pred.squeeze(1)

                        # Shape check
                        if not isinstance(criterion, (nn.CrossEntropyLoss, FocalLoss, nn.NLLLoss, GHMC_Loss)):
                            if hasattr(criterion, 'reduction') and criterion.reduction != 'none' and y_pred_for_loss.shape != val.shape:
                                print(f"ERROR Shape mismatch step {global_step}: Pred {y_pred_for_loss.shape}, Target {val.shape}. Skipping.")
                                global_step += 1; step_in_epoch += 1; progress_bar.update(1); continue

                        loss_input = y_pred_for_loss.to(torch.float32)
                        if isinstance(criterion, nn.NLLLoss): loss_input = F.log_softmax(loss_input, dim=-1)
                        loss = criterion(loss_input, val.to(loss_input.device))

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/Inf loss at step {global_step}. Skipping BWD/step.")
                        loss = None

                    if loss is not None:
                        accumulated_loss += loss.detach()
                        accumulation_steps += 1

                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer_stepped = True

                except Exception as e:
                    print(f"Error in FWD/BWD step {global_step}: {e}")
                    import traceback; traceback.print_exc()
                    global_step += 1; step_in_epoch += 1; progress_bar.update(1); continue
                # --- End Forward/Backward ---

                # --- Scheduler Step ---
                if scheduler is not None and optimizer_stepped: scheduler.step()
                # --- End Scheduler Step ---

                # --- Update Counters ---
                global_step += 1
                step_in_epoch += 1 # Crucially, increment steps *within* this epoch
                progress_bar.update(1)
                wrapper.update_step(global_step)
                # --- End Update Counters ---

                # --- Logging, Saving, Postfix Update (remains same, based on global_step) ---
                if global_step % LOG_EVERY_N == 0 and global_step > 0:
                    # Calculate Average Loss
                    avg_loss_value = float('nan')
                    if accumulation_steps > 0:
                        avg_loss_tensor = accumulated_loss / accumulation_steps
                        avg_loss_value = avg_loss_tensor.item()
                        accumulated_loss.zero_(); accumulation_steps = 0

                    # Perform validation
                    eval_loss_val = float('nan')
                    if val_loader:
                        eval_loss_val = wrapper.evaluate_on_validation_set(val_loader)
                        if not model.training: model.train() # Ensure back in train mode

                    # Log metrics
                    if math.isnan(eval_loss_val): print(f"Warning: Eval loss is NaN at Global Step {global_step}.")
                    wrapper.log_main(step=global_step, train_loss_batch=avg_loss_value, eval_loss=eval_loss_val)

                    # Update Progress Bar Postfix
                    lr = optimizer.param_groups[0]['lr']
                    postfix_data = {"Epoch": epoch + 1, "AvgLoss": avg_loss_value}
                    if not math.isnan(eval_loss_val): postfix_data["EvalLoss"] = eval_loss_val
                    postfix_data["LR"] = lr
                    progress_bar.set_postfix(postfix_data, refresh=False)

                    # Save best model
                    if not math.isnan(eval_loss_val) and eval_loss_val < best_eval_loss:
                        best_eval_loss = eval_loss_val
                        print(f"\nNew best validation loss: {best_eval_loss:.4e} at Global Step {global_step}. Saving best model...")
                        wrapper.save_model(step=global_step, epoch=epoch, suffix="_best_val")

                # Periodic saving
                if args.nsave > 0 and global_step % args.nsave == 0:
                     if global_step > initial_global_step or global_step == args.nsave:
                          print(f"\nSaving periodic checkpoint at Global Step {global_step}...")
                          wrapper.save_model(step=global_step, epoch=epoch, save_aux=False)
                # --- End Logging and Saving ---

            # --- End Inner Step Loop (while step_in_epoch < steps_per_epoch) ---

            # Check global step limit again after inner loop finishes
            if global_step >= total_steps_to_run:
                break

        # --- End Outer Epoch Loop ---

    except KeyboardInterrupt:
         print("\nTraining interrupted by user.")
    finally:
        progress_bar.close()
        print(f"\nTraining loop finished. Reached Global Step: {global_step}")

# ================================================
#        Main Execution Block
# ================================================
def main():
    """Main function to run the training process."""
    # 1. Load base config from YAML + CMD overrides
    args = parse_and_load_args()
    print(f"Target device: {TARGET_DEV}")

    # 2. Setup basic components (precision, logging, data)
    amp_dtype, enabled_amp = setup_precision(args)
    wandb_run = setup_wandb(args)
    dataset, train_loader, val_loader = setup_dataloaders(args)

    # 3. Calculate final training duration (steps/epochs) based on loader size
    if not train_loader: exit("Error: Train loader is empty.")
    try:
         args.steps_per_epoch = len(train_loader)
         if args.steps_per_epoch == 0: raise ValueError
    except (TypeError, ValueError):
         exit("Error: Could not determine train loader length (steps_per_epoch).")

    if args.max_train_steps is not None and args.max_train_steps > 0:
        # Prioritize max_train_steps
        args.num_train_epochs = math.ceil(args.max_train_steps / args.steps_per_epoch)
        # Keep args.max_train_steps as the primary limit
    elif args.max_train_epochs is not None and args.max_train_epochs > 0:
        # Calculate steps from epochs
        args.num_train_epochs = args.max_train_epochs
        args.max_train_steps = args.num_train_epochs * args.steps_per_epoch
    else:
        exit("Error: Must specify either max_train_epochs or max_train_steps in config.")
    # Now args.num_train_epochs and args.max_train_steps are finalized

    # 4. Setup Model, Criterion, Optimizer, Scheduler (using final args)
    model, criterion = setup_model_criterion(args, dataset)
    # setup_optimizer_scheduler uses args.max_train_steps for scheduler defaults
    optimizer, scheduler, is_schedule_free = setup_optimizer_scheduler(args, model)
    scaler = torch.amp.GradScaler(device=TARGET_DEV, enabled=(enabled_amp and TARGET_DEV == 'cuda'))

    # 5. Load Checkpoint state (finds start epoch and global step)
    start_epoch, initial_global_step = load_checkpoint(args, model, optimizer, scheduler, scaler)

    # 6. Write Final Config (now includes calculated steps/epochs)
    print("\n--- Final Calculated Args ---")
    for k, v in sorted(vars(args).items()): print(f"  {k}: {v}")
    print("--------------------------\n")
    write_config(args) # Uses updated function from utils.py

    # 7. Setup Wrapper
    wrapper = ModelWrapper(
        name=args.name, model=model, device=TARGET_DEV,
        num_labels=getattr(args, 'num_labels', 1), criterion=criterion,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        wandb_run=wandb_run # Pass WandB run object
        # log_file_path is handled internally now
    )
    # Optional: Restore best_val_loss to wrapper if loaded from state?
    # wrapper.best_val_loss = loaded_best_loss_from_state_if_available

    # 8. Run Training Loop
    try:
        train_loop(args, model, criterion, optimizer, scheduler, scaler,
                   train_loader, val_loader, wrapper,
                   start_epoch, initial_global_step,  # Pass correct start points
                   enabled_amp, amp_dtype)
    finally:
        # 9. Final Save & Cleanup
        print(f"Saving final model...")
        final_step = wrapper.get_current_step() # Get final global step
        wrapper.save_model(step=final_step, epoch="final", suffix="_final", save_aux=False) # Save aux for final
        wrapper.close() # Close logs
        if wandb_run: wandb_run.finish() # End WandB run
        print("Training script finished.")

if __name__ == "__main__":
    main()