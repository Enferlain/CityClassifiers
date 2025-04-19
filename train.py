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

            # v2.2.1: Corrected logic for BCE/CE/Focal target shapes and dtypes
            target_val_from_batch = batch.get("val") # Get original tensor first
            if target_val_from_batch is None:
                print(f"Error: 'val' is None in batch at step {current_step}. Skipping.")
                progress.update(args.batch); continue

            val = None # Define val outside try block
            try:
                # --- Score Architecture ---
                if args.arch == "score":
                    # Needs Float target, usually shape [B] to match model output
                    val = target_val_from_batch.to(TARGET_DEV, dtype=torch.float32)
                    # Ensure shape is [B] if model output is [B]
                    # (Check y_pred shape *after* model call, before loss)
                    if val.ndim == 2 and val.shape[1] == 1:
                        val = val.squeeze(1)
                    elif val.ndim != 1:
                        val = val.squeeze() # General squeeze if needed
                    # Final check might be needed after y_pred is known

                # --- Class Architecture ---
                elif args.arch == "class":
                    if isinstance(criterion, nn.BCEWithLogitsLoss):
                        # BCE expects Float target with shape matching input (usually [B] for num_classes=1)
                        val = target_val_from_batch.to(dtype=torch.float32, device=TARGET_DEV)
                        if val.ndim == 2 and val.shape[1] == 1:
                            val = val.squeeze(1) # [B, 1] -> [B]
                        elif val.ndim != 1: # Handle other cases like [B] or maybe errors
                            val = val.squeeze()
                        if val.ndim != 1: # Final check for 1D
                             raise ValueError(f"BCE target shape error. Expected 1D [B], got {val.shape}")

                    else:  # CrossEntropy / Focal
                        # Expects Long target with shape [B]
                        val = target_val_from_batch.squeeze().to(dtype=torch.long, device=TARGET_DEV)
                        if val.ndim != 1: # Final check for 1D
                            raise ValueError(f"CE/Focal target shape error. Expected 1D [B], got {val.shape}")

                else: # Should not happen if arch is validated earlier
                     raise ValueError(f"Unknown args.arch '{args.arch}' during target prep.")

            except Exception as e:
                print(f"Error processing target tensor 'val' at step {current_step}: {e}")
                print(f"  Original val shape: {batch.get('val').shape}, dtype: {batch.get('val').dtype}")
                progress.update(args.batch); continue
            # --- End Target Prep ---

            # --- Get Model Prediction ---
            # Do this *after* initial target checks, but *before* final shape check/loss
            y_pred_for_loss = None
            try:
                 with torch.amp.autocast(device_type=TARGET_DEV, enabled=enabled_amp, dtype=amp_dtype):
                     y_pred = model(emb) # Prediction on device

                 # Prepare prediction shape to match target shape expected by loss
                 # For num_classes=1 (BCE/Score), model outputs [B] after squeeze in forward.
                 # For num_classes>1 (CE/Focal), model outputs [B, C].
                 # BCE/L1/MSE expect [B] input if target is [B].
                 # CE/Focal expect [B, C] input if target is [B].

                 if isinstance(criterion, nn.BCEWithLogitsLoss):
                      y_pred_for_loss = y_pred # Expects [B] from model
                 elif isinstance(criterion, (nn.L1Loss, nn.MSELoss)):
                      y_pred_for_loss = y_pred # Expects [B] from model (assuming Tanh scaled output)
                 else: # CrossEntropy / Focal
                      y_pred_for_loss = y_pred # Expects [B, C] from model

                 if y_pred_for_loss is None: raise ValueError("y_pred_for_loss is None") # Safety

            except Exception as e_pred:
                 print(f"Error during model prediction at step {current_step}: {e_pred}")
                 progress.update(args.batch); continue
            # --- End Model Prediction ---

            # --- Final Shape Check ---
            if val is None: # Safety check
                 print(f"Error: Target tensor 'val' is None before loss calculation. Skipping.")
                 progress.update(args.batch); continue

            # --- FIX: Add NLLLoss to the expected mismatch check ---
            if y_pred_for_loss.shape != val.shape:
                 # Specific check for losses expecting [B, C] input and [B] target
                 if isinstance(criterion, (nn.CrossEntropyLoss, FocalLoss, nn.NLLLoss, GHMC_Loss)):
                      if y_pred_for_loss.ndim == 2 and val.ndim == 1 and y_pred_for_loss.shape[0] == val.shape[0]:
                           # This shape mismatch IS EXPECTED for CE/Focal/NLL ([B, C] vs [B])
                           pass # Don't print error
                      else:
                            # Other mismatches ARE errors for CE/Focal/NLL
                            print(f"ERROR: Shape mismatch before {type(criterion).__name__} loss! Input: {y_pred_for_loss.shape}, Target: {val.shape}. Skipping step.")
                            progress.update(args.batch); continue
                 else:
                      # For other losses (BCE, L1, MSE), shapes MUST match
                      print(f"ERROR: Shape mismatch before {type(criterion).__name__} loss! Input: {y_pred_for_loss.shape}, Target: {val.shape}. Skipping step.")
                      progress.update(args.batch); continue
            # --- End Final Shape Check ---

            # --- Loss Calculation & Backward Pass ---
            try:
                 with torch.amp.autocast(device_type=TARGET_DEV, enabled=enabled_amp, dtype=amp_dtype):
                     # y_pred_for_loss should have shape [B, C] and contain linear logits
                     # Target 'val' should have shape [B] and contain Long class indices

                     loss_input = y_pred_for_loss.to(torch.float32) # Ensure float32 logits

                     # --- FIX: Apply LogSoftmax for NLLLoss ---
                     if isinstance(criterion, nn.NLLLoss):
                          # NLLLoss requires log-probabilities! Apply LogSoftmax.
                          print("DEBUG train_loop: Applying LogSoftmax before NLLLoss.")
                          loss_input = F.log_softmax(loss_input, dim=-1)
                     # --- END FIX ---
                     # Other losses (CE, Focal, BCE) work directly on logits (y_pred_for_loss)
                     # L1/MSE might need specific output modes handled earlier or need y_pred here

                     # Pass log-probabilities (for NLL) or logits (for others) to criterion
                     loss = criterion(loss_input, val)

                 if torch.isnan(loss) or torch.isinf(loss):
                     print(f"Warning: NaN or Inf loss detected at step {current_step}. Skipping step.")
                     progress.update(args.batch)
                     continue

                 optimizer.zero_grad(set_to_none=True)
                 scaler.scale(loss).backward()
                 scaler.step(optimizer)
                 scaler.update()

                 if scheduler is not None:
                     scheduler.step()

            except Exception as e_loss:
                 print(f"Error during loss calculation or backward pass at step {current_step}: {e_loss}")
                 print(f"  Input shape: {y_pred_for_loss.shape}, Target shape: {val.shape}, Target dtype: {val.dtype}")
                 progress.update(args.batch); continue
            # --- End Loss & Backward ---

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