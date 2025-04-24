# Version: 2.3.0 (Handles E2E Dataset Loading)
import collections
import inspect
import os
import sys
import traceback

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import wandb
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor # <<< Added AutoProcessor import

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
    ModelWrapper, get_embed_params, parse_and_load_args, write_config,
    FocalLoss, load_optimizer_state, load_scheduler_state, load_scaler_state # Add load helpers if needed here
)
# Import datasets conditionally or both if needed
from dataset import EmbeddingDataset
try:
    from image_dataset import ImageFolderDataset, collate_group_by_size
    IMAGE_DATASET_AVAILABLE = True
except ImportError:
    print("Warning: image_dataset.py not found. End-to-end training will not be available.")
    IMAGE_DATASET_AVAILABLE = False

from model_early_extract import EarlyExtractAnatomyModel
from model import PredictorModel
from losses import GHMC_Loss # Keep loss imports here

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

# Version 2.3.0: Updated setup_dataloaders function
def setup_dataloaders(args, image_processor=None):
    """
    Sets up the dataset and dataloaders based on args.
    Uses ImageFolderDataset if args.is_end_to_end is True, otherwise EmbeddingDataset.
    """
    data_root_path = getattr(args, 'data_root', 'data')
    dataset = None
    train_loader = None
    val_loader = None
    collate_fn = None # Define collate_fn based on dataset type

    if getattr(args, 'is_end_to_end', False):
        # --- End-to-End Path ---
        print("Setting up ImageFolderDataset for end-to-end training.")
        if not IMAGE_DATASET_AVAILABLE:
             exit("Error: image_dataset.py is required for end-to-end training but could not be imported.")
        if image_processor is None:
            exit("Error: Image processor is required for ImageFolderDataset.")

        # The root directory should contain the class subfolders (e.g., data/0/, data/1/)
        image_data_dir = data_root_path
        print(f"Looking for class folders (0, 1, ...) directly inside: {image_data_dir}")

        try:
            dataset = ImageFolderDataset(
                root_dir=image_data_dir,
                transform=image_processor, # Pass the loaded processor!
                validation_split_count=args.val_split_count,
                seed=getattr(args, 'seed', 218)
            )
            # Update args with num_labels discovered by dataset
            args.num_labels = dataset.num_labels
            print(f"DEBUG: Updated args.num_labels from ImageFolderDataset: {args.num_labels}")
            # <<< Use the new collate function >>>
            collate_fn = collate_group_by_size
            print("DEBUG: Using collate_group_by_size for end-to-end dataloader.")
        except Exception as e:
            print(f"Error creating ImageFolderDataset: {e}")
            print(f"Check if class folders (e.g., '0', '1') exist directly under: {image_data_dir}")
            exit(1)

    else:
        # --- Embedding Path (Existing Logic) ---
        if not hasattr(args, 'embed_ver') or not args.embed_ver:
             exit("Error: 'embed_ver' is required for embedding-based training.")
        dataset_version = args.embed_ver # This is the subfolder name within data_root
        embedding_data_dir = os.path.join(data_root_path, dataset_version)
        print(f"Setting up EmbeddingDataset using version folder: {embedding_data_dir}")

        try:
            # Note: EmbeddingDataset init takes the version folder *name*, not the full path.
            # The root argument points to the parent dir ('data_root_path').
            dataset = EmbeddingDataset(
                ver=dataset_version, # Pass just the version name
                root=data_root_path, # Pass the parent directory
                mode=args.arch,
                preload=getattr(args, 'preload_data', True),
                validation_split_count=args.val_split_count,
                seed=getattr(args, 'seed', 218)
            )
            # Update args with num_labels discovered by dataset
            if args.arch == 'class':
                 args.num_labels = dataset.num_labels
                 print(f"DEBUG: Updated args.num_labels from EmbeddingDataset: {args.num_labels}")
            # <<< Use the old collate function (or None for default) >>>
            # EmbeddingDataset already provides collate_ignore_none, let's get it
            collate_fn = getattr(dataset, 'collate_ignore_none', torch.utils.data.dataloader.default_collate)
            print(f"DEBUG: Using {getattr(collate_fn, '__name__', 'default_collate')} for embedding dataloader.")

        except FileNotFoundError:
            print(f"Error: EmbeddingDataset root directory not found: {embedding_data_dir}")
            print(f"Ensure the folder '{dataset_version}' exists inside '{data_root_path}'.")
            exit(1)
        except Exception as e:
            print(f"Error creating EmbeddingDataset: {e}")
            exit(1)

    # --- Create DataLoaders ---
    if dataset is None:
         exit("Error: Dataset initialization failed.")
    if len(dataset) == 0:
         print("Warning: Training dataset is empty! Check data path and configuration.")
         # Optionally exit or continue, depending on desired behavior
         # exit("Exiting due to empty training dataset.")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        pin_memory=False, # Generally False for datasets returning dicts or PIL images
        num_workers=getattr(args, 'num_workers', 0),
        collate_fn=collate_fn # Use the determined collate function
    )

    val_loader = dataset.get_validation_loader(
        batch_size=args.batch, # Can often use larger batch for validation
        num_workers=getattr(args, 'num_workers', 0)
        # Validation loader within dataset should also use the robust collate_fn if needed
    )

    # Common print statement
    print(f"Created training loader with {len(train_loader)} batches ({len(dataset)} samples).")
    if val_loader and hasattr(val_loader, 'dataset') and len(val_loader.dataset) > 0:
        print(f"Created validation loader with {len(val_loader)} batches ({len(val_loader.dataset)} samples).")
    elif args.val_split_count > 0:
         print("Validation split requested, but validation loader is empty or could not be created.")
    else:
        print("No validation split requested or validation data available.")

    return dataset, train_loader, val_loader
# --- End setup_dataloaders ---

# Version 2.3.0: Handles E2E and Embedding model/criterion setup
def setup_model_criterion(args, dataset):
    """
    Sets up the model and criterion based on args.
    Instantiates either PredictorModel or EarlyExtractAnatomyModel.
    Determines criterion based on args.loss_function.
    """
    print("DEBUG setup_model_criterion: Setting up model and criterion...")

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
    model_output_mode = getattr(args, 'head_output_mode') if getattr(args, 'is_end_to_end', False) else getattr(args, 'output_mode')
    if model_output_mode is None: exit("Error: Model output mode is not defined in args.")
    model_output_mode = model_output_mode.lower()

    if loss_function_name == 'l1':
        print("DEBUG: Setting criterion to L1Loss.")
        criterion = nn.L1Loss(reduction='mean')
        if num_classes != 1: print(f"Warning: L1Loss typically used with num_classes=1, but found {num_classes}.")
        if model_output_mode not in ['tanh_scaled', 'sigmoid', 'linear']: print(f"Warning: L1Loss typically paired with scaled/linear output, got '{model_output_mode}'.")

    elif loss_function_name == 'mse':
        print("DEBUG: Setting criterion to MSELoss.")
        criterion = nn.MSELoss(reduction='mean')
        if num_classes != 1: print(f"Warning: MSELoss typically used with num_classes=1, but found {num_classes}.")
        if model_output_mode not in ['tanh_scaled', 'linear', 'sigmoid']: print(f"Warning: MSELoss typically paired with linear/scaled output, got '{model_output_mode}'.")

    elif loss_function_name == 'focal':
        print("DEBUG: Setting criterion to FocalLoss.")
        if num_classes <= 1: exit(f"Error: FocalLoss requires num_classes > 1, but found {num_classes}.")
        criterion = FocalLoss(gamma=getattr(args, 'focal_loss_gamma', 2.0))
        if model_output_mode != 'linear':
             print(f"Warning: FocalLoss usually expects output_mode='linear', but got '{model_output_mode}'. Training with non-logit input.")
             # Allow non-linear input, but loss calculation might be suboptimal.

    elif loss_function_name == 'crossentropy':
        print("DEBUG: Setting criterion to CrossEntropyLoss.")
        if num_classes <= 1: exit(f"Error: CrossEntropyLoss requires num_classes > 1, but found {num_classes}.")
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        if model_output_mode != 'linear':
             print(f"Warning: CrossEntropyLoss usually expects output_mode='linear', but got '{model_output_mode}'. Training with non-logit input.")

    elif loss_function_name == 'bce':
        print("DEBUG: Setting criterion to BCEWithLogitsLoss.")
        # BCEWithLogitsLoss *can* handle multi-label if target is multi-label float,
        # but for single-label classification, num_classes=1 is standard.
        if num_classes != 1:
            print(f"Warning: BCEWithLogitsLoss selected, but num_classes={num_classes}. Ensure targets are correctly formatted (e.g., multi-label floats) or consider using CrossEntropyLoss.")
        criterion = nn.BCEWithLogitsLoss(weight=class_weights_tensor if num_classes > 1 else None) # Weight applied per element for multi-label
        if model_output_mode != 'linear':
             print(f"Warning: BCEWithLogitsLoss expects output_mode='linear', but got '{model_output_mode}'. Training with non-logit input.")

    elif loss_function_name == 'nll':
        print("DEBUG: Setting criterion to NLLLoss (expects LogSoftmax input).")
        if num_classes <= 1: exit(f"Error: NLLLoss requires num_classes > 1, but found {num_classes}.")
        criterion = nn.NLLLoss(weight=class_weights_tensor)
        if model_output_mode != 'linear':
             # NLL *requires* log-probabilities. If the model isn't outputting logits, this won't work correctly.
             exit(f"Error: loss_function='nll' selected, but model output_mode is '{model_output_mode}'. Must be 'linear' for NLLLoss (LogSoftmax applied in train loop).")

    elif loss_function_name == 'ghm':
        print("DEBUG: Setting criterion to GHMC_Loss.")
        if num_classes <= 1: exit(f"Error: GHMC_Loss currently requires num_classes > 1, but found {num_classes}.")
        ghm_bins = getattr(args, 'ghm_bins', 10)
        ghm_momentum = getattr(args, 'ghm_momentum', 0.75)
        criterion = GHMC_Loss(bins=ghm_bins, momentum=ghm_momentum, reduction='mean')
        if model_output_mode != 'linear':
             exit(f"Error: GHMC_Loss requires output_mode='linear', but got '{model_output_mode}'.")
        if class_weights_tensor is not None: print("Warning: Class weights specified but GHMC_Loss does not use them directly.")

    else:
        raise ValueError(f"Unknown loss_function '{loss_function_name}' specified in config.")

    if criterion is None:
        exit("Error: Criterion setup failed.")

    # --- Instantiate the Correct Model ---
    model = None
    amp_dtype, enabled_amp = setup_precision(args) # Get compute dtype

    if getattr(args, 'is_end_to_end', False):
        print(f"DEBUG: Instantiating EarlyExtractAnatomyModel ...")
        try:
            # Pass device from args or global TARGET_DEV
            model = EarlyExtractAnatomyModel(
                base_model_name=args.base_vision_model,
                device=TARGET_DEV, # <<< Pass device >>>
                extract_layer=args.extract_layer,
                pooling_strategy=args.pooling_strategy,
                head_features=None,
                head_hidden_dim=args.head_hidden_dim,
                head_num_classes=num_classes,
                head_num_res_blocks=args.head_num_res_blocks,
                head_dropout_rate=args.head_dropout_rate,
                head_output_mode=args.head_output_mode,
                attn_pool_heads=getattr(args, 'attn_pool_heads', 8),
                attn_pool_dropout=getattr(args, 'attn_pool_dropout', 0.1),
                freeze_base_model=args.freeze_base_model,
                compute_dtype=amp_dtype
            )
        except Exception as e:
            print(f"Error details during EarlyExtractAnatomyModel instantiation: {e}")
            import traceback
            traceback.print_exc()
            exit(f"Error instantiating EarlyExtractAnatomyModel.")

    else: # Embedding Path
        print(f"DEBUG: Instantiating PredictorModel v2 (Output Classes: {num_classes}, Output Mode: {args.output_mode})")
        try:
            if args.features is None: exit("Error: Features not determined for embedding model.")
            model = PredictorModel(
                features=args.features,
                hidden_dim=args.hidden_dim, # Should be loaded by parse_args
                num_classes=num_classes, # Use final num_classes
                use_attention=args.use_attention,
                num_attn_heads=args.num_attn_heads,
                attn_dropout=args.attn_dropout,
                num_res_blocks=args.num_res_blocks,
                dropout_rate=args.dropout_rate,
                output_mode=args.output_mode # Use predictor output mode
            )
        except Exception as e:
            print(f"Error details during PredictorModel instantiation: {e}")
            import traceback
            traceback.print_exc()
            exit(f"Error instantiating PredictorModel.")

    if model is None:
        exit("Model instantiation failed.")

    model.to(TARGET_DEV)
    print("Model and criterion setup complete.")

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
#        Main Training Loop (Epoch-Based)
# ================================================
# Version 3.5.2: Uses (loss / num_sub_batches).backward() for grad accum
def train_loop(args, model, criterion, optimizer, scheduler, scaler,
               train_loader, val_loader, wrapper, start_epoch, initial_global_step,
               enabled_amp, amp_dtype, is_schedule_free):
    """
    Runs the main training loop.
    Handles list of mini-batches using (loss / num_sub_batches).backward().
    """
    # --- Initial Setup ---
    if not hasattr(args, 'num_train_epochs') or not hasattr(args, 'steps_per_epoch') or not hasattr(args, 'max_train_steps'):
         print("ERROR: num_train_epochs, steps_per_epoch, or max_train_steps missing from args.")
         return

    model.train()
    if hasattr(optimizer, 'train') and callable(optimizer.train):
        try: optimizer.train()
        except Exception as e: print(f"Warning: Error calling optimizer.train(): {e}")

    total_steps_to_run = args.max_train_steps
    total_epochs_to_run = args.num_train_epochs
    steps_per_epoch = args.steps_per_epoch
    is_e2e = getattr(args, 'is_end_to_end', False)
    # Use LOG_EVERY_N constant from utils.py if available, else default
    log_every_n = getattr(args, 'log_every_n', 1) # Use 10 as default?
    validate_every_n = getattr(args, 'validate_every_n', 50) # Use 150 as default?

    print(f"Starting training loop. Target Epochs: {total_epochs_to_run}, Target Steps: {total_steps_to_run}")
    print(f"Steps per epoch: {steps_per_epoch}. End-to-End Mode: {is_e2e}. Logging every {log_every_n} steps.")

    global_step = initial_global_step
    best_eval_loss = float('inf')
    if hasattr(wrapper, 'best_val_loss') and wrapper.best_val_loss != float('inf'):
         best_eval_loss = wrapper.best_val_loss
         print(f"Retrieved best validation loss from wrapper: {best_eval_loss:.4e}")

    # <<< INITIALIZE last_eval_loss_val HERE >>>
    last_eval_loss_val = float('nan')

    progress_bar = tqdm(initial=initial_global_step, total=total_steps_to_run, desc="Overall Training", unit="step", dynamic_ncols=True)

    accumulated_loss_for_log = torch.tensor(0.0, device=TARGET_DEV)
    accumulation_steps_for_log = 0

    print("Initializing DataLoader iterator...")
    train_iterator = iter(train_loader)

    # --- Outer Epoch Loop ---
    try:
        for epoch in range(start_epoch, total_epochs_to_run):
            if global_step >= total_steps_to_run: break

            if epoch == start_epoch or (epoch + 1) % 50 == 0 or epoch == total_epochs_to_run - 1:
                 print(f"\n--- Starting Epoch {epoch + 1} / {total_epochs_to_run} (Global Step: {global_step}) ---")
            wrapper.current_epoch = epoch

            model.train()
            if hasattr(optimizer, 'train') and callable(optimizer.train): optimizer.train()

            step_in_epoch = 0
            # --- Inner Step Loop (Conceptual Batches from DataLoader) ---
            while step_in_epoch < steps_per_epoch:
                if global_step >= total_steps_to_run: break

                try: batch_data = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    try: batch_data = next(train_iterator)
                    except StopIteration: print("ERROR: DataLoader empty after restart."); global_step = total_steps_to_run; break
                except Exception as e_iter: print(f"\nError getting batch data: {e_iter}"); global_step = total_steps_to_run; break

                if global_step < initial_global_step:
                    global_step += 1; progress_bar.update(1); continue

                if batch_data is None or not batch_data:
                    print(f"Warning: Skipping step {global_step} due to invalid batch_data.")
                    global_step += 1; step_in_epoch += 1; progress_bar.update(1); continue

                if isinstance(batch_data, dict): batch_data_list = [batch_data]
                elif isinstance(batch_data, list): batch_data_list = batch_data
                else:
                     print(f"Warning: Unexpected batch_data type {type(batch_data)} at step {global_step}. Skipping.")
                     global_step += 1; step_in_epoch += 1; progress_bar.update(1); continue

                optimizer_stepped = False # Initialize for this conceptual batch
                optimizer.zero_grad(set_to_none=True)

                processed_samples_in_step = 0
                minibatch_losses = []
                num_sub_batches = len(batch_data_list)
                if num_sub_batches == 0: # Should be caught earlier, but safety check
                    print(f"Warning: Empty batch_data_list at step {global_step}. Skipping.")
                    global_step += 1; step_in_epoch += 1; progress_bar.update(1); continue

                # --- Innermost Loop: Iterate through Mini-Batches ---
                for sub_batch in batch_data_list:
                    loss = torch.tensor(float('nan'), device=TARGET_DEV)
                    try:
                        if not model.training: model.train()
                        if hasattr(optimizer, 'train') and callable(optimizer.train): optimizer.train()

                        target_val_from_sub_batch = sub_batch.get("label" if is_e2e else "val")
                        if target_val_from_sub_batch is None: continue

                        model_input_dict = {}; emb_input = None; current_sub_batch_size = 0
                        if is_e2e:
                            pixel_values = sub_batch.get("pixel_values")
                            if pixel_values is None: continue
                            model_input_dict["pixel_values"] = pixel_values.to(TARGET_DEV)
                            current_sub_batch_size = pixel_values.size(0)
                        else:
                            emb_input = sub_batch.get("emb")
                            if emb_input is None: continue
                            emb_input = emb_input.to(TARGET_DEV); current_sub_batch_size = emb_input.size(0)

                        try: # Prepare target tensor
                            if args.num_classes == 1: target = target_val_from_sub_batch.to(TARGET_DEV, dtype=torch.float32)
                            else: target = target_val_from_sub_batch.to(TARGET_DEV, dtype=torch.long)
                            if target.shape[0] != current_sub_batch_size: target = target.view(current_sub_batch_size, -1).squeeze()
                            if target.shape[0] != current_sub_batch_size: raise ValueError("Target shape mismatch")
                        except Exception as e_target: print(f"Error processing target: {e_target}"); continue

                        # --- Sub-Batch Forward/Backward ---
                        with torch.amp.autocast(device_type=TARGET_DEV, enabled=enabled_amp, dtype=amp_dtype):
                            if is_e2e: y_pred = model(**model_input_dict)
                            else: y_pred = model(emb_input)

                            y_pred_for_loss = y_pred
                            if isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)) and args.num_classes == 1:
                                if y_pred.ndim > 1 and y_pred.shape[1] == 1: y_pred_for_loss = y_pred.squeeze(-1)

                            loss_input = y_pred_for_loss.to(torch.float32)
                            target_for_loss = target.to(loss_input.device)

                            if isinstance(criterion, nn.NLLLoss): loss_input = F.log_softmax(loss_input.unsqueeze(0), dim=-1).squeeze(0) # Check NLLLoss input shape requirements

                            loss = criterion(loss_input, target_for_loss.long() if isinstance(criterion, (nn.CrossEntropyLoss, FocalLoss, nn.NLLLoss, GHMC_Loss)) else target_for_loss.float())

                        if torch.isnan(loss) or torch.isinf(loss): loss = None
                        else:
                            minibatch_losses.append(loss.detach()) # Log detached loss
                            processed_samples_in_step += current_sub_batch_size
                            # <<< Gradient Accumulation by averaging loss before backward >>>
                            scaler.scale(loss / num_sub_batches).backward()
                            # <<< End Change >>>

                    except Exception as e:
                        print(f"\nError processing sub-batch step {global_step}: {e}")
                        traceback.print_exc(); continue
                # --- End Innermost Loop ---

                # --- Optimizer Step ---
                if processed_samples_in_step > 0:
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer_stepped = True
                    except Exception as e_optim: print(f"\nError optimizer step {global_step}: {e_optim}")
                else:
                    print(f"Warning: No samples processed for step {global_step}. Skipping optimizer step.")
                    optimizer.zero_grad(set_to_none=True) # Still zero grads

                # --- Scheduler Step ---
                if scheduler is not None and not is_schedule_free and optimizer_stepped:
                     try: scheduler.step()
                     except Exception as e_sched: print(f"Warning: Error scheduler step: {e_sched}")

                # --- Accumulate Loss for Logging ---
                if minibatch_losses:
                     avg_step_loss = torch.mean(torch.stack(minibatch_losses)).item()
                     if not math.isnan(avg_step_loss):
                         accumulated_loss_for_log += avg_step_loss; accumulation_steps_for_log += 1

                # --- Update Counters & Logging/Saving ---
                global_step += 1; step_in_epoch += 1; progress_bar.update(1); wrapper.update_step(global_step)

                # --- Frequent Logging & Postfix Update ---
                if global_step % log_every_n == 0 and global_step > 0:
                    avg_loss_value_log = float('nan')
                    if accumulation_steps_for_log > 0:
                        avg_loss_value_log = (accumulated_loss_for_log / accumulation_steps_for_log).item()
                    accumulated_loss_for_log.zero_(); accumulation_steps_for_log = 0

                    # Call log_main to log to WandB/CSV and update internal state
                    # Pass the last known eval loss
                    wrapper.log_main(step=global_step, train_loss_batch=avg_loss_value_log, eval_loss=last_eval_loss_val)

                    # Update Progress Bar Postfix
                    lr = optimizer.param_groups[0]['lr']
                    # <<< Build the postfix string >>>
                    postfix_dict = collections.OrderedDict() # Use ordered dict for consistent order
                    postfix_dict["Epoch"] = epoch + 1
                    postfix_dict["AvgLoss"] = f"{avg_loss_value_log:.3e}" if not math.isnan(avg_loss_value_log) else "N/A"
                    # Show last eval loss if available
                    if not math.isnan(last_eval_loss_val): postfix_dict["LastEval"] = f"{last_eval_loss_val:.3e}"
                    postfix_dict["LR"] = f"{lr:.1e}"
                    progress_bar.set_postfix(ordered_dict=postfix_dict, refresh=False) # Use ordered_dict arg

                # --- Less Frequent Validation & Best Model Saving ---
                if global_step % validate_every_n == 0 and global_step > 0:
                    print(f"\n--- Running Validation @ Step {global_step} ---") # Indicate validation start
                    eval_loss_val = float('nan')
                    if val_loader:
                        eval_loss_val = wrapper.evaluate_on_validation_set(val_loader, args)
                        last_eval_loss_val = eval_loss_val # Update last known eval loss
                        if not model.training: model.train()

                    # Log validation loss specifically (if using WandB etc)
                    if wrapper.wandb_run and not math.isnan(eval_loss_val):
                         try: wrapper.wandb_run.log({"eval/loss": eval_loss_val}, step=global_step)
                         except Exception as e: print(f"Wandb eval log error: {e}")

                    if math.isnan(eval_loss_val): print(f"Warning: Eval loss is NaN at Step {global_step}.")
                    else: print(f"--- Validation Complete @ Step {global_step}: Eval Loss = {eval_loss_val:.4e} ---")

                    # Check for best model only after validation
                    if not math.isnan(eval_loss_val) and eval_loss_val < best_eval_loss:
                        best_eval_loss = eval_loss_val
                        print(f"New best val loss: {best_eval_loss:.4e}. Saving best model...")
                        wrapper.save_model(step=global_step, epoch=epoch, suffix="_best_val", save_aux=False, args=args)

                # --- Periodic saving (independent of validation) ---
                if args.nsave > 0 and global_step % args.nsave == 0:
                     if global_step > initial_global_step:
                          print(f"\nSaving periodic checkpoint @ step {global_step}...")
                          wrapper.save_model(step=global_step, epoch=epoch, args=args)

            # --- End Inner Step Loop ---
            if global_step >= total_steps_to_run: break
        # --- End Outer Epoch Loop ---

    except KeyboardInterrupt: print("\nTraining interrupted by user.")
    finally: progress_bar.close(); print(f"\nTraining loop finished. Reached Global Step: {global_step}")

# ================================================
#        Main Execution Block (Updated)
# ================================================
# Version 2.3.0: Updated main execution flow
def main():
    """Main function to run the training process."""
    # 1. Load base config from YAML + CMD overrides
    args = parse_and_load_args() # Should now contain is_end_to_end and relevant params
    print(f"Target device: {TARGET_DEV}")

    # Set seed AFTER parsing args
    seed = getattr(args, 'seed', 0) # Default seed 0 if not specified
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # Seed all GPUs
    print(f"Set random seed to: {seed}")

    # 2. Setup basic components (precision, logging)
    amp_dtype, enabled_amp = setup_precision(args)
    wandb_run = setup_wandb(args) # Needs args.name to be set

    # <<< NEW: Load Processor BEFORE Dataset if End-to-End >>>
    image_processor = None
    if getattr(args, 'is_end_to_end', False):
        print("DEBUG Main: Loading processor for end-to-end model...")
        if not args.base_vision_model:
             exit("Error: base_vision_model needed for end-to-end processor loading.")
        try:
             # Load processor using AutoProcessor
             # Added trust_remote_code=True as AIMv2 might require it
             image_processor = AutoProcessor.from_pretrained(args.base_vision_model, trust_remote_code=True)
             print(f"Loaded processor: {image_processor.__class__.__name__}")
             # TODO: Verify if AutoProcessor handles AIMv2 Native correctly or if manual transforms are needed.
             # e.g., from aim.v2.utils import val_transforms
             # requires: pip install 'git+https://github.com/apple/ml-aim.git#subdirectory=aim-v2'
        except ImportError:
            exit("Error: 'transformers' library not found or AutoProcessor failed. Is it installed?")
        except Exception as e:
             exit(f"Error loading image processor for {args.base_vision_model}: {e}")

    # 3. Setup Dataloaders (Pass processor)
    # This function now updates args.num_labels based on the dataset
    dataset, train_loader, val_loader = setup_dataloaders(args, image_processor)

    # 4. Calculate final training duration (after dataset length is known)
    if not train_loader: exit("Error: Train loader is empty or failed to initialize.")
    try:
         # If drop_last=True, len(train_loader) is floor(num_samples / batch_size)
         # If drop_last=False, it's ceil(num_samples / batch_size)
         # Using len(train_loader) directly is simpler.
         args.steps_per_epoch = len(train_loader)
         if args.steps_per_epoch == 0: raise ValueError("Train loader length is zero.")
    except (TypeError, ValueError) as e:
         exit(f"Error: Could not determine train loader length (steps_per_epoch): {e}.")

    if args.max_train_steps is not None and args.max_train_steps > 0:
        args.num_train_epochs = math.ceil(args.max_train_steps / args.steps_per_epoch)
    elif args.max_train_epochs is not None and args.max_train_epochs > 0:
        args.num_train_epochs = args.max_train_epochs
        args.max_train_steps = args.num_train_epochs * args.steps_per_epoch
    else:
        # Should have been caught by parse_and_load_args default/validation
        exit("Error: Training duration (max_train_steps or max_train_epochs) not properly set.")

    # 5. Setup Model, Criterion (Needs implementation)
    # setup_model_criterion uses args (including num_labels updated by dataset)
    model, criterion = setup_model_criterion(args, dataset)
    if model is None or criterion is None: exit("Error: Model or Criterion setup failed.")

    # 6. Setup Optimizer, Scheduler (Needs implementation)
    optimizer, scheduler, is_schedule_free = setup_optimizer_scheduler(args, model)
    if optimizer is None: exit("Error: Optimizer setup failed (using placeholders).")

    scaler = torch.amp.GradScaler(device=TARGET_DEV, enabled=(enabled_amp and TARGET_DEV == 'cuda'))

    # 7. Load Checkpoint state (Needs implementation)
    start_epoch, initial_global_step = load_checkpoint(args, model, optimizer, scheduler, scaler)

    # 8. Write Final Config (now includes calculated steps/epochs & dataset labels)
    print("\n--- Final Calculated Args ---")
    for k, v in sorted(vars(args).items()): print(f"  {k}: {v}")
    print("--------------------------\n")
    write_config(args) # Uses updated function from utils.py

    # 9. Setup Wrapper
    wrapper = ModelWrapper(
        name=args.name, model=model, device=TARGET_DEV,
        num_labels=getattr(args, 'num_labels', 1), criterion=criterion,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        wandb_run=wandb_run
    )
    # Optional: Restore best_val_loss here if loaded from state

    # 10. Run Training Loop (Needs implementation)
    try:
        train_loop(args, model, criterion, optimizer, scheduler, scaler,
                   train_loader, val_loader, wrapper,
                   start_epoch, initial_global_step,
                   enabled_amp, amp_dtype,
                   is_schedule_free)
    finally:
        # 11. Final Save & Cleanup
        print(f"Saving final model...")
        final_step = wrapper.get_current_step()
        # Save aux=True for final to allow easier resuming/analysis
        wrapper.save_model(step=final_step, epoch="final", suffix="_final", save_aux=False, args=args)
        wrapper.close()
        if wandb_run: wandb_run.finish()
        print("Training script finished.")

if __name__ == "__main__":
    main()