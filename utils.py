# Version: 2.0.0
# Desc: Integrate wandb logging and more args

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
		# CLIPVisionModelWithProjection
		#  openai/clip-vit-large-patch14-336
		return {
			"features": 768,
			"hidden": 1024,
		}
	elif ver == "META":
		# open_clip
		#  metaclip_fullcc | ViT-H-14-quickgelu
		print("META ver. was only meant for testing!")
		return {
			"features": 1024,
			"hidden": 1280,
		}
	else:
		raise ValueError(f"Unknown model '{ver}'")


# --- Modify parse_args ---
def parse_args():
	parser = argparse.ArgumentParser(description="Train aesthetic predictor/classifier")  # Updated description slightly
	parser.add_argument("--config", required=True, help="Training config YAML file")
	parser.add_argument('--resume', help="Checkpoint (.safetensors model file) to resume from")
	parser.add_argument('--images', action=argparse.BooleanOptionalAction, default=False,
						help="Live process images instead of using pre-computed embeddings")
	parser.add_argument("--nsave", type=int, default=10000,
						help="Save model every N steps (set to 0 or negative to disable)")  # Give a default like 10k?

	# --- NEW ARGUMENTS START HERE ---

	parser.add_argument('--optimizer', type=str, default='AdamW',
						choices=['AdamW', 'FMARSCropV3ExMachina', 'ADOPT',
								 'ADOPTScheduleFree', 'ADOPTAOScheduleFree'],  # Add more as needed
						help='Optimizer to use.')
	parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'],
						help='Training precision (fp32, fp16, bf16).')

	# General Optimizer Params (can be used by multiple optimizers)
	parser.add_argument('--betas', type=float, nargs='+', default=None,
						help='Optimizer beta parameters (e.g., 0.9 0.999 or 0.9 0.999 0.99)')  # Allow variable number
	parser.add_argument('--eps', type=float, default=None, help='Optimizer epsilon term.')
	parser.add_argument('--weight_decay', type=float, default=None, help='Optimizer weight decay.')

	# FMARSCrop / ADOPTMARS Specific Params
	parser.add_argument('--gamma', type=float, default=None, help='Gamma for MARS correction (FMARSCrop/ADOPTMARS).')

	# ScheduleFree Specific Params
	parser.add_argument('--r_sf', type=float, default=None, help='ScheduleFree r parameter (polynomial weight power).')
	parser.add_argument('--wlpow_sf', type=float, default=None, help='ScheduleFree weight_lr_power parameter.')
	# Note: sf_momentum is usually fixed like beta1, might not need cmd arg unless experimenting

	# ADOPTAOScheduleFree Specific Params
	parser.add_argument('--state_precision', type=str, default='parameter',
						choices=['parameter', 'q8bit', 'q4bit', 'qfp8'],
						help='Precision for optimizer state (ADOPTAOScheduleFree).')

	# --- NEW ARGUMENTS END HERE ---

	args = parser.parse_args()
	if not os.path.isfile(args.config):
		parser.error(f"Can't find config file '{args.config}'")

	# Now, merge YAML config with command-line args
	args = get_training_args(args)

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
def get_training_args(args):
	"""Loads YAML config and merges with argparse args."""
	with open(args.config) as f:
		conf = yaml.safe_load(f)

	# --- Load Training Params from YAML ---
	train_conf = conf.get("train", {})
	args.lr = float(
		train_conf.get("lr", args.lr if hasattr(args, 'lr') and args.lr is not None else 1e-4))  # Get base LR
	args.steps = int(train_conf.get("steps", args.steps if hasattr(args, 'steps') else 100000))
	args.batch = int(train_conf.get("batch", args.batch if hasattr(args, 'batch') else 1))

	# Allow YAML override for general optimizer params if not set via command line
	if args.betas is None:
		args.betas = tuple(map(float, train_conf.get("betas", []))) or None  # Convert list from YAML to tuple
	if args.eps is None:
		args.eps = train_conf.get("eps", None)
	if args.weight_decay is None:
		args.weight_decay = train_conf.get("weight_decay", None)

	# Allow YAML override for specific optimizer params
	if args.gamma is None:
		args.gamma = train_conf.get("gamma", None)
	if args.r_sf is None:
		args.r_sf = train_conf.get("r_sf", None)
	if args.wlpow_sf is None:
		args.wlpow_sf = train_conf.get("wlpow_sf", None)

	# Scheduler specific (might need adjustment if using schedule-free)
	args.cosine = train_conf.get("cosine", getattr(args, 'cosine', True))  # Default to True if not set
	args.warmup_steps = int(train_conf.get("warmup_steps", getattr(args, 'warmup_steps', 5000)))  # Add warmup_steps

	# --- Load Model Params from YAML (Keep as is) ---
	assert "model" in conf.keys(), "Model config not optional!"
	args.base = conf["model"].get("base", "unknown")
	args.rev = conf["model"].get("rev", "v0.0")
	args.arch = conf["model"].get("arch", None)
	args.clip = conf["model"].get("clip", "CLIP")
	args.name = f"{args.base}-{args.rev}"
	assert args.arch in ["score", "class"], f"Unknown arch '{args.arch}'"
	assert args.clip in ["CLIP", "META"], f"Unknown CLIP '{args.clip}'"

	# --- Load Labels/Weights from YAML (Keep as is) ---
	labels = conf.get("labels", {})
	if args.arch == "class" and labels:
		args.labels = {str(k): v.get("name", str(k)) for k, v in labels.items()}
		args.num_labels = max([int(x) for x in labels.keys()]) + 1
		weights = [1.0 for _ in range(args.num_labels)]
		for k in labels.keys():
			weights[k] = labels[k].get("loss", 1.0)
		args.weights = weights  # This is used later in train.py
	else:
		# Ensure these exist even for score mode, maybe set to None or default
		args.num_labels = 1
		args.labels = None
		args.weights = None

	# Pass through any other args added by argparse
	return args


# --- Keep write_config (No change needed for args) ---
def write_config(args):
	conf = {
		"name": args.base,
		"rev": args.rev,
		"arch": args.arch,
		"labels": args.labels,
		# Add training and optimizer args to config dump for reproducibility
		"train_args": {
			"lr": args.lr,
			"steps": args.steps,
			"batch": args.batch,
			"cosine": args.cosine,
			"warmup_steps": args.warmup_steps,
			"optimizer": args.optimizer,
			"precision": args.precision,
			"betas": args.betas,
			"eps": args.eps,
			"weight_decay": args.weight_decay,
			"gamma": args.gamma,
			"r_sf": args.r_sf,
			"wlpow_sf": args.wlpow_sf,
			"state_precision": args.state_precision,
			# Add any other relevant args here...
		}
	}
	conf["model_params"] = get_embed_params(args.clip)
	conf["model_params"]["outputs"] = args.num_labels

	os.makedirs(SAVE_FOLDER, exist_ok=True)
	with open(f"{SAVE_FOLDER}/{args.name}.config.json", "w") as f:
		f.write(json.dumps(conf, indent=2))


# --- Updated ModelWrapper Class ---
class ModelWrapper:
	def __init__(self, name, model, optimizer, criterion, scheduler=None, device="cpu", dataset=None, stdout=True,
				 scaler=None, wandb_run=None):  # Added scaler, wandb_run
		self.name = name
		self.device = device
		self.losses = []

		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.scheduler = scheduler
		self.scaler = scaler  # Store scaler
		self.wandb_run = wandb_run  # Store wandb object

		self.dataset = dataset
		# Ensure eval data uses correct device and dtype
		# Note: If eval_src becomes very large, consider loading/processing it inside eval_model
		if dataset and dataset.eval_data:
			self.eval_src = dataset.eval_data.get("emb").to(device=self.device)  # dtype handled by autocast later
			self.eval_dst = dataset.eval_data.get("val").to(device=self.device,
															dtype=torch.float32)  # target usually fp32
		else:
			print("Warning: No evaluation data found in dataset.")
			self.eval_src = None
			self.eval_dst = None

		os.makedirs(SAVE_FOLDER, exist_ok=True)
		# --- Log file handling ---
		self.log_file_path = f"{SAVE_FOLDER}/{self.name}.csv"
		# Check if resuming, open in append mode if CSV exists
		# Check if optimizer has state before assuming resume for append mode
		file_mode = "a" if os.path.exists(self.log_file_path) and any(self.optimizer.state.values()) else "w"
		try:
			self.csvlog = open(self.log_file_path, file_mode)
			if file_mode == "w":  # Write header only if creating new file
				self.csvlog.write("step,train_loss_avg,eval_loss,learning_rate\n")
		except IOError as e:
			print(f"Warning: Could not open CSV log file {self.log_file_path}: {e}")
			self.csvlog = None  # Set to None if opening fails
		self.stdout = stdout

	# --- End log file handling ---

	def log_step(self, loss, step=None):
		self.losses.append(loss)
		current_step = step or len(self.losses)  # Use consistent naming
		if current_step % LOG_EVERY_N == 0 and current_step > 0:  # Log every N steps, skip step 0 maybe
			self.log_main(current_step)  # Pass the calculated step

	def log_main(self, step):  # Renamed argument for clarity
		lr = float(self.optimizer.param_groups[0]['lr'])
		avg_len = min(len(self.losses), LOSS_MEMORY)
		avg = sum(self.losses[-avg_len:]) / avg_len if avg_len > 0 else 0.0

		# --- Perform Evaluation ---
		# This call now handles setting eval/train modes internally
		eval_loss_val, eval_pred_tensor = self.eval_model()
		# --- End Evaluation ---

		# --- Stdout Logging ---
		if self.stdout:
			pred_str = "N/A"
			if eval_pred_tensor is not None:
				# Check tensor shape and content before processing
				try:
					if eval_pred_tensor.ndim >= 2 and eval_pred_tensor.shape[1] == 1:  # score (batch_size, 1)
						pred_str = f"{eval_pred_tensor.mean().item() * 100:.2f}%"
					elif eval_pred_tensor.ndim >= 2 and eval_pred_tensor.shape[
						0] > 0:  # class (batch_size, num_classes)
						conf, _ = torch.max(eval_pred_tensor[0], dim=0)  # Max conf for first sample
						pred_str = f"cls_conf={conf.item():.3f}"
					else:
						pred_str = f"Invalid shape: {eval_pred_tensor.shape}"
				except Exception as e:
					pred_str = f"Error ({e})"

			tqdm.write(
				f"{str(step):<10} Loss(avg): {avg:.4e} | Eval Loss: {eval_loss_val:.4e} | LR: {lr:.4e} | Eval Pred: {pred_str}")
		# --- End Stdout Logging ---

		# --- Wandb Logging ---
		if self.wandb_run:
			log_data = {
				"train/loss_avg": avg,  # Use groups for clarity
				"eval/loss": eval_loss_val,
				"train/learning_rate": lr,
				# Removed step from here, wandb uses its own step counter by default
				# or you can log with step=step if needed: self.wandb_run.log(log_data, step=step)
			}
			# Example: Add eval prediction confidence if available
			if eval_pred_tensor is not None and eval_pred_tensor.ndim >= 2 and eval_pred_tensor.shape[0] > 0 and \
					eval_pred_tensor.shape[1] > 1:
				try:
					conf, _ = torch.max(eval_pred_tensor[0], dim=0)
					log_data["eval/pred_max_conf_sample0"] = conf.item()
				except:
					pass  # Ignore errors in optional logging

			self.wandb_run.log(log_data)  # Log the dictionary
		# --- End Wandb Logging ---

		# --- CSV Logging ---
		if self.csvlog:
			try:
				self.csvlog.write(f"{step},{avg},{eval_loss_val},{lr}\n")
				self.csvlog.flush()
			except IOError as e:
				print(f"Warning: Could not write to CSV log file: {e}")
				self.csvlog = None  # Stop trying if write fails

	# --- End CSV Logging ---

	# --- eval_model with mode handling ---
	def eval_model(self):
		"""Performs evaluation and handles model/optimizer modes."""
		if self.eval_src is None or self.eval_dst is None:
			# Return NaN or suitable default if no eval data
			return float('nan'), None

		eval_loss = torch.tensor(float('nan'), device=self.device)  # Default to NaN
		eval_pred = None

		# --- Set Eval Modes ---
		original_model_mode = self.model.training
		original_optimizer_mode_is_training = False  # Track if optimizer was training

		# Check if optimizer needs mode switching (has state and eval method)
		needs_optim_switch = (hasattr(self.optimizer, 'eval') and callable(self.optimizer.eval) and
							  hasattr(self.optimizer, 'train') and callable(self.optimizer.train) and
							  hasattr(self.optimizer, 'state') and any(self.optimizer.state.values()))

		if needs_optim_switch:
			# Check if optimizer has a 'train_mode' attribute, otherwise assume it's training
			if hasattr(self.optimizer, 'train_mode'):
				original_optimizer_mode_is_training = self.optimizer.train_mode
			elif original_model_mode:  # Heuristic: if model is training, optimizer probably is too
				original_optimizer_mode_is_training = True

			if original_optimizer_mode_is_training:
				self.optimizer.eval()  # Switch to eval if it was training

		self.model.eval()  # Always set model to eval
		# --- End Set Eval Modes ---

		try:
			autocast_enabled = self.scaler is not None and self.scaler.is_enabled()
			# --- Determine correct dtype ---
			if autocast_enabled:
				# Infer dtype based on scaler state (less direct) or pass from train.py
				# For simplicity, let's assume bf16 if supported, else fp16, if scaler enabled
				# A better way is to pass args.precision or amp_dtype to ModelWrapper's __init__
				if torch.cuda.is_bf16_supported():
					current_amp_dtype = torch.bfloat16
				else:
					current_amp_dtype = torch.float16
			else:
				current_amp_dtype = torch.float32
			# --- End determine dtype ---

			# Use the determined dtype
			with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=current_amp_dtype):
				with torch.no_grad():
					eval_pred = self.model(self.eval_src)
					eval_loss = self.criterion(eval_pred, self.eval_dst)
		except Exception as e:
			print(f"Error during evaluation inference/loss calculation: {e}")
			eval_loss = torch.tensor(float('nan'), device=self.device)
			eval_pred = None
		finally:
			# --- Restore Modes ---
			if needs_optim_switch and original_optimizer_mode_is_training:
				self.optimizer.train()  # Switch back to train only if it was originally training

			if original_model_mode:  # Restore model mode only if it was originally training
				self.model.train()
		# No need for else self.model.eval() because it was set above
		# --- End Restore Modes ---

		return eval_loss.item(), eval_pred  # Return loss value and prediction tensor

	# --- End eval_model modification ---

	# --- save_model with scaler state saving ---
	def save_model(self, step=None, epoch=None):
		# Determine step number robustly
		current_step_num = 0
		if step is not None:
			current_step_num = step
		elif self.losses:
			current_step_num = len(self.losses)

		if epoch is None and current_step_num >= 10 ** 6:
			epoch_str = f"_s{round(current_step_num / 10 ** 6, 2)}M"
		elif epoch is None and current_step_num > 0:
			epoch_str = f"_s{round(current_step_num / 10 ** 3)}K"
		elif epoch is not None:
			epoch_str = f"_e{epoch}"
		else:
			epoch_str = "_s0"

		output_name = f"./{SAVE_FOLDER}/{self.name}{epoch_str}"
		print(f"\nSaving checkpoint: {output_name} (Step: {current_step_num})")

		try:
			save_file(self.model.state_dict(), f"{output_name}.safetensors")
			torch.save(self.optimizer.state_dict(), f"{output_name}.optim.pth")
			if self.scheduler is not None:
				torch.save(self.scheduler.state_dict(), f"{output_name}.sched.pth")
			# --- Save Scaler State ---
			if self.scaler is not None and self.scaler.is_enabled():
				torch.save(self.scaler.state_dict(), f"{output_name}.scaler.pth")
			# --- End Save Scaler State ---
			print("Checkpoint saved successfully.")
		except Exception as e:
			print(f"Error saving checkpoint {output_name}: {e}")

	# --- End save_model modification ---

	def close(self):
		if self.csvlog:
			try:
				self.csvlog.close()
			except Exception as e:
				print(f"Warning: Error closing CSV log file: {e}")
			finally:
				self.csvlog = None  # Ensure it's None after trying to close
# Optional: Close wandb run if passed and managed here?
# if self.wandb_run:
#     self.wandb_run.finish()
#     self.wandb_run = None
# Usually wandb.finish() is called at the end of the main script.

# --- End ModelWrapper Modifications ---