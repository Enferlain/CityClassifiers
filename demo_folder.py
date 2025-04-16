import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from shutil import copyfile

from inference import CityAestheticsPipeline, CityClassifierPipeline

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]

def parse_args():
	parser = argparse.ArgumentParser(description="Test model by running it on an entire folder")
	parser.add_argument('--src', default="test", help="Folder with images to score")
	parser.add_argument('--dst', default="pass", help="Folder with images that fit the score threshold")
	parser.add_argument('--max', type=int, default=100, help="Upper limit for score")
	parser.add_argument('--min', type=int, default=  0, help="Lower limit for score")
	parser.add_argument('--model', required=True, help="Model file to use")
	parser.add_argument("--arch", choices=["score", "class"], default="score", help="Model type")
	parser.add_argument('--label', type=str, default=0, help="Target class to use when model type is classifier. Comma separated.")
	parser.add_argument('--copy', action=argparse.BooleanOptionalAction, help="Copy files to the dst folder")
	parser.add_argument('--keep', action=argparse.BooleanOptionalAction, help="Keep original folder structure")
	return parser.parse_args()

def process_file(pipeline, src_path, dst_path):
	pred = pipeline(Image.open(src_path), tile_strat='mean')
	if args.arch == "score":
		pred = int(pred * 100) # [float]=>[int](percentage)
	elif args.arch == "class":
		target_label_indices = [str(x).strip() for x in args.label.split(',')]  # e.g., ['1']
		scores_for_target_indices = []

		# The 'pred' dict keys are label names (or index strings if config failed)
		# We need to find the score corresponding to the *index* specified in --label
		# We need the mapping from index to name (which inference.py has but demo_folder doesn't easily)
		# WORKAROUND: Assume the order in the returned dict corresponds *roughly*
		# OR, a better fix is needed in inference.py/how results are passed.
		# Let's try a simple assumption first: if labels are {'0': 'Bad', '1': 'Good'},
		# the value for the key 'Good' corresponds to label index 1.

		# A slightly more robust way if we know the number of classes (usually 2 here):
		try:
			if len(pred) == 2:  # Specific handling for our binary case
				# Assume labels are 0 and 1 from training setup
				score_label_0 = pred.get('0', pred.get('Bad Anatomy', None))  # Try index key then name key
				score_label_1 = pred.get('1', pred.get('Good Anatomy', None))

				if '1' in target_label_indices and score_label_1 is not None:
					scores_for_target_indices.append(int(score_label_1 * 100))
				if '0' in target_label_indices and score_label_0 is not None:
					scores_for_target_indices.append(int(score_label_0 * 100))

			else:  # Generic fallback (less reliable) - tries matching index string key directly
				for lbl_idx_str in target_label_indices:
					score = pred.get(lbl_idx_str)
					if score is not None:
						scores_for_target_indices.append(int(score * 100))

			# If we found any scores for the requested label(s), take the max, else default to -1?
			if scores_for_target_indices:
				pred = max(scores_for_target_indices)
			else:
				print(f"Warning: Could not find score for requested label(s) '{args.label}' in prediction dict: {pred}")
				pred = -1  # Indicate error
		except Exception as e:
			print(f"Error processing classifier prediction: {e}")
			print(f"  Prediction dict: {pred}")
			pred = -1  # Indicate error

	tqdm.write(f" {pred:>3}% [{os.path.basename(src_path)}]")
	if args.min <= pred <= args.max:
		if dst_path: copyfile(src_path, dst_path)

def process_folder(pipeline, src_root, dst_root):
	dst_folders = [] # avoid excessive mkdir
	for path, _, files in os.walk(src_root):
		for fname in files:
			dst_path = None
			if args.copy:
				dst_dir = dst_root
				src_rel = os.path.relpath(path, src_root)
				if args.keep and src_rel != ".":
					dst_dir = os.path.join(dst_root, src_rel)
				if dst_dir not in dst_folders:
					os.makedirs(dst_dir, exist_ok=True)
					dst_folders.append(dst_dir)
				dst_path = os.path.join(dst_dir, fname)
			src_path = os.path.join(path, fname)
			if os.path.splitext(fname)[1] not in IMAGE_EXTS: continue
			process_file(pipeline, src_path, dst_path)
			# try: process_file(pipeline, src_path, dst_path)
			# except: pass # e.g. for skipping file errors

if __name__ == "__main__":
	args = parse_args()

	os.makedirs(args.dst, exist_ok=True)
	print(f"Predictor using model {os.path.basename(args.model)}")

	pipeline_args = {}
	if torch.cuda.is_available():
		pipeline_args["device"] = "cuda"
		pipeline_args["clip_dtype"] = torch.float32

	if args.arch == "score":
		pipeline = CityAestheticsPipeline(args.model, **pipeline_args)
	elif args.arch == "class":
		pipeline = CityClassifierPipeline(args.model, **pipeline_args)
	else:
		raise ValueError(f"Unknown model architecture '{args.arch}'")

	process_folder(pipeline, args.src, args.dst)
