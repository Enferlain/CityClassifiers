# image_dataset.py
# Version 1.0.0: Dataset for loading images and applying transforms end-to-end.

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import random
from collections import defaultdict
import traceback

# List of common image extensions
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]

# --- Helper: Collate function to handle potential None from failed loads ---
def collate_skip_none(batch):
    """Collate function that filters out None items."""
    batch = [item for item in batch if item is not None]
    if not batch:
        # Return an empty dictionary or signal error if batch becomes empty
        # Returning empty dict might work if train loop checks batch content
        # print("Warning: collate_skip_none resulted in empty batch.")
        return {} # Or maybe None? Needs handling in train loop. Let's try empty dict.
    try:
        # Use default collate on the filtered batch
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        print(f"Error in collate_skip_none default_collate: {e}")
        print("Problematic batch contents (first item type):", type(batch[0]) if batch else "Empty")
        # Maybe print shapes if possible
        if isinstance(batch[0], dict):
            for key, value in batch[0].items():
                if hasattr(value, 'shape'): print(f"  Item 0, Key '{key}', Shape: {value.shape}")
        return {} # Return empty dict on error
# --- End Helper ---


class ImageFolderDataset(Dataset):
    """
    Dataset to load images directly from class-based folders and apply transforms.
    Handles train/validation splitting based on fixed count per class.
    """
    def __init__(self, root_dir, transform=None, validation_split_count=0, seed=42):
        """
        Args:
            root_dir (str): Path to the directory containing class subfolders (e.g., data/my_images/).
            transform (callable, optional): A function/transform to apply to the PIL image
                                           (e.g., the processor from Hugging Face).
            validation_split_count (int): Number of samples per class for validation.
            seed (int): Random seed for shuffling and splitting.
        """
        print(f"Initializing ImageFolderDataset v1.0.0...")
        self.root_dir = root_dir
        self.transform = transform # This will be our AIMv2 processor
        self.validation_split_count = validation_split_count
        self.seed = seed
        self.num_labels = 0

        self.train_items = [] # List of tuples: (image_path, label_idx)
        self.val_items = []   # List of tuples: (image_path, label_idx)
        self.label_map = {}   # Maps class folder name (e.g., "0") to integer index (0)
        self.idx_to_label = {} # Maps integer index back to folder name (optional)

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")

        # Parse, map labels, and split
        all_items_by_class_name = self._parse_image_files()
        self._map_labels_and_split(all_items_by_class_name)

        print(f"ImageFolderDataset: OK. Training items: {len(self.train_items)}, Validation items: {len(self.val_items)}")
        print(f"  Found {self.num_labels} classes. Label map: {self.label_map}")


    def _parse_image_files(self):
        """Scans the root directory for class folders and image files."""
        print("Dataset: Scanning for image files...")
        items_by_class_name = defaultdict(list)
        found_classes = set()

        for class_folder_name in tqdm(os.listdir(self.root_dir), desc="Parsing folders"):
            class_dir = os.path.join(self.root_dir, class_folder_name)
            if not os.path.isdir(class_dir): continue

            # Basic validation: Check if folder name looks like a class label (e.g., integer)
            # You might adjust this if your folder names are different
            if not class_folder_name.isdigit() and class_folder_name != "test": # Allow 'test' folder but maybe skip?
                 print(f"Warning: Skipping non-class-like folder '{class_folder_name}'.")
                 continue

            found_classes.add(class_folder_name)
            image_count = 0
            for filename in os.listdir(class_dir):
                if os.path.splitext(filename)[1].lower() in IMAGE_EXTS:
                    image_path = os.path.join(class_dir, filename)
                    items_by_class_name[class_folder_name].append(image_path)
                    image_count += 1
            # print(f"  Found {image_count} images in folder '{class_folder_name}'.") # Optional verbose log

        sorted_classes = sorted(list(found_classes), key=lambda x: int(x) if x.isdigit() else float('inf'))
        print(f"Dataset: Found class folders: {sorted_classes}")

        # Assign integer indices based on sorted folder names
        self.num_labels = 0
        for name in sorted_classes:
             # Check if name is digit before adding to label map
             if name.isdigit():
                  self.label_map[name] = self.num_labels
                  self.idx_to_label[self.num_labels] = name
                  self.num_labels += 1
             # else: print(f"  Skipping non-digit folder '{name}' for label mapping.") # Skip non-numeric folders

        if self.num_labels == 0:
             print("Warning: No valid class folders (named with digits) found for training/validation.")

        return items_by_class_name

    def _map_labels_and_split(self, all_items_by_class_name):
        """Assigns integer labels and splits into train/validation sets."""
        print(f"Dataset: Splitting data (Validation count per class: {self.validation_split_count})...")
        random.seed(self.seed)
        self.train_items = []
        self.val_items = []

        for class_name, image_paths in all_items_by_class_name.items():
            if class_name not in self.label_map: continue # Skip folders not mapped to labels

            label_idx = self.label_map[class_name]
            items_with_labels = [(path, label_idx) for path in image_paths]
            num_items = len(items_with_labels)
            val_count = self.validation_split_count

            if val_count <= 0: # No validation split
                self.train_items.extend(items_with_labels)
            elif num_items <= val_count: # Not enough samples for split
                 print(f"Warning: Class '{class_name}' ({num_items} samples) <= validation count ({val_count}). Using all for training.")
                 self.train_items.extend(items_with_labels)
            else: # Perform split
                 random.shuffle(items_with_labels)
                 self.val_items.extend(items_with_labels[:val_count])
                 self.train_items.extend(items_with_labels[val_count:])

        if self.train_items: random.shuffle(self.train_items) # Shuffle final training list
        # Validation set is usually not shuffled by default in loader


    def __len__(self):
        return len(self.train_items)

    def __getitem__(self, index):
        img_path, label_idx = self.train_items[index]

        try:
            # Load Image
            img = Image.open(img_path).convert("RGB")

            # Apply Transforms (e.g., AIMv2 Processor)
            pixel_values = None
            if self.transform:
                # Processor typically returns a dict-like object (BatchFeature)
                # Handle potential errors during transform
                try:
                     processed_output = self.transform(images=img, return_tensors="pt")
                     # Extract the pixel_values tensor, remove the batch dimension
                     pixel_values = processed_output['pixel_values'].squeeze(0)
                except Exception as transform_e:
                     print(f"ERROR applying transform to {img_path}: {transform_e}")
                     traceback.print_exc()
                     return None # Skip this item if transform fails

            # Return dictionary expected by training loop
            # Ensure label is a tensor
            label_tensor = torch.tensor(label_idx, dtype=torch.long)

            if pixel_values is not None:
                 return {"pixel_values": pixel_values, "label": label_tensor}
            else: # Should not happen if transform worked, but safety check
                 print(f"Warning: pixel_values is None after transform for {img_path}")
                 return None

        except UnidentifiedImageError:
            print(f"Warning: Skipping file (cannot identify image): {img_path}")
            return None # Skip corrupt images
        except Exception as e:
            print(f"ERROR in ImageFolderDataset __getitem__ for index {index}, path {img_path}: {e}")
            traceback.print_exc()
            return None # Skip item on other errors

    def get_validation_loader(self, batch_size, num_workers=0):
        if not self.val_items:
            print("Validation set is empty.")
            return None
        # Use a simple sub-dataset for validation items
        val_dataset = ValidationSubDataset(self.val_items, self.transform)
        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, # No shuffle for validation
            drop_last=False,
            pin_memory=False, # Generally False for PIL loading
            num_workers=num_workers,
            collate_fn=collate_skip_none # Use the collate function to handle None items
        )

# --- Simple Validation Dataset Wrapper ---
class ValidationSubDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items # List of (path, label_idx) tuples
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # Same logic as main dataset's __getitem__
        img_path, label_idx = self.items[index]
        try:
            img = Image.open(img_path).convert("RGB")
            pixel_values = None
            if self.transform:
                try:
                     processed_output = self.transform(images=img, return_tensors="pt")
                     pixel_values = processed_output['pixel_values'].squeeze(0)
                except Exception as transform_e:
                     print(f"ERROR applying transform to VAL {img_path}: {transform_e}")
                     traceback.print_exc()
                     return None
            label_tensor = torch.tensor(label_idx, dtype=torch.long)
            if pixel_values is not None:
                 return {"pixel_values": pixel_values, "label": label_tensor}
            else: return None
        except UnidentifiedImageError: print(f"Warning: Skipping VAL file (cannot identify): {img_path}"); return None
        except Exception as e: print(f"ERROR in VAL __getitem__ for index {index}, path {img_path}: {e}"); traceback.print_exc(); return None