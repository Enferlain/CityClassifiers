# sequence_dataset.py
# Version 1.1.0: Improved validation loader, __getitem__ robustness, clarified comments.

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler # Added Sampler
from collections import defaultdict
import random
from tqdm import tqdm
import math
import traceback # Added traceback


# --- Collate Function ---
def collate_sequences(batch):
    """
    Collate function for sequence dataset. Assumes BucketBatchSampler provides
    batches where all sequences have the same length. Stacks tensors.
    Filters None items resulting from errors in __getitem__.
    """
    # Filter out None items (e.g., from failed file loads)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the whole batch failed

    # Use default collate since lengths within the batch should be consistent
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        # This error suggests the BucketBatchSampler failed or logic is wrong
        print(f"\n!!! FATAL ERROR in collate_sequences: Sequence length mismatch in batch! Check BucketBatchSampler. Bailing out. Error: {e}")
        # Print shapes for debugging
        for i, item in enumerate(batch):
             if isinstance(item, dict) and 'sequence' in item and hasattr(item['sequence'], 'shape'):
                  print(f"  Item {i} shape: {item['sequence'].shape}")
             else:
                  print(f"  Item {i}: Invalid item type or missing sequence: {type(item)}")
        raise RuntimeError(f"Sequence length mismatch in collate_sequences: {e}") from e
    except Exception as e_coll:
         # Catch other potential collation errors
         print(f"\nError during collation: {e_coll}")
         traceback.print_exc()
         return None # Allow skipping batch on other errors


# --- Main Dataset Class ---
class FeatureSequenceDataset(Dataset):
    """
    Dataset for loading pre-computed feature sequences saved as .npz files.
    Reads sequence lengths, handles train/val split, and prepares for bucketing.
    """
    def __init__(self, feature_root_dir: str, validation_split_count: int = 0, seed: int = 42, preload: bool = False, preload_limit_gb: float = 10.0):
        """
        Args:
            feature_root_dir (str): Path to the directory containing class subfolders
                                     (e.g., data/aimv2..._SeqFeatures_fp16/).
            validation_split_count (int): Number of samples per class for validation.
            seed (int): Random seed for shuffling and splitting.
            preload (bool): Whether to preload features into RAM.
            preload_limit_gb (float): Approx RAM limit in GB for preloading.
        """
        print(f"Initializing FeatureSequenceDataset from: {feature_root_dir}")
        if not os.path.isdir(feature_root_dir):
             raise FileNotFoundError(f"Feature root directory not found: {feature_root_dir}")

        self.feature_root_dir = feature_root_dir
        self.validation_split_count = validation_split_count
        self.seed = seed
        self.preload = preload

        # Data storage
        self.metadata = [] # List of dicts: {'path': str, 'label': int, 'seq_len': int, 'orig_shape': tuple(int, int)}
        self.preloaded_data = None # Holds tensors if preload=True
        self.label_map = {}
        self.idx_to_label = {}
        self.num_labels = 0

        # Bucketing info
        self.buckets = defaultdict(list) # {seq_len: [metadata_idx1, ...]}
        self.bucket_lengths = []

        # Train/Validation split (stores indices into self.metadata)
        self.train_indices = []
        self.val_indices = []

        # --- Initialization Steps ---
        self._scan_and_load_metadata()
        if not self.metadata: exit("Initialization failed: No valid feature files found.")

        if self.preload:
            self._preload_features(preload_limit_gb) # Attempt preload after scanning

        self._make_buckets()
        self._split_data()
        # --- End Initialization ---

        # Final print summary
        print(f"Dataset Initialized.")
        print(f"  Total sequences found: {len(self.metadata)}")
        print(f"  Classes found: {self.num_labels} ({list(self.label_map.keys())})")
        print(f"  Buckets created (by sequence length): {len(self.buckets)}")
        print(f"  Training samples: {len(self.train_indices)}")
        print(f"  Validation samples: {len(self.val_indices)}")
        print(f"  Preloading enabled: {self.preload} ({len(self.preloaded_data) if self.preloaded_data else 0} items preloaded)")

    def _scan_and_load_metadata(self):
        """Scans directories, loads .npz, extracts sequence length metadata."""
        print("Scanning for .npz feature files and reading sequence lengths...")
        found_classes = sorted([d for d in os.listdir(self.feature_root_dir) if os.path.isdir(os.path.join(self.feature_root_dir, d)) and not d.startswith('.')])

        for class_folder in tqdm(found_classes, desc="Scanning Classes"):
            if not class_folder.isdigit():
                print(f"Warning: Skipping non-digit folder '{class_folder}'.")
                continue

            # Assign sequential integer labels starting from 0
            if class_folder not in self.label_map:
                 label_idx = self.num_labels
                 self.label_map[class_folder] = label_idx
                 self.idx_to_label[label_idx] = class_folder
                 self.num_labels += 1
            else:
                 label_idx = self.label_map[class_folder]

            class_dir = os.path.join(self.feature_root_dir, class_folder)
            try:
                 files_in_dir = os.listdir(class_dir)
            except OSError as e:
                 print(f"Warning: Could not list files in {class_dir}: {e}. Skipping folder.")
                 continue

            for filename in tqdm(files_in_dir, desc=f"Class {class_folder}", leave=False):
                if filename.lower().endswith(".npz") and not filename.startswith('.'):
                    filepath = os.path.join(class_dir, filename)
                    try:
                        # Efficiently get shape without loading full array?
                        # np.load allows mmap, maybe fast enough? Let's try loading.
                        with np.load(filepath) as data:
                            if 'sequence' not in data:
                                print(f"Warning: 'sequence' key missing in {filepath}. Skipping.")
                                continue
                            # Get sequence length (number of patches)
                            seq_len = data['sequence'].shape[0]
                            # Store metadata
                            self.metadata.append({'path': filepath, 'label': label_idx, 'seq_len': seq_len})
                    except Exception as e:
                        print(f"Warning: Could not load or read shape from {filepath}: {e}. Skipping.")

        print(f"Scan complete. Found {len(self.metadata)} valid sequences.")

    def _preload_features(self, limit_gb):
         """Attempts to preload feature sequences into RAM if enabled and within limit."""
         if not self.metadata: return
         print(f"Preloading requested. Estimating memory usage...")
         estimated_size = 0
         # Estimate based on first few files? Or assume average? Let's estimate roughly.
         # Assume average seq_len * hidden_dim * bytes_per_elem
         avg_seq_len = sum(m['seq_len'] for m in self.metadata) / len(self.metadata)
         # Get hidden dim from first file? Assume 1024 for now. Assume fp16 save.
         hidden_dim = 1024
         bytes_per_elem = 2 # fp16
         estimated_size = len(self.metadata) * avg_seq_len * hidden_dim * bytes_per_elem
         estimated_gb = estimated_size / (1024**3)
         print(f"  Estimated RAM needed: {estimated_gb:.2f} GB (Limit: {limit_gb:.2f} GB)")

         if estimated_gb > limit_gb:
              print(f"  Estimated size exceeds limit. Preloading DISABLED.")
              self.preload = False
              return

         print(f"  Preloading features into RAM...")
         self.preloaded_data = [None] * len(self.metadata)
         load_errors = 0
         for i, meta in enumerate(tqdm(self.metadata, desc="Preloading")):
              try:
                   with np.load(meta['path']) as data:
                        sequence_np = data['sequence']
                        # Preload as fp16 tensor on CPU
                        self.preloaded_data[i] = torch.from_numpy(sequence_np).to(torch.float16)
              except Exception as e:
                   print(f"Error preloading {meta['path']}: {e}")
                   self.preloaded_data[i] = None # Mark as failed
                   load_errors += 1

         if load_errors > 0:
              print(f"Warning: Failed to preload {load_errors} feature files.")
         print("Preloading complete.")

    def _make_buckets(self):
        """Groups metadata indices into buckets based on sequence length."""
        print("Creating buckets based on sequence length...")
        for i, meta in enumerate(self.metadata):
            seq_len = meta['seq_len']
            self.buckets[seq_len].append(i) # Store index of metadata item

        self.bucket_lengths = sorted(self.buckets.keys())

        # Optional: Log bucket counts
        print(f"Created {len(self.bucket_lengths)} buckets.")
        # for length in self.bucket_lengths:
        #     print(f"  Length {length}: {len(self.buckets[length])} sequences")

    def _split_data(self):
        """Splits metadata indices into train/validation stratified by class."""
        print(f"Splitting data (Validation count per class: {self.validation_split_count})...")
        if self.validation_split_count <= 0:
            print("No validation split requested. Using all data for training.")
            self.train_indices = list(range(len(self.metadata)))
            self.val_indices = []
            random.seed(self.seed) # Seed shuffle even if no validation
            random.shuffle(self.train_indices)
            return

        random.seed(self.seed)
        metadata_by_class = defaultdict(list)
        for idx, meta in enumerate(self.metadata):
            metadata_by_class[meta['label']].append(idx) # Group indices by label

        self.train_indices = []
        self.val_indices = []

        print("Stratifying split by class:")
        for label_idx, indices in metadata_by_class.items():
            class_name = self.idx_to_label.get(label_idx, str(label_idx))
            num_items = len(indices)
            val_count = self.validation_split_count
            random.shuffle(indices) # Shuffle indices for this class

            if num_items <= val_count:
                print(f"  Class '{class_name}': {num_items} samples <= {val_count} validation count. Using all {num_items} for training.")
                self.train_indices.extend(indices)
            else:
                self.val_indices.extend(indices[:val_count])
                self.train_indices.extend(indices[val_count:])
                print(f"  Class '{class_name}': {num_items} samples -> {len(indices[val_count:])} train, {val_count} val.")

        # Final shuffle of training indices
        random.shuffle(self.train_indices)
        # Validation indices are kept in class-grouped order from split for potential analysis
        print(f"Split complete. Train indices: {len(self.train_indices)}, Val indices: {len(self.val_indices)}")

    def __len__(self):
        """Returns the number of training samples."""
        return len(self.train_indices) # Length based on train split

    def get_metadata(self, index):
         """Safely retrieves metadata for a given index."""
         if 0 <= index < len(self.metadata):
              return self.metadata[index]
         return None

    def get_item_by_metadata_index(self, metadata_index):
        """Loads data given an index into the main self.metadata list."""
        try:
            if not (0 <= metadata_index < len(self.metadata)):
                raise IndexError(f"Metadata index {metadata_index} out of bounds.")
            meta = self.metadata[metadata_index]

            # Check if preloaded
            if self.preload and self.preloaded_data is not None and metadata_index < len(self.preloaded_data) and self.preloaded_data[metadata_index] is not None:
                 sequence_tensor = self.preloaded_data[metadata_index].to(torch.float32) # Convert to fp32 for model input? Or keep fp16? Let's keep fp16.
                 sequence_tensor = self.preloaded_data[metadata_index]
            else: # Load from disk
                 filepath = meta['path']
                 try:
                     data = np.load(filepath)
                     sequence_np = data['sequence']
                     sequence_tensor = torch.from_numpy(sequence_np).to(torch.float16) # Load as fp16
                 except Exception as e_load:
                      print(f"Error loading npz file {filepath}: {e_load}")
                      return None

            label = meta['label']
            return {'sequence': sequence_tensor, 'label': torch.tensor(label, dtype=torch.long)}

        except IndexError as e_index: print(f"Error in get_item_by_metadata_index: {e_index}"); return None
        except Exception as e:
             meta_info = self.metadata[metadata_index] if 'metadata_index' in locals() and 0 <= metadata_index < len(self.metadata) else {'path': 'unknown'}
             print(f"Error processing metadata index {metadata_index} (path: {meta_info.get('path', 'unknown')}): {e}"); traceback.print_exc(); return None

    # v1.1.1: Corrected __getitem__ for direct metadata indexing from sampler
    def __getitem__(self, index):
        """
        Loads sequence data for a given METADATA index.
        The BucketBatchSampler yields indices that directly map to self.metadata.
        """
        try:
             # <<< Use index directly on self.metadata >>>
             if index >= len(self.metadata):
                   raise IndexError(f"Metadata index {index} out of range (len {len(self.metadata)})")
             meta = self.metadata[index]
             # <<< End Change >>>

             # Check if preloaded
             if self.preload and self.preloaded_data is not None and index < len(self.preloaded_data) and self.preloaded_data[index] is not None:
                  sequence_tensor = self.preloaded_data[index] # Already fp16 tensor
             else: # Load from disk
                  filepath = meta['path']
                  try:
                      data = np.load(filepath)
                      sequence_np = data['sequence']
                      sequence_tensor = torch.from_numpy(sequence_np).to(torch.float16) # Load as fp16
                  except Exception as e_load:
                       print(f"Error loading npz file {filepath}: {e_load}")
                       return None # Signal error

             label = meta['label']
             # Return fp16 sequence, fp32 handled by autocast in training loop
             return {'sequence': sequence_tensor, 'label': torch.tensor(label, dtype=torch.long)}

        except IndexError as e_index:
             print(f"Error in __getitem__ with metadata index {index}: {e_index}")
             return None
        except Exception as e:
             # Use index directly in error message
             path_info = self.metadata[index].get('path', 'unknown') if 0 <= index < len(self.metadata) else 'unknown'
             print(f"Error processing data for metadata index {index} (path: {path_info}): {e}")
             traceback.print_exc()
             return None

    # --- get_validation_loader ---
    # This also needs to be consistent if using BucketBatchSampler
    def get_validation_loader(self, batch_size, num_workers=0):
         print("Creating validation loader using BucketBatchSampler...")
         if not self.val_indices:
              print("Validation set is empty."); return None

         # Create buckets specifically for validation indices
         val_buckets = defaultdict(list)
         for idx in self.val_indices: # val_indices ARE indices into metadata
              meta = self.get_metadata(idx)
              if meta: val_buckets[meta['seq_len']].append(idx)

         if not any(val_buckets.values()):
              print("No valid samples found for validation buckets."); return None

         # Create validation sampler
         val_sampler = BucketBatchSampler(
             buckets=val_buckets, batch_size=batch_size, drop_last=False,
             shuffle_buckets=False, shuffle_within_bucket=False, seed=self.seed
         )

         # Create DataLoader - pass main dataset, sampler handles indices
         val_loader = DataLoader(
             self, # Pass self (main dataset instance)
             batch_sampler=val_sampler, # Use the batch sampler
             num_workers=num_workers,
             collate_fn=collate_sequences # Use the sequence collate function
         )
         print(f"  Created validation loader with {len(val_sampler)} batches.")
         return val_loader


# --- Simple Validation Dataset Wrapper ---
# (Assumes indices map to the main dataset's metadata)
class ValidationSubDatasetFeatures(Dataset):
     def __init__(self, metadata_list, validation_indices):
          self.metadata = metadata_list
          self.val_indices = validation_indices
          print(f"ValidationSubDataset initialized with {len(self.val_indices)} indices.")

     def __len__(self):
          return len(self.val_indices)

     def __getitem__(self, index):
          metadata_index = -1 # Initialize
          try:
               if index >= len(self.val_indices):
                    raise IndexError(f"Validation index {index} out of bounds for val_indices (len {len(self.val_indices)})")
               metadata_index = self.val_indices[index] # Get index into main metadata
               if metadata_index >= len(self.metadata):
                    raise IndexError(f"Validation metadata index {metadata_index} out of bounds for metadata (len {len(self.metadata)})")
               meta = self.metadata[metadata_index]
               filepath = meta['path']
               label = meta['label']

               try:
                   data = np.load(filepath)
                   sequence_np = data['sequence']
               except Exception as e_load:
                    print(f"Error loading validation npz file {filepath}: {e_load}")
                    return None

               sequence_tensor = torch.from_numpy(sequence_np).to(torch.float16)
               return {'sequence': sequence_tensor, 'label': torch.tensor(label, dtype=torch.long)}

          except IndexError as e_index:
               print(f"Error in validation __getitem__: {e_index}")
               return None
          except Exception as e:
               path_info = 'unknown'
               if metadata_index != -1 and metadata_index < len(self.metadata):
                    path_info = self.metadata[metadata_index].get('path', 'unknown')
               print(f"Error processing validation data for index {index} (metadata index {metadata_index}, path: {path_info}): {e}")
               traceback.print_exc()
               return None

class BucketBatchSampler(Sampler[list[int]]):
    """
    Sampler that yields batches of indices, ensuring all indices in a batch
    belong to the same bucket (same sequence length).

    Args:
        buckets (dict[int, list[int]]): Dictionary mapping sequence length to list of metadata indices.
        batch_size (int): The desired batch size.
        drop_last (bool): If True, drop the last incomplete batch from each bucket.
        shuffle_buckets (bool): If True, shuffle the order of buckets each epoch.
        shuffle_within_bucket (bool): If True, shuffle indices within each bucket each epoch.
        seed (int): Random seed for shuffling.
    """
    def __init__(self, buckets: dict[int, list[int]], batch_size: int, drop_last: bool,
                 shuffle_buckets: bool = True, shuffle_within_bucket: bool = True, seed: int = 42):
        # Pass None as data_source since we don't use it directly here
        super().__init__(data_source=None)

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value")
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value")

        self.buckets = buckets
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_buckets = shuffle_buckets
        self.shuffle_within_bucket = shuffle_within_bucket
        self.seed = seed
        self.epoch = 0

        # Pre-calculate batches for length calculation
        self.batches_per_bucket = {}
        self.num_batches = 0
        self.bucket_keys = sorted(list(self.buckets.keys())) # Store sorted keys

        for seq_len in self.bucket_keys:
            bucket_indices = self.buckets[seq_len]
            num_samples_in_bucket = len(bucket_indices)
            if num_samples_in_bucket == 0:
                 self.batches_per_bucket[seq_len] = 0
                 continue

            num_batches_for_bucket = num_samples_in_bucket // self.batch_size
            if not self.drop_last and num_samples_in_bucket % self.batch_size != 0:
                num_batches_for_bucket += 1

            self.batches_per_bucket[seq_len] = num_batches_for_bucket
            self.num_batches += num_batches_for_bucket

        print(f"BucketBatchSampler initialized. Total batches per epoch: {self.num_batches}")

    def __iter__(self):
        # Use epoch number and seed for deterministic shuffling across epochs/resumes
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Determine order of buckets
        bucket_processing_order = self.bucket_keys
        if self.shuffle_buckets:
            # Shuffle the keys using torch generator for reproducibility
            rand_indices = torch.randperm(len(bucket_processing_order), generator=g).tolist()
            bucket_processing_order = [self.bucket_keys[i] for i in rand_indices]

        # Generate batches for this epoch
        all_batches = []
        for seq_len in bucket_processing_order:
            bucket_indices = self.buckets[seq_len][:] # Get a copy
            if not bucket_indices: continue # Skip empty buckets

            if self.shuffle_within_bucket:
                # Shuffle indices within the bucket using torch generator
                rand_indices = torch.randperm(len(bucket_indices), generator=g).tolist()
                bucket_indices = [bucket_indices[i] for i in rand_indices]

            # Create mini-batches for this bucket
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue # Skip last incomplete batch if drop_last is True
                all_batches.append(batch)

        # Shuffle the final list of batches (optional, batches are already from shuffled buckets)
        # random.shuffle(all_batches) # Maybe not needed if buckets are shuffled

        self.epoch += 1 # Increment epoch for next iteration's seed
        return iter(all_batches)

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        # Allows DistributedSampler compatibility if needed later
        self.epoch = epoch