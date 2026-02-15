"""Dataloader setup for embedding-based training mode."""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from dataset import EmbeddingDataset


def build_embedding_training_dataloaders(args, image_processor=None):
    """Build dataset + dataloaders for embedding-mode training."""
    if image_processor is not None:
        print("Warning: image_processor is ignored in embedding mode.")
    if not hasattr(args, "embed_ver") or not args.embed_ver:
        raise RuntimeError("'embed_ver' is required for embedding-based training.")

    data_root_path = args.data_root
    dataset_version = args.embed_ver
    embedding_data_dir = os.path.join(data_root_path, dataset_version)
    print(f"Setting up EmbeddingDataset using version folder: {embedding_data_dir}")

    dataset = EmbeddingDataset(
        ver=dataset_version,
        root=data_root_path,
        mode=args.arch,
        preload=getattr(args, "preload_data", True),
        validation_split_count=args.val_split_count,
        seed=args.seed,
    )
    if args.arch == "class":
        args.num_labels = dataset.num_labels
        print(f"DEBUG: Updated args.num_labels from EmbeddingDataset: {args.num_labels}")
    collate_fn = getattr(dataset, "collate_ignore_none", torch.utils.data.dataloader.default_collate)
    print(f"DEBUG: Using {getattr(collate_fn, '__name__', 'default_collate')} for embedding dataloader.")

    if len(dataset) == 0:
        print("Warning: Training dataset is empty! Check data path and configuration.")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=getattr(args, "num_workers", 0),
        collate_fn=collate_fn,
    )
    val_loader = dataset.get_validation_loader(
        batch_size=args.batch,
        num_workers=getattr(args, "num_workers", 0),
    )

    print(f"Created embedding training loader with {len(train_loader)} batches ({len(dataset)} samples).")
    if val_loader and hasattr(val_loader, "dataset") and len(val_loader.dataset) > 0:
        print(
            f"Created embedding validation loader with {len(val_loader)} batches "
            f"({len(val_loader.dataset)} samples)."
        )
    elif args.val_split_count > 0:
        print("Validation split requested, but embedding validation loader is empty or could not be created.")
    else:
        print("No validation split requested or validation data available.")

    return dataset, train_loader, val_loader
