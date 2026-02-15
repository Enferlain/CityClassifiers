"""Dataloader setup for end-to-end image training mode."""

from __future__ import annotations

from torch.utils.data import DataLoader

from image_dataset import ImageFolderDataset, collate_group_by_size


def build_image_training_dataloaders(args, image_processor):
    """Build dataset + dataloaders for image-mode training."""
    if image_processor is None:
        raise RuntimeError("Image processor is required for ImageFolderDataset.")

    image_data_dir = args.data_root
    print(f"Setting up ImageFolderDataset from: {image_data_dir}")
    print(f"Looking for class folders (0, 1, ...) directly inside: {image_data_dir}")

    dataset = ImageFolderDataset(
        root_dir=image_data_dir,
        transform=image_processor,
        validation_split_count=args.val_split_count,
        seed=args.seed,
    )
    args.num_labels = dataset.num_labels
    print(f"DEBUG: Updated args.num_labels from ImageFolderDataset: {args.num_labels}")
    print("DEBUG: Using collate_group_by_size for image-mode dataloader.")

    if len(dataset) == 0:
        print("Warning: Training dataset is empty! Check image path and configuration.")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=args.num_workers,
        collate_fn=collate_group_by_size,
    )
    val_loader = dataset.get_validation_loader(
        batch_size=args.batch,
        num_workers=args.num_workers,
    )

    print(f"Created image training loader with {len(train_loader)} batches ({len(dataset)} samples).")
    if val_loader and hasattr(val_loader, "dataset") and len(val_loader.dataset) > 0:
        print(f"Created image validation loader with {len(val_loader)} batches ({len(val_loader.dataset)} samples).")
    elif args.val_split_count > 0:
        print("Validation split requested, but image validation loader is empty or could not be created.")
    else:
        print("No validation split requested or validation data available.")

    return dataset, train_loader, val_loader
