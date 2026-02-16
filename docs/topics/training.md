# Training

This guide explains how training works in CityClassifiers and how to start your own training runs.

## Overview

Training in CityClassifiers is driven by **Configuration Files**. You don't need to write code to change the model architecture, learning rate, or dataset path—you just edit a YAML file.

We support two main training modes, each with its own command-line interface (CLI):

1.  **Embedding/Image Training**: The most common mode. Fast and efficient.
2.  **Sequence Training**: For advanced use cases requiring spatial awareness.

## Starting a Training Run

### 1. Training on Embeddings (Recommended)

If you have generated single-vector embeddings using `generate_embeddings.py` (see [Quickstart](../quickstart.md)), use this command:

```bash
python -m cityclassifiers.cli.train_embeddings --config config/your_config.yaml
```

This script loads your pre-computed embeddings into memory and trains a "Head" model to classify them. Because the heavy lifting (vision feature extraction) is already done, this training is extremely fast—often finishing in minutes.

### 2. End-to-End Training

If you want to train directly from images (fine-tuning the vision backbone), you use the **same command** as above, but with a different config setting:

*   **Command:** `python -m cityclassifiers.cli.train_embeddings --config config/your_config.yaml`
*   **Config:** Set `data.mode: "images"` and ensure `e2e_params.is_end_to_end: true`.

### 3. Training on Feature Sequences

If you generated feature sequences (using `generate_feature_sequences.py`), you need a different training script that handles the extra dimension of data:

```bash
python -m cityclassifiers.cli.train_features --config config/your_config.yaml
```

This mode trains a model that attends to specific parts of the image (patches), offering potentially higher accuracy for complex tasks.

## Key Configuration Options

Your YAML config file controls the training process. Here are the most important sections to look at:

*   **`data`**: Where your data lives (`feature_dir_name` or `image_dir`).
*   **`model`**: Which backbone was used (`base_vision_model`) and the architecture of your head (`head_params`).
*   **`train`**: Hyperparameters like `learning_rate`, `batch_size`, and `epochs`.
*   **`wandb`**: Settings for logging to Weights & Biases.

For a full reference of configuration options, see [Configs](configs.md).

## resuming Training

To resume a training run that was interrupted, simply add the `resume` path to your config file or pass it as an argument (if supported by your specific script version, though config is preferred):

```yaml
# In your config.yaml
args:
  resume: "models/your_run_name/checkpoint_last.pth"
```

## Validation & Testing

During training, the model is automatically validated against a hold-out set (if you provided one or split your data). Metrics like Accuracy and Loss are logged to the console and W&B.

To verify the codebase integrity before training (e.g., after modifying code), you can run the quality gate:

```bash
scripts/quality/run_quality.sh
```
