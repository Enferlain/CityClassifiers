# Quickstart Guide

Welcome to CityClassifiers! This guide will walk you through the process of training your own image classifier or aesthetic predictor. We've designed this framework to be flexible yet easy to use, letting you leverage powerful state-of-the-art vision models.

## 1. Setup

Before we begin, make sure you have the environment set up. We recommend using `uv` for fast and reliable dependency management, but standard `pip` works too.

```bash
# Clone the repository
git clone https://github.com/Enferlain/CityClassifiers.git
cd CityClassifiers

# Create a virtual environment (Python 3.11+ recommended)
uv venv .venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (adjust torch version for your CUDA if needed)
uv sync --python .venv/bin/python
```

## 2. Prepare Your Data

The framework expects your images to be organized in folders, where the folder name represents the class label.

**Example Structure:**
```
data/
  my_dataset/
    cats/
      cat1.jpg
      cat2.png
    dogs/
      dog1.jpg
      dog2.webp
```

## 3. Step 1: Generate Features (The "Entrypoint")

Training modern vision models from scratch is expensive. Instead, we use a powerful pre-trained "base" model (like SigLIP or DINOv2) to "look" at your images and extract their essence into numerical features (embeddings). We then train a smaller, faster model on top of these features.

This is the first **Entrypoint** into the backend: `generate_embeddings.py`.

**Run this command to create embeddings:**

```bash
python generate_embeddings.py \
  --image_dir data/my_dataset \
  --output_dir_root data \
  --model_name google/siglip-so400m-patch14-384 \
  --preprocess_mode fit_pad
```

**What happens here?**
*   It reads images from `data/my_dataset`.
*   It passes them through the `siglip-so400m` model.
*   It saves the resulting embeddings (compact vector representations) into a new folder in `data/`. Look for a folder name starting with `google_siglip...`.

> **Tip:** If you want higher accuracy and have more disk space, you can generate *feature sequences* instead using `generate_feature_sequences.py`. This preserves spatial information but takes up more room.

## 4. Step 2: Train Your Model

Now that you have embeddings, you can train your custom classifier. This is very fast!

This is the second **Entrypoint**: `cityclassifiers.cli.train_embeddings`.

**1. Create a Config File**
Find a template in the `config/` directory (e.g., `config/anatomy_so400nf.yaml`) and make a copy, say `config/my_run.yaml`.

Edit these key fields:
*   `data.feature_dir_name`: The name of the folder created in Step 3.
*   `model.base_vision_model`: The same model you used for generation (`google/siglip-so400m-patch14-384`).
*   `head_params`: Configure your model head (defaults are usually fine).

**2. Start Training**

```bash
python -m cityclassifiers.cli.train_embeddings --config config/my_run.yaml
```

The script will:
*   Load your embeddings.
*   Train a new "Head" model to classify them.
*   Save checkpoints to a `models/` folder.

## 5. Step 3: Use Your Model (Inference)

Once training is done, you have a `.safetensors` model file. You can use it to classify new images!

**Batch Process a Folder:**

```bash
python demo_folder.py \
  --src path/to/new_images \
  --model models/my_run_best_val.safetensors \
  --arch class \
  --target_label_name "cats" \
  --copy_passed
```

**Interactive Demo:**

Fire up a web interface to drag-and-drop images:

```bash
python demo_class_gradio.py
```
*(Note: You'll need to edit `demo_class_gradio.py` to point to your new model path).*

---

That's it! You've successfully trained and deployed a state-of-the-art computer vision model. Check the `docs/topics/` folder for deeper dives into specific components.
