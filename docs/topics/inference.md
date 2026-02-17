# Inference

Inference is the process of using your trained model to make predictions on new images. CityClassifiers provides flexible pipelines for this, whether you need to process a single image, a whole folder, or run a web demo.

## Using the Demos

The easiest way to perform inference is using the provided demo scripts.

### 1. Batch Processing a Folder

If you have a folder of images and want to sort them or get scores for all of them, use `demo_folder.py`.

```bash
python demo_folder.py \
  --src data/my_new_images \
  --dst output/good_images \
  --model models/my_model.safetensors \
  --arch class \
  --target_label_name "good_quality" \
  --copy_passed
```

*   `--src`: Input folder.
*   `--dst`: Output folder (for copying matching images).
*   `--model`: Path to your trained `.safetensors` file.
*   `--arch`: Model architecture (`class` for classifiers, `score` for aesthetic predictors).
*   `--copy_passed`: If set, images meeting the criteria (e.g., predicted as "good_quality") are copied to `--dst`.

### 2. Interactive Web Demos

To test your model interactively, use the Gradio scripts. These launch a local web server where you can upload images and see predictions instantly.

*   **For Classifiers:** `python demo_class_gradio.py`
*   **For Aesthetic Scorers:** `python demo_score_gradio.py`

*Note: You will need to edit the `MODELS` list inside these scripts to point to your specific model file.*

## Programmatic Inference (Python API)

You can also use the inference pipelines directly in your own Python code.

```python
from PIL import Image
from cityclassifiers.inference import CityClassifierPipeline

# Load the pipeline
pipeline = CityClassifierPipeline(
    model_path="models/my_model.safetensors",
    device="cuda"
)

# Load an image
img = Image.open("test.jpg")

# Get prediction
result = pipeline(img)
print(result)
# Output: {'label_0': 0.1, 'label_1': 0.9, ...}
```

### Available Pipelines

*   **`CityClassifierPipeline`**: For single-model classification.
*   **`CityAestheticsPipeline`**: For single-model aesthetic scoring.
*   **`CityClassifierMultiModelPipeline`**: For running multiple classifiers at once.
*   **`CityAestheticsMultiModelPipeline`**: For running multiple aesthetic scorers at once.

These pipelines handle all the necessary preprocessing (resizing, padding, feature extraction) automatically, ensuring that inference matches training conditions.
