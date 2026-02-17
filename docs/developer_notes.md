# Developer Notes

This document contains notes for developers working on the CityClassifiers codebase, including information about known issues, dependencies, and recent changes.

## Known Issues and Missing Dependencies

### `dinov3_7b_quant_bnb`
The codebase references a module named `dinov3_7b_quant_bnb` for loading quantized DINOv3 models (specifically `dinov3-vit7b16-pretrain-lvd1689m-8bit`). This module is **not present** in the repository and appears to be a custom or private dependency.

-   **Impact:** Attempts to use the `dinov3_7b_8bit_bnb` mode in `generate_embeddings.py` or load this specific model in inference pipelines will fail with an `ImportError`.
-   **Mitigation:** The import has been made conditional in `cityclassifiers/inference/pipeline.py` and `generate_embeddings.py`. The code will now raise a clear error only when this specific functionality is requested, allowing other parts of the library to function normally without this dependency.

## Codebase Changes (Refactor)

### Import Paths
The demo scripts (`demo_class_gradio.py`, `demo_score_gradio.py`, `demo_folder.py`) previously attempted to import directly from `inference`. This has been corrected to import from the package structure `cityclassifiers.inference`.

### `CityAestheticsMultiModelPipeline`
The `CityAestheticsMultiModelPipeline` class was referenced in `demo_score_gradio.py` but was missing from `cityclassifiers/inference/pipeline.py`. It has been implemented to support running multiple aesthetic scorer models and returning a dictionary of `{model_name: score}`.

## Entrypoints

The primary entrypoints for using this library are:

1.  **Data Preparation:**
    -   `generate_embeddings.py`: For generating single-vector embeddings.
    -   `generate_feature_sequences.py`: For generating sequence features.

2.  **Training:**
    -   `python -m cityclassifiers.cli.train_embeddings`: For training on embeddings or end-to-end images.
    -   `python -m cityclassifiers.cli.train_features`: For training on pre-computed feature sequences.

3.  **Inference:**
    -   `demo_folder.py`: For batch processing folders.
    -   `demo_class_gradio.py` / `demo_score_gradio.py`: For interactive web demos.
