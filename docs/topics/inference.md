# Inference

Inference logic is package-based and reused by demo scripts.

## Core Modules

1. Pipelines: `cityclassifiers/inference/pipeline.py`
2. Output formatting/postprocess: `cityclassifiers/inference/postprocess.py`
3. Inference entry module: `cityclassifiers/cli/infer.py`

## Common Actions

1. Batch folder inference:
```bash
python demo_folder.py --help
```
2. Classifier Gradio demo:
```bash
python demo_class_gradio.py
```
3. Score/regression Gradio demo:
```bash
python demo_score_gradio.py
```

## Notes

1. Training and inference share model heads from `cityclassifiers.models.*`.
2. Avoid introducing inference-only model logic in root scripts; keep behavior in package modules.
