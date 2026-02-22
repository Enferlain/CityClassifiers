# Inference

Inference logic is package-based and exposed through package CLIs.

## Core Modules

1. Pipelines: `cityclassifiers/inference/pipeline.py`
2. Output formatting/postprocess: `cityclassifiers/inference/postprocess.py`
3. Folder inference CLI: `cityclassifiers/cli/infer_folder.py`

## Common Actions

1. Batch folder inference:
```bash
python -m cityclassifiers.cli.infer_folder --help
```

## Notes

1. Training and inference share model heads from `cityclassifiers.models.*`.
2. Avoid introducing inference-only model logic in root scripts; keep behavior in package modules.
