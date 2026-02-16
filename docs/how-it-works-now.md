# How It Works Now

This document describes the current runtime flow after the refactor and where to extend behavior.
For action-oriented navigation by topic, see `docs/README.md`.

## 1. Entrypoints
Main commands:

1. `python -m cityclassifiers.cli.train_embeddings --config <config.yaml>`
2. `python -m cityclassifiers.cli.train_features --config <config.yaml>`
3. Inference and demos:
`cityclassifiers/inference/pipeline.py`, `demo_folder.py`, `demo_class_gradio.py`, `demo_score_gradio.py`

The old root training wrappers are removed. Package CLIs are the canonical path.

## 2. Config Flow

1. CLI reads config path from `--config`.
2. `cityclassifiers.config.loader.load_experiment_config(...)` loads YAML.
3. Loader normalizes legacy and modern YAML shapes into `ExperimentConfig`.
4. CLI stores normalized runtime args and writes run config snapshots via `cityclassifiers.config.runtime_args.write_config`.

Primary config modules:

1. `cityclassifiers/config/schema.py`
2. `cityclassifiers/config/loader.py`
3. `cityclassifiers/config/runtime_args.py`
4. `cityclassifiers/config/embed_params.py`

## 3. Training Flow

### 3.1 Bootstrap

1. Runtime/device/precision setup from `cityclassifiers.training.bootstrap`.
2. Optional optimizer/scheduler registry loading.
3. Optional checkpoint restore of model, optimizer, scheduler, scaler, and step state.

### 3.2 Model + Criterion Construction

1. Registry and factory logic in:
`cityclassifiers.models.registry`, `cityclassifiers.models.factory`.
2. Heads and task losses live under:
`cityclassifiers.models.heads.*`, `cityclassifiers.models.tasks.losses`.
3. For embeddings/features mode, CLI selects and instantiates model variants via factory.

### 3.3 Data Pipeline

1. Dataloader builders:
`cityclassifiers.data.embeddings`, `cityclassifiers.data.sequences`, `cityclassifiers.data.images`.
2. Shared loader helpers:
`cityclassifiers.data.dataloaders`.
3. Dataset implementations:
`cityclassifiers.data.datasets.*`.
4. Shared batch contracts:
`cityclassifiers.data.contracts`.

### 3.4 Train Loop

1. Loop implementations:
`cityclassifiers.training.loops`.
2. Shared step helpers:
`cityclassifiers.training.engine`.
3. Validation helpers:
`cityclassifiers.training.validation`.
4. Metrics helpers:
`cityclassifiers.training.metrics`.
5. Checkpoint helpers:
`cityclassifiers.training.checkpoint`, `cityclassifiers.training.state_io`.

## 4. Inference Flow

Inference core lives in:

1. `cityclassifiers.inference.pipeline`
2. `cityclassifiers.inference.postprocess`

Training and inference share model-head implementations from `cityclassifiers.models.*`.

## 5. Extension Points

### 5.1 Add a Model

1. Implement module under `cityclassifiers/models/backbones`, `cityclassifiers/models/heads`, or `cityclassifiers/models/tasks`.
2. Register it in `cityclassifiers/models/registry.py`.
3. Reference it from config (`model.model_id` or related model params).

See `docs/how-to-add-model.md`.

### 5.2 Add a Dataset/Loader

1. Add dataset/collate under `cityclassifiers/data/datasets`.
2. Wire a loader builder in `cityclassifiers/data/*.py`.
3. Keep batch contract consistent with `cityclassifiers/data/contracts.py`.

See `docs/how-to-add-dataset.md`.

## 6. Quality and Type Check Baseline

Primary quality command:

1. `scripts/quality/run_quality.sh`

Current `ty` gate is intentionally scoped to refactored core modules used by the package CLIs.
Legacy, optional-dependency-heavy, and exploratory surfaces are not part of the required type-check baseline yet.
