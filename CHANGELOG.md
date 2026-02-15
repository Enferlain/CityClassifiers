# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Package-level CLIs for training and inference:
  - `cityclassifiers/cli/train_embeddings.py`
  - `cityclassifiers/cli/train_features.py`
  - `cityclassifiers/cli/infer.py`
- Package-level config layer with typed schema and loader:
  - `cityclassifiers/config/schema.py`
  - `cityclassifiers/config/loader.py`
- Package-level model registry/factory:
  - `cityclassifiers/models/registry.py`
  - `cityclassifiers/models/factory.py`
- Package-level model adapter modules:
  - `cityclassifiers/models/backbones/*`
  - `cityclassifiers/models/heads/*`
  - `cityclassifiers/models/tasks/*`
- Package-level data-loading layer:
  - `cityclassifiers/data/__init__.py`
  - `cityclassifiers/data/contracts.py`
  - `cityclassifiers/data/embeddings.py`
  - `cityclassifiers/data/images.py`
  - `cityclassifiers/data/sequences.py`
  - `cityclassifiers/data/transforms.py`
- Package-level training-loop helpers:
  - `cityclassifiers/training/engine.py`
- Package-level training setup helpers:
  - `cityclassifiers/training/optim.py`
  - `cityclassifiers/training/checkpoint.py`
- Package-level metric logging helpers:
  - `cityclassifiers/training/metrics.py`
- Package-level training loop implementations:
  - `cityclassifiers/training/loops.py`
- Package-level inference module:
  - `cityclassifiers/inference/pipeline.py`
- Refactor docs:
  - `docs/refactor-plan.md`
  - `docs/architecture.md`
  - `docs/migration-map.md`
- Smoke test harness and smoke tests:
  - `scripts/smoke/run_smoke.sh`
  - `tests/smoke/test_refactor_smoke.py`

### Changed
- Root entrypoints are now compatibility wrappers that forward to package modules:
  - `train.py`
  - `train_features.py`
  - `inference.py`
- Training CLIs now load a normalized experiment config via `load_experiment_config(...)`.
- `train_features` and `train_embeddings` now use registry/factory helpers for loss and model construction.
- Embedding model selection can now be configured with `model.model_id` (defaults to `hybrid_head_model` if unspecified).
- Training CLI dataloader setup now delegates to `cityclassifiers.data` modules.
- Training CLIs now share train-mode/scheduler/postfix loop helpers via `cityclassifiers.training.engine`.
- Training CLIs now share optimizer/scheduler construction and checkpoint load wrappers via `cityclassifiers.training.*`.
- Validation metric logging and periodic/best checkpoint save flow now routes through shared `cityclassifiers.training` helpers.
- Per-step prediction/loss shaping now routes through shared helpers in `cityclassifiers.training.engine`.
- Training loops now route batch/target preparation through shared `cityclassifiers.training.engine` helpers.
- Optimizer-step execution and loss-window bookkeeping now route through shared `cityclassifiers.training.engine` helpers.
- Training loops now route global-step/progress/wrapper updates through shared `cityclassifiers.training.engine.advance_global_step`.
- Training loops now route step-limit checks and periodic log/validation interval gating through shared `cityclassifiers.training.engine` helpers.
- Validation post-run train-mode restoration and last-eval-loss tracking now route through shared `cityclassifiers.training` helpers.
- Training CLIs now delegate `train_loop(...)` execution to `cityclassifiers.training.loops`.
- Model and loss imports now route through package adapter modules instead of root-level model/loss imports in registry/factory/engine paths.
- Inference pipeline model-head imports now route through `cityclassifiers.models.heads`.
- Config normalization now infers mode from raw config only (no runtime-args fallback coupling), including `model.is_end_to_end` inference for image mode.
- Config schema now exposes typed mode-specific sections (`predictor_params`, `head_params`, `e2e_params`) for orchestration paths.
- Training model/criterion setup paths now consume typed normalized config sections instead of ad-hoc `getattr` defaults for core model behavior.
- Image-mode dataloader setup now lives in `cityclassifiers.data.images` and embeddings loader no longer owns end-to-end image mode branching.
- Data layer now documents explicit batch-key contracts via `cityclassifiers.data.contracts`.
- End-to-end image processor loading now routes through `cityclassifiers.data.transforms.load_image_processor` instead of direct CLI import/use.

### Fixed
- Config normalization supports both new and legacy YAML shapes, with numeric coercion for string numeric values.
- Inference local model path resolution was updated to be repo-root aware from package location.
- Smoke checks cover refactor-critical modules and integration points to catch structural regressions early.
- Feature-sequence loop no longer performs duplicate best-checkpoint save checks in the same validation pass.
