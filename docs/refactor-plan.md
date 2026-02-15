# CityClassifiers Refactor Plan

## Purpose
This plan defines the target structure and an incremental migration path to move the repo from flat, oversized scripts to a modular package with clearer ownership, easier model support, and safer long-term maintenance.

## Refactor Principles
1. Preserve runtime behavior while restructuring.
2. Make changes in small slices with verification after each slice.
3. Keep compatibility wrappers at the repo root until migration is complete.
4. New model support should require adapter modules + registration, not training loop edits.
5. Update `CHANGELOG.md` (`Unreleased`) at the end of each migration slice.

## Target Repository Shape
```text
cityclassifiers/
  pyproject.toml
  README.md
  config/
  cityclassifiers/
    __init__.py
    cli/
      train_embeddings.py
      train_features.py
      infer.py
    config/
      schema.py
      loader.py
    data/
      embeddings.py
      sequences.py
      images.py
      transforms.py
    models/
      registry.py
      factory.py
      backbones/
      heads/
      tasks/
    training/
      engine.py
      bootstrap.py
      optim.py
      checkpoint.py
      metrics.py
    inference/
      pipeline.py
      postprocess.py
    utils/
      logging.py
      seed.py
  tests/
    unit/
    integration/
```

## Phase 0 - Baseline Freeze and Architecture Contract
### Goals
- Capture current behavior so refactors can be validated.
- Define boundaries and ownership for each module area.

### Deliverables
- `docs/architecture.md`:
  - module boundaries
  - data/model/training/inference contracts
  - dependency direction rules
- `docs/migration-map.md`:
  - old file/function -> new module target mapping
- `scripts/smoke/` commands or documented smoke command list:
  - train embeddings smoke
  - train features smoke
  - inference smoke

### Done Criteria
- Baseline smoke commands are documented and repeatable.
- We can compare pre/post-refactor behavior with consistent command set.

## Phase 1 - Entrypoint Thinning (Compatibility First)
### Goals
- Move orchestration into package modules.
- Keep root scripts as wrappers.

### Deliverables
- `cityclassifiers/cli/train_embeddings.py`
- `cityclassifiers/cli/train_features.py`
- `cityclassifiers/cli/infer.py`
- Root wrappers:
  - `train.py`
  - `train_features.py`
  - `inference.py`

### Done Criteria
- Root commands still work unchanged.
- Most business logic is no longer in root scripts.

## Phase 2 - Unified Configuration Layer
### Goals
- One normalized config object across all modes.
- Fail fast on invalid/missing config.

### Deliverables
- `cityclassifiers/config/schema.py`:
  - typed config models
- `cityclassifiers/config/loader.py`:
  - YAML parse + defaults + normalization + validation
- Replace ad-hoc `getattr` defaulting in core paths.

### Done Criteria
- Training/inference consume a validated config object.
- Config errors are clear and early.

## Phase 3 - Model/Task Registry and Factory
### Goals
- Remove model/loss hardcoding from train scripts.
- Make model support extensible by registration.

### Deliverables
- `cityclassifiers/models/registry.py`
  - backbone/head/task registries
- `cityclassifiers/models/factory.py`
  - construct model + criterion + task adapter
- `cityclassifiers/models/backbones/*`
- `cityclassifiers/models/heads/*`
- `cityclassifiers/models/tasks/*`

### Done Criteria
- Existing supported setups are registry-driven.
- Adding a model family requires:
  1. adapter module
  2. registry entry

## Phase 4 - Data Pipeline Normalization
### Goals
- Standardize dataset interfaces and collate contracts.
- Isolate mode-specific preprocessing.

### Deliverables
- `cityclassifiers/data/embeddings.py`
- `cityclassifiers/data/sequences.py`
- `cityclassifiers/data/images.py`
- `cityclassifiers/data/transforms.py`
- common dataloader builder util

### Done Criteria
- Training engine consumes a common batch contract.
- Mode differences live in adapter modules, not engine internals.

## Phase 5 - Training Engine Modularization
### Goals
- Centralize training loop mechanics.
- Reuse checkpoint, optimizer, and logging logic.

### Deliverables
- `cityclassifiers/training/engine.py`
- `cityclassifiers/training/optim.py`
- `cityclassifiers/training/checkpoint.py`
- `cityclassifiers/training/metrics.py`
- callback hooks:
  - validation
  - logging
  - checkpointing

### Done Criteria
- Train loops are shared and mode-agnostic.
- Checkpoint/optimizer/scheduler behavior is centralized.

## Phase 6 - Inference Pipeline Refactor
### Goals
- Reuse training-time model/config construction.
- Isolate IO from inference core logic.

### Deliverables
- `cityclassifiers/inference/pipeline.py`
- `cityclassifiers/inference/postprocess.py`
- CLI wrapper in `cityclassifiers/cli/infer.py`

### Done Criteria
- Inference logic is reusable and testable.
- Folder/single-image flows share core inference path.

## Phase 7 - Tests and Quality Gates
### Goals
- Add confidence for continuous refactoring.

### Deliverables
- `tests/unit/`:
  - config parsing/validation
  - registry resolution
  - target/loss shaping
  - checkpoint IO
- `tests/integration/`:
  - one-step training smoke per mode
  - inference smoke
- `pyproject.toml` quality tasks:
  - lint
  - type-check
  - test

### Done Criteria
- A single quality command catches structural regressions.
- Core refactor risk areas have coverage.

## Phase 8 - Surface Cleanup and Contributor Docs
### Goals
- Reduce root clutter.
- Document extension workflows.

### Deliverables
- Lean root directory with compatibility wrappers only.
- `docs/how-to-add-model.md`
- `docs/how-to-add-dataset.md`
- deprecation notes for legacy internals

### Done Criteria
- New contributors can extend models/datasets without touching core engine logic.
- Structure is discoverable and documented.

## Recommended Execution Order
1. Phase 0
2. Phase 1
3. Phase 3
4. Phase 2
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8

## Verification Strategy Per Slice
1. Move code with compatibility shim.
2. Run syntax checks.
3. Run smoke command(s) for affected path.
4. Confirm no CLI or config regressions.
5. Update `CHANGELOG.md` with Added/Changed/Fixed notes.
6. Commit slice independently.

## Progress Snapshot (2026-02-15)
1. Phase 0: completed
2. Phase 1: completed
3. Phase 2: completed
4. Phase 3: completed
5. Phase 4: in progress
6. Phase 5: completed
7. Phase 6: partially completed
8. Phase 7: in progress
9. Phase 8: not started

## Completed Highlights
1. Root wrappers and package CLIs are in place (`train.py`, `train_features.py`, `inference.py` -> `cityclassifiers/cli/*`).
2. Config schema/loader now includes stricter normalization with typed mode-specific sections used by training setup paths.
3. Model registry/factory now drives both training CLIs.
4. Model package adapters now expose explicit module paths under:
   - `cityclassifiers/models/backbones/*`
   - `cityclassifiers/models/heads/*`
   - `cityclassifiers/models/tasks/*`
5. Data loader setup is extracted to `cityclassifiers/data/*` for embeddings, sequences, and images.
6. Data batch contracts are now documented centrally in `cityclassifiers/data/contracts.py`.
7. End-to-end processor loading is now isolated in `cityclassifiers/data/transforms.py`.
8. Shared training helpers now cover:
   - train-mode + scheduler + progress postfix
   - optimizer/scheduler setup
   - checkpoint load + periodic/best-save helpers
   - validation metric logging helpers
   - per-step prediction/loss shaping
   - batch/target preparation
   - global-step/progress/wrapper propagation
   - step-limit checks and periodic interval gating
   - validation loss-state updates and post-validation mode restoration
9. Training-loop implementations live in `cityclassifiers/training/loops.py`, and CLI training loops now delegate to package code.
10. Refactor smoke suite is active and passing.

## Remaining Work Queue
1. Continue Phase 4:
   - optional follow-up: add a common dataloader-builder helper to reduce remaining duplication
2. Continue Phase 6:
   - split postprocess from `cityclassifiers/inference/pipeline.py` into `cityclassifiers/inference/postprocess.py`
3. Continue Phase 7:
   - add focused unit tests for `training.engine`, `training.metrics`, `training.checkpoint`
   - add minimal integration test(s) that execute one forward/backward step with synthetic data
4. Phase 8 cleanup:
   - contributor docs for adding models/datasets
   - final deprecation notes and surface cleanup
