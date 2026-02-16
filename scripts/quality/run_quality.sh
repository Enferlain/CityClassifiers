#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv-wsl/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

VENV_BIN_DIR="$(cd "$(dirname "$PYTHON_BIN")" && pwd)"
RUFF_BIN="${RUFF_BIN:-$VENV_BIN_DIR/ruff}"
TY_BIN="${TY_BIN:-$VENV_BIN_DIR/ty}"
TYPECHECK_MODE="${TYPECHECK_MODE:-auto}" # auto|required|off

if [[ ! -x "$RUFF_BIN" ]]; then
  echo "error: ruff not found: $RUFF_BIN" >&2
  exit 1
fi

LINT_TARGETS=(
  "cityclassifiers/config"
  "cityclassifiers/data"
  "cityclassifiers/models"
  "cityclassifiers/training/state_io.py"
  "cityclassifiers/training/validation.py"
  "cityclassifiers/training/wrapper.py"
  "cityclassifiers/training/checkpoint.py"
  "cityclassifiers/training/engine.py"
  "cityclassifiers/training/loops.py"
  "cityclassifiers/training/metrics.py"
  "tests"
)

echo "[quality] lint"
"$RUFF_BIN" check "${LINT_TARGETS[@]}"

echo "[quality] root-surface"
"$PYTHON_BIN" scripts/quality/check_root_surface.py

echo "[quality] wrapper-refs"
"$PYTHON_BIN" scripts/quality/check_wrapper_references.py

echo "[quality] compile"
"$PYTHON_BIN" -m compileall -q cityclassifiers tests

if [[ "$TYPECHECK_MODE" != "off" ]]; then
  if [[ -x "$TY_BIN" ]]; then
    TY_TARGETS=(
      "cityclassifiers/config"
      "cityclassifiers/data"
      "cityclassifiers/models/factory.py"
      "cityclassifiers/models/registry.py"
      "cityclassifiers/models/heads"
      "cityclassifiers/models/tasks"
      "cityclassifiers/training/checkpoint.py"
      "cityclassifiers/training/engine.py"
      "cityclassifiers/training/loops.py"
      "cityclassifiers/training/metrics.py"
      "cityclassifiers/training/state_io.py"
      "cityclassifiers/training/validation.py"
      "cityclassifiers/training/wrapper.py"
      "cityclassifiers/cli/train_embeddings.py"
      "cityclassifiers/cli/train_features.py"
    )
    echo "[quality] type-check"
    "$TY_BIN" check "${TY_TARGETS[@]}" --output-format concise
  elif [[ "$TYPECHECK_MODE" == "required" ]]; then
    echo "error: TYPECHECK_MODE=required but ty not found: $TY_BIN" >&2
    exit 1
  else
    echo "[quality] type-check skipped (ty not installed)."
  fi
fi

echo "[quality] unit"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest tests/unit -q -s

echo "[quality] integration"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest tests/integration -q -s

echo "[quality] smoke"
scripts/smoke/run_smoke.sh

echo "[quality] all checks passed"
