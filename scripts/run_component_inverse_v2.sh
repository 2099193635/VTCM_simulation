#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${WORKERS:-3}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-1}"
TARGET_SEEDS="${TARGET_SEEDS:-150}"

if [[ "$TARGET_SEEDS" == "150" ]]; then
  MANIFEST="configs/sweeps/component_inverse_v2_stage1.yaml"
  CATALOG_BASE="configs/sweeps/component_inverse_v2_stage1_catalog"
elif [[ "$TARGET_SEEDS" == "300" ]]; then
  MANIFEST="configs/sweeps/component_inverse_v2.yaml"
  CATALOG_BASE="configs/sweeps/component_inverse_v2_catalog"
else
  echo "TARGET_SEEDS must be 150 or 300" >&2
  exit 2
fi

"$PYTHON_BIN" utils/generate_component_inverse_v2.py \
  --target-seed-count "$TARGET_SEEDS" \
  --output "$MANIFEST" \
  --catalog-json "${CATALOG_BASE}.json" \
  --catalog-csv "${CATALOG_BASE}.csv"

"$PYTHON_BIN" utils/run_param_sweep.py \
  --manifest "$MANIFEST" \
  --build-first \
  --skip-completed \
  --stop-on-error \
  --workers "$WORKERS" \
  --threads-per-worker "$THREADS_PER_WORKER"
