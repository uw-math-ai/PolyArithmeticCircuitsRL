#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
N_VARS="${N_VARS:-2}"
MAX_OPS="${MAX_OPS:-6}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/data}"
EVAL_FRAC="${EVAL_FRAC:-0.2}"
SEED="${SEED:-0}"
MAX_GRAPH_NODES="${MAX_GRAPH_NODES:-500000}"
MAX_SECONDS="${MAX_SECONDS:-900}"

mkdir -p "$OUT_DIR"

cd "$REPO_DIR"
exec "$PYTHON" scripts/build_dataset.py \
  --n_vars "$N_VARS" \
  --max_ops "$MAX_OPS" \
  --out_dir "$OUT_DIR" \
  --eval_frac "$EVAL_FRAC" \
  --seed "$SEED" \
  --max_graph_nodes "$MAX_GRAPH_NODES" \
  --max_seconds "$MAX_SECONDS" \
  "$@"
