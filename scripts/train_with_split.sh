#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
N_VARS="${N_VARS:-2}"
MAX_OPS="${MAX_OPS:-6}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
SEED="${SEED:-42}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/split_train_${SEED}_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
DATASET_TAG="polys_nvars${N_VARS}_maxops${MAX_OPS}"
TRAIN_JSONL="${TRAIN_JSONL:-$DATA_DIR/${DATASET_TAG}.train.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-$DATA_DIR/${DATASET_TAG}.eval.jsonl}"
AUTO_BUILD_DATASET="${AUTO_BUILD_DATASET:-1}"

cd "$REPO_DIR"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  exec "$PYTHON" scripts/train.py --help
fi

if [ ! -f "$TRAIN_JSONL" ] || [ ! -f "$EVAL_JSONL" ]; then
  if [ "$AUTO_BUILD_DATASET" = "1" ]; then
    echo "Dataset split missing; building it now..."
    N_VARS="$N_VARS" MAX_OPS="$MAX_OPS" OUT_DIR="$DATA_DIR" PYTHON="$PYTHON" \
      bash scripts/make_dataset.sh
  else
    echo "Missing split files:"
    echo "  $TRAIN_JSONL"
    echo "  $EVAL_JSONL"
    echo "Run scripts/make_dataset.sh first or set AUTO_BUILD_DATASET=1."
    exit 2
  fi
fi

mkdir -p "$LOG_DIR"
exec "$PYTHON" scripts/train.py \
  --n_vars "$N_VARS" \
  --max_ops "$MAX_OPS" \
  --total_steps "$TOTAL_STEPS" \
  --seed "$SEED" \
  --log_dir "$LOG_DIR" \
  --interesting "$TRAIN_JSONL" \
  --eval_jsonl "$EVAL_JSONL" \
  "$@"
