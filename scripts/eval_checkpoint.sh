#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/eval_checkpoint.sh <checkpoint.pt> [extra evaluate.py args...]"
  exit 2
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CHECKPOINT="$1"
shift
EPISODES="${EPISODES:-500}"

ARGS=(--checkpoint "$CHECKPOINT" --episodes "$EPISODES")
if [ -n "${MAX_OPS_OVERRIDE:-}" ]; then
  ARGS+=(--max_ops "$MAX_OPS_OVERRIDE")
fi

cd "$REPO_DIR"
exec "$PYTHON" scripts/evaluate.py "${ARGS[@]}" "$@"
