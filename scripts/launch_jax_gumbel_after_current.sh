#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WAIT_PID="${WAIT_PID:-2859913}"
SOURCE_RESULTS_DIR="${SOURCE_RESULTS_DIR:-results/ppo-mcts-jax_fl_C5_C8_blackwell_resume850_policyfix_extend3000}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-jax-blackwell-python3.13/bin/python}"

GUMBEL_ITERS="${GUMBEL_ITERS:-500}"
MCTS_BATCH_SIZE="${MCTS_BATCH_SIZE:-1024}"
GUMBEL_NUM_SIMULATIONS="${GUMBEL_NUM_SIMULATIONS:-64}"
GUMBEL_MAX_NUM_CONSIDERED_ACTIONS="${GUMBEL_MAX_NUM_CONSIDERED_ACTIONS:-16}"
GUMBEL_SCALE="${GUMBEL_SCALE:-1.0}"
GUMBEL_C_VISIT="${GUMBEL_C_VISIT:-50.0}"
GUMBEL_C_SCALE="${GUMBEL_C_SCALE:-0.1}"
WANDB_PROJECT="${WANDB_PROJECT:-PolyArithmeticCircuitsRL}"
WANDB_ENTITY="${WANDB_ENTITY:-zengrf-university-of-washington}"

mkdir -p results/detached

echo "[scheduler] root_dir=$ROOT_DIR"
echo "[scheduler] waiting for PID $WAIT_PID to exit before starting JAX Gumbel"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done

echo "[scheduler] PID $WAIT_PID exited; waiting 30s for files/GPU state to settle"
sleep 30

LATEST_CHECKPOINT="$({ find "$SOURCE_RESULTS_DIR" -maxdepth 1 -name 'checkpoint_*.pkl' | sort | tail -n 1; } || true)"
if [[ -z "$LATEST_CHECKPOINT" ]]; then
    echo "[scheduler] no checkpoint_*.pkl found under $SOURCE_RESULTS_DIR" >&2
    exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-results/ppo-mcts-jax_fl_C5_C8_gumbel_from_latest_${timestamp}}"
RUN_NAME="${WAND_RUN_NAME:-$(basename "$RESULTS_DIR")}"
RUN_LOG="$RESULTS_DIR/run.log"

mkdir -p "$RESULTS_DIR"

echo "[scheduler] selected checkpoint: $LATEST_CHECKPOINT"
echo "[scheduler] results_dir: $RESULTS_DIR"
echo "[scheduler] run_name: $RUN_NAME"
echo "[scheduler] log: $RUN_LOG"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

exec "$PYTHON_BIN" -u -m src.main \
    --algorithm ppo-mcts-jax \
    --search gumbel \
    --fixed-complexities 5 6 7 8 \
    --max-degree 8 \
    --max-steps 14 \
    --iterations "$GUMBEL_ITERS" \
    --mcts-batch-size "$MCTS_BATCH_SIZE" \
    --gumbel-num-simulations "$GUMBEL_NUM_SIMULATIONS" \
    --gumbel-max-num-considered-actions "$GUMBEL_MAX_NUM_CONSIDERED_ACTIONS" \
    --gumbel-scale "$GUMBEL_SCALE" \
    --gumbel-c-visit "$GUMBEL_C_VISIT" \
    --gumbel-c-scale "$GUMBEL_C_SCALE" \
    --ppo-lr 1e-4 \
    --ppo-epochs 1 \
    --ent-coef 0.01 \
    --factor-subgoal-reward 1.0 \
    --factor-library-bonus 0.5 \
    --completion-bonus 3.0 \
    --results-dir "$RESULTS_DIR" \
    --wandb-run-name "$RUN_NAME" \
    --checkpoint "$LATEST_CHECKPOINT" \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    > "$RUN_LOG" 2>&1