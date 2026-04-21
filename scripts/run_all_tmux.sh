#!/usr/bin/env bash
# Launch full result-generation pipeline in a detached tmux session.
# Each pane redirects stdout+stderr to logs/ so crashes leave a trail.
# Checkpoints land in runs/<tag>/ — training resumes where best_lvl*.pt were saved.
#
# Usage:
#   bash scripts/run_all_tmux.sh              # start everything
#   tmux attach -t polyrl                     # watch live
#   tmux kill-session -t polyrl               # abort
#
# Override with env vars:
#   SEEDS="42 43 44" TOTAL_STEPS=1000000 bash scripts/run_all_tmux.sh

set -euo pipefail

SESSION="${SESSION:-polyrl}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SEEDS="${SEEDS:-42 43 44}"
N_VARS="${N_VARS:-2}"
MAX_OPS="${MAX_OPS:-6}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
NUM_TARGETS="${NUM_TARGETS:-500}"
PYTHON="${PYTHON:-python}"
LEVELS="${LEVELS:-$(seq 1 "$MAX_OPS")}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_DIR/logs/$STAMP"
RUN_ROOT="$REPO_DIR/runs/$STAMP"
DATA_DIR="$REPO_DIR/data"
mkdir -p "$LOG_DIR" "$RUN_ROOT" "$DATA_DIR"

DATASET_TAG="polys_nvars${N_VARS}_maxops${MAX_OPS}"
TRAIN_JSONL="$DATA_DIR/${DATASET_TAG}.train.jsonl"
EVAL_JSONL="$DATA_DIR/${DATASET_TAG}.eval.jsonl"

if [ ! -f "$TRAIN_JSONL" ] || [ ! -f "$EVAL_JSONL" ]; then
    echo "Building dataset (one-time, ~few minutes)..."
    $PYTHON scripts/build_dataset.py \
        --n_vars $N_VARS --max_ops $MAX_OPS \
        --out_dir "$DATA_DIR" \
        --eval_frac 0.2 --seed 0 \
        2>&1 | tee "$LOG_DIR/build_dataset.log"
else
    echo "Reusing cached dataset: $TRAIN_JSONL"
fi

echo "Session:   $SESSION"
echo "Repo:      $REPO_DIR"
echo "Logs:      $LOG_DIR"
echo "Runs:      $RUN_ROOT"
echo "Seeds:     $SEEDS"
echo "n_vars=$N_VARS  max_ops=$MAX_OPS  total_steps=$TOTAL_STEPS"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Kill it first: tmux kill-session -t $SESSION"
    exit 1
fi

# Wrapper: autorestart a crashed training so a transient OOM / segfault doesn't
# lose the run. Checkpoints persist via best_lvl*.pt so resumption has value.
# Max 3 retries, 30s backoff.
run_with_retry() {
    local log="$1"; shift
    local max=3
    local attempt=1
    while (( attempt <= max )); do
        echo "[attempt $attempt] $*" | tee -a "$log"
        if "$@" >>"$log" 2>&1; then
            echo "[done] $*" | tee -a "$log"
            return 0
        fi
        echo "[crash $attempt] exit=$?  retrying in 30s..." | tee -a "$log"
        sleep 30
        attempt=$((attempt + 1))
    done
    echo "[give-up] $*" | tee -a "$log"
    return 1
}
export -f run_with_retry

cd "$REPO_DIR"

tmux new-session -d -s "$SESSION" -n ctl "cd '$REPO_DIR'; echo 'controller: watch other windows'; exec bash"

# --- One training window per seed -------------------------------------------
for SEED in $SEEDS; do
    TAG="train_s${SEED}"
    RUN_LOG="$LOG_DIR/${TAG}.log"
    LOG_SUB_DIR="$RUN_ROOT/seed${SEED}"
    mkdir -p "$LOG_SUB_DIR"

    CMD="run_with_retry '$RUN_LOG' $PYTHON scripts/train.py \
        --n_vars $N_VARS --max_ops $MAX_OPS \
        --total_steps $TOTAL_STEPS \
        --reward_mode full \
        --seed $SEED \
        --log_dir '$LOG_SUB_DIR' \
        --interesting '$TRAIN_JSONL' \
        --eval_jsonl '$EVAL_JSONL'"

    tmux new-window -t "$SESSION" -n "$TAG" \
        "cd '$REPO_DIR'; $CMD; echo '--- finished ($TAG) ---'; exec bash"
done

# --- Baselines window (runs immediately; doesn't depend on training) --------
BASE_LOG="$LOG_DIR/baselines.log"
BASE_SKIP_EXHAUSTIVE=""
if [ "$MAX_OPS" -gt 4 ]; then
    BASE_SKIP_EXHAUSTIVE="--skip_exhaustive"
fi
BASE_CMD="$PYTHON scripts/run_baselines.py \
    --n_vars $N_VARS --max_ops $MAX_OPS \
    --eval_jsonl '$EVAL_JSONL' \
    $BASE_SKIP_EXHAUSTIVE \
    --seed 42 \
    --out '$RUN_ROOT/baselines.json'"
tmux new-window -t "$SESSION" -n baselines \
    "cd '$REPO_DIR'; echo '$BASE_CMD' | tee -a '$BASE_LOG'; $BASE_CMD 2>&1 | tee -a '$BASE_LOG'; echo '--- baselines done ---'; exec bash"

# --- Eval window: waits for training, then sweeps every best_lvl*.pt --------
EVAL_LOG="$LOG_DIR/eval.log"
EVAL_SCRIPT=$(cat <<EOF
echo "waiting for training windows to close..."
for s in $SEEDS; do
    while tmux list-windows -t $SESSION -F '#{window_name}' | grep -q "^train_s\${s}\$"; do
        # window removed once the final 'exec bash' shell exits. while any 'train_s*'
        # window exists (even post-exit bash) keep waiting; break after trainer wrote final.pt
        if [ -f "$RUN_ROOT/seed\${s}/final.pt" ]; then break; fi
        sleep 60
    done
done
echo "sweeping checkpoints..."
for s in $SEEDS; do
    for k in $LEVELS; do
        ckpt="$RUN_ROOT/seed\${s}/best_lvl\${k}.pt"
        [ -f "\$ckpt" ] || { echo "missing \$ckpt"; continue; }
        echo "=== seed=\$s lvl=\$k ===" | tee -a "$EVAL_LOG"
        $PYTHON scripts/evaluate.py --checkpoint "\$ckpt" --max_ops \$k --episodes 500 \
            2>&1 | tee -a "$EVAL_LOG"
    done
    ckpt="$RUN_ROOT/seed\${s}/final.pt"
    if [ -f "\$ckpt" ]; then
        echo "=== seed=\$s final ===" | tee -a "$EVAL_LOG"
        $PYTHON scripts/evaluate.py --checkpoint "\$ckpt" --episodes 500 \
            2>&1 | tee -a "$EVAL_LOG"
    fi
done
echo "--- eval done ---"
EOF
)
tmux new-window -t "$SESSION" -n eval \
    "cd '$REPO_DIR'; bash -c '$EVAL_SCRIPT'; exec bash"

# --- htop pane for quick vitals --------------------------------------------
tmux new-window -t "$SESSION" -n top "command -v htop >/dev/null && htop || top"

tmux select-window -t "$SESSION":ctl

echo
echo "Launched. Attach with:  tmux attach -t $SESSION"
echo "Kill all:               tmux kill-session -t $SESSION"
