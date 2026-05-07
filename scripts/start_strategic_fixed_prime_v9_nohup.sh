#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/Polynomial2"
RUN_ID="strategic-fixed-prime-v9-nohup-20260422_145500"
OUT_DIR="$ROOT/artifacts/$RUN_ID"
LOG_PATH="$OUT_DIR/nohup.log"
PID_PATH="$OUT_DIR/nohup.pid"

mkdir -p "$OUT_DIR"

nohup setsid /bin/bash -lc "
cd '$ROOT'
exec env \
  HOME='$ROOT/.sage_home' \
  XDG_CACHE_HOME='$ROOT/.sage_home/.cache' \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 \
  '$ROOT/.venv/bin/python' \
  '$ROOT/scripts/run_full_experiment.py' \
  --output-dir '$OUT_DIR' \
  --prime 3 \
  --seed 0 \
  --initial-supervised-count 3072 \
  --holdout-count 192 \
  --cycles 14 \
  --search-targets-per-cycle 96 \
  --recent-distill-sample-size 96 \
  --replay-sample-size 2048 \
  --elite-sample-size 384 \
  --synthetic-sample-size 2048 \
  --replay-capacity 49152 \
  --elite-capacity 6144 \
  --supervised-epochs 12 \
  --cycle-epochs 6 \
  --learning-rate 3e-4 \
  --cycle-learning-rate 2e-4 \
  --weight-decay 1e-4 \
  --value-loss-weight 0.25 \
  --search-simulations 160 \
  --cycle-search-retries 2 \
  --cycle-search-fresh-search-per-target \
  --cycle-search-progress-interval 8 \
  --device cuda \
  --batch-size 1024 \
  --auto-batch-size \
  --max-auto-batch-size 16384 \
  --gpu-memory-target-fraction 0.94 \
  --model-hidden-dim 4096 \
  --model-shared-layers 6 \
  --model-value-hidden-dim 2048 \
  --model-value-layers 4 \
  --model-activation gelu \
  --reserve-cpu-cores 4 \
  --torch-cpu-threads 8 \
  --torch-interop-threads 1 \
  --nice-level 10 \
  --replay-uniform-fraction 0.1 \
  --curriculum-extra-primes= \
  --curriculum-max-vars 6 \
  --curriculum-max-support 7 \
  --curriculum-max-degree 5 \
  --curriculum-max-horner-degree 10 \
  --curriculum-max-inner-support 5 \
  --wandb-run-id '$RUN_ID' \
  --wandb-mode disabled
" >"$LOG_PATH" 2>&1 </dev/null &

PID=$!
printf '%s\n' "$PID" >"$PID_PATH"
printf 'run_id=%s\npid=%s\nlog=%s\n' "$RUN_ID" "$PID" "$LOG_PATH"
