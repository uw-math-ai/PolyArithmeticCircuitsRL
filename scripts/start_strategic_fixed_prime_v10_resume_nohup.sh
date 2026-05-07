#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/Polynomial2"
USER_HOME="/home/ec2-user"
DEFAULT_RESUME_CKPT="$ROOT/artifacts/strategic-fixed-prime-v9-nohup-20260422_145500/checkpoints/stage_a.pt"
RESUME_CKPT="${RESUME_CKPT:-$DEFAULT_RESUME_CKPT}"
RUN_ID="${RUN_ID:-strategic-fixed-prime-v10-resume-$(date -u +%Y%m%d_%H%M%S)}"
WANDB_ENTITY="${WANDB_ENTITY:-p-agi}"
WANDB_PROJECT="${WANDB_PROJECT:-PolyArithmeticCircuitsRL}"
WANDB_MODE="${WANDB_MODE:-online}"
OUT_DIR="$ROOT/artifacts/$RUN_ID"
LOG_PATH="$OUT_DIR/nohup.log"
PID_PATH="$OUT_DIR/nohup.pid"

mkdir -p "$OUT_DIR"

nohup setsid /bin/bash -lc "
cd '$ROOT'
exec env \
  HOME='$USER_HOME' \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 \
  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
  '$ROOT/.venv/bin/python' \
  '$ROOT/scripts/run_full_experiment.py' \
  --output-dir '$OUT_DIR' \
  --resume-checkpoint '$RESUME_CKPT' \
  --prime 3 \
  --seed 0 \
  --initial-supervised-count 3072 \
  --holdout-count 192 \
  --cycles 14 \
  --search-targets-per-cycle 128 \
  --recent-distill-sample-size 128 \
  --replay-sample-size 49152 \
  --elite-sample-size 1024 \
  --synthetic-sample-size 8192 \
  --replay-capacity 131072 \
  --elite-capacity 12288 \
  --supervised-epochs 12 \
  --cycle-epochs 8 \
  --learning-rate 3e-4 \
  --cycle-learning-rate 2e-4 \
  --weight-decay 1e-4 \
  --value-loss-weight 0.25 \
  --search-simulations 192 \
  --cycle-search-retries 2 \
  --cycle-search-fresh-search-per-target \
  --cycle-search-progress-interval 8 \
  --device cuda \
  --batch-size 4096 \
  --auto-batch-size \
  --max-auto-batch-size 65536 \
  --gpu-memory-target-fraction 0.96 \
  --cache-dataset-on-device \
  --model-hidden-dim 4096 \
  --model-shared-layers 6 \
  --model-value-hidden-dim 2048 \
  --model-value-layers 4 \
  --model-activation gelu \
  --reserve-cpu-cores 4 \
  --torch-cpu-threads 8 \
  --torch-interop-threads 1 \
  --nice-level 10 \
  --replay-uniform-fraction 0.05 \
  --curriculum-extra-primes= \
  --curriculum-max-vars 6 \
  --curriculum-max-support 7 \
  --curriculum-max-degree 5 \
  --curriculum-max-horner-degree 10 \
  --curriculum-max-inner-support 5 \
  --wandb-entity '$WANDB_ENTITY' \
  --wandb-project '$WANDB_PROJECT' \
  --wandb-run-id '$RUN_ID' \
  --wandb-mode '$WANDB_MODE'
" >"$LOG_PATH" 2>&1 </dev/null &

PID=$!
printf '%s\n' "$PID" >"$PID_PATH"
printf 'run_id=%s\npid=%s\nresume=%s\nlog=%s\n' "$RUN_ID" "$PID" "$RESUME_CKPT" "$LOG_PATH"
