#!/usr/bin/env bash
set -euo pipefail

cd /home/ec2-user/Polynomial2

exec env \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 \
  HOME=/home/ec2-user/Polynomial2/.sage_home \
  XDG_CACHE_HOME=/home/ec2-user/Polynomial2/.sage_home/.cache \
  /home/ec2-user/Polynomial2/.venv/bin/python \
  /home/ec2-user/Polynomial2/scripts/run_full_experiment.py \
  --output-dir /home/ec2-user/Polynomial2/artifacts/strategic-fixed-prime-v7-nohup-20260422_001900 \
  --prime 3 \
  --seed 0 \
  --initial-supervised-count 2048 \
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
  --search-simulations 160 \
  --device cuda \
  --batch-size 1024 \
  --auto-batch-size \
  --max-auto-batch-size 16384 \
  --gpu-memory-target-fraction 0.94 \
  --model-hidden-dim 6144 \
  --model-shared-layers 7 \
  --model-value-hidden-dim 3072 \
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
  --wandb-run-id strategic-fixed-prime-v7-nohup-20260422_001900 \
  --wandb-mode disabled
