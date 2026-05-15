#!/bin/bash
#SBATCH --job-name=lgs_learned_only
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/gscratch/stf/kj66/learn-guided-search}"
SCRIPT_DIR="${REPO_ROOT}/slurm"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:-results/server_runs/server_train_eval_full_001/ranker.pt}"
MAX_INSTANCES_PER_FAMILY="${MAX_INSTANCES_PER_FAMILY:-2}"
BEAM_WIDTHS="${BEAM_WIDTHS:-1,2,4,8,16}"
CANDIDATE_KS="${CANDIDATE_KS:-4,8,16,32,64}"
TIER2_MS="${TIER2_MS:-64,128}"
LAMBDA_MODEL="${LAMBDA_MODEL:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/diagnose_learned_only}"

source "${SCRIPT_DIR}/hyak_python.sh"
ENABLE_WANDB=0
PYTHON="$(bootstrap_lgs_python)"

mkdir -p results/slurm "${OUTPUT_DIR}"

echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output dir: ${OUTPUT_DIR}"

"${PYTHON}" scripts/run_benchmark_sweep.py \
  --checkpoint "${CHECKPOINT}" \
  --learned-only \
  --lambda-model "${LAMBDA_MODEL}" \
  --max-instances-per-family "${MAX_INSTANCES_PER_FAMILY}" \
  --beam-widths "${BEAM_WIDTHS}" \
  --candidate-ks "${CANDIDATE_KS}" \
  --tier2-ms "${TIER2_MS}" \
  --no-wandb \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/run.log"
