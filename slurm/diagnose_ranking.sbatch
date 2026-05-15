#!/bin/bash
#SBATCH --job-name=lgs_diagnose
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/gscratch/stf/kj66/learn-guided-search}"
SCRIPT_DIR="${REPO_ROOT}/slurm"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:-results/server_runs/server_train_eval_full_001/ranker.pt}"
MAX_INSTANCES_PER_FAMILY="${MAX_INSTANCES_PER_FAMILY:-2}"
TOP_K="${TOP_K:-10}"
MAX_STATES="${MAX_STATES:-5}"
LEARNED_ONLY="${LEARNED_ONLY:-0}"

source "${SCRIPT_DIR}/hyak_python.sh"
ENABLE_WANDB=0
PYTHON="$(bootstrap_lgs_python)"

mkdir -p results/slurm

echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Learned-only: ${LEARNED_ONLY}"

CMD=(
  "${PYTHON}" scripts/diagnose_ranking.py
  --max-instances-per-family "${MAX_INSTANCES_PER_FAMILY}"
  --top-k "${TOP_K}"
  --max-states "${MAX_STATES}"
)

if [[ -n "${CHECKPOINT}" && -f "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}")
  echo "Using checkpoint: ${CHECKPOINT}"
else
  echo "No checkpoint found — running heuristic-only baseline"
fi

if [[ "${LEARNED_ONLY}" == "1" ]]; then
  CMD+=(--learned-only)
fi

echo "Command: ${CMD[*]}"
echo "---"
"${CMD[@]}"
