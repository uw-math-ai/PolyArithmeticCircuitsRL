#!/bin/bash
#SBATCH --job-name=lgs_noisy_baseline
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/gscratch/stf/kj66/learn-guided-search}"
SCRIPT_DIR="${REPO_ROOT}/slurm"
cd "${REPO_ROOT}"

NOISE_SIGMA="${NOISE_SIGMA:-1.0}"
MAX_INSTANCES_PER_FAMILY="${MAX_INSTANCES_PER_FAMILY:-20}"
BEAM_WIDTHS="${BEAM_WIDTHS:-1,2,4,8}"
CANDIDATE_KS="${CANDIDATE_KS:-4,8,16,32}"
TIER2_MS="${TIER2_MS:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-results/noisy_sigma_${NOISE_SIGMA}}"

source "${SCRIPT_DIR}/hyak_python.sh"
ENABLE_WANDB=0
PYTHON="$(bootstrap_lgs_python)"

mkdir -p results/slurm "${OUTPUT_DIR}"

echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Noise sigma: ${NOISE_SIGMA}"
echo "Output dir: ${OUTPUT_DIR}"

"${PYTHON}" scripts/run_benchmark_sweep.py \
  --noise-sigma "${NOISE_SIGMA}" \
  --max-instances-per-family "${MAX_INSTANCES_PER_FAMILY}" \
  --beam-widths "${BEAM_WIDTHS}" \
  --candidate-ks "${CANDIDATE_KS}" \
  --tier2-ms "${TIER2_MS}" \
  --no-wandb \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/run.log"
