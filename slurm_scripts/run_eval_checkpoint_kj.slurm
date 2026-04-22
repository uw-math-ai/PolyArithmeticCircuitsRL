#!/bin/bash
#SBATCH --job-name=eval_checkpoint_kj
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

cd /gscratch/stf/kj66/PolyArithmeticCircuitsRL
mkdir -p results/slurm

module load apptainer

if [ ! -f container.sif ]; then
  echo "ERROR: container.sif not found in $(pwd)"
  exit 2
fi

if [ -z "${CHECKPOINT:-}" ]; then
  echo "ERROR: CHECKPOINT is required."
  echo "Usage: sbatch --export=ALL,CHECKPOINT=runs/<run_dir>/best_lvl6.pt slurm_scripts/run_eval_checkpoint_kj.slurm"
  exit 2
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: CHECKPOINT file not found: $CHECKPOINT"
  exit 2
fi

export PYTHONUNBUFFERED=1
export EPISODES="${EPISODES:-500}"

echo "[eval] Starting at $(date)"
echo "[eval] CHECKPOINT=$CHECKPOINT"
echo "[eval] EPISODES=$EPISODES"
if [ -n "${MAX_OPS_OVERRIDE:-}" ]; then
  echo "[eval] MAX_OPS_OVERRIDE=$MAX_OPS_OVERRIDE"
fi

apptainer exec --bind /gscratch:/gscratch container.sif \
  bash scripts/eval_checkpoint.sh "$CHECKPOINT"

echo "[eval] Done at $(date)"

