#!/bin/bash
#SBATCH --job-name=build_dataset_kj
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

cd /gscratch/stf/kj66/PolyArithmeticCircuitsRL
mkdir -p results/slurm data

module load apptainer

if [ ! -f container.sif ]; then
  echo "ERROR: container.sif not found in $(pwd)"
  exit 2
fi

export PYTHONUNBUFFERED=1
export N_VARS="${N_VARS:-2}"
export MAX_OPS="${MAX_OPS:-6}"
export OUT_DIR="${OUT_DIR:-data}"
export EVAL_FRAC="${EVAL_FRAC:-0.2}"
export SEED="${SEED:-0}"
export MAX_GRAPH_NODES="${MAX_GRAPH_NODES:-500000}"
export MAX_SECONDS="${MAX_SECONDS:-900}"

echo "[dataset] Starting at $(date)"
apptainer exec --bind /gscratch:/gscratch container.sif \
  bash scripts/make_dataset.sh
echo "[dataset] Done at $(date)"

