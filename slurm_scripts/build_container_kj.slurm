#!/bin/bash
#SBATCH --job-name=build_container_kj
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

# Prevent mksquashfs thread creation failures on shared nodes.
export APPTAINER_MKSQUASHFS_PROCS=1

echo "[build] Starting container build at $(date)"
apptainer build --fakeroot container.sif container.def

echo "[smoke] Verifying imports inside container"
apptainer exec container.sif \
  python -c "import torch, gymnasium, networkx, sympy, wandb; print('ok')"

echo "[done] container.sif ready at $(date)"
