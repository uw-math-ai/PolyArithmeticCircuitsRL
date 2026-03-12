#!/bin/bash
#SBATCH --job-name=jax_c5_c8
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=results/slurm/%x_%j.out

# ---- IMPORTANT ----
# If you see a Python 3.6 SyntaxError, rebuild the container:
#   apptainer build container.sif container.def
# The old container.sif uses Python 3.6; the new def uses Ubuntu 22.04 / Python 3.10+.

cd /gscratch/stf/rpande/AI_Lab/PolyArithmeticCircuitsRL
module load cuda

# PYTHONUNBUFFERED=1 ensures logs stream immediately (no buffering).
apptainer run --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container.sif \
  python -u -m src.main \
    --algorithm ppo-mcts-jax \
    --fixed-complexities 5 6 7 8 \
    --max-degree 6 \
    --max-steps 14 \
    --iterations 500 \
    --mcts-batch-size 512 \
    --mcts-simulations 32 \
    --ent-coef 0.05 \
    --wandb
