#!/bin/bash
#SBATCH --job-name=jax_c5_c8
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

cd /gscratch/stf/rpande/AI_Lab/PolyArithmeticCircuitsRL
module load cuda
mkdir -p results/slurm

# PYTHONUNBUFFERED=1 ensures logs stream immediately (no buffering).
apptainer run --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container_jax.sif \
  python -u -m src.main \
    --algorithm ppo-mcts-jax \
    --fixed-complexities 5 6 7 8 \
    --max-degree 6 \
    --max-steps 14 \
    --iterations 500 \
    --mcts-batch-size 512 \
    --mcts-simulations 32 \
    --ent-coef 0.05 \
    --factor-subgoal-reward 1.0 \
    --factor-library-bonus 0.5 \
    --completion-bonus 3.0 \
    --results-dir results/ppo-mcts-jax_fl_C5_C8 \
    --wandb-run-name ppo-mcts-jax_fl_C5_C8 \
    --wandb
