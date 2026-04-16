#!/bin/bash
#SBATCH --job-name=eval_ppo_mcts_2var
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

cd /gscratch/stf/rpande/AI_Lab/PolyArithmeticCircuitsRL
module load cuda
mkdir -p results/slurm

# Evaluate C5 2-var checkpoints
apptainer run --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container_jax.sif \
  python -u eval_ppo_mcts_jax.py \
    --checkpoint-dir results/ppo-mcts-jax_fl_C5_2var \
    --iterations 50 100 150 200 \
    --complexities 5 \
    --num-trials 1000 \
    --mcts-simulations 16 \
    --temperature 0.1 \
    --wandb

# Evaluate C6 2-var checkpoints
apptainer run --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container_jax.sif \
  python -u eval_ppo_mcts_jax.py \
    --checkpoint-dir results/ppo-mcts-jax_fl_C6_2var \
    --iterations 50 100 150 200 \
    --complexities 6 \
    --num-trials 1000 \
    --mcts-simulations 16 \
    --temperature 0.1 \
    --wandb
