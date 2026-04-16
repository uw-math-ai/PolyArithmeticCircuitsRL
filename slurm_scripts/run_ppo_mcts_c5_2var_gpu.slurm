#!/bin/bash
#SBATCH --job-name=ppo_mcts_c5_2var_gpu
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

cd /gscratch/stf/rpande/AI_Lab/PolyArithmeticCircuitsRL
module load cuda
mkdir -p results/slurm

# Verify JAX/CUDA is visible before starting.
apptainer exec --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container_jax.sif \
  python -u -c "import jax; print('JAX devices:', jax.devices())"

# Run JAX PPO+MCTS with the factor library on fixed C5 2-variable targets.
apptainer exec --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container_jax.sif \
  python -u -m src.main \
    --algorithm ppo-mcts-jax \
    --n-variables 2 \
    --fixed-complexities 5 \
    --iterations 200 \
    --mcts-batch-size 256 \
    --mcts-simulations 16 \
    --ent-coef 0.05 \
    --factor-subgoal-reward 1.0 \
    --factor-library-bonus 0.5 \
    --completion-bonus 3.0 \
    --results-dir results/ppo-mcts-jax_fl_C5_2var \
    --wandb-run-name ppo-mcts-jax_fl_C5_2var \
    --wandb
