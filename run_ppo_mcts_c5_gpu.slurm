#!/bin/bash
#SBATCH --job-name=ppo_mcts_c5_gpu
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=results/slurm/%x_%j.out

cd /gscratch/stf/rpande/AI_Lab/PolyArithmeticCircuitsRL
module load cuda

# Verify GPU is visible before starting
apptainer run --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container_jax.sif \
  python -u -c "import jax; print('JAX devices:', jax.devices())"

# Run C5 training
apptainer run --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container_jax.sif \
  python -u -m src.main \
    --algorithm ppo-mcts-jax \
    --fixed-complexities 5 \
    --iterations 200 \
    --mcts-batch-size 256 \
    --mcts-simulations 16 \
    --ent-coef 0.05 \
    --wandb
