#!/bin/bash
#SBATCH --job-name=ppo_mcts_c5_gpu
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out

cd /gscratch/stf/rpande/AI_Lab/PolyArithmeticCircuitsRL
module load cuda

apptainer run --nv --bind /gscratch:/gscratch \
  container.sif \
  python -m src.main \
    --algorithm ppo-mcts-jax \
    --max-complexity 5 \
    --iterations 200 \
    --mcts-simulations 50 \
    --steps-per-update 512 \
    --wandb
  
