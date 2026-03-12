#!/bin/bash
#SBATCH --job-name=sac_c5_gpu
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out

cd /gscratch/stf/rpande/AI_Lab/PolyArithmeticCircuitsRL
module load cuda

apptainer run --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container.sif \
  python -u -m src.main \
    --algorithm sac \
    --max-complexity 5 \
    --iterations 500 \
    --wandb
