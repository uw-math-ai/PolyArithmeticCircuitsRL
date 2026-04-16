#!/bin/bash
#SBATCH --job-name=sac_c5_gpu
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

cd /gscratch/stf/rpande/AI_Lab/PolyArithmeticCircuitsRL
module load cuda
mkdir -p results/slurm

apptainer run --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container.sif \
  python -u -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

apptainer run --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container.sif \
  python -u -m src.main \
    --algorithm sac \
    --n-variables 2 \
    --no-curriculum \
    --max-complexity 5 \
    --device cuda \
    --iterations 500 \
    --factor-subgoal-reward 1.0 \
    --factor-library-bonus 0.5 \
    --completion-bonus 3.0 \
    --results-dir results/sac_fl_C5_2var \
    --wandb-run-name sac_fl_C5_2var \
    --wandb
