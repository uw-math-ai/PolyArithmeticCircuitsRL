#!/bin/bash
#SBATCH --job-name=sac_c6_gpu
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

cd /gscratch/stf/kj66/PolyArithmeticCircuitsRL_kj66-hyak
module load cuda
mkdir -p results/slurm

apptainer exec --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container.sif \
  python -u -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

apptainer exec --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container.sif \
  python -u -m src.main \
    --algorithm sac \
    --n-variables 2 \
    --max-complexity 6 \
    --device cuda \
    --iterations 500 \
    --sac-fixed-complexity-iters 0 \
    --factor-subgoal-reward 1.0 \
    --factor-library-bonus 0.5 \
    --completion-bonus 3.0 \
    --results-dir results/sac_fl_C6_2var \
    --wandb-run-name sac_fl_C6_2var \
    --wandb
