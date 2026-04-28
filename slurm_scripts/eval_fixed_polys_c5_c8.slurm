#!/bin/bash
#SBATCH --job-name=eval_fixed_poly
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:30:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

cd /gscratch/stf/kj66/PolyArithmeticCircuitsRL_kj66-hyak
module load cuda
mkdir -p results/slurm

CHECKPOINT_DIR=${CHECKPOINT_DIR:-results/ppo-mcts-jax_fl_C5_C8}
ITERATIONS=${ITERATIONS:-"500"}
NUM_TRIALS=${NUM_TRIALS:-100}
MCTS_SIMULATIONS=${MCTS_SIMULATIONS:-32}
BATCH_SIZE=${BATCH_SIZE:-32}
TEMPERATURE=${TEMPERATURE:-0.0}
OUT_PREFIX=${OUT_PREFIX:-${CHECKPOINT_DIR}/fixed_poly_eval}
MAX_SAMPLES_PER_OUTCOME=${MAX_SAMPLES_PER_OUTCOME:-1}
XLA_FLAGS=${XLA_FLAGS:-"--xla_gpu_strict_conv_algorithm_picker=false"}
XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

echo "Fixed polynomial eval"
echo "  checkpoint_dir: ${CHECKPOINT_DIR}"
echo "  iterations: ${ITERATIONS}"
echo "  num_trials: ${NUM_TRIALS}"
echo "  mcts_simulations: ${MCTS_SIMULATIONS}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  temperature: ${TEMPERATURE}"
echo "  out_prefix: ${OUT_PREFIX}"
echo "  XLA_FLAGS: ${XLA_FLAGS}"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE: ${XLA_PYTHON_CLIENT_PREALLOCATE}"

# shellcheck disable=SC2086
apptainer exec --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  --env XLA_FLAGS="${XLA_FLAGS}" \
  --env XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE}" \
  container_jax.sif \
  python -u eval_fixed_polys_jax.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --iterations ${ITERATIONS} \
    --num-trials "${NUM_TRIALS}" \
    --mcts-simulations "${MCTS_SIMULATIONS}" \
    --batch-size "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --out-prefix "${OUT_PREFIX}" \
    --max-samples-per-outcome "${MAX_SAMPLES_PER_OUTCOME}"
