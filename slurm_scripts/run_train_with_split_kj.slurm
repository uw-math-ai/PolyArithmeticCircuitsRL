#!/bin/bash
#SBATCH --job-name=train_with_split_kj
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

cd /gscratch/stf/kj66/PolyArithmeticCircuitsRL
mkdir -p results/slurm runs

module load cuda
module load apptainer

if [ ! -f container.sif ]; then
  echo "ERROR: container.sif not found in $(pwd)"
  exit 2
fi

export PYTHONUNBUFFERED=1
export N_VARS="${N_VARS:-2}"
export MAX_OPS="${MAX_OPS:-6}"
export TOTAL_STEPS="${TOTAL_STEPS:-500000}"
export SEED="${SEED:-42}"
export DATA_DIR="${DATA_DIR:-data}"
export AUTO_BUILD_DATASET=0
export ENABLE_WANDB="${ENABLE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-PolyArithmeticCircuitsRL}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-kj_split_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"

DATASET_TAG="polys_nvars${N_VARS}_maxops${MAX_OPS}"
TRAIN_JSONL="${TRAIN_JSONL:-$DATA_DIR/${DATASET_TAG}.train.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-$DATA_DIR/${DATASET_TAG}.eval.jsonl}"

if [ ! -f "$TRAIN_JSONL" ] || [ ! -f "$EVAL_JSONL" ]; then
  echo "ERROR: Required split files missing:"
  echo "  $TRAIN_JSONL"
  echo "  $EVAL_JSONL"
  echo "Run slurm_scripts/run_build_dataset_kj.slurm first."
  exit 2
fi

if [ -z "${LOG_DIR:-}" ]; then
  LOG_DIR="runs/split_train_${SEED}_$(date +%Y%m%d_%H%M%S)"
fi
export LOG_DIR

echo "[train] Starting at $(date)"
echo "[train] N_VARS=$N_VARS MAX_OPS=$MAX_OPS TOTAL_STEPS=$TOTAL_STEPS SEED=$SEED"
echo "[train] LOG_DIR=$LOG_DIR"
echo "[train] ENABLE_WANDB=$ENABLE_WANDB"

EXTRA_ARGS=()
if [ "$ENABLE_WANDB" = "1" ]; then
  EXTRA_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME")
  if [ -n "$WANDB_ENTITY" ]; then
    EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[train] WARNING: ENABLE_WANDB=1 but WANDB_API_KEY is not set in this job environment."
  fi
fi

apptainer exec --nv --bind /gscratch:/gscratch container.sif \
  bash scripts/train_with_split.sh "${EXTRA_ARGS[@]}"

echo "[train] Done at $(date)"
