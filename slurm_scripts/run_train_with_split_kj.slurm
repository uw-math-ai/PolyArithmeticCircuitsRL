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
export SKIP_OPTIMAL_PRECOMPUTE="${SKIP_OPTIMAL_PRECOMPUTE:-1}"
export SKIP_EXPERT_DEMOS="${SKIP_EXPERT_DEMOS:-1}"
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
echo "[train] SKIP_OPTIMAL_PRECOMPUTE=$SKIP_OPTIMAL_PRECOMPUTE"
echo "[train] SKIP_EXPERT_DEMOS=$SKIP_EXPERT_DEMOS"
echo "[train] ENABLE_WANDB=$ENABLE_WANDB"

EXTRA_ARGS=()
if [ "$ENABLE_WANDB" = "1" ]; then
  EXTRA_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME")
  if [ -n "$WANDB_ENTITY" ]; then
    EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.netrc" ]; then
    WANDB_API_KEY_FROM_NETRC="$(
      awk '
        $1=="machine" {inblock=($2=="api.wandb.ai"); next}
        inblock && $1=="password" {print $2; exit}
      ' "$HOME/.netrc"
    )"
    if [ -n "$WANDB_API_KEY_FROM_NETRC" ]; then
      export WANDB_API_KEY="$WANDB_API_KEY_FROM_NETRC"
      echo "[train] Loaded WANDB_API_KEY from $HOME/.netrc"
    fi
    unset WANDB_API_KEY_FROM_NETRC
  fi
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[train] WARNING: ENABLE_WANDB=1 but WANDB_API_KEY is not set and could not be read from $HOME/.netrc."
  fi
fi

export APPTAINERENV_SKIP_OPTIMAL_PRECOMPUTE="$SKIP_OPTIMAL_PRECOMPUTE"
export APPTAINERENV_SKIP_EXPERT_DEMOS="$SKIP_EXPERT_DEMOS"
if [ -n "${WANDB_API_KEY:-}" ]; then
  export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
fi

apptainer exec --nv --bind /gscratch:/gscratch --bind "$HOME:$HOME" container.sif \
  bash scripts/train_with_split.sh "${EXTRA_ARGS[@]}"

echo "[train] Done at $(date)"
