#!/bin/bash
#SBATCH --job-name=baselines_kj
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

cd /gscratch/stf/kj66/PolyArithmeticCircuitsRL
mkdir -p results/slurm runs

module load apptainer

if [ ! -f container.sif ]; then
  echo "ERROR: container.sif not found in $(pwd)"
  exit 2
fi

export PYTHONUNBUFFERED=1
export N_VARS="${N_VARS:-2}"
export MAX_OPS="${MAX_OPS:-6}"
export SEED="${SEED:-42}"

DATA_DIR="${DATA_DIR:-data}"
DATASET_TAG="polys_nvars${N_VARS}_maxops${MAX_OPS}"
EVAL_JSONL="${EVAL_JSONL:-$DATA_DIR/${DATASET_TAG}.eval.jsonl}"

if [ ! -f "$EVAL_JSONL" ]; then
  echo "ERROR: Eval split not found: $EVAL_JSONL"
  echo "Run slurm_scripts/run_build_dataset_kj.slurm first."
  exit 2
fi

OUT_PATH="${OUT_PATH:-runs/baselines_nvars${N_VARS}_maxops${MAX_OPS}_seed${SEED}.json}"

echo "[baselines] Starting at $(date)"
echo "[baselines] EVAL_JSONL=$EVAL_JSONL"
echo "[baselines] OUT_PATH=$OUT_PATH"

apptainer exec --bind /gscratch:/gscratch container.sif \
  python scripts/run_baselines.py \
    --n_vars "$N_VARS" \
    --max_ops "$MAX_OPS" \
    --eval_jsonl "$EVAL_JSONL" \
    --skip_exhaustive \
    --seed "$SEED" \
    --out "$OUT_PATH"

echo "[baselines] Done at $(date)"

