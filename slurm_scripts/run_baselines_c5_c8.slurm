#!/bin/bash
#SBATCH --job-name=baselines_c5_c8
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

cd /gscratch/stf/kj66/PolyArithmeticCircuitsRL_kj66-hyak
module load cuda
mkdir -p results/slurm results/baselines

# Knobs (override at submission via env vars).
COMPLEXITIES=${COMPLEXITIES:-"5 6 7 8"}
NUM_TRIALS=${NUM_TRIALS:-50}
SEED=${SEED:-42}
N_VARIABLES=${N_VARIABLES:-2}
MOD=${MOD:-5}
MAX_DEGREE=${MAX_DEGREE:-6}
MAX_STEPS=${MAX_STEPS:-14}
MCTS_SIMULATIONS=${MCTS_SIMULATIONS:-32}
CHECKPOINT=${CHECKPOINT:-results/ppo-mcts-jax_fl_C5_C8/checkpoint_00500.pkl}
TRUE_COMPLEXITY_JSON=${TRUE_COMPLEXITY_JSON:-results/random_circuit_true_complexity.json}
TARGET_CACHE=${TARGET_CACHE:-results/baselines/target_cache.npz}
OUT_CSV=${OUT_CSV:-results/baselines/comparison_C5_C8.csv}
METHODS=${METHODS:-"greedy_1step uniform_mcts_32 ppo_mcts_checkpoint"}
WANDB=${WANDB:-1}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-baselines_C5_C8}

echo "Baseline comparison run"
echo "  complexities:     $COMPLEXITIES"
echo "  num_trials:       $NUM_TRIALS"
echo "  seed:             $SEED"
echo "  mcts_simulations: $MCTS_SIMULATIONS"
echo "  checkpoint:       $CHECKPOINT"
echo "  target_cache:     $TARGET_CACHE"
echo "  out_csv:          $OUT_CSV"
echo "  methods:          $METHODS"

# Step 1: build the shared target cache (deterministic given --seed).
apptainer exec --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container_jax.sif \
  python -u scripts/build_baseline_target_cache.py \
    --complexities $COMPLEXITIES \
    --num-trials "$NUM_TRIALS" \
    --n-variables "$N_VARIABLES" \
    --mod "$MOD" \
    --max-degree "$MAX_DEGREE" \
    --max-steps "$MAX_STEPS" \
    --seed "$SEED" \
    --true-complexity-json "$TRUE_COMPLEXITY_JSON" \
    --out "$TARGET_CACHE"

# Step 2: run all methods on the shared cache.
WANDB_ARGS=()
if [[ "$WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-run-name "$WANDB_RUN_NAME")
fi

# shellcheck disable=SC2086
apptainer exec --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  --env XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  container_jax.sif \
  python -u scripts/run_baseline_comparison.py \
    --target-cache "$TARGET_CACHE" \
    --methods $METHODS \
    --mcts-simulations "$MCTS_SIMULATIONS" \
    --checkpoint "$CHECKPOINT" \
    --out-csv "$OUT_CSV" \
    --seed "$SEED" \
    "${WANDB_ARGS[@]}"
