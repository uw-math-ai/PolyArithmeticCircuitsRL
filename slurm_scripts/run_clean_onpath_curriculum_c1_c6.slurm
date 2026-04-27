#!/bin/bash
#SBATCH --job-name=clean_onpath_2var_c1_c6
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

REPO_DIR=${REPO_DIR:-/gscratch/stf/kj66/PolyArithmeticCircuitsRL_kj66-hyak}
CONTAINER=${CONTAINER:-container_jax.sif}

N_VARIABLES=${N_VARIABLES:-2}
MOD=${MOD:-5}
MAX_COMPLEXITY=${MAX_COMPLEXITY:-6}
MAX_DEGREE=${MAX_DEGREE:-6}
MAX_STEPS=${MAX_STEPS:-12}

CACHE_DIR=${CACHE_DIR:-on_path_cache/n2_mod5_deg6_C1_C6_routes32_seed42}
SPLIT_SEED=${SPLIT_SEED:-42}
MAX_ON_PATH_SIZE=${MAX_ON_PATH_SIZE:-4096}
ON_PATH_NUM_ROUTES=${ON_PATH_NUM_ROUTES:-32}
ON_PATH_PHI_MODE=${ON_PATH_PHI_MODE:-max_step}
GRAPH_ONPATH_SHAPING_COEFF=${GRAPH_ONPATH_SHAPING_COEFF:-1.0}

ITERATIONS=${ITERATIONS:-2000}
# Conservative GPU defaults. Increase to 256/32 only after confirming memory.
MCTS_BATCH_SIZE=${MCTS_BATCH_SIZE:-128}
MCTS_SIMULATIONS=${MCTS_SIMULATIONS:-16}
PPO_EPOCHS=${PPO_EPOCHS:-8}
ENT_COEF=${ENT_COEF:-0.05}

STARTING_COMPLEXITY=${STARTING_COMPLEXITY:-1}
ADVANCE_THRESHOLD=${ADVANCE_THRESHOLD:-0.7}
BACKOFF_THRESHOLD=${BACKOFF_THRESHOLD:-0.05}
CURRICULUM_WINDOW=${CURRICULUM_WINDOW:-512}
# MIN_DWELL=1 means one completed outer PPO iteration at the current level
# before another advance/backoff check; the success window is the main debounce.
CURRICULUM_MIN_DWELL_ITERATIONS=${CURRICULUM_MIN_DWELL_ITERATIONS:-4}

RESULTS_DIR=${RESULTS_DIR:-results/ppo-mcts-jax_clean_onpath_curriculum_2var_C1_C6}
WANDB=${WANDB:-1}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-ppo-mcts-jax_clean_onpath_curriculum_2var_C1_C6}

cd "$REPO_DIR"
module load cuda
mkdir -p results/slurm "$RESULTS_DIR"

for c in $(seq "$STARTING_COMPLEXITY" "$MAX_COMPLEXITY"); do
  cache_file="$CACHE_DIR/on_path_C${c}.npz"
  if [[ ! -f "$cache_file" ]]; then
    echo "Missing required OnPath cache file: $cache_file"
    echo "Build it first with slurm_scripts/build_on_path_cache_c1_c6.slurm"
    exit 1
  fi
done

echo "Starting large clean_onpath curriculum run"
echo "  repo: $REPO_DIR"
echo "  cache_dir: $CACHE_DIR"
echo "  results_dir: $RESULTS_DIR"
echo "  max_complexity: $MAX_COMPLEXITY"
echo "  iterations: $ITERATIONS"
echo "  mcts_batch_size: $MCTS_BATCH_SIZE"
echo "  mcts_simulations: $MCTS_SIMULATIONS"
echo "  ppo_epochs: $PPO_EPOCHS"
echo "  on_path_phi_mode: $ON_PATH_PHI_MODE"
echo "  starting_complexity: $STARTING_COMPLEXITY"
echo "  advance_threshold: $ADVANCE_THRESHOLD"
echo "  backoff_threshold: $BACKOFF_THRESHOLD"
echo "  curriculum_window: $CURRICULUM_WINDOW"
echo "  curriculum_min_dwell_iterations: $CURRICULUM_MIN_DWELL_ITERATIONS"
echo "  on_path_num_routes: $ON_PATH_NUM_ROUTES"

apptainer exec --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  --env XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \
  "$CONTAINER" \
  python -u -c "import jax; print('JAX devices:', jax.devices())"

WANDB_ARGS=()
if [[ "$WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-run-name "$WANDB_RUN_NAME")
fi

apptainer exec --nv --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  --env XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \
  "$CONTAINER" \
  python -u -m src.main \
    --algorithm ppo-mcts-jax \
    --n-variables "$N_VARIABLES" \
    --mod "$MOD" \
    --max-complexity "$MAX_COMPLEXITY" \
    --starting-complexity "$STARTING_COMPLEXITY" \
    --advance-threshold "$ADVANCE_THRESHOLD" \
    --backoff-threshold "$BACKOFF_THRESHOLD" \
    --curriculum-window "$CURRICULUM_WINDOW" \
    --curriculum-min-dwell-iterations "$CURRICULUM_MIN_DWELL_ITERATIONS" \
    --max-degree "$MAX_DEGREE" \
    --max-steps "$MAX_STEPS" \
    --iterations "$ITERATIONS" \
    --mcts-batch-size "$MCTS_BATCH_SIZE" \
    --mcts-simulations "$MCTS_SIMULATIONS" \
    --ppo-epochs "$PPO_EPOCHS" \
    --ent-coef "$ENT_COEF" \
    --reward-mode clean_onpath \
    --graph-onpath-cache-dir "$CACHE_DIR" \
    --graph-onpath-shaping-coeff "$GRAPH_ONPATH_SHAPING_COEFF" \
    --on-path-phi-mode "$ON_PATH_PHI_MODE" \
    --on-path-max-size "$MAX_ON_PATH_SIZE" \
    --on-path-split-seed "$SPLIT_SEED" \
    --on-path-num-routes "$ON_PATH_NUM_ROUTES" \
    --results-dir "$RESULTS_DIR" \
    "${WANDB_ARGS[@]}"
