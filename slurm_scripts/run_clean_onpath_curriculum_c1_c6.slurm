#!/bin/bash
#SBATCH --job-name=clean_onpath_2var_c1_c5
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
MAX_COMPLEXITY=${MAX_COMPLEXITY:-5}
MAX_BUILD_COMPLEXITY=${MAX_BUILD_COMPLEXITY:-$MAX_COMPLEXITY}
BUILD_COMPLEXITY_SLACK=${BUILD_COMPLEXITY_SLACK:--1}
MAX_DEGREE=${MAX_DEGREE:-5}
MAX_STEPS=${MAX_STEPS:-12}

CACHE_DIR=${CACHE_DIR:-on_path_cache/c1_c5}
SPLIT_SEED=${SPLIT_SEED:-42}
MAX_ON_PATH_SIZE=${MAX_ON_PATH_SIZE:-4096}
ON_PATH_NUM_ROUTES=${ON_PATH_NUM_ROUTES:-32}
ON_PATH_PHI_MODE=${ON_PATH_PHI_MODE:-depth_weighted}
ON_PATH_DEPTH_WEIGHT_POWER=${ON_PATH_DEPTH_WEIGHT_POWER:-1.0}
ON_PATH_ROUTE_CONSISTENCY_MODE=${ON_PATH_ROUTE_CONSISTENCY_MODE:-best_route_phi}
GRAPH_ONPATH_SHAPING_COEFF=${GRAPH_ONPATH_SHAPING_COEFF:-5.0}
ON_ROUTE_BONUS_COEFF=${ON_ROUTE_BONUS_COEFF:-0.0}

ITERATIONS=${ITERATIONS:-2000}
# Conservative GPU defaults. Increase to 256/32 only after confirming memory.
MCTS_BATCH_SIZE=${MCTS_BATCH_SIZE:-128}
MCTS_SIMULATIONS=${MCTS_SIMULATIONS:-32}
PPO_LR=${PPO_LR:-1e-4}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.25}
PPO_LOG_RATIO_CLIP=${PPO_LOG_RATIO_CLIP:-10.0}
PPO_EPOCHS=${PPO_EPOCHS:-2}
TARGET_KL=${TARGET_KL:-0.05}
ENT_COEF=${ENT_COEF:-0.15}
ENT_COEF_FINAL=${ENT_COEF_FINAL:-0.01}
ENT_COEF_ANNEAL_FRACTION=${ENT_COEF_ANNEAL_FRACTION:-0.5}
VF_COEF=${VF_COEF:-0.5}
VALUE_CLIP_RANGE=${VALUE_CLIP_RANGE:-0.2}
LOG_INTERVAL=${LOG_INTERVAL:-10}
PROGRESS_BAR=${PROGRESS_BAR:-0}

STARTING_COMPLEXITY=${STARTING_COMPLEXITY:-2}
ADVANCE_THRESHOLD=${ADVANCE_THRESHOLD:-0.7}
# Negative backoff disables curriculum backoff for stable C1->C3/C1->C6 debugging.
BACKOFF_THRESHOLD=${BACKOFF_THRESHOLD:--1.0}
CURRICULUM_WINDOW=${CURRICULUM_WINDOW:-3000}
# MIN_DWELL is counted in completed outer PPO iterations at the current level.
CURRICULUM_MIN_DWELL_ITERATIONS=${CURRICULUM_MIN_DWELL_ITERATIONS:-50}
CURRICULUM_BACKOFF_PATIENCE_ITERATIONS=${CURRICULUM_BACKOFF_PATIENCE_ITERATIONS:-0}

RESULTS_DIR=${RESULTS_DIR:-results/onpath_2var_C1_C5}
WANDB=${WANDB:-1}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-onpath_2var_C1_C5}
FIXED_COMPLEXITIES=${FIXED_COMPLEXITIES:-}

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
echo "  max_build_complexity: $MAX_BUILD_COMPLEXITY"
echo "  build_complexity_slack: $BUILD_COMPLEXITY_SLACK"
echo "  max_steps: $MAX_STEPS"
echo "  iterations: $ITERATIONS"
echo "  mcts_batch_size: $MCTS_BATCH_SIZE"
echo "  mcts_simulations: $MCTS_SIMULATIONS"
echo "  ppo_lr: $PPO_LR"
echo "  max_grad_norm: $MAX_GRAD_NORM"
echo "  ppo_log_ratio_clip: $PPO_LOG_RATIO_CLIP"
echo "  ppo_epochs: $PPO_EPOCHS"
echo "  target_kl: $TARGET_KL"
echo "  ent_coef: $ENT_COEF"
echo "  ent_coef_final: $ENT_COEF_FINAL"
echo "  ent_coef_anneal_fraction: $ENT_COEF_ANNEAL_FRACTION"
echo "  vf_coef: $VF_COEF"
echo "  value_clip_range: $VALUE_CLIP_RANGE"
echo "  on_path_phi_mode: $ON_PATH_PHI_MODE"
echo "  on_route_bonus_coeff: $ON_ROUTE_BONUS_COEFF"
echo "  on_path_depth_weight_power: $ON_PATH_DEPTH_WEIGHT_POWER"
echo "  on_path_route_consistency_mode: $ON_PATH_ROUTE_CONSISTENCY_MODE"
echo "  starting_complexity: $STARTING_COMPLEXITY"
echo "  advance_threshold: $ADVANCE_THRESHOLD"
echo "  backoff_threshold: $BACKOFF_THRESHOLD"
echo "  curriculum_window: $CURRICULUM_WINDOW"
echo "  curriculum_min_dwell_iterations: $CURRICULUM_MIN_DWELL_ITERATIONS"
echo "  curriculum_backoff_patience_iterations: $CURRICULUM_BACKOFF_PATIENCE_ITERATIONS"
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
PROGRESS_ARGS=()
if [[ "$PROGRESS_BAR" != "1" ]]; then
  PROGRESS_ARGS+=(--no-progress-bar)
fi
FIXED_ARGS=()
if [[ -n "$FIXED_COMPLEXITIES" ]]; then
  # shellcheck disable=SC2206
  FIXED_ARGS=(--fixed-complexities $FIXED_COMPLEXITIES)
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
    --max-build-complexity "$MAX_BUILD_COMPLEXITY" \
    --build-complexity-slack "$BUILD_COMPLEXITY_SLACK" \
    --starting-complexity "$STARTING_COMPLEXITY" \
    --advance-threshold "$ADVANCE_THRESHOLD" \
    --backoff-threshold "$BACKOFF_THRESHOLD" \
    --curriculum-window "$CURRICULUM_WINDOW" \
    --curriculum-min-dwell-iterations "$CURRICULUM_MIN_DWELL_ITERATIONS" \
    --curriculum-backoff-patience-iterations "$CURRICULUM_BACKOFF_PATIENCE_ITERATIONS" \
    --max-degree "$MAX_DEGREE" \
    --max-steps "$MAX_STEPS" \
    --iterations "$ITERATIONS" \
    --mcts-batch-size "$MCTS_BATCH_SIZE" \
    --mcts-simulations "$MCTS_SIMULATIONS" \
    --ppo-lr "$PPO_LR" \
    --ppo-epochs "$PPO_EPOCHS" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --ppo-log-ratio-clip "$PPO_LOG_RATIO_CLIP" \
    --target-kl "$TARGET_KL" \
    --ent-coef "$ENT_COEF" \
    --ent-coef-final "$ENT_COEF_FINAL" \
    --ent-coef-anneal-fraction "$ENT_COEF_ANNEAL_FRACTION" \
    --vf-coef "$VF_COEF" \
    --value-clip-range "$VALUE_CLIP_RANGE" \
    --log-interval "$LOG_INTERVAL" \
    --reward-mode clean_onpath \
    --graph-onpath-cache-dir "$CACHE_DIR" \
    --graph-onpath-shaping-coeff "$GRAPH_ONPATH_SHAPING_COEFF" \
    --on-route-bonus-coeff "$ON_ROUTE_BONUS_COEFF" \
    --on-path-phi-mode "$ON_PATH_PHI_MODE" \
    --on-path-depth-weight-power "$ON_PATH_DEPTH_WEIGHT_POWER" \
    --on-path-route-consistency-mode "$ON_PATH_ROUTE_CONSISTENCY_MODE" \
    --on-path-max-size "$MAX_ON_PATH_SIZE" \
    --on-path-split-seed "$SPLIT_SEED" \
    --on-path-num-routes "$ON_PATH_NUM_ROUTES" \
    --results-dir "$RESULTS_DIR" \
    "${FIXED_ARGS[@]}" \
    "${PROGRESS_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
