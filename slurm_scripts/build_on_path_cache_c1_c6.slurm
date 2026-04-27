#!/bin/bash
#SBATCH --job-name=onpath_cache_2var_c1_c6
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

REPO_DIR=${REPO_DIR:-/gscratch/stf/kj66/PolyArithmeticCircuitsRL_kj66-hyak}
CONTAINER=${CONTAINER:-container_jax.sif}

N_VARIABLES=${N_VARIABLES:-2}
MOD=${MOD:-5}
MAX_DEGREE=${MAX_DEGREE:-6}
COMPLEXITIES=${COMPLEXITIES:-"1 2 3 4 5 6"}
SPLIT_SEED=${SPLIT_SEED:-42}
MAX_ON_PATH_SIZE=${MAX_ON_PATH_SIZE:-4096}
ON_PATH_NUM_ROUTES=${ON_PATH_NUM_ROUTES:-32}
MAX_ROUTE_TRUNCATION_RATE=${MAX_ROUTE_TRUNCATION_RATE:-0.25}
CACHE_DIR=${CACHE_DIR:-on_path_cache/n2_mod5_deg6_SEQ_C1_C6_routes32_seed42}

cd "$REPO_DIR"
mkdir -p results/slurm "$CACHE_DIR"

read -r -a COMPLEXITY_ARGS <<< "$COMPLEXITIES"

echo "Building cached OnPath target set"
echo "  repo: $REPO_DIR"
echo "  cache_dir: $CACHE_DIR"
echo "  n_variables: $N_VARIABLES"
echo "  mod: $MOD"
echo "  max_degree: $MAX_DEGREE"
echo "  complexities: ${COMPLEXITY_ARGS[*]}"
echo "  split_seed: $SPLIT_SEED"
echo "  max_on_path_size: $MAX_ON_PATH_SIZE"
echo "  on_path_num_routes: $ON_PATH_NUM_ROUTES"
echo "  max_route_truncation_rate: $MAX_ROUTE_TRUNCATION_RATE"

apptainer exec --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  "$CONTAINER" \
  python -u -m src.game_board.on_path \
    --complexities "${COMPLEXITY_ARGS[@]}" \
    --n-variables "$N_VARIABLES" \
    --mod "$MOD" \
    --max-degree "$MAX_DEGREE" \
    --cache-dir "$CACHE_DIR" \
    --split-seed "$SPLIT_SEED" \
    --max-on-path-size "$MAX_ON_PATH_SIZE" \
    --num-routes "$ON_PATH_NUM_ROUTES" \
    --max-route-truncation-rate "$MAX_ROUTE_TRUNCATION_RATE"

echo "OnPath cache complete: $CACHE_DIR"
