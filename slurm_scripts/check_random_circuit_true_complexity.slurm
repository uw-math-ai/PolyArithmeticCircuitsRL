#!/bin/bash
#SBATCH --job-name=check_true_c
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

cd /gscratch/stf/kj66/PolyArithmeticCircuitsRL_kj66-hyak
mkdir -p results/slurm

LABELS=${LABELS:-"5 6 7 8"}
SAMPLES_PER_LABEL=${SAMPLES_PER_LABEL:-1000}
MAX_TRUE_COMPLEXITY=${MAX_TRUE_COMPLEXITY:-8}
N_VARIABLES=${N_VARIABLES:-2}
MOD=${MOD:-5}
MAX_DEGREE=${MAX_DEGREE:-6}
SEED=${SEED:-42}
ROUTE_CAP=${ROUTE_CAP:-512}
MAX_POLYS=${MAX_POLYS:-300000}
OUT_PREFIX=${OUT_PREFIX:-results/random_circuit_true_complexity}

echo "Random circuit true-complexity check"
echo "  labels: ${LABELS}"
echo "  samples_per_label: ${SAMPLES_PER_LABEL}"
echo "  max_true_complexity: ${MAX_TRUE_COMPLEXITY}"
echo "  n_variables: ${N_VARIABLES}"
echo "  mod: ${MOD}"
echo "  max_degree: ${MAX_DEGREE}"
echo "  route_cap: ${ROUTE_CAP}"
echo "  max_polys: ${MAX_POLYS}"
echo "  out_prefix: ${OUT_PREFIX}"

# shellcheck disable=SC2086
apptainer exec --bind /gscratch:/gscratch \
  --env PYTHONUNBUFFERED=1 \
  container_jax.sif \
  python -u check_random_circuit_true_complexity.py \
    --labels ${LABELS} \
    --samples-per-label "${SAMPLES_PER_LABEL}" \
    --max-true-complexity "${MAX_TRUE_COMPLEXITY}" \
    --n-variables "${N_VARIABLES}" \
    --mod "${MOD}" \
    --max-degree "${MAX_DEGREE}" \
    --seed "${SEED}" \
    --route-cap "${ROUTE_CAP}" \
    --max-polys "${MAX_POLYS}" \
    --out-prefix "${OUT_PREFIX}"
