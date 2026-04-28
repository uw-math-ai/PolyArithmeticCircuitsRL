#!/bin/bash
#SBATCH --job-name=jax_gumbel_4var_c10
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-ppo-mcts-jax_gumbel_4var_C10_fromscratch}"
RESULTS_DIR="${RESULTS_DIR:-results/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-results/nohup}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}_${TIMESTAMP}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
CONTAINER_PATH="${CONTAINER_PATH:-${REPO_ROOT}/container_jax.sif}"

ITERATIONS="${ITERATIONS:-2000}"
N_VARIABLES="${N_VARIABLES:-4}"
MAX_COMPLEXITY="${MAX_COMPLEXITY:-10}"
MAX_DEGREE="${MAX_DEGREE:-10}"
MAX_STEPS="${MAX_STEPS:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
MCTS_BATCH_SIZE="${MCTS_BATCH_SIZE:-64}"
GUMBEL_NUM_SIMULATIONS="${GUMBEL_NUM_SIMULATIONS:-32}"
GUMBEL_MAX_NUM_CONSIDERED_ACTIONS="${GUMBEL_MAX_NUM_CONSIDERED_ACTIONS:-16}"
GUMBEL_SCALE="${GUMBEL_SCALE:-1.0}"
GUMBEL_C_VISIT="${GUMBEL_C_VISIT:-50.0}"
GUMBEL_C_SCALE="${GUMBEL_C_SCALE:-0.1}"
PPO_LR="${PPO_LR:-1e-4}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
ENT_COEF="${ENT_COEF:-0.01}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-PolyArithmeticCircuitsRL}"
WANDB_ENTITY="${WANDB_ENTITY:-zengrf-university-of-washington}"
WANDB_MODE_DEFAULT="${WANDB_MODE_DEFAULT:-online}"

PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "${candidate}")"
    break
  fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No usable python3 interpreter found" >&2
  exit 1
fi

PYTHON_VERSION="$(${PYTHON_BIN} - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  echo "Python 3.10+ is required for the JAX trainer; found ${PYTHON_VERSION} at ${PYTHON_BIN}" >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-jax-blackwell-python3.13}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" results/slurm

if [[ -z "${SLURM_JOB_ID:-}" && "${RUN_FOREGROUND:-0}" != "1" ]]; then
  echo "Launching ${RUN_TAG} with nohup"
  echo "Log file: ${LOG_FILE}"
  nohup env \
    RUN_FOREGROUND=1 \
    RUN_TAG="${RUN_TAG}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    LOG_DIR="${LOG_DIR}" \
    LOG_FILE="${LOG_FILE}" \
    PID_FILE="${PID_FILE}" \
    VENV_DIR="${VENV_DIR}" \
    CONTAINER_PATH="${CONTAINER_PATH}" \
    ITERATIONS="${ITERATIONS}" \
    N_VARIABLES="${N_VARIABLES}" \
    MAX_COMPLEXITY="${MAX_COMPLEXITY}" \
    MAX_DEGREE="${MAX_DEGREE}" \
    MAX_STEPS="${MAX_STEPS}" \
    HIDDEN_DIM="${HIDDEN_DIM}" \
    MCTS_BATCH_SIZE="${MCTS_BATCH_SIZE}" \
    GUMBEL_NUM_SIMULATIONS="${GUMBEL_NUM_SIMULATIONS}" \
    GUMBEL_MAX_NUM_CONSIDERED_ACTIONS="${GUMBEL_MAX_NUM_CONSIDERED_ACTIONS}" \
    GUMBEL_SCALE="${GUMBEL_SCALE}" \
    GUMBEL_C_VISIT="${GUMBEL_C_VISIT}" \
    GUMBEL_C_SCALE="${GUMBEL_C_SCALE}" \
    PPO_LR="${PPO_LR}" \
    PPO_EPOCHS="${PPO_EPOCHS}" \
    ENT_COEF="${ENT_COEF}" \
    WAIT_FOR_PID="${WAIT_FOR_PID}" \
    WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE}" \
    GPU_POLL_SECONDS="${GPU_POLL_SECONDS}" \
    ENABLE_WANDB="${ENABLE_WANDB}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_ENTITY="${WANDB_ENTITY}" \
    WANDB_MODE_DEFAULT="${WANDB_MODE_DEFAULT}" \
    bash "$0" >"${LOG_FILE}" 2>&1 &
  child_pid=$!
  echo "${child_pid}" > "${PID_FILE}"
  echo "PID: ${child_pid}"
  exit 0
fi

if command -v module >/dev/null 2>&1; then
  module load cuda >/dev/null 2>&1 || true
fi

export PYTHONUNBUFFERED=1
unset JAX_PLATFORM_NAME
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export WANDB_MODE="${WANDB_MODE:-${WANDB_MODE_DEFAULT}}"

bootstrap_venv() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi

  if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
    "${VENV_DIR}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
  fi

  if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "${VENV_DIR}/bin/python" /tmp/get-pip.py
  fi

  if ! "${VENV_DIR}/bin/python" - <<'PY' >/dev/null 2>&1
import flax
import jax
import mctx
import optax
import tqdm
import wandb
PY
  then
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
    "${VENV_DIR}/bin/python" -m pip install --no-cache-dir -r "${REPO_ROOT}/requirements-jax.txt"
  fi
}

PYTHON_CMD=()
CONTAINER_EXEC=()
if command -v apptainer >/dev/null 2>&1 && [[ -f "${CONTAINER_PATH}" ]]; then
  if [[ -d /gscratch ]]; then
    CONTAINER_EXEC+=(--bind /gscratch:/gscratch)
  fi
  PYTHON_CMD=(apptainer exec --nv "${CONTAINER_EXEC[@]}" "${CONTAINER_PATH}" python)
elif command -v singularity >/dev/null 2>&1 && [[ -f "${CONTAINER_PATH}" ]]; then
  if [[ -d /gscratch ]]; then
    CONTAINER_EXEC+=(--bind /gscratch:/gscratch)
  fi
  PYTHON_CMD=(singularity exec --nv "${CONTAINER_EXEC[@]}" "${CONTAINER_PATH}" python)
else
  bootstrap_venv
  PYTHON_CMD=("${VENV_DIR}/bin/python")
fi

wait_for_pid_exit() {
  local pid="$1"
  if [[ -z "${pid}" ]]; then
    return 0
  fi

  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    return 0
  fi

  echo "Waiting for PID ${pid} to exit before starting training"
  while kill -0 "${pid}" >/dev/null 2>&1; do
    printf '  [%s] blocked on PID %s\n' "$(date '+%F %T')" "${pid}"
    sleep "${GPU_POLL_SECONDS}"
  done
}

wait_for_gpu_idle() {
  if [[ "${WAIT_FOR_GPU_IDLE}" != "1" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')
  if [[ ${#gpu_pids[@]} -eq 0 ]]; then
    return 0
  fi

  echo "Waiting for GPU to become idle before starting training"
  while true; do
    mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')
    if [[ ${#gpu_pids[@]} -eq 0 ]]; then
      break
    fi
    printf '  [%(%F %T)T] active GPU PIDs:' -1
    for pid in "${gpu_pids[@]}"; do
      printf ' %s' "${pid}"
    done
    printf '\n'
    sleep "${GPU_POLL_SECONDS}"
  done
}

wait_for_pid_exit "${WAIT_FOR_PID}"
wait_for_gpu_idle

echo "Repo root: ${REPO_ROOT}"
echo "Results dir: ${RESULTS_DIR}"
echo "Log file: ${LOG_FILE}"
echo "JAX Gumbel settings: n_vars=${N_VARIABLES} max_complexity=${MAX_COMPLEXITY} batch=${MCTS_BATCH_SIZE} sims=${GUMBEL_NUM_SIMULATIONS} lr=${PPO_LR} epochs=${PPO_EPOCHS} ent=${ENT_COEF} max_degree=${MAX_DEGREE} max_steps=${MAX_STEPS} iterations=${ITERATIONS} hidden_dim=${HIDDEN_DIM}"
if [[ "${ENABLE_WANDB}" == "1" ]]; then
  echo "W&B target: ${WANDB_ENTITY}/${WANDB_PROJECT} (mode=${WANDB_MODE})"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,utilization.gpu --format=csv,noheader
fi

"${PYTHON_CMD[@]}" - <<'PY'
import jax
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())
PY

TRAIN_CMD=(
  "${PYTHON_CMD[@]}"
  -u -m src.main
  --algorithm ppo-mcts-jax
  --search gumbel
  --n-variables "${N_VARIABLES}"
  --max-complexity "${MAX_COMPLEXITY}"
  --max-degree "${MAX_DEGREE}"
  --max-steps "${MAX_STEPS}"
  --hidden-dim "${HIDDEN_DIM}"
  --iterations "${ITERATIONS}"
  --mcts-batch-size "${MCTS_BATCH_SIZE}"
  --gumbel-num-simulations "${GUMBEL_NUM_SIMULATIONS}"
  --gumbel-max-num-considered-actions "${GUMBEL_MAX_NUM_CONSIDERED_ACTIONS}"
  --gumbel-scale "${GUMBEL_SCALE}"
  --gumbel-c-visit "${GUMBEL_C_VISIT}"
  --gumbel-c-scale "${GUMBEL_C_SCALE}"
  --ppo-lr "${PPO_LR}"
  --ppo-epochs "${PPO_EPOCHS}"
  --ent-coef "${ENT_COEF}"
  --factor-subgoal-reward 1.0
  --factor-library-bonus 0.5
  --completion-bonus 3.0
  --results-dir "${RESULTS_DIR}"
  --wandb-run-name "${RUN_TAG}"
)

if [[ "${ENABLE_WANDB}" == "1" ]]; then
  TRAIN_CMD+=(
    --wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-entity "${WANDB_ENTITY}"
  )
fi

printf 'Training command:\n'
printf '  %q' "${TRAIN_CMD[@]}"
printf '\n'

"${TRAIN_CMD[@]}"