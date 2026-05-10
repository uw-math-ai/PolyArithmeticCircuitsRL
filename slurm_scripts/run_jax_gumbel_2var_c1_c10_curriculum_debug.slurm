#!/bin/bash
#SBATCH --job-name=jax_gumbel_2var_c1_c10_debug
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-ppo-mcts-jax_2var_C1_C10_gumbel_curriculum_debug}"
RESULTS_DIR="${RESULTS_DIR:-results/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_${TIMESTAMP}.log}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-${LOG_DIR}/gpu_monitor_${TIMESTAMP}.log}"
CONTAINER_PATH="${CONTAINER_PATH:-${REPO_ROOT}/container_jax.sif}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-jax-blackwell-python3.13}"

# Task/model scale. Model architecture defaults come from src/config.py:
# hidden_dim=384, embedding_dim=256, num_gnn_layers=5,
# poly_embedding_dim=128, node_id_embedding_dim=32, pair_hidden_dim=384.
N_VARIABLES="${N_VARIABLES:-2}"
MAX_COMPLEXITY="${MAX_COMPLEXITY:-10}"
MAX_DEGREE="${MAX_DEGREE:-10}"
MAX_STEPS="${MAX_STEPS:-14}"

# Data/search/update settings.
MCTS_BATCH_SIZE="${MCTS_BATCH_SIZE:-1024}"
GUMBEL_NUM_SIMULATIONS="${GUMBEL_NUM_SIMULATIONS:-64}"
GUMBEL_MAX_NUM_CONSIDERED_ACTIONS="${GUMBEL_MAX_NUM_CONSIDERED_ACTIONS:-16}"
GUMBEL_SCALE="${GUMBEL_SCALE:-1.0}"
GUMBEL_C_VISIT="${GUMBEL_C_VISIT:-50.0}"
GUMBEL_C_SCALE="${GUMBEL_C_SCALE:-0.1}"
PPO_LR="${PPO_LR:-1e-4}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
ENT_COEF="${ENT_COEF:-0.01}"

# Eval settings. Eval after each phase by default; increase EVAL_EVERY_PHASES
# to reduce overhead. Final eval always runs when RUN_FINAL_EVAL=1.
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_EVERY_PHASES="${EVAL_EVERY_PHASES:-1}"
EVAL_COMPLEXITIES="${EVAL_COMPLEXITIES:-1 2 3 4 5 6 7 8 9 10}"
EVAL_NUM_TRIALS="${EVAL_NUM_TRIALS:-128}"
FINAL_EVAL_NUM_TRIALS="${FINAL_EVAL_NUM_TRIALS:-512}"
EVAL_MCTS_SIMULATIONS="${EVAL_MCTS_SIMULATIONS:-64}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_SEED="${EVAL_SEED:-20260510}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"
EVAL_WANDB="${EVAL_WANDB:-0}"

# W&B is useful for train metrics. Eval W&B is off by default to avoid many
# separate eval runs; set EVAL_WANDB=1 if you want them too.
ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-PolyArithmeticCircuitsRL}"
WANDB_ENTITY="${WANDB_ENTITY:-zengrf-university-of-washington}"
WANDB_MODE_DEFAULT="${WANDB_MODE_DEFAULT:-online}"

GPU_MONITOR_SECONDS="${GPU_MONITOR_SECONDS:-300}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

# Manual curriculum. Each phase resumes from the previous phase checkpoint.
# The trainer saves checkpoints every 50 iterations and at phase end.
PHASE_ITERATIONS=(30 500 500 500 500 500 500 500 500 500)
PHASE_COMPLEXITIES=(
  "1"
  "1 2"
  "2 3"
  "3 4"
  "4 5"
  "5 6"
  "6 7"
  "7 8"
  "8 9"
  "9 10"
)

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" results/slurm
exec > >(tee -a "${LOG_FILE}") 2>&1

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

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  echo "Python 3.10+ is required for the JAX trainer" >&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module load cuda >/dev/null 2>&1 || true
fi

export PYTHONUNBUFFERED=1
unset JAX_PLATFORM_NAME
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
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

start_gpu_monitor() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  (
    while true; do
      echo "=== $(date '+%F %T') ==="
      nvidia-smi
      sleep "${GPU_MONITOR_SECONDS}"
    done
  ) >> "${GPU_MONITOR_LOG}" 2>&1 &
  GPU_MONITOR_PID=$!
  echo "GPU monitor PID: ${GPU_MONITOR_PID}"
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}

trap stop_gpu_monitor EXIT

write_run_metadata() {
  {
    echo "RUN_TAG=${RUN_TAG}"
    echo "RESULTS_DIR=${RESULTS_DIR}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "N_VARIABLES=${N_VARIABLES}"
    echo "MAX_COMPLEXITY=${MAX_COMPLEXITY}"
    echo "MAX_DEGREE=${MAX_DEGREE}"
    echo "MAX_STEPS=${MAX_STEPS}"
    echo "MCTS_BATCH_SIZE=${MCTS_BATCH_SIZE}"
    echo "GUMBEL_NUM_SIMULATIONS=${GUMBEL_NUM_SIMULATIONS}"
    echo "GUMBEL_MAX_NUM_CONSIDERED_ACTIONS=${GUMBEL_MAX_NUM_CONSIDERED_ACTIONS}"
    echo "GUMBEL_SCALE=${GUMBEL_SCALE}"
    echo "GUMBEL_C_VISIT=${GUMBEL_C_VISIT}"
    echo "GUMBEL_C_SCALE=${GUMBEL_C_SCALE}"
    echo "PPO_LR=${PPO_LR}"
    echo "PPO_EPOCHS=${PPO_EPOCHS}"
    echo "ENT_COEF=${ENT_COEF}"
    echo "RUN_EVAL=${RUN_EVAL}"
    echo "EVAL_EVERY_PHASES=${EVAL_EVERY_PHASES}"
    echo "EVAL_COMPLEXITIES=${EVAL_COMPLEXITIES}"
    echo "EVAL_NUM_TRIALS=${EVAL_NUM_TRIALS}"
    echo "FINAL_EVAL_NUM_TRIALS=${FINAL_EVAL_NUM_TRIALS}"
    echo "EVAL_MCTS_SIMULATIONS=${EVAL_MCTS_SIMULATIONS}"
    echo "EVAL_TEMPERATURE=${EVAL_TEMPERATURE}"
  } > "${RESULTS_DIR}/run_config.env"

  echo "phase,iterations,complexities,cumulative_iterations,episodes,max_env_steps,root_simulations" \
    > "${RESULTS_DIR}/curriculum_schedule.csv"
  total_iters=0
  for idx in "${!PHASE_ITERATIONS[@]}"; do
    phase_num=$((idx + 1))
    phase_iters="${PHASE_ITERATIONS[idx]}"
    phase_complexities="${PHASE_COMPLEXITIES[idx]}"
    total_iters=$((total_iters + phase_iters))
    episodes=$((phase_iters * MCTS_BATCH_SIZE))
    max_env_steps=$((episodes * MAX_STEPS))
    root_sims=$((max_env_steps * GUMBEL_NUM_SIMULATIONS))
    printf '%d,%d,"%s",%d,%d,%d,%d\n' \
      "${phase_num}" "${phase_iters}" "${phase_complexities}" \
      "${total_iters}" "${episodes}" "${max_env_steps}" "${root_sims}" \
      >> "${RESULTS_DIR}/curriculum_schedule.csv"
  done
}

print_system_snapshot() {
  echo "=== Run Started: $(date '+%F %T') ==="
  echo "Host: $(hostname)"
  echo "Repo root: ${REPO_ROOT}"
  echo "Results dir: ${RESULTS_DIR}"
  echo "Log file: ${LOG_FILE}"
  echo "GPU monitor log: ${GPU_MONITOR_LOG}"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
  echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "JAX_PLATFORMS=${JAX_PLATFORMS}"
  echo "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"

  echo
  echo "=== Git Snapshot ==="
  git rev-parse --show-toplevel || true
  git rev-parse HEAD || true
  git branch --show-current || true
  git status --short || true
  git diff --stat -- src/algorithms/jax_env.py src/algorithms/jax_net.py src/config.py tests/test_jax.py slurm_scripts || true

  echo
  echo "=== Python/JAX Snapshot ==="
  printf 'Python command:'
  printf ' %q' "${PYTHON_CMD[@]}"
  printf '\n'
  "${PYTHON_CMD[@]}" - <<'PY'
import platform
import jax
from src.config import Config
cfg = Config()
print("Python:", platform.python_version())
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())
print("Config model dims:", {
    "hidden_dim": cfg.hidden_dim,
    "embedding_dim": cfg.embedding_dim,
    "num_gnn_layers": cfg.num_gnn_layers,
    "poly_embedding_dim": getattr(cfg, "poly_embedding_dim", None),
    "node_id_embedding_dim": getattr(cfg, "node_id_embedding_dim", None),
    "op_embedding_dim": getattr(cfg, "op_embedding_dim", None),
    "pair_hidden_dim": getattr(cfg, "pair_hidden_dim", None),
})
print("Config defaults:", {
    "steps_per_update": cfg.steps_per_update,
    "log_interval": cfg.log_interval,
    "checkpoint_interval": "50 iterations in PPOMCTSJAXTrainer.train",
})
PY

  echo
  echo "=== GPU Snapshot ==="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi not found"
  fi

  echo
  echo "=== Curriculum Schedule ==="
  column -s, -t "${RESULTS_DIR}/curriculum_schedule.csv" || cat "${RESULTS_DIR}/curriculum_schedule.csv"
}

run_eval_for_phase() {
  local phase_num="$1"
  local phase_iters="$2"
  local phase_results_dir="$3"
  local out_prefix="$4"
  local num_trials="$5"
  shift 5
  local eval_complexities=("$@")

  EVAL_CMD=(
    "${PYTHON_CMD[@]}"
    -u scripts/eval_ppo_mcts_jax.py
    --checkpoint-dir "${phase_results_dir}"
    --iterations "${phase_iters}"
    --complexities "${eval_complexities[@]}"
    --num-trials "${num_trials}"
    --mcts-simulations "${EVAL_MCTS_SIMULATIONS}"
    --temperature "${EVAL_TEMPERATURE}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --seed "${EVAL_SEED}"
    --out-prefix "${out_prefix}"
  )

  if [[ "${EVAL_WANDB}" == "1" ]]; then
    EVAL_CMD+=(
      --wandb
      --wandb-project "${WANDB_PROJECT}"
      --wandb-entity "${WANDB_ENTITY}"
      --wandb-run-name "${RUN_TAG}_eval_phase$(printf '%02d' "${phase_num}")"
    )
  fi

  echo
  echo "=== Eval after phase $(printf '%02d' "${phase_num}") ==="
  echo "Trials per complexity: ${num_trials}"
  echo "Complexities: ${eval_complexities[*]}"
  printf 'Eval command:\n'
  printf '  %q' "${EVAL_CMD[@]}"
  printf '\n'
  "${EVAL_CMD[@]}"
}

wait_for_pid_exit "${WAIT_FOR_PID}"
wait_for_gpu_idle
write_run_metadata
print_system_snapshot
start_gpu_monitor

resume_checkpoint=""
final_phase_results_dir=""
final_phase_iters=""
read -r -a ALL_EVAL_COMPLEXITY_ARGS <<< "${EVAL_COMPLEXITIES}"

for idx in "${!PHASE_ITERATIONS[@]}"; do
  phase_num=$((idx + 1))
  phase_iters="${PHASE_ITERATIONS[idx]}"
  phase_complexities="${PHASE_COMPLEXITIES[idx]}"
  read -r -a PHASE_COMPLEXITY_ARGS <<< "${phase_complexities}"

  phase_label="$(printf '%s' "${phase_complexities}" | tr ' ' '_')"
  phase_results_dir="${RESULTS_DIR}/phase$(printf '%02d' "${phase_num}")_C${phase_label}"
  phase_run_tag="${RUN_TAG}_phase$(printf '%02d' "${phase_num}")"
  mkdir -p "${phase_results_dir}"

  echo
  echo "=== Phase $(printf '%02d' "${phase_num}") / ${#PHASE_ITERATIONS[@]} ==="
  echo "Started: $(date '+%F %T')"
  echo "Iterations this phase: ${phase_iters}"
  echo "Complexities this phase: ${phase_complexities}"
  echo "Episodes this phase: $((phase_iters * MCTS_BATCH_SIZE))"
  echo "Max env steps this phase: $((phase_iters * MCTS_BATCH_SIZE * MAX_STEPS))"
  echo "Root simulations upper bound: $((phase_iters * MCTS_BATCH_SIZE * MAX_STEPS * GUMBEL_NUM_SIMULATIONS))"
  if [[ -n "${resume_checkpoint}" ]]; then
    echo "Resume checkpoint: ${resume_checkpoint}"
  else
    echo "Resume checkpoint: <fresh start>"
  fi

  TRAIN_CMD=(
    "${PYTHON_CMD[@]}"
    -u -m src.main
    --algorithm ppo-mcts-jax
    --search gumbel
    --n-variables "${N_VARIABLES}"
    --fixed-complexities "${PHASE_COMPLEXITY_ARGS[@]}"
    --max-complexity "${MAX_COMPLEXITY}"
    --max-degree "${MAX_DEGREE}"
    --max-steps "${MAX_STEPS}"
    --iterations "${phase_iters}"
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
    --results-dir "${phase_results_dir}"
    --wandb-run-name "${phase_run_tag}"
  )

  if [[ -n "${resume_checkpoint}" ]]; then
    TRAIN_CMD+=(--checkpoint "${resume_checkpoint}")
  fi

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

  resume_checkpoint="${phase_results_dir}/checkpoint_$(printf '%05d' "${phase_iters}").pkl"
  if [[ ! -f "${resume_checkpoint}" ]]; then
    echo "Expected checkpoint missing after phase ${phase_num}: ${resume_checkpoint}" >&2
    exit 1
  fi

  printf '%s\n' "${resume_checkpoint}" > "${RESULTS_DIR}/latest_checkpoint.txt"
  final_phase_results_dir="${phase_results_dir}"
  final_phase_iters="${phase_iters}"

  echo "Phase $(printf '%02d' "${phase_num}") finished: $(date '+%F %T')"
  echo "Latest checkpoint: ${resume_checkpoint}"
  du -sh "${phase_results_dir}" || true
  ls -lh "${phase_results_dir}" || true

  if [[ "${RUN_EVAL}" == "1" ]] && (( phase_num % EVAL_EVERY_PHASES == 0 )); then
    run_eval_for_phase \
      "${phase_num}" \
      "${phase_iters}" \
      "${phase_results_dir}" \
      "${phase_results_dir}/eval_phase$(printf '%02d' "${phase_num}")_final" \
      "${EVAL_NUM_TRIALS}" \
      "${ALL_EVAL_COMPLEXITY_ARGS[@]}"
  fi
done

if [[ "${RUN_FINAL_EVAL}" == "1" && -n "${final_phase_results_dir}" ]]; then
  run_eval_for_phase \
    "${#PHASE_ITERATIONS[@]}" \
    "${final_phase_iters}" \
    "${final_phase_results_dir}" \
    "${RESULTS_DIR}/final_eval_C1_C10" \
    "${FINAL_EVAL_NUM_TRIALS}" \
    "${ALL_EVAL_COMPLEXITY_ARGS[@]}"
fi

echo
echo "Curriculum run finished successfully: $(date '+%F %T')"
echo "Latest checkpoint: ${resume_checkpoint}"
echo "Schedule: ${RESULTS_DIR}/curriculum_schedule.csv"
echo "Run config: ${RESULTS_DIR}/run_config.env"
echo "Main log: ${LOG_FILE}"
echo "GPU monitor log: ${GPU_MONITOR_LOG}"
