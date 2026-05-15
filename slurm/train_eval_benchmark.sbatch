#!/bin/bash
#SBATCH --job-name=lgs_train_eval
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mmfs1/gscratch/stf/kj66/learn-guided-search}"
SCRIPT_DIR="${REPO_ROOT}/slurm"
cd "${REPO_ROOT}"

RUN_NAME="${RUN_NAME:-train_eval_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/server_runs/${RUN_NAME}}"
ROUNDS="${ROUNDS:-5}"
MAX_INSTANCES_PER_FAMILY="${MAX_INSTANCES_PER_FAMILY:-25}"
EPOCHS_PER_ROUND="${EPOCHS_PER_ROUND:-30}"
BEAM_WIDTHS="${BEAM_WIDTHS:-1,2,4,8}"
CANDIDATE_KS="${CANDIDATE_KS:-4,8,16,32}"
TIER2_MS="${TIER2_MS:-64,128}"
SEED="${SEED:-0}"
LAMBDA_MODEL="${LAMBDA_MODEL:-}"
CHECKPOINT="${CHECKPOINT:-}"
RANDOM_LABELS="${RANDOM_LABELS:-0}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-PolyArithmeticCircuitsRL}"
WANDB_ENTITY="${WANDB_ENTITY:-zengrf-university-of-washington}"
WANDB_MODE_DEFAULT="${WANDB_MODE_DEFAULT:-online}"

source "${SCRIPT_DIR}/hyak_python.sh"

mkdir -p results/slurm results/server_runs "${OUTPUT_DIR}"
METADATA_FILE="${OUTPUT_DIR}/run_metadata.txt"
START_TIME="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
START_SECONDS="$(date +%s)"
: > "${METADATA_FILE}"

log_kv() {
  printf '%s=%s\n' "$1" "$2" | tee -a "${METADATA_FILE}"
}

PYTHON="$(bootstrap_lgs_python)"
export WANDB_MODE="${WANDB_MODE:-${WANDB_MODE_DEFAULT}}"

CMD=(
  "${PYTHON}" scripts/train_and_eval_benchmark.py
  --rounds "${ROUNDS}"
  --max-instances-per-family "${MAX_INSTANCES_PER_FAMILY}"
  --epochs-per-round "${EPOCHS_PER_ROUND}"
  --beam-widths "${BEAM_WIDTHS}"
  --candidate-ks "${CANDIDATE_KS}"
  --tier2-ms "${TIER2_MS}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
)
if [[ "${RANDOM_LABELS}" == "1" ]]; then
  CMD+=(--random-labels)
fi
if [[ "${ENABLE_WANDB}" == "1" ]]; then
  CMD+=(
    --wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-run-name "${RUN_NAME}"
  )
  if [[ -n "${WANDB_ENTITY}" ]]; then
    CMD+=(--wandb-entity "${WANDB_ENTITY}")
  fi
else
  CMD+=(--no-wandb)
fi

log_environment() {
  log_kv "run_name" "${RUN_NAME}"
  log_kv "slurm_job_id" "${SLURM_JOB_ID:-none}"
  log_kv "slurm_array_task_id" "${SLURM_ARRAY_TASK_ID:-none}"
  log_kv "hostname" "$(hostname)"
  log_kv "start_time" "${START_TIME}"
  log_kv "working_directory" "${PWD}"
  log_kv "git_commit" "$(git rev-parse HEAD 2>/dev/null || printf 'not-a-git-repo')"
  log_kv "git_status_summary" "$(git status --short 2>/dev/null | tr '\n' ';' || printf 'unavailable')"
  log_kv "python_path" "${PYTHON}"
  log_kv "python_version" "$("${PYTHON}" --version 2>&1)"
  log_kv "hyak_python_module" "${HYAK_PYTHON_MODULE:-auto}"
  log_kv "venv_dir" "${VENV_DIR:-not-used}"
  log_kv "bootstrap_venv" "${BOOTSTRAP_VENV:-1}"
  log_kv "package_versions" "$("${PYTHON}" - <<'PY'
import importlib.metadata as md
for name in ("torch", "numpy", "pytest", "wandb"):
    try:
        print(f"{name}:{md.version(name)}", end=" ")
    except md.PackageNotFoundError:
        print(f"{name}:not-installed", end=" ")
PY
)"
  log_kv "cpu_count" "$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf 'unknown')"
  log_kv "memory_info" "$(free -h 2>/dev/null | tr '\n' ';' || sysctl -n hw.memsize 2>/dev/null || printf 'unknown')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    log_kv "gpu_info" "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  else
    log_kv "gpu_info" "nvidia-smi-not-found"
  fi
  log_kv "rounds" "${ROUNDS}"
  log_kv "max_instances_per_family" "${MAX_INSTANCES_PER_FAMILY}"
  log_kv "epochs_per_round" "${EPOCHS_PER_ROUND}"
  log_kv "beam_widths" "${BEAM_WIDTHS}"
  log_kv "candidate_ks" "${CANDIDATE_KS}"
  log_kv "tier2_ms" "${TIER2_MS}"
  log_kv "lambda_model" "${LAMBDA_MODEL:-not-used}"
  log_kv "checkpoint" "${CHECKPOINT:-not-used}"
  log_kv "random_labels" "${RANDOM_LABELS}"
  log_kv "enable_wandb" "${ENABLE_WANDB}"
  log_kv "wandb_project" "${WANDB_PROJECT}"
  log_kv "wandb_entity" "${WANDB_ENTITY}"
  log_kv "wandb_mode" "${WANDB_MODE}"
  log_kv "output_dir" "${OUTPUT_DIR}"
  printf 'Command:\n' | tee -a "${METADATA_FILE}"
  printf '  %q' "${CMD[@]}" | tee -a "${METADATA_FILE}"
  printf '\n' | tee -a "${METADATA_FILE}"
}

finalize_run() {
  local exit_code="$1"
  local end_seconds
  end_seconds="$(date +%s)"
  log_kv "exit_code" "${exit_code}"
  log_kv "end_time" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  log_kv "elapsed_seconds" "$((end_seconds - START_SECONDS))"
  printf 'generated_files:\n' | tee -a "${METADATA_FILE}"
  find "${OUTPUT_DIR}" -maxdepth 2 -type f | sort | tee -a "${METADATA_FILE}" || true
  for path in sweep_rows.jsonl sweep_summary.json sweep_failures.json ranker.pt bootstrap_metrics.json; do
    if [[ -f "${OUTPUT_DIR}/${path}" ]]; then
      log_kv "exists_${path}" "yes"
    else
      log_kv "exists_${path}" "no"
    fi
  done
  if [[ -f "${OUTPUT_DIR}/sweep_rows.jsonl" ]]; then
    log_kv "sweep_rows_count" "$(wc -l < "${OUTPUT_DIR}/sweep_rows.jsonl" | tr -d ' ')"
  fi
  log_kv "summary_json" "${OUTPUT_DIR}/sweep_summary.json"
  log_kv "failure_summary_json" "${OUTPUT_DIR}/sweep_failures.json"
}

echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON} ($("${PYTHON}" --version 2>&1))"
echo "Output dir: ${OUTPUT_DIR}"
if [[ "${ENABLE_WANDB}" == "1" ]]; then
  echo "W&B target: ${WANDB_ENTITY:+${WANDB_ENTITY}/}${WANDB_PROJECT} (mode=${WANDB_MODE})"
fi
log_environment

set +e
"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
exit_code="${PIPESTATUS[0]}"
set -e
finalize_run "${exit_code}"
exit "${exit_code}"
