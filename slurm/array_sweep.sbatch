#!/bin/bash
#SBATCH --job-name=lgs_array_sweep
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/%x_%A_%a.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)}"
cd "${REPO_ROOT}"

MATRIX_FILE="${MATRIX_FILE:-slurm/sweep_matrix.tsv}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${TASK_ID}" ]]; then
  echo "SLURM_ARRAY_TASK_ID must be set for array_sweep.sbatch" >&2
  exit 1
fi
if [[ ! -f "${MATRIX_FILE}" ]]; then
  echo "Matrix file not found: ${MATRIX_FILE}" >&2
  exit 1
fi

MATRIX_LINES=()
while IFS= read -r line; do
  if [[ "${line}" =~ ^[[:space:]]*# ]]; then
    continue
  fi
  if [[ -z "${line//[[:space:]]/}" ]]; then
    continue
  fi
  MATRIX_LINES+=("${line}")
done < "${MATRIX_FILE}"
LINE_INDEX=$((TASK_ID - 1))
if (( LINE_INDEX < 0 || LINE_INDEX >= ${#MATRIX_LINES[@]} )); then
  echo "Array task ${TASK_ID} is out of range for ${#MATRIX_LINES[@]} matrix rows" >&2
  exit 1
fi

IFS=$'\t' read -r RUN_NAME MODE CHECKPOINT MAX_INSTANCES_PER_FAMILY BEAM_WIDTHS CANDIDATE_KS TIER2_MS LAMBDA_MODEL <<< "${MATRIX_LINES[LINE_INDEX]}"
OUTPUT_DIR="${OUTPUT_DIR:-results/server_runs/${RUN_NAME}}"
ROUNDS="${ROUNDS:-5}"
EPOCHS_PER_ROUND="${EPOCHS_PER_ROUND:-30}"
SEED="${SEED:-0}"
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

case "${MODE}" in
  heuristic)
    CMD=(
      "${PYTHON}" scripts/run_benchmark_sweep.py
      --max-instances-per-family "${MAX_INSTANCES_PER_FAMILY}"
      --beam-widths "${BEAM_WIDTHS}"
      --candidate-ks "${CANDIDATE_KS}"
      --tier2-ms "${TIER2_MS}"
      --output-dir "${OUTPUT_DIR}"
    )
    ;;
  guided)
    if [[ -z "${CHECKPOINT}" || "${CHECKPOINT}" == "-" ]]; then
      echo "Guided matrix row requires a checkpoint path" >&2
      exit 1
    fi
    if [[ ! -f "${CHECKPOINT}" ]]; then
      echo "Checkpoint not found: ${CHECKPOINT}" >&2
      exit 1
    fi
    CMD=(
      "${PYTHON}" scripts/run_benchmark_sweep.py
      --checkpoint "${CHECKPOINT}"
      --lambda-model "${LAMBDA_MODEL}"
      --max-instances-per-family "${MAX_INSTANCES_PER_FAMILY}"
      --beam-widths "${BEAM_WIDTHS}"
      --candidate-ks "${CANDIDATE_KS}"
      --tier2-ms "${TIER2_MS}"
      --output-dir "${OUTPUT_DIR}"
    )
    ;;
  train_eval)
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
    ;;
  *)
    echo "Unknown mode in matrix row: ${MODE}" >&2
    exit 1
    ;;
esac
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
  log_kv "mode" "${MODE}"
  log_kv "matrix_file" "${MATRIX_FILE}"
  log_kv "matrix_row" "${TASK_ID}"
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
  log_kv "lambda_model" "${LAMBDA_MODEL}"
  log_kv "checkpoint" "${CHECKPOINT}"
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
