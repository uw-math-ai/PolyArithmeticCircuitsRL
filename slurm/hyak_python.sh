#!/bin/bash

load_hyak_python_module() {
  if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    # Hyak exposes Lmod through this shell initialization file in batch jobs.
    # Source it only when needed so local runs remain unaffected.
    source /etc/profile.d/modules.sh
  fi

  if ! command -v module >/dev/null 2>&1; then
    return 0
  fi

  if [[ -n "${HYAK_PYTHON_MODULE:-}" ]]; then
    module load "${HYAK_PYTHON_MODULE}"
    return 0
  fi

  for module_name in python/3.12 python/3.11 python/3.10 python; do
    if module load "${module_name}" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 0
}

find_base_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${PYTHON_BIN}" ]] || command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      command -v "${PYTHON_BIN}" 2>/dev/null || printf '%s\n' "${PYTHON_BIN}"
      return 0
    fi
    echo "PYTHON_BIN is set but not executable/found: ${PYTHON_BIN}" >&2
    return 1
  fi

  for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done

  echo "No usable Python interpreter found. Set HYAK_PYTHON_MODULE or PYTHON_BIN." >&2
  return 1
}

bootstrap_lgs_python() {
  load_hyak_python_module

  local base_python
  base_python="$(find_base_python)"

  if ! "${base_python}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
  then
    echo "Python 3.10+ is required; found $("${base_python}" --version 2>&1)" >&2
    return 1
  fi

  VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-hyak-$(basename "${base_python}")}"
  PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REPO_ROOT}/.cache/pip}"
  export PIP_CACHE_DIR
  mkdir -p "${PIP_CACHE_DIR}"

  if [[ "${BOOTSTRAP_VENV:-1}" != "1" ]]; then
    printf '%s\n' "${base_python}"
    return 0
  fi

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Creating Python venv: ${VENV_DIR}" >&2
    "${base_python}" -m venv "${VENV_DIR}" >&2
  fi

  local venv_python="${VENV_DIR}/bin/python"
  if ! "${venv_python}" -m pip --version >/dev/null 2>&1; then
    "${venv_python}" -m ensurepip --upgrade >/dev/null 2>&1 || true
  fi

  local check_imports='import torch'
  if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
    check_imports="${check_imports}"$'\n''import wandb'
  fi

  if ! "${venv_python}" - <<PY >/dev/null 2>&1
${check_imports}
PY
  then
    echo "Installing learn-guided-search dependencies into ${VENV_DIR}" >&2
    "${venv_python}" -m pip install --upgrade pip setuptools wheel >&2
    "${venv_python}" -m pip install --no-cache-dir -e "${REPO_ROOT}" >&2
    "${venv_python}" -m pip install --no-cache-dir torch >&2
  fi

  printf '%s\n' "${venv_python}"
}
