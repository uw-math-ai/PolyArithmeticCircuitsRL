#!/usr/bin/env bash
# Bootstrap a fresh Linux/macOS cloud machine for PPO+MCTS training.
#
# Idempotent: re-running is safe. Creates a local .venv, installs the package
# with the `train` + `dev` extras (so PyTorch is pulled in), and runs pytest
# as a smoke test. SymPy is used as the factorization backend by default; pass
# --with-sage to additionally bootstrap the optional Sage CAS environment via
# scripts/setup_cas_env.sh (slower, but recommended for serious runs).
#
# Usage:
#   scripts/setup_cloud.sh                # CPU PyTorch, SymPy backend
#   scripts/setup_cloud.sh --with-sage    # also build .cas_env/ via micromamba
#   scripts/setup_cloud.sh --skip-tests   # skip pytest smoke run
#   scripts/setup_cloud.sh --gpu          # install CUDA PyTorch wheel (cu121)
#
# Requires: Python >=3.10 on PATH. On Ubuntu 22.04/24.04 we will install
# python3-venv via apt-get if it is missing (sudo).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WITH_SAGE=0
SKIP_TESTS=0
WANT_GPU=0
for arg in "$@"; do
  case "$arg" in
    --with-sage)  WITH_SAGE=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --gpu)        WANT_GPU=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

echo "==> Repo root: $ROOT_DIR"

# 1. Python check + python3-venv on Debian/Ubuntu if needed.
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required. Install Python >=3.10 first." >&2
  exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "==> python3 = $(command -v python3) ($PY_VER)"

if ! python3 -m venv --help >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "==> Installing python3-venv via apt-get (sudo)"
    sudo apt-get update -y
    sudo apt-get install -y "python${PY_VER}-venv" python3-pip
  else
    echo "python3 -m venv is unavailable and apt-get is not present." >&2
    echo "Install the venv module for your distro and re-run." >&2
    exit 1
  fi
fi

# 2. Create / refresh the virtualenv.
if [[ ! -d .venv ]]; then
  echo "==> Creating .venv"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# 3. PyTorch wheel: CPU by default, CUDA 12.1 with --gpu.
echo "==> Installing PyTorch ($([[ $WANT_GPU -eq 1 ]] && echo cu121 || echo cpu))"
if [[ $WANT_GPU -eq 1 ]]; then
  pip install --index-url https://download.pytorch.org/whl/cu121 "torch>=2.4"
else
  pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
fi

# 4. Package + dev/train extras (skip torch, already pinned above).
echo "==> Installing decomp-rl in editable mode with [train,dev]"
pip install -e '.[train,dev]'

# 5. Optional Sage CAS bootstrap.
if [[ $WITH_SAGE -eq 1 ]]; then
  echo "==> Bootstrapping Sage CAS environment (slow, several minutes)"
  bash scripts/setup_cas_env.sh
fi

# 6. Smoke test.
if [[ $SKIP_TESTS -eq 0 ]]; then
  echo "==> Running pytest smoke test"
  pytest -q
fi

echo ""
echo "==> Setup complete. To use the environment in this shell:"
echo "        source .venv/bin/activate"
echo ""
echo "==> Try a short PPO+MCTS run:"
echo "        python scripts/run_ppo_finetune.py --use-mcts --iterations 5 --seed 0"
