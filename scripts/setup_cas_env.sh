#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/.tools/micromamba"
MAMBA_BIN="$TOOLS_DIR/bin/micromamba"
CAS_ENV="$ROOT_DIR/.cas_env"

mkdir -p "$TOOLS_DIR"

if [[ ! -x "$MAMBA_BIN" ]]; then
  curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/micromamba.tar.bz2
  tar -xjf /tmp/micromamba.tar.bz2 -C /tmp
  mkdir -p "$TOOLS_DIR/bin"
  cp /tmp/bin/micromamba "$MAMBA_BIN"
fi

"$MAMBA_BIN" create -y -p "$CAS_ENV" -c conda-forge python=3.11 sagelib

echo "CAS environment ready at $CAS_ENV"
