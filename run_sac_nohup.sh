#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_sac_nohup.sh [log_file] [iterations] [extra args...]
# Example:
#   ./run_sac_nohup.sh tests/sac_v5.log 10000 --n-variables 2 --max-complexity 6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_FILE="${1:-tests/sac_training.log}"
ITERATIONS="${2:-10000}"
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))

mkdir -p models "$(dirname "${LOG_FILE}")"

CMD=(
  python3 -u -m src.main
  --algorithm sac
  --iterations "${ITERATIONS}"
  --save-path "models/sac_checkpoint.pt"
  --log-interval 1
)
if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Started SAC training with nohup."
echo "PID: ${PID}"
echo "Log file: ${LOG_FILE}"
echo "Tail logs with: tail -f \"${LOG_FILE}\""
