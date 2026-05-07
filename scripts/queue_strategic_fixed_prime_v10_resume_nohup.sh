#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ec2-user/Polynomial2"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-strategic-fixed-prime-v10-resume}"
QUEUE_ID="${QUEUE_ID:-${RUN_ID_PREFIX}-queue-$(date -u +%Y%m%d_%H%M%S)}"
BLOCKING_PID="${BLOCKING_PID:-}"
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
OUT_DIR="$ROOT/artifacts/$QUEUE_ID"
LOG_PATH="$OUT_DIR/queue.log"

mkdir -p "$OUT_DIR"

echo "queue_id=$QUEUE_ID"
echo "log=$LOG_PATH"
echo "blocking_pid=${BLOCKING_PID:-none}"
echo "min_free_mib=$MIN_FREE_MIB"
echo "poll_seconds=$POLL_SECONDS"

while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  blocking_alive=0
  if [[ -n "$BLOCKING_PID" ]] && kill -0 "$BLOCKING_PID" 2>/dev/null; then
    blocking_alive=1
  fi
  echo "[$timestamp] free_mib=$free_mib blocking_alive=$blocking_alive" | tee -a "$LOG_PATH"

  if [[ "$blocking_alive" -eq 0 && "$free_mib" -ge "$MIN_FREE_MIB" ]]; then
    RUN_ID="${RUN_ID:-${RUN_ID_PREFIX}-$(date -u +%Y%m%d_%H%M%S)}"
    echo "[$timestamp] launching_run_id=$RUN_ID" | tee -a "$LOG_PATH"
    RUN_ID="$RUN_ID" "$ROOT/scripts/start_strategic_fixed_prime_v10_resume_nohup.sh" | tee -a "$LOG_PATH"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
