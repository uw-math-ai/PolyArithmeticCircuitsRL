#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Slurm job script for decomp-rl training on Hyak (or any Slurm cluster).
#
# Quick start on Hyak:
#   1. Edit the #SBATCH directives below (account, partition, time).
#   2. sbatch scripts/run_hyak.slurm
#
# Key env-var overrides (pass via --export or set before sbatch):
#   RUN_ID          – unique name for this run (default: decomp-rl-<timestamp>)
#   OUT_DIR         – full path for outputs   (default: $REPO/artifacts/$RUN_ID)
#   RESUME_CKPT     – path to a specific .pt checkpoint to resume from
#                     (omit to auto-resume from the latest cycle checkpoint
#                      already in OUT_DIR/checkpoints/)
#   WANDB_MODE      – online | offline | disabled (default: online)
#   WANDB_ENTITY    – W&B entity (default: p-agi)
#   WANDB_PROJECT   – W&B project (default: PolyArithmeticCircuitsRL)
#
# Requeue / checkpoint-partition usage:
#   The script auto-detects the newest cycle_NNN.pt inside OUT_DIR and resumes
#   from it. So you can set a fixed OUT_DIR and resubmit the same sbatch
#   command after preemption; training continues seamlessly.
#
#   Example with checkpoint (preemptible) partition:
#     OUT_DIR=/gscratch/mylab/myuser/decomp-rl-run1 \
#     sbatch --partition=ckpt-all --time=8:00:00 scripts/run_hyak.slurm
# ═══════════════════════════════════════════════════════════════════════════

# ── Slurm resource requests ──────────────────────────────────────────────
#SBATCH --job-name=decomp-rl
#SBATCH --account=CHANGEME          # <── replace with your Hyak account
#SBATCH --partition=gpu-a40         # gpu-a40 | gpu-l40s | ckpt-all | etc.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00             # max 24h on gpu-a40; 8h on ckpt-all
#SBATCH --output=/dev/null          # redirected to OUT_DIR below
#SBATCH --error=/dev/null

set -euo pipefail

# ── Resolve repo root ────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Load cluster modules ─────────────────────────────────────────────────
# Hyak: adjust the CUDA version to match what your venv was built against.
# Run `module avail cuda` on the login node to see available versions.
if command -v module &>/dev/null; then
    module load cuda/12.1.1 2>/dev/null || \
    module load cuda/11.8.0 2>/dev/null || \
    true  # fall through if neither is available (e.g. non-Hyak clusters)
fi

# ── Run identity ─────────────────────────────────────────────────────────
RUN_ID="${RUN_ID:-decomp-rl-$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO/artifacts/$RUN_ID}"
mkdir -p "$OUT_DIR"

# Redirect all further output to a file inside the run directory.
exec > "$OUT_DIR/slurm_${SLURM_JOB_ID:-local}.log" 2>&1

echo "job_id=${SLURM_JOB_ID:-local}"
echo "run_id=$RUN_ID"
echo "out_dir=$OUT_DIR"
echo "repo=$REPO"
echo "node=$(hostname)"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Auto-resume ──────────────────────────────────────────────────────────
# If RESUME_CKPT is not set, look for the newest cycle checkpoint already
# saved in this OUT_DIR so a requeued job continues seamlessly.
RESUME_CKPT="${RESUME_CKPT:-}"
CKPT_DIR="$OUT_DIR/checkpoints"
if [[ -z "$RESUME_CKPT" && -d "$CKPT_DIR" ]]; then
    LATEST="$(ls -1 "$CKPT_DIR"/cycle_*.pt 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "$LATEST" ]]; then
        RESUME_CKPT="$LATEST"
        echo "auto_resume=$RESUME_CKPT"
    fi
fi

RESUME_ARGS=()
[[ -n "$RESUME_CKPT" ]] && RESUME_ARGS=(--resume-checkpoint "$RESUME_CKPT")

# ── W&B settings ─────────────────────────────────────────────────────────
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-p-agi}"
WANDB_PROJECT="${WANDB_PROJECT:-PolyArithmeticCircuitsRL}"

# ── Python / venv ─────────────────────────────────────────────────────────
PYTHON="$REPO/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: virtualenv not found at $REPO/.venv" >&2
    echo "Run: bash scripts/setup_cloud.sh --gpu" >&2
    exit 1
fi

# ── Launch training ───────────────────────────────────────────────────────
exec env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    HOME="$HOME" \
    WANDB_API_KEY="${WANDB_API_KEY:-}" \
    "$PYTHON" \
    "$REPO/scripts/run_full_experiment.py" \
    --output-dir "$OUT_DIR" \
    "${RESUME_ARGS[@]}" \
    --prime 3 \
    --seed 0 \
    --initial-supervised-count 3072 \
    --holdout-count 192 \
    --cycles 14 \
    --search-targets-per-cycle 128 \
    --recent-distill-sample-size 128 \
    --replay-sample-size 49152 \
    --elite-sample-size 1024 \
    --synthetic-sample-size 8192 \
    --replay-capacity 131072 \
    --elite-capacity 12288 \
    --supervised-epochs 12 \
    --cycle-epochs 8 \
    --learning-rate 3e-4 \
    --cycle-learning-rate 2e-4 \
    --weight-decay 1e-4 \
    --value-loss-weight 0.25 \
    --search-simulations 192 \
    --cycle-search-retries 2 \
    --cycle-search-fresh-search-per-target \
    --cycle-search-progress-interval 8 \
    --device cuda \
    --batch-size 4096 \
    --auto-batch-size \
    --max-auto-batch-size 65536 \
    --gpu-memory-target-fraction 0.96 \
    --cache-dataset-on-device \
    --model-hidden-dim 4096 \
    --model-shared-layers 6 \
    --model-value-hidden-dim 2048 \
    --model-value-layers 4 \
    --model-activation gelu \
    --reserve-cpu-cores 4 \
    --torch-cpu-threads 8 \
    --torch-interop-threads 1 \
    --nice-level 0 \
    --replay-uniform-fraction 0.05 \
    --curriculum-extra-primes= \
    --curriculum-max-vars 6 \
    --curriculum-max-support 7 \
    --curriculum-max-degree 5 \
    --curriculum-max-horner-degree 10 \
    --curriculum-max-inner-support 5 \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-id "$RUN_ID" \
    --wandb-mode "$WANDB_MODE"
