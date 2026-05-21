#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Slurm job script for top-down PPO (and PPO+MCTS) training on UW Hyak Klone.
#
# Runs scripts/run_ppo_finetune.py — the trainer described in README.md under
# "PPO and PPO+MCTS Fine-Tuning", which implements the top-down decomposition
# loop (frontier expansion + additive split f = g + h + auto-factorization).
#
# Quick start on Klone:
#   1. ssh klone.hyak.uw.edu
#   2. cd to your repo checkout
#   3. bash scripts/setup_cloud.sh --gpu          # one-time .venv bootstrap
#   4. Edit the #SBATCH --account directive below to your Klone account.
#   5. sbatch scripts/run_hyak_ppo.slurm
#
# Common env-var overrides (set before sbatch or pass via --export):
#   RUN_ID         – unique name for this run (default: decomp-rl-ppo-<ts>)
#   OUT_DIR        – outputs directory      (default: $REPO/artifacts/$RUN_ID)
#   ITERATIONS     – PPO update iterations  (default: 2000)
#   ROLLOUTS       – episodes per update    (default: 32)
#   CANDIDATES     – split candidates/step  (default: 32)
#   MAX_STEPS      – max episode length     (default: 32)
#   LR             – Adam learning rate     (default: 3e-4)
#   ENT_COEF       – entropy bonus weight   (default: 0.01)
#   LIB_REWARD_W   – library shaping weight (default: 1.0)
#   SEED           – training seed          (default: 0)
#   PRIME          – finite field prime     (default: 3)
#   CKPT_EVERY     – periodic ckpt interval (default: 25 iters)
#   USE_MCTS       – 1 enables PPO+MCTS     (default: 1)
#   MCTS_SIMS      – PUCT simulations/step  (default: 64)
#   MCTS_DEPTH     – AndOrSearch max depth  (default: 6)
#   MCTS_TEMP      – visit-count temperature(default: 1.0)
#   MCTS_DISTILL   – distillation weight    (default: 1.0)
#   WANDB_API_KEY  – your W&B API key (NEVER commit this; export before sbatch
#                    or run `wandb login` once on the login node)
#   WANDB_ENTITY   – W&B entity/user/team   (empty disables W&B)
#   WANDB_PROJECT  – W&B project            (default: PolyArithmeticCircuitsRL)
#   WANDB_MODE     – online|offline|disabled|auto (default: auto)
#
# Auto-resume:
#   The script always passes --resume latest to the trainer. The trainer scans
#   $OUT_DIR/checkpoints/iter_*.pt and continues from the newest one. So you
#   can submit, requeue after preemption, or simply rerun the same sbatch with
#   a stable OUT_DIR and training will pick up where it left off.
#
#   Example with preemptible partition:
#     OUT_DIR=/gscratch/mylab/myuser/decomp-rl-ppo-run1 \
#     ITERATIONS=5000 \
#     sbatch --partition=ckpt-all --time=8:00:00 scripts/run_hyak_ppo.slurm
# ═══════════════════════════════════════════════════════════════════════════

# ── Slurm resource requests ──────────────────────────────────────────────
#SBATCH --job-name=decomp-rl-ppo
#SBATCH --account=CHANGEME          # <── replace with your Klone account
#SBATCH --partition=gpu-a40         # gpu-a40 | gpu-l40s | ckpt-all | etc.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00             # max 24h on gpu-a40; 8h on ckpt-all
#SBATCH --output=/dev/null          # redirected to OUT_DIR below
#SBATCH --error=/dev/null

set -euo pipefail

# ── Resolve repo root ────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Load cluster modules ─────────────────────────────────────────────────
# Klone: adjust the CUDA version to match the one your .venv torch was built
# against. Run `module avail cuda` on the login node to see options.
if command -v module &>/dev/null; then
    module load cuda/12.1.1 2>/dev/null || \
    module load cuda/11.8.0 2>/dev/null || \
    true  # fall through if neither is available
fi

# ── Run identity ─────────────────────────────────────────────────────────
RUN_ID="${RUN_ID:-decomp-rl-ppo-$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO/artifacts/$RUN_ID}"
CKPT_DIR="$OUT_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

# Redirect all further output to a file inside the run directory.
exec > "$OUT_DIR/slurm_${SLURM_JOB_ID:-local}.log" 2>&1

echo "job_id=${SLURM_JOB_ID:-local}"
echo "run_id=$RUN_ID"
echo "out_dir=$OUT_DIR"
echo "repo=$REPO"
echo "node=$(hostname)"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Python / venv ─────────────────────────────────────────────────────────
PYTHON="$REPO/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: virtualenv not found at $REPO/.venv" >&2
    echo "Run: bash scripts/setup_cloud.sh --gpu" >&2
    exit 1
fi

# ── PPO hyperparameters (env-overridable) ────────────────────────────────
ITERATIONS="${ITERATIONS:-2000}"
ROLLOUTS="${ROLLOUTS:-32}"
CANDIDATES="${CANDIDATES:-32}"
MAX_STEPS="${MAX_STEPS:-32}"
LR="${LR:-3e-4}"
LIB_REWARD_W="${LIB_REWARD_W:-1.0}"
ENT_COEF="${ENT_COEF:-0.01}"
SEED="${SEED:-0}"
PRIME="${PRIME:-3}"
CKPT_EVERY="${CKPT_EVERY:-25}"
USE_MCTS="${USE_MCTS:-1}"
MCTS_SIMS="${MCTS_SIMS:-64}"
MCTS_DEPTH="${MCTS_DEPTH:-6}"
MCTS_TEMP="${MCTS_TEMP:-1.0}"
MCTS_DISTILL="${MCTS_DISTILL:-1.0}"

MCTS_ARGS=()
if [[ "$USE_MCTS" == "1" ]]; then
    MCTS_ARGS=(
        --use-mcts
        --mcts-simulations "$MCTS_SIMS"
        --mcts-max-depth   "$MCTS_DEPTH"
        --mcts-temperature "$MCTS_TEMP"
        --mcts-distill-coef "$MCTS_DISTILL"
    )
fi

# Optional warm-start from a supervised/ExIt checkpoint. If $WARM_START is set
# and no periodic ckpt is found, the trainer warm-starts from that file.
WARM_START="${WARM_START:-}"
WARM_ARGS=()
[[ -n "$WARM_START" ]] && WARM_ARGS=(--checkpoint-in "$WARM_START")

# ── W&B settings ─────────────────────────────────────────────────────────
# WANDB_API_KEY must come from the submitter's environment (or ~/.netrc via
# `wandb login`); it is never written to disk by this script.
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-PolyArithmeticCircuitsRL}"
WANDB_MODE="${WANDB_MODE:-auto}"
WANDB_ARGS=(
    --wandb-entity  "$WANDB_ENTITY"
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-id  "$RUN_ID"
    --wandb-mode    "$WANDB_MODE"
)

# ── Launch training ───────────────────────────────────────────────────────
exec env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    HOME="$HOME" \
    WANDB_API_KEY="${WANDB_API_KEY:-}" \
    WANDB_DIR="$OUT_DIR" \
    "$PYTHON" \
    "$REPO/scripts/run_ppo_finetune.py" \
    --prime "$PRIME" \
    --iterations "$ITERATIONS" \
    --rollouts-per-update "$ROLLOUTS" \
    --candidates-per-step "$CANDIDATES" \
    --max-episode-steps "$MAX_STEPS" \
    --learning-rate "$LR" \
    --library-reward-weight "$LIB_REWARD_W" \
    --entropy-coef "$ENT_COEF" \
    --seed "$SEED" \
    --device cuda \
    --checkpoint-dir "$CKPT_DIR" \
    --checkpoint-every "$CKPT_EVERY" \
    --resume latest \
    --checkpoint-out "$CKPT_DIR/final.pt" \
    --metrics-out   "$OUT_DIR/metrics.jsonl" \
    "${WARM_ARGS[@]}" \
    "${MCTS_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
