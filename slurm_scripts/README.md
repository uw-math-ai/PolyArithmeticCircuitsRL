# Slurm Scripts

This folder is the canonical home for Hyak job scripts for the retained JAX
PPO+MCTS pipeline.

Current assumption for job submission:

- use `#SBATCH --account=stf`
- use `#SBATCH --partition=ckpt`

If Hyak rejects that combination for your user, update the scripts to the
account/partition pair reported by `hyakalloc`.

## Reward Modes

Existing scripts do not pass `--reward-mode`, so they run the default
`legacy` reward baseline. For clean sparse ablations add:

```bash
--reward-mode clean_sparse
```

For cached OnPath teacher shaping, first build/cache the target set, then add:

```bash
--reward-mode clean_onpath \
--graph-onpath-cache-dir on_path_cache \
--on-path-phi-mode max_step
```

The OnPath signal is disabled during normal evaluation.

## Cached OnPath Target Sets

Build the cached board-step OnPath target set before running
`clean_onpath` training:

```bash
sbatch slurm_scripts/build_on_path_cache_c1_c6.slurm
```

By default this writes:

```text
on_path_cache/n2_mod5_deg6_C1_C6_routes32_seed42
```

for 2-variable, mod-5 targets with `max_degree=6` and curriculum
complexities `1 2 3 4 5 6`. The cache stores the actual train/val/test
target ID splits, coherent optimal-route masks, and training loads those
splits directly.
The route-mask cache records how many targets hit the configured route cap and
fails by default if more than `MAX_ROUTE_TRUNCATION_RATE=0.25` of a complexity
hits `ON_PATH_NUM_ROUTES`.

The cache geometry must match the training run:

```text
n_variables, mod, max_degree, split_seed, and requested complexities
```

The requested curriculum complexities may be a subset of the cached
complexities. To override defaults at submission time:

```bash
CACHE_DIR=on_path_cache/n2_mod5_deg6_C1_C6_routes32_seed42 \
COMPLEXITIES="1 2 3 4 5 6" \
ON_PATH_NUM_ROUTES=32 \
MAX_ON_PATH_SIZE=8192 \
MAX_ROUTE_TRUNCATION_RATE=0.25 \
sbatch slurm_scripts/build_on_path_cache_c1_c6.slurm
```

To run the same clean OnPath flow on 3-variable targets later, override the
geometry and output names:

```bash
N_VARIABLES=3 \
CACHE_DIR=on_path_cache/n3_mod5_deg6_C1_C6_routes32_seed42 \
RESULTS_DIR=results/ppo-mcts-jax_clean_onpath_curriculum_3var_C1_C6 \
WANDB_RUN_NAME=ppo-mcts-jax_clean_onpath_curriculum_3var_C1_C6 \
sbatch slurm_scripts/run_clean_onpath_curriculum_c1_c6.slurm
```

## Training Scripts

- `run_ppo_mcts_c5_2var_gpu.slurm`: JAX PPO+MCTS legacy reward baseline on
  2-variable C5 targets.
- `run_ppo_mcts_c6_2var_gpu.slurm`: JAX PPO+MCTS legacy reward baseline on
  2-variable C6 targets.
- `run_ppo_mcts_c5_gpu.slurm`: JAX PPO+MCTS legacy reward baseline on
  3-variable C5 targets.
- `run_ppo_mcts_c6_gpu.slurm`: JAX PPO+MCTS legacy reward baseline on
  3-variable C6 targets.
- `run_jax_c5_c8.slurm`: JAX PPO+MCTS legacy reward baseline over fixed
  complexities 5, 6, 7, and 8.
- `run_clean_onpath_curriculum_c1_c6.slurm`: large JAX PPO+MCTS adaptive
  curriculum run from C1 through C6 on 2-variable targets using cached
  `clean_onpath` teacher shaping.

The adaptive curriculum samples only the current complexity, then advances or
backs off using the current-level success window. When complexity changes, the
success window is cleared and dwell is reset, so the next decision only uses
episodes from the new level. Min dwell is symmetric: it blocks both advance and
backoff. `BACKOFF_THRESHOLD < 0` disables backoff. The default
`CURRICULUM_MIN_DWELL_ITERATIONS=64` means 64 completed
outer PPO iterations at the current level before another level-change check;
setting it to `1` allows level changes on consecutive iterations while still
preventing same-iteration churn.

The large clean run defaults to `ON_PATH_PHI_MODE=depth_weighted` with
`ON_PATH_DEPTH_WEIGHT_POWER=1.0`, which gives deeper cached OnPath hits more
potential while staying bounded and monotonic. Use `ON_PATH_PHI_MODE=count` for
the denser uniform-count ablation, or `ON_PATH_PHI_MODE=max_step` for deepest-hit
only progress.

Clean OnPath caches now store `ON_PATH_NUM_ROUTES=32` coherent optimal-route
masks per target. The default `ON_PATH_ROUTE_CONSISTENCY_MODE=best_route_phi`
records every unique OnPath hit but computes potential as the best progress
within one coherent route. Use `lock_on_first_hit` to reproduce the older
irreversible route-mask narrowing, or `off` for union-style reward.
Inspect route-mask structure before a run with:

```bash
python scripts/inspect_on_path_cache.py \
  --cache-dir on_path_cache/n2_mod5_deg6_C1_C6_routes32_seed42 \
  --complexity 2
```

The clean OnPath Slurm defaults are intended to be stable for one GPU:
`MCTS_BATCH_SIZE=128`, `MCTS_SIMULATIONS=32`, `PPO_LR=1e-4`,
`MAX_GRAD_NORM=0.25`, `PPO_LOG_RATIO_CLIP=10.0`, `PPO_EPOCHS=2`,
`TARGET_KL=0.03`, `ENT_COEF=0.05`, `GRAPH_ONPATH_SHAPING_COEFF=3.0`,
`MAX_COMPLEXITY=6`, and `MAX_STEPS=12`. Progress bars are disabled by default
for cleaner Slurm logs; set `PROGRESS_BAR=1` to re-enable.

Run the large cached curriculum job after the cache job finishes:

```bash
sbatch slurm_scripts/run_clean_onpath_curriculum_c1_c6.slurm
```

Useful overrides:

```bash
ITERATIONS=4000 \
PPO_EPOCHS=2 \
PPO_LR=1e-4 \
MAX_GRAD_NORM=0.25 \
MCTS_BATCH_SIZE=128 \
MCTS_SIMULATIONS=32 \
TARGET_KL=0.03 \
ON_PATH_PHI_MODE=depth_weighted \
ON_PATH_DEPTH_WEIGHT_POWER=1.0 \
ON_PATH_ROUTE_CONSISTENCY_MODE=best_route_phi \
ADVANCE_THRESHOLD=0.97 \
BACKOFF_THRESHOLD=-1.0 \
CURRICULUM_WINDOW=2048 \
CURRICULUM_MIN_DWELL_ITERATIONS=64 \
sbatch slurm_scripts/run_clean_onpath_curriculum_c1_c6.slurm
```

For a C1→C3 stable debug run against the C1→C6 cache you already built, keep
the cache path as C1→C6 and only lower `MAX_COMPLEXITY`:

```bash
CACHE_DIR=on_path_cache/n2_mod5_deg6_C1_C6_routes32_seed42 \
MAX_COMPLEXITY=3 \
RESULTS_DIR=results/ppo-mcts-jax_clean_onpath_curriculum_2var_C1_C3 \
WANDB_RUN_NAME=ppo-mcts-jax_clean_onpath_curriculum_2var_C1_C3 \
ON_PATH_PHI_MODE=depth_weighted \
ON_PATH_DEPTH_WEIGHT_POWER=1.0 \
ON_PATH_ROUTE_CONSISTENCY_MODE=best_route_phi \
ADVANCE_THRESHOLD=0.97 \
BACKOFF_THRESHOLD=-1.0 \
CURRICULUM_WINDOW=2048 \
CURRICULUM_MIN_DWELL_ITERATIONS=64 \
sbatch slurm_scripts/run_clean_onpath_curriculum_c1_c6.slurm
```

## Evaluation Scripts

- `run_eval_ppo_mcts_2var_gpu.slurm`: evaluates retained 2-variable JAX
  PPO+MCTS checkpoints.
- `run_eval_ppo_mcts_gpu.slurm`: evaluates retained 3-variable JAX PPO+MCTS
  checkpoints.

## Expected Output Directories

- `results/ppo-mcts-jax_fl_C5_2var`
- `results/ppo-mcts-jax_fl_C6_2var`
- `results/ppo-mcts-jax_fl_C5`
- `results/ppo-mcts-jax_fl_C6`
- `results/ppo-mcts-jax_fl_C5_C8`
- `results/ppo-mcts-jax_clean_onpath_curriculum_2var_C1_C6`
