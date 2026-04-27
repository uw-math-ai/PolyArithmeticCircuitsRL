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
sbatch slurm_scripts/build_on_path_cache_c2_c8.slurm
```

By default this writes:

```text
on_path_cache/n2_mod5_deg6_C2_C8_seed42
```

for 2-variable, mod-5 targets with `max_degree=6` and curriculum
complexities `2 3 4 5 6 7 8`. The cache stores the actual train/val/test
target ID splits, and training loads those splits directly.

The cache geometry must match the training run:

```text
n_variables, mod, max_degree, split_seed, and requested complexities
```

The requested curriculum complexities may be a subset of the cached
complexities. To override defaults at submission time:

```bash
CACHE_DIR=on_path_cache/n2_mod5_deg6_C2_C6_seed42 \
COMPLEXITIES="2 3 4 5 6" \
MAX_ON_PATH_SIZE=8192 \
sbatch slurm_scripts/build_on_path_cache_c2_c8.slurm
```

To run the same clean OnPath flow on 3-variable targets later, override the
geometry and output names:

```bash
N_VARIABLES=3 \
CACHE_DIR=on_path_cache/n3_mod5_deg6_C2_C8_seed42 \
RESULTS_DIR=results/ppo-mcts-jax_clean_onpath_curriculum_3var_C2_C8 \
WANDB_RUN_NAME=ppo-mcts-jax_clean_onpath_curriculum_3var_C2_C8 \
sbatch slurm_scripts/run_clean_onpath_curriculum_c2_c8.slurm
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
- `run_clean_onpath_curriculum_c2_c8.slurm`: large JAX PPO+MCTS adaptive
  curriculum run from C2 through C8 on 2-variable targets using cached
  `clean_onpath` teacher shaping.

The adaptive curriculum samples only the current complexity, then advances or
backs off using the current-level success window. When complexity changes, the
success window is cleared and dwell is reset, so the next decision only uses
episodes from the new level. Min dwell is symmetric: it blocks both advance and
backoff. The default `CURRICULUM_MIN_DWELL_ITERATIONS=1` means one completed
outer PPO iteration at the current level before another level-change check.

The large clean run defaults to `ON_PATH_PHI_MODE=max_step` because it rewards
deep progress on high-complexity targets and is less vulnerable to collecting
incompatible nodes from the union of optimal routes. Use
`ON_PATH_PHI_MODE=count` for the denser count-based ablation.

Run the large cached curriculum job after the cache job finishes:

```bash
sbatch slurm_scripts/run_clean_onpath_curriculum_c2_c8.slurm
```

Useful overrides:

```bash
ITERATIONS=4000 \
PPO_EPOCHS=8 \
MCTS_BATCH_SIZE=512 \
MCTS_SIMULATIONS=32 \
ON_PATH_PHI_MODE=max_step \
CURRICULUM_WINDOW=512 \
CURRICULUM_MIN_DWELL_ITERATIONS=1 \
sbatch slurm_scripts/run_clean_onpath_curriculum_c2_c8.slurm
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
- `results/ppo-mcts-jax_clean_onpath_curriculum_2var_C2_C8`
