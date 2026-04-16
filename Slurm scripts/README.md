# Slurm Scripts

This folder is the canonical home for Hyak job scripts.

Current assumption for job submission:

- use `#SBATCH --partition=ckpt`
- do not set an explicit `#SBATCH --account=...`

If Hyak rejects that combination for your user, update the scripts to the
account/partition pair reported by `hyakalloc`.

Current FL-enabled training scripts:

- `run_ppo_mcts_c5_2var_gpu.slurm`:
  JAX PPO+MCTS with factor-library rewards on 2-variable C5 targets.
- `run_ppo_mcts_c6_2var_gpu.slurm`:
  JAX PPO+MCTS with factor-library rewards on 2-variable C6 targets.
- `run_sac_c5_gpu.slurm`:
  SAC with factor-library rewards on 2-variable C5 targets.
- `run_sac_c6_gpu.slurm`:
  SAC with factor-library rewards on 2-variable C6 targets.

Additional scripts:

- `run_ppo_mcts_c5_gpu.slurm`, `run_ppo_mcts_c6_gpu.slurm`:
  JAX PPO+MCTS with factor-library rewards for the 3-variable runs.
- `run_jax_c5_c8.slurm`:
  JAX PPO+MCTS with factor-library rewards over fixed complexities 5, 6, 7, 8.
- `run_eval_ppo_mcts_2var_gpu.slurm`, `run_eval_ppo_mcts_gpu.slurm`:
  Evaluation scripts for JAX PPO+MCTS checkpoints.

Output directories:

- PPO+MCTS JAX 2-var FL:
  `results/ppo-mcts-jax_fl_C5_2var`, `results/ppo-mcts-jax_fl_C6_2var`
- PPO+MCTS JAX FL:
  `results/ppo-mcts-jax_fl_C5`, `results/ppo-mcts-jax_fl_C6`,
  `results/ppo-mcts-jax_fl_C5_C8`
- SAC FL:
  `results/sac_fl_C5_2var`, `results/sac_fl_C6_2var`
- Slurm logs:
  `results/slurm/`
