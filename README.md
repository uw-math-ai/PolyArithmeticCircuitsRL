# Polynomial Arithmetic Circuits PPO+MCTS

This repository trains agents to construct arithmetic circuits for target
polynomials over a finite field. The retained training pipeline is PPO+MCTS:
MCTS improves action selection during rollouts, and PPO updates the shared
policy-value network from the collected trajectories.

Two implementations are kept:

- `ppo-mcts`: PyTorch policy-value network with sequential neural MCTS.
- `ppo-mcts-jax`: JAX/Flax policy-value network with batched MCTS through
  `mctx`, intended for GPU/HPC runs.

## Project Structure

```text
src/
├── config.py                     # Shared environment, reward, PPO, MCTS, and logging config
├── environment/
│   ├── fast_polynomial.py        # Dense finite-field polynomial backend
│   ├── polynomial_utils.py       # SymPy helpers and conversions
│   ├── action_space.py           # Action encode/decode and valid masks
│   ├── circuit_game.py           # PyTorch-side circuit construction environment
│   └── factor_library.py         # Optional subgoal/reuse reward library
├── game_board/
│   ├── generator.py              # BFS board builder and target samplers
│   └── on_path.py                # Cached board-step OnPath teacher signal
├── models/
│   ├── gnn_encoder.py            # PyTorch GNN graph encoder
│   └── policy_value_net.py       # PyTorch policy-value network
├── algorithms/
│   ├── mcts.py                   # Neural MCTS for PyTorch PPO+MCTS
│   ├── ppo_mcts.py               # PyTorch PPO+MCTS trainer
│   ├── jax_env.py                # Pure JAX environment and polynomial ops
│   ├── jax_net.py                # Flax policy-value network
│   └── ppo_mcts_jax.py           # JAX batched PPO+MCTS trainer
├── evaluation/
│   └── evaluate.py               # PyTorch PPO+MCTS evaluation harness
└── main.py                       # CLI entry point
```

Other retained top-level files:

- `eval_ppo_mcts_jax.py`: evaluates JAX checkpoint series.
- `container.def` and `container_jax.def`: Apptainer container definitions.
- `slurm_scripts/`: Hyak job scripts for JAX PPO+MCTS training/evaluation.
- `results/ppo-mcts*` and `results/ppo-mcts-jax*`: retained PPO+MCTS run assets.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU JAX runs, use a CUDA-compatible JAX install or build the JAX container:

```bash
apptainer build container_jax.sif container_jax.def
```

## Training

PyTorch PPO+MCTS:

```bash
python -m src.main --algorithm ppo-mcts --iterations 100
```

Smaller local PyTorch sanity run:

```bash
python -m src.main \
  --algorithm ppo-mcts \
  --iterations 1 \
  --steps-per-update 16 \
  --mcts-simulations 2
```

JAX batched PPO+MCTS:

```bash
python -m src.main \
  --algorithm ppo-mcts-jax \
  --iterations 100 \
  --mcts-batch-size 256 \
  --mcts-simulations 16
```

JAX fixed-complexity training:

```bash
python -m src.main \
  --algorithm ppo-mcts-jax \
  --fixed-complexities 5 6 7 8 \
  --max-complexity 8 \
  --max-degree 8 \
  --iterations 200
```

Useful shared flags:

- `--n-variables`, `--mod`, `--max-complexity`, `--max-steps`
- `--max-degree`
- `--mcts-simulations`
- `--steps-per-update`
- `--reward-mode {legacy,clean_sparse,clean_onpath}`
- `--no-curriculum`
- `--no-factor-library`
- `--factor-subgoal-reward`, `--factor-library-bonus`, `--completion-bonus`
- `--results-dir`
- `--wandb`, `--wandb-project`, `--wandb-entity`, `--wandb-run-name`

## Reward Modes

The default `legacy` mode preserves the existing term-similarity, factor-library,
and completion rewards for baseline ablations.

`clean_sparse` uses only:

```text
terminal_success_reward + step_penalty
```

`clean_onpath` adds potential shaping from cached board-step shortest-path
structure:

```text
terminal_success_reward
+ step_penalty
+ graph_onpath_shaping_coeff * (gamma * phi_after - phi_before)
```

Build an OnPath cache before using `clean_onpath`:

```bash
python -m src.game_board.on_path \
  --complexities 5 6 \
  --n-variables 3 \
  --mod 5 \
  --max-degree 6 \
  --cache-dir on_path_cache \
  --split-seed 42
```

Then train with:

```bash
python -m src.main \
  --algorithm ppo-mcts-jax \
  --fixed-complexities 6 \
  --reward-mode clean_onpath \
  --graph-onpath-cache-dir on_path_cache
```

The OnPath cache is a teacher signal under the current game-board step/depth
metric. It is not a proof of globally operation-minimal circuit size. Evaluation
is oracle-free by default; checkpoints trained with `clean_onpath` are evaluated
with `clean_sparse` reward semantics unless an explicit ablation says otherwise.

## Evaluation

Evaluate a PyTorch PPO+MCTS checkpoint:

```bash
python -m src.main \
  --algorithm ppo-mcts \
  --eval-only \
  --checkpoint results/ppo-mcts_C5/checkpoint.pt
```

Evaluate JAX checkpoints:

```bash
python eval_ppo_mcts_jax.py \
  --checkpoint-dir results/ppo-mcts-jax_C6 \
  --iterations 50 100 150 200 \
  --complexities 5 6 \
  --num-trials 1000 \
  --mcts-simulations 16
```

## Tests

```bash
pytest tests
```

The JAX test module uses `pytest.importorskip`, so it is skipped when JAX,
Flax, Optax, or mctx are not installed.
