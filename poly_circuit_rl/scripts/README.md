# scripts/ - Training and evaluation entrypoints

Command-line wrappers for training and checkpoint evaluation.

## `train.py`

Runs curriculum DQN+HER+MCTS training.

### Examples

```bash
# Basic (MCTS enabled by default)
python scripts/train.py

# Use precomputed interesting-polynomial JSONL
python scripts/train.py \
  --interesting ../Game-Board-Generation/pre-training-data/game_board_C4.analysis.jsonl

# Disable fallback auto-generation
python scripts/train.py --no-auto-interesting

# Bound auto-generation graph growth
python scripts/train.py --gen-max-graph-nodes 20000 --gen-max-successors 30

# Disable MCTS (fall back to epsilon-greedy)
python scripts/train.py --no-mcts

# Tune MCTS parameters
python scripts/train.py --mcts_simulations 100 --mcts_c_puct 2.0 --mcts_temperature 0.5

# Enable Weights & Biases logging
python scripts/train.py --wandb_project poly-circuit-rl --wandb_entity your-team
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--n_vars` | `2` | Number of variables |
| `--max_ops` | `4` | Max op budget per episode |
| `--L` | `16` | Visible node slots |
| `--m` | `16` | Eval-point count |
| `--step_cost` | `0.05` | Base penalty for `ADD`/`MUL` |
| `--shaping_coeff` | `0.3` | Eval-distance shaping bonus scale (disabled in config default) |
| `--d_model` | `64` | Transformer hidden size |
| `--n_heads` | `4` | Attention heads |
| `--n_layers` | `3` | Transformer layers |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `256` | Replay batch size |
| `--buffer_size` | `100000` | Replay capacity |
| `--eps_decay_steps` | `50000` | Epsilon decay horizon |
| `--total_steps` | `500000` | Total env steps |
| `--seed` | `42` | RNG seed |
| `--log_dir` | `runs/` | Checkpoint directory |
| `--interesting` | `None` | JSONL path for precomputed interesting polynomials |
| `--no-auto-interesting` | `False` | Disable fallback auto-generation |
| `--gen-max-graph-nodes` | `None` | Auto-generation graph node cap |
| `--gen-max-successors` | `None` | Auto-generation per-node expansion cap |
| `--no-mcts` | `False` | Disable MCTS (use epsilon-greedy instead) |
| `--mcts_simulations` | `50` | Number of MCTS simulations per action |
| `--mcts_c_puct` | `1.5` | PUCT exploration constant |
| `--mcts_temperature` | `1.0` | Temperature for MCTS action selection |
| `--wandb_project` | `None` | W&B project name (enables logging) |
| `--wandb_entity` | `None` | W&B team/user namespace |
| `--wandb_run_name` | `None` | Optional run name override |

If `--interesting` is omitted and auto-generation is enabled, training uses `GenerativeInterestingPolynomialSampler` as fallback.

Checkpoints:
- periodic best: `log_dir/best_lvl{level}.pt`
- final: `log_dir/final.pt`

## `evaluate.py`

Evaluates a saved checkpoint with deterministic policy.

```bash
python scripts/evaluate.py \
  --checkpoint runs/best_lvl2.pt \
  --max_ops 4 \
  --episodes 200
```

Outputs success rate, average reward, and average steps.
