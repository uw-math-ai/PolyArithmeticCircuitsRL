# scripts/ - Training and evaluation entrypoints

Command-line wrappers for training and checkpoint evaluation.

## `train.py`

Runs curriculum DQN+HER training.

### Examples

```bash
# Basic
python scripts/train.py

# Use precomputed interesting-polynomial JSONL
python scripts/train.py \
  --interesting ../Game-Board-Generation/pre-training-data/game_board_C4.analysis.jsonl

# Disable fallback auto-generation
python scripts/train.py --no-auto-interesting

# Bound auto-generation graph growth
python scripts/train.py --gen-max-graph-nodes 20000 --gen-max-successors 30

# Tune shaping reward
python scripts/train.py --shaping_coeff 0.2
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--n_vars` | `2` | Number of variables |
| `--max_ops` | `4` | Max op budget per episode |
| `--L` | `16` | Visible node slots |
| `--m` | `16` | Eval-point count |
| `--step_cost` | `0.05` | Base penalty for `ADD`/`MUL` |
| `--shaping_coeff` | `0.3` | Eval-distance shaping bonus scale |
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
