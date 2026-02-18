# scripts/ — Training and Evaluation Entrypoints

These are the command-line scripts you run directly. They are thin wrappers that parse arguments and call into the library.

## train.py — Train the Agent

Runs the full DQN+HER curriculum training loop.

```bash
# Basic training with defaults (2 variables, max 4 ops, 500K steps)
python scripts/train.py

# With interesting polynomial data (recommended for better learning)
python scripts/train.py \
  --interesting ../Game-Board-Generation/pre-training-data/game_board_C4.analysis.jsonl

# Custom hyperparameters
python scripts/train.py \
  --n_vars 2 \
  --max_ops 4 \
  --d_model 128 \
  --n_layers 4 \
  --total_steps 1000000 \
  --log_dir runs/large_model
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_vars` | 2 | Number of polynomial variables |
| `--max_ops` | 4 | Max operations per episode |
| `--d_model` | 64 | Transformer hidden dim |
| `--n_heads` | 4 | Attention heads |
| `--n_layers` | 3 | Transformer layers |
| `--total_steps` | 500000 | Total environment steps |
| `--log_dir` | `runs/` | Where to save checkpoints |
| `--interesting` | None | Path to analysis JSONL file |

Checkpoints are saved to `log_dir/best_lvlN.pt` whenever eval success rate improves, and `log_dir/final.pt` at the end.

## evaluate.py — Evaluate a Checkpoint

Runs a trained agent with a deterministic policy and reports success rate.

```bash
python scripts/evaluate.py \
  --checkpoint runs/best_lvl2.pt \
  --max_ops 4 \
  --episodes 200
```

Output:
```
Checkpoint: runs/best_lvl2.pt
Max ops:    4
Episodes:   200
Success:    73.50%
Avg reward: 0.582
Avg steps:  3.2
```
