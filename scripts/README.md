# scripts/ - CLI entrypoints and helper wrappers

This folder contains Python entrypoints and shell wrappers for common workflows.

## Quick workflow (recommended)

```bash
# 1) Build dataset split for complexity up to 6
bash scripts/make_dataset.sh

# 2) Train using that split
bash scripts/train_with_split.sh

# 3) Evaluate a checkpoint
bash scripts/eval_checkpoint.sh runs/split_train_*/best_lvl6.pt
```

You can override defaults through environment variables, for example:

```bash
N_VARS=2 MAX_OPS=6 TOTAL_STEPS=800000 SEED=43 bash scripts/train_with_split.sh
```

## Python entrypoints

### `train.py`

Trains curriculum DQN + HER.

```bash
python scripts/train.py
python scripts/train.py --n_vars 2 --max_ops 6 --total_steps 500000
python scripts/train.py --interesting data/polys_nvars2_maxops6.train.jsonl \
  --eval_jsonl data/polys_nvars2_maxops6.eval.jsonl
```

Key args:
- `--max_ops` default: `6`
- `--interesting`: training split JSONL
- `--eval_jsonl`: held-out eval split JSONL
- `--no-auto-interesting`: disable auto graph-generated fallback

### `evaluate.py`

Evaluates one checkpoint.

```bash
python scripts/evaluate.py --checkpoint runs/best_lvl3.pt --episodes 200
python scripts/evaluate.py --checkpoint runs/best_lvl3.pt --max_ops 6 --episodes 200
```

### `build_dataset.py`

One-time offline dataset generation and train/eval split.

```bash
python scripts/build_dataset.py --n_vars 2 --max_ops 6 --out_dir data/
```

Important options:
- `--only_shortcut` / `--no-only_shortcut`
- `--eval_frac 0.2`
- `--max_graph_nodes`, `--max_successors`, `--max_seconds`

For very small budgets (for example `max_ops <= 2`), `--no-only_shortcut` is often needed to avoid empty splits.

### `run_baselines.py`

Runs symbolic/search baselines on random targets or a held-out split.

```bash
python scripts/run_baselines.py --n_vars 2 --max_ops 6 \
  --eval_jsonl data/polys_nvars2_maxops6.eval.jsonl \
  --skip_exhaustive
```

For `max_ops > 4`, use `--skip_exhaustive` unless you intentionally want a very expensive exhaustive run.

## Shell wrappers

### `make_dataset.sh`

Builds `data/polys_nvars{N}_maxops{K}.{train,eval}.jsonl`.

Defaults:
- `N_VARS=2`
- `MAX_OPS=6`
- `OUT_DIR=./data`

Example:

```bash
N_VARS=2 MAX_OPS=6 bash scripts/make_dataset.sh
```

### `train_with_split.sh`

Ensures split files exist (auto-build enabled by default), then trains with
`--interesting` and `--eval_jsonl` wired up.

Defaults:
- `N_VARS=2`
- `MAX_OPS=6`
- `TOTAL_STEPS=500000`
- `AUTO_BUILD_DATASET=1`

### `eval_checkpoint.sh`

Evaluates one checkpoint with default `EPISODES=500`.

Optional env override:
- `MAX_OPS_OVERRIDE=6`

## Full multi-run orchestrator

### `run_all_tmux.sh`

Launches dataset build, multi-seed training, baselines, and checkpoint sweeps in tmux.

```bash
bash scripts/run_all_tmux.sh
```

Defaults:
- `MAX_OPS=6`
- `SEEDS="42 43 44"`
- `LEVELS="1 2 3 4 5 6"` (derived from `MAX_OPS`)

Attach / stop:

```bash
tmux attach -t polyrl
tmux kill-session -t polyrl
```
