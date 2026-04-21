# PolyArithmeticCircuitsRL

Reinforcement learning for minimum-operation arithmetic-circuit synthesis over exact multivariate polynomials.

The agent builds circuits with `ADD`, `MUL`, `SET_OUTPUT`, `STOP` actions and learns with DQN + HER + curriculum.

## What This Repo Contains

- Goal-conditioned Gymnasium environment for circuit construction
- DQN agent with transformer Q-network and action masking
- HER replay with decomposed rewards
- Optional MCTS action selection
- Offline/online target polynomial samplers
- Dataset builder for reproducible train/eval JSONL splits
- Symbolic/search baselines for comparison

## Default Complexity Setting

Current defaults target **complexity up to 6 operations**:
- `Config.max_ops = 6`
- `Config.curriculum_levels = (1, 2, 3, 4, 5, 6)`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

### 1) Build dataset split

```bash
bash scripts/make_dataset.sh
```

This writes:
- `data/polys_nvars2_maxops6.train.jsonl`
- `data/polys_nvars2_maxops6.eval.jsonl`
- `data/polys_nvars2_maxops6.meta.json`

### 2) Train with split

```bash
bash scripts/train_with_split.sh
```

Outputs checkpoints under `runs/...`:
- `best_lvl{level}.pt`
- `final.pt`

### 3) Evaluate checkpoint

```bash
bash scripts/eval_checkpoint.sh runs/<run_dir>/best_lvl6.pt
```

Optional override:

```bash
MAX_OPS_OVERRIDE=6 EPISODES=500 bash scripts/eval_checkpoint.sh runs/<run_dir>/best_lvl6.pt
```

## Main Entry Scripts

- Training: `scripts/train.py`
- Evaluation: `scripts/evaluate.py`
- Dataset generation: `scripts/build_dataset.py`
- Baselines: `scripts/run_baselines.py`
- Full tmux pipeline: `scripts/run_all_tmux.sh`

Detailed usage for each script is in `scripts/README.md`.

## One-Command Multi-Run Pipeline

Run full pipeline (dataset + multi-seed training + baselines + eval sweeps):

```bash
bash scripts/run_all_tmux.sh
```

Then:

```bash
tmux attach -t polyrl
```

## Repository Layout

- `poly_circuit_rl/`: core package
- `scripts/`: CLI entrypoints and wrappers
- `configs/`: config reference templates
- `tests/`: test suite

## Notes

- For `max_ops > 4`, exhaustive baseline search is usually too expensive; use `--skip_exhaustive` in `scripts/run_baselines.py` unless intentionally benchmarking exhaustive search.
- If you pass explicit dataset paths to training, path errors fail fast to avoid silently training on a different distribution.
