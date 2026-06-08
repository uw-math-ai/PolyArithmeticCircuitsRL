# Learned Symbolic Search for Polynomial Arithmetic Circuit Synthesis

This project synthesizes arithmetic circuits for target polynomials by combining exact symbolic arithmetic, target-conditioned candidate generation, learned candidate ranking, and bounded symbolic search. The implementation is intentionally small and auditable: every arithmetic operation is exact over a finite field, every returned circuit is verified by polynomial equality, and the learned model only scores candidates inside a symbolic search loop.

```text
ProblemInstance
  -> CircuitState
  -> CandidateGenerator
  -> HeuristicScorer / CandidateRanker
  -> BeamSearch / GumbelSearch
  -> ExactVerifier
  -> PreferenceExtractor
  -> BootstrapTrainer
  -> BenchmarkSweeps
```

## Current Status

Implemented:

- exact sparse finite-field polynomial backend
- fail-loud arithmetic and domain validation
- circuit state abstractions and exact trace verification
- target-conditioned add/mul candidate generation
- support-geometry / Minkowski candidate features
- candidate recall and candidate-rank diagnostics
- heuristic-only beam search
- Gumbel / Sequential-Halving planner over generated candidates
- search-history based preference extraction
- fixed-feature MLP candidate ranker with the current 49-feature schema
- ranker-guided beam search
- bootstrapped ranker training loop
- structured benchmark generation and sweep scripts
- matched-budget four-method planner sweep:
  - `beam_heuristic`
  - `beam_guided`
  - `gumbel_heuristic`
  - `gumbel_guided`
- fair expansion accounting through `SearchHistory.num_expansions` and `history_expansion_count`
- optional W&B logging
- Slurm wrappers for server runs

Not implemented yet:

- value head
- full AlphaZero-style MCTS with value targets
- Transformer or GNN ranker
- learned candidate proposal or hybrid candidate-set quotas
- scalar penalties or model-aware candidate preselection outside heuristic top-K
- local rewrite or post-search circuit improvement
- old factor-library logic or reward-shaped RL training

The project is independent from the sibling `gumbel` directory. That older code can be used as a historical baseline or reference point, but this implementation does not import from it or depend on it.

## Installation

Use Python 3.10 or newer. The core tests require `pytest`; ranker training requires PyTorch.

```bash
cd learn-guided-search
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install torch
```

If PyTorch is already available in your environment, the last command may not be needed.

## Test

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is used because some global environments load a Hydra/OmegaConf pytest plugin that can conflict with this local test environment before collection begins.

## Quickstart

Run a tiny heuristic-only benchmark sweep:

```bash
python scripts/run_benchmark_sweep.py \
  --max-instances-per-family 1 \
  --beam-widths 1 \
  --candidate-ks 4 \
  --tier2-ms 16 \
  --no-wandb
```

Run a tiny matched-budget planner sweep:

```bash
python scripts/run_planner_sweep.py \
  --max-instances-per-family 1 \
  --beam-widths 1 \
  --candidate-ks 4 \
  --tier2-ms 16 \
  --expansion-budgets 16 \
  --gumbel-seeds 0 \
  --output-dir results/local_planner_smoke \
  --no-wandb
```

Run a tiny train-and-evaluate benchmark:

```bash
python scripts/train_and_eval_benchmark.py \
  --rounds 1 \
  --max-instances-per-family 1 \
  --epochs-per-round 3 \
  --beam-widths 1 \
  --candidate-ks 4 \
  --tier2-ms 16 \
  --no-wandb
```

Outputs are written to `results/` by default:

- `sweep_rows.jsonl`
- `sweep_summary.json`
- `sweep_failures.json`
- `planner_sweep_rows.jsonl` for planner sweeps
- `planner_sweep_summary.json` for planner sweeps
- `planner_sweep_deltas.json` for planner sweeps
- `ranker.pt` for train-and-eval runs
- `bootstrap_metrics.json` for train-and-eval runs

## Slurm

Server wrappers live in `slurm/`.

```bash
sbatch slurm/train_eval_benchmark.sbatch
sbatch slurm/heuristic_sweep.sbatch
CHECKPOINT=results/server_runs/server_run_001/ranker.pt sbatch slurm/guided_sweep.sbatch
CHECKPOINT=results/server_runs/server_run_001/ranker.pt sbatch slurm/planner_sweep.sbatch
```

The experiment scripts support W&B through `--wandb` / `--no-wandb`. Slurm
defaults are documented per wrapper; set `ENABLE_WANDB=0` or pass `--no-wandb`
to run without it.
See [slurm/README.md](slurm/README.md) for array jobs, metadata logging, and output layout.

## Key Concepts

- `FastPoly`: sparse exact polynomial arithmetic over `F_p`.
- `ProblemInstance`: target polynomial, variable names, field, degree cap, and operation budget.
- `Action`: canonical `add(i, j)` or `mul(i, j)`.
- `CircuitState`: immutable circuit state containing base nodes, constructed nodes, action trace, parents, and node keys.
- `Candidate`: proposed action plus exact result polynomial, features, tags, and scores.
- `CandidateFeatureEncoder`: deterministic encoder for the 49-feature ranker schema; checkpoints save their feature names.
- `beam_search`: exact symbolic search with optional learned ranker scoring.
- `gumbel_search`: Gumbel / Sequential-Halving planner over generated candidates, with exact verification and explicit expansion accounting.
- `compute_recall_for_best_finished`: candidate recall/rank diagnostics for solved traces.
- `extract_preferences`: converts successful beam-search histories into pairwise candidate preferences.
- `run_bootstrap_training`: repeats search, preference extraction, ranker training, validation, and lambda promotion.
- `run_planner_sweep`: matched-budget comparison of Beam and Gumbel, heuristic and guided.

## Limitations

- Candidate generation still returns heuristic top-K before model reranking. If the right action is not in that set, the ranker cannot recover it.
- The learned ranker has only been tested on early structured polynomial families so far.
- Benchmarks are structured synthesis tasks, not arbitrary dense polynomial minimization.
- The MLP ranker uses hand-built features rather than a graph, Transformer, or learned symbolic representation.
- There is no value head yet, so delayed-good branches can still be mis-scored.
- The current Gumbel planner is a clean planner over generated `Candidate` objects, not a full AlphaZero/MCTS system with value targets or visit-count training.
- Learned candidate proposal and hybrid candidate-set quotas are not implemented yet.
- The code currently optimizes verified construction under search budgets. It does not claim optimal arithmetic circuit minimization.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Method](docs/METHOD.md)
- [Experiments](docs/EXPERIMENTS.md)
- [Development](docs/DEVELOPMENT.md)
- [Related Work And Rationale](docs/RELATED_WORK_AND_RATIONALE.md)
- [Roadmap](docs/ROADMAP.md)
