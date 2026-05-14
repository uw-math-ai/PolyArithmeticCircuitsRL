# Development

This document describes development practices for `learn-guided-search`.

## Setup

```bash
cd learn-guided-search
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install torch
```

The code targets Python 3.10+.

## Tests

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
```

Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` when global pytest plugins interfere with local collection.

## Code Style

- Keep changes scoped.
- Prefer explicit validation over clever shortcuts.
- Preserve exact arithmetic semantics.
- Add small, focused tests for each behavior change.
- Do not introduce new dependencies unless they directly serve a milestone.
- Use deterministic seeds in tests and scripts.

## Fail-Loud Arithmetic

Arithmetic must fail loudly. Do not silently do this:

- no silent truncation
- no silent overflow
- no silent domain mismatch
- no approximate equality for verification
- no invalid action normalization outside `Action`
- no dependency on stored factor-library memory
- no hidden import from sibling `gumbel`

Polynomial domains include:

- field prime `p`
- number of variables
- degree cap

Adding, subtracting, multiplying, comparing, or verifying across mismatched domains should raise or fail explicitly.

## Adding A Target Family

Target generators live in `src/lgs/data/target_generators.py`.

Recommended pattern:

1. Build exact `FastPoly` variables.
2. Construct the target with exact arithmetic.
3. Choose an operation budget that reflects intended construction difficulty.
4. Use `ProblemInstance`.
5. Include metadata:
   - `id`
   - `target_id`
   - `family`
   - `description`
   - `intended_complexity` or `generative_ops`
6. Add the family to `make_structured_benchmark` if it should be part of sweeps.
7. Add tests that initial states construct and search can run without crashing.

Do not use random dense polynomial targets as the main benchmark. Prefer structured targets with interpretable generation.

## Adding A Candidate Feature

Candidate features are computed in `src/lgs/search/candidate_generator.py` and encoded in `src/lgs/models/feature_encoder.py`.

Steps:

1. Add exact feature computation in `compute_tier1_features` or `compute_tier2_features`.
2. Add a default value path for missing features.
3. Add the feature name to `CANDIDATE_FEATURE_NAMES` if it should reach the MLP.
4. Update tests for deterministic feature length and no NaN/inf.
5. Consider whether old checkpoints remain compatible. Current checkpoint loading stores feature names.

Do not mutate `CircuitState` or `ProblemInstance` while computing features.

## Adding An Evaluation Metric

Evaluation code lives in:

- `src/lgs/eval/evaluate_search.py`
- `src/lgs/eval/sweep.py`

For sweep metrics:

1. Add fields to `SweepRow` if the metric is per-run.
2. Add aggregation in `summarize_sweep` or `summarize_failures`.
3. Update JSON-output scripts only if needed.
4. Add a deterministic test with hand-built `SweepRow`s where possible.

Keep grouping keys explicit. Guided-vs-heuristic deltas should only compare matching method groups with the same family and search budget.

## `gumbel` Boundary

Do not import from `gumbel`.

Acceptable:

- inspect old code manually for historical context
- compare against old runs as an external baseline
- document high-level differences

Not acceptable:

- import old polynomial code
- reuse old factor-library state
- add reward shaping copied from the old RL setup
- make new tests depend on old modules

## Milestone History

- Milestone 1: exact sparse finite-field polynomial backend and core environment abstractions.
- Milestone 2: fresh target-conditioned candidate generation.
- Milestone 3: heuristic-only beam search.
- Milestone 4: preference extraction from successful search histories.
- Milestone 5: fixed-feature MLP candidate ranker and pairwise training.
- Milestone 6A: optional ranker-guided beam search.
- Milestone 6B: bootstrapped training loop with validation-gated lambda promotion.
- Milestone 7: structured benchmark suite, sweep aggregation, and Slurm launch scripts.

## Before Submitting Large Runs

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
python scripts/run_benchmark_sweep.py --max-instances-per-family 1 --beam-widths 1 --candidate-ks 4 --tier2-ms 16
python scripts/train_and_eval_benchmark.py --rounds 1 --max-instances-per-family 1 --epochs-per-round 3 --beam-widths 1 --candidate-ks 4 --tier2-ms 16
```

For Slurm scripts:

```bash
bash -n slurm/*.sbatch
```
