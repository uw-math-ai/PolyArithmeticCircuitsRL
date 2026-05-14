# Architecture

This repository implements a learned symbolic search pipeline for polynomial arithmetic circuit synthesis. The code is organized around exact symbolic state transitions and learned candidate ranking. Exact verification is the correctness boundary.

The implementation is independent from the sibling `gumbel` directory. `gumbel` is historical context and may be used as a baseline candidate, but code in `src/lgs` should not import from it or depend on its factor-library, reward-shaping, or RL machinery.

## Directory Layout

```text
src/lgs/poly/
src/lgs/env/
src/lgs/search/
src/lgs/models/
src/lgs/training/
src/lgs/data/
src/lgs/eval/
scripts/
slurm/
tests/
```

## `src/lgs/poly/`

Purpose: exact finite-field polynomial arithmetic.

Main files:

- `fast_poly.py`
  - `Polynomial`
  - `FastPoly`
  - `PolynomialDegreeError`
  - domain and degree validation
- `poly_utils.py`
  - `make_variables`
  - `poly_from_terms`
  - `exact_divides`
  - domain helpers

Representation:

- sparse terms keyed by exponent tuple
- coefficients normalized modulo `p`
- zero coefficients removed
- canonical keys include `(p, n_vars, degree_cap, sorted_terms)`

Must not depend on:

- search
- models
- training
- `gumbel`
- approximate symbolic packages for verification

## `src/lgs/env/`

Purpose: exact synthesis task and circuit state abstractions.

Main classes:

- `ProblemInstance`: target polynomial, variable names, field, degree cap, operation budget, family metadata.
- `Action`: canonical `add(i, j)` or `mul(i, j)`.
- `CircuitState`: immutable state with nodes, action trace, parents, node keys, and budget.
- `Candidate`: proposed action plus exact polynomial result, features, tags, and scores.
- verification helpers:
  - `verify_state`
  - `verify_trace`
  - `execute_trace`
  - `verify_candidate_transition`

Must not depend on:

- neural models
- training loops
- benchmark suites
- `gumbel`

The environment layer defines correctness. A circuit is successful only when a state contains a polynomial exactly equal to the target in the same domain.

## `src/lgs/search/`

Purpose: generate and search over exact candidate actions.

Main files:

- `candidate_generator.py`
  - `generate_candidates`
  - `enumerate_basic_pair_candidates`
  - `compute_tier1_features`
  - `compute_tier2_features`
  - `unique_by_result_polynomial`
- `heuristic_score.py`
  - tier-1 and tier-2 symbolic scoring
- `beam_search.py`
  - `beam_search`
  - optional ranker-guided scoring
  - exact finished-state detection
- `search_history.py`
  - `SearchHistory`
  - `ExpandedStateRecord`

Search dependencies:

- may depend on `poly` and `env`
- may optionally call a ranker passed in by the caller
- should not own training logic
- should not depend on old factor-library state or `gumbel`

The current beam search always requires exact verification through `state.contains(instance.target)`. Model scores affect ordering, not correctness.

## `src/lgs/models/`

Purpose: offline candidate feature encoding and candidate scoring.

Main classes:

- `CandidateFeatureEncoder`
  - deterministic fixed-length feature vector
  - uses exact symbolic data and candidate features
- `CandidateRanker`
  - PyTorch MLP
  - maps `[batch, input_dim]` feature tensors to scalar scores

Must not depend on:

- beam-search internals
- benchmark split state
- stored cross-episode memory
- `gumbel`

The model learns `score(instance, state, candidate)`. It does not generate circuits directly.

## `src/lgs/training/`

Purpose: search-generated preference extraction and ranker training.

Main files:

- `preference_dataset.py`
  - `PreferenceExample`
  - `extract_preferences`
  - trace-step matching by action-prefix equality
- `train_ranker.py`
  - pairwise hinge ranking loss
  - training loop
  - checkpoint save/load
- `bootstrap_loop.py`
  - `BootstrapConfig`
  - `run_bootstrap_training`
  - validation-gated lambda promotion

Training code may call search and evaluation. It must not change exact verification semantics, import from `gumbel`, or use validation instances for preference extraction.

## `src/lgs/data/`

Purpose: deterministic target generation, benchmark suites, and curricula.

Main files:

- `target_generators.py`
  - power-of-sum targets
  - common-factor targets
  - product-of-sums targets
  - nested/reuse targets
  - structured random-circuit targets
- `benchmark_suite.py`
  - `BenchmarkSpec`
  - `make_structured_benchmark`
- `curriculum.py`
  - `FixedCurriculum`

Data code should only create `ProblemInstance`s. It should not train, evaluate, or call `gumbel`.

## `src/lgs/eval/`

Purpose: evaluate search and aggregate benchmark sweeps.

Main files:

- `evaluate_search.py`
  - `SearchEvalMetrics`
  - `evaluate_beam_search`
- `compare_rankers.py`
  - direct heuristic-vs-guided comparison helper
- `sweep.py`
  - `SweepConfig`
  - `SweepRow`
  - `run_search_sweep`
  - `summarize_sweep`
  - `summarize_failures`

Evaluation code runs search and reports metrics. It should not train models or mutate benchmark instances.

## `scripts/`

Purpose: local runnable entry points.

- `run_benchmark_sweep.py`: heuristic-only sweep, or guided sweep if a checkpoint is provided.
- `train_and_eval_benchmark.py`: structured benchmark split, bootstrap training, evaluation sweep, checkpoint output.
- `train_bootstrap.py`: tiny bootstrap smoke run.

Script defaults are intentionally small enough for local smoke tests. Larger sweeps are controlled through flags.

## `slurm/`

Purpose: server launch wrappers.

- `train_eval_benchmark.sbatch`
- `heuristic_sweep.sbatch`
- `guided_sweep.sbatch`
- `array_sweep.sbatch`

These scripts are CPU-first, write run metadata, and avoid W&B, containers, GPUs, and old `gumbel` logic.

## `tests/`

Purpose: regression tests for arithmetic, state transitions, candidate generation, search, preference extraction, ranker training, bootstrapping, benchmark generation, sweep aggregation, and Slurm-adjacent assumptions.

The standard command is:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
```
