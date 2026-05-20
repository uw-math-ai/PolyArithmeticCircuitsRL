# Decomposition-Search RL for Arithmetic Circuit Discovery

This branch implements a top-down reinforcement-learning stack for discovering
low-cost arithmetic circuits for sparse polynomials over finite fields.

The central idea is to build a circuit by repeatedly decomposing a polynomial
`f` into additive split pieces `f = g + h`. Each split piece is automatically
factored over a finite field, cheap factored parts are rebuilt immediately, and
unresolved factors are pushed onto a frontier for later expansion. The learner
is trained to choose splits that reduce the final circuit cost.

## What Is Included

- Sparse finite-field polynomial arithmetic with canonical hashing.
- Finite-field factorization through a persistent Sage worker, with SymPy
  fallback when Sage is unavailable.
- A split-based decomposition environment, `DecompEnv`.
- Candidate split generation from support partitions, Horner pivots, common
  factors, family templates, random masks, and factorizable-library matches.
- A live `FactorizableLibrary` of useful discovered factorizations, including
  exact/scalar/permutation subset matching.
- Memoized AND/OR PUCT search over decomposition traces.
- A policy/value model for variable-sized split candidate sets.
- Supervised warm-start, search distillation, prioritized replay, elite
  self-imitation, and PPO/PPO+MCTS fine-tuning.
- Evaluation utilities against multiple closed-form/search baselines.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[train,dev]'
pytest
```

For a CPU-only development setup, `pip install -e '.[dev]'` is enough for the
core symbolic pipeline and tests. The `train` extra installs PyTorch and W&B.

## Optional Sage CAS Setup

The factorizer automatically looks for a Sage-capable Python at
`.cas_env/bin/python`. If it exists, multivariate factorization is handled by a
persistent helper process. If it does not exist, the code falls back to SymPy,
which is enough for tests and small local runs.

```bash
./scripts/setup_cas_env.sh
```

## Core Architecture

### Environment

`src/decomp_rl/decomp_env.py` defines `DecompEnv`. An episode starts with a
frontier containing the target polynomial. At each step:

1. The active frontier polynomial is selected.
2. The policy chooses an additive split `f = g + h`.
3. `FiniteFieldFactorizer` factors `g` and `h`.
4. Rebuild costs are charged for factored pieces.
5. unresolved child factors are added back to the frontier.
6. The reward is the cost saving against the current baseline estimate.

The environment memoizes solved subproblems, deduplicates frontier children, and
can add an extra shaping reward when a split piece hits the `FactorizableLibrary`.

### Policy/Value Model

`src/decomp_rl/model.py` contains the model interface and Torch MLP baseline.
The action space is variable-sized: each state exposes a fresh list of candidate
splits. For each candidate, the network receives handcrafted features

```text
[target_features, g_features, h_features]
```

and emits one policy logit. A separate value head predicts normalized expected
improvement for the target polynomial.

### Search

`src/decomp_rl/andor_search.py` implements memoized AND/OR search with
cost-minimizing PUCT selection. Nodes are polynomials, actions are additive
splits, and child nodes are unresolved factors produced after factoring the
split pieces. Search returns:

- the best cost found,
- the best decomposition trace,
- the root candidate list,
- root visit-count policy targets,
- value estimates and search/cache statistics.

### Factorizable Library

`src/decomp_rl/factor_library.py` stores polynomials with confirmed non-trivial
factorizations that save at least one operation over direct construction. The
library can match entries inside a target polynomial by exact term inclusion,
finite-field scalar multiples, and variable permutations. Matching can guide
split proposals, and exact library hits can shape RL reward.

### Baselines

`src/decomp_rl/baselines.py` defines `BaselineBundle`, used for reward shaping
and evaluation. It takes the minimum cost across:

- sparse direct construction,
- the simple baseline Horner upper bound,
- multivariate/gap-aware Horner,
- common-subexpression style shared powers,
- memoized top-down power-pivot search.

PPO can apply a terminal bonus when a completed episode beats this five-baseline
minimum.

## Training Modes

### Full ExIt-Style Training

The main training entrypoint is:

```bash
python scripts/run_full_experiment.py --help
```

The full loop is:

1. Generate synthetic curriculum examples from planted factorizations, Horner
   forms, elementary symmetric polynomials, and exact-small decompositions.
2. Train a supervised policy/value model from these traces.
3. Run `AndOrSearch` on curriculum targets using the current model as prior and
   value guide.
4. Distill root search policies and values back into supervised examples.
5. Mix recent search-distill examples, prioritized replay, elite traces, and
   fresh synthetic examples.
6. Continue training, evaluate holdout targets, log metrics, and checkpoint.

Useful outputs live under `artifacts/<run_id>/`:

- `config.json`
- `metrics.jsonl`
- `checkpoints/stage_a.pt`
- `checkpoints/cycle_NNN.pt`
- `checkpoints/best_holdout.pt`
- `checkpoints/final.pt`

Checkpoint payloads include model weights, metadata, timestamp, and, when
available, optimizer state for resume.

### PPO and PPO+MCTS Fine-Tuning

For a short PPO+MCTS run:

```bash
python scripts/run_ppo_finetune.py --use-mcts --iterations 10 --seed 0
```

Plain PPO samples from the network policy over the current candidate split set.
PPO+MCTS runs `AndOrSearch` at each action step, samples from the root visit
distribution, and adds a cross-entropy distillation term from MCTS visits to the
PPO loss.

Important flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--iterations` | `50` | PPO update iterations |
| `--rollouts-per-update` | `8` | Episodes before each update |
| `--candidates-per-step` | `16` | Split candidates per active polynomial |
| `--library-reward-weight` | `1.0` | Weight for factorizable-library reward |
| `--use-mcts` | off | Enable AlphaZero-style MCTS guidance |
| `--mcts-simulations` | `32` | PUCT simulations per action step |
| `--mcts-distill-coef` | `1.0` | MCTS policy distillation weight |
| `--checkpoint-in` | unset | Warm-start checkpoint |
| `--checkpoint-out` | `artifacts/ppo/finetuned.pt` | Output checkpoint |

## Evaluation

Evaluate checkpoints on a fixed 20-polynomial suite:

```bash
python scripts/evaluate_checkpoints.py \
  --checkpoint-dir artifacts/<run_id>/checkpoints \
  --search-sims 96
```

The evaluator reports greedy rollout and `AndOrSearch` costs for each
checkpoint. A run is counted as successful on a polynomial when its discovered
cost is less than or equal to the `BaselineBundle` minimum.

The full training loop also logs a uniform random split-policy evaluator. This
baseline samples random candidates from the same proposal set, rolls out for a
bounded number of split steps, then directly solves any remaining frontier
polynomials. Metrics appear in `metrics.jsonl` and W&B as `holdout_random/*`
and `cycle_random/*`; tune it with `--random-rollouts-per-target`,
`--random-rollout-max-steps`, and `--random-rollout-candidates`.

## W&B Logging

`scripts/run_full_experiment.py` logs to W&B when the `train` extra is installed
and W&B credentials are available. Defaults:

```text
entity:  p-agi
project: PolyArithmeticCircuitsRL
mode:    auto
```

For a cluster run, set one of these before submission:

```bash
export WANDB_API_KEY=<your-key>
# or run once in the same environment:
wandb login
```

The trainer writes local metrics regardless of W&B status. If W&B initialization
fails in `auto` mode, it falls back to offline/disabled behavior and writes a
warning file in the run directory.

## Slurm Training

The repository includes Slurm launchers for long runs:

- `scripts/run_hyak.slurm` for Hyak-style Slurm environments.
- `scripts/run_tilicum.slurm` for UW Tilicum. Tilicum uses QoS instead of
  Klone-style account/partition selection, and jobs must request at least one
  GPU.

Typical submission shape:

```bash
RUN_ID=decomp-rl-tilicum-001 \
OUT_DIR=/path/to/artifacts/decomp-rl-tilicum-001 \
WANDB_MODE=online \
WANDB_API_KEY=<your-key> \
WANDB_ENTITY=p-agi \
WANDB_PROJECT=PolyArithmeticCircuitsRL \
sbatch scripts/run_tilicum.slurm
```

Use a stable `OUT_DIR` for preemptible or requeued jobs. The Slurm launcher
auto-detects the newest `cycle_NNN.pt` in `OUT_DIR/checkpoints/` and resumes
from it.

The Tilicum launcher defaults to `--qos=normal`, `--gres=gpu:1`,
`--cpus-per-task=8`, `--mem=200G`, and `--time=24:00:00`. Set `WANDB_API_KEY`
or run `wandb login` before submitting with `WANDB_MODE=online`; otherwise the
job exits before training starts so missing W&B credentials are not missed.

## Repository Layout

- `src/decomp_rl/polynomial.py`: sparse polynomial representation.
- `src/decomp_rl/factor_fp.py`: finite-field factorization and Sage worker.
- `src/decomp_rl/factor_library.py`: factorizable polynomial library.
- `src/decomp_rl/split_proposals.py`: candidate split generation.
- `src/decomp_rl/decomp_env.py`: RL decomposition environment.
- `src/decomp_rl/andor_search.py`: memoized AND/OR PUCT search.
- `src/decomp_rl/model.py`: heuristic and Torch policy/value models.
- `src/decomp_rl/evaluate.py`: search, supervision, and random-rollout metrics.
- `src/decomp_rl/train_supervised.py`: supervised policy/value training.
- `src/decomp_rl/train_search_distill.py`: search distillation and elite traces.
- `src/decomp_rl/train_ppo.py`: PPO and PPO+MCTS fine-tuning.
- `scripts/run_full_experiment.py`: main end-to-end training loop.
- `scripts/evaluate_checkpoints.py`: fixed-suite checkpoint evaluation.
- `tests/`: regression tests for the symbolic, search, and training stack.
