# Experiments

The central experimental question is:

```text
Does learned ranking improve symbolic beam search over heuristic-only search
on harder structured polynomial targets under fixed search budgets?
```

The key comparison is:

- heuristic-only beam search
- heuristic plus learned-ranker beam search

## Test Command

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
```

Use the plugin autoload flag when the global environment contains pytest plugins that conflict with the local project environment.

## Smoke Sweep

Run a tiny heuristic-only sweep:

```bash
python scripts/run_benchmark_sweep.py \
  --max-instances-per-family 1 \
  --beam-widths 1 \
  --candidate-ks 4 \
  --tier2-ms 16
```

This is a quick sanity check, not a meaningful benchmark.

## Train And Evaluate

Run a tiny train-and-evaluate benchmark:

```bash
python scripts/train_and_eval_benchmark.py \
  --rounds 1 \
  --max-instances-per-family 1 \
  --epochs-per-round 3 \
  --beam-widths 1 \
  --candidate-ks 4 \
  --tier2-ms 16
```

Run a larger local or server-scale experiment:

```bash
python scripts/train_and_eval_benchmark.py \
  --rounds 5 \
  --max-instances-per-family 100 \
  --epochs-per-round 50 \
  --beam-widths 1,2,4,8,16 \
  --candidate-ks 4,8,16,32,64 \
  --tier2-ms 64,128 \
  --output-dir results/server_run_001
```

## Heuristic-Only Sweep

```bash
python scripts/run_benchmark_sweep.py \
  --max-instances-per-family 100 \
  --beam-widths 1,2,4,8,16 \
  --candidate-ks 4,8,16,32,64 \
  --tier2-ms 64,128 \
  --output-dir results/heuristic_full_001
```

## Guided Sweep From Checkpoint

```bash
python scripts/run_benchmark_sweep.py \
  --checkpoint results/server_run_001/ranker.pt \
  --lambda-model 1.0 \
  --max-instances-per-family 100 \
  --beam-widths 1,2,4,8,16 \
  --candidate-ks 4,8,16,32,64 \
  --tier2-ms 64,128 \
  --output-dir results/guided_full_001
```

## Slurm

Server scripts live in `slurm/`.

```bash
sbatch slurm/train_eval_benchmark.sbatch
sbatch slurm/heuristic_sweep.sbatch
CHECKPOINT=results/server_runs/server_run_001/ranker.pt sbatch slurm/guided_sweep.sbatch
```

See [../slurm/README.md](../slurm/README.md) for run metadata, array sweeps, and output layout.

## Output Files

Sweep outputs:

- `sweep_rows.jsonl`: one row per `(method, instance, beam_width, candidate_k, tier2_m)` run
- `sweep_summary.json`: grouped aggregate metrics
- `sweep_failures.json`: failed rows grouped by family, complexity, method, and budget

Train-and-eval outputs also include:

- `ranker.pt`: saved model checkpoint and encoder feature schema
- `bootstrap_metrics.json`: per-round preference and validation metrics

Slurm wrappers additionally write:

- `run_metadata.txt`
- `run.log`

## Metrics

Per-row metrics:

- `success`: whether exact target construction succeeded
- `best_ops`: operation count for the smallest verified finished circuit, or `null` on failure
- `expansions`: number of expanded candidate records
- `runtime_sec`: wall-clock runtime for that search

Summary metrics:

- `solve_rate`: solved instances divided by total instances in the group
- `avg_best_ops`: average best operation count over solved instances only
- `avg_expansions`: average expansions over all instances
- `median_expansions`: median expansions over all instances
- `avg_runtime_sec`: average runtime over all instances
- `guided_minus_heuristic_solve_rate`: guided solve rate minus heuristic solve rate for matching `(family, beam_width, candidate_k, tier2_m)` groups
- `solved_by_complexity`: solve rate grouped by intended target complexity

## How To Interpret Results

Start with tight budgets:

```text
beam_width in {1, 2}
candidate_k in {4, 8, 16}
```

These settings expose whether ranking matters. If both methods solve everything, the benchmark or budget is too easy. If both fail similarly, inspect whether the necessary candidate appears in the heuristic top-K. If it does not appear, candidate generation is the bottleneck. If it appears but beam search drops delayed-good branches, value-guided search or MCTS may be the next appropriate extension.

The first paper-relevant result is not absolute solve rate. It is whether learned ranking improves success, expansions, or circuit size under fixed, fair search budgets.
