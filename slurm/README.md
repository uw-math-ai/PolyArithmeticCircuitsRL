# Slurm Benchmark Jobs

These scripts run `learn-guided-search` benchmark sweeps on CPU nodes. They
intentionally avoid W&B, containers, GPUs, old `gumbel` logic, and any external
logging service.

Outputs go to:

```text
results/slurm/
results/server_runs/<run_name>/
```

Each run writes:

```text
run_metadata.txt
run.log
sweep_rows.jsonl
sweep_summary.json
sweep_failures.json
```

Train/eval runs also write:

```text
ranker.pt
bootstrap_metrics.json
```

## Scripts

- `train_eval_benchmark.sbatch`: trains through the bootstrap loop, evaluates
  heuristic and guided search, and saves a checkpoint.
- `heuristic_sweep.sbatch`: runs heuristic-only sweeps with no checkpoint.
- `guided_sweep.sbatch`: runs guided sweeps from a saved `ranker.pt`.
- `array_sweep.sbatch`: dispatches multiple independent sweep jobs from a TSV
  matrix using `SLURM_ARRAY_TASK_ID`.

SBATCH headers use CPU defaults: account `stf`, partition `ckpt`, one node, one
task, and no GPU. Override Slurm resources with `sbatch` flags if the cluster
requires different account, partition, time, memory, or CPU settings. Do not
request a GPU unless a future model actually needs it.

The scripts find Python in this order:

1. `$PYTHON_BIN`
2. `.venv/bin/python`
3. `python3`
4. `python`

Python 3.10+ is required. Jobs do not install dependencies.

## Train And Evaluate

Moderate default:

```bash
sbatch slurm/train_eval_benchmark.sbatch
```

Override settings:

```bash
RUN_NAME=server_run_001 \
MAX_INSTANCES_PER_FAMILY=100 \
ROUNDS=8 \
EPOCHS_PER_ROUND=50 \
BEAM_WIDTHS=1,2,4,8,16 \
CANDIDATE_KS=4,8,16,32,64 \
TIER2_MS=64,128 \
sbatch slurm/train_eval_benchmark.sbatch
```

The trained checkpoint will be:

```text
results/server_runs/server_run_001/ranker.pt
```

## Heuristic-Only Sweep

```bash
sbatch slurm/heuristic_sweep.sbatch
```

With overrides:

```bash
RUN_NAME=heuristic_full_001 \
MAX_INSTANCES_PER_FAMILY=100 \
BEAM_WIDTHS=1,2,4,8,16 \
CANDIDATE_KS=4,8,16,32,64 \
TIER2_MS=64,128 \
sbatch slurm/heuristic_sweep.sbatch
```

## Guided Sweep

```bash
CHECKPOINT=results/server_runs/server_run_001/ranker.pt \
sbatch slurm/guided_sweep.sbatch
```

With overrides:

```bash
RUN_NAME=guided_full_001 \
CHECKPOINT=results/server_runs/server_run_001/ranker.pt \
LAMBDA_MODEL=1.0 \
MAX_INSTANCES_PER_FAMILY=100 \
BEAM_WIDTHS=1,2,4,8,16 \
CANDIDATE_KS=4,8,16,32,64 \
TIER2_MS=64,128 \
sbatch slurm/guided_sweep.sbatch
```

## Array Sweep

Create a matrix file such as `slurm/sweep_matrix.tsv`:

```text
# run_name	mode	checkpoint	max_instances_per_family	beam_widths	candidate_ks	tier2_ms	lambda_model
heuristic_bw1	heuristic	-	100	1	4,8,16,32,64	64,128	0.0
heuristic_bw2	heuristic	-	100	2	4,8,16,32,64	64,128	0.0
guided_bw1	guided	results/server_runs/server_run_001/ranker.pt	100	1	4,8,16,32,64	64,128	1.0
train_eval_small	train_eval	-	25	1,2,4	4,8,16	64	0.0
```

Then submit:

```bash
sbatch --array=1-4 slurm/array_sweep.sbatch
```

Use a custom matrix path with:

```bash
MATRIX_FILE=slurm/my_matrix.tsv sbatch --array=1-4 slurm/array_sweep.sbatch
```

For `train_eval` rows, the checkpoint column can be `-`; `ROUNDS`,
`EPOCHS_PER_ROUND`, and `SEED` remain environment-overridable.

## Inspect Jobs

```bash
squeue -u $USER
tail -f results/slurm/<job_name>_<job_id>.out
tail -f results/server_runs/<run_name>/run.log
cat results/server_runs/<run_name>/run_metadata.txt
```

If the global pytest environment has plugin conflicts, use:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
```

## Notes

- The Slurm wrappers are CPU-only by default.
- There is no W&B, container, PID waiting, or old `gumbel` integration.
- `run_metadata.txt` records the git commit/status, Python/package versions,
  full command, run parameters, exit code, elapsed seconds, and generated files.
