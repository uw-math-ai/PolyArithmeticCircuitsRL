# Slurm Benchmark Jobs

These scripts run `learn-guided-search` benchmark sweeps on CPU nodes. They
support W&B logging and intentionally avoid containers, GPUs, old `gumbel`
logic, and PID waiting.

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

Planner sweeps write:

```text
planner_sweep_rows.jsonl
planner_sweep_summary.json
planner_sweep_deltas.json
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
- `planner_sweep.sbatch`: runs the matched-budget four-method planner sweep:
  heuristic Beam, guided Beam, heuristic Gumbel, and guided Gumbel when a
  checkpoint is provided.
- `array_sweep.sbatch`: dispatches multiple independent sweep jobs from a TSV
  matrix using `SLURM_ARRAY_TASK_ID`.

SBATCH headers use CPU defaults: account `stf`, partition `ckpt`, one node, one
task, and no GPU. Override Slurm resources with `sbatch` flags if the cluster
requires different account, partition, time, memory, or CPU settings. Do not
request a GPU unless a future model actually needs it.

The scripts bootstrap Python inside the Slurm job. They first try to load a Hyak
Python module, then create/reuse a repo-local venv, and install this package,
W&B, and PyTorch if they are missing.

Python selection order:

1. `$HYAK_PYTHON_MODULE`, if set, through `module load`
2. auto module attempts: `python/3.12`, `python/3.11`, `python/3.10`, `python`
3. `$PYTHON_BIN`, if set
4. `python3.13`, `python3.12`, `python3.11`, `python3.10`, `python3`, `python`

Python 3.10+ is required. The default venv is:

```text
.venv-hyak-<python-binary-name>/
```

Useful overrides:

```bash
HYAK_PYTHON_MODULE=python/3.11.4   # exact module if Hyak's default name differs
VENV_DIR=/gscratch/scrubbed/$USER/lgs-venv
BOOTSTRAP_VENV=0                   # use selected Python directly; deps must already exist
PIP_CACHE_DIR=/gscratch/scrubbed/$USER/pip-cache
```

W&B defaults vary by wrapper. The train/eval and benchmark-sweep wrappers enable
it by default; `planner_sweep.sbatch` defaults to `ENABLE_WANDB=0` so local
pre-flight planner runs do not require W&B credentials. Override with:

```bash
ENABLE_WANDB=1
WANDB_PROJECT=PolyArithmeticCircuitsRL
WANDB_ENTITY=zengrf-university-of-washington
WANDB_MODE=online
```

Set `ENABLE_WANDB=0` to disable it, or override `WANDB_PROJECT`, `WANDB_ENTITY`,
or `WANDB_MODE`.

## Train And Evaluate

Moderate default:

```bash
sbatch slurm/train_eval_benchmark.sbatch
```

Override settings:

```bash
RUN_NAME=server_run_001 \
ENABLE_WANDB=1 \
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
ENABLE_WANDB=1 \
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
ENABLE_WANDB=1 \
LAMBDA_MODEL=1.0 \
MAX_INSTANCES_PER_FAMILY=100 \
BEAM_WIDTHS=1,2,4,8,16 \
CANDIDATE_KS=4,8,16,32,64 \
TIER2_MS=64,128 \
sbatch slurm/guided_sweep.sbatch
```

## Matched-Budget Planner Sweep

Heuristic-only Beam/Gumbel planner sweep:

```bash
RUN_NAME=planner_heuristic_001 \
sbatch slurm/planner_sweep.sbatch
```

Guided plus heuristic four-method sweep from a checkpoint:

```bash
RUN_NAME=planner_guided_001 \
CHECKPOINT=results/server_runs/server_train_eval_full_001/ranker.pt \
ENABLE_WANDB=1 \
sbatch slurm/planner_sweep.sbatch
```

Moderate pre-flight server sweep:

```bash
RUN_NAME=planner_sweep_medium_001 \
CHECKPOINT=results/server_runs/server_train_eval_full_001/ranker.pt \
ENABLE_WANDB=1 \
MAX_INSTANCES_PER_FAMILY=10 \
BEAM_WIDTHS=1,2,4 \
CANDIDATE_KS=4,8,16 \
TIER2_MS=64 \
EXPANSION_BUDGETS=64,128 \
GUMBEL_SEEDS=0,1,2,3,4 \
LAMBDA_MODEL=1.0 \
sbatch slurm/planner_sweep.sbatch
```

Full sweep template:

```bash
RUN_NAME=planner_sweep_full_001 \
CHECKPOINT=results/server_runs/server_train_eval_full_001/ranker.pt \
ENABLE_WANDB=1 \
MAX_INSTANCES_PER_FAMILY=25 \
BEAM_WIDTHS=1,2,4,8 \
CANDIDATE_KS=4,8,16,32 \
TIER2_MS=64,128 \
EXPANSION_BUDGETS=64,128,256 \
GUMBEL_SEEDS=0,1,2,3,4,5,6,7,8,9 \
LAMBDA_MODEL=1.0 \
sbatch slurm/planner_sweep.sbatch
```

The first deltas to inspect in `planner_sweep_deltas.json` are:

```text
gumbel_heuristic - beam_heuristic
beam_guided - beam_heuristic
gumbel_guided - gumbel_heuristic
gumbel_guided - beam_guided
```

These are grouped by family, intended complexity, beam width, candidate count,
tier-2 count, and expansion budget.

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
- W&B logging is controlled by `ENABLE_WANDB`; no W&B import happens when it is disabled.
- There is no container, PID waiting, or old `gumbel` integration.
- `run_metadata.txt` records the git commit/status, Python/package versions,
  W&B settings, full command, run parameters, exit code, elapsed seconds, and
  generated files.
