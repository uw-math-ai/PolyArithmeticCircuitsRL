# Decomposition-Search RL for Arithmetic Circuit Discovery

This repository contains an MVP implementation of the split-based arithmetic
circuit discovery program described in
[`AlphaZero RL Circuit Discovery Plan.md`](./AlphaZero%20RL%20Circuit%20Discovery%20Plan.md).

The current implementation focuses on the early phases of the roadmap:

- sparse finite-field polynomial arithmetic with canonical hashing,
- finite-field factorization through a CAS-backed Sage worker with SymPy fallback,
- baseline and rebuild cost models,
- split proposals, decomposition environment, and memoized AND/OR search,
- synthetic trace generators and lightweight training/evaluation utilities.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## CAS Environment

The factorizer auto-detects a Sage CAS environment at `.cas_env/bin/python` and
uses a persistent helper subprocess for multivariate factorization. If that
environment is unavailable, the code falls back to the SymPy prototype backend.

To create the Sage environment locally:

```bash
./scripts/setup_cas_env.sh
```

## Scripts

- `scripts/generate_pretrain_dataset.py`
- `scripts/queue_strategic_fixed_prime_v10_resume_nohup.sh`
- `scripts/start_strategic_fixed_prime_v10_resume_nohup.sh`
- `scripts/start_strategic_fixed_prime_v9_nohup.sh`
- `scripts/start_strategic_fixed_prime_v8_nohup.sh`
- `scripts/launch_strategic_fixed_prime_v7.sh`
- `scripts/run_supervised_pretrain.py`
- `scripts/run_search_distill.py`
- `scripts/run_full_experiment.py`
- `scripts/run_training_smoke.py`
- `scripts/run_eval_suite.py`
- `scripts/demo_environment.py`

The training stack defaults to a heuristic policy/value model so the symbolic
pipeline works immediately. Install the optional `train` extra to experiment
with the Torch MLP baseline in `src/decomp_rl/model.py`.

## Repo Layout

- `src/decomp_rl/`: core library code for polynomials, factorization, search, generators, and training
- `scripts/run_full_experiment.py`: main end-to-end training entrypoint
- `scripts/start_strategic_fixed_prime_v10_resume_nohup.sh`: resume from the last strong stage-A checkpoint and relaunch detached
- `scripts/queue_strategic_fixed_prime_v10_resume_nohup.sh`: wait for GPU memory to free up, then launch the resumed detached run
- `scripts/start_strategic_fixed_prime_v8_nohup.sh`: validated detached GPU launcher
- `artifacts/<run_id>/`: logs, config, checkpoints, and metrics for each run

## Detached GPU Training

The validated detached-launch pattern on this machine is:

```bash
nohup setsid /bin/bash -lc 'cd /home/ec2-user/Polynomial2; exec env ... .venv/bin/python scripts/run_full_experiment.py ...' > artifacts/<run_id>/nohup.log 2>&1 </dev/null &
```

Plain `nohup ... &` was not sufficient in this environment; the trainer would
exit immediately after the parent shell returned. Using `nohup` together with
`setsid` keeps the training process alive and produces a reliable log file.

For the current fixed-prime large run, use the checked-in launcher:

```bash
./scripts/start_strategic_fixed_prime_v10_resume_nohup.sh
```

It writes the detached process id to `artifacts/<run_id>/nohup.pid` and the
live log to `artifacts/<run_id>/nohup.log`.

The runner also supports checkpoint resume directly:

```bash
HOME=/home/ec2-user/Polynomial2/.sage_home \
XDG_CACHE_HOME=/home/ec2-user/Polynomial2/.sage_home/.cache \
.venv/bin/python scripts/run_full_experiment.py \
  --resume-checkpoint artifacts/<old_run>/checkpoints/stage_a.pt \
  --output-dir artifacts/<new_run> ...
```

The current resume path is exact for `stage_a.pt` / `best_holdout.pt` from
cycle 0 because the initial supervised set, holdout set, and replay bootstrap
are regenerated deterministically from the saved config. Resuming from later
cycle checkpoints reloads model weights but rebuilds replay and elite state
approximately.

W&B is now configured to default to:

```text
p-agi/PolyArithmeticCircuitsRL
```

The detached `v10` launcher reads credentials from `/home/ec2-user/.netrc`, so
it should come up online automatically as long as that file contains a valid
API key.

If the GPU is busy with another job, queue the detached relaunch instead of
starting immediately:

```bash
BLOCKING_PID=<other_gpu_pid> \
MIN_FREE_MIB=70000 \
./scripts/queue_strategic_fixed_prime_v10_resume_nohup.sh
```

For an interactive foreground run, the canonical entrypoint is:

```bash
HOME=/home/ec2-user/Polynomial2/.sage_home \
XDG_CACHE_HOME=/home/ec2-user/Polynomial2/.sage_home/.cache \
.venv/bin/python scripts/run_full_experiment.py ...
```

Cycle search is now run with a fresh search/factorizer instance per target plus
target-level progress logging, which avoids silent state buildup across the
whole target batch and makes failures much easier to localize. Training batches
are also packed once per phase instead of rebuilding feature tensors every
epoch, and the trainer can cache that packed dataset on GPU to reduce host-side
stalling during larger CUDA runs.
