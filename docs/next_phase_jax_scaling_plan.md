# Next-Phase JAX And Systems Plan

This note records the current bottlenecks in the decomposition-search project
and the recommended next implementation wave while the GPU is occupied by
another job.

## Current State

The repo already has:

- symbolic finite-field polynomial arithmetic,
- CAS-backed factorization via Sage,
- split proposals and AND/OR search,
- heuristic and learned policy/value scoring,
- supervised warm start,
- search distillation,
- prioritized replay and elite self-imitation,
- curriculum generation and evaluation scripts,
- detached launchers and checkpoint resume from cycle-0 checkpoints.

The main training issue is no longer missing functionality. It is systems
throughput.

## Current Bottlenecks

### 1. CPU/CAS startup dominates wall-clock time

Initial supervised data generation, holdout generation, synthetic refreshes, and
search-distillation targets all use Python + Sage factorization on CPU. The GPU
often sits idle before stage-A or between learner updates.

### 2. Learner and search share one process

The current `run_full_experiment.py` interleaves:

- generate targets,
- run search,
- build replay batches,
- run learner updates,
- evaluate,

inside one Python process. This is simple but underfeeds the GPU.

### 3. Model scoring is not served as a true batched inference service

Search calls into the model inline. Even though the learner now uses packed
batches, search-time network scoring is still not aggregated across many CPU
workers.

### 4. Torch remains a light baseline, not a throughput-oriented learner stack

The current Torch MLP is useful for validation, but it does not yet provide:

- shape-stable compiled training steps,
- large asynchronous device queues,
- actor/learner separation,
- aggressive batch bucketing,
- efficient batched inference serving for search.

### 5. Several research-plan items are still missing

The highest-value missing scientific items are:

- final DAG / CSE scoring,
- frontier teacher forcing and optional learned frontier selector,
- auxiliary heads and auxiliary losses,
- value calibration,
- search-vs-policy and memoization ablations,
- optional PPO/SAC only after the above are stable.

## Recommendation: Dual-Stack Architecture

Do **not** port the symbolic search and CAS logic to JAX first.

Recommended split:

- keep symbolic search, Sage, generators, and exact cost logic in Python,
- add a JAX/Flax learner + inference plane,
- communicate between CPU search workers and the GPU learner through fixed-shape
  batched tensors and queue-based RPC-like interfaces.

This keeps the hard symbolic logic where it already works and uses JAX exactly
where it brings the biggest win: dense batched learning and inference.

## Recommended Next-Phase Architecture

### A. Actor / learner split

Create two subsystems:

1. `actor/search workers` on CPU
2. `learner/inference server` on GPU

Actors are responsible for:

- curriculum target generation,
- Sage-backed factorization,
- AND/OR search,
- trace extraction,
- replay / elite record emission.

The learner server is responsible for:

- batched policy/value/aux inference,
- JAX `jit`-compiled updates,
- checkpointing,
- parameter publication to actors.

### B. Fixed-shape batch schema

JAX performs best with stable shapes. The learner interface should therefore use
fixed-shape padded arrays instead of Python objects.

Recommended batch objects:

- `target_features`: `(B, T)`
- `candidate_features`: `(B, K, C)`
- `candidate_mask`: `(B, K)`
- `policy_targets`: `(B, K)`
- `value_targets`: `(B,)`
- optional `frontier_features`: `(B, F, H)`
- optional auxiliary labels:
  - `branch_factorable`: `(B,)`
  - `saving_bucket`: `(B, S)`

Use a small set of candidate-count buckets rather than one globally maximal
shape. Example buckets: `K in {8, 16, 32}`.

### C. JAX learner stack

Use:

- `jax`
- `flax`
- `optax`

Core learner components:

- `JAXPolicyValueNetwork`
- `TrainState`
- `jit`-compiled `train_step`
- `jit`-compiled `infer_step`
- bf16 activations / parameters where numerically safe
- prefetch-to-device queues

### D. Batched inference server

Search workers should not score candidates one target at a time. Instead:

- push candidate-scoring requests into an inference queue,
- aggregate requests for a short interval or until full,
- pad/bucket them,
- run one `jax.jit` / `jax.vmap` inference pass,
- return priors and values back to workers.

This is the main change needed if the goal is to keep the GPU busy during
search-distillation rather than only during learner updates.

### E. Bounded Sage worker pool

The factorizer leak bug has been fixed, but the next architecture should still
avoid unbounded process creation.

Implement a bounded host-side factorization pool:

- fixed worker count,
- explicit checkout / return,
- per-worker health checks,
- metrics for queue depth and restart count.

### F. Scientific extensions to add in the same wave

1. final DAG / CSE scorer
2. frontier teacher forcing
3. optional learned frontier selector
4. auxiliary heads:
   - factorable-branch prediction
   - immediate-saving bucket
5. value calibration evaluation
6. ablation runners

These are all compatible with the actor/learner split and should be scaffolded
alongside it.

## How JAX Should Be Used Here

### What to move to JAX now

- policy/value/aux network
- learner updates
- batched scoring for search
- replay minibatch processing
- optional frontier selector inference and training

### What to keep in Python/CAS for now

- sparse polynomial canonicalization
- split proposal generation
- Sage factorization
- exact rebuild and baseline costs
- AND/OR tree bookkeeping
- DAG / CSE symbolic scoring

### What not to do in the next wave

Do not attempt all of the following at once:

- full JAX rewrite of search,
- full JAX rewrite of the environment,
- CAS-in-JAX emulation,
- PPO-from-scratch before the search-distill path is stable.

That would be high-risk and would delay useful experiments.

## Aggressive GPU Utilization Plan

To actually consume a large GPU, the next run needs more than a larger model.

### 1. Separate producers from learner

Keep CPU actors generating data continuously while the learner updates on the
latest replay shards.

### 2. Increase update-to-data ratio

Per actor round, run multiple learner steps over:

- recent distill data,
- replay data,
- elite data,
- synthetic refresh data.

### 3. Cache packed data on device

This has already been partially added for Torch. The JAX version should:

- pack once,
- bucket by static shape,
- prefetch onto device,
- run many `jit` updates without repacking.

### 4. Batch inference aggressively

If search workers do not batch model queries, GPU use during distillation will
remain low even with a JAX learner.

### 5. Use free-memory-aware sizing

The batch auto-sizer should consider currently free memory, not just total GPU
memory. This matters when other jobs are already on the GPU.

### 6. Keep CPU reserved for SSH and system stability

Continue reserving a small set of CPU cores for SSH / system tasks while giving
search workers the remaining cores.

## Proposed Package Skeleton

When scaffolding the next implementation wave, add:

- `src/decomp_rl/jax_backend/config.py`
- `src/decomp_rl/jax_backend/types.py`
- `src/decomp_rl/jax_backend/network.py`
- `src/decomp_rl/jax_backend/train_state.py`
- `src/decomp_rl/jax_backend/learner.py`
- `src/decomp_rl/jax_backend/inference_server.py`
- `src/decomp_rl/jax_backend/batching.py`
- `src/decomp_rl/jax_backend/replay_dataset.py`
- `src/decomp_rl/jax_backend/frontier_head.py`
- `src/decomp_rl/dag_scoring.py`
- `src/decomp_rl/ablations.py`
- `scripts/run_full_experiment_jax.py`
- `scripts/run_ablation_suite.py`

## Immediate Consultation Decision

There are two sensible ways to proceed.

### Option A: Recommended

Build a **dual-stack** system:

- current Python/Sage search remains,
- JAX handles learner and batched inference,
- next run is still search-distillation-first,
- PPO remains optional later.

### Option B: Higher risk

Start a much more ambitious rewrite:

- JAX environment representation,
- JAX-native batched search loop,
- early PPO/MCTS integration.

This may eventually be cleaner, but it is a larger rewrite and a slower path to
the next scientifically useful run.

## Recommendation

Choose **Option A** first.

It directly addresses the current observed bottlenecks without destabilizing the
symbolic core that is already working.
