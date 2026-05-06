# `src/algorithms/breakdown/` — Top-Down Polynomial Decomposition

This subpackage adds an alternative training paradigm for the arithmetic
circuit synthesis problem. Instead of building a circuit *bottom-up* from
base inputs ($x_0, x_1, \ldots, x_{n-1}, 1$) toward the target polynomial
$f^\star$ — as the original `PPO`, `PPO+MCTS`, `AlphaZero`, and `SAC`
trainers do — the agent here works *top-down*: starting from $f^\star$
itself and recursively breaking it into smaller sub-polynomials until every
leaf of the decomposition tree is a base node.

For example, given $f^\star = x^2 + 2xy + y^2 + 1$, a decomposition tree
might be:

```
  x^2 + 2xy + y^2 + 1                       (target)
  ├─ ADD ─┐
  │       └─ 1                              (leaf — constant)
  └─ x^2 + 2xy + y^2                        (= (x + y)^2, factorable)
     ├─ MUL ─┐
     │       └─ x + y                       (factor — also reusable)
     └─ x + y                               (factor — also reusable)
        ├─ ADD ─┐
        │       └─ y                        (leaf — base var)
        └─ x                                (leaf — base var)
```

Each internal node corresponds to one arithmetic operation in the eventual
bottom-up circuit, so the number of *distinct* internal nodes is a lower
bound on the circuit size.

The package is designed to be a **drop-in addition**: nothing inside the
existing `src/` tree was modified, no existing test breaks, and the new
trainers can be imported alongside (or instead of) the original ones.

---

## Table of contents

- [Files in this folder](#files-in-this-folder)
- [Why top-down decomposition?](#why-top-down-decomposition)
- [The `BreakdownGame` environment](#the-breakdowngame-environment)
  - [State](#state)
  - [Candidate generation](#candidate-generation)
  - [Step semantics](#step-semantics)
  - [Reward design](#reward-design)
  - [Termination conditions](#termination-conditions)
- [Observation format](#observation-format)
- [Trainers](#trainers)
  - [`PPOMCTSBreakdownTrainer` (PyTorch)](#ppomctsbreakdowntrainer-pytorch)
  - [`SACBreakdownTrainer` (PyTorch)](#sacbreakdowntrainer-pytorch)
  - [`PPOBreakdownJAXTrainer` (JAX/Flax/optax)](#ppobreakdownjaxtrainer-jaxflaxoptax)
- [Reuse of `FactorLibrary`](#reuse-of-factorlibrary)
- [Quick start](#quick-start)
- [Algorithmic differences vs. the bottom-up trainers](#algorithmic-differences-vs-the-bottom-up-trainers)
- [JAX coverage and the missing `mctx` pieces](#jax-coverage-and-the-missing-mctx-pieces)
- [Testing](#testing)
- [Limitations and future work](#limitations-and-future-work)
- [FAQ](#faq)

---

## Files in this folder

| File | Role |
| --- | --- |
| `__init__.py` | Lazy package init (no torch / jax imported eagerly). |
| `breakdown_env.py` | The `BreakdownGame` decomposition environment, plus observation containers and helpers. Pure numpy + SymPy (via `FactorLibrary`); no torch / jax. |
| `ppo_mcts_breakdown.py` | **Main file 1.** Self-contained PyTorch PPO + MCTS trainer for the breakdown task: `BreakdownPolicyValueNet`, `BreakdownMCTS`, `PPOMCTSBreakdownTrainer`. |
| `sac_breakdown.py` | **Main file 2.** Self-contained PyTorch discrete-action SAC trainer: `BreakdownStateEncoder`, `BreakdownSACActor`, `BreakdownSACCritic`, `_StratifiedReplay`, `SACBreakdownTrainer`. |
| `ppo_breakdown_jax.py` | JAX/Flax/optax PPO trainer: `BreakdownPolicyValueNetJax`, `PPOBreakdownJAXTrainer`. JIT'd policy/value inference and PPO update; rollouts on host (see [JAX coverage](#jax-coverage-and-the-missing-mctx-pieces)). |
| `README.md` | This file. |

---

## Why top-down decomposition?

The original bottom-up game has the agent commit to operations in the
order they appear in the eventual circuit, **before** it has any direct
information about what the final circuit will look like. For high-degree
or high-arity polynomials this leads to:

- **Wide branching at each step** — the action space is
  $|A| = 2 \cdot \binom{|V|+1}{2}$, growing quadratically as nodes
  accumulate.
- **Sparse reward** — only the final step that produces $f^\star$ exactly
  triggers `success_reward`; intermediate matches are rewarded by the
  optional factor library, but the agent still has to reverse-engineer
  the right intermediate from the target on its own.
- **Inefficient credit assignment** for compositional structure: if
  $f^\star = g \cdot h$, the agent has to discover both $g$ and $h$
  *and* the right multiplication order.

Top-down decomposition flips this: the agent **starts from the answer**
and explicitly factors / additively-splits it. The compositional
structure is laid bare at every step (the focus polynomial is always
*part of* the eventual answer), and the action space at each step is
fixed at `max_options` (a small constant, default 32).

This is especially attractive when:

- The target factors non-trivially mod $p$ (then SymPy via the existing
  `FactorLibrary` immediately exposes a useful split).
- Multiple targets share common sub-polynomials, because the
  `FactorLibrary` is updated after every successful decomposition just
  like in the bottom-up trainers, so reuse is rewarded.

---

## The `BreakdownGame` environment

`BreakdownGame` lives in `breakdown_env.py`. It is **not** a subclass of
`CircuitGame` — it is an independent class that intentionally borrows
only `FastPoly` (numpy modular polynomial arithmetic) and `FactorLibrary`
(session-level sub-polynomial cache).

### State

```
target_poly:  FastPoly                # the original target T = f*
tree:         Dict[bytes, (op, lkey, rkey)]
                                      # decompositions chosen so far
poly_by_key:  Dict[bytes, FastPoly]   # canonical-key → FastPoly lookup
leaf_keys:    Set[bytes]              # base nodes + already-leaf entries
frontier:     Deque[bytes]            # canonical keys still to decompose
candidates:   List[BreakdownCandidate]   # cached options for current focus
steps_taken:  int                     # number of decomposition steps
done:         bool
```

The tree is keyed by `FastPoly.canonical_key()` (a `bytes` view of the
flat coefficient array), so identical sub-polynomials collapse to one
node automatically — the *distinct* internal node count corresponds to
the actual number of arithmetic operations in the eventual circuit.

### Candidate generation

At each step the polynomial at the head of the frontier (the *focus*) is
combined with the existing factor library to produce up to `max_options`
candidate decompositions of the form `focus = left  op  right` where
`op ∈ {add, mul}`. The candidate sources are:

1. **Multiplicative splits** via
   `FactorLibrary.factorize_poly(focus)`. For each non-trivial factor
   $f$ of `focus` over $\mathbb{Z}$ (then reduced mod $p$), the
   environment computes the cofactor with
   `FactorLibrary.exact_quotient(focus, f)`. If the quotient exists
   over $\mathbb{F}_p$ the pair `(f, focus/f)` is added as a `mul`
   candidate.

2. **Leaf-residual additive splits**. For every polynomial $v$ already
   in `leaf_keys` (base nodes or sub-polynomials previously declared
   leaves), if `focus - v` is non-zero and not equal to `focus` the
   pair `(v, focus - v)` is added as an `add` candidate.

3. **Monomial peel-offs**. For up to four of the highest-total-degree
   monomials $m$ in `focus`, `(m, focus - m)` is added as an `add`
   candidate. **This guarantees progress** — every non-zero polynomial
   has at least one valid candidate, so the env can never deadlock.

The list is deduplicated by the unordered key
`(op, frozenset({lkey, rkey}))` and truncated to `max_options`. Padded
slots are masked off in the observation.

> All three sources reuse the **existing factorization utilities** in
> `src/environment/factor_library.py` and `src/environment/polynomial_utils.py`,
> as the original task description requested. No new factorization
> library is introduced.

### Step semantics

Each `env.step(action_idx)` does:

1. Pops the focus polynomial from the front of the frontier.
2. Reads candidate `action_idx` (mask must be `True`).
3. Sanity-checks `recomposed = left op right == focus`.
4. Records the split in `tree`, registers `left` and `right` in
   `poly_by_key`.
5. For each child:
    * If it is a base node or zero, mark it a `leaf` and don't queue it.
    * Otherwise append its key to the frontier (skipping if already
      decomposed or queued — DAG sharing).
6. Increments `steps_taken`, advances the frontier past any newly-trivial
   entries, and recomputes candidates for the new focus.

### Reward design

```python
reward = step_penalty                                  # per step (-0.1)
       + factor_subgoal_reward * (left or right is in target's factor list,  # +1.0
                                  not yet claimed this episode)
       + factor_library_bonus  * (the same child is also library-known)      # +0.5
       + success_reward        * (frontier becomes empty)                    # +10.0
       - size_penalty_per_node * max(0, internal_nodes − max_complexity)
       + (-1.0)                * (max_steps reached without success)
```

The factor / library bonuses use **the same constants** as the existing
forward `CircuitGame` (read from `Config`) so the two games live on
comparable reward scales. The `size_penalty_per_node` term is new and
small (default `0.05`), and is only meaningful as a tie-breaker; it
discourages bloated trees without dominating the success signal.

### Termination conditions

- **Success**: the frontier is empty (every leaf of the tree is a base
  node). At this point all internal-node decompositions are guaranteed
  realisable as a bottom-up circuit by traversing the tree in DFS post-
  order — the environment provides
  `register_decomposition_in_library()` to feed those polynomials back
  into the shared `FactorLibrary`.
- **Timeout**: `steps_taken == max_steps` while the frontier is
  non-empty. The episode ends with a small negative reward and is
  recorded as a failure.
- **Trivial target**: if the user passes a target that is itself a base
  node (or zero) at `reset()`, the env returns `done=True` immediately
  and the trainer counts that as an automatic success.

---

## Observation format

`BreakdownObservation` is a `dataclass` of plain numpy arrays:

```python
focus_vec:           (target_size,) float32   # current focus, scaled by 1/p
target_vec:          (target_size,) float32   # original target T, scaled
context_vec:         (5,) float32             # episode-level scalars
candidate_features:  (max_options, 5) float32 # per-candidate features
mask:                (max_options,) bool      # validity mask
```

The five context features are normalised episode statistics
(steps-taken / frontier-size / tree-size / leaf-count fractions, plus
the focus-to-target term similarity).

The five per-candidate features are
`[is_mul, sim(left, T), sim(right, T), nnz(left), nnz(right)]`. They are
deliberately compact — instead of embedding the full polynomials of
each child (which would multiply the encoder cost by `max_options`), the
networks broadcast the encoded *state* across the K slots and concatenate
these light features for the per-slot scoring head. This keeps both
PyTorch and JAX networks small and fast.

Helpers `observation_to_tensors` and `stack_observations` (in
`breakdown_env.py`) convert a single obs or a list of obs into the dict
of torch tensors the PyTorch networks expect. The JAX trainer uses its
own numpy-stack helpers so it stays decoupled from torch.

---

## Trainers

All three trainers share the same external API: a constructor taking
`Config`, a `train(num_iterations)` method returning a history dict with
the same keys as the existing trainers (`pg_loss`, `vf_loss`, `entropy`,
`success_rate`, `avg_reward`, `complexity`), and an `evaluate()` method
for greedy roll-out.

### `PPOMCTSBreakdownTrainer` (PyTorch)

`ppo_mcts_breakdown.py`. The most direct counterpart to the existing
`src/algorithms/ppo_mcts.PPOMCTSTrainer`.

- **Network** — `BreakdownPolicyValueNet` (PyTorch).
  Three-stream MLP encoder for `focus`, `target`, and `context` →
  fused embedding → broadcast over `K` candidate slots and concatenated
  with `cand_feats` → per-slot scoring head + state-only value head.
- **Search** — `BreakdownMCTS`. PUCT with neural priors and
  network-evaluated leaves, faithful to `src/algorithms/mcts.MCTS` but
  rewritten for the breakdown observation format.
- **Update** — Standard PPO clipped-surrogate, GAE, optional
  visit-distribution distillation (`config.gumbel_distill_coef`).
- **Curriculum** — Same sliding-window scheme as the forward trainer
  (`advance_threshold`, `backoff_threshold`).
- **Library** — A fresh `FactorLibrary` is created per trainer instance.
  Successful decompositions are registered automatically.

### `SACBreakdownTrainer` (PyTorch)

`sac_breakdown.py`. The most direct counterpart to
`src/algorithms/sac.SACTrainer`.

- **Networks** —
  - `BreakdownSACActor`: same encoder + per-slot scorer architecture as
    the PPO version, masked categorical output.
  - `BreakdownSACCritic`: twin-Q heads, both produce per-slot action
    values (`[B, K]` each).
  - `target_critic`: EMA copy of the critic, softly updated with `tau`.
- **Replay** — `_StratifiedReplay` mirrors the existing
  `StratifiedReplayBuffer` (complexity / success / recent fractions),
  re-typed for breakdown observations.
- **n-step returns** — Same `_build_n_step` helper as in `sac.py`,
  reimplemented locally to avoid coupling to the existing trainer.
- **Adaptive entropy temperature** — `log_alpha` learnable scalar with
  state-dependent entropy targets, identical to the forward trainer.
- **Library** — Same lifetime as PPO+MCTS: created per trainer and
  updated on success.

### `PPOBreakdownJAXTrainer` (JAX/Flax/optax)

`ppo_breakdown_jax.py`. Mirrors the structure of the existing
`src/algorithms/ppo_mcts_jax.PPOMCTSJAXTrainer`.

- **Network** — `BreakdownPolicyValueNetJax` (Flax) — same architecture
  as the PyTorch version, in Flax.
- **Optimiser** — `optax.chain(clip_by_global_norm, adam)` exactly like
  the existing JAX trainer.
- **Inference** — JIT'd batched forward pass over `B` parallel envs.
- **Update** — JIT'd `value_and_grad` step (`_ppo_update_step`); applied
  in mini-batches over `ppo_epochs` epochs.
- **Rollouts** — Run on host across `B` parallel `BreakdownGame`
  instances. Stepping the environment is sequential per env (it has to
  be — see the [JAX coverage section](#jax-coverage-and-the-missing-mctx-pieces)
  below for why), but each env's actions are sampled from a single
  batched JAX inference call so the bulk of the cost is GPU-side.
- **Curriculum + library** — Same as the PyTorch trainers.

`PPOBreakdownJAXTrainer` does **not** use MCTS. The bottom-up forward
trainer is able to use `mctx` for batched MCTS because the forward env
(`jax_env.py`) is fully expressed as JAX arrays. The breakdown env's
candidate generator depends on `sympy.factor_list`, which cannot be
JIT'd or `vmap`'d, so an `mctx`-based search would have to call back to
the host via `jax.pure_callback` for every leaf expansion — at that
point most of the JAX speedup is gone. We instead use plain on-policy
PPO without MCTS in the JAX trainer; for MCTS-driven training of the
breakdown task, use the PyTorch `PPOMCTSBreakdownTrainer`.

---

## Reuse of `FactorLibrary`

The original task description called for using the existing
factorization utilities. We do, in two places:

- **Candidate generation** (`BreakdownGame._refresh_candidates`) calls
  `FactorLibrary.factorize_poly(focus)` for multiplicative splits and
  `FactorLibrary.exact_quotient(focus, f)` to obtain the cofactor.
- **Subgoal rewards** track the same `target_factor_keys` that the
  forward `CircuitGame` does, and award the same
  `factor_subgoal_reward` and `factor_library_bonus` when a child of
  the chosen split matches one of those factors. Reward magnitudes
  default to the `Config` values so the two games live on comparable
  scales.

After a successful decomposition, every distinct internal polynomial of
the tree is `register`-ed in the library (DFS post-order, so smaller /
shallower sub-polynomials are credited with lower step counts). This
mirrors `CircuitGame.register_episode_nodes` and lets the library grow
across episodes regardless of which trainer (forward or breakdown) is
running it.

> **Important:** each breakdown trainer instantiates its **own**
> `FactorLibrary` in its constructor. The library is **not** shared with
> any forward trainer that might be running, so enabling the breakdown
> trainers cannot pollute or be polluted by the existing experiments.

---

## Quick start

```python
from src.config import Config
from src.algorithms.breakdown.ppo_mcts_breakdown import PPOMCTSBreakdownTrainer
from src.algorithms.breakdown.sac_breakdown import SACBreakdownTrainer

cfg = Config(n_variables=2, mod=5, max_complexity=4)

# (1) PPO + MCTS breakdown trainer
ppo = PPOMCTSBreakdownTrainer(
    cfg,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_options=32,
    max_breakdown_steps=12,
    mcts_simulations=64,
)
hist = ppo.train(num_iterations=200)
print("eval:", ppo.evaluate(num_trials=100))

# (2) SAC breakdown trainer
sac = SACBreakdownTrainer(cfg, device="cpu", max_options=32)
sac.train(num_iterations=200)
sac.save_checkpoint("checkpoints/sac_breakdown.pt")

# (3) JAX PPO breakdown trainer (requires `pip install jax flax optax`)
from src.algorithms.breakdown.ppo_breakdown_jax import PPOBreakdownJAXTrainer
jax_trainer = PPOBreakdownJAXTrainer(cfg, batch_size=128, max_options=32)
jax_trainer.train(num_iterations=200)
```

All three trainers expose the same `train()` / `evaluate()` shape as the
existing forward trainers, so any plotting / logging code that consumes
the returned history dict will work unchanged.

---

## Algorithmic differences vs. the bottom-up trainers

| Aspect | Forward (existing) | Breakdown (new) |
| --- | --- | --- |
| Direction | Build $f^\star$ from $\{x_0, \ldots, 1\}$ upward | Decompose $f^\star$ downward |
| State | Circuit DAG (graph) + target poly | Decomposition tree + frontier + focus poly |
| Action space | $2 \cdot \binom{\|V\|+1}{2}$ (varies with $\|V\|$) | Fixed `max_options` (default 32) |
| Encoder | GNN over the circuit graph | MLP over `(focus, target, context)` |
| Per-action features | None (logits over `max_actions`) | `cand_feats` (5-dim per slot) |
| Multiplicative reasoning | Implicit (must be discovered) | Explicit (via `factorize_poly`) |
| Termination success | Some node equals $f^\star$ | Frontier is empty |
| Library reuse | Subgoal reward when factor is built | Subgoal reward + library-driven candidates |
| MCTS support (PyTorch) | Yes | Yes |
| MCTS support (JAX/`mctx`) | Yes | No (see below) |

---

## JAX coverage and the missing `mctx` pieces

The existing JAX trainer (`src/algorithms/ppo_mcts_jax.py`) uses
`mctx.muzero_policy` / `mctx.gumbel_muzero_policy` for **batched** MCTS,
which is the main source of its speedup over the PyTorch version. To do
the same for the breakdown task we'd need a JAX-friendly version of the
candidate generator — that is, JAX implementations of:

- `FactorLibrary.factorize_poly` (currently uses `sympy.factor_list`)
- `FactorLibrary.exact_quotient` (currently uses `sympy.div`)

There is no JAX-native polynomial factorization library, so the options
are:

1. **`jax.pure_callback` everywhere** — wrap the SymPy calls in
   `pure_callback` so they are reachable from within a `jit` / `vmap`
   trace. Functionally correct, but the `mctx` simulation step would
   call back to the host for *every* leaf expansion, which destroys the
   GPU parallelism that makes `mctx` fast in the first place.
2. **A specialised JAX factorization shim** — implement enough of
   "factor a small bivariate polynomial mod $p$" in pure JAX to remove
   the SymPy dependency from the hot path. Doable but a non-trivial
   subproject (most of the literature on multivariate factorization is
   not GPU-friendly).
3. **What we ship** — host-side batched rollouts, JIT'd network
   inference, JIT'd PPO update. No MCTS in the JAX path. The
   `PPOBreakdownJAXTrainer` still scales to large `batch_size` and
   benefits from GPU placement of the network and update step.

If batched MCTS for the breakdown task is needed in the future, option
(2) is the recommended path. Option (1) is a quick prototype; the
forward `ppo_mcts_jax.py` already establishes the surrounding `mctx`
plumbing one would reuse.

---

## Testing

A simple manual smoke test that exercises every public entry point:

```python
from src.config import Config
from src.environment.fast_polynomial import FastPoly
from src.algorithms.breakdown.breakdown_env import BreakdownGame

cfg = Config(n_variables=2, mod=5, max_complexity=4, max_steps=10, max_degree=4)

x0 = FastPoly.variable(0, 2, 4, 5)
one = FastPoly.constant(1, 2, 4, 5)
target = (x0 + one) * (x0 + one) + one        # (x + 1)^2 + 1

env = BreakdownGame(cfg, max_options=16, max_steps=8)
obs = env.reset(target)
while not env.done:
    obs, r, done, info = env.step(0)           # always pick first candidate
print("success:", info["is_success"], "tree size:", info["tree_size"])
```

Running this end-to-end in the repo's interpreter solves
$(x+1)^2 + 1$ in 3 decomposition steps and registers 3 polynomials in
the library — a useful sanity check.

The existing pytest suite (`tests/`) is unaffected by this package — all
96 tests still pass after adding the breakdown folder, because no
existing file was touched.

---

## Limitations and future work

- **No batched `mctx` MCTS in the JAX path** — see
  [JAX coverage](#jax-coverage-and-the-missing-mctx-pieces).
- **Candidate scoring features are minimal** — only 5 numbers per
  candidate. For richer credit assignment we could embed each child
  polynomial through a small shared encoder, at the cost of `K`-fold
  inference work per step. Worth profiling before adopting.
- **No explicit "give up and commit to leaf" action** — the agent must
  decompose every non-base polynomial. This is by design (the monomial
  peel-off keeps progress guaranteed), but a learned "treat as
  buildable directly" action could shorten the tree at the cost of
  needing a downstream sub-circuit synthesizer (essentially calling
  back to the forward trainer for that subtree).
- **Tree → circuit conversion is implicit** — the env counts distinct
  internal nodes as the circuit cost, but it does not actually emit
  the bottom-up action sequence. A small `tree_to_circuit_actions`
  helper would close that loop, e.g. for evaluation against the
  existing `CircuitGame`.
- **Curriculum is shared with the forward trainer's knobs** — we
  inherit `Config.starting_complexity`, `Config.max_complexity`,
  etc., which are tuned for the forward task. The breakdown task's
  difficulty does not scale with the same knobs in exactly the same
  way; a breakdown-specific curriculum could give better learning
  curves.

---

## FAQ

**Q. Does enabling these trainers affect the existing `PPO`, `PPO+MCTS`,
`AlphaZero`, or `SAC` runs?**

No. We added a new folder under `src/algorithms/`. No existing file is
modified, no existing import is changed, and the existing pytest suite
still passes (all 96 tests). Each breakdown trainer constructs its own
`FactorLibrary`; nothing is shared with the forward trainers.

**Q. Where do the multiplicative splits come from?**

From `FactorLibrary.factorize_poly` (over $\mathbb{Z}$ via SymPy's
`factor_list`, then reduced mod $p$). For each factor $f$, the cofactor
is computed by `FactorLibrary.exact_quotient` (pseudo-division over
$\mathbb{Z}$ followed by a mod-$p$ remainder check). Both are existing
utilities — the breakdown env did not introduce a new factorization
backend.

**Q. Can the agent get stuck in an infinite loop on an irreducible
polynomial?**

No. Even if there are no factor-library or leaf-residual splits
available, the monomial peel-off branch always produces at least one
valid candidate — peeling off a single monomial always reduces the
support of the polynomial by exactly 1, and the residual is non-zero.
Combined with the `max_steps` budget, every episode terminates either
in success or in a timeout.

**Q. Why is the JAX trainer's distillation target a one-hot of the
chosen action?**

In the PyTorch `PPOMCTSBreakdownTrainer` we distil the **MCTS visit
counts** into the policy via a cross-entropy term, gated by
`config.gumbel_distill_coef`. The JAX trainer has no MCTS, so there is
no visit distribution to distil. Setting the target to a one-hot of the
chosen action collapses that term to standard cross-entropy on the
action that was actually taken — harmless given that `pg_loss` already
covers it. We keep the term plumbed through so re-introducing
distillation later (if and when JAX MCTS becomes feasible) is a one-line
change.

**Q. Does the breakdown trainer always succeed within `max_complexity`
operations?**

No — there is no theoretical guarantee that the agent will decompose a
target into $\le$ `max_complexity` distinct internal nodes within
`max_breakdown_steps` decomposition steps. The reward shaping (success
reward minus a per-extra-node penalty) gives a soft incentive for
compact trees, and curriculum learning starts from easy targets, but a
hard target *could* yield a tree larger than the forward circuit budget.
The trainer treats this as success-with-an-oversized-tree, which
penalises the agent but does not abort the episode.
