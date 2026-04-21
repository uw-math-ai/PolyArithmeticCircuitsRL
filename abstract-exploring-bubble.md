# PolyArithmeticCircuitsRL — Fix Plan

## Context

This plan is the actionable subset of the prior 42-finding audit. Items
deferred (D4 policy head, D8 factored action head, D7 float16 storage, L4
tau tuning, L9 greedy O(L²) optimization) are listed at the bottom with the
reason they're out of scope.

Each entry below is concrete: file paths, the change, and a one-line
verification step. Tier order = priority: Tier 1 is correctness and
reproducibility, Tier 2 is methodology and ergonomics, Tier 3 is polish.
Do tiers in order; within a tier, top-down.

---

## Tier 1 — Correctness & Reproducibility

### T1.1  Config validator layer  *(was D9, L6, L7)*
**Why:** Closes a whole class of "fails 200 lines into the run" bugs and
makes `L >= max_nodes` enforced rather than hoped-for.

**File:** [poly_circuit_rl/config.py](poly_circuit_rl/config.py)

**Fix:** Add `__post_init__` to `Config` (frozen dataclass — pure asserts
since no normalization needed):
- `assert L >= max_nodes`
- `assert d_model % n_heads == 0`
- `assert all(curriculum_levels[i] < curriculum_levels[i+1] for i in
   range(len(curriculum_levels)-1))`
- `assert m >= 1 and eval_norm_scale > 0`
- `assert 0 < gamma < 1`
- `assert 0 <= eps_end <= eps_start <= 1`
- If `factor_shaping_coeff > 0`: try `import sympy`, raise
  `RuntimeError("factor_shaping_coeff>0 requires sympy")` if missing.

**Verify:** `Config(L=4, max_nodes=8)` raises; `Config()` succeeds.

---

### T1.2  HER must gate "solved" on exact poly equality  *(was D2/H3/H4)*
**Why:** Highest-leverage correctness fix. Today HER marks transitions
"solved" via `np.allclose` on tanh-saturated fingerprints; the env requires
exact polynomial equality. Agent learns to satisfy fingerprints, not
polynomials.

**Files:**
- [poly_circuit_rl/rl/replay_buffer.py:126-128](poly_circuit_rl/rl/replay_buffer.py#L126)
- [poly_circuit_rl/env/circuit_env.py](poly_circuit_rl/env/circuit_env.py) (trajectory storage)
- [poly_circuit_rl/rl/trainer.py:117](poly_circuit_rl/rl/trainer.py#L117) (per-step capture)

**Fix:**
1. Extend trajectory entries in `circuit_env.py` to also store the
   `PolyKey` of the achieved goal (use existing `poly_hashkey(node.poly)`).
   Add a parallel `ep_achieved_goal_keys: List[Optional[PolyKey]]` field
   in `add_episode_with_her`.
2. In `replay_buffer.py:126-128`, after the fingerprint `allclose` check,
   also require `ep_achieved_goal_keys[t] == new_goal_key`. Both must be
   true to set `relabeled_solved=True`.

**Verify:** Construct two distinct polynomials whose tanh-normalised eval
vectors collide (saturated MUL chain); assert HER does not relabel as
"solved".

---

### T1.3  Thread MCTS sampling through the seeded RNG  *(was H1)*
**Why:** Last remaining global-RNG hole; runs are not bit-reproducible
under MCTS.

**File:** [poly_circuit_rl/rl/mcts.py:243,248](poly_circuit_rl/rl/mcts.py#L243)

**Fix:** Construct `self.rng = np.random.default_rng(config.seed + 31)`
in `MCTS.__init__`. Replace `np.random.choice(valid_actions)` →
`self.rng.choice(valid_actions)` and `np.random.choice(len(probs),
p=probs)` → `self.rng.choice(len(probs), p=probs)`.

**Verify:** Two runs of `MCTS.search(...)` with `temperature > 0` and
identical env state produce identical actions.

---

### T1.4  `try/finally` around mutable env-state swaps  *(was H2, H6)*
**Why:** Two exception-leak paths today: MCTS leaves `env._simulation=True`
on raise; `circuit_env.reset` leaves the sampler's `max_steps` overridden
on raise.

**Files:**
- [poly_circuit_rl/rl/mcts.py:114-115](poly_circuit_rl/rl/mcts.py#L114)
- [poly_circuit_rl/env/circuit_env.py:137-140](poly_circuit_rl/env/circuit_env.py#L137)

**Fix:** Wrap both swap blocks in `try/finally` so the cleanup line always
runs, even on exception.

**Verify:** Inject an exception into `env.step`; assert
`env._simulation == False` and `target_sampler.max_steps == old_max`
after.

---

### T1.5  Curriculum advance must work without periodic eval  *(was H8)*
**Why:** Today, `eval_every=0` deadlocks the curriculum at level 0 silently.

**File:** [poly_circuit_rl/rl/trainer.py:443-449](poly_circuit_rl/rl/trainer.py#L443)

**Fix:** Either:
- (Simpler) `assert config.eval_every > 0` at the top of `train()`.
- (Better) Add a fallback path: if `config.eval_every == 0`, the advance
  condition uses only `train_sr >= curriculum_train_threshold`, ignoring
  `last_eval_sr`.

Pick (simpler) unless eval-free training is a real workflow you want.

**Verify:** `Config(eval_every=0)` either raises immediately, or trains
through curriculum levels using train SR alone.

---

### T1.6  Oracle mask must not silently fail open  *(was H9)*
**Why:** When the oracle returns no actions, the code keeps the original
mask and the diagnostic disables itself with no log.

**File:** [poly_circuit_rl/env/circuit_env.py:368-387](poly_circuit_rl/env/circuit_env.py#L368)

**Fix:** When `combined.sum() == 0`, emit `warnings.warn(...)` (once per
target via a `set` cache to avoid log spam), then keep the original mask
as today. Don't raise — that would break training mid-run.

**Verify:** Set up an oracle that returns empty actions for one specific
target; assert exactly one warning is emitted.

---

### T1.7  Separate eval env from training env  *(was D1, H10)*
**Why:** Currently the same `PolyCircuitEnv` is used for both phases.
Snapshot/restore of the factor library helps but doesn't isolate sampler
RNG, eval points, or trajectory state.

**Files:**
- [poly_circuit_rl/core/factor_library.py](poly_circuit_rl/core/factor_library.py) — add `frozen_view()`
- [poly_circuit_rl/rl/trainer.py](poly_circuit_rl/rl/trainer.py) — construct `eval_env` once, pass to `evaluate()`

**Fix:**
1. `FactorLibrary.frozen_view()` returns a wrapper instance whose
   `register_episode_nodes`, `register_node`, etc. are no-ops; reads pass
   through to the underlying library.
2. In `train()`, after `env = PolyCircuitEnv(config, factor_library=...)`,
   add:
   ```python
   eval_env = PolyCircuitEnv(
       config,
       factor_library=factor_library.frozen_view() if factor_library else None,
   )
   eval_env.eval_points = env.eval_points  # share fixed points only
   eval_env.rng = random.Random(config.seed + 200)  # distinct RNG
   ```
3. Pass `eval_env` into `evaluate(...)` instead of `env`. Drop the
   snapshot/restore plumbing in `evaluate()` since it's no longer needed.

**Verify:** Run train→eval→train→eval. Assert factor library size equals
the count from training-only registrations (no eval contamination).

---

### T1.8  End-to-end "learning improves" smoke test  *(was D5)*
**Why:** Unit tests cover behavior; nothing asserts that the agent
actually learns. Given the breadth of T1 fixes, regressions are otherwise
silent.

**File:** new — `tests/test_train_smoke.py`

**Fix:**
```python
def test_sparse_reward_one_op_one_var_learns():
    config = Config(
        n_vars=1, max_ops=1, L=4, max_nodes=4, m=8,
        total_steps=300, eval_every=100, eval_episodes=20,
        reward_mode="sparse",  # added in T2.1
        expert_demo_count=0, use_mcts=False, dropout=0.0,
        factor_library_enabled=False, auto_interesting=False,
    )
    agent = train(config)
    final_sr = evaluate(...)["success_rate"]
    assert final_sr > 0.5, f"agent failed to learn 1-op task: SR={final_sr}"
```
Mark `@pytest.mark.slow` if runtime > 30 s on CPU.

**Verify:** `pytest tests/test_train_smoke.py` passes locally.

---

## Tier 2 — Methodology & Stability

### T2.1  `reward_mode` flag for clean ablation  *(was D3)*
**Why:** Today `reward = -step_cost + shaping + factor_library + completion
+ solve_bonus` with HER stripping subsets inconsistently. No way to
isolate "does the underlying agent work at all."

**Files:**
- [poly_circuit_rl/config.py](poly_circuit_rl/config.py)
- [poly_circuit_rl/env/circuit_env.py](poly_circuit_rl/env/circuit_env.py)

**Fix:** Add `Config.reward_mode: Literal["sparse", "shaped", "full"] =
"full"`. In `circuit_env.step`, gate shaping/factor/completion contributions
on `reward_mode != "sparse"` (and shaped excludes factor/completion).
T1.8's smoke test runs in `"sparse"`.

**Verify:** With `reward_mode="sparse"`, every reward in a successful
episode equals `-step_cost` per ADD/MUL plus `+1.0` solve bonus.

---

### T2.2  Held-out eval target set + Wilson CI + gap-to-optimal  *(was D6)*
**Why:** Train and eval currently sample from the same distribution. No
confidence intervals. The exhaustive baseline exists but isn't used as a
benchmark.

**File:** [poly_circuit_rl/rl/trainer.py](poly_circuit_rl/rl/trainer.py)

**Fix:**
1. **Held-out set:** at trainer init, pre-sample `config.eval_episodes`
   targets per curriculum level using `Random(config.seed + 90_000)`.
   Cache as `eval_targets[level] = [Poly, ...]`. `evaluate()` iterates
   this list instead of resampling each call.
2. **Wilson CI:** in the eval print line, replace
   `f"SR={success_rate:.2%}"` with success rate plus 95% Wilson interval.
   ~5 lines of stats math.
3. **Gap-to-optimal:** when `cur_max_ops <= 4`, run
   `ExhaustiveSearch(config).build(cur_max_ops)` once at trainer init;
   cache `optimal_ops[poly_hashkey(target)]`. In `evaluate()`, log
   `mean(actual_ops - optimal_ops)` over solved episodes.

**Verify:** Successive eval calls on the same checkpoint return identical
success rates (held-out determinism). Gap metric is `>= 0` and reported.

---

### T2.3  Coordinate completion bonus stacking  *(was M1)*
**Why:** A single new node can trigger both additive and multiplicative
completion bonuses in one step.

**File:** [poly_circuit_rl/env/circuit_env.py:413-419,449-453](poly_circuit_rl/env/circuit_env.py#L413)

**Fix:** Replace the two `_*_complete_hit` flags with a single
`_completion_hit` flag. Once it's set by either path, neither path adds
again.

**Verify:** Construct an episode where both `T-v_new in circuit` AND
`T/v_new in circuit` for one node; assert reward delta is exactly one
`completion_bonus`, not two.

---

### T2.4  Cap factor library size  *(was M5)*
**Why:** Library grows monotonically across runs; long training slows
`factorize_target` and grows memory.

**File:** [poly_circuit_rl/core/factor_library.py](poly_circuit_rl/core/factor_library.py)

**Fix:** Add `Config.factor_library_max_size: int = 10_000`. Implement
LRU eviction in `register_episode_nodes` once size exceeds cap. Use
`collections.OrderedDict` for `_known`.

**Verify:** Run 20k episodes synthetically; assert
`len(factor_library) <= 10_000`.

---

### T2.5  Explicit length check in `eval_poly`  *(was M8)*
**Why:** `zip(mono, values)` truncates silently when `len(values) <
n_vars`. Real bug if a caller ever passes a mis-shaped value list.

**File:** [poly_circuit_rl/core/poly.py:173](poly_circuit_rl/core/poly.py#L173)

**Fix:** Compute `n_vars = max((len(m) for m in poly), default=0)` and
`assert len(values) >= n_vars` at function entry. Add a unit test for
the failure case.

---

## Tier 3 — Polish

### T3.1  Regenerate `tests/README.md`  *(was M3)*
**Why:** Says "32 tests"; actual count is 94 across 12 modules.

**File:** [tests/README.md](tests/README.md)

**Fix:** Regenerate the test-count and module list. Add a one-line note
that `test_train_smoke.py` (added in T1.8) is the end-to-end gate.

---

### T3.2  Bounds-check `encode_action(SET_OUTPUT, i, …)`  *(was L1)*
**Why:** API inconsistency: ADD/MUL raise on bad indices, SET_OUTPUT
silently overflows.

**File:** [poly_circuit_rl/core/action_codec.py:98-101](poly_circuit_rl/core/action_codec.py#L98)

**Fix:** Add `if not 0 <= i < L: raise ValueError(...)` before computing
the index. Mirror the SET_OUTPUT validation pattern from `pair_to_index`.

---

### T3.3  Log SymPy fallback paths  *(was L3)*
**Why:** Broad `except Exception` swallows SymPy failures with no trace.
Hard to diagnose factor-library degradation in training logs.

**Files:** [circuit_env.py:175,420,438,461](poly_circuit_rl/env/circuit_env.py#L175),
[factor_library.py](poly_circuit_rl/core/factor_library.py)

**Fix:** At each broad-except site, replace `pass` with a single
`logging.debug("sympy fallback in %s: %s", __name__, e)`. Keep the
exception swallowed (existing behavior), but make it inspectable via log
level.

---

### T3.4  Throttle W&B artifact uploads  *(was L5)*
**Why:** With `eval_every=1000` and `total_steps=500_000`, the trainer can
upload dozens-to-hundreds of "best" artifacts before SR stabilizes.

**File:** [poly_circuit_rl/rl/trainer.py:498-517](poly_circuit_rl/rl/trainer.py#L498)

**Fix:** Add `Config.wandb_artifact_min_interval_steps: int = 50_000`.
Track `last_artifact_step`; only upload when
`agent.total_steps - last_artifact_step >= interval` AND it's a new best.

**Verify:** With interval=50k and total_steps=500k, at most ~10 artifacts
upload regardless of how often SR improves.

---

### T3.5  Warn on partial expert demo prefill  *(was L8)*
**Why:** Today, partial demo failure is logged as a `Warning:` line and
training continues silently with an under-sampled demo buffer. The 20%
demo reservation in `replay_buffer.sample` then under-samples too.

**File:** [poly_circuit_rl/rl/trainer.py:350-366](poly_circuit_rl/rl/trainer.py#L350)

**Fix:** After demo generation, if
`len(demos) < 0.5 * config.expert_demo_count`, emit a louder warning and
optionally `assert config.allow_partial_demos` (a new flag, default
True for back-compat). Same change makes accidental "no demos at all"
visible.

---

### T3.6  Clone shouldn't alias mutable list  *(was M4)*
**Why:** `CircuitBuilder.clone()` shares `eval_points` reference. Safe
today (nothing mutates), but a future bug waiting.

**File:** [poly_circuit_rl/core/builder.py](poly_circuit_rl/core/builder.py)

**Fix:** Replace `cloned.eval_points = self.eval_points` with
`cloned.eval_points = tuple(self.eval_points)` so it's immutable by
construction. Update type hint accordingly.

---

### T3.7  Wall-clock budget for graph enumeration  *(was M7)*
**Why:** Only safety guard is `gen_max_graph_nodes=100_000`. No
wall-clock budget. Demo generation can hang on hard configs.

**File:** [poly_circuit_rl/env/graph_enumeration.py](poly_circuit_rl/env/graph_enumeration.py)

**Fix:** Add `Config.gen_max_seconds: float = 60.0`. Pass through; check
`time.perf_counter()` against deadline in the inner expansion loop;
break and warn on overrun.

---

### T3.8  Stale `runs/` artifacts  *(was L10)*
**Why:** `git status` shows deleted log files, `best_*.pt`, `final.pt`,
`train.pid` from a previous repo layout. Clutters provenance.

**Fix:** `.gitignore` `runs/` if it isn't already, then `git rm -r
--cached runs/` to drop tracked artifacts. (Confirm with user first
before running — destructive on intent, not on disk.)

---

### T3.9  Clarify experimental modules  *(was D10, D11)*
**Why:** `bandits.py` / `bandit_trainer.py` aren't invoked from
`train.py` and have no README coverage. `env/factor.py` is used but
unobvious.

**Files:**
- top-level `README.md` (or `CLAUDE.md`)
- module docstrings on the three files

**Fix:**
- Add a one-paragraph "Module map" section to the README naming the
  canonical entrypoint (`scripts/train.py`) and marking
  `bandit_trainer.py` as experimental.
- Add a `# EXPERIMENTAL — not invoked from train.py` banner to the bandit
  module docstrings.
- Confirm `env/factor.py` is referenced from `baselines/factorization.py`
  and `baselines/symbolic_utils.py`; if any function inside is unused,
  delete it.

---

## Out of Scope (with reason)

These were in the prior audit but **deliberately not implementing**:

- **D4 — Separate policy head for MCTS priors.** Costly architecture
  refactor, replay schema change, new loss balancing, likely checkpoint
  break. Premature before T1 correctness fixes settle. Revisit only if
  metrics show MCTS-via-Q has plateaued.
- **D8 — Factored action head.** Large refactor across action encoding,
  masking, Q-target semantics, MCTS, and tests. No payoff at L=16.
  Revisit when scaling L past ~32.
- **D7 — Float16 replay storage.** Memory isn't a current pain point.
  Saturation interaction with T1.2 adds risk for marginal benefit.
- **L4 — Polyak `tau=0.005` tuning.** Hyperparameter churn. Not a bug.
- **L9 — Greedy baseline O(L²) optimization.** Engineering distraction;
  doesn't touch RL correctness.

Also kept as observations only (no fix planned, low impact today):
H5 invalid-action penalty path, H7 HER reward-scale mismatch (intentional
tradeoff), H11 (overlap with H7), M2 legacy-test brittleness, M6 expert
demos mid-replay drift, M9 parent-index clamp speculation, L2 float-sqrt
precision.

---

## Order of Operations

**Sprint 1 (correctness):** T1.1 → T1.4 → T1.3 → T1.5 → T1.6.
Cheap, no schema changes, all surgical.

**Sprint 2 (the big two):** T1.2 (HER exact equality) → T1.7 (eval env
isolation). These are the high-leverage fixes that change observed
behavior; do them after the surgical fixes so you can attribute SR
changes correctly.

**Sprint 3 (gating):** T1.8 (smoke test) → T2.1 (reward_mode) — added
together since the smoke test depends on `reward_mode="sparse"`.

**Sprint 4 (methodology):** T2.2 → T2.3 → T2.4 → T2.5.

**Sprint 5 (polish):** all T3.* in any order.

After each sprint: full pytest, run training for ~5 min, sanity-check
that SR is not catastrophically worse than the pre-sprint baseline.