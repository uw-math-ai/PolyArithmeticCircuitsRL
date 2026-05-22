# Handoff — Top-Down PPO, Ck Complexity Framing & Factorization

Branch: `top-down-branch`. Last meaningful commit: `db823e7`
("Add FLINT factorization backend, Ck complexity framing, and whole-poly factor action").

This document is a handoff for the next agent/developer: what exists, why, how to
reproduce, and what is worth doing next. Read it alongside `README.md` (the
project overview) — this file only covers the recent Ck/factorization work.

---

## 1. TL;DR of the journey

The top-down PPO learner builds circuits for sparse polynomials over F_p by
repeatedly choosing additive splits `f = g + h` and auto-factoring the pieces.
We made it actually work and gave it an honest, complexity-graded evaluation:

1. **Found the eval was meaningless** — the original `run_ppo_finetune.py`
   default targets are two trivial polynomials with no learnable structure.
2. **Built a real curriculum** with cost-saving structure and rich W&B metrics.
3. **Fixed factorization** — SymPy cannot factor multivariate polys over GF(p)
   and silently produced wrong (integer) factorizations. Replaced with a FLINT
   backend (Sage is not installable on this Windows box: no WSL/conda).
4. **Added "Ck" complexity framing** — Ck = op count of the shortest circuit;
   curriculum bucketed by exact Ck (C2..C9).
5. **Added a whole-polynomial factorization action** so the agent can reach
   optima like `(x+y)^2 = C2` that additive splits structurally cannot.
6. **Best run so far:** PPO+MCTS, 1000 iters → **0.90 Ck-match, mean gap 0.09
   ops** from the true shortest circuit (C2–C5 perfect).

---

## 2. Environment / hardware facts (this machine)

- GPU: **RTX 4070 Ti**, CUDA available. `torch 2.7.0+cu128`.
- Python: **3.13** (Windows Store python at `C:\Program Files\WindowsApps\...`).
  There is **no `.venv`** — packages are installed into that global interpreter.
- Installed via `pip install -e ".[train]"` plus `python-flint` (the `cas` extra).
- **No Sage, no conda, no WSL.** Do not try to set up Sage here; use FLINT.
- W&B: logged in as `wz82` (key in `C:\Users\86007\_netrc`).

### W&B routing (the user's requirement)
All runs go to: **entity `zengrf-university-of-washington`**, **project
`PolyArithmeticCircuitsRL`**, **group `top-down`** (`wz82` is a member of that
entity even though their default entity is `wz82-university-of-washington`).
Pass `--wandb-entity zengrf-university-of-washington --wandb-group top-down`.

---

## 3. What was built (files)

### New modules
- `src/decomp_rl/complexity.py` — **hybrid Ck**. `circuit_complexity(poly, ...)`
  returns `(ck, method)`. Exact memoized search over {direct build, whole-poly
  factorization, additive split} for small polys (support ≤ 7, degree ≤ 6),
  `BaselineBundle.min_cost` upper bound for larger ones. The exact search is the
  only thing that knows `(x+y)^2 = C2`, because it allows factoring the *whole*
  polynomial.

### New scripts
- `scripts/generate_ck_curriculum.py` — generates a pool, computes hybrid Ck,
  buckets by **exact Ck value** (C2, C3, …), keeps N per bucket. Emits JSONL
  with fields `prime, variables, terms, ck, method, level, source`.
- `scripts/generate_target_curriculum.py` — earlier generator graded by term
  count / degree into levels 1–4 (superseded by the Ck generator, kept for
  reference).
- `scripts/run_ppo_curriculum.py` — the **main launcher**. Wraps `train_ppo`,
  adds periodic greedy evaluation, logs per-Ck metrics to W&B, supports
  `--use-mcts`. Success = greedy circuit reaches Ck (`cost <= ck`).

### Modified core
- `src/decomp_rl/factor_fp.py` — **FLINT backend** (`_factor_via_flint`,
  `_flint_to_sparse`, `_flint_available`). Backend preference in `auto` mode:
  Sage > FLINT > SymPy. Trusts FLINT's monic factors + leading unit verbatim
  (do NOT re-monicize — that was a bug).
- `src/decomp_rl/split_proposals.py` — `SplitAction.kind` field
  (`"split"`/`"factor"`); `factor_whole_action(poly, factorization)` builds a
  `kind="factor"` action (`g=poly, h=0`) when the poly factors non-trivially.
- `src/decomp_rl/decomp_env.py` — `get_candidate_splits` prepends the factor
  action; `step` charges `add_op = 0` for factor actions (no `+1` addition).
- `src/decomp_rl/andor_search.py` — same wiring for MCTS: `_expand` injects the
  factor action, `_evaluate_action`/heuristic use `add_op`.
- `pyproject.toml` — added `cas = ["python-flint>=0.6"]`.

### NOT mine (left uncommitted, do not assume)
- `scripts/run_hyak_ppo.slurm` has a 6-line Klone module-load change in the
  working tree that predates / is unrelated to this work. It was deliberately
  excluded from commit `db823e7`.

---

## 4. How to reproduce

```bash
# 1. deps (once)
python -m pip install -e ".[train]"
python -m pip install python-flint            # or: pip install -e ".[cas]"

# 2. tests (should be 44 passed, 1 skipped)
python -m pytest -q

# 3. generate a Ck-bucketed curriculum (C2..C9, 10 each = 80 targets)
python scripts/generate_ck_curriculum.py \
  --output artifacts/ck_curriculum/targets.jsonl \
  --prime 3 --variables x y z --ck-min 2 --ck-max 9 --per-bucket 10 --seed 0

# 4a. plain PPO (fast, minutes)
python scripts/run_ppo_curriculum.py \
  --target-file artifacts/ck_curriculum/targets.jsonl \
  --iterations 500 --device cuda --seed 0 \
  --wandb-entity zengrf-university-of-washington --wandb-group top-down --wandb-mode online

# 4b. PPO + MCTS (the strong run; ~9 s/iter, ~2.5 h for 1000)
python scripts/run_ppo_curriculum.py \
  --target-file artifacts/ck_curriculum/targets.jsonl \
  --iterations 1000 --device cuda --seed 0 \
  --use-mcts --mcts-simulations 48 --mcts-max-depth 6 --mcts-distill-coef 1.0 \
  --rollouts-per-update 16 --candidates-per-step 16 --max-episode-steps 24 \
  --eval-every 20 \
  --checkpoint-dir artifacts/ck_curriculum/mcts-1000/checkpoints --checkpoint-every 25 \
  --metrics-out artifacts/ck_curriculum/mcts-1000/metrics.jsonl \
  --wandb-entity zengrf-university-of-washington --wandb-group top-down \
  --wandb-run-id <unique-id> --wandb-mode online
```

W&B metrics to watch: `eval/ck_match_rate`, `eval/mean_gap_to_optimal`,
`eval/C{2..9}/match_rate`, `eval/C{k}/mean_gap`, `baseline/random_ck_match_rate`,
plus `train/*` (reward, savings, policy/value loss, entropy, approx_kl,
terminal_bonus).

---

## 5. Results so far (W&B group `top-down`)

| Run id | Setup | Headline metric |
| --- | --- | --- |
| `top-down-ppo-4070ti-500` | plain PPO, 2 trivial built-in targets | flat / meaningless (baseline sanity only) |
| `top-down-ppo-curriculum-500` | plain PPO, term/degree-graded curriculum, success vs 5-baseline min | 0.833 success |
| `top-down-ppo-ck-500` | plain PPO, Ck curriculum, success = reach Ck | 0.588 Ck-match, gap +0.61 |
| `top-down-ppo-ck-mcts-1000` | **PPO+MCTS + factor action**, Ck curriculum | **0.900 Ck-match, gap +0.09** |

Final per-Ck match (MCTS-1000): C2 1.00, C3 1.00, C4 1.00, C5 1.00, C6 0.60,
C7 0.90, C8 0.80, C9 0.90.

---

## 6. Key concepts / gotchas for the next agent

- **Action space.** The agent picks among candidate `SplitAction`s. A
  `kind="factor"` action is encoded as `g=poly, h=0` — its candidate feature
  vector is `[target, target, zero]`. The model has **no explicit "is factor"
  feature** (we did not change feature dims to avoid invalidating checkpoints);
  it learns to recognize the encoding. Adding an explicit flag feature is a
  reasonable future improvement but requires retraining and a dim bump in
  `model.py` (`TARGET_FEATURE_DIM` / `CANDIDATE_FEATURE_DIM`).
- **Ck is an upper bound, exactly tight for small polys.** Exact Ck searches
  {factor whole, split, direct} — it does NOT model cross-circuit common
  subexpression sharing, so the true arithmetic-circuit complexity could be
  slightly lower in principle. For the small curriculum polys it matches
  intuition (`(x+y)^2`=2, `(x+1)(y+1)`=3, etc.).
- **Value head saturation.** `TorchPolicyValueNetwork`'s value head ends in
  `Tanh` (range [-1, 1]), but PPO returns are inflated by
  `PPOConfig.terminal_bonus_weight = 10`, so `value_loss` looks huge on the
  graphs. This is cosmetic for the policy (advantages still work) but hurts GAE
  quality. Fix options: normalize returns, scale the value target, or lower the
  bonus weight.
- **No resume in the curriculum launcher.** It supports `--checkpoint-in` for
  warm start but not the `--resume latest` auto-resume that
  `run_ppo_finetune.py` has. Long runs are not preemption-safe yet.
- **`artifacts/` is gitignored.** Checkpoints/metrics/wandb dirs are not
  committed. Curriculum JSONL under `artifacts/` is also not committed — the
  generator is deterministic given `--seed`, so regenerate rather than rely on
  the file.
- **FLINT API note (python-flint 0.8):** `flint.nmod_mpoly_ctx.get(list(vars),
  modulus=p)`; `ctx.from_dict({exp_tuple: coeff})`; `poly.monoms()/coeffs()`;
  `poly.factor() -> (leading_unit, [(monic_factor, exp), ...])`.

---

## 7. Future work (concrete, roughly prioritized)

1. **Investigate the C6 bucket (0.60).** It lags C7/C9 (0.90). Likely a few
   split/factor-resistant polys or small-sample noise (10/bucket). Dump the
   greedy traces for the C6 failures (see the trace-printing pattern used in
   conversation: run `greedy_episode_cost` but keep the `EnvState` and read
   `state.history`). Increase `--per-bucket` to reduce noise.
2. **Extend the curriculum.** Push Ck range to C10–C14 and raise `--per-bucket`
   to 20+. Larger polys will be `method="heuristic"` (exact search too slow);
   consider raising `DEFAULT_EXACT_SUPPORT_LIMIT` cautiously or adding a budgeted
   exact search.
3. **Explicit factor-action feature** in `model.py` (dim bump + retrain) so the
   policy doesn't have to infer "factor" from the `[target,target,zero]` shape.
4. **Value-head fix** (see gotcha above) — likely improves sample efficiency and
   makes the loss curves legible.
5. **Resume support** in `run_ppo_curriculum.py` (mirror `run_ppo_finetune.py`'s
   `--resume latest` + periodic checkpoints) for long/preemptible runs.
6. **Scale MCTS** — more simulations (64–128) / depth, compare Ck-match and
   wall-clock. Current run used 48 sims, depth 6.
7. **Persisted FactorizableLibrary** — currently rebuilt empty each run and
   populated in-memory. Seeding/saving it could speed up and improve reuse.
8. **Multi-prime / multi-variable studies** — everything here is F_3, vars
   (x,y,z). The pipeline is field/var agnostic; vary and measure.

---

## 8. Pointers

- Main launcher: `scripts/run_ppo_curriculum.py`
- Ck logic: `src/decomp_rl/complexity.py`
- Factor action: `src/decomp_rl/split_proposals.py::factor_whole_action`,
  wired in `decomp_env.py` and `andor_search.py`
- FLINT backend: `src/decomp_rl/factor_fp.py::_factor_via_flint`
- Agent memory (machine-specific facts): `.claude/.../memory/` —
  `wandb-routing`, `factorization-backend`, `ck-action-space-gap`.
