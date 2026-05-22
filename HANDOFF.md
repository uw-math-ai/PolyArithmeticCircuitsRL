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
| `top-down-ppo-ck-mcts-1000` | **PPO+MCTS + factor action**, Ck curriculum (80) | **0.900 Ck-match, gap +0.09** |
| `top-down-ppo-ck-mcts-big-1000` | PPO+MCTS + factor action, 450 train / 207 held-out | **train 0.918 = held-out 0.918** (no overfit), gap ~0.01 (see §11) |
| `top-down-sac-ck-big-1000` | discrete SAC (target-entropy 0.7), big curriculum | held-out 0.831 final / **0.928 peak** — α over-grew (over-exploration) |
| `top-down-sac-ck-big-tuned-1000` | discrete SAC (target-entropy 0.4) | held-out **0.889 final / 0.928 best**, α stable; ~30 min vs ~2.5 h MCTS |

**PPO+MCTS vs SAC** (held-out, same benchmark/metric): PPO 0.918 (most stable),
tuned SAC 0.889 final / 0.928 best, both ≫ random ~0.40 and both generalize.
SAC trains ~5× faster (no search). Comparison figures + writeup in
`paper_plots/comparison_ppo_vs_sac/` (`compare_runs.py`). Shared weak spot: C8.

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
- SAC trainer: `src/decomp_rl/train_sac.py`; launcher
  `scripts/run_sac_finetune.py`; critic `model.py::TorchQNetwork` (see §10)
- Agent memory (machine-specific facts): `.claude/.../memory/` —
  `wandb-routing`, `factorization-backend`, `ck-action-space-gap`.

---

## 9. Paper plots

Figures for the paper are generated by **`scripts/plot_training.py`** directly
from a run's `metrics.jsonl` (the same data W&B shows — `metrics.jsonl` is the
canonical local source, so no W&B API/network call is needed).

Regenerate the PPO+MCTS-1000 figures:

```bash
python scripts/plot_training.py \
  --metrics artifacts/ck_curriculum/mcts-1000/metrics.jsonl \
  --outdir  paper_plots/top-down-ppo-ck-mcts-1000 \
  --title   "PPO+MCTS, Ck curriculum, factor action (1000 iters)"
```

Output folder `paper_plots/top-down-ppo-ck-mcts-1000/` (each figure as **PNG @
300 dpi + vector PDF**):

| File | What it shows |
| --- | --- |
| `01_success_rate_over_iterations` | overall Ck-match rate vs iteration, with the random-policy reference as a dashed line |
| `02_success_rate_by_complexity`   | per-Ck match rate vs iteration; one line per bucket, viridis-colored by complexity + colorbar |
| `03_final_success_by_complexity`  | final per-Ck match rate as a labeled bar chart |
| `04_gap_to_optimal_over_iterations`| mean gap (cost − Ck, in ops) vs iteration; 0 = optimal |
| `05_training_curves`              | 2×2: reward, entropy, policy loss, value loss (raw + rolling mean on noisy panels) |

### Styling conventions (all in the `plt.rcParams` block at the top of the script)
- **Readability first:** large fonts (axes 14, title 16 bold), light grid
  (`alpha 0.30`), top/right spines removed, `constrained_layout` so nothing is
  clipped, `lines.linewidth 2.2`.
- **Output:** `savefig.dpi 300`, `bbox="tight"`, **both PNG and PDF** (PDF is
  vector — preferred for LaTeX `\includegraphics`).
- **Color:** single accent blue `#1f6feb` for the primary line; reference lines
  dashed grey `#888888`; **complexity is always encoded with the perceptually
  uniform, color-blind-safe `viridis` colormap** (low Ck dark → high Ck yellow),
  shown via both a colorbar and a small legend.
- **Smoothing:** noisy per-iteration training curves (reward, policy loss) are
  drawn as a faint raw trace (`alpha 0.25`) under a centered rolling mean
  (`_rolling`, window 21); eval curves are plotted as-is with `o` markers since
  they are sampled every `--eval-every` iters.
- **Axes:** rates are pinned to `[0, 1]`; the value-loss panel is left
  unscaled and the y-label flags that it is `Tanh`-capped (see §6) so reviewers
  don't misread its magnitude.

To produce figures for any other run, point `--metrics` at that run's
`metrics.jsonl` and pick a fresh `--outdir`. `paper_plots/` is tracked (it is
small vector/PNG output); large `artifacts/` stays gitignored.

---

## 10. SAC (discrete Soft Actor-Critic) — top-down

Off-policy counterpart to the PPO trainer, following the same Decomposition-
Search RL theory and using the same environment, factor library, and
whole-polynomial factor action. No SAC code existed before this; it was built
from scratch (the legacy "PPO/SAC optional" notes never had an implementation).

### Files
- `src/decomp_rl/train_sac.py` — the trainer. `SACConfig`, `ReplayBuffer`,
  `SACTransition`, `collect_episode`, `sac_update`, `train_sac`, `SACMetrics`.
- `src/decomp_rl/model.py::TorchQNetwork` — candidate-scoring critic (scalar
  `Q(s, a)` per candidate; same `[target, g, h]` features, no Tanh, no value
  head).
- `scripts/run_sac_finetune.py` — launcher mirroring `run_ppo_finetune.py`
  (targets / `--target-file`, checkpoint in-out, W&B, metrics, device).
- `tests/test_train_sac.py` — smoke tests (collection, replay buffer, training
  finiteness, fixed-alpha path).

### Algorithm (discrete SAC over a variable action set)
This is Christodoulou (2019) "SAC for Discrete Action Settings", adapted to the
fact that every state exposes a *different-sized* candidate list:

- **Actor** = `TorchPolicyValueNetwork` (only its candidate logits are used;
  the value head is ignored). Reusing it means SAC can **warm-start from a PPO
  checkpoint** via `--checkpoint-in`, and the saved actor is loadable by the
  existing greedy/eval utilities.
- **Twin critics** Q1, Q2 (`TorchQNetwork`) + Polyak-averaged **target critics**
  (`tau` soft updates).
- **Soft state value:** `V(s') = Σ_a' π(a'|s') [ min(Q1_t, Q2_t)(s',a') −
  α·log π(a'|s') ]`; TD target `y = r + γ·V(s')` (zeroed when terminal).
- **Critic loss:** MSE of the chosen action's `Q1`, `Q2` against `y`.
- **Actor loss:** `Σ_a π(a|s) [ α·log π(a|s) − min(Q1,Q2)(s,a) ]` (Q detached).
- **Temperature α:** auto-tuned (default) toward
  `target_entropy = target_entropy_scale · log(num_candidates)` — note the
  target scales with the *per-state* candidate count, since the action set size
  varies. Disable with `--no-autotune-alpha`.
- **Replay buffer** of `(s, a, r, s', done)` where a "state" is an active
  polynomial + its candidate set. Episode decisions are linked so transition
  `i`'s next state is decision `i+1` (terminal at episode end / truncation).
- **Reward shaping** is identical to PPO: env cost-savings reward +
  `library_reward_weight · library_reward` + optional terminal bonus when the
  finished circuit beats the five-baseline minimum (`BaselineBundle`).

### How to run
```bash
# basic (built-in 2 targets) / or point at a curriculum JSONL
python scripts/run_sac_finetune.py \
  --target-file artifacts/ck_curriculum/targets.jsonl --variables x y z --prime 3 \
  --iterations 500 --device cuda --seed 0 \
  --rollouts-per-update 8 --gradient-steps 16 --batch-size 64 --learning-starts 256 \
  --wandb-entity zengrf-university-of-washington --wandb-group top-down --wandb-mode online
# warm-start from a PPO policy:
#   --checkpoint-in models/top-down-ppo-ck-mcts-1000/final.pt
```

W&B / metrics keys: `mean_episode_reward`, `mean_episode_savings`,
`critic_loss`, `actor_loss`, `alpha_loss`, `alpha`, `entropy`,
`mean_terminal_bonus`, `buffer_size`.

### Sanity check (25 iters, Ck curriculum, RTX 4070 Ti)
Learns as expected: reward 0.13 → ~10+, savings turns positive (+1.4),
actor loss decreases, entropy anneals from ~0.7, α auto-tunes around 0.2,
buffer fills. All 4 SAC tests + the full suite (48 passed, 1 skipped) green.

### Differences from PPO / gotchas
- **Per-item update loop.** `sac_update` loops over the batch computing per-item
  forwards (candidate sets are variable-length, like `ppo_update`). Simple but
  not vectorized — fine at this scale; pad+mask if you need speed.
- **No MCTS variant** for SAC (MCTS guidance is a PPO-side feature). SAC is pure
  off-policy actor-critic.
- **Not yet wired into the Ck curriculum launcher.** `run_sac_finetune.py` logs
  training metrics but not the periodic per-Ck greedy `ck_match_rate` eval that
  `run_ppo_curriculum.py` does. Easy follow-up: add the same
  `evaluate_curriculum` callback (the greedy eval is policy-only and works on
  the SAC actor unchanged), or factor the eval out of the PPO launcher into a
  shared module and call it from both.
- **train_optional_rl.py still rejects `algorithm="sac"`** (its test asserts
  this). SAC has its own entrypoint; wire it in there only if you also update
  `tests/test_train_ppo.py::test_run_optional_rl_rejects_non_ppo_algorithm`.

## 11. Evaluation rigor (train/test split, random per-Ck, unfiltered dist)

After a "success rate too high?" review, the eval was hardened. Key facts:

- **Training never sees Ck.** The PPO/SAC reward is env cost-savings + library
  bonus + terminal bonus vs the *baseline bundle* — never vs Ck. Ck is only the
  eval metric, so there is no label leak.
- **"Success" = reaching the exact optimum.** The agent's move set
  (split/factor/direct) is a subset of what the exact Ck search uses, so greedy
  cost >= Ck always; a "match" (cost <= ck) means it found the verified shortest
  circuit. (~2/80 old "successes" were vs heuristic-labeled Ck upper bounds.)
- **C2 is near-trivial** — a random split policy already matches it ~0.85+; the
  learned policy adds little there. Footnote or drop C2 in the paper.
- **The win grows with complexity.** On the old 80-target set, agent-vs-random
  per Ck: C2 1.00/0.89, C3 1.00/0.47, ... C9 0.90/0.09. Lead with the *lift*,
  per complexity, not the absolute 0.90.

Tooling added for defensible numbers:

- **Train/test split in `run_ppo_curriculum.py`:** `--heldout-file` evaluates a
  disjoint test set each cycle under `heldout/*` metrics. On the old run the
  held-out match (0.875) ≈ train (0.90) — i.e. it generalizes, not memorizes.
- **Per-Ck random reference logged in-run:** `random_ck_baseline` logs
  `baseline_train/C{k}/random_match_rate` and `baseline_heldout/...` (overall +
  per bucket), controllable via `--random-rollouts`.
- **`scripts/eval_unfiltered.py`:** evaluates on the *unfiltered* random-poly
  distribution (no Ck bucketing), computing exact Ck only where tractable
  (support<=7, degree<=6). On the mcts-1000 model: **agent 0.853 vs random
  0.421 over 300 random polys**, lift holding across C3–C12. Polys too large for
  exact Ck are skipped — that is the verifiable regime / the standing limitation.
- **Larger curricula:** `artifacts/ck_curriculum/train_big.jsonl` (450 targets,
  50/bucket C2–C10) and `heldout_big.jsonl` (207, disjoint). C9–C10 are
  partly heuristic-labeled (exact search runs out past support 7); treat
  "verified optimal" claims as C2–C8.
- **Plots:** `plot_training.py` gained `06_train_vs_heldout` and
  `07_agent_vs_random_by_complexity`.

**Result — `top-down-ppo-ck-mcts-big-1000`** (PPO+MCTS, 1000 iters, 450 train /
207 disjoint held-out, factor action). The headline generalization result:

- **Train Ck-match = 0.918, held-out = 0.918** (identical — no overfitting),
  random ≈ 0.41. Mean gap to optimal: train 0.03, held-out 0.01 ops.
- Held-out reached 0.918 by iter ~100 and stayed flat through iter 1000.
- Per-Ck (held-out) vs random, lift widens with complexity:
  C2 1.00/0.81 · C3 0.95/0.71 · C4 1.00/0.57 · C5 1.00/0.44 · C6 1.00/0.30 ·
  C7 0.76/0.32 · C8 0.72/0.23 · C9 0.96/0.31 · C10 0.92/0.18.
  (C8 is a real difficulty pocket — ~0.72 on both train and held-out.)
- Figures: `paper_plots/top-down-ppo-ck-mcts-big-1000/` (06 = train-vs-held-out
  generalization, 07 = per-Ck learned-vs-random bars are the paper money figs).

### SAC-specific future work
1. Add the per-Ck greedy eval to `run_sac_finetune.py` (or a shared launcher)
   so SAC produces the same dashboards/plots as PPO for a head-to-head.
2. Run a full 500–1000 iter SAC on the Ck curriculum and compare Ck-match /
   gap-to-optimal against PPO and PPO+MCTS.
3. Tune SAC knobs: `gradient_steps` per env step (UTD ratio), `batch_size`,
   `tau`, `target_entropy_scale`, buffer size; consider prioritized replay
   (the project already has `replay.py` for prioritized *target* replay).
4. Vectorize `sac_update` (pad candidate sets + mask) if throughput matters.
