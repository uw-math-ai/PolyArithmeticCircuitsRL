# Analysis — `top-down-ppo-ck-mcts-big-1000`

Final analysis of the large PPO+MCTS run on the Ck-graded curriculum.

- **Algorithm:** PPO + MCTS (48 simulations/step, depth 6, distillation 1.0),
  with the whole-polynomial factor action enabled.
- **Data:** 450 train / 207 disjoint held-out targets, complexity buckets C2–C10
  (50 per bucket train; held-out de-duplicated against train, overlap = 0).
- **Compute:** 1000 iterations on an RTX 4070 Ti (~2.5 h).
- **Field/domain:** F_3, variables (x, y, z), small polynomials (≤7 terms in the
  exact-Ck regime).
- **W&B:** run `top-down-ppo-ck-mcts-big-1000`, entity
  `zengrf-university-of-washington`, project `PolyArithmeticCircuitsRL`,
  group `top-down`.

## Metric definition
"Success" = the greedy policy's circuit cost reaches the polynomial's
shortest-circuit complexity **Ck** (`cost <= ck`). Ck is the exact minimum op
count over the move set {additive split, whole-poly factorization, direct
build}, computed offline (see `src/decomp_rl/complexity.py`). The agent's moves
are a subset of that set, so `cost >= ck` always and a match means it found the
*verified shortest* circuit. Training never sees Ck (reward is env cost-savings
+ factor-library bonus + a terminal bonus vs the 5-baseline minimum), so there
is no label leak.

## Headline: generalization (fig `06_train_vs_heldout`)

| | train (450) | held-out (207, unseen) | random |
| --- | --- | --- | --- |
| Ck-match rate | **0.918** | **0.918** | ~0.41 |
| mean gap to optimal | 0.03 ops | 0.01 ops | — |

Train and held-out are identical, and the curves overlap from iteration ~100
onward with no drift through iteration 1000. On 207 polynomials it never trained
on, the policy finds the shortest circuit at the same rate as on training — i.e.
it learned a transferable split/factor policy, not a memorized lookup.

## Lift over random, per complexity (fig `07_agent_vs_random_by_complexity`)

The advantage over a uniform-random split policy **widens with complexity** —
the signature of genuine learning rather than trivially easy problems:

| Ck | learned (held-out) | random | lift |
| --- | --- | --- | --- |
| C2  | 1.00 | 0.81 | +0.19 |
| C3  | 0.95 | 0.71 | +0.24 |
| C4  | 1.00 | 0.57 | +0.43 |
| C5  | 1.00 | 0.44 | +0.56 |
| C6  | 1.00 | 0.30 | +0.70 |
| C7  | 0.76 | 0.32 | +0.44 |
| C8  | 0.72 | 0.23 | +0.49 |
| C9  | 0.96 | 0.31 | +0.65 |
| C10 | 0.86 | 0.18 | +0.68 |

Random collapses from 0.81 (C2) to 0.16–0.18 (C10) as circuits get longer, while
the learned policy stays in the 0.72–1.00 band. **C2 is near-trivial** (random
already 0.81) and should be footnoted or dropped from headline claims.

## Anomaly
**C8 is a genuine difficulty pocket: ~0.72 on both train and held-out** (not
noise — consistent across sets), and non-monotonic (C9 = 0.96 > C8 = 0.72). The
C8 instances appear structurally harder for the split/factor moves; worth a
trace-level look if chasing 100%.

## How this supersedes the earlier 0.90 (80-target) run
- **8× larger** benchmark (450 vs 80) → far less per-bucket noise.
- **Proper held-out test set** → 0.918 is *generalization*, not training-set fit.
- **Per-Ck random logged in-run** → the agent-vs-random comparison is produced by
  the run itself, not a post-hoc script.

## Limitations to disclose
- All results are in the **exact-Ck-computable regime** (≤7 terms). C9–C10
  buckets are partly heuristic-labeled (exact search stops past support 7), so
  "verified optimal" claims should be scoped to **C2–C8**.
- "Shortest circuit" = optimal within {split, factor, direct}; cross-circuit
  common-subexpression sharing is not modeled. For these small polynomials it is
  almost certainly the true optimum, but the wording should be scoped.
- Larger polynomials are skipped (optimum not certifiable) — that is the real
  frontier for follow-up work.

## Figures (this folder; PNG @ 300 dpi + vector PDF)
- `01_success_rate_over_iterations` — overall Ck-match vs iteration.
- `02_success_rate_by_complexity` — per-Ck match vs iteration.
- `03_final_success_by_complexity` — final per-Ck bars.
- `04_gap_to_optimal_over_iterations` — mean gap (ops) vs iteration.
- `05_training_curves` — reward / entropy / policy & value loss.
- `06_train_vs_heldout` — **generalization (key figure)**.
- `07_agent_vs_random_by_complexity` — **learned vs random per Ck (key figure)**.

## Reproduce
```bash
python scripts/run_ppo_curriculum.py \
  --target-file artifacts/ck_curriculum/train_big.jsonl \
  --heldout-file artifacts/ck_curriculum/heldout_big.jsonl \
  --iterations 1000 --device cuda --use-mcts --mcts-simulations 48 \
  --rollouts-per-update 16 --candidates-per-step 16 --max-episode-steps 24 \
  --seed 0 --eval-every 25 --random-rollouts 25 \
  --wandb-entity zengrf-university-of-washington --wandb-group top-down --wandb-mode online
python scripts/plot_training.py \
  --metrics artifacts/ck_curriculum/mcts-big/metrics.jsonl \
  --outdir  paper_plots/top-down-ppo-ck-mcts-big-1000 \
  --title "PPO+MCTS, 450 train / 207 held-out, factor action"
```
