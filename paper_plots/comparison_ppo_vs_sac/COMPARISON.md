# Comparison — PPO+MCTS vs SAC (top-down circuit discovery)

Head-to-head on the **same** benchmark and metric: 450 train / 207 disjoint
held-out targets (C2–C10), greedy Ck-match (`cost <= ck`), whole-polynomial
factor action enabled, F_3 / (x, y, z). Both report the identical
train/held-out/per-Ck-random metrics from the shared launcher.

## Summary

| | PPO + MCTS | SAC (final) | SAC (best, iter 175) |
| --- | --- | --- | --- |
| held-out Ck-match | **0.918** | 0.889 | **0.928** |
| train Ck-match | 0.918 | 0.844 | ~0.92 |
| mean gap to optimal (held-out) | **0.01 ops** | 0.05 ops | ~0.02 ops |
| random reference | 0.40 | 0.40 | 0.40 |
| generalization (train≈held-out) | yes | yes | yes |
| wall-clock (1000 iters, 4070 Ti) | ~2.5 h | **~30 min** | — |
| uses MCTS / search at train time | yes | no | no |

## Read

- **Both far exceed random** and **both generalize** (train ≈ held-out with no
  overfitting) — the core result is method-independent.
- **PPO+MCTS is the most stable and slightly strongest at convergence**
  (0.918 flat from iter ~100). MCTS lookahead + distillation buys that
  stability, at ~5× the wall-clock.
- **SAC is competitive and far cheaper.** Its best checkpoint (0.928) edges out
  PPO; its final (0.889) is just below, after mild post-peak drift. No search
  needed.
- **Per-Ck** (`cmp_02_final_per_ck`): the two are within a few points across
  buckets — PPO leads slightly at C5/C6/C9, SAC leads at C7 — and the
  learned-vs-random gap **widens with complexity** for both (random 0.81→0.18,
  learned 0.68→1.00).
- **Shared weak spot:** C8 (~0.68–0.72 for both) is a genuine difficulty pocket,
  not a method artifact.

## Practical guidance
- For the **strongest, most stable** model: PPO+MCTS (use `final.pt`).
- For **fast iteration / ablations / compute-limited** settings: SAC reaches
  parity in ~30 min; use the **best held-out checkpoint** (`iter_00175.pt`),
  and consider `target_entropy_scale 0.3` or held-out early stopping to make the
  final checkpoint the best one.

## Figures (this folder)
- `cmp_01_heldout_over_iterations` — held-out Ck-match vs iteration, both runs +
  random reference.
- `cmp_02_final_per_ck` — final held-out Ck-match per complexity, PPO vs SAC vs
  random (grouped bars).

Per-method detail: `../top-down-ppo-ck-mcts-big-1000/ANALYSIS.md` and
`../top-down-sac-ck-big-tuned-1000/ANALYSIS.md`.

## Caveats (apply to both)
Exact-Ck-computable regime only (≤7 terms); C9–C10 partly heuristic-labeled, so
"verified optimal" claims scope to C2–C8. "Shortest" = optimal within
{split, factor, direct}; no cross-circuit CSE modeled. Same train/held-out
distribution (Ck-balanced small polynomials).
