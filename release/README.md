# Release — top-down arithmetic-circuit discovery (PPO & SAC)

Self-contained bundle of the final models, figures, and writeups for the
top-down decomposition RL agent over F₃ / (x, y, z).

## Start here
- **`PROBLEM_FORMULATION.md`** — what the problem is, the cost model, how the
  shortest-circuit complexity **Ck** is computed, worked examples, the success
  metric. Read this first.

## Contents
```
release/
├── PROBLEM_FORMULATION.md      problem setup, cost model, Ck, examples
├── checkpoints/                BOTH final models in one place
│   ├── ppo_mcts_final.pt       PPO+MCTS  (held-out Ck-match 0.918)
│   ├── sac_best.pt             SAC       (held-out Ck-match 0.923, iter 150)
│   └── CHECKPOINTS.md          provenance, SHAs, load instructions
└── figures/
    ├── ppo/                    PPO+MCTS plots (7) + ANALYSIS.md
    ├── sac/                    SAC plots (7) + ANALYSIS.md
    └── comparison/             PPO-vs-SAC plots (2) + COMPARISON.md
```

## Headline result
On a 207-target held-out set (disjoint from 450 train), both methods reach the
**verified shortest circuit** far above a random-policy baseline (~0.40), and
**generalize** (train ≈ held-out):

| | PPO+MCTS | SAC (best) |
| --- | --- | --- |
| held-out Ck-match | 0.918 | 0.923 |
| gap to optimal | 0.01 ops | ~0.02 ops |
| train wall-clock (1000 iters, RTX 4070 Ti) | ~2.5 h | ~30 min |

The learned-vs-random advantage **widens with circuit complexity** — see
`figures/comparison/cmp_02_final_per_ck.{png,pdf}`. All figures are PNG (300 dpi)
+ vector PDF for direct use in LaTeX.

## Reproduce
See the "Reproduce" section in each `ANALYSIS.md` and `../HANDOFF.md` (§4, §10,
§11) for the exact commands and curriculum generation.
