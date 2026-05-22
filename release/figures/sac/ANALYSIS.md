# Analysis — `top-down-sac-ck-big-tuned-1000`

Discrete Soft Actor-Critic (SAC) on the same Ck-graded curriculum and metric as
the PPO run, with a tuned entropy target.

- **Algorithm:** discrete SAC (twin Q-critics, Polyak target nets, auto-tuned
  temperature α), whole-polynomial factor action enabled. No MCTS.
- **Key fix vs the first SAC run:** `--target-entropy-scale 0.4` (was 0.7). The
  first run's α ballooned 0.20 → 1.09, over-exploring and decaying held-out from
  a 0.928 peak to 0.831. With 0.4, α stays stable (~0.18–0.23).
- **Data / compute:** 450 train / 207 disjoint held-out, C2–C10; 1000 iters on
  an RTX 4070 Ti in **~30 min** (vs ~2.5 h for PPO+MCTS).
- **Metric:** identical to PPO — greedy Ck-match (`cost <= ck`), train + held-out
  + per-Ck random reference (see the PPO analysis for the full definition).

## Results (held-out)

| | value |
| --- | --- |
| final Ck-match (iter 999) | **0.889** |
| best Ck-match (iter 175) | **0.928** |
| train Ck-match (final) | 0.844 |
| mean gap to optimal | 0.05 ops |
| random reference | ~0.40 |

Generalizes cleanly: train ≈ held-out at every eval point (`06_train_vs_heldout`).

## α and stability (the headline of this run)
```
iter   0: heldout=0.502  alpha=0.200
iter 200: heldout=0.918  alpha=0.219   <- peak region
iter 400: heldout=0.903  alpha=0.234
iter 600: heldout=0.884  alpha=0.211
iter 999: heldout=0.889  alpha=0.184
```
α is now flat — the over-exploration collapse is gone. Held-out peaks ~0.92
around iter 175–200 and settles ~0.89, oscillating in a tight 0.85–0.92 band
(mild off-policy drift, not the α blow-up of the untuned run).

## Per-Ck (held-out, final) vs random
C2 1.00 · C3 0.95 · C4 1.00 · C5 0.91 · C6 0.96 · C7 0.80 · C8 0.68 · C9 0.88 ·
C10 0.88, all far above random (0.81 → 0.18). Same C8 difficulty pocket as PPO.

## Takeaways
- **SAC is competitive with PPO+MCTS** (final 0.889 vs 0.918; best 0.928 vs
  0.918) at **~5× lower wall-clock**, with no MCTS.
- Use the **best checkpoint** (`iter_00175.pt`) for a deployed model; `final.pt`
  is slightly past the peak.
- Further stability could come from a touch lower `target_entropy_scale`
  (0.3), a smaller actor LR, or early stopping on held-out.

## Figures (this folder, PNG @ 300 dpi + PDF)
Same 7-figure set as the PPO run; `06_train_vs_heldout` and
`07_agent_vs_random_by_complexity` are the key ones. Cross-method comparison
figures are in `paper_plots/comparison_ppo_vs_sac/`.

## Reproduce
```bash
python scripts/run_ppo_curriculum.py --algorithm sac \
  --target-file artifacts/ck_curriculum/train_big.jsonl \
  --heldout-file artifacts/ck_curriculum/heldout_big.jsonl \
  --iterations 1000 --device cuda \
  --rollouts-per-update 16 --gradient-steps 16 --batch-size 64 --learning-starts 512 \
  --target-entropy-scale 0.4 --seed 0 --eval-every 25 --random-rollouts 25 \
  --wandb-entity zengrf-university-of-washington --wandb-group top-down --wandb-mode online
```
