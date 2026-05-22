# Model: top-down-ppo-ck-mcts-1000

Policy/value network for the top-down split-based circuit-discovery agent,
trained with **PPO + MCTS** on a Ck-graded curriculum with the
whole-polynomial factorization action enabled.

- **File:** `final.pt` (state_dict for `decomp_rl.model.TorchPolicyValueNetwork`)
- **SHA256:** `ba30d0b235096ffdce73df40ea3bf0c8b4a8e2aec981be682c6393d94c86f1e0`
- **Size:** ~151 KB (small MLP)
- **Commit / branch:** trained on `top-down-branch` at `db823e7`
- **W&B run:** `top-down-ppo-ck-mcts-1000` (entity `zengrf-university-of-washington`,
  project `PolyArithmeticCircuitsRL`, group `top-down`)

## Architecture
`TorchPolicyValueNetwork` — shared MLP over candidate features
`[target_feats, g_feats, h_feats]` (8 dims each = 24 in), one policy logit per
candidate split + a `Tanh` value head over the target features. See
`src/decomp_rl/model.py`.

## Field / domain
Trained over **F_3**, variables **(x, y, z)**. The curriculum
(`curriculum_targets.jsonl`, 80 polynomials, 10 per bucket C2–C9) labels each
target with its exact shortest-circuit complexity `ck` (see
`src/decomp_rl/complexity.py`).

## Training setup
- PPO + MCTS: `--mcts-simulations 48 --mcts-max-depth 6 --mcts-distill-coef 1.0`
- `--rollouts-per-update 16 --candidates-per-step 16 --max-episode-steps 24`
- `--learning-rate 3e-4 --entropy-coef 0.01 --library-reward-weight 1.0 --seed 0`
- 1000 iterations on an RTX 4070 Ti (~2.5 h).

## Results (final, greedy eval; success = greedy circuit reaches Ck)
- **Overall Ck-match rate: 0.900**, mean gap to optimal: **+0.09 ops**
- Random-policy reference: 0.453
- Per-Ck match: C2 1.00 · C3 1.00 · C4 1.00 · C5 1.00 · C6 0.60 · C7 0.90 · C8 0.80 · C9 0.90
- Full per-iteration curves in `metrics.jsonl`.

## How to load and run
```python
import torch
from decomp_rl.model import TorchPolicyValueNetwork

net = TorchPolicyValueNetwork()
net.load_state_dict(torch.load("models/top-down-ppo-ck-mcts-1000/final.pt", map_location="cpu"))
net.eval()
```
For greedy evaluation against Ck, see `greedy_episode_cost` /
`evaluate_curriculum` in `scripts/run_ppo_curriculum.py`. To reproduce training,
see `HANDOFF.md` §4.

## Caveats
- Trained specifically on F_3 / (x,y,z) small polynomials — not expected to
  generalize to other fields/variable sets without retraining.
- The agent's only moves are additive split and whole-poly factorization; the
  Ck targets it is scored against allow exactly these, so "match" means
  "reached the optimum within this move set" (see `HANDOFF.md` §6).
