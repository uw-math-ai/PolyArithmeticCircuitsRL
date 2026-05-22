# Released checkpoints

Both final/best policy/value networks for the top-down circuit-discovery agent,
over F_3 with variables (x, y, z). Each is a state_dict for
`decomp_rl.model.TorchPolicyValueNetwork` (~151 KB).

| File | Algorithm | Source | Held-out Ck-match | SHA256 (prefix) |
| --- | --- | --- | --- | --- |
| `ppo_mcts_final.pt` | PPO + MCTS | `mcts-big` run, iter 1000 (final = best, stable) | **0.918** | `d7c093c2…` |
| `sac_best.pt` | discrete SAC | `sac-big-tuned` run, **iter 150** (best held-out) | **0.923** | `79676212…` |

Both trained on the 450-target curriculum and evaluated on the disjoint
207-target held-out set (C2–C10). Random-policy reference ≈ 0.40.

Note: for SAC, `iter_00150.pt` is the best *saved* checkpoint (held-out 0.923);
the run's final checkpoint drifted slightly to 0.889 (see the SAC analysis).
PPO+MCTS was flat at 0.918, so its final checkpoint is also its best.

## Load and run
```python
import torch
from decomp_rl.model import TorchPolicyValueNetwork

net = TorchPolicyValueNetwork()
net.load_state_dict(torch.load("release/checkpoints/ppo_mcts_final.pt", map_location="cpu"))
net.eval()
# greedy rollout: see greedy_episode_cost in scripts/run_ppo_curriculum.py
```
Both checkpoints share the identical architecture, so the same loading code
works for either. The SAC actor is the same network class (its critics are not
needed for inference).
