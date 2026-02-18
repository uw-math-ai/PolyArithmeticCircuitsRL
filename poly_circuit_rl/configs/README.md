# configs/ — Hyperparameter Reference

`default.yaml` documents all hyperparameters and their defaults. It is **not** loaded automatically — the defaults live in `Config` (see [config.py](../src/poly_circuit_rl/config.py)).

Use this file as a reference when running experiments or as a template for overriding settings via `scripts/train.py` arguments.

## Hyperparameter Groups

### Environment
Controls the task difficulty and observation space size.
- `n_vars`: number of polynomial variables. Increasing this exponentially increases polynomial complexity.
- `max_ops`: max ADD/MUL operations per episode. Curriculum advances from 1 → max_ops.
- `L`: max visible nodes (obs size scales with L). Increase if circuits run out of space.
- `m`: evaluation point count. Higher = more reliable polynomial identity checks.

### Transformer
Controls model capacity.
- `d_model`: main hidden dimension. 64 is a good start; 128 for harder tasks.
- `n_heads`: attention heads. Must divide `d_model`.
- `n_layers`: transformer depth. 3 layers works well for max_ops ≤ 4.

### DQN
Standard DQN hyperparameters — usually don't need tuning.
- `eps_decay_steps`: how long epsilon anneals from 1.0 → 0.1. Increase for harder tasks.
- `learning_starts`: steps before training begins. Gives the buffer time to fill.

### Curriculum
- `curriculum_levels`: list of max_ops values the agent advances through.
- `curriculum_threshold`: success rate needed to advance (0.80 = 80%).
- `curriculum_window`: rolling window size for computing success rate.

### Mixed Sampling
- `interesting_ratio`: fraction of episodes using interesting polynomials (those with multiple optimal circuits). Only active at curriculum level ≥ 1.
