# rl/ - DQN agent, MCTS, transformer Q-network, replay buffer, trainer

Reinforcement-learning components for polynomial circuit construction.

## Files

### `network.py` - `CircuitTransformerQ`

Transformer Q-network over flat observations.

Pipeline:
1. Parse flat obs into per-node continuous fields + index fields
2. Embed parent/position indices
3. Encode target eval vector
4. Concatenate with normalized `steps_left`
5. Run transformer encoder with causal + padding masks
6. Score actions with bilinear (`ADD`/`MUL`) and linear (`SET_OUTPUT`, `STOP`) heads

Outputs `Q(s, a)` for the full flattened action space.

### `agent.py` - `DQNAgent`

Double DQN with:
- epsilon-greedy exploration (used when MCTS is disabled)
- action masking (`invalid -> -1e9` before argmax)
- Huber loss + gradient clipping
- soft target updates (`tau`)
- checkpoint save/load

### `mcts.py` - `MCTS`

AlphaZero-style Monte Carlo Tree Search using the Q-network for both policy priors and leaf evaluation (no rollouts).

- **Policy prior**: softmax over Q-values for valid actions gives `P(s,a)`
- **Leaf evaluation**: `max(Q[valid_actions])` — the best Q-value among valid actions
- **PUCT formula**: `Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(child))`
- **Action selection**: temperature-scaled sampling from visit counts at the root

MCTS uses `env.get_state()`/`set_state()` to simulate actions without modifying the real episode. The env's `_simulation` flag is set during MCTS to skip expensive SymPy factorization checks.

Key config parameters:
- `mcts_simulations` (default 50): number of tree simulations per action
- `mcts_c_puct` (default 1.5): exploration constant in PUCT
- `mcts_temperature` (default 1.0): temperature for converting visit counts to action probabilities
- `use_mcts` (default True): toggle MCTS on/off (falls back to epsilon-greedy when off)

### `replay_buffer.py` - `HERReplayBuffer`

Circular replay with future-goal HER relabeling.

For each transition, stores:
- original transition
- up to `her_k` relabeled transitions using future node eval vectors as goals

Relabeled goals operate in the same normalized goal space as observations.

### `trainer.py`

`train(config, interesting_jsonl=None)`:
- creates env + agent + optional MCTS
- sampler selection:
  - if JSONL path exists: `InterestingPolynomialSampler`
  - else if `auto_interesting=True`: `GenerativeInterestingPolynomialSampler`
  - else: random-only targets
- curriculum over `curriculum_levels` (default `(1,2,3,4,5,6)`)
- mixed interesting/random sampling at higher levels via `interesting_ratio`
- interleaved train updates (`train_freq`) after `learning_starts`
- periodic deterministic evaluation and checkpointing
- optional Weights & Biases logging

`collect_episode(...)` gathers episode trajectories using MCTS (if enabled) or epsilon-greedy for action selection, then pushes HER-augmented transitions.

`evaluate(...)` reports success rate, average reward, average steps.
