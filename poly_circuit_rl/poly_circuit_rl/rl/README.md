# rl/ - DQN agent, transformer Q-network, replay buffer, trainer

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
- epsilon-greedy exploration
- action masking (`invalid -> -1e9` before argmax)
- Huber loss + gradient clipping
- soft target updates (`tau`)
- checkpoint save/load

### `replay_buffer.py` - `HERReplayBuffer`

Circular replay with future-goal HER relabeling.

For each transition, stores:
- original transition
- up to `her_k` relabeled transitions using future node eval vectors as goals

Relabeled goals operate in the same normalized goal space as observations.

### `trainer.py`

`train(config, interesting_jsonl=None)`:
- creates env + agent
- sampler selection:
  - if JSONL path exists: `InterestingPolynomialSampler`
  - else if `auto_interesting=True`: `GenerativeInterestingPolynomialSampler`
  - else: random-only targets
- curriculum over `curriculum_levels` (default `(1,2,3,4,5,6)`)
- mixed interesting/random sampling at higher levels via `interesting_ratio`
- interleaved train updates (`train_freq`) after `learning_starts`
- periodic deterministic evaluation and checkpointing

`collect_episode(...)` gathers episode trajectories and pushes HER-augmented transitions.

`evaluate(...)` reports success rate, average reward, average steps.
