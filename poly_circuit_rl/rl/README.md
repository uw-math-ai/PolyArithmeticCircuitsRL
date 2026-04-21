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
- epsilon-greedy exploration (linear anneal from 1.0 to 0.02)
- action masking (`invalid -> -1e9` before argmax)
- Huber loss + gradient clipping
- soft target updates (`tau`)
- checkpoint save/load

### `mcts.py` - `MCTS`

AlphaZero-style Monte Carlo Tree Search using the Q-network for both policy priors and leaf evaluation (no rollouts).

- **Policy prior**: softmax over Q-values for valid actions
- **Leaf evaluation**: `max(Q[valid_actions])`
- **PUCT formula**: `Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(child))`
- **Action selection**: temperature-scaled sampling from visit counts at the root

MCTS uses `env.get_state()`/`set_state()` to simulate actions without modifying the real episode.

### `replay_buffer.py` - `HERReplayBuffer`

Circular replay with future-goal HER relabeling and decomposed reward channels.

**Decomposed rewards** enable HER + factor library compatibility:
- Original transitions: `reward = base_reward + shaping_reward + solve_bonus`
- HER-relabeled transitions: `reward = base_reward + relabeled_solve_bonus` (shaping stripped)

This means factor library rewards are preserved in original transitions but correctly
excluded from relabeled transitions (since they depend on the original target).

Expert demo awareness: when `is_demo` transitions exist, reserves 20% of each
sampled batch for demos to maintain representation during training.

### `trainer.py`

`train(config, interesting_jsonl=None)`:
- creates factor library, env, agent, and optional MCTS
- **expert demo pre-fill**: generates BFS-optimal demonstrations and loads into buffer
- sampler selection (JSONL, auto-generative, or random-only)
- curriculum over `curriculum_levels` (default `(1,2,3,4,5,6)`)
- mixed interesting/random sampling at higher levels
- interleaved train updates after `learning_starts`
- periodic evaluation and checkpointing
- optional Weights & Biases logging with factor library metrics

`collect_episode(...)` gathers trajectories with decomposed rewards and passes them
to `add_episode_with_her()` for proper HER relabeling.
