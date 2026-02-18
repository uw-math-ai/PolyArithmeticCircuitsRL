# tests/ — Unit Tests

32 tests covering all components. Run with:

```bash
# From poly_circuit_rl/ directory
python -m unittest discover -s tests -p 'test_*.py' -v

# Or with pytest (if no hydra plugin conflict):
python -m pytest tests/ -v
```

## Test Files

### test_obs.py — Observation Encoding (10 tests)
Tests the flat observation layout produced by `env/obs.py`:
- Shape is exactly `(obs_dim,)` with correct `d_node_raw` per node
- Type/op one-hots set correctly for VAR, CONST, ADD, MUL nodes
- Parent indices: correct node IDs for ops, sentinel `L` for leaf nodes
- Position index set to node's slot number
- Steps-left correctly normalized to `[0, 1]`
- Eval vector matches polynomial evaluated at the fixed points
- Leaf ID one-hot correct for each variable and const
- `extract_goal` / `replace_goal` roundtrip works
- Empty slots filled with sentinel values and empty-type flag
- `get_num_real_nodes` correctly counts non-empty nodes

### test_env.py — Gymnasium Environment (9 tests)
Tests `env/circuit_env.py`:
- Reset produces correct observation and mask shapes
- All actions in the mask are actually executable without error
- Steps-left counter decrements correctly
- ADD/MUL incur the step cost reward (-0.05)
- Correct solve detection: output node matches target polynomial
- Agent can solve x0 + x1 by building ADD(0,1) and setting output
- Episode truncates when operation budget is exhausted
- Trajectory length matches number of steps taken
- Two envs with the same seed produce identical initial observations

### test_network.py — Transformer Q-Network (6 tests)
Tests `rl/network.py`:
- Forward pass output shape is `(B, action_dim)`
- Works with batch size = 1
- Parameter count is > 0 and < 500K
- Different observations produce different Q-values
- Deterministic in eval mode (same input → same output)
- Gradients flow through all parameters

### test_replay_buffer.py — HER Replay Buffer (3 tests)
Tests `rl/replay_buffer.py`:
- `add` and `sample` work correctly, return right shapes
- `add_episode_with_her` stores more transitions than the raw episode (HER augmentation)
- Circular buffer wraps correctly: capacity is respected

### test_agent.py — DQN Agent (3 tests)
Tests `rl/agent.py`:
- `select_action` with `deterministic=True` always picks from valid masked actions
- With `eps=1.0`, exploration samples valid actions randomly and covers all valid choices
- Save/load checkpoint preserves `total_steps` and produces identical Q-values
