# tests/ - Unit test suite

Current suite: 32 tests across env/obs/network/agent/replay buffer.

## Run tests

```bash
# From poly_circuit_rl/
python -m unittest discover -s tests -p 'test_*.py' -v

# Optional pytest path
python -m pytest tests/ -v
```

## Test modules

### `test_obs.py` (10)
Validates observation encoding:
- layout, dimensions, dtypes
- parent/position/leaf fields
- eval vector correctness
- goal extract/replace utilities
- empty-slot and real-node counting behavior

### `test_env.py` (9)
Validates environment behavior:
- reset/step shapes and mask dtype
- mask-valid action executability
- step-cost and solve logic
- truncation behavior
- trajectory logging
- deterministic reset under fixed seed

### `test_network.py` (6)
Validates transformer Q-network:
- forward shapes
- deterministic eval-mode behavior
- parameter-count sanity
- gradient flow

### `test_replay_buffer.py` (3)
Validates replay + HER:
- add/sample behavior
- HER augmentation increases stored transitions
- circular overwrite at capacity

### `test_agent.py` (3)
Validates DQN agent:
- action selection respects mask
- exploration behavior at high epsilon
- checkpoint save/load roundtrip
