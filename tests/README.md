# tests/ - Test suite

Current suite: **113 tests across 17 modules**.

Coverage areas:
- core arithmetic and action encoding
- environment dynamics and reward modes
- replay/HER relabeling
- trainer/eval behavior
- MCTS
- factor-library behavior
- baseline algorithms
- script-level config loading
- graph enumeration safeguards
- end-to-end training smoke gate

## Run tests

```bash
# all tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q

# exclude slow smoke test
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -m "not slow" -q
```

## Modules

- `test_action_codec.py` (1)
- `test_agent.py` (7)
- `test_baselines.py` (14)
- `test_config.py` (4)
- `test_env.py` (14)
- `test_evaluate.py` (3)
- `test_expert_demos.py` (5)
- `test_factor_library.py` (22)
- `test_graph_enumeration.py` (1)
- `test_her_factor_compat.py` (4)
- `test_mcts.py` (5)
- `test_network.py` (6)
- `test_obs.py` (11)
- `test_poly.py` (1)
- `test_replay_buffer.py` (7)
- `test_train_smoke.py` (1, slow end-to-end learning gate)
- `test_trainer.py` (7)
