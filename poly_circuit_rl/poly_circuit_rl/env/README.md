# env/ - Gym environment, observation encoding, and samplers

This module defines the goal-conditioned environment and target-polynomial samplers.

## Files

### `circuit_env.py` - `PolyCircuitEnv`

Environment where the agent incrementally builds arithmetic circuits.

Observation dict:
- `obs`: flat `float32` vector `(obs_dim,)`
- `action_mask`: binary mask `(action_dim,)`

Actions:
- `ADD(i,j)`, `MUL(i,j)`, `SET_OUTPUT(i)`, `STOP`

Reward:
- `ADD`/`MUL`: `-step_cost`
- factorization shaping: `-factor_shaping_coeff` when `ADD` produces a factorizable polynomial (should have used `MUL` of factors instead). Only applied in real episodes, not during MCTS simulation.
- eval-distance shaping (disabled by default, `shaping_coeff=0.0`): bonus when a new node improves best eval-distance to target
- solve bonus: `+1.0` when output polynomial equals target
- invalid action: `-1.0` and immediate truncation

Termination and truncation:
- `terminated=True` on solve
- `truncated=True` on `STOP`
- op-budget truncation after post-budget resolve step
- hard episode cap at `max_episode_steps` (or derived default if unset)

State snapshots (for MCTS):
- `get_state()` - returns a serializable dict with cloned builder, step counter, target info
- `set_state(state)` - restores env from a snapshot; sets `_simulation=True` to skip expensive factorization shaping during MCTS tree search
- `reset()` clears `_simulation` flag for real episodes

### `obs.py` - observation encoder

Per-node layout (`d_node_raw`):

```text
[type_onehot(3) | op_onehot(2) | parent1_idx | parent2_idx | pos_idx | leaf_id(n_vars+1) | eval_vector(m)]
```

Full observation:

```text
[L * d_node_raw | target eval vector (m) | steps_left_norm]
```

Normalization:
- node and target eval vectors use tanh normalization: `tanh(v / eval_norm_scale)`
- set `eval_norm_scale <= 0` to disable normalization

Utilities:
- `extract_goal(obs, config)`
- `replace_goal(obs, new_goal, config)`
- `get_num_real_nodes(obs, config)`

### `samplers.py` - target samplers

- `RandomCircuitSampler`: random circuit targets
- `InterestingPolynomialSampler`: loads precomputed JSONL (typically from `Game-Board-Generation`)
- `GenerativeInterestingPolynomialSampler`:
  - lazy graph construction up to needed curriculum depth
  - **shortcut filtering** (`only_shortcut`): selects polynomials where the optimal circuit is significantly shorter than naive monomial-by-monomial construction (gap >= `min_shortcut_gap`, default 2)
  - grouped sampling by `shortest_length <= max_ops`

Dependencies for interesting samplers:
- `sympy`
- `networkx` (for generative sampler)

### `graph_enumeration.py`

Pure-Python graph enumeration + path analysis used by `GenerativeInterestingPolynomialSampler`.
Contains:
- game-graph construction (`build_game_graph`)
- shortest/total path multiplicity analysis (`analyze_graph`)
- `estimate_naive_ops(expr, num_vars)`: estimates the number of operations needed for naive monomial-by-monomial construction
- shortcut detection: compares `shortest_length` from graph search against `naive_ops` to identify polynomials with non-obvious optimal circuits
