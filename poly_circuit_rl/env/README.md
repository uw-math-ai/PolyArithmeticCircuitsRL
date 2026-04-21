# env/ - Gym environment, observation encoding, samplers, and expert demos

This module defines the goal-conditioned environment, target-polynomial samplers,
and expert demonstration generation.

## Files

### `circuit_env.py` - `PolyCircuitEnv`

Environment where the agent incrementally builds arithmetic circuits.

Observation dict:
- `obs`: flat `float32` vector `(obs_dim,)`
- `action_mask`: binary mask `(action_dim,)`

Actions:
- `ADD(i,j)`, `MUL(i,j)`, `SET_OUTPUT(i)`, `STOP`

Reward (decomposed into three channels):
- **base_reward**: `-step_cost` for ADD/MUL, 0 for SET_OUTPUT/STOP
- **shaping_reward**: factor library subgoal/completion bonuses + eval-distance shaping
- **solve_bonus**: `+1.0` when output polynomial equals target
- invalid action: `-1.0` and immediate truncation

Factor library integration:
- At `reset()`: factorizes target into subgoals via `FactorLibrary.factorize_target()`
- At `step()`: checks new nodes against subgoals, awards `factor_subgoal_reward` (+0.3),
  `factor_library_bonus` (+0.15), and `completion_bonus` (+0.5)
- Dynamic subgoal discovery when library-known nodes are built (T-v, T/v)
- On success: registers all constructed nodes in the factor library

State snapshots (for MCTS):
- `get_state()` - includes factor library per-episode state (subgoals, hits, completion flags)
- `set_state(state)` - restores env; sets `_simulation=True` to skip expensive operations

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
- `InterestingPolynomialSampler`: loads precomputed JSONL
- `GenerativeInterestingPolynomialSampler`:
  - lazy graph construction up to needed curriculum depth
  - shortcut filtering: selects polynomials where optimal is significantly shorter than naive

### `expert_demos.py` - expert demonstration generator

`ExpertDemoGenerator` uses the BFS game DAG to generate optimal trajectories:
- Builds the game graph via `graph_enumeration.build_game_graph()`
- Maps DAG nodes to internal `Poly` representations
- For each target, traces shortest path backward and converts to action sequences
- Replays actions through the environment to collect valid `Transition` objects
- All demo transitions are marked with `is_demo=True`

Used to pre-fill the replay buffer before training starts, directly addressing
the exploration catastrophe.

### `graph_enumeration.py`

Pure-Python graph enumeration + path analysis used by samplers and expert demos.
Contains:
- `build_game_graph` - constructs the arithmetic DAG
- `analyze_graph` - shortest/total path multiplicity analysis
- `shortest_path_intermediates` - BFS backward for path extraction
- `estimate_naive_ops` - naive monomial-by-monomial cost

### `factor.py` - SymPy conversion utilities

- `poly_dict_to_expr` - convert internal Poly dict to SymPy expression

### `oracle_mask.py` - diagnostic oracle action mask

Restricts actions to optimal DAG paths for diagnostic testing.
