# env/ — Gymnasium Environment and Observation Encoding

This module implements the polynomial circuit construction environment as a standard Gymnasium environment, plus observation encoding and target polynomial samplers.

## Files

### circuit_env.py — PolyCircuitEnv

A goal-conditioned Gymnasium environment where the agent builds arithmetic circuits step by step.

**Episode flow:**
1. `reset()` — sample a target polynomial, create an empty circuit with input variables `(x0, x1, ..., const_1)`
2. Agent picks actions: `ADD(i,j)`, `MUL(i,j)`, `SET_OUTPUT(i)`, or `STOP`
3. Episode ends when: the output node matches the target (+1.0 reward), the operation budget runs out (truncated), or the agent stops

**Observation space** (Dict):
- `"obs"`: flat float32 vector of shape `(obs_dim,)` — see obs.py
- `"action_mask"`: int8 vector of shape `(action_dim,)` — 1 for valid actions, 0 for invalid

**Reward:**
- `-step_cost` for each ADD/MUL operation (default: -0.05)
- `+1.0` when output matches target polynomial
- `-1.0` for invalid actions (should not happen with mask)

**Key methods:**
- `reset(options={"max_ops": 3, "target_poly": poly})` — start new episode
- `step(action_id)` — execute one action
- `get_trajectory()` — returns episode history (used by HER)

### obs.py — Observation Encoding

Encodes the circuit state as a flat numpy array for the neural network.

**Per-node layout** (`d_node_raw` floats):
```
[type_onehot(3) | op_onehot(2) | parent1_idx(1) | parent2_idx(1) | pos_idx(1) | leaf_id(n_leaf) | eval_vector(m)]
```

- `type_onehot`: [input, op, empty] — distinguishes variable/constant nodes, operation nodes, and empty padding slots
- `op_onehot`: [add, mul] — which operation (zero for non-op nodes)
- `parent1_idx, parent2_idx`: raw integer indices of parent nodes (the network embeds these via `nn.Embedding`). Sentinel value `L` for leaf nodes
- `pos_idx`: node position in construction order
- `leaf_id`: one-hot identifying which variable (x0, x1, ...) or constant
- `eval_vector`: polynomial evaluated at `m` fixed random points

**Full observation:**
```
[L nodes × d_node_raw | target_eval(m) | steps_left(1)]
```

**Goal functions** (used by HER):
- `extract_goal(obs, config)` → target eval vector
- `replace_goal(obs, new_goal, config)` → obs with new target

### samplers.py — Target Polynomial Samplers

**RandomCircuitSampler**: builds a random circuit with `max_steps` operations and picks a random node's polynomial as the target. This is the default sampler — fast and diverse, but most targets only have one construction path.

**InterestingPolynomialSampler**: loads pre-computed polynomials from analysis JSONL files (from `Game-Board-Generation/`). These polynomials have *multiple shortest paths* — multiple distinct optimal circuits. The sampler:
- Converts SymPy expression strings to our internal `Poly` format
- Groups polynomials by `shortest_length` for curriculum-aware sampling
- `sample(rng, max_ops)` returns polynomials with `shortest_length ≤ max_ops`

Requires `sympy` (install via `pip install -e ".[interesting]"`).
