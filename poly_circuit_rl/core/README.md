# core/ - Polynomial arithmetic and circuit primitives

Mathematical foundations used by the environment and agent.

## Files

### `poly.py` - exact sparse polynomial arithmetic

`Poly = Dict[Tuple[int, ...], Fraction]`

Key functions:
- constructors: `make_zero`, `make_const`, `make_var`
- ops: `add`, `sub`, `mul`, `canonicalize`, `equal`
- hashing: `poly_hashkey` — canonical hashable key for deduplication and library lookups
- predicates: `is_scalar` — True if polynomial is degree 0
- analysis: `stats`
- evaluation: `eval_poly`

Arithmetic is exact (`Fraction`), avoiding floating-point identity errors.

### `fingerprints.py` - eval-vector fingerprints

- `sample_eval_points(rng, n_vars, m, low, high)`
- `eval_poly_points(poly, points)`
- `eval_distance(a, b)`

Used for target representation and per-node eval features.

### `node.py` - circuit node type

`Node` stores:
- `node_id`, `op`, `args`
- exact `poly`
- `depth`
- canonical `signature` (for dedup)
- optional cached `evals`

Op constants:
- `OP_VAR`, `OP_CONST`, `OP_ADD`, `OP_MUL`

### `builder.py` - `CircuitBuilder`

Incremental DAG builder with canonical deduplication:
- pre-populates leaves (`x0..x{n-1}`, optional constants)
- `add_add(i, j)` / `add_mul(i, j)`
- `set_output(node_id)`
- `clone()` - creates a deep copy for MCTS state snapshots (nodes are immutable dataclasses, so list copy suffices)

Commutativity is handled in signatures, so structurally equivalent `ADD/MUL` nodes are reused.

### `action_codec.py` - flat action mapping

Flattened action layout for `L` visible nodes:

```text
[ADD pairs] + [MUL pairs] + [SET_OUTPUT nodes] + [STOP]
```

Utilities:
- `pair_to_index`, `index_to_pair`
- `action_space_size(L)`
- `encode_action(...)`, `decode_action(...)`

### `factor_library.py` - cross-episode factor library

Session-level cache that tracks which polynomials the agent has built in past episodes.

Key features:
- **Factorization**: Decomposes target polynomials over Q using SymPy `factor_list()`.
  Non-trivial factors become subgoals for the episode.
- **Subgoal rewards**: Agent earns `factor_subgoal_reward` when constructing a factor.
  Extra `factor_library_bonus` if the factor was previously built.
- **Dynamic discovery**: When a library-known node is built, discovers new subgoals
  via T-v (additive) and T/v (multiplicative, via exact polynomial division).
- **Completion bonus**: Awarded when the circuit is one operation from the target.
- **HER compatibility**: Factor rewards are stripped from HER-relabeled transitions.

Public API:
- `FactorLibrary(n_vars)` — create empty library
- `factorize_target(target)` — factorize and return non-trivial factors
- `register(poly, step_num)` / `register_episode_nodes(nodes, n_initial)` — record successful constructions
- `contains(poly)` / `filter_known(factors)` — library membership queries
- `exact_quotient(dividend, divisor)` — exact polynomial division over Q
