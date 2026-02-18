# core/ - Polynomial arithmetic and circuit primitives

Mathematical foundations used by the environment and agent.

## Files

### `poly.py` - exact sparse polynomial arithmetic

`Poly = Dict[Tuple[int, ...], Fraction]`

Key functions:
- constructors: `make_zero`, `make_const`, `make_var`
- ops: `add`, `sub`, `mul`, `canonicalize`, `equal`
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
