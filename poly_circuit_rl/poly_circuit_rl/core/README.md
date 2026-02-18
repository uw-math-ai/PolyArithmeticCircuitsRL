# core/ — Polynomial Arithmetic and Circuit Primitives

This module provides the mathematical foundation: sparse polynomial arithmetic, circuit construction with signature-based deduplication, and a flat action encoding for the RL agent.

## Files

### poly.py — Sparse Polynomial Arithmetic

The central type is `Poly = Dict[Tuple[int, ...], Fraction]`, mapping exponent tuples to exact rational coefficients.

For a 2-variable system:
- `x0` → `{(1, 0): Fraction(1)}`
- `2*x0*x1 + 3` → `{(1, 1): Fraction(2), (0, 0): Fraction(3)}`

Key functions:
- `make_var(n_vars, idx)` — create a single variable polynomial
- `make_const(n_vars, value)` — create a constant polynomial
- `add(a, b)`, `mul(a, b)`, `sub(a, b)` — polynomial arithmetic
- `equal(a, b)` — exact equality check (both canonicalized)
- `eval_poly(poly, values)` — evaluate at a point, returns `Fraction`
- `stats(poly, n_vars)` — degree, number of terms, L1 norm

Uses `Fraction` throughout for exact arithmetic — no floating point errors in polynomial identity checks.

### fingerprints.py — Evaluation-Based Fingerprinting

Implements the Schwartz-Zippel polynomial fingerprinting scheme: evaluate a polynomial at `m` random points to get an evaluation vector. Two polynomials are (probabilistically) equal iff their eval vectors match.

- `sample_eval_points(rng, n_vars, m, low, high)` — generate m random evaluation points
- `eval_poly_points(poly, points)` — evaluate polynomial at all points, returns `List[Fraction]`
- `eval_distance(a, b)` — L1 distance between two eval vectors

The eval vector is the primary per-node feature in the observation space and the target representation for goal-conditioned RL.

### node.py — Circuit Node

```python
@dataclass
class Node:
    node_id: int                    # index in builder.nodes
    op: str                         # OP_VAR, OP_CONST, OP_ADD, OP_MUL
    args: Tuple[int, int] | Tuple   # parent node IDs (for ADD/MUL), or (var_idx,) / (value,)
    poly: Poly                      # the polynomial this node computes
    depth: int                      # circuit depth
    signature: Tuple                # canonical form for deduplication
    evals: Optional[Tuple[Fraction, ...]]  # cached eval vector
```

### builder.py — Circuit Builder

`CircuitBuilder` manages the incremental construction of arithmetic circuits:

- Initializes with input variables (`x0, x1, ...`) and optionally `const_1`
- `add_add(i, j)` / `add_mul(i, j)` — create a new node combining nodes i and j
- `set_output(node_id)` — designate a node as the circuit output
- **Signature deduplication**: if `x0 + x1` already exists and you try to add it again, the builder returns the existing node instead of creating a duplicate

The builder automatically computes and caches the polynomial and eval vector for each node.

### action_codec.py — Flat Action Encoding

Maps structured actions `(op, i, j)` to a single integer for DQN compatibility:

```
Action space layout (for L visible nodes):
  [0, pairs)           → ADD(i, j) for all i ≤ j
  [pairs, 2*pairs)     → MUL(i, j) for all i ≤ j
  [2*pairs, 2*pairs+L) → SET_OUTPUT(i)
  [2*pairs+L]          → STOP

  where pairs = L*(L+1)/2
```

- `encode_action(kind, i, j, L)` → flat action ID
- `decode_action(action_id, L)` → `DecodedAction(kind, i, j)`
- `action_space_size(L)` → total number of actions
