# baselines/ - Symbolic and search baselines

Reference baselines for evaluating agent performance.

## Files

### `exhaustive.py` - `ExhaustiveSearch`

BFS layer-by-layer enumeration of all reachable polynomials up to `max_ops`.

Pipeline:
1. Layer 0: all base polynomials (variables `x0..x{n-1}` + constant `1`)
2. Each subsequent layer: combine all pairs from prior layers via ADD and MUL
3. Track minimum op count per unique polynomial (keyed by `poly_hashkey`)

Public API:
- `build(max_ops)` — enumerate up to given depth
- `find_optimal(target) -> Optional[int]` — minimum ops, or None if unreachable
- `gap_to_optimal(target, agent_ops) -> Optional[int]` — `agent_ops - optimal_ops`
- `reachable_count() -> int` — number of distinct reachable polynomials

Practical for small instances (2 vars, <=4 ops). Provides ground-truth optimal
op counts for NeurIPS gap-to-optimal metrics.

### `greedy.py` - `GreedyBaseline`

Greedy heuristic: at each step, pick the ADD/MUL pair whose result has minimum
L1 eval-distance to the target.

Pipeline:
1. Initialize `CircuitBuilder` with eval points
2. At each step, clone builder and test all valid (op, i, j) combinations
3. Select the action minimizing `sum(|result_eval - target_eval|)`
4. Stop when target is matched or `max_ops` exhausted

Public API:
- `solve(target, max_ops) -> dict` — returns `solved`, `ops_used`, `nodes_built`
- `evaluate_batch(targets, max_ops) -> dict` — aggregate stats over target list

### `factorization.py` - `FactorizationBaseline`

Uses SymPy `factor()` to produce algebraically restructured expressions and
estimates operation count from resulting ADD/MUL structure.

Public API:
- `solve(target, max_ops=None) -> dict`
- `evaluate_batch(targets, max_ops=None) -> dict`

### `horner.py` - `HornerBaseline`

Applies nested Horner decomposition (`sympy.horner`) over a selected variable
order, then estimates operation count from the transformed expression.

Public API:
- `solve(target, max_ops=None, var_order=None) -> dict`
- `evaluate_batch(targets, max_ops=None, var_order=None) -> dict`

### `memoized.py` - `MemoizedCSEBaseline`

Uses common-subexpression elimination (`sympy.cse`) as a memoized reuse baseline
and scores the resulting temporary assignments + reduced expression.

Public API:
- `solve(target, max_ops=None) -> dict`
- `evaluate_batch(targets, max_ops=None) -> dict`
