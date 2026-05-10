# Factor Library — Worked Example

A concrete walkthrough of what happens internally when the factor library is active, from the training command to the reward signal.

---

## Setup

You run:
```bash
python -m src.main --algorithm sac --iterations 200
```

During `SACTrainer.__init__()`, this is created:
```python
factor_library = FactorLibrary(mod=5, n_vars=2, max_degree=6)
env = CircuitGame(config, factor_library=factor_library)
```

The library is empty. It also pre-computes the **base node keys** — canonical fingerprints of `x0`, `x1`, and `1` — because those are always free starting nodes and should never be treated as subgoals.

---

## Episode Starts — `env.reset(target_poly)`

Say the sampled target is:

$$f^* = x_0^2 + 2x_0 + 1$$

which over the integers is $(x_0 + 1)^2$.

Inside `reset()`, this line runs **once**:
```python
factors = self.factor_library.factorize_target(target_poly)
```

### Inside `factorize_target` → `factorize_poly`

**Step 1** — Convert `FastPoly` → SymPy expression using `fast_to_sympy`:
```
x0**2 + 2*x0 + 1
```

**Step 2** — Call `sympy.factor_list(x0**2 + 2*x0 + 1, x0, x1)` over the integers (Z).

SymPy returns:
```python
(1,  [(x0 + 1,  2)])
#^       ^       ^
# integer  factor  multiplicity
# content  expr
```

The `1` is the integer GCD of all coefficients (the "content"). The list says: the only irreducible factor over Z is `x0 + 1`, appearing with multiplicity 2.

**Step 3** — Iterate over `[(x0 + 1, 2)]` and apply filters:
- Is `x0 + 1` a pure number? **No.**
- Is its total degree 0? **No**, it's degree 1.
- Convert to `FastPoly` (reduces coefficients mod 5): produces `coeffs[0,0]=1, coeffs[1,0]=1` — i.e., `1 + x0`.
- Is it the zero polynomial after mod-p reduction? **No.**
- Is its canonical key equal to the poly's key? **No** (target is degree 2, factor is degree 1).
- Is its canonical key one of the base keys (`x0`, `x1`, or `1`)? **No** — `x0+1` is distinct from bare `x0`.
- Is it in `exclude_keys`? **No** (exclude_keys = {target_key}, and factor ≠ target).
- Has it appeared already in the result (deduplication)? **No** — first time.
- ✅ Add to result.

Multiplicity is **ignored** — `(x0+1, 2)` and `(x0+1, 3)` both produce a single subgoal entry `{x0+1}`.

**`factorize_target` returns:** `[FastPoly(x0 + 1)]`

Back in `reset()`:
```python
self._subgoal_keys        = { canonical_key(x0+1) }  # bytes fingerprint
self._library_known_keys  = {}                        # library is empty → nothing known yet
self._subgoals_hit        = set()                     # nothing collected yet
self._additive_complete_hit   = False
self._mult_complete_hit       = False
```

SymPy is now done for reset. The rest of the episode uses fast operations.

---

## During the Episode — `env.step(action)`

### Step A: Agent builds `x0 + 1`

The agent selects **"add node 0 (x0) and node 2 (constant 1)"**, producing:

$$v_{\text{new}} = x_0 + 1$$

Inside `step()`, after computing and appending `new_poly`:

```python
new_key = new_poly.canonical_key()
# new_key == canonical_key(x0+1)  ← same bytes as the subgoal fingerprint

if new_key in self._subgoal_keys and new_key not in self._subgoals_hit:  # YES ✓
    self._subgoals_hit.add(new_key)
    factor_hit = True
    reward += 1.0                    # factor_subgoal_reward

    if new_key in self._library_known_keys:  # NO ✗ (library is empty)
        ...
```

**Completion + dynamic discovery block** runs next (library is empty, so `contains(new_poly)` is False → SymPy gated block skipped):

```python
residual = target - new_poly  # = (x0^2 + 2x0 + 1) - (x0 + 1) = x0^2 + x0

# Additive completion: is x0^2 + x0 in [x0, x1, 1]? NO.
# Library gate: factor_library.contains(x0+1) → False (library empty). Skip.
```

**Reward this step:** `step_penalty(−0.1) + shaping(≈0.x) + factor_subgoal_reward(+1.0)`

---

## Episode Succeeds

The agent then adds `(x0+1)` to itself (multiply: `x0+1` × `x0+1`), producing `x0^2 + 2x0 + 1` which matches the target.

Before `step()` returns:

```python
if is_success and self.factor_library is not None:
    n_initial = 3  # x0, x1, constant_1
    self.factor_library.register_episode_nodes(self.nodes, n_initial)
```

`self.nodes` at this point is `[x0, x1, 1, x0+1, x0^2+2x0+1]`.
The slice `nodes[3:]` gives `[x0+1, x0^2+2x0+1]`. Both get registered:

```
library = {
    canonical_key(x0+1):        step_num=1,
    canonical_key(x0^2+2x0+1): step_num=2,
}
```

---

## Next Episode — Same Target

`env.reset(target_poly)` runs again. `factorize_target` returns `[FastPoly(x0+1)]` as before. But now:

```python
self._library_known_keys = lib.filter_known([FastPoly(x0+1)])
# → { canonical_key(x0+1) }   ← it's in the library now!
```

### Step A: Agent builds `x0 + 1` again

```python
reward += 1.0   # factor_subgoal_reward  (factor of the target)
reward += 0.5   # factor_library_bonus   (was also seen in a prior successful episode)
```

**Dynamic discovery now runs** because `factor_library.contains(x0+1)` is True:

```python
residual = target - (x0+1)  # = x0^2 + 2x0 + 1 - x0 - 1 = x0^2 + x0

# 2a. Add residual as direct subgoal:
#   x0^2 + x0 is non-zero, not a base node → add to _subgoal_keys
#   _subgoal_keys now = { key(x0+1), key(x0^2+x0) }

# 2b. Factorize x0^2 + x0 = x0*(x0+1):
#   Factors: x0 (filtered → base node), x0+1 (already in _subgoal_keys → excluded)
#   → no new factors from this

# 2c. Exact quotient: T / (x0+1) = x0+1 (exact! remainder = 0 mod 5)
#   quotient = x0+1, which is NOT scalar
#   Multiplicative completion: is x0+1 in existing_keys?
#     existing_keys = {key(x0), key(x1), key(1)}  ← nodes BEFORE x0+1 was added
#     x0+1 is NOT in that set yet → completion bonus does NOT fire here
#   x0+1 is already in _subgoal_keys → not added again
```

### Step B: Agent multiplies `x0+1` × `x0+1`

The circuit is now `[x0, x1, 1, x0+1]`. New poly = `(x0+1)^2 = x0^2+2x0+1` = target.
`is_success = True` → `success_reward(+10.0)` fires. The completion/discovery block is **skipped** (`not is_success` guard).

---

## Completion Bonus in Action

Consider a different episode where `target = x0 + x1 + 1`.

The circuit starts as `[x0, x1, 1]`. The agent builds `x0 + x1`:

```python
new_poly = x0 + x1

residual = target - new_poly  # = (x0+x1+1) - (x0+x1) = 1

existing_keys = {key(x0), key(x1), key(1)}  # nodes BEFORE this step

# Additive completion: key(1) in existing_keys?  YES! ✓
reward += 3.0   # completion_bonus
self._additive_complete_hit = True
```

The agent is one ADD away from the target (`(x0+x1) + 1 = x0+x1+1`). It gets a +3.0 bonus to strongly encourage doing that final step.

---

## What Counts as a Factor — Reference Table

| Target | `sympy.factor_list` result | Initial subgoals |
|--------|--------------------------|------------------|
| `(x0+1)^3` | `(1, [(x0+1, 3)])` | `{x0+1}` |
| `3*(x0+1)^2` | `(3, [(x0+1, 2)])` | `{x0+1}` — scalar `3` filtered |
| `(x0+1)*(x1+2)` | `(1, [(x0+1,1), (x1+2,1)])` | `{x0+1, x1+2}` — two independent subgoals |
| `x0*(x0+1)` | `(1, [(x0,1), (x0+1,1)])` | `{x0+1}` — `x0` is a base node, filtered |
| `x0 + x1` | `(1, [(x0+x1, 1)])` | `{}` — irreducible; single factor = target itself, filtered |
| `x0^2 + x1` | irreducible | `{}` — no subgoals |

**A factor counts if and only if:**
1. It is a polynomial of total degree ≥ 1 (not a pure scalar like `3`).
2. It is not one of the free starting nodes (`x0`, `x1`, ..., `1`).
3. It is not identical to the polynomial being factorized.
4. It is not the zero polynomial after mod-p reduction.

---

## Dynamic Subgoals — Reference Table

After the agent builds a library-known node `v`, SymPy may discover additional subgoals:

| `v` built | Target `T` | `T - v` | Dynamic additive subgoals | `T / v` | Dynamic multiplicative subgoals |
|-----------|-----------|---------|--------------------------|---------|--------------------------------|
| `x0+1` | `(x0+1)^2` | `x0^2+x0` | `{x0^2+x0}` (direct) | `x0+1` | already known |
| `x0+1` | `(x0+1)*(x1+2)` | `x1^2+2x1+x0*x1+2x0-1` | factors of that | `x1+2` | `{x1+2}` added as subgoal |
| `x0+1` | `3*(x0+1)` | `2*x0+2 = 2*(x0+1)` | `{}` (factor = target) | `3` (scalar!) | completion bonus if 3 in circuit |
