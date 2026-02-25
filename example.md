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

### Inside `factorize_target`

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
- Is its canonical key equal to the target's key? **No** (target is degree 2, factor is degree 1).
- Is its canonical key one of the base keys (`x0`, `x1`, or `1`)? **No** — `x0+1` is distinct from bare `x0`.
- Has it appeared already in the result (deduplication)? **No** — first time.
- ✅ Add to result.

Multiplicity is **ignored** — `(x0+1, 2)` and `(x0+1, 3)` both produce a single subgoal entry `{x0+1}`. We only care that it's a factor, not how many times it appears.

**`factorize_target` returns:** `[FastPoly(x0 + 1)]`

Back in `reset()`:
```python
self._subgoal_keys        = { canonical_key(x0+1) }  # bytes fingerprint of the coefficient array
self._library_known_keys  = {}                        # library is empty → nothing known yet
self._subgoals_hit        = set()                     # nothing collected this episode yet
```

SymPy is now done. The rest of the episode uses only O(1) set lookups.

---

## During the Episode — `env.step(action)`

The agent takes several steps. At some point it selects the action **"add node 0 (x0) and node 2 (constant 1)"**, producing:

$$v_{\text{new}} = x_0 + 1$$

Inside `step()`, after computing `new_poly`:

```python
new_key = new_poly.canonical_key()
# new_key == canonical_key(x0+1)  ← same bytes as the subgoal fingerprint

if new_key in self._subgoal_keys:          # YES ✓
    if new_key not in self._subgoals_hit:  # YES ✓ (first time this episode)
        self._subgoals_hit.add(new_key)
        factor_hit = True
        reward += 1.0                      # factor_subgoal_reward

        if new_key in self._library_known_keys:  # NO ✗ (library is still empty)
            ...
```

**Reward this step:** `step_penalty(−0.1) + shaping(≈0.x) + factor_subgoal_reward(+1.0)`

If the agent somehow builds `x0+1` again later in the same episode, it gets **nothing extra** — `_subgoals_hit` prevents double-rewarding the same factor.

---

## Episode Succeeds

After the agent eventually builds `x0^2 + 2x0 + 1` and matches the target, before `step()` returns:

```python
if is_success and self.factor_library is not None:
    n_initial = 3  # x0, x1, constant_1
    self.factor_library.register_episode_nodes(self.nodes, n_initial)
```

`self.nodes` at this point is something like `[x0, x1, 1, x0+1, x0^2+2x0+1]`.
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

When the agent builds `x0+1` this time:

```python
reward += 1.0   # factor_subgoal_reward  (factor of the target)
reward += 0.5   # factor_library_bonus   (was also seen in a prior successful episode)
```

Total factor reward this step: **+1.5** instead of +1.0. The agent is being rewarded for recognising a sub-computation it has already learned.

---

## What Counts as a Factor — Reference Table

| Target | `sympy.factor_list` result | Subgoals produced |
|--------|--------------------------|-------------------|
| `(x0+1)^3` | `(1, [(x0+1, 3)])` | `{x0+1}` |
| `3*(x0+1)^2` | `(3, [(x0+1, 2)])` | `{x0+1}` — scalar `3` filtered (not a polynomial factor) |
| `(x0+1)*(x1+2)` | `(1, [(x0+1,1), (x1+2,1)])` | `{x0+1, x1+2}` — two independent subgoals |
| `x0*(x0+1)` | `(1, [(x0,1), (x0+1,1)])` | `{x0+1}` — `x0` is a base node, filtered |
| `x0 + x1` | `(1, [(x0+x1, 1)])` | `{}` — irreducible; single factor equals the target itself, filtered |
| `x0^2 + x1` | irreducible | `{}` — no subgoals, no factor rewards |

**A factor counts if and only if:**
1. It is a polynomial of total degree ≥ 1 (not a pure scalar like `3`).
2. It is not one of the free starting nodes (`x0`, `x1`, ..., `1`).
3. It is not identical to the target polynomial itself.
4. It is not the zero polynomial after mod-p reduction.
