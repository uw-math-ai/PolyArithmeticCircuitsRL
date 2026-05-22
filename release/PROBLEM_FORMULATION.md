# Problem formulation, the cost model, and how Ck is determined

This document explains what the agent is actually solving, how circuit cost is
counted, how the shortest-circuit complexity **Ck** is computed, and how to read
the success metric — with worked examples. Read this before the per-algorithm
analyses (`figures/ppo/ANALYSIS.md`, `figures/sac/ANALYSIS.md`) and the
`figures/comparison/COMPARISON.md`.

---

## 1. The problem

Given a sparse multivariate polynomial `f` over a finite field `F_p`, build an
**arithmetic circuit** (a straight-line program of `+` and `×`) that evaluates
`f`, using as few operations as possible. Example: `x² + 2xy + y²` over F₃ is
best built as `(x + y)²` — one addition then one squaring — rather than from its
five-ish naive monomial operations.

We restrict to a structured, top-down construction (below), so "shortest" means
shortest within that construction's move set, which for small polynomials
matches the true optimum in practice.

## 2. Cost model (what one "op" is)

Implemented in `src/decomp_rl/cost_model.py`:

- **1 op = one polynomial addition OR one polynomial multiplication.**
- **Free:** multiplying by a field scalar, and the bare variables `x, y, z`
  themselves.
- A power `x^k` is built by **repeated squaring**:
  `rep_sq(k) = (bitlength(k) − 1) + (popcount(k) − 1)` multiplications
  (e.g. `x²` = 1 mult, `x³` = 2, `x⁴` = 2, `x⁵` = 3).
- A monomial costs its power-builds plus `(#distinct variables − 1)` mults to
  combine them.
- Sparse-direct construction of `f` = sum of each monomial's cost +
  `(support_size − 1)` additions. This is the naive baseline.

So "Ck" / "C5" etc. just means **the circuit uses k operations**.

## 3. The top-down decomposition MDP

Implemented in `src/decomp_rl/decomp_env.py`. An episode maintains a **frontier**
of not-yet-built polynomials, starting with the target `f`. At each step the
agent acts on `frontier[0]` by choosing one **action** from a variable-sized
candidate set:

1. **Additive split** `f = g + h` (costs **+1** for the addition). Each piece
   `g, h` is then **automatically factored** over `F_p`; cheap factors are
   rebuilt immediately and unresolved factors are pushed back onto the frontier.
2. **Whole-polynomial factorization** `f = c · ∏ qᵢ^{eᵢ}` (costs **+0** for an
   addition — there is none; you pay only the `rebuild_cost` to recombine the
   irreducible factors). This is what lets perfect powers like `(x+y)²` reach
   their true optimum; without it, additive-split-only agents cannot.

The episode ends when the frontier is empty; the accumulated op count is the
circuit cost. Two ingredients make this efficient:

- **Factorizer** (`factor_fp.py`) — correct multivariate factorization over
  `F_p` via a FLINT backend (`python-flint`); it also *populates a live*
  **`FactorizableLibrary`** of useful factorizations it discovers.
- **Reward** = cost saved vs the current baseline estimate + a small bonus when
  a split piece hits the factor library + a terminal bonus when the finished
  circuit beats the minimum of five closed-form/search baselines. **The reward
  never uses Ck**, so there is no leakage from the evaluation metric into
  training.

## 4. How Ck (shortest-circuit complexity) is determined

Implemented in `src/decomp_rl/complexity.py`. For a given polynomial we want the
minimum op count over the construction moves. We compute it with a **memoized
exhaustive search** that, at every node, takes the best of:

- **direct build** (the sparse-direct cost — an upper bound), and
- **whole-poly factorization**: `rebuild_cost(factors) + Σ cost(distinct
  factor)`, and
- **additive splits** `f = g + h`: `1 + rebuild(g) + rebuild(h) + Σ cost(child)`.

Every child is strictly smaller (proper factors / split halves), so the
recursion terminates; memoization keeps it fast. This is the **hybrid** Ck:

- **exact** for small polynomials (support ≤ 7 and total degree ≤ 6), which is
  the regime we report and trust;
- a **multi-baseline upper bound** (`BaselineBundle.min_cost`) for larger ones
  (labeled `method = "heuristic"`).

`circuit_complexity(poly, ...)` returns `(ck, method)`.

### Why Ck is a fair, tight bar
The agent's move set (split / factor / direct) is a **subset** of what the Ck
search explores, so the agent's circuit cost is always **≥ Ck**. Therefore a
"match" (`cost ≤ Ck`) means the agent found the **verified shortest** circuit —
not merely a good one. (For the few `heuristic`-labeled targets, Ck is an upper
bound, so "match" there is weaker; we scope verified-optimal claims to C2–C8.)

## 5. Worked examples (all over F₃, vars x, y, z)

| polynomial | Ck | shortest circuit | why |
| --- | --- | --- | --- |
| `x + y` | **1** | `x + y` | one addition |
| `x² + 2xy + y²` | **2** | `(x + y)²` | build `x+y` (1 add), square it (1 mult) — needs the **factor action** |
| `x² + y²` | **3** | `x²·1 + y²` | irreducible over F₃; `x²` (1 mult) + `y²` (1 mult) + 1 add |
| `x³ + 1` | **3** | `(x + 1)³` | over F₃, `(x+1)³ = x³+1` (Frobenius); build `x+1` (1 add) + cube (`rep_sq(3)=2` mults) |
| `xy + x + y` | **3** | — | `xy` (1 mult) + 2 adds |
| `(x+1)(y+1)` | **3** | `(x+1)(y+1)` | `x+1` (1 add) + `y+1` (1 add) + 1 mult |

### A C5 example, step by step
**Target:** `2xyz + xz + xy + 2y + 2x` (Ck = 5, robust at search widths 12/24/40).
The trained agent's greedy circuit (cost 5):
```
SPLIT  target = [2y] + [2xyz + xz + xy + 2x]
   └ the right piece factors as  2 · x · (y+2) · (z+2)   (2 mults to combine; scalar 2 is free)
SPLIT  z + 2 = [z] + [2]      (1 add)
SPLIT  y + 2 = [y] + [2]      (1 add)
```
Total = 1 (top add) + 2 (factor combine) + 1 + 1 = **5 ops**. Verified to compute
the target: `2·x·(y+2)·(z+2) + 2y = 2xyz + xy + xz + 2x + 2y` ✓. The environment
guarantees correctness — every step checks `g + h == node` and every
factorization is checked to reconstruct its node.

## 6. The success metric (and the random baseline)

- **Ck-match rate** = fraction of targets where the greedy policy's circuit cost
  reaches Ck (`cost ≤ Ck`) — i.e. it found the shortest circuit.
- **Gap to optimal** = mean `(cost − Ck)` in ops.
- **Random reference** = a uniform-random split/factor policy over the *same*
  candidate sets, scored the same way (per-Ck and overall). Reported as a flat
  line; it climbs on tiny circuits (C2 ≈ 0.8) and collapses as complexity grows
  (C10 ≈ 0.16), so the learned policy's *lift* — and how it **widens with
  complexity** — is the real signal.
- **Train / held-out split.** Models are evaluated on a disjoint held-out set;
  matching train and held-out rates demonstrate generalization, not memorization.

## 7. Scope and limitations

- Everything is in the **exact-Ck regime** (≤ 7 terms); larger polynomials are
  skipped in verified evaluations because their optimum is not certifiable.
  C9–C10 buckets are partly heuristic-labeled.
- "Shortest" = optimal within {additive split, whole-poly factorization, direct
  build}. Cross-circuit common-subexpression sharing across unrelated branches
  is not modeled; for the small polynomials here it is almost always the true
  optimum, but claims should be scoped to this move set.
- Field/vars fixed to F₃ / (x, y, z) for these runs; the pipeline is
  field- and variable-agnostic.

## 8. Pointers
- Cost model: `src/decomp_rl/cost_model.py`
- Environment + actions: `src/decomp_rl/decomp_env.py`,
  `src/decomp_rl/split_proposals.py` (`factor_whole_action`)
- Factorizer + library: `src/decomp_rl/factor_fp.py`,
  `src/decomp_rl/factor_library.py`
- Ck: `src/decomp_rl/complexity.py`
- Trainers: `src/decomp_rl/train_ppo.py`, `src/decomp_rl/train_sac.py`
- Launcher (PPO/SAC, shared eval): `scripts/run_ppo_curriculum.py`
- Full project handoff: `../HANDOFF.md`
