# Related Work And Rationale

TODO: Replace citation placeholders with exact references from the provided papers before public release.

This document explains the rationale for the current learned-search design. It does not claim final experimental results or broad generalization. The current contribution should be framed as algebraically informed learned symbolic search, centered on a compact high-recall target-conditioned candidate interface.

## AI-For-CO Framing

Polynomial arithmetic circuit synthesis can be framed as deterministic constructive combinatorial optimization:

- state: current arithmetic circuit
- action: add/mul candidate
- transition: exact symbolic operation
- objective: construct the target with a small verified circuit

This resembles deterministic MDP formulations used in combinatorial optimization, where feasible actions are state-dependent and the goal is to construct a high-quality solution step by step. [TODO: cite RL4CO/AI4CO survey]

The raw action space is large because every state with `m` nodes has roughly:

```text
2 * m * (m + 1) / 2
```

canonical add/mul actions. The difficulty is not just choosing an operation. It is finding algebraically useful intermediate expressions that may only pay off several steps later.

## Why Learned Search, Not Pure RL

The design is motivated by a conservative reading of AI-for-CO and RL-for-CO discussions: learned components often work best when they guide search, pruning, construction heuristics, or local improvement rather than replacing combinatorial reasoning entirely. [TODO: cite neural combinatorial optimization survey]

Pure model-free construction is fragile here because:

- branching factors grow with the circuit state
- exact success rewards are sparse
- shaped rewards can bias toward misleading local patterns
- useful intermediates may not directly overlap target support
- failure modes are hard to diagnose

In this project, learning is inserted as a candidate ranking heuristic inside exact symbolic planners. The symbolic backend and exact verifier remain responsible for correctness.

## Why Candidate Generation

Candidate generation reduces the effective branching factor and exposes meaningful choices to the model.

The current generator still enumerates all canonical add/mul pairs, but it applies exact symbolic computation, safe filtering, deduplication, and target-conditioned features before ranking. This makes the learning problem:

```text
rank candidates for this exact target and state
```

rather than:

```text
discover all algebraic structure and long-horizon action consequences from scratch
```

Candidate generation is also a useful diagnostic boundary. Candidate recall@K and candidate-rank diagnostics measure whether actions on a verified best trace appear in the generated candidate list at each source state. If search fails because the needed action never appears in the candidate set, the next step is candidate generation or learned proposal. If the candidate appears but search drops the branch, the next step may be value guidance, MCTS, or a better planner.

## Why Support Geometry

Polynomial supports are finite sets of exponent vectors. This gives a lightweight algebraic geometry signal without leaving exact finite-field arithmetic.

Multiplication induces a Minkowski-style relationship between operand supports and product support. Addition is closer to support union and residual coverage. These observations motivate support-geometry candidate features for examples such as:

- `x + y` as a reusable factor for `(x + y)^2`
- `a + b` and `c + d` for `(a + b)(c + d)`
- reused subexpressions in `(x + y)^2 + z(x + y)`

These are heuristic signals, not proof certificates. Exact verification remains the only correctness boundary.

## Why Beam Search First

Beam search is simple, deterministic, and debuggable. It gives clear ablations:

- heuristic-only
- heuristic plus learned ranker
- different beam widths
- different candidate budgets

Before adding value heads or full MCTS, beam search is enough to test whether learned candidate ranking improves success, expansions, or solution size under fixed budgets.

## Why Gumbel / Sequential-Halving Now

The current Gumbel planner is a bounded planner over already generated symbolic candidates. It uses heuristic/model total scores as logits, adds Gumbel noise for selection, and applies a simple Sequential-Halving style branch filter. This tests whether stochastic branch preservation helps when deterministic beam search drops delayed-good states under tight budgets.

This is not claimed to be a full AlphaZero-style MCTS system:

- no value head
- no visit-count training target
- no PPO/RL loop
- no learned candidate proposal

Its role is diagnostic and comparative. The matched-budget planner sweep compares:

- `beam_heuristic`
- `beam_guided`
- `gumbel_heuristic`
- `gumbel_guided`

using explicit expansion accounting.

## Why Exact Verification Matters

Unlike approximate neural program generation, every returned circuit is checked by exact polynomial equality. The model may guide the search toward a candidate, but it cannot certify correctness.

This matters because arithmetic circuit synthesis has a hard correctness constraint. Approximate equality, floating-point evaluation, or probabilistic testing is not the correctness boundary in this implementation.

## Previous Direction Versus Current Direction

The previous setup in the sibling `gumbel` direction was closer to flat-action RL or reward-shaped construction. That style can be useful as a baseline, but the current approach addresses several likely failure modes directly.

| Failure Mode | Previous Direction | Current Direction |
| --- | --- | --- |
| Action explosion | policy faces many raw `(op, i, j)` choices | candidate generation and deduplication structure the choice set |
| Sparse rewards | reward shaping needed to provide signal | preferences are extracted from successful search histories |
| Useful intermediates | model must discover them through trial and reward | permissive candidate generation keeps low-overlap intermediates |
| Diagnosability | failures can mix reward, policy, representation, and search | separate ablations for candidate generation, heuristic, ranker, Beam, and Gumbel |
| Correctness | training signal and verification can be entangled | exact verifier is the final boundary |

The current approach is not claimed to be universally better. It is a cleaner experimental decomposition for testing learned symbolic search.

## Hypothesis

The working hypothesis is:

```text
Given a structured candidate set, exact symbolic features, and fair expansion accounting,
a learned ranker and/or Gumbel-style planner can improve symbolic search under fixed budgets.
```

This should be evaluated against heuristic-only beam search on held-out structured families and tighter search budgets. Planner gains, ranker gains, and combined gains should be separated with controlled four-method sweeps.

## Citation Placeholders

Replace these before public release:

- [TODO: cite RL4CO/AI4CO survey]
- [TODO: cite neural combinatorial optimization survey]
- [TODO: cite learning-to-search or learning-augmented search references]
- [TODO: cite symbolic regression or program synthesis baselines if used]
- [TODO: cite arithmetic circuit complexity background if discussed]
