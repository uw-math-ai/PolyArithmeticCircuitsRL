# Learned Symbolic Search for Polynomial Circuit Synthesis

This README specifies the **main implementable training pipeline** for a polynomial arithmetic circuit synthesis solver.

The solver is framed as an **AI-for-combinatorial-optimization style learned search system**:

```text
ProblemGenerator
    ↓
ProblemInstance
    ↓
SearchState / exact symbolic environment
    ↓
CandidateGenerator
    ↓
CandidatePolicy / learned ranker
    ↓
Planner / decoder: beam search first, MCTS later
    ↓
Verifier
    ↓
Search-history-to-preference Trainer
```

The key idea is:

> Do not train a flat-action RL agent to choose directly from every possible `(op, i, j)` action. Instead, generate a compact target-conditioned candidate set, learn to rank those candidate transitions, and use search to plan multi-step circuit constructions.

This README focuses only on the **main training pipeline**. Baselines, old RL systems, and rewrite-improvement extensions can be added later.

---

## 0. Core implementation decisions

These decisions should be treated as fixed unless explicitly changed.

### 0.1 Coefficient domain

Use a finite field:

```text
F_p
```

where `p` is a configurable prime.

Recommended default:

```text
a large but bounded prime, e.g. a 28-bit prime
```

The exact value should live in one config field:

```yaml
field:
  p: 268435399   # example; can be changed
```

Reason:

```text
finite-field arithmetic gives exact equality,
bounded coefficient representation,
stable model features,
and finite reachable state spaces for small diagnostic boards.
```

Avoid very small primes such as `p = 5` or `p = 7` except for fast diagnostics, because small primes create unnatural coefficient collisions.

### 0.2 Overflow safety

If the polynomial backend uses fixed-width integer arithmetic, multiplication must be protected by runtime assertions.

A conservative check is:

```text
accumulation_count * (p - 1)^2 < 2^63
```

where `accumulation_count` is the maximum number of product terms that may contribute to a single output coefficient during multiplication.

If the condition fails, raise an error.

Never silently overflow.

### 0.3 Polynomial semantics

Use one consistent polynomial representation across the repo.

If the backend stores dense coefficient tensors with a per-variable degree cap `d`, then the implemented ring is:

```text
F_p[x_0, ..., x_{n-1}] / (x_0^{d+1}, ..., x_{n-1}^{d+1})
```

In this case:

1. `d` must be at least the maximum per-variable degree needed by the target family.
2. conversion from external symbolic form to internal polynomial form must raise if a monomial exceeds the degree cap;
3. multiplication must not silently drop terms unless quotient-ring semantics are explicitly intended;
4. tests must verify that known-good target constructions do not unintentionally truncate.

If the intended semantics are ordinary polynomials rather than the quotient ring, then the backend must not silently truncate high-degree terms.

### 0.4 Complexity definition

Complexity means:

```text
number of arithmetic gates / operations
```

Each `+` or `*` action that creates one new non-base node counts as one operation.

Thus:

```text
complexity 10 = at most 10 constructed gates after the initial variables/constants
```

Use this definition consistently in target generation, search depth, evaluation, and logging.

### 0.5 Exact verification

Correctness is decided only by exact polynomial equality.

The learned model and symbolic heuristic are search guides only. They must never decide correctness.

The implementation must follow this rule:

```text
exact arithmetic must fail loudly
```

That means:

```text
no silent truncation
no silent overflow
no silent operand-domain mismatch
no silent invalid action
no silent coefficient-layout mismatch
```

---

## 1. Method abstraction

This project should use a modular separation similar to AI-for-CO libraries, but adapted to exact symbolic circuit synthesis.

### 1.1 ProblemGenerator

A `ProblemGenerator` samples synthesis tasks from structured distributions.

Examples:

```text
CommonFactorGenerator
PowerOfSumGenerator
ProductOfSumsGenerator
HornerPolynomialGenerator
SymmetricPolynomialGenerator
RandomCircuitGenerator
```

It returns a `ProblemInstance`.

### 1.2 ProblemInstance

A `ProblemInstance` contains static data for one synthesis task:

```python
@dataclass(frozen=True)
class ProblemInstance:
    target: Polynomial
    variables: tuple[str, ...]
    field_p: int
    degree_cap: int
    op_budget: int
    family_name: str
    metadata: dict[str, Any]
```

It should not contain a solution trace unless explicitly used for tiny diagnostics.

### 1.3 SearchState

A `SearchState` is the dynamic partial solution.

It contains:

```text
current circuit nodes
action trace
parent pointers
canonical node keys
remaining operation budget
```

This is analogous to a partial solution in constructive combinatorial optimization.

### 1.4 CandidateGenerator

The candidate generator defines the state-dependent feasible transition subset:

```python
CandidateGenerator(instance, state) -> list[Candidate]
```

It shrinks the raw action space:

```text
all canonical add/mul pairs
```

into a compact target-conditioned set.

The learned policy is defined over this candidate set, not over the raw global action space.

### 1.5 CandidatePolicy / Ranker

The learned model is a candidate policy or transition heuristic:

```python
π_θ(instance, state, candidate) -> scalar score
```

It does not directly output a full circuit. It scores candidate symbolic transitions.

### 1.6 Planner / Decoder

The planner converts candidate scores into a multi-step circuit construction.

Initial planner:

```text
beam search
```

Planned later:

```text
MCTS / Gumbel-style search with a value head
```

### 1.7 Verifier

The verifier checks exact polynomial equality.

It is the only component allowed to decide whether a circuit is correct.

### 1.8 Trainer

The trainer converts search histories into ranking preferences and trains the candidate policy.

---

## 2. Problem setting

Given a target polynomial:

```text
f(x_0, ..., x_{n-1})
```

we want to synthesize a small arithmetic circuit using binary operations:

```text
+
*
```

A circuit state is an ordered list of polynomial nodes:

```text
C = [v_0, v_1, ..., v_{m-1}]
```

The initial nodes are usually:

```text
x_0, x_1, ..., x_{n-1}, 1
```

An action is:

```text
(op, i, j)
```

where:

```text
op ∈ {add, mul}
0 <= i <= j < len(C)
```

Applying an action creates a new node:

```text
v_new = C[i] + C[j]      if op = add
v_new = C[i] * C[j]      if op = mul
```

A state succeeds if any node equals the target exactly:

```text
exists k such that C[k] == f
```

The solver searches for a verified circuit with small operation count under a fixed expansion budget.

---

## 3. Main training pipeline

The training pipeline is bootstrapped learning from search.

```text
initialize candidate policy / ranker π_θ

for each training round:
    choose a batch of target instances from the curriculum

    for each instance:
        run guided beam search using:
            - fresh target-conditioned candidate generation
            - symbolic heuristic scoring
            - current learned ranker π_θ

        collect:
            - verified circuits found
            - expanded states
            - candidate sets
            - selected actions
            - downstream outcomes

        convert search history into pairwise preferences:
            actions leading to better verified circuits
            >
            actions leading to worse or failed branches

    train π_θ on accumulated preferences

at test time:
    run guided search with candidate generator + heuristic + trained ranker
    return the smallest verified circuit found
```

No optimal path oracle is assumed for normal training. For larger complexity, optimal circuits are too expensive. The model learns from the best circuits found by search under a budget.

Tiny exact boards may be used only for diagnostics or optional warm-start on very small instances.

---

## 4. Recommended repository structure

```text
src/
  poly/
    fast_poly.py              # exact polynomial backend
    poly_utils.py             # degree/support/division/residual helpers

  env/
    problem_instance.py       # ProblemInstance dataclass
    circuit_state.py          # SearchState / circuit state
    action.py                 # Action dataclass
    verification.py           # exact equality and trace verification

  search/
    candidate.py              # Candidate dataclass
    candidate_generator.py    # fresh target-conditioned candidates
    heuristic_score.py        # cheap symbolic scoring
    beam_search.py            # first planner / decoder
    mcts.py                   # planned later
    search_history.py         # records expanded states and outcomes

  models/
    candidate_ranker.py       # score(instance, state, candidate) -> scalar
    feature_encoder.py        # engineered feature extraction
    value_model.py            # planned later

  training/
    collect_traces.py         # run search and collect histories
    preference_dataset.py     # convert histories to ranking examples
    train_ranker.py           # pairwise/listwise ranker training
    bootstrap_loop.py         # outer search-train loop

  data/
    target_generators.py      # structured polynomial families
    curriculum.py             # adaptive difficulty scheduling

  eval/
    evaluate_search.py        # solve rate, circuit size, expansions
    compare_rankers.py        # heuristic vs heuristic+ranker

  scripts/
    train_bootstrap.py
    run_search.py
    evaluate.py

  configs/
    default.yaml
```

---

## 5. Core data structures

### 5.1 Polynomial

The polynomial backend must support:

```python
p + q
p * q
p == q
key(p)
degree(p)
support(p)
coefficients(p)
exact_divides(f, g)
```

`key(p)` must be a canonical hashable representation. Equal keys must imply equal polynomials under the chosen exact semantics.

### 5.2 ProblemInstance

```python
@dataclass(frozen=True)
class ProblemInstance:
    target: Polynomial
    variables: tuple[str, ...]
    field_p: int
    degree_cap: int
    op_budget: int
    family_name: str
    metadata: dict[str, Any]
```

### 5.3 CircuitState / SearchState

```python
@dataclass
class CircuitState:
    nodes: list[Polynomial]
    actions: list[Action]
    parents: list[tuple[int, int, str] | None]
    node_keys: set[Hashable]
    op_budget: int
```

Required methods:

```python
state.apply(action) -> CircuitState
state.contains(target) -> bool
state.get_node(index) -> Polynomial
state.num_nodes() -> int
state.num_ops() -> int
state.remaining_budget() -> int
state.node_key(index) -> Hashable
```

`apply(action)` must:

1. validate operand indices;
2. validate remaining budget;
3. compute the exact result polynomial;
4. append the result node;
5. record parent/action metadata;
6. update `node_keys`;
7. raise on invalid arithmetic or backend mismatch.

### 5.4 Action

```python
@dataclass(frozen=True)
class Action:
    op: Literal["add", "mul"]
    i: int
    j: int
```

Always canonicalize operands:

```python
i <= j
```

because `+` and `*` are commutative in this setting.

### 5.5 Candidate

```python
@dataclass
class Candidate:
    action: Action
    result_poly: Polynomial
    source_tags: set[str]
    features: dict[str, float]
    tier1_score: float = 0.0
    tier2_score: float = 0.0
    heuristic_score: float = 0.0
    model_score: float = 0.0
    total_score: float = 0.0
```

Example source tags:

```text
basic_pair
target_factor_seed
degree_ok
support_overlap
residual_relevant
quotient_relevant
one_step_completion_add
one_step_completion_mul
two_step_hint
exact_target
duplicate_filtered
```

---

## 6. Candidate generation

Candidate generation is computed fresh from:

```text
ProblemInstance
current SearchState
```

It must not use stored cross-episode memories.

The candidate generator should use **two tiers**:

```text
Tier 1:
    cheap features over many raw actions

Tier 2:
    expensive symbolic checks only on top-M tier-1 candidates
```

This prevents exact division, factorization, and lookahead from running over every raw pair.

Recommended defaults:

```yaml
search:
  candidate_k: 64
  tier2_m: 128
```

---

### 6.1 Candidate generation overview

```python
def generate_candidates(
    instance: ProblemInstance,
    state: CircuitState,
    K: int,
    tier2_m: int,
) -> list[Candidate]:
    raw = enumerate_basic_pair_candidates(state)

    raw += proactive_target_factor_candidates(instance, state)

    tier1 = []
    for cand in raw:
        if hard_filter(cand, instance, state):
            continue
        cand.features.update(compute_tier1_features(cand, instance, state))
        cand.tier1_score = tier1_score(cand.features)
        tier1.append(cand)

    tier1 = unique_by_result_polynomial(tier1)
    tier1.sort(key=lambda c: c.tier1_score, reverse=True)

    tier2_pool = tier1[:tier2_m]

    for cand in tier2_pool:
        cand.features.update(compute_tier2_features(cand, instance, state))
        cand.tier2_score = tier2_score(cand.features)
        cand.heuristic_score = cand.tier1_score + cand.tier2_score

    tier2_pool.sort(key=lambda c: c.heuristic_score, reverse=True)
    return tier2_pool[:K]
```

---

### 6.2 Raw basic pair candidates

Enumerate all canonical add/mul pairs:

```python
for i in range(len(nodes)):
    for j in range(i, len(nodes)):
        yield Action("add", i, j)
        yield Action("mul", i, j)
```

Compute result polynomial:

```python
g = nodes[i] + nodes[j]
g = nodes[i] * nodes[j]
```

---

### 6.3 Proactive target factorization candidates

Add a stateless, target-conditioned target factorization step.

This is **not** a stored factor library. It is computed fresh for the current target.

At search start, optionally compute:

```text
factor_list(target over F_p)
```

The nontrivial factors become sub-target seeds.

Example:

```text
target = ab + ac = a(b+c)
```

Target factorization yields:

```text
a
b+c
```

The candidate generator should use these as scoring hints and candidate sources.

Candidate source examples:

```text
if an action result equals a target factor:
    tag target_factor_seed

if an action result divides a target factor:
    tag divides_target_factor

if an existing node times candidate result equals a target factor:
    tag builds_target_factor
```

Target factorization should be optional and cached per target. If it is unavailable or too expensive, candidate generation must still work from basic pairs, residual checks, quotient checks, and one-step completion.

---

### 6.4 Hard filters

Apply hard filters before expensive scoring.

#### Duplicate result

If the result polynomial already exists in the current state, usually drop it:

```python
if key(result) in state.node_keys:
    drop
```

Exception: if the action itself immediately creates the target, keep it.

#### Zero / invalid result

Drop zero if zero is not useful under the current operation set.

Raise on invalid backend/domain mismatch.

#### Degree filter

If ordinary polynomial semantics are intended, drop candidates with degree exceeding target degree:

```python
if degree(result) > degree(instance.target):
    drop
```

If quotient-ring semantics are used, the filter must match that semantics. Do not apply a filter that assumes a different ring from the backend.

#### Step/node budget

Drop candidates that would exceed configured complexity or node cap.

---

### 6.5 Tier-1 cheap features

Tier-1 features should be cheap enough to compute over all raw candidates.

#### Basic result features

```python
features["degree_result"]
features["degree_target"]
features["degree_gap"] = degree(target) - degree(result)
features["support_size_result"]
features["support_size_target"]
features["equals_target"]
```

#### Support overlap

```python
supp_g = support(result)
supp_f = support(target)

features["support_overlap_count"] = len(supp_g & supp_f)
features["support_overlap_frac"] = len(supp_g & supp_f) / max(1, len(supp_g))
features["target_coverage_frac"] = len(supp_g & supp_f) / max(1, len(supp_f))
features["outside_support_count"] = len(supp_g - supp_f)
```

Do not hard-drop all candidates with zero direct support overlap. Useful intermediates such as `x+y` may not directly overlap a degree-2 target but can still be essential.

#### Residual-exists check

For candidate result `g`:

```python
r = target - g
```

Compute:

```python
features["residual_exists"] = float(key(r) in state.node_keys)
features["residual_support_size"]
features["residual_target_overlap"]
```

If `residual_exists = 1`, then `g` can complete the target by one addition.

#### Target factor seed features

If target factors are available:

```python
features["equals_target_factor"]
features["support_overlap_with_factor"]
features["divides_known_factor_hint"]
```

Avoid expensive exact division here unless it is trivial.

---

### 6.6 Tier-2 expensive features

Run these only on top-M candidates from tier 1.

#### Exact quotient / divisibility

Check whether candidate result `g` exactly divides target:

```python
q = exact_divides(target, g)
```

If exact:

```python
features["divides_target"] = 1.0
features["quotient_exists"] = float(key(q) in state.node_keys)
features["quotient_degree"]
features["quotient_support_size"]
```

If `quotient_exists = 1`, then `g` can complete the target by multiplication.

Implementation requirements:

```text
- cache by (target_key, g_key)
- prefilter by degree before exact division
- optionally prefilter with cheap support/variable checks
- check exact remainder under F_p
```

If using SymPy:

```text
perform multivariate division under a chosen monomial order,
then reduce/check remainder modulo p.
```

The routine must return `None` unless the quotient is exact under the chosen polynomial semantics.

#### One-step completion

For candidate result `g`, check whether it can combine with any existing node `u`, or itself, to produce target:

```python
for u in state.nodes + [g]:
    if g + u == target:
        features["one_step_completion_add"] = 1.0

    if g * u == target:
        features["one_step_completion_mul"] = 1.0
```

This catches:

```text
g = x+y
g*g = x^2 + 2xy + y^2
```

#### Two-step lookahead

Two-step lookahead is expensive. Do not run it on all candidates.

If implemented, run only for a tiny subset of tier-2 candidates.

Example:

```text
target = (x+y)^3
candidate g = x+y

h = g*g
h*g = target
```

Set:

```python
features["two_step_completion_hint"] = 1.0
```

The learned ranker should eventually learn to predict this kind of delayed usefulness from cheaper features, so avoid making two-step lookahead mandatory at search time.

#### Residual or quotient factorization

Optional. If used, only run on top-M or top-N candidates, never all raw actions.

---

### 6.7 Heuristic scoring

Example score:

```python
score = 0.0

score += 1000.0 * features["equals_target"]

score += 250.0 * features.get("one_step_completion_add", 0.0)
score += 250.0 * features.get("one_step_completion_mul", 0.0)

score += 80.0 * features.get("quotient_exists", 0.0)
score += 40.0 * features.get("divides_target", 0.0)
score += 80.0 * features.get("residual_exists", 0.0)

score += 30.0 * features.get("equals_target_factor", 0.0)

score += 5.0 * features["support_overlap_count"]
score += 5.0 * features["target_coverage_frac"]

score -= 2.0 * features["outside_support_count"]
score -= 0.5 * max(0, features["degree_gap"])
```

Keep weights in config.

The heuristic should be strong enough to produce useful search traces before the ranker is trained, but the central experiment is whether the learned ranker improves over this heuristic.

---

## 7. Planner / decoder: guided beam search

Beam search is the first planner to implement.

### 7.1 Purpose

Beam search keeps the top `B` partial circuits at each depth.

It avoids expanding the full action tree:

```text
full branching: all possible (op, i, j)
beam branching: top-K target-conditioned candidates from each beam state
```

### 7.2 Inputs

```python
beam_search(
    instance: ProblemInstance,
    candidate_generator,
    ranker=None,
    beam_width=16,
    candidate_k=64,
    tier2_m=128,
    max_depth=None,
    lambda_model=0.0,
    exploration_eps=0.05,
)
```

If `max_depth` is `None`, use:

```python
max_depth = instance.op_budget
```

### 7.3 Candidate score

```python
candidate.total_score = (
    candidate.heuristic_score
    + lambda_model * candidate.model_score
)
```

If no ranker exists:

```python
lambda_model = 0.0
```

### 7.4 State score

Start simple:

```python
state_score = candidate.total_score - alpha * state.num_ops()
```

A cheap improvement is to include a potential-like state progress signal:

```python
phi(state, target) = max over current nodes of similarity(node, target)
```

where similarity may use support overlap, coefficient match, or residual closeness.

Then:

```python
state_score = candidate.total_score + beta * phi(next_state, target) - alpha * num_ops
```

Later, replace or augment this with a learned value head.

### 7.5 Expansion loop

```python
beam = [initial_state(instance)]
history = SearchHistory(instance=instance)

for depth in range(instance.op_budget):

    all_expansions = []

    for state in beam:

        candidates = generate_candidates(
            instance=instance,
            state=state,
            K=candidate_k,
            tier2_m=tier2_m,
        )

        if ranker is not None and lambda_model > 0:
            for cand in candidates:
                cand.model_score = ranker.score(instance, state, cand)
                cand.total_score = cand.heuristic_score + lambda_model * cand.model_score
        else:
            for cand in candidates:
                cand.total_score = cand.heuristic_score

        selected = select_for_expansion(candidates, exploration_eps)

        for cand in selected:
            next_state = state.apply(cand.action)

            history.record(
                state=state,
                candidates=candidates,
                candidate=cand,
                next_state=next_state,
                depth=depth,
            )

            if next_state.contains(instance.target):
                history.finished.append(next_state)

            all_expansions.append(
                (score_state(next_state, instance, cand), next_state)
            )

    beam = keep_top_diverse(all_expansions, beam_width)

return history
```

### 7.6 Diversity and deduplication

When selecting the next beam:

```text
- deduplicate identical node sets
- deduplicate identical final result node
- prefer smaller circuits when states are equivalent
- optionally enforce diversity by last action source tag
```

This prevents the beam from collapsing into near-identical states.

---

## 8. Search history

Search history must store enough information to train the ranker.

```python
@dataclass
class ExpandedStateRecord:
    instance: ProblemInstance
    state: CircuitState
    candidates: list[Candidate]
    candidate: Candidate
    next_state: CircuitState
    depth: int
```

```python
@dataclass
class SearchHistory:
    instance: ProblemInstance
    records: list[ExpandedStateRecord]
    finished: list[CircuitState]
```

After search, annotate records with approximate downstream outcomes:

```python
record.branch_success
record.best_final_size_after_action
record.expansions_to_success_after_action
record.downstream_return
```

These can be computed by tracing which expanded branches eventually led to verified circuits.

---

## 9. Preference extraction

The ranker is trained from search-generated preferences.

### 9.1 Select best verified circuit

For each instance:

```python
verified = [state for state in history.finished if state.contains(instance.target)]
best = min(verified, key=lambda s: s.num_ops())
```

If no verified circuit is found, do not create strong positive labels. You may create weak negative labels for structurally useless actions.

### 9.2 Recover the best trace

The state should contain parent/action history, so recover:

```text
(C_0, a_0)
(C_1, a_1)
...
(C_T, a_T)
```

for the best verified circuit.

### 9.3 Compare good actions against alternatives

At each state on the best trace:

```text
good action = action used by the best trace
alternatives = other candidates generated at that state
```

Create preferences only when the return gap is meaningful.

```python
if R(good) > R(other) + delta:
    add_preference(good > other)
```

### 9.4 Downstream return

Suggested return:

```python
if branch_success:
    R = 100.0 - final_circuit_size - 0.001 * expansions_used
else:
    R = -20.0
```

This is not a true optimal value. It is a search-generated training signal.

### 9.5 Preference gap schedule

Early search traces are noisy. Start with a large preference threshold:

```text
early rounds: delta = 5.0
middle rounds: delta = 2.0
later rounds: delta = 1.0
```

This means early training only learns from clear wins:

```text
action A solved, action B failed
action A produced much smaller circuit than B
action B was structurally useless
```

As search improves, use smaller gaps.

### 9.6 Preference weights

Suggested weights:

```text
good solved vs failed branch: 1.0
good solved vs much larger verified circuit: 1.0
good solved vs slightly larger verified circuit: 0.25
good vs duplicate/useless action: 2.0
good vs noisy failed branch: 0.5
```

Do not over-weight early “1 operation smaller” comparisons. Those may reflect search noise rather than true action quality.

---

## 10. CandidatePolicy / Ranker

The candidate policy predicts:

```python
score = ranker(instance, state, candidate)
```

The first implementation should be an MLP over engineered features.

Do not start with a complex GNN unless the feature MLP fails.

### 10.1 Input features

#### ProblemInstance / target features

```text
number of variables
target degree
target support size
target coefficient statistics
target family embedding or one-hot
target factorization summary if available
operation budget
```

For small settings, a dense coefficient vector is acceptable.

For larger settings, prefer:

```text
sparse support features
hashed monomial features
degree histogram
coefficient histogram
```

Avoid committing to a dense vector layout that explodes with `n_vars` and `degree`.

#### State features

```text
number of nodes
number of constructed ops
remaining operation budget
best current support overlap with target
best current residual closeness
whether target already exists
```

#### Operand node features

For nodes `i` and `j`:

```text
degree
support size
target support overlap
outside-target support
whether node divides target, if cached
whether target - node exists in state
graph depth / reuse count, if available
```

#### Candidate result features

These are the most important:

```text
operation type
degree(result)
support size(result)
support overlap with target
target coverage fraction
outside-target support count
equals target
equals target factor
residual exists
divides target
quotient exists
one-step completion add
one-step completion mul
two-step hint, if computed
heuristic score
```

### 10.2 First model

First model:

```text
feature vector
    ↓
MLP
    ↓
scalar score
```

Example:

```python
MLP(input_dim, hidden_dim=256, num_layers=3, dropout=0.1)
```

### 10.3 Later value head

After the ranker improves beam search, add a value head:

```python
V(instance, state) -> predicted downstream search value
```

The value target can be:

```text
best verified circuit size reachable from this state under search budget
success/failure under budget
estimated return
```

This value head is mainly for MCTS or Gumbel-style search later.

---

## 11. Ranker training

### 11.1 Pairwise ranking loss

For each preference:

```python
s_better = ranker(instance, state, better)
s_worse = ranker(instance, state, worse)

loss = weight * max(0, margin - s_better + s_worse)
```

Recommended:

```text
margin = 1.0
```

### 11.2 Replay buffer

Use a replay buffer of preference examples:

```python
PreferenceReplayBuffer(max_size=N)
```

Keep old data, but prioritize recent data and best-circuit updates.

### 11.3 Regeneration cadence

Every few rounds, rerun search on previously solved targets with the current ranker.

If a smaller circuit is found, supersede old preferences for that target.

This prevents the model from being anchored to early mediocre traces.

---

## 12. Bootstrap loop

### 12.1 Validation-gated model schedule

Do not increase the model weight purely by round number.

Use a held-out validation set.

Compare:

```text
heuristic-only beam search
vs
heuristic + learned ranker beam search
```

Only increase `lambda_model` if the learned ranker does not hurt validation performance.

Example schedule:

```text
start: lambda_model = 0.0
candidate next: 0.25
then: 0.5
then: 1.0
```

Promotion rule:

```text
promote λ only if heuristic+ranker >= heuristic-only
on solve rate and/or expansions-to-solution
over the validation set.
```

If the ranker hurts, keep λ fixed or reduce it.

### 12.2 Bootstrap pseudocode

```python
ranker = CandidateRanker()
buffer = PreferenceReplayBuffer()

lambda_model = 0.0

for round_idx in range(num_rounds):

    instances = curriculum.sample_training_instances()

    for instance in instances:
        history = beam_search(
            instance=instance,
            ranker=ranker,
            lambda_model=lambda_model,
            beam_width=config.beam_width,
            candidate_k=config.candidate_k,
            tier2_m=config.tier2_m,
            max_depth=instance.op_budget,
            exploration_eps=config.exploration_eps,
        )

        prefs = extract_preferences(
            history=history,
            instance=instance,
            delta=current_delta(round_idx),
        )
        buffer.add(prefs)

    train_ranker(ranker, buffer)

    if should_promote_lambda(ranker, validation_instances):
        lambda_model = next_lambda(lambda_model)

    curriculum.update_from_recent_results()
```

---

## 13. Problem generation and curriculum

Use structured target families. Random dense polynomials may not have compact circuits.

### 13.1 Families

#### Common-factor targets

```text
ab + ac
ab + ac + ad
x^2y + xy^2
a(b+c+d)
```

#### Powers of sums

```text
(x+y)^2
(x+y)^3
(x+y+z)^2
(x+y+z)^3
```

#### Product-of-sums

```text
(a+b)(c+d)
(x+y)(x+z)
(a+b+c)(d+e)
```

#### Horner-style targets

```text
x^3 + 2x^2 + 3x + 4
x(x(x+2)+3)+4
```

#### Symmetric polynomials

```text
xy + xz + yz
xyz + xyw + xzw + yzw
```

#### Random circuit targets

Generate random circuits of known operation count, then use their output polynomial as the target.

This gives targets with known generative complexity, not necessarily optimal complexity.

### 13.2 Adaptive curriculum

Track solve rate by:

```text
complexity
degree
number of variables
target family
```

Start with low complexity and increase when solve rate exceeds a threshold.

Example:

```text
if solve_rate(complexity=k) > 80% for two consecutive rounds:
    allow complexity k+1
```

Also oversample failing families.

### 13.3 Tiny-board diagnostics

For very small instances only, optionally use exhaustive BFS boards to:

```text
grade target difficulty
test candidate generator recall on known optimal traces
warm-start the ranker on tiny examples
```

Do not rely on BFS boards for the main training pipeline at complexity 10.

---

## 14. Exact verification and circuit extraction

Every returned circuit must be exactly verified.

```python
def verify_state(state, target):
    return any(node == target for node in state.nodes)
```

For final output, reconstruct the action trace and re-execute it from base nodes:

```python
def verify_trace(trace, instance):
    state = initial_state(instance)
    for action in trace:
        state = state.apply(action)
    return state.contains(instance.target)
```

If verification fails, discard the circuit and raise a diagnostic error.

---

## 15. Main evaluation during development

The central development comparison is:

```text
heuristic-only search
vs
heuristic + learned ranker search
```

This should be measured continuously, not left as a later baseline.

### 15.1 Core grid

Evaluate:

```text
random top-K ordering
heuristic-only beam search
heuristic + learned-ranker beam search
heuristic + learned-ranker + value/MCTS, later
```

Across:

```text
complexity
target family
number of variables
search budget
```

### 15.2 Metrics

```text
solve rate at fixed budget
expansions to first verified solution
best verified circuit size
search time
expansions saved at equal solve rate
final circuit size at equal budget
```

### 15.3 Honest framing

If the learned ranker is only marginally better than the symbolic heuristic, that is still useful information.

The goal is to identify:

```text
where learning helps symbolic search
where heuristic search is already enough
where the candidate generator is the bottleneck
```

---

## 16. Logging and diagnostics

### 16.1 Search logs

For each target:

```text
target id
family
degree
number of variables
configured complexity
beam width
candidate K
tier2 M
max depth
expanded states
generated candidates
filtered candidates
success/failure
best verified circuit size
depth of first solution
wall time
```

### 16.2 Candidate logs

For expanded states:

```text
raw candidate count
after hard filter count
after dedup count
tier2 pool size
returned top-K size
number of exact target candidates
number of one-step completions
number of quotient candidates
number of residual-exists candidates
number of target-factor candidates
number of duplicates removed
```

### 16.3 Training logs

```text
number of preference examples
ranking loss
pairwise accuracy: score(better) > score(worse)
average score margin
validation heuristic-only solve rate
validation heuristic+ranker solve rate
lambda_model value
```

---

## 17. Milestones

### Milestone 1: exact backend safety

Implement or verify:

```text
finite-field arithmetic
canonical polynomial keys
overflow assertions
no silent truncation
domain/layout consistency checks
exact verification
```

### Milestone 2: two-tier candidate generator

Implement:

```text
raw add/mul enumeration
hard filters
deduplication
tier-1 cheap features
target factorization candidate seeds
tier-2 exact quotient checks
one-step completion checks
heuristic top-K selection
```

Manual expected behavior:

```text
x^2 + 2xy + y^2:
    x+y appears early
    after x+y exists, (x+y)*(x+y) appears at top

ab + ac:
    b+c appears early
    after b+c exists, a*(b+c) appears at top

(x+y+z)^2:
    x+y or x+y+z path appears
```

### Milestone 3: heuristic beam search

Run beam search with:

```text
lambda_model = 0
```

Confirm that heuristic-only search solves easy structured targets.

### Milestone 4: search history and preference extraction

Implement:

```text
SearchHistory
best verified circuit selection
trace recovery
downstream return annotation
pairwise preference extraction
```

### Milestone 5: MLP ranker

Train the first candidate ranker with pairwise ranking loss.

### Milestone 6: bootstrapped training

Run:

```text
heuristic search
→ collect preferences
→ train ranker
→ validation-gated lambda update
→ guided search
→ regenerate better preferences
```

### Milestone 7: value head and MCTS/Gumbel search

After beam + ranker is stable, add:

```text
state value head
MCTS or Gumbel/Sequential-Halving planner
```

This is planned, but not required before the beam-search pipeline works.

---

## 18. Reuse guidance

Only reuse components that pass tests under the new exact-search assumptions.

### 18.1 Reuse if compatible

```text
exact polynomial arithmetic
canonical polynomial equality
SymPy ↔ internal polynomial conversions, after making them fail loudly
target generation utilities
basic circuit transition code
action canonicalization i <= j
existing curriculum scaffold, if clean
```

### 18.2 Do not reuse as the main method

```text
old reward machinery
stored cross-episode factor memory
flat-action RL training loop
old factor-library cache/state
old valid-action integer codec if it forces flat action selection
```

These can remain as separate baselines later, but they should not be dependencies of the new pipeline.

### 18.3 Add tests before reusing neural code

If reusing any existing GNN or batching code, add tests for:

```text
multi-example batches
padding behavior
actual node counts
pooling over real nodes only
consistent target feature layout
```

Do not assume batch-1 tests catch batching/padding bugs.

---

## 19. Summary

The desired system is:

```text
ProblemGenerator
    → ProblemInstance
    → exact SearchState transitions
    → fresh target-conditioned CandidateGenerator
    → symbolic heuristic
    → learned CandidatePolicy / Ranker
    → beam search first
    → search-generated ranking data
    → validation-gated bootstrapping
    → exact verification
```

The most important implementation principles are:

```text
1. The candidate generator shrinks the action space.
2. The learned ranker improves candidate ordering.
3. The planner handles multi-step construction.
4. Exact arithmetic handles correctness.
5. Nothing should silently truncate, overflow, or lie.
```

The first successful version should prove:

```text
heuristic + learned ranker
beats
heuristic-only search
```

under the same candidate budget, beam width, search depth, and target distribution.
