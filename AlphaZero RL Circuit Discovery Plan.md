# Decomposition-Search RL for Arithmetic Circuit Discovery over \(\mathbf F_p\)

A codex-oriented implementation specification and milestone checklist.

---

## 0. Purpose

This document specifies a concrete implementation plan for a new **split-based, AlphaZero-style / Expert-Iteration-style** arithmetic-circuit discovery system over finite fields \(\mathbf F_p\).

The core design change is to replace the original **gate-by-gate circuit growth** environment with a **recursive decomposition game**:

- input state: a target polynomial \(f\), or a frontier of unresolved target polynomials,
- action: choose an **addition split** \(f=g+h\),
- environment step: factor \(g\) and \(h\) over \(\mathbf F_p\),
- continuation: recursively solve the resulting factors,
- objective: minimize the final arithmetic-circuit cost after memoized reuse and symbolic factorization.

This is intended to reduce sparse rewards, shorten effective horizons, and align more naturally with self-improving search and decomposition-based reasoning.

This plan is designed to be implemented inside or alongside the existing `PolyArithmeticCircuitsRL` codebase.

## 0.1 Implementation Status Snapshot (April 19, 2026)

- [x] Symbolic backbone implemented: sparse polynomials, canonicalization, hashing, rebuild costs, and CAS-backed factorization.
- [x] MVP environment and AND/OR search implemented with memoization, frontier handling, and direct-solve fallback.
- [x] Supervised warm start implemented with synthetic decomposition traces and a trainable Torch policy/value model.
- [x] Search-distillation pipeline implemented with iterative training scripts.
- [x] Prioritized replay and elite self-imitation implemented and integrated into the experiment loop.
- [x] Sage CAS environment integrated via persistent helper process.
- [~] Stronger proposal families implemented beyond MVP: common-factor and elementary-symmetric family templates are in; broader family coverage is still open.
- [~] Broader curricula and evaluation coverage are partially implemented.
- [ ] Final DAG scoring / CSE pass not implemented yet.
- [ ] Optional PPO/SAC fine-tuning not implemented yet.

---

## 1. Guiding references

Primary project references:

1. **CircuitBuilder: From Polynomials to Circuits via Reinforcement Learning**  
   arXiv: 2603.17075  
   https://arxiv.org/abs/2603.17075

2. **PolyArithmeticCircuitsRL repository**  
   https://github.com/uw-math-ai/PolyArithmeticCircuitsRL

Training-strategy references that motivate the search-distillation / self-improvement recipe:

3. **Expert Iteration (ExIt)** — tree search produces improved policies, neural net imitates them  
   https://proceedings.neurips.cc/paper_files/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf

4. **Prioritized Level Replay (PLR)** — prioritize revisiting training tasks with high learning potential  
   https://proceedings.mlr.press/v139/jiang21b/jiang21b.pdf

5. **Self-imitation learning** — replay and imitate successful past trajectories in sparse-reward settings  
   Representative reference: https://openreview.net/forum?id=HyxzRsR9Y7

---

## 2. High-level system design

### 2.1 Core idea

Instead of letting the agent construct an arithmetic circuit from the variable nodes upward, train the agent to repeatedly discover useful **addition split points**

\[
f = g + h.
\]

The environment then:

1. factors \(g\) over \(\mathbf F_p\),
2. factors \(h\) over \(\mathbf F_p\),
3. symbolically records multiplication structure for those factorizations,
4. returns the nontrivial factor subproblems as the next unresolved items.

If a chosen split causes one or both sides to factor substantially, the agent has effectively uncovered a compact subcircuit.

### 2.2 Why this should help

This formulation has several advantages over gate-by-gate DAG growth:

- every move is semantically meaningful,
- rewards can be shaped by **immediate estimated cost reduction**,
- planning depth is reduced because symbolic factorization collapses many multiplication decisions,
- recursive families such as Horner schemes and elementary symmetric polynomials naturally become short decomposition traces,
- the setup fits **search-guided self-improvement** naturally.

### 2.3 Correct mental model

This is not just a standard MCTS tree over a single branching process. It is best viewed as an **AND/OR decomposition search**:

- **OR node:** choose a split \(f=g+h\),
- **AND node:** solve all resulting unresolved factors from \(g\) and \(h\).

---

## 3. Mathematical problem formulation

Work over a fixed prime field \(\mathbf F_p\).

Given a polynomial
\[
f \in \mathbf F_p[x_1,\dots,x_n],
\]
we seek an arithmetic circuit computing \(f\) with minimal or near-minimal cost under a chosen size model.

### 3.1 Proposed recursive cost objective

Let \(C(f)\) denote the best cost found for \(f\).

We define
\[
C(f) = \min\Big(B(f),\; \min_{a=(g,h):\, f=g+h} Q(f,a)\Big),
\]
where:

- \(B(f)\) is a fallback direct-construction baseline cost,
- \(Q(f,a)\) is the cost induced by split \(a=(g,h)\).

If
\[
g = u_g \prod_i q_i^{e_i},
\qquad
h = u_h \prod_j r_j^{d_j},
\]
then
\[
Q(f,a)
=
1 + R(g) + R(h) + \sum_{u\in \mathcal U(g,h)} C(u),
\]
where:

- the `1` is the top addition gate,
- \(R(g)\) is the multiplication/exponentiation cost to rebuild \(g\) from its factors,
- \(R(h)\) is the analogous rebuild cost for \(h\),
- \(\mathcal U(g,h)\) is the set or multiset of unresolved child subproblems after factorization and memo-reuse.

This objective allows the planner to compare “split further” vs “solve directly now”.

---

## 4. Environment design

### 4.1 Full environment state

Use a frontier-based state:

\[
S = (\mathcal U, \mathcal C, c_{\text{acc}}),
\]
where:

- \(\mathcal U\): multiset of unresolved polynomials,
- \(\mathcal C\): cache / transposition table / memoized solved subproblems,
- \(c_{\text{acc}}\): accumulated cost so far.

### 4.2 Training simplification

For the neural net, the fundamental decision unit should be a **single active polynomial** \(f\). The model learns:

- policy over candidate splits for \(f\),
- value estimate for the best achievable normalized saving on \(f\).

Frontier selection can be:

- teacher-forced during early training,
- heuristic later,
- learned only after split selection is stable.

### 4.3 Episode start

At reset:

- \(\mathcal U = \{f_{\text{target}}\}\),
- \(\mathcal C = \varnothing\),
- \(c_{\text{acc}} = 0\).

### 4.4 Episode termination

Terminate when the unresolved frontier is empty.

A polynomial should count as **solved** if it is one of:

- a constant,
- a variable,
- a monomial,
- an item handled by the base solver,
- a cache hit with known best cost.

### 4.5 Environment API

Proposed API:

```python
class DecompEnv:
    def reset(self, target_poly) -> State:
        ...

    def get_active_items(self, state) -> list[PolyHandle]:
        ...

    def get_candidate_splits(self, state, poly_handle, k: int) -> list[SplitAction]:
        ...

    def step(self, state, poly_handle, action: SplitAction):
        # returns next_state, reward, done, info
        ...
```

`info` should include:

- chosen \(g,h\),
- factorization of \(g\),
- factorization of \(h\),
- rebuild costs \(R(g), R(h)\),
- child cache hits,
- baseline costs before and after,
- immediate saving estimate,
- debugging metadata.

---

## 5. Polynomial representation and canonicalization

### 5.1 Sparse polynomial format

Represent polynomials sparsely as ordered tuples of monomials:

\[
((c_1, \alpha_1), \dots, (c_m, \alpha_m)),
\]
where:

- \(c_i \in \mathbf F_p\),
- \(\alpha_i \in \mathbf Z_{\ge 0}^n\),
- terms are sorted canonically.

### 5.2 Canonicalization rules

Every polynomial object must satisfy:

- coefficients reduced mod \(p\),
- zero coefficients removed,
- monomials sorted by a fixed order,
- duplicate monomials merged,
- scalar units normalized in factor outputs,
- deterministic string / hash serialization.

### 5.3 Required polynomial metadata

For each polynomial, precompute and cache:

- support size,
- total degree,
- per-variable max degree,
- homogeneous flag,
- monomial gcd exponent vector,
- coefficient histogram mod \(p\),
- sparsity density statistics,
- current baseline cost estimate,
- factorization status if already known.

### 5.4 Critical requirement

**Canonical hashing is mandatory.**

Without canonical hashing, memoization and subproblem reuse will fail, and the system will behave like formula search instead of DAG-aware search.

---

## 6. Action space: addition-split proposals

### 6.1 Action definition

An action is a canonical split
\[
a = (g,h), \qquad f = g + h,
\]
with a fixed ordering rule so \((g,h)\) and \((h,g)\) are not duplicated.

### 6.2 Do not enumerate all splits

The full split space is enormous. The policy should only rank a bounded **candidate set**
\[
\{a_1,\dots,a_K\}.
\]
So implement a candidate generator:

```python
def propose_splits(f: SparsePoly, k: int, config) -> list[SplitAction]:
    ...
```

### 6.3 Candidate proposal families

#### A. Support-partition splits
Assign each term of \(f\) wholly to \(g\) or \(h\).

This is the simplest first version and should be the MVP action family.

#### B. Common-factor template splits
Detect clusters of terms sharing a common monomial or low-degree polynomial factor and propose:

- one side = factorable cluster,
- other side = complement.

#### C. Horner-style splits
For each variable \(x_i\), write
\[
f = r_i + x_i q_i,
\]
where:

- \(r_i\) collects terms with \(x_i\)-exponent zero,
- \(q_i\) is the quotient after factoring one \(x_i\) from the remaining terms.

This should be included explicitly as a candidate family.

#### D. Recognized-family splits
Inject known recurrence-based splits when the polynomial belongs to a structured family, e.g. elementary symmetric polynomials.

#### E. Random-but-biased support masks
Include a small number of stochastic proposals to maintain diversity.

### 6.4 Candidate set size

Start with something like:

- `K = 16` or `32` in early versions,
- later expand to `64` or `128` if needed.

Candidate generation quality is a major bottleneck. It should be treated as a first-class component.

---

## 7. Factorization and symbolic multiplication

### 7.1 Factorization requirement

After each split, the environment must factor \(g\) and \(h\) over \(\mathbf F_p\). This is central to the design.

### 7.2 Implementation advice

Use a robust symbolic algebra backend for finite-field polynomial factorization. Acceptable options include:

- SymPy for simpler prototypes,
- Sage for stronger finite-field support,
- a custom wrapper if the codebase already has a preferred CAS route.

### 7.3 Factorization output format

Standardize output as:

```python
[(factor_poly_1, exponent_1), ..., (factor_poly_t, exponent_t)]
```
plus a separate scalar unit.

### 7.4 Rebuild cost

For a factorization
\[
g = u \prod_i q_i^{e_i},
\]
compute the multiplication/exponentiation rebuild cost \(R(g)\) by a consistent cost model.

Recommended initial rule:

- repeated squaring or short addition-chain estimate for powers,
- multiplication count to combine the powered factors,
- scalar units cost-free or near-free depending on convention.

### 7.5 Cache factorization aggressively

Factorization will be expensive. Cache all factorization results keyed by canonical polynomial.

---

## 8. Base solver and fallback cost model

### 8.1 Why a base solver is necessary

The system must have the option to stop splitting and solve a polynomial directly. Otherwise it will learn to split uselessly.

### 8.2 Base cases

At minimum:

- constant: cost 0,
- variable: cost 0,
- monomial: exact monomial-construction cost,
- tiny-support polynomial: optional exact small solver,
- recognized trivial family instance: direct known cost.

### 8.3 Baseline direct-construction cost \(B(f)\)

Implement a fallback cost estimate, e.g.:

- sparse summation plus monomial construction,
- Horner-like heuristic,
- dense evaluation-inspired heuristic if needed,
- exact search for tiny cases.

The search objective always compares against \(B(f)\).

---

## 9. Reward shaping and value targets

### 9.1 Immediate shaped reward

Let
\[
\widehat Q(f;g,h)
=
1 + R(g) + R(h) + \sum_{u \in \text{children}(g,h)} B(u).
\]
Define immediate reward
\[
r = B(f) - \widehat Q(f;g,h).
\]
Interpretation:

- positive reward: this split seems better than solving \(f\) directly,
- negative reward: likely a bad split.

This is the primary anti-sparsity mechanism.

### 9.2 Search-time value target

When search or memoized solutions are available, replace the child baselines by improved estimates.

### 9.3 Value head target

Predict normalized saving
\[
V(f) \approx \frac{B(f)-C^*(f)}{B(f)+\varepsilon}.
\]
This stabilizes targets across scales and curriculum stages.

---

## 10. Search: AND/OR PUCT with memoization

### 10.1 Search objective

This should not use vanilla single-path AlphaZero backup directly. The correct search is **PUCT selection on OR nodes with AND-style cost aggregation**.

### 10.2 OR node logic

At polynomial \(f\):

1. generate candidate splits,
2. score them with the policy prior,
3. select via PUCT,
4. expand chosen split,
5. factor children,
6. recursively solve unresolved children.

### 10.3 AND backup

For split \(a=(g,h)\), backup should aggregate child costs:

\[
\widehat Q(f,a)
=
1 + R(g) + R(h) + \sum_{u\in\text{children}(a)} \widehat C(u).
\]

Selection is therefore minimizing cost, or maximizing savings.

### 10.4 Transposition table

Maintain a transposition table keyed by canonical polynomial, storing at least:

- visit count,
- best cost found,
- value estimate,
- best split found,
- candidate priors if cached,
- factorization cache.

### 10.5 Final CSE pass

After building a decomposition tree, run a final common-subexpression-elimination pass to score the resulting DAG fairly.

This is essential; otherwise the planner may undervalue reusable structure.

---

## 11. Neural architecture

### 11.1 State encoder

The state is not naturally a circuit graph anymore. Use a term-level encoder for sparse polynomials.

Each term token should encode:

- coefficient embedding mod \(p\),
- exponent vector,
- total monomial degree,
- optional simple handcrafted statistics.

Recommended first architecture:

- Transformer encoder over term tokens, or
- Set Transformer / DeepSets if a simpler start is preferred.

### 11.2 Candidate-scoring policy head

The policy does not output over a global action space. It scores a finite candidate set.

For each candidate split \((g,h)\), encode:

- the target polynomial \(f\),
- side polynomials \(g\) and \(h\),
- candidate metadata such as support sizes, degree balance, common factors, immediate estimated saving, factorization summary if available.

Then score each candidate with an MLP or cross-attention scorer.

### 11.3 Value head

A scalar head on the encoded target polynomial predicting normalized saving.

### 11.4 Optional auxiliary heads

Highly recommended:

- `factorable_branch_head`: predicts whether a split yields at least one nontrivial factorable branch,
- `immediate_saving_bucket_head`: predicts rough short-term benefit class,
- optional family classifier during pretraining only.

---

## 12. Pretraining strategy

Pretraining should be done on **decomposition traces**, not just final circuits.

### 12.1 Why decomposition traces matter

The environment decision is “how should I split this polynomial?”, so the supervised targets should be traces of good split decisions and their induced recursive subproblems.

### 12.2 Pretraining source A: planted factorizable splits

Generate examples by sampling factorable \(g,h\), then setting
\[
f=g+h.
\]
Keep examples where the planted split is actually useful under the current cost model.

Store:

- target \(f\),
- candidate set,
- planted split label,
- factorization tree of \(g\) and \(h\),
- resulting total cost target.

### 12.3 Pretraining source B: Horner traces

Horner decomposition maps perfectly to the split environment.

For a univariate polynomial:
\[
f(x)=a_0 + x q_1(x),
\]
use the supervised split:

- \(g = a_0\),
- \(h = x q_1(x)\).

Then recurse on \(q_1\).

For multivariate Horner, choose a variable order and repeatedly apply
\[
f = r + x_i q.
\]

This produces excellent early supervision because one branch is immediately factorable at every step.

### 12.4 Pretraining source C: elementary symmetric polynomial traces

For
\[
e_k^{(n)} = e_k(x_1,\dots,x_n),
\]
use the recurrence
\[
e_k^{(n)} = e_k^{(n-1)} + x_n e_{k-1}^{(n-1)}.
\]

This is an ideal split label:

- \(g = e_k^{(n-1)}\),
- \(h = x_n e_{k-1}^{(n-1)}\).

This family is especially important because it forces the system to learn memoized reuse.

### 12.5 Pretraining source D: exact-small-solver traces

For tiny problem sizes, solve the decomposition problem exactly by exhaustive search or dynamic programming.

This gives ground-truth optimal split labels and exact cost targets.

### 12.6 Mixing schedule for early pretraining

Recommended initial data mixture:

- 40% planted factorizable splits,
- 25% Horner traces,
- 20% elementary symmetric traces,
- 15% exact-small-solver traces.

This can later shift toward more exact and search-distilled data.

### 12.7 Mod-\(p\) validation requirement

Every synthetic generator must validate that over the chosen field:

- the polynomial is nonzero,
- the split is nontrivial,
- the intended recurrence still holds after reduction mod \(p\),
- the cost advantage remains real.

Characteristic effects can otherwise silently corrupt the dataset.

---

## 13. Training strategy: self-improving search, not pure PPO-from-scratch

### 13.1 Recommended training paradigm

Use an **Expert-Iteration-style** loop:

1. current model guides search,
2. search finds improved split policies,
3. model imitates search,
4. repeat.

This is preferable to raw policy-gradient exploration from scratch.

### 13.2 Four training stages

#### Stage A: supervised warm start
Train on synthetic and exact decomposition traces.

#### Stage B: search distillation
For each target:

- run AND/OR PUCT search with the current model,
- collect root search policy distribution,
- collect backed-up cost / value targets,
- train the model on these improved targets.

#### Stage C: prioritized self-improvement
Maintain a replay buffer of training targets and revisit those with highest learning potential.

#### Stage D: optional RL fine-tuning
Only after the above pipeline is stable, optionally add PPO or SAC fine-tuning using the decomposition environment.

### 13.3 Strong recommendation

Treat PPO/SAC as optional later refinements. The backbone should be:

- supervised traces,
- search distillation,
- elite self-imitation,
- prioritized task replay.

---

## 14. Curriculum design

Curriculum is necessary, but it must be structured around the correct axes.

### 14.1 Complexity axis

Increase gradually by:

- number of variables,
- support size,
- total degree,
- baseline construction cost.

### 14.2 Splitability axis

Start with targets where useful splits are obvious or common:

- Horner-like,
- elementary-symmetric recurrence,
- planted factorable branches,
- low-support structured targets.

Then move toward:

- less obvious groupings,
- fewer high-gain splits among many distractors,
- greater reliance on search.

### 14.3 Horizon axis

Gradually increase decomposition depth:

- shallow traces,
- medium traces,
- deep recursive decomposition trees.

### 14.4 Reuse axis

Schedule in tasks where memoization is essential:

- elementary symmetric families,
- repeated-structure synthetic families,
- batched related targets.

### 14.5 Prime axis

Do not randomize aggressively over many primes at the beginning. Start with one or two small primes, then broaden.

---

## 15. Prioritized replay over target polynomials

Uniform sampling is not ideal.

Maintain a priority score for each training target:

\[
P(f) = \alpha\,\mathrm{policy\_gap}(f)
+ \beta\,\mathrm{value\_error}(f)
+ \gamma\,\mathrm{search\_gain}(f)
+ \delta\,\mathrm{novelty}(f).
\]

Where:

- `policy_gap`: discrepancy between network policy and search-improved policy,
- `value_error`: value prediction error,
- `search_gain`: amount by which search improved over raw policy,
- `novelty`: structural rarity / diversity contribution.

Sample targets proportional to this priority, with a floor of uniform random sampling.

---

## 16. Self-imitation and elite-buffer training

### 16.1 Why self-imitation is natural here

Good split traces will be rare and valuable. Once the system finds a high-quality decomposition trace, that trace should be replayed and imitated.

### 16.2 Elite buffer contents

Maintain an elite buffer storing:

- best complete decomposition traces found so far,
- best partial subproblem traces,
- associated cost reductions,
- family tags / structural metadata.

### 16.3 How to use it

Periodically interleave supervised updates from the elite buffer into the main training stream.

This is a natural self-improvement mechanism and aligns well with self-improving-search workshop themes.

---

## 17. Frontier selection policy

### 17.1 Early training

Do **teacher forcing** for which frontier item to expand.

### 17.2 Mid training

Use a heuristic, e.g. expand the unresolved item with:

- highest baseline cost,
- largest support size,
- largest value uncertainty,
- largest predicted split benefit.

### 17.3 Late training

Optionally learn a frontier-selection policy as a separate lightweight module.

Do not couple frontier selection and split selection from day one.

---

## 18. Evaluation metrics

Track more than episode reward.

### 18.1 Per-state metrics

- top-1 candidate ranking accuracy,
- top-k hit rate for best split,
- value calibration error,
- factorable-branch prediction accuracy,
- immediate saving prediction error.

### 18.2 Search metrics

- search gain over raw policy,
- average number of simulations per improvement,
- transposition-table hit rate,
- factorization cache hit rate,
- average branch factor after candidate generation.

### 18.3 End-task metrics

- final circuit cost found,
- relative saving over baseline,
- solved fraction,
- wall-clock to improvement,
- reuse statistics (unique subproblems vs repeated hits).

### 18.4 Generalization metrics

- held-out polynomial families,
- held-out degrees / support sizes,
- held-out primes,
- held-out variable counts.

---

## 19. Ablation plan

Mandatory ablations:

1. No memoization.
2. No search, policy only.
3. No value head.
4. No elite self-imitation buffer.
5. Uniform replay vs prioritized replay.
6. Support-partition proposals only vs richer proposals.
7. Without Horner / elementary symmetric pretraining.
8. With and without frontier teacher forcing.
9. With and without final common-subexpression elimination.
10. Greedy search targets vs exploratory rollout targets.

---

## 20. Proposed module layout

Create a parallel implementation path rather than mutating the old code immediately.

```text
src/decomp_rl/
    __init__.py
    polynomial.py
    canonical.py
    factor_fp.py
    baseline_cost.py
    cost_model.py
    split_proposals.py
    family_generators.py
    decomp_env.py
    frontier_policy.py
    andor_search.py
    model.py
    losses.py
    replay.py
    elite_buffer.py
    train_supervised.py
    train_search_distill.py
    train_optional_rl.py
    evaluate.py
    config.py

scripts/
    generate_pretrain_dataset.py
    run_supervised_pretrain.py
    run_search_distill.py
    run_full_experiment.py
    run_eval_suite.py

tests/
    test_polynomial.py
    test_canonical.py
    test_factor_fp.py
    test_split_proposals.py
    test_horner_generator.py
    test_elementary_symmetric_generator.py
    test_decomp_env.py
    test_andor_search.py
    test_cost_model.py
    test_end_to_end_small.py
```

---

## 21. Detailed component specifications

### 21.1 `polynomial.py`

Responsibilities:

- sparse polynomial object over \(\mathbf F_p\),
- addition / subtraction / multiplication,
- monomial extraction,
- support operations,
- serialization / deserialization,
- metadata helpers.

### 21.2 `canonical.py`

Responsibilities:

- canonical ordering,
- coefficient normalization,
- factor normalization,
- stable hash keys,
- equivalence checks.

### 21.3 `factor_fp.py`

Responsibilities:

- wrap finite-field factorization backend,
- return normalized factor lists,
- cache results,
- log failures and timeouts.

### 21.4 `baseline_cost.py`

Responsibilities:

- exact cost for constants / variables / monomials,
- heuristic direct-construction cost,
- optional exact tiny-case solver.

### 21.5 `split_proposals.py`

Responsibilities:

- support-partition proposals,
- common-factor proposals,
- Horner proposals,
- family-template proposals,
- deduplication,
- candidate truncation to top-K heuristically before model scoring if necessary.

### 21.6 `family_generators.py`

Responsibilities:

- planted factorizable example generation,
- Horner trace generation,
- elementary symmetric trace generation,
- repeated-substructure synthetic families.

### 21.7 `decomp_env.py`

Responsibilities:

- frontier state transitions,
- factorization after splits,
- reward shaping,
- cache integration,
- episode termination.

### 21.8 `andor_search.py`

Responsibilities:

- PUCT selection over split candidates,
- AND backup,
- transposition-table updates,
- search trace extraction for distillation.

### 21.9 `model.py`

Responsibilities:

- polynomial encoder,
- candidate scoring head,
- value head,
- optional auxiliary heads.

### 21.10 `replay.py`

Responsibilities:

- prioritized task replay,
- search-distillation buffer,
- exact-trace buffer,
- mixed sampling.

### 21.11 `elite_buffer.py`

Responsibilities:

- store best found decomposition traces,
- rank by cost improvement,
- support self-imitation minibatches.

---

## 22. Pseudocode sketches

### 22.1 Environment step

```python
def step(state, poly_handle, action):
    f = state.frontier[poly_handle]
    g, h = action.g, action.h
    assert add(g, h) == f

    g_factors = factor_cache.get_or_factor(g)
    h_factors = factor_cache.get_or_factor(h)

    rebuild_g = rebuild_cost(g_factors)
    rebuild_h = rebuild_cost(h_factors)

    children = []
    for child in unresolved_children(g_factors, h_factors):
        if is_base_case(child):
            state.acc_cost += exact_base_cost(child)
        elif child in state.memo:
            state.acc_cost += state.memo[child].best_cost
        else:
            children.append(child)

    est_before = baseline_cost(f)
    est_after = 1 + rebuild_g + rebuild_h + sum(baseline_cost(c) for c in children)
    reward = est_before - est_after

    remove f from frontier
    add children to frontier
    state.acc_cost += 1 + rebuild_g + rebuild_h

    done = len(frontier) == 0
    return state, reward, done, info
```

### 22.2 AND/OR search backup

```python
def evaluate_split(f, split):
    g, h = split.g, split.h
    children, rebuild_g, rebuild_h = expand_split(g, h)

    total = 1 + rebuild_g + rebuild_h
    for child in children:
        total += estimate_cost(child)  # search, memo, or value net
    return total
```

### 22.3 Search-distillation loop

```python
def self_improve_step(batch_of_targets):
    training_examples = []
    for f in batch_of_targets:
        search_result = andor_search(root=f, model=current_model)
        training_examples.append(
            make_distillation_example(
                target=f,
                candidate_set=search_result.root_candidates,
                search_policy=search_result.root_policy,
                value_target=search_result.root_value,
                best_trace=search_result.best_trace,
            )
        )
        elite_buffer.maybe_add(search_result.best_trace)
    train_model(training_examples)
```

---

## 23. Recommended development order

### Phase 0: symbolic backbone

1. [x] Implement sparse polynomial representation over \(\mathbf F_p\).
2. [x] Implement canonicalization and hashing.
3. [x] Implement factorization wrapper and cache.
4. [x] Implement baseline and monomial cost functions.
5. [x] Write exhaustive tests for all of the above.

### Phase 1: non-learned decomposition solver

6. [x] Implement split candidate generation.
7. [x] Implement decomposition environment.
8. [x] Implement heuristic best-first or AO*-style search without neural nets.
9. [x] Confirm this already beats simple direct baselines on small structured families.

### Phase 2: supervised learning

10. [x] Implement synthetic trace generators.
11. [x] Build pretraining datasets.
12. [x] Implement polynomial encoder and candidate-scoring model.
13. [x] Train policy/value network on fixed datasets.
14. [x] Evaluate ranking accuracy and value calibration.

### Phase 3: search-guided self-improvement

15. [x] Implement AND/OR PUCT with network priors.
16. [x] Implement search-distillation dataset creation.
17. [x] Implement prioritized replay and elite-buffer self-imitation.
18. [x] Run iterative search-distill-train cycles.

### Phase 4: frontier and deployment refinements

19. [x] Add frontier-selection heuristics / policy.
20. [ ] Add DAG scoring with final CSE pass.
21. [x] Add stronger candidate templates.
22. [~] Add broader prime curriculum and larger target families.

### Phase 5: optional RL fine-tuning

23. [ ] Integrate PPO or SAC only if needed.
24. [ ] Compare against the original environment on common benchmarks.

---

## 24. Hyperparameter starting points

These are not final; they are sensible first settings.

### 24.1 Candidate generation

- `K_candidates = 32`
- support-partition fraction: 50%
- Horner/family templates: 25%
- common-factor templates: 15%
- random-biased proposals: 10%

### 24.2 Search

- simulations per root: 64 to 128 for small problems,
- PUCT exploration constant: start with standard AlphaZero-like values, tune empirically,
- search depth cap for first prototype: moderate,
- transposition reuse: always on.

### 24.3 Training

- optimizer: AdamW,
- learning rate: standard small-transformer range,
- value loss weight initially modest,
- auxiliary-loss weights small but nonzero,
- mixed minibatches from supervised/search/elite buffers.

### 24.4 Replay mixture example

For search-distillation stage:

- 40% recent search-distilled targets,
- 20% elite self-imitation traces,
- 20% exact-small traces,
- 20% structured synthetic traces.

---

## 25. Known failure modes and mitigations

### 25.1 Failure mode: useless splitting loops

Mitigation:

- always compare against direct baseline \(B(f)\),
- allow “stop and solve directly” fallback,
- prune clearly bad candidates.

### 25.2 Failure mode: formula bias / no reuse

Mitigation:

- canonical hashing,
- memoization,
- transposition table,
- final CSE pass,
- reuse-heavy curriculum.

### 25.3 Failure mode: candidate generator too weak

Mitigation:

- treat proposal quality as a core research problem,
- inject family templates,
- use heuristic and stochastic diversity together,
- add proposal ablations early.

### 25.4 Failure mode: factorization bottleneck

Mitigation:

- cache aggressively,
- precompute factorization for popular synthetic families,
- set fallback timeouts and logging.

### 25.5 Failure mode: overfitting to Horner / \(e_k\)

Mitigation:

- mix in random planted splits and exact-small optimal data,
- evaluate on held-out generic targets,
- decrease family-data fraction over time.

---

## 26. Benchmark plan

Use three benchmark categories.

### 26.1 Structured families

- Horner-generated targets,
- elementary symmetric polynomials,
- other known compact-recursive families if added later.

### 26.2 Synthetic generic targets

- random sparse polynomials with controlled support/degree,
- planted factorable-branch targets,
- repeated-substructure synthetic families.

### 26.3 Original project benchmark overlap

Where possible, compare against targets from the existing circuit-growth project so progress is measurable against the prior formulation.

---

## 27. Minimal viable product (MVP)

The MVP should be deliberately narrow.

### MVP scope

- 2 variables,
- one small prime, e.g. \(p=3\),
- support-partition and Horner proposals only,
- exact monomial/base costs,
- factorization cache,
- non-learned heuristic search,
- supervised candidate-ranking model,
- no frontier-selection learning yet.

### MVP success criterion

The non-learned plus supervised decomposition system should already show at least one of:

- lower average cost than a direct baseline,
- stronger performance than random split policies,
- easier learning curves than the original gate-growth environment on comparable toy tasks.

If the MVP does **not** show this, do not move to large-scale RL yet.

---

## 28. Detailed implementation checklist

Use this as the main codex checklist.

### A. Symbolic core

- [x] Implement sparse polynomial class over \(\mathbf F_p\).
- [x] Implement canonical coefficient reduction mod \(p\).
- [x] Implement canonical monomial ordering.
- [x] Implement deterministic serialization and hashing.
- [x] Implement polynomial add/subtract/multiply.
- [x] Implement metadata extraction (support size, degree profile, etc.).
- [x] Write unit tests for all polynomial ops.

### B. Factorization and cost model

- [x] Implement finite-field factorization wrapper.
- [x] Normalize factorization outputs consistently.
- [x] Cache factorization results by canonical hash.
- [x] Implement monomial-construction cost.
- [x] Implement rebuild cost from factor lists.
- [x] Implement direct baseline cost \(B(f)\).
- [x] Implement exact tiny-case solver if feasible.
- [x] Write tests for factorization and cost consistency.

### C. Split proposal system

- [x] Implement support-partition proposal generator.
- [x] Implement Horner proposal generator.
- [x] Implement common-factor proposal generator.
- [x] Implement recognized-family proposal injection.
- [x] Implement candidate deduplication and truncation.
- [x] Write tests ensuring each proposal satisfies \(f=g+h\).

### D. Environment

- [x] Implement frontier state structure.
- [x] Implement `reset`.
- [x] Implement `get_candidate_splits`.
- [x] Implement `step` with factorization and child insertion.
- [x] Implement immediate shaped reward.
- [x] Implement base-case termination logic.
- [x] Implement memoization integration.
- [x] Write environment-step tests.

### E. Search without learning

- [x] Implement heuristic best-first / AO* decomposition search.
- [x] Implement transposition table.
- [x] Implement cost backup.
- [x] Implement trace extraction.
- [x] Verify search on Horner and elementary symmetric examples.
- [ ] Benchmark against naive/random splitting.

### F. Pretraining data generation

- [x] Implement planted factorizable-split generator.
- [x] Implement univariate Horner trace generator.
- [x] Implement multivariate Horner trace generator.
- [x] Implement elementary symmetric trace generator.
- [x] Implement exact-small-trace generator.
- [x] Add mod-\(p\) validation checks to all generators.
- [x] Save datasets in a reusable format.

### G. Model

- [x] Implement polynomial encoder.
- [x] Implement candidate-scoring policy head.
- [x] Implement value head.
- [ ] Implement factorable-branch auxiliary head.
- [ ] Implement immediate-saving-bucket auxiliary head.
- [ ] Add config-driven architecture variants.

### H. Supervised training

- [x] Implement dataloaders for mixed pretraining sources.
- [x] Implement policy loss.
- [x] Implement value loss.
- [ ] Implement auxiliary losses.
- [x] Train initial candidate-ranker.
- [x] Evaluate ranking accuracy on held-out structured data.
- [ ] Evaluate value calibration.

### I. Search-guided self-improvement

- [x] Implement AND/OR PUCT.
- [x] Use model priors in search.
- [x] Extract root search policy targets.
- [x] Extract backed-up value targets.
- [x] Implement search-distillation training loop.
- [ ] Verify search improves over raw policy.

### J. Replay and self-imitation

- [x] Implement prioritized target replay.
- [x] Compute policy-gap priority signal.
- [x] Compute value-error priority signal.
- [x] Compute search-gain priority signal.
- [x] Implement elite buffer for best traces.
- [x] Interleave elite self-imitation minibatches.

### K. Frontier policy and DAG scoring

- [ ] Implement teacher-forced frontier choice in early training.
- [x] Implement heuristic frontier selection.
- [ ] Optionally implement learned frontier selector.
- [ ] Implement final common-subexpression-elimination DAG scoring.
- [ ] Verify reuse-heavy families benefit from memoization.

### L. Curriculum and training control

- [x] Implement curriculum over support size and degree.
- [ ] Implement curriculum over decomposition depth.
- [x] Implement curriculum over structured vs generic targets.
- [x] Implement curriculum over number of variables.
- [x] Implement curriculum over primes.
- [x] Track curriculum-stage progression metrics.

### M. Evaluation and ablations

- [x] Implement held-out evaluation suite.
- [x] Measure final cost reduction over baseline.
- [x] Measure search gain over raw policy.
- [x] Measure memoization hit rate.
- [ ] Run no-memoization ablation.
- [ ] Run no-search ablation.
- [ ] Run no-family-pretraining ablation.
- [ ] Run uniform-vs-prioritized replay ablation.
- [ ] Compare against original gate-growth baseline where feasible.

### N. Optional RL fine-tuning

- [ ] Wrap decomposition environment for PPO or SAC.
- [ ] Add replay-bootstrapped fine-tuning if needed.
- [ ] Compare fine-tuned results to pure search-distilled training.

### O. High-throughput learner and JAX acceleration

- [ ] Add optional JAX/Flax/Optax training stack.
- [ ] Implement fixed-shape padded batch schema for learner and inference.
- [ ] Implement bucketed candidate-count batching for shape-stable compilation.
- [ ] Implement JAX policy/value network with optional auxiliary heads.
- [ ] Implement JIT-compiled learner update step.
- [ ] Implement batched inference server for CPU search workers.
- [ ] Separate CPU search actors from GPU learner updates.
- [ ] Implement bounded Sage worker pool with health checks.
- [ ] Make batch sizing depend on currently free GPU memory.
- [ ] Verify GPU is busy during both learning and search-distillation inference.

### P. Research-system refinements for the next wave

- [ ] Implement teacher-forced frontier supervision in early training.
- [ ] Implement optional learned frontier selector.
- [ ] Implement final DAG / CSE scoring pass.
- [ ] Add value calibration evaluation.
- [ ] Add naive/random-split benchmark.
- [ ] Add ablation runners for memoization, search, pretraining mix, and replay strategy.

---

## 29. Concrete intermediate goals

These should be treated as release-style milestones.

### Milestone 1: symbolic substrate complete

Deliverables:

- polynomial class,
- factorization wrapper,
- cost model,
- 100% passing unit tests for symbolic core.

### Milestone 2: decomposition environment complete

Deliverables:

- working frontier environment,
- working split proposals,
- correct factorization-based transitions,
- correct reward shaping,
- small manual examples passing.

### Milestone 3: non-learned decomposition search beats random

Deliverables:

- heuristic search,
- best-first/AO* or simple AND/OR planner,
- measurable gain over random splits on toy tasks.

### Milestone 4: supervised pretraining is viable

Deliverables:

- trace generators,
- mixed dataset,
- candidate ranking significantly above random,
- value estimates correlated with actual savings.

### Milestone 5: search distillation improves the model

Deliverables:

- search-guided training loop,
- policy-search gap shrinking across iterations,
- elite buffer populated with useful traces.

### Milestone 6: memoized reuse is learned

Deliverables:

- improved performance on elementary symmetric families,
- measurable memo-hit rate,
- DAG scoring better than formula-only scoring.

### Milestone 7: benchmark-ready system

Deliverables:

- held-out evaluation suite,
- ablations,
- comparison to original formulation,
- reproducible training scripts.

---

## 30. Current repo status and next-wave plan (2026-04-22)

The repo now has a working symbolic/search backbone plus supervised warm start,
search distillation, replay, elite self-imitation, a Sage-backed factorization
worker, detached launch scripts, and cycle-0 checkpoint resume.

However, recent runs show that the next bottleneck is systems throughput rather
than missing basic functionality:

1. CPU/Sage generation and search dominate wall-clock time.
2. The learner, search, and evaluation still share one Python process.
3. GPU usage is bursty and often low because search-time inference is not yet
   served as a true batched GPU service.
4. The current Torch learner is still a lightweight baseline rather than a
   throughput-oriented actor/learner stack.

The recommended next implementation wave is therefore:

1. Keep symbolic search, split proposals, exact costs, and Sage factorization in
   Python.
2. Add a JAX learner plus batched inference plane for policy/value/aux training
   and search-time scoring.
3. Split the system into CPU search actors and a GPU learner/inference server.
4. Add final DAG/CSE scoring, frontier supervision, auxiliary heads, and
   ablation runners in the same wave.

Detailed implementation notes are recorded in:

- `docs/next_phase_jax_scaling_plan.md`

---

## 31. Final recommendations

1. **Start with the symbolic and search substrate first.** If the non-learned decomposition system does not already show promise, neural training will not rescue it.
2. **Use decomposition traces as the central supervised object.** Do not reduce everything to final circuit labels.
3. **Treat candidate generation as a first-class research component.** It will dominate performance.
4. **Prioritize memoization and canonicalization early.** Otherwise reuse will never emerge correctly.
5. **Build the system as self-improving search with imitation, not as PPO-from-scratch.**
6. **Use Horner and elementary symmetric families aggressively in warm start, then phase in generic targets.**
7. **Track search gain and memo-hit rate as primary scientific metrics.**

---

## 32. Suggested first week execution plan

If codex wants a concrete first implementation slice, do this first:

### Day 1–2
- implement sparse polynomial class,
- implement canonicalization,
- write tests.

### Day 3
- implement factorization wrapper,
- implement factorization cache,
- implement base-case / baseline costs.

### Day 4
- implement support-partition split generator,
- implement Horner split generator,
- test split correctness.

### Day 5
- implement `DecompEnv.step`,
- run manual toy rollouts,
- verify reward signs on obvious good/bad splits.

### Day 6
- implement non-learned best-first decomposition search,
- test on Horner and elementary symmetric toy instances.

### Day 7
- implement planted / Horner trace dataset generator,
- train a first candidate-ranking model.

This first-week slice should already tell whether the split-based formulation is substantially more learnable than gate-by-gate circuit construction.

---

## 33. Closing note

This proposal is not just a small tweak to the original RL environment. It is a reframing of the task from **constructive circuit synthesis** into **recursive decomposition discovery with symbolic closure under factorization**. That should make it a stronger platform both for efficient arithmetic-circuit search and for broader self-improving-search themes.
