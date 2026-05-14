# Method

This project treats arithmetic circuit synthesis as exact constructive search guided by learned candidate ranking. The model is a search heuristic, not a substitute for symbolic arithmetic or verification.

## Problem Formulation

Input:

- a finite field `F_p`
- a target polynomial `f`
- named variables `x_0, ..., x_{n-1}`
- a per-variable degree cap
- an operation budget

The initial circuit state contains:

```text
x_0, x_1, ..., x_{n-1}, 1
```

At each step, the search may apply an action:

```text
add(i, j)
mul(i, j)
```

where `i` and `j` index existing circuit nodes. The result becomes a new node. The circuit complexity used here is the number of constructed gates or operations.

Success is exact:

```text
target polynomial key is present in CircuitState.node_keys
```

Equivalently, replaying the action trace with exact polynomial arithmetic reconstructs the target exactly in the same field and domain.

## Why Not Flat-Action RL First

A flat action formulation exposes all possible `(op, i, j)` choices directly to the policy. This is difficult for symbolic circuit synthesis because:

- the action space grows roughly quadratically with the number of current nodes
- rewards are sparse and delayed
- useful intermediates may have little direct target-support overlap
- the policy must simultaneously discover algebraic structure, action relevance, and long-horizon consequences
- reward shaping can obscure whether a failure came from representation, search, reward design, or the action space

The older `gumbel` direction was closer to a flat-action or reward-shaped construction approach. It is useful historical context, but the current implementation deliberately separates exact symbolic generation, heuristic search, learned ranking, and verification.

## Current Approach

The current pipeline is:

```text
ProblemInstance
  -> CircuitState
  -> CandidateGenerator
  -> HeuristicScorer
  -> CandidateRanker
  -> BeamSearch
  -> ExactVerifier
  -> PreferenceExtractor
  -> BootstrapTrainer
```

### Candidate Generation

For a state, the generator enumerates canonical add/mul pairs:

```text
for i in range(num_nodes):
    for j in range(i, num_nodes):
        add(i, j)
        mul(i, j)
```

Each result is computed exactly. Degree-cap overflow raises in the polynomial backend and is skipped as an invalid candidate for the current capped search domain.

Only safe hard filters are used:

- zero result, unless zero is the target
- duplicate result already in the current state, unless it is the target
- exhausted operation budget

The generator intentionally does not filter out candidates solely because they have zero direct target-support overlap. Intermediates such as `x + y` for `(x + y)^2` are important even when they do not look like a target term.

### Two-Tier Features

Tier 1 computes cheap symbolic features for every surviving candidate:

- target equality
- degree and degree gap
- support size
- support overlap
- target coverage
- residual existence in current nodes

Tier 2 computes more expensive exact features for top tier-1 candidates:

- exact divisibility by checking `target = result * quotient`
- quotient existence in current nodes
- one-step add completion
- one-step multiply completion

Exact divisibility never returns a quotient unless multiplication verifies exactly.

### Heuristic Scoring

The heuristic score is a symbolic baseline. It rewards:

- exact target construction
- residual completion
- support overlap and coverage
- exact divisibility
- quotient existence
- one-step completion

It penalizes some outside-support and size features, but it is intentionally conservative. It is a baseline to beat, not a final solver.

### Beam Search

Beam search keeps a bounded set of states ranked by candidate score minus a small operation cost. It records every expanded candidate in `SearchHistory`, including:

- source state
- full candidate list
- chosen candidate
- next state
- depth
- state score

This makes preference extraction and debugging possible after search.

### Learned Candidate Ranker

The ranker learns:

```text
score(instance, state, candidate)
```

It does not directly emit actions or circuits. The current model is a fixed-feature MLP. Its score is combined with the symbolic score:

```text
total_score = heuristic_score + lambda_model * model_score
```

Correctness still depends only on exact verification.

### Preference Extraction

When a search succeeds, `extract_preferences` selects the best verified circuit, recovers its trace, and creates pairwise examples:

```text
candidate on best trace > alternative candidate from the same source state
```

Trace steps are matched by action-prefix equality, not object identity:

```text
record.state.actions == best.actions[:t]
record.candidate.action == best.actions[t]
```

This keeps the training signal tied to the actual search history.

### Bootstrapped Training

The bootstrap loop repeats:

```text
search
-> extract preferences
-> train ranker
-> evaluate heuristic-only and guided search
-> promote lambda_model only if validation is not worse
```

Validation instances are used for lambda promotion and evaluation only. They are not used for preference extraction.

## Why This Approach

The design is motivated by lessons from AI-for-combinatorial-optimization discussions: learned methods often work best when they guide search or heuristics rather than replace combinatorial reasoning entirely. [TODO: cite RL4CO/AI4CO survey]

This problem can be viewed as a deterministic MDP:

- state: current circuit
- action: add/mul candidate
- transition: exact symbolic operation
- objective: construct a verified target with a small circuit

The feasible action set is state-dependent and grows quickly. Candidate generation and beam search expose useful structure before learning is applied. The learned model gets a smaller, more meaningful ranking task instead of a raw construction problem.

## Comparison With Previous Direction

| Aspect | Previous Direction | Current Direction |
| --- | --- | --- |
| Action choice | flat or mostly flat action selection | candidate generator proposes plausible actions before search |
| Guidance | reward/factor guidance after actions | symbolic heuristic score before expansion |
| Learning burden | model has to discover useful actions among many pairs | model learns to reorder generated candidates |
| Search | closer to reward-shaped construction | beam search handles multi-step construction |
| Correctness | dependent on training/search behavior plus checks | exact verifier is the correctness boundary |
| Diagnosability | harder to isolate representation, reward, search, or action-space failures | clean ablations: heuristic-only vs heuristic+ranker |

## What Is Learned

The model learns a scalar candidate scoring function. It consumes:

- problem and target features
- current state features
- action features
- candidate symbolic features
- heuristic scores and tags

The current learner is deliberately simple so benchmark results can isolate whether learned ranking helps at all before adding more expressive models.

## What Is Not Yet Implemented

- MCTS
- value head
- Transformer or GNN ranker
- local rewrite improvement
- scalar penalties
- model-aware candidate preselection beyond heuristic top-K
- broad baseline comparisons against the older flat-RL path
