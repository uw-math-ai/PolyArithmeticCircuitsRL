# Roadmap

This roadmap is intentionally conditional. The next step depends on sweep results, not a fixed feature list.

## Milestone 8: Large Sweeps And Failure Analysis

Run structured benchmark sweeps across:

- beam width
- candidate budget
- tier-2 budget
- heuristic-only versus guided
- train/eval splits

Analyze:

- solve-rate deltas
- expansions
- solution sizes
- failures by family and intended complexity
- whether correct actions appear in candidate sets

Decision:

- if guided improves search, expand benchmark coverage and write results
- if guided and heuristic match everywhere, inspect whether the heuristic top-K already solves the benchmark
- if search fails because candidates are missing, improve candidate generation
- if candidates exist but beam drops delayed-good branches, add value guidance or MCTS

## Milestone 9: Candidate Generation Improvements

Only if failure analysis shows missing candidates.

Possible additions:

- target factorization over `F_p`
- structured residual candidates
- controlled scalar/low-information handling
- more algebraic completion checks

Constraint: keep exact arithmetic and avoid old factor-library state.

## Milestone 10: Value Head Or MCTS

Only if failure analysis shows that useful candidates exist but beam search drops them.

Possible additions:

- value estimate for partial states
- MCTS with learned policy/ranker prior
- delayed-return preference targets

Correctness remains exact verification.

## Milestone 11: Transformer Or GNN Ranker

Only if MLP fixed features appear insufficient.

Possible inputs:

- node graph
- action endpoints
- polynomial support features
- target-conditioned attention

The goal would be improved ranking, not direct unverified circuit generation.

## Milestone 12: Baselines And Comparisons

Compare against:

- heuristic-only beam search
- learned-ranker guided beam search
- older `gumbel` or flat-RL baselines as external baselines
- possibly symbolic baselines if added later

Do not import old code into the new pipeline. Treat old systems as separate experiment runners.
