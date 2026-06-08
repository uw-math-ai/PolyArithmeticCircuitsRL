# Roadmap

This roadmap is intentionally conditional. The next step depends on sweep results, not a fixed feature list.

## Completed Recent Milestones

Done:

- candidate recall and candidate-rank diagnostics
- support-geometry / Minkowski candidate features
- Gumbel / Sequential-Halving planner over generated candidates
- fair expansion accounting through `SearchHistory.num_expansions`
- matched-budget four-method planner sweep:
  - `beam_heuristic`
  - `beam_guided`
  - `gumbel_heuristic`
  - `gumbel_guided`
- W&B and Slurm support for server runs

## Milestone 8: Large Sweeps And Failure Analysis

Run structured benchmark sweeps across:

- beam width
- candidate budget
- tier-2 budget
- expansion budget
- heuristic Beam, guided Beam, heuristic Gumbel, and guided Gumbel
- multiple Gumbel seeds
- train/eval splits

Analyze:

- solve-rate deltas
- expansions
- solution sizes
- failures by family and intended complexity
- whether correct actions appear in candidate sets
- whether candidate recall misses or planner branch dropping is the dominant failure mode

Decision:

- if guided improves search, expand benchmark coverage and write results
- if Gumbel improves search, inspect families where Beam drops delayed-good branches
- if guided and heuristic match everywhere, inspect whether the heuristic top-K already solves the benchmark
- if search fails because candidates are missing, improve candidate generation
- if candidates exist but beam drops delayed-good branches, add value guidance or MCTS

## Milestone 9: Candidate Generation Improvements

Only if candidate recall diagnostics show missing candidates.

Possible additions:

- target factorization over `F_p`
- structured residual candidates
- controlled scalar/low-information handling
- more algebraic completion checks
- learned candidate proposal / hybrid candidate-set quotas

Constraint: keep exact arithmetic and avoid old factor-library state.

## Milestone 10: Value Head Or MCTS

Only if failure analysis shows that useful candidates exist but Beam and current Gumbel search still drop them.

Possible additions:

- value estimate for partial states
- MCTS with learned policy/ranker prior
- delayed-return preference targets

Correctness remains exact verification. The current Gumbel planner is not a full value-guided MCTS system.

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
- heuristic Gumbel / Sequential-Halving search
- learned-ranker guided Gumbel / Sequential-Halving search
- older `gumbel` or flat-RL baselines as external baselines
- possibly symbolic baselines if added later

Do not import old code into the new pipeline. Treat old systems as separate experiment runners.
