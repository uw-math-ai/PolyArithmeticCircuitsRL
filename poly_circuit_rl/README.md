# poly_circuit_rl

RL agent for discovering short polynomial arithmetic circuits with DQN + Hindsight Experience Replay (HER) + Monte Carlo Tree Search (MCTS).

Given a target polynomial (for example `x0^2 + x0*x1`), the agent builds a circuit using only `ADD` / `MUL` over input variables and `const_1`, then chooses an output node. MCTS with Q-network priors enables lookahead planning to discover optimal (shortest) circuits rather than naive term-by-term constructions.

## Architecture at a glance

- State: flat observation with `L` node slots + target eval vector + normalized `steps_left`
- Node features: type/op one-hots, parent indices, position index, leaf id, eval vector
- Target representation: eval vector at fixed random points
- Encoder: transformer with causal mask (construction order)
- Action head: bilinear scores for `ADD(i,j)` and `MUL(i,j)`, linear scores for `SET_OUTPUT(i)` and `STOP`
- Action selection: MCTS with PUCT (AlphaZero-style, Q-network as prior and leaf evaluator) or epsilon-greedy fallback

## Project layout

```text
poly_circuit_rl/
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   └── README.md
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── README.md
├── poly_circuit_rl/
│   ├── config.py
│   ├── core/
│   │   └── README.md
│   ├── env/
│   │   └── README.md
│   └── rl/
│       └── README.md
└── tests/
    └── README.md
```

## Module docs

- [core/README.md](poly_circuit_rl/core/README.md)
- [env/README.md](poly_circuit_rl/env/README.md)
- [rl/README.md](poly_circuit_rl/rl/README.md)
- [scripts/README.md](scripts/README.md)
- [configs/README.md](configs/README.md)
- [tests/README.md](tests/README.md)

## Install

```bash
cd poly_circuit_rl
pip install -e .

# Optional extras:
pip install -e ".[interesting]"  # sympy + networkx
pip install -e ".[dev]"          # pytest
pip install -e ".[wandb]"        # Weights & Biases logging
```

## Train

```bash
# Basic training (MCTS enabled by default)
python scripts/train.py

# Use precomputed interesting-polynomial JSONL
python scripts/train.py \
  --interesting ../Game-Board-Generation/pre-training-data/game_board_C4.analysis.jsonl

# Disable auto-generated interesting polynomials
python scripts/train.py --no-auto-interesting

# Bound auto-generation graph size
python scripts/train.py --gen-max-graph-nodes 20000 --gen-max-successors 30

# Disable MCTS (fall back to epsilon-greedy)
python scripts/train.py --no-mcts

# Tune MCTS parameters
python scripts/train.py --mcts_simulations 100 --mcts_c_puct 2.0 --mcts_temperature 0.5

# Enable Weights & Biases logging
python scripts/train.py --wandb_project poly-circuit-rl --wandb_entity your-team
```

## Evaluate

```bash
python scripts/evaluate.py --checkpoint runs/best_lvl2.pt --max_ops 4 --episodes 200
```

## Reward and termination

| Event | Reward / effect |
|---|---|
| `ADD` / `MUL` | `-step_cost` |
| `ADD` producing factorizable result | additional `-factor_shaping_coeff` penalty |
| `SET_OUTPUT` | `0.0` |
| `STOP` | `0.0`, truncates episode |
| Output matches target | `+1.0` and terminates |
| Invalid action | `-1.0` and truncates |

**Factorization shaping** (`factor_shaping_coeff`, default `0.1`): when an `ADD` operation produces a polynomial that SymPy can factor, the agent receives a penalty. The intuition is that factorizable results should have been built via `MUL` of their factors, which uses fewer operations. This shaping is only applied during real episodes, not during MCTS simulations (for performance).

**Eval-distance shaping** (`shaping_coeff`, default `0.0`): disabled by default. When enabled, gives a bonus when a newly created node improves best eval-distance to target. Disabled because it can mislead the agent toward naive term-by-term construction.

Episode can truncate by:
- explicit `STOP`
- exhausting op budget after a post-budget resolve step
- hard step cap (`max_episode_steps`, default derived from `max_ops + max_nodes + 5` when unset)

## Target polynomial sources

1. `RandomCircuitSampler` (default): random circuits, random node target.
2. `InterestingPolynomialSampler`: precomputed JSONL with path-multiplicity metadata.
3. `GenerativeInterestingPolynomialSampler`: auto-generates interesting targets from graph enumeration when JSONL is not provided and auto-generation is enabled. Filters for **shortcut polynomials** where the optimal circuit is significantly shorter than naive monomial-by-monomial construction (gap >= 2 operations by default).

At curriculum levels `>= 1`, training can mix interesting and random targets via `interesting_ratio`.

## Config highlights (`poly_circuit_rl/config.py`)

| Parameter | Default | Purpose |
|---|---|---|
| `n_vars` | `2` | Number of polynomial variables |
| `max_ops` | `4` | Max `ADD`/`MUL` budget per episode |
| `L` | `16` | Visible node slots in observation |
| `m` | `16` | Eval-point count for fingerprints |
| `shaping_coeff` | `0.0` | Eval-distance shaping strength (disabled) |
| `factor_shaping_coeff` | `0.1` | Factorization penalty for ADD nodes |
| `eval_norm_scale` | `100.0` | `tanh(v/scale)` normalization for eval vectors |
| `max_episode_steps` | `None` | Hard episode step cap override |
| `curriculum_levels` | `(1,2,3,4,5,6)` | Curriculum op levels |
| `auto_interesting` | `True` | Enable fallback auto-generation |
| `gen_max_graph_nodes` | `100000` | Auto-generation graph size cap |
| `gen_max_successors` | `50` | Auto-generation per-node expansion cap |
| `use_mcts` | `True` | Enable MCTS for action selection |
| `mcts_simulations` | `50` | MCTS simulations per action |
| `mcts_c_puct` | `1.5` | PUCT exploration constant |
| `mcts_temperature` | `1.0` | Temperature for visit-count action selection |
| `total_steps` | `500000` | Total environment steps |

## Tests

```bash
# From poly_circuit_rl/
python -m unittest discover -s tests -p 'test_*.py' -v

# Optional
python -m pytest tests/ -v
```
