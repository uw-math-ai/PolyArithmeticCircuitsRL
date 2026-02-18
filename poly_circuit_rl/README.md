# poly_circuit_rl

RL agent for discovering short polynomial arithmetic circuits with DQN + Hindsight Experience Replay (HER).

Given a target polynomial (for example `x0^2 + x0*x1`), the agent builds a circuit using only `ADD` / `MUL` over input variables and `const_1`, then chooses an output node.

## Architecture at a glance

- State: flat observation with `L` node slots + target eval vector + normalized `steps_left`
- Node features: type/op one-hots, parent indices, position index, leaf id, eval vector
- Target representation: eval vector at fixed random points
- Encoder: transformer with causal mask (construction order)
- Action head: bilinear scores for `ADD(i,j)` and `MUL(i,j)`, linear scores for `SET_OUTPUT(i)` and `STOP`
- Action selection: invalid actions masked before argmax

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
```

## Train

```bash
# Basic training
python scripts/train.py

# Use precomputed interesting-polynomial JSONL
python scripts/train.py \
  --interesting ../Game-Board-Generation/pre-training-data/game_board_C4.analysis.jsonl

# Disable auto-generated interesting polynomials
python scripts/train.py --no-auto-interesting

# Bound auto-generation graph size
python scripts/train.py --gen-max-graph-nodes 20000 --gen-max-successors 30

# Adjust shaping reward strength
python scripts/train.py --shaping_coeff 0.2
```

## Evaluate

```bash
python scripts/evaluate.py --checkpoint runs/best_lvl2.pt --max_ops 4 --episodes 200
```

## Reward and termination

| Event | Reward / effect |
|---|---|
| `ADD` / `MUL` | `-step_cost` + optional shaping bonus |
| `SET_OUTPUT` | `0.0` |
| `STOP` | `0.0`, truncates episode |
| Output matches target | `+1.0` and terminates |
| Invalid action | `-1.0` and truncates |

Shaping bonus is controlled by `shaping_coeff` and is given only when a newly created node improves best eval-distance-to-target.

Episode can truncate by:
- explicit `STOP`
- exhausting op budget after a post-budget resolve step
- hard step cap (`max_episode_steps`, default derived from `max_ops + max_nodes + 5` when unset)

## Target polynomial sources

1. `RandomCircuitSampler` (default): random circuits, random node target.
2. `InterestingPolynomialSampler`: precomputed JSONL with path-multiplicity metadata.
3. `GenerativeInterestingPolynomialSampler`: auto-generates interesting targets from graph enumeration when JSONL is not provided and auto-generation is enabled.

At curriculum levels `>= 1`, training can mix interesting and random targets via `interesting_ratio`.

## Config highlights (`poly_circuit_rl/config.py`)

| Parameter | Default | Purpose |
|---|---|---|
| `n_vars` | `2` | Number of polynomial variables |
| `max_ops` | `4` | Max `ADD`/`MUL` budget per episode |
| `L` | `16` | Visible node slots in observation |
| `m` | `16` | Eval-point count for fingerprints |
| `shaping_coeff` | `0.3` | Eval-distance shaping strength |
| `eval_norm_scale` | `100.0` | `tanh(v/scale)` normalization for eval vectors |
| `max_episode_steps` | `None` | Hard episode step cap override |
| `curriculum_levels` | `(1,2,3,4,5,6)` | Curriculum op levels |
| `auto_interesting` | `True` | Enable fallback auto-generation |
| `gen_max_graph_nodes` | `100000` | Auto-generation graph node cap |
| `gen_max_successors` | `50` | Auto-generation per-node expansion cap |
| `total_steps` | `500000` | Total environment steps |

## Tests

```bash
# From poly_circuit_rl/
python -m unittest discover -s tests -p 'test_*.py' -v

# Optional
python -m pytest tests/ -v
```
