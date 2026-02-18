# poly_circuit_rl

RL agent for discovering optimal polynomial arithmetic circuits using DQN + Hindsight Experience Replay (HER).

Given a target polynomial (e.g. `x0^2 + x0*x1`), the agent learns to construct a minimal-step arithmetic circuit using only addition and multiplication operations over input variables.

## Architecture

```
STATE: Node token matrix  X ∈ R^{L × d_node}
  Per node: type_onehot | op_onehot | parent_embeddings | pos_embedding | leaf_id | eval_vector
  + Target eval vector → TargetEncoder → target_embed
  + Steps left (normalized)

ENCODER: Transformer with causal mask (construction order)
  Input projection → N layers × TransformerEncoderLayer → node embeddings H

ACTION HEAD: Bilinear structured scoring
  Q_add[i,j] = h_i^T W_add h_j     (for all pairs i ≤ j)
  Q_mul[i,j] = h_i^T W_mul h_j
  Q_set_output[i] = w_out^T h_i
  Q_stop = w_stop^T mean_pool(H)
  → Flatten to action_dim Q-values → mask invalid actions → argmax
```

Key design choices:
- **Parent pointer embeddings**: encode DAG structure in O(N) via learned index embeddings
- **Eval vectors**: polynomial evaluated at m random points (Schwartz-Zippel lemma ensures uniqueness)
- **Step cost reward**: -0.05 per ADD/MUL + 1.0 on solve → incentivizes minimal circuits
- **HER**: relabels failed episodes with achievable goals → dense reward signal
- **Curriculum**: starts with 1-op targets, advances to 4-op when success rate > 80%

## Project Structure

```
poly_circuit_rl/
├── pyproject.toml              # Package metadata and dependencies
├── configs/
│   └── default.yaml            # Default hyperparameters (reference)
├── scripts/
│   ├── train.py                # Training entrypoint
│   └── evaluate.py             # Evaluation entrypoint
├── src/poly_circuit_rl/        # Main package
│   ├── config.py               # Config dataclass with all hyperparameters
│   ├── core/                   # Polynomial math and circuit primitives
│   ├── env/                    # Gymnasium environment and observation encoding
│   └── rl/                     # DQN agent, transformer network, replay buffer
└── tests/                      # Unit tests (32 tests)
```

See each module's README for details:
- [core/README.md](src/poly_circuit_rl/core/README.md) — polynomial arithmetic, circuit builder, action encoding
- [env/README.md](src/poly_circuit_rl/env/README.md) — gymnasium environment, observation layout, target samplers
- [rl/README.md](src/poly_circuit_rl/rl/README.md) — transformer Q-network, DQN agent, HER replay buffer, training loop

## Quick Start

### Install

```bash
cd poly_circuit_rl
pip install -e .

# For interesting polynomial data (requires sympy):
pip install -e ".[interesting]"

# For running tests:
pip install -e ".[dev]"
```

### Train

```bash
# Basic training with default config
python scripts/train.py

# With interesting polynomial data for mixed sampling
python scripts/train.py --interesting ../Game-Board-Generation/pre-training-data/game_board_C4.analysis.jsonl

# Override hyperparameters
python scripts/train.py --n_vars 2 --max_ops 4 --d_model 64 --total_steps 500000 --log_dir runs/exp1
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint runs/best_lvl2.pt --max_ops 4 --episodes 200
```

### Run Tests

```bash
# From the poly_circuit_rl/ directory
python -m pytest tests/ -v

# If pytest fails due to a hydra plugin conflict in your environment:
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Reward Design

| Event | Reward |
|-------|--------|
| ADD or MUL operation | -0.05 (step cost) |
| SET_OUTPUT or STOP | 0.0 |
| Output matches target | +1.0 |

A k-step optimal circuit earns total return: `1.0 - k × 0.05`

## Training Data

Two sources of target polynomials:

1. **RandomCircuitSampler** (default): builds random circuits and picks a random node's polynomial as the target. Fast, but most targets only have one construction path.

2. **InterestingPolynomialSampler** (optional): loads pre-computed polynomials from analysis JSONL files where each polynomial has *multiple* shortest construction paths. These are more interesting for learning optimal circuits.

At curriculum level >= 1, training uses a mix: 70% interesting polynomials + 30% random (configurable via `interesting_ratio`).

## Config

All hyperparameters are in `Config` (see [config.py](src/poly_circuit_rl/config.py)):

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_vars | 2 | Number of polynomial variables |
| max_ops | 4 | Max ADD/MUL operations per episode |
| L | 16 | Max visible nodes in observation |
| m | 16 | Number of eval points (fingerprint) |
| d_model | 64 | Transformer hidden dimension |
| n_heads | 4 | Number of attention heads |
| n_layers | 3 | Number of transformer layers |
| step_cost | 0.05 | Per-operation reward penalty |
| total_steps | 500,000 | Total environment steps |
| curriculum_levels | (1,2,3,4) | Max ops per curriculum level |
