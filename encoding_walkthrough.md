# Polynomial Encoding and PPO+MCTS Lifecycle Walkthrough

This note explains, end-to-end, how polynomial state/action encoding works in this project and how PPO+MCTS uses it during training.

## 1) Core Representations

### 1.1 Polynomial representation (`FastPoly`)
Polynomials are stored as dense coefficient tensors over `F_p[x0, ..., x_{n-1}]`.

- Tensor index = monomial exponents `(a0, a1, ..., an-1)`
- Tensor value = coefficient modulo `p`

Implemented in:
- `src/environment/fast_polynomial.py`

For `n_variables=2`, `max_degree=2`, shape is `(3,3)`.
Entry `coeffs[1,2]` means coefficient of `x0^1 * x1^2`.

### 1.2 Observation encoding used by the network
Environment observation has 3 parts:
- `graph`: current circuit graph (node type features + edges)
- `target`: flattened coefficient vector for target polynomial, normalized by `/ mod`
- `mask`: valid action mask over flat action space

Implemented in:
- `src/environment/circuit_game.py` (`get_observation`, `_build_graph`, `_encode_target`)

Important: current circuit nodes are encoded structurally (graph features), not by full coefficient vectors. The target polynomial gets the dense coefficient vector.

### 1.3 Action encoding
Action is `(op, i, j)` where:
- `op=0` add, `op=1` multiply
- `i, j` are node indices, with `i <= j`

Encoded into one integer:

- `pair_idx = i * N - i*(i-1)/2 + (j - i)`
- `action_idx = 2 * pair_idx + op`

where `N = max_nodes`.

Implemented in:
- `src/environment/action_space.py` (`encode_action`, `decode_action`)

---

## 2) Where `pair_idx` and `action_idx` live

### `pair_idx`
- Temporary local variable only inside encode/decode math
- Not stored persistently

### `action_idx`
- Persistently used across the full pipeline:
  - chosen by MCTS / policy
  - executed by environment (`decode_action`)
  - stored in PPO+MCTS rollout buffer
  - used for valid-action masking

Key files:
- `src/environment/action_space.py`
- `src/environment/circuit_game.py`
- `src/algorithms/mcts.py`
- `src/algorithms/ppo_mcts.py`

---

## 3) Full PPO+MCTS lifecycle (start to finish)

## 3.1 Startup
Running:

```bash
python -m src.main --algorithm ppo-mcts ...
```

creates:
- `PolicyValueNet`
- `PPOMCTSTrainer`
- inside trainer: `MCTS` + `CircuitGame`

Code:
- `src/main.py`
- `src/algorithms/ppo_mcts.py`

## 3.2 Target selection
Each episode samples a target polynomial from BFS game-board at current curriculum complexity.

Code:
- `src/game_board/generator.py` (`build_game_board`, `sample_target`)
- `src/algorithms/ppo_mcts.py` (`collect_rollouts`)

## 3.3 Episode reset
`env.reset(target_poly)` sets:
- `nodes = [x0, x1, ..., x_{n-1}, 1]`
- clears `edges`
- resets counters and optional factor-library episode state
- returns observation (`graph`, `target`, `mask`)

Code:
- `src/environment/circuit_game.py`

## 3.4 Per-step decision (PPO+MCTS)
For each step in episode:
1. network estimates value `V(s)`
2. MCTS runs simulations from current state (`game.clone()` internally)
3. MCTS returns visit-count distribution over `action_idx`
4. sample/select action
5. environment executes action (`env.step(action_idx)`)
6. store transition in rollout buffer:
   - `obs, action, reward, mcts_log_prob, value, done`

Code:
- `src/algorithms/mcts.py`
- `src/algorithms/ppo_mcts.py`

## 3.5 Environment transition details
Inside `env.step(action_idx)`:
1. decode `action_idx -> (op, i, j)`
2. compute new polynomial (`+` or `*`, mod `p`, degree-truncated)
3. append to `self.nodes`
4. append node type to `self.node_types`
5. append operand edges to `self.edges`
6. compute reward / done / info

Code:
- `src/environment/circuit_game.py`

This is where the newly created node is stored as learning proceeds:
- `self.nodes` (primary polynomial store)
- plus structural metadata in `self.node_types` and `self.edges`

## 3.6 Learning update
After collecting `steps_per_update` transitions:
1. compute GAE advantages/returns
2. PPO update with clipped ratio:

`ratio = pi_theta(a|s) / pi_MCTS(a|s)`

3. update network weights
4. optional curriculum step

Code:
- `src/algorithms/ppo_mcts.py`

## 3.7 Iterative improvement
Updated model improves MCTS priors/values in next iteration, creating Expert Iteration loop.

---

## 4) Concrete action-index example

Question: why is `encode_action(0,1,3,max_nodes=5) = 14`?

Compute:
- `pair_idx = 1*5 - (1*0)/2 + (3-1) = 7`
- `action_idx = 2*7 + 0 = 14`

Interpretation:
- pair `(1,3)` is upper-triangle pair index 7 (0-based)
- op `0` (add) takes even slot

---

## 5) Concrete mini trajectory example

Assume:
- `n_variables=2`
- initial nodes after reset: `[x0, x1, 1]`
- target: `2*x0 + x1`
- `max_nodes=5`

One successful 2-step sequence:

1. `add(x0, x0)`
   - `(op,i,j) = (0,0,0)`
   - `action_idx = 0`
   - new node index 3 = `2*x0`

2. `add(x1, node3)`
   - `(op,i,j) = (0,1,3)`
   - `action_idx = 14`
   - new node index 4 = `x1 + 2*x0` (target reached)

Storage progression:
- after step 1: `nodes = [x0, x1, 1, 2*x0]`
- after step 2: `nodes = [x0, x1, 1, 2*x0, x1+2*x0]`

---

## 6) Quick file map

- Polynomial backend: `src/environment/fast_polynomial.py`
- Action encoding: `src/environment/action_space.py`
- Environment state/obs/step: `src/environment/circuit_game.py`
- Target sampling/game board: `src/game_board/generator.py`
- MCTS search: `src/algorithms/mcts.py`
- PPO+MCTS training loop: `src/algorithms/ppo_mcts.py`
- Entrypoint: `src/main.py`
- Policy-value network: `src/models/policy_value_net.py`
