# Polynomial Arithmetic Circuits RL

Learning to construct minimal arithmetic circuits for target polynomials using reinforcement learning. An agent builds circuits by composing addition and multiplication operations over polynomial nodes, with all arithmetic performed modulo a small prime $p$.

Two algorithms are implemented for comparison:
- **PPO** (Proximal Policy Optimization) — a policy-gradient baseline
- **AlphaZero** — MCTS guided by a learned neural network (primary method)
- **SAC** (Soft Actor-Critic, discrete masked-action variant) — off-policy baseline

---

## Table of Contents

- [Problem Formulation](#problem-formulation)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Algorithms](#algorithms)
  - [Environment and Reward](#environment-and-reward)
  - [PPO](#ppo-proximal-policy-optimization)
  - [AlphaZero (MCTS + Neural Network)](#alphazero-mcts--neural-network)
  - [Curriculum Learning](#curriculum-learning)
- [Neural Network Architecture](#neural-network-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)

---

## Problem Formulation

**Given:** A target polynomial $f^* \in \mathbb{F}_p[x_0, x_1, \ldots, x_{n-1}]$.

**Goal:** Construct an arithmetic circuit $C$ using the fewest operations such that $C$ computes $f^*$.

The agent starts with input nodes $\{x_0, x_1, \ldots, x_{n-1}, 1\}$ and at each step selects:
1. An **operation** $\mathrm{op} \in \{+, \times\}$
2. Two **existing nodes** $v_i, v_j$ (with $i \le j$)

This produces a new node $v_k = v_i \;\mathrm{op}\; v_j$ whose polynomial is reduced modulo $p$. The episode succeeds when any node computes $f^*$.

**Default configuration:** $p = 5$, $n = 2$ variables, max 6 operations, max 10 steps per episode.

---

## Project Structure

```
src/
├── config.py                     # Centralized configuration dataclass
├── environment/
│   ├── fast_polynomial.py        # Fast numpy-based polynomial arithmetic (primary backend)
│   ├── polynomial_utils.py       # SymPy helpers + SymPy↔FastPoly conversions
│   ├── action_space.py           # Action encoding/decoding with O(1) closed-form inverse
│   ├── circuit_game.py           # Core game environment (Gymnasium-style API)
│   └── factor_library.py         # Session-level factor cache + subgoal reward logic
├── game_board/
│   └── generator.py              # BFS DAG builder and target polynomial sampling
├── models/
│   ├── gnn_encoder.py            # GCN encoder for circuit graphs
│   └── policy_value_net.py       # Shared policy-value network (GNN + target encoder)
├── algorithms/
│   ├── ppo.py                    # PPO training loop with GAE
│   ├── sac.py                    # Discrete SAC with masked actions + stratified replay
│   ├── mcts.py                   # Neural MCTS (AlphaZero-style, no random rollouts)
│   └── alphazero.py              # AlphaZero self-play + training
├── evaluation/
│   └── evaluate.py               # Evaluation harness across complexity levels
└── main.py                       # CLI entry point for training and evaluation

tests/
├── test_fast_polynomial.py       # Tests for fast numpy polynomial backend
├── test_environment.py           # Tests for polynomial utils, action space, game
├── test_game_board.py            # Tests for BFS board builder and sampling
└── test_models.py                # Tests for GNN and policy-value network
```

---

## File Descriptions

### `src/config.py` — Configuration

A single `@dataclass Config` holding every hyperparameter. Derived properties compute values from the base parameters:

| Property | Formula | Description |
|----------|---------|-------------|
| `max_nodes` | $n_{\text{vars}} + 1 + C_{\max}$ | Input variables + constant node + max operations |
| `max_actions` | $N_{\max}(N_{\max} + 1)$ | Size of the flat action space |
| `effective_max_degree` | `max_degree` if set, else $C_{\max}$ | Max degree per variable in polynomial representation |
| `target_size` | $(d + 1)^{n}$ | Dimension of the flattened coefficient vector |

For defaults ($n=2$, $C_{\max}=6$, $d=6$): `max_nodes = 9`, `max_actions = 90`, `target_size = 49`.

**Factor library parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `factor_library_enabled` | `True` | Enable factor subgoal rewards and library accumulation |
| `factor_subgoal_reward` | `1.0` | Reward for building a non-trivial factor subgoal |
| `factor_library_bonus` | `0.5` | Extra reward when that factor was built in a previous episode |
| `completion_bonus` | `3.0` | Bonus when the circuit has both pieces for one final add/mul to reach T |

---

### `src/environment/fast_polynomial.py` — Fast Polynomial Backend

The primary polynomial arithmetic engine, replacing SymPy for all hot-path operations. Polynomials are stored as dense numpy `int64` arrays of shape $(d+1)^n$ where $d$ is `max_degree` and $n$ is the number of variables. Entry `coeffs[a0, a1, ...]` stores the coefficient of $x_0^{a_0} x_1^{a_1} \cdots$ modulo $p$.

**Performance:** ~87x faster than SymPy for BFS game board generation, ~4,600 environment steps/sec.

**Key operations:**

| Operation | Implementation | Complexity |
|-----------|---------------|------------|
| Addition | Element-wise `(A + B) % p` | $O((d+1)^n)$ |
| Multiplication | N-dimensional convolution + truncate + mod | $O((d+1)^{2n})$ |
| Equality | `np.array_equal` | $O((d+1)^n)$ |
| Canonical key | `coeffs.tobytes()` | $O((d+1)^n)$ |
| Term similarity | Vectorized comparison | $O((d+1)^n)$ |

For the default settings ($n=2$, $d=6$), each polynomial is a $7 \times 7 = 49$-entry array. Multiplication via 2D convolution is a tight loop over at most $49 \times 49 = 2401$ operations — trivial for numpy.

**Constructors:** `FastPoly.zero()`, `FastPoly.constant(v)`, `FastPoly.variable(i)` for building base polynomials.

---

### `src/environment/polynomial_utils.py` — SymPy Utilities and Conversions

SymPy-based polynomial functions retained for display, debugging, and conversion:

- **`mod_reduce(expr, syms, mod)`** — Expand and reduce coefficients mod $p$:

$$6x_0^2 + 3x_1 \;\xrightarrow{\bmod\; 5}\; x_0^2 + 3x_1$$

- **`sympy_to_fast(expr, syms, mod, max_degree)`** — Convert SymPy expression to `FastPoly`
- **`fast_to_sympy(poly, syms)`** — Convert `FastPoly` back to SymPy for display
- **`canonical_key`**, **`poly_equal`**, **`poly_to_coefficient_vector`**, **`term_similarity`** — SymPy versions kept for backward compatibility and testing

The **shaping potential** used for reward computation:

$$\phi(s) = \frac{|\{m : c_m^{\text{current}} = c_m^{\text{target}} \text{ and } c_m^{\text{target}} \ne 0\}|}{|\{m : c_m^{\text{target}} \ne 0\}|}$$

---

### `src/environment/action_space.py` — Action Encoding

Actions are triples $(\mathrm{op}, i, j)$ with $i \le j$, encoded as a single integer. Pairs $(i, j)$ use upper-triangular indexing; each pair gets 2 slots (add = even, multiply = odd).

**Encoding:**

$$\text{pair\_idx}(i, j) = i \cdot N_{\max} - \frac{i(i-1)}{2} + (j - i)$$

$$\text{action\_idx} = 2 \cdot \text{pair\_idx} + \mathrm{op}$$

**Decoding** uses a closed-form inverse via the triangular number formula. Given `pair_idx`:

$$i = \left\lfloor \frac{(2N_{\max} + 1) - \sqrt{(2N_{\max} + 1)^2 - 8 \cdot \text{pair\_idx}}}{2} \right\rfloor$$

$$j = i + (\text{pair\_idx} - \text{pair\_idx}(i, i))$$

This is $O(1)$ rather than the $O(N)$ linear search in the old codebase.

**`get_valid_actions_mask(num_current_nodes, max_nodes)`** — Returns a boolean tensor where action $(\mathrm{op}, i, j)$ is valid iff both $i < n_{\text{current}}$ and $j < n_{\text{current}}$.

---

### `src/environment/factor_library.py` — Factor Library

An in-memory cache that accumulates knowledge about sub-computations across episodes within a single training run. It enables three complementary mechanisms:

**Subgoal detection** — At the start of each episode, the target polynomial is factorized over $\mathbb{Z}$ (integers) using SymPy's `factor_list`. Non-trivial polynomial factors (degree $\geq 1$, not equal to any base input node) are stored as per-episode subgoals. Subgoals grow dynamically during stepping (see below).

**Reuse bonus** — When a subgoal is also present in the library (meaning it was built in a past successful episode), the agent earns an extra library bonus on top of the standard subgoal reward.

**Dynamic subgoal discovery** — At each step, when the newly built node $v$ is library-known, two additional operations run (both gated to avoid SymPy overhead on every step): (1) compute $T - v$ in FastPoly (fast) and factorize it, adding new additive subgoals; (2) attempt exact division $T / v$ using SymPy and factorize the quotient, adding new multiplicative subgoals. The residual $T - v$ and quotient $T/v$ themselves are also added as direct subgoals.

**Completion bonus** — At each step, two cheap checks determine whether the circuit is exactly one operation away from success:
- *Additive*: $T - v$ is already in the circuit → the agent could do $v + (T-v) = T$ next.
- *Multiplicative*: the exact quotient $T/v$ is already in the circuit, or is a scalar → the agent could do $v \times (T/v) = T$ next.

Each direction fires at most once per episode to prevent exploitation.

**Key design choices:**
- No disk I/O — the library lives entirely in RAM for one training session.
- Additive completion check is always $O(n_{\text{nodes}})$ (just FastPoly subtraction + key lookup). SymPy is only called when $v$ is library-known.
- Factorization is over $\mathbb{Z}$ (not $\mathbb{F}_p$) for stability. Exact division is verified by reducing the $\mathbb{Z}$-remainder mod $p$ (handles cases where the ZZ remainder is a non-zero multiple of $p$).
- Each subgoal key can only be rewarded once per episode (`_subgoals_hit` set).
- After each successful episode, all agent-built nodes are registered in the library.

**Example:** For target $3(x_0+1)^2$, `factor_list` returns `(3, [(x_0+1, 2)])`. Subgoal: $\{x_0+1\}$. When the agent later builds $v = x_0+1$ (library-known), SymPy is called on $T - v = x_0^2 + x_0$ and $T/v = 3(x_0+1) = 3x_0+3$, adding further subgoals. If the residual $x_0^2+x_0$ was already in the circuit, the additive completion bonus (+3.0) fires.

| Method | Description |
|--------|-------------|
| `factorize_target(target)` | Factorize target over Z, return non-trivial factors mod p (called at reset) |
| `factorize_poly(poly, exclude_keys)` | General factorization helper; `factorize_target` delegates here |
| `exact_quotient(dividend, divisor)` | SymPy exact division over Z; returns quotient FastPoly if exact mod p, else None |
| `is_base(poly)` | True if poly is one of the free initial nodes (x0, x1, ..., 1) |
| `filter_known(factors)` | Return canonical keys of factors already in the library |
| `register(poly, step_num)` | Add/update one polynomial in the library |
| `register_episode_nodes(nodes, n_initial)` | Bulk-register all agent-built nodes after success |
| `contains(poly)` | O(1) membership test |

---

### `src/environment/circuit_game.py` — Game Environment

Gymnasium-style environment for the circuit construction game. Uses `FastPoly` internally for all polynomial arithmetic.

**State representation:**
- `nodes`: list of `FastPoly` objects (one per circuit node)
- `node_types`: per-node feature vectors $[\text{is\_input},\; \text{is\_constant},\; \text{is\_op},\; \text{op\_type}]$
  - Input node: $(1, 0, 0, 0)$
  - Constant node $1$: $(0, 1, 0, 0)$
  - Add result: $(0, 0, 1, 0.5)$
  - Multiply result: $(0, 0, 1, 1.0)$
- `edges`: directed edges from operand nodes to result nodes (stored bidirectionally for GNN message passing)

**`reset(target_poly)`** — Initializes nodes to $[x_0, x_1, \ldots, x_{n-1}, 1]$ and returns the initial observation.

**`step(action_idx)`** — Decodes the action, computes $v_k = \texttt{mod\_reduce}(v_i \;\mathrm{op}\; v_j)$, appends the new node, checks for success, and returns `(obs, reward, done, info)`.

**`get_observation()`** — Returns a dict:
- `graph`: PyG `Data` object (or dict fallback) with node features and edge indices, padded to `max_nodes`
- `target`: flattened coefficient array of the target polynomial, normalized to $[0, 1]$ by dividing by $p$. Shape: $(d+1)^n$ where $d$ = `max_degree`
- `mask`: boolean valid-action mask

**`clone()`** — Deep copies the game state for use in MCTS tree search. Numpy coefficient arrays are explicitly copied since they're mutable (unlike SymPy expressions).

---

### `src/game_board/generator.py` — Game Board Generation

Builds a DAG of all reachable polynomials via BFS, used for sampling meaningful training targets.

**`build_game_board(config, complexity)`** — BFS construction:
- Layer 0: initial nodes $\{x_0, x_1, \ldots, x_{n-1}, 1\}$
- Layer $k+1$: all new polynomials obtainable by applying $+$ or $\times$ to any pair of polynomials from layers $\le k$
- Deduplicates via `canonical_key`; tracks multiple construction paths per polynomial

**`find_interesting_targets(board, min_paths=2)`** — Returns polynomials reachable via $\ge$ `min_paths` distinct circuits, sorted by complexity then path count. These are interesting because the agent must discover non-trivial constructions.

**`sample_target(config, complexity, board=None)`** — Samples a target polynomial at a given complexity level from the board.

**`generate_random_circuit(config, complexity)`** — Generates a random circuit by performing `complexity` random operations. Returns `(polynomial, action_sequence)`.

---

### `src/models/gnn_encoder.py` — Graph Neural Network

A 3-layer Graph Convolutional Network (GCN) with residual connections and LayerNorm.

**Architecture:**

$$h^{(0)} = W_{\text{in}} \cdot x$$

For each layer $\ell = 1, \ldots, L$:

$$h^{(\ell)} = h^{(\ell-1)} + \text{ReLU}\!\left(\text{LayerNorm}\!\left(\text{GCN}^{(\ell)}(h^{(\ell-1)}, A)\right)\right)$$

$$z = W_{\text{out}} \cdot h^{(L)}$$

Graph-level embedding via global mean pooling over actual (non-padding) nodes:

$$\mathbf{g} = \frac{1}{|\mathcal{V}|}\sum_{v \in \mathcal{V}} z_v$$

Uses PyTorch Geometric `GCNConv` when available, otherwise falls back to a simple neighbor-aggregation message passing.

---

### `src/models/policy_value_net.py` — Policy-Value Network

Shared network used by both PPO and AlphaZero.

**Architecture:**

$$\mathbf{g} = \text{GNN}(\text{graph})$$

$$\mathbf{t} = \text{MLP}_{\text{target}}(\text{coeff\_vector})$$

$$\mathbf{f} = \text{MLP}_{\text{fusion}}([\mathbf{g} \;\|\; \mathbf{t}])$$

$$\boldsymbol{\pi} = \text{MLP}_{\text{policy}}(\mathbf{f}) \in \mathbb{R}^{|\mathcal{A}|}$$

$$v = \tanh(\text{MLP}_{\text{value}}(\mathbf{f})) \in [-1, 1]$$

Invalid actions are masked by setting their logits to $-\infty$ before softmax.

The target encoder is a 2-layer MLP that embeds the coefficient vector. Fusion concatenates the graph embedding and target embedding, then projects through a 2-layer MLP. This is simpler than the 6-layer transformer decoder in the old codebase — concatenation + linear fusion suffices for combining two embeddings.

**Key methods:**
- `forward(obs)` — Returns `(masked_logits, value)`
- `get_action_and_value(obs)` — Samples action, returns `(action, log_prob, entropy, value)` (used by PPO)
- `get_policy_and_value(obs)` — Returns `(softmax_probs, value)` (used by MCTS)

---

### `src/algorithms/ppo.py` — PPO Training

---

### `src/algorithms/mcts.py` — Neural MCTS

---

### `src/algorithms/alphazero.py` — AlphaZero Training

See the [Algorithms](#algorithms) section below for full mathematical descriptions.

---

### `src/evaluation/evaluate.py` — Evaluation Harness

Evaluates a trained model across multiple complexity levels.

- For **PPO**: uses greedy action selection (argmax over logits)
- For **AlphaZero**: runs full MCTS search with temperature 0 (deterministic)

Reports per-complexity success rate, average steps to solution, and overall aggregates.

---

### `src/main.py` — Entry Point

CLI for training and evaluation. Handles argument parsing, config overrides, device auto-detection (CUDA/MPS/CPU), seed setting, checkpoint save/load, and dispatches to the appropriate trainer.

---

## Algorithms

### Environment and Reward

Each episode is a Markov Decision Process:
- **State** $s_t$: the current circuit graph + target polynomial
- **Action** $a_t = (\mathrm{op}, i, j)$: apply operation to nodes $i, j$
- **Transition**: deterministic — add node $v_k = \texttt{mod\_reduce}(v_i \;\mathrm{op}\; v_j)$ to the circuit
- **Termination**: success (some node equals target), max steps reached, or max nodes reached

**Reward function:**

$$r_t = r_{\text{step}} + r_{\text{success}} + r_{\text{shaping}} + r_{\text{factor}} + r_{\text{library}} + r_{\text{complete}}$$

where:
- $r_{\text{step}} = -0.1$ (per-step penalty to encourage short circuits)
- $r_{\text{success}} = +10.0$ if the target is matched, else $0$
- $r_{\text{shaping}} = \gamma \cdot \phi(s_{t+1}) - \phi(s_t)$ (potential-based shaping, when enabled)
- $r_{\text{factor}} \in \{0, r_f\}$: bonus for building a non-trivial subgoal (default $r_f = 1.0$); fired at most once per subgoal per episode
- $r_{\text{library}} \in \{0, r_b\}$: additional bonus if that subgoal was also previously seen in the library (default $r_b = 0.5$)
- $r_{\text{complete}} \in \{0, r_c\}$: completion bonus when the circuit is exactly one operation from $T$ (default $r_c = 3.0$); additive and multiplicative directions each fire at most once per episode

The shaping potential is:

$$\phi(s) = \max_{v \in \text{nodes}(s)} \text{term\_similarity}(v, f^*)$$

By the **potential-based shaping theorem** (Ng et al., 1999), $r_{\text{shaping}}$ preserves the optimal policy while providing denser learning signal.

The completion bonus fires when the newly built node $v$ creates a "one-step gap" to success:
- **Additive:** $T - v$ already exists in the circuit (one ADD away from $T$)
- **Multiplicative:** $T / v$ divides exactly and the quotient is in the circuit or is a scalar (one MUL away from $T$)

---

### Factor Library and Subgoal Rewards

**File:** `src/environment/factor_library.py`

#### Motivation

Many target polynomials share common factors. For example, both $(x_0+1)^2$ and $(x_0+1)^3$ require building $x_0+1$ as an intermediate node. Once the agent has learned to construct $x_0+1$ efficiently, it should be guided to reuse that sub-circuit whenever $x_0+1$ appears as a factor of a new target.

#### Mechanism

**At episode start (`reset`):**
1. The target polynomial is factorized over $\mathbb{Z}$ using SymPy's `factor_list`.
2. Non-trivial polynomial factors (total degree $\geq 1$, not a base input node) are collected as the initial **subgoal set** $\mathcal{F}$ for the episode.
3. Subgoals already present in the factor library (from previous successful episodes) are marked as **library-known**.

**During the episode (`step`):**
- When the agent constructs node $v$ with $v \in \mathcal{F}$ and $v$ not yet rewarded this episode:

$$r_{\text{factor}} = r_f \quad (\text{default } +1.0)$$

- If $v$ is also library-known:

$$r_{\text{library}} = r_b \quad (\text{default } +0.5, \text{ stacks on } r_f)$$

- **Additive completion** (always checked): if $T - v$ is already in the circuit:

$$r_{\text{complete}} = r_c \quad (\text{default } +3.0, \text{ fires once per episode})$$

- If $v$ is **library-known**, dynamic discovery runs (SymPy gated):
  1. Factorize $T - v$ over $\mathbb{Z}$ → new additive subgoals extend $\mathcal{F}$.
  2. Compute exact quotient $T / v$ over $\mathbb{Z}$ → if exact, add quotient and its factors to $\mathcal{F}$ as multiplicative subgoals.
  3. **Multiplicative completion**: if $T/v$ is a scalar or is already in the circuit → $r_{\text{complete}} = r_c$ (separate from additive, fires once).

**At episode end (on success):**
- All agent-constructed nodes $v_{n_{\text{init}}}, \ldots, v_{N-1}$ are registered in the factor library, available as subgoals in future episodes.

#### Example

Target: $(x_0+1)^2$

1. `factor_list` returns $(1,\, [(x_0+1,\, 2)])$. Initial subgoal: $\{x_0+1\}$.
2. Episode 1: agent builds $x_0+1$ → $r_f = +1.0$. Library now contains $\{x_0+1\}$.
3. Next step: $x_0+1$ is library-known, so dynamic discovery runs: $T - (x_0+1) = x_0^2 + x_0 + 1 - x_0 - 1 = x_0^2 + x_0$. This gets factorized to produce additional subgoals, and $T/(x_0+1) = x_0+1$ is confirmed exact. If $x_0+1$ is already in the circuit (which it is — just built), multiplicative completion fires: $r_c = +3.0$.
4. Episode 2 (same target): builds $x_0+1$ → $r_f + r_b = +1.5$ (library hit).

#### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `factor_library_enabled` | `True` | Enable/disable the entire feature |
| `factor_subgoal_reward` | `1.0` | Bonus for constructing any subgoal |
| `factor_library_bonus` | `0.5` | Extra bonus when that subgoal was previously seen |
| `completion_bonus` | `3.0` | Bonus when both pieces for a final add/mul are in the circuit |

CLI overrides: `--no-factor-library`, `--factor-subgoal-reward`, `--factor-library-bonus`, `--completion-bonus`.

---

### PPO (Proximal Policy Optimization)

**File:** `src/algorithms/ppo.py`

PPO collects rollout trajectories, computes advantages via GAE, and updates the policy with a clipped surrogate objective.

#### Generalized Advantage Estimation (GAE)

For a trajectory of length $T$:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{k=0}^{T-t-1} (\gamma \lambda)^k \delta_{t+k}$$

Computed efficiently via backward recursion:

$$\hat{A}_T = 0, \qquad \hat{A}_t = \delta_t + \gamma \lambda \cdot \hat{A}_{t+1}$$

Returns are $\hat{R}_t = \hat{A}_t + V(s_t)$.

Defaults: $\gamma = 0.99$, $\lambda = 0.95$.

#### PPO Clipped Objective

Let $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ be the probability ratio. The clipped surrogate loss is:

$$L^{\text{CLIP}}(\theta) = -\mathbb{E}_t\left[\min\!\left(r_t(\theta)\,\hat{A}_t,\;\; \text{clip}(r_t(\theta),\; 1-\epsilon,\; 1+\epsilon)\,\hat{A}_t\right)\right]$$

with $\epsilon = 0.2$.

#### Value and Entropy Losses

$$L^{\text{VF}}(\theta) = \mathbb{E}_t\left[\bigl(V_\theta(s_t) - \hat{R}_t\bigr)^2\right]$$

$$H[\pi_\theta] = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$$

#### Total PPO Loss

$$L(\theta) = L^{\text{CLIP}} + c_v \cdot L^{\text{VF}} - c_e \cdot H[\pi_\theta]$$

Defaults: $c_v = 0.5$, $c_e = 0.01$, gradient clipping at $\|\nabla\|_{\max} = 0.5$.

#### Training Loop

```
for each iteration:
    1. Collect 2048 steps of rollout data using current policy
    2. Compute GAE advantages and returns
    3. Normalize advantages: A_hat = (A - mean) / (std + 1e-8)
    4. For 4 epochs over mini-batches of size 64:
        a. Recompute log_probs, entropy, values
        b. Compute PPO loss
        c. Backprop with gradient clipping
    5. Adjust curriculum complexity
```

---

### AlphaZero (MCTS + Neural Network)

**Files:** `src/algorithms/mcts.py` and `src/algorithms/alphazero.py`

#### Neural MCTS

Each node in the search tree stores:
- $N(s, a)$: visit count
- $W(s, a)$: total value
- $Q(s, a) = W(s,a) / N(s,a)$: mean value
- $P(s, a)$: prior probability from the neural network

**Selection** uses the PUCT formula (Polynomial Upper Confidence Trees):

$$a^* = \arg\max_a \left[ Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]$$

where $c_{\text{puct}} = 1.4$ balances exploitation ($Q$) vs. exploration (prior-weighted).

**Expansion:** When a leaf node is reached, the neural network evaluates the state:

$$\mathbf{P}(s, \cdot),\; v = f_\theta(s)$$

Child nodes are created for all valid actions with their prior probabilities.

**Evaluation:** At terminal states, the value is the actual game outcome ($+1$ for success, $-1$ for failure). At non-terminal leaves, the neural network's value estimate $v$ is used. **No random rollouts** — this is the core AlphaZero insight.

**Backpropagation:** For every node along the selection path:

$$N(s, a) \leftarrow N(s, a) + 1$$

$$W(s, a) \leftarrow W(s, a) + v$$

**Action selection** from the root after all simulations:

$$\pi(a | s) \propto N(s_{\text{root}}, a)^{1/\tau}$$

where $\tau$ is a temperature parameter that decays from $1.0$ (exploratory) to $0.1$ (exploitative) over the course of each game.

Default: 100 simulations per move.

#### AlphaZero Loss Function

Training data consists of tuples $(s, \boldsymbol{\pi}, z)$ where:
- $s$ is a game state
- $\boldsymbol{\pi}$ is the MCTS visit-count policy
- $z \in \{+1, -1\}$ is the game outcome

The loss combines policy and value:

$$\mathcal{L}(\theta) = \underbrace{(v_\theta - z)^2}_{\text{value loss}} - \underbrace{\boldsymbol{\pi}^\top \log \mathbf{p}_\theta}_{\text{policy loss}}$$

where $\mathbf{p}_\theta = \text{softmax}(f_\theta^{\text{policy}}(s))$ and $v_\theta = f_\theta^{\text{value}}(s)$.

This trains the network to predict both the MCTS-improved policy and the game outcome simultaneously.

#### Self-Play Training Loop

```
for each iteration:
    1. Self-play phase (model.eval):
        - Play 100 games using MCTS
        - Store (state, MCTS_policy, outcome) tuples in replay buffer
    2. Training phase (model.train):
        - Sample mini-batches from replay buffer (capacity 50,000)
        - Minimize AlphaZero loss for 10 epochs
    3. Adjust curriculum complexity
```

---

### Curriculum Learning

Both algorithms use adaptive curriculum learning to gradually increase problem difficulty.

The agent starts at complexity $C = 2$ (2 operations). A sliding window tracks the recent success rate $\rho$:

$$C \leftarrow \begin{cases} C + 1 & \text{if } \rho \ge 0.7 \text{ and } C < C_{\max} \\[4pt] C - 1 & \text{if } \rho \le 0.4 \text{ and } C > C_{\min} \\[4pt] C & \text{otherwise} \end{cases}$$

The window size is 50 episodes for PPO and 100 games for AlphaZero. History is cleared after each transition to allow the agent to adapt.

---

## Neural Network Architecture

```
Input: game observation
├── Circuit Graph (PyG Data)
│   └── CircuitGNN (3-layer GCN + residual + LayerNorm)
│       └── Global mean pool → graph_embedding (128-dim)
│
├── Target Polynomial (flattened coefficient array, 49-dim for n=2, d=6)
│   └── MLP (49 → 128 → 128) → target_embedding (128-dim)
│
└── Fusion
    └── Concat [graph_emb ∥ target_emb] (256-dim)
        └── MLP (256 → 128 → 128) → fused (128-dim)
            ├── Policy Head: MLP (128 → 128 → 90) → logits + mask
            └── Value Head:  MLP (128 → 128 → 1) → tanh → value
```

Total parameters: ~173K (with default `hidden_dim=128`).

---

## Installation

**Requirements:** Python 3.9+, PyTorch, SymPy, NumPy.

```bash
pip install torch sympy numpy pytest
```

**Optional** (for GCN via PyTorch Geometric):

```bash
pip install torch-geometric
```

If PyTorch Geometric is not installed, the GNN encoder falls back to a simple neighbor-aggregation message passing scheme.

---

## Usage

All commands are run from the project root directory.

### Training with PPO

```bash
python -m src.main --algorithm ppo --iterations 100
```

### Training with SAC

```bash
python -m src.main --algorithm sac --iterations 100
```

Optional SAC stabilizers:

```bash
# Enable CQL-lite regularization and BC warm start from board demonstrations
python -m src.main --algorithm sac --iterations 100 \
    --sac-use-cql --sac-cql-alpha 0.1 --sac-bc-warmstart
```

### Training with AlphaZero

```bash
python -m src.main --algorithm alphazero --iterations 50
```

### Training with SAC

```bash
python -m src.main --algorithm sac --iterations 100
```

### Run SAC with `nohup` (old workflow style)

```bash
./run_sac_nohup.sh tests/sac_training.log 10000
tail -f tests/sac_training.log
```

### Configuration Overrides

```bash
# 3 variables, mod 7, up to 8 operations, on GPU
python -m src.main --algorithm ppo --iterations 200 \
    --n-variables 3 --mod 7 --max-complexity 8 --device cuda

# Disable curriculum learning
python -m src.main --algorithm alphazero --iterations 100 --no-curriculum

# Custom hidden dimension and seed
python -m src.main --algorithm ppo --hidden-dim 256 --seed 123
```

### Factor Library Options

The factor library is enabled by default. To adjust or disable it:

```bash
# Disable factor library entirely (no subgoal rewards, no completion bonus)
python -m src.main --algorithm ppo --iterations 100 --no-factor-library

# Tune reward magnitudes
python -m src.main --algorithm sac --iterations 200 \
    --factor-subgoal-reward 2.0 --factor-library-bonus 1.0 --completion-bonus 5.0
```

### Evaluation Only

```bash
python -m src.main --eval-only --checkpoint checkpoint.pt --algorithm ppo
python -m src.main --eval-only --checkpoint checkpoint.pt --algorithm sac
python -m src.main --eval-only --checkpoint checkpoint.pt --algorithm alphazero
python -m src.main --eval-only --checkpoint checkpoint.pt --algorithm sac
```

### Saving and Loading Checkpoints

```bash
# Save to custom path
python -m src.main --algorithm ppo --iterations 100 --save-path models/ppo_v1.pt

# Load and continue or evaluate
python -m src.main --algorithm ppo --checkpoint models/ppo_v1.pt --eval-only
```

---

## Testing

Run the full test suite (90 tests):

```bash
python -m pytest tests/ -v
```

Run individual test files:

```bash
# Fast polynomial backend tests
python -m pytest tests/test_fast_polynomial.py -v

# Environment tests: polynomial math, action encoding, game flow
python -m pytest tests/test_environment.py -v

# Game board tests: BFS construction, target sampling
python -m pytest tests/test_game_board.py -v

# Model tests: GNN, policy-value network
python -m pytest tests/test_models.py -v
```

### What the Tests Verify

**`test_fast_polynomial.py`** (28 tests):
- Construction: zero, constant, variable polynomials
- Arithmetic: addition, multiplication with mod reduction
- Degree truncation on multiply
- Equality, hashing, canonical key consistency
- Term similarity computation
- Coefficient vector (to_vector) shape and values
- Copy independence (mutating copy doesn't affect original)
- 3-variable generalization (n-dimensional arrays)

**`test_environment.py`** (42 tests):
- SymPy polynomial mod reduction correctness (e.g., $6x_0 \bmod 5 = x_0$)
- SymPy canonical key deduplication (e.g., $x_0 + x_1 = x_1 + x_0$)
- SymPy polynomial equality under mod arithmetic
- Monomial enumeration and coefficient vector construction
- Term similarity computation
- SymPy $\leftrightarrow$ FastPoly round-trip conversion correctness
- Action encode/decode roundtrips for all valid $(op, i, j)$
- No duplicate action indices across the full action space
- Valid action masking with varying node counts
- Game reset, single-step, and multi-step episode flow
- Success detection and reward computation
- Game state cloning for MCTS
- Constant node (value 1) usage

**`test_game_board.py`** (12 tests):
- BFS produces correct base nodes at layer 0
- Polynomials appear at correct complexity layers
- No duplicate canonical keys in the board
- Known polynomials (e.g., $x_0 + x_1$) are reachable
- Board size grows monotonically with complexity
- Interesting targets have multiple construction paths
- Target sampling returns valid polynomials
- Random circuit generation produces valid action sequences

**`test_models.py`** (8 tests):
- GNN output shape correctness
- GNN handles empty edge sets
- Gradient flow through GNN
- Policy-value network forward pass shape
- Invalid actions receive $-\infty$ logits
- Sampled actions are always valid (masked)
- Policy probabilities sum to 1
- End-to-end gradient flow through the full network
