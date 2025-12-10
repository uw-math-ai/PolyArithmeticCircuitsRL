# Polynomial Arithmetic Circuits Reinforcement Learning

## Overview

This project implements a combined **Supervised Learning + PPO Reinforcement Learning** pipeline for discovering optimal arithmetic circuits that generate target polynomial expressions. The system uses a hybrid neural network architecture combining **Graph Neural Networks (GNNs)** and **Transformer Decoders**, integrated with **Monte Carlo Tree Search (MCTS)** for lookahead planning during policy training.

The core task is: given a target polynomial, find the minimal sequence of addition and multiplication operations on input variables to construct that polynomial.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Encoding System](#encoding-system)
3. [Pipeline Phases](#pipeline-phases)
4. [MCTS Integration](#mcts-integration)
5. [Implementation Details](#implementation-details)
6. [Configuration](#configuration)
7. [Training](#training)

---

## Architecture Overview

### Problem Formulation

**State Space**: An arithmetic circuit is represented as a directed acyclic graph (DAG) where:
- **Nodes** represent intermediate polynomial expressions
- **Edges** represent data dependencies between operations
- The circuit maintains a list of polynomial expressions, starting with input variables and the constant 1

**Action Space**: At each step, the agent can:
- Select two existing polynomial nodes (i, j) where i ≤ j
- Choose an operation: **add** (p_i + p_j) or **multiply** (p_i × p_j)
- This produces a new polynomial node

**Reward Signal**:
- **+100** (success_reward) for constructing the target polynomial
- **-1** (step_penalty) for each unsuccessful step
- Similarity bonus based on term-wise match with target

---

## Encoding System

### Compact One-Hot Graph Encoder

The circuit state is encoded into a fixed-size vector representation optimized for neural network consumption. The encoding consists of three components:

#### 1. **Operation Type Encoding** (num_op_nodes = 2N)
- For N maximum operations allowed, maintain a binary vector of size 2N
- Each operation gets 2 slots: one for addition, one for multiplication
- At position 2k + op_type, set 1 if operation k is of type op_type (0=add, 1=multiply)

```python
# Example: 3 operations performed, last two are (add, multiply, add)
operation_type = [0,1, 1,0, 1,0]  # positions: (add), (mul), (add)
```

#### 2. **Edge Encoding** (num_edge_nodes = N × (2(D+P-1) + N-1))
- Encodes the connectivity between nodes
- For each operation k:
  - Can connect to any of the first D+P-1+k possible nodes (D variables + P constant terms + k previous ops)
  - Two edges per operation (source1, source2)
  - Stored as binary: 1 at position indicating which nodes are connected

```python
# For operation k connecting nodes i and j:
# base = k * (I + I + k - 1)  # I = D + P - 1
# edges[base + i] = 1       # first edge
# edges[base + num_ids_possible + j] = 1  # second edge
```

#### 3. **Node ID Tracking** (num_id_nodes = N)
- Tracks which is the "current" node (last generated)
- One-hot encoding: position k is 1 if node k is the latest output

**Why This Encoding?**
- Fixed size regardless of circuit complexity (up to N operations)
- Captures both structure (edges) and semantics (operation types)
- Efficiently represents the causal ordering of operations
- Amenable to gradient-based learning

---

## Pipeline Phases

### Phase 1: Supervised Learning

The model first learns from a curated dataset of interesting polynomials and their optimal construction sequences.

**Data Generation**:
1. Load pre-computed polynomials from `Game-Board-Generation/pre-training-data/`
   - Each polynomial has been analyzed for optimal circuit depth
   - Multiple paths may exist; the system learns from all valid sequences
2. Convert each circuit construction into training pairs: (partial_circuit, next_action)
   - For a circuit with actions [a₀, a₁, ..., aₙ], generate pairs:
     - (state[a₀], action=a₁)
     - (state[a₀, a₁], action=a₂)
     - ... up to (state[a₀...aₙ₋₁], action=aₙ)

**Model Training**:
```python
# Supervised objective combines:
Loss = CrossEntropy(action_logits, target_action) + λ_v × MSE(value_pred, trajectory_return)
```

Where:
- **Action loss**: Predicts the next action in the construction sequence
- **Value loss**: Estimates expected remaining steps to completion
- **Cosine annealing scheduler**: Learning rate decays over training

**Convergence**: Train until test accuracy plateaus, typically at 70-85% action prediction accuracy.

**Output**: `best_supervised_model_n{N}_C{C}.pt` - pretrained weights for the neural network

---

### Phase 2: Reinforcement Learning with PPO

Once the model has learned behavioral patterns, PPO fine-tunes it to handle novel polynomials and improve beyond supervised accuracy.

#### Key Components:

**Generalized Advantage Estimation (GAE)**:
```python
advantages = []
gae = 0.0
for t in reversed(range(T)):
    delta = reward[t] + γ × V(s[t+1]) × (1 - done[t]) - V(s[t])
    gae = delta + γ × λ × (1 - done[t]) × gae
    advantages[t] = gae
returns = advantages + values
```

This provides lower-variance advantage estimates than raw discounted returns.

**PPO Objective**:
```python
L_clip = E[min(r_t × Ā_t, clip(r_t, 1-ε, 1+ε) × Ā_t)]
L_value = MSE(V(s), G_t)
L_entropy = -H(π(s))

Loss = -L_clip + c_v × L_value - c_ent × L_entropy
```

Where:
- **r_t** = exp(log π_new(a|s) - log π_old(a|s)) (importance sampling ratio)
- **Clipping** with ε=0.2 prevents catastrophically large policy updates
- **Value loss** keeps value predictions accurate
- **Entropy bonus** maintains exploration throughout training

**Curriculum Learning**:
- Start with complexity 1 (single operations)
- Gradually increase to harder complexity levels
- Complexity advances when success rate > 60% over 200 recent games
- Keeps agent learning tasks at appropriate difficulty

---

## MCTS Integration

MCTS provides **lookahead planning** during policy data collection, helping the policy learn better trajectories.

### Architecture

**CircuitGameState**: Lightweight, fully cloneable state representation
```python
state.clone()              # Deep copy for tree search
state.available_actions()  # List all valid next actions
state.apply_action(a)      # Deterministic state transition
state.is_terminal()        # Check if goal reached or steps exhausted
state.evaluate_reward()    # Reward heuristic (similarity + success bonus)
```

**MCTSNode**: Tree node with UCB statistics
```python
class MCTSNode:
    parent              # Pointer to parent
    state               # Cloned CircuitGameState
    children            # Dict[action -> child_node]
    visit_count         # Number of visits
    total_value         # Cumulative return
    untried_actions     # Actions not yet explored
```

### Search Algorithm

Four phases per iteration:

1. **Selection**: Starting from root, follow UCB formula to reach leaf
   ```
   UCB(child) = Q(child)/N(child) + C × √(ln(N(parent))/N(child))
   ```
   - **Exploitation**: High average return nodes
   - **Exploration**: Under-visited nodes (C=1.4 by default)

2. **Expansion**: Add one child to the tree
   - Pick untried action from selected node
   - Create new child with cloned state

3. **Rollout**: Simulate random play from expanded node
   - Each step: randomly select action from available set
   - Continue until terminal state (success or max_steps reached)
   - Return final reward heuristic

4. **Backpropagation**: Update all ancestors
   ```python
   while node:
       node.visit_count += 1
       node.total_value += rollout_reward
       node = node.parent
   ```

### Integration with PPO

During PPO rollout collection:

```python
planner = MCTSPlanner(config) if config.use_mcts else None

while not game.is_done():
    state = game.observe()
    
    # With probability mcts_policy_mix, ask MCTS for action
    planner_action = None
    if planner and random.random() < config.mcts_policy_mix:
        planner_action = planner.select_action(game)
    
    # Let policy predict, optionally constrained by MCTS
    action, log_prob, entropy, value = model.get_action_and_value(
        state, 
        action_idx=planner_action
    )
    
    game.take_action(action)
    # ... collect trajectory ...
```

**Configuration Parameters**:
- `use_mcts`: Enable/disable MCTS guidance
- `mcts_simulations`: Number of tree search iterations (e.g., 64)
- `mcts_exploration`: UCB exploration constant (e.g., 1.4)
- `mcts_policy_mix`: Probability of using MCTS suggestion (0.35 = 35%)

**Benefits**:
- MCTS sees further ahead than single-step policy
- Provides better action suggestions for critical decision points
- Policy learns to match MCTS's lookahead judgments
- Improves success rates by 10-20% with modest computational cost

---

## Implementation Details

### Neural Network Architecture

**CircuitBuilder** (Main Model):

```
GNN Branch:
    Input: Graph representation of current circuit
    ├─ 4-dim node features (type encoding [3] + scalar value [1])
    ├─ GCN layers (3 layers, 256→256→256→embedding_dim)
    ├─ Residual connections + LayerNorm
    └─ Global mean pooling → graph embedding (embedding_dim,)

Transformer Branch:
    Input: Target polynomial encoding + graph embedding
    ├─ Polynomial Linear (state_size → embedding_dim)
    ├─ Stack both as memory (2, batch_size, embedding_dim)
    ├─ Transformer Decoder (6 layers, 4 heads, 256 FFN dim)
    └─ Query token learned parameter
    
Output Heads:
    ├─ Action head: embedding_dim → max_actions (policy)
    └─ Value head: embedding_dim → 1 (value estimate)
```

**Why This Design?**
- **GNN** captures circuit structure and node relationships
- **Transformer** attends to target polynomial requirements
- **Global pooling** aggregates node-level info to graph level
- **Value head** provides bootstrap for advantage estimation
- Handles variable-length graphs (PAD graphs together in batch)

### Graph Representation

Each circuit state becomes a `torch_geometric.Data` object:

```python
Data(
    x=[num_nodes, 4],           # Node features
    edge_index=[2, num_edges],  # Edge indices (undirected + self-loops)
    batch=[num_nodes]           # Batch assignment (for pooling)
)
```

**Node Features** (4-dim):
- **[1,0,0]** + position → Input variable node
- **[0,1,0]** + 1.0 → Constant node
- **[0,0,1]** + op_type → Operation node (0=add, 1=multiply)

**Edge Structure**:
- Directed edges from operands to operation results
- Self-loops on all nodes (standard GCN practice)
- No edge features (uniform operations)

### Action Space Encoding

Actions are encoded into single integers for efficient sampling:

```python
# Given max_nodes possible nodes and pair (i,j) with i ≤ j:
pair_index = i × max_nodes - i(i-1)/2 + (j - i)
operation_index = 0 if add, 1 if multiply
action_id = pair_index × 2 + operation_index

# Example: max_nodes=10, add(3,7)
pair_index = 3×10 - 3×2/2 + (7-3) = 30 - 3 + 4 = 31
action_id = 31 × 2 + 0 = 62
```

This bijection enables:
- Masking invalid actions (operating on non-existent nodes)
- Efficient categorical sampling
- Dense action space for neural network output

---

## Configuration

Edit `src/PPO RL/PPO.py` Config class:

```python
class Config:
    # Problem
    n_variables = 3              # Number of input variables
    max_complexity = 1           # Max arithmetic operations
    max_degree = 2               # Max polynomial degree
    
    # Model architecture
    hidden_dim = 256             # GNN/Transformer hidden size
    embedding_dim = 256          # Final embedding dimension
    num_gnn_layers = 3           # GCN layers
    num_transformer_layers = 6   # Transformer decoder layers
    transformer_heads = 4        # Multi-head attention heads
    
    # Supervised learning
    learning_rate = 0.0003
    batch_size = 128
    epochs = 20
    
    # PPO hyperparameters
    rl_learning_rate = 1e-5
    ppo_iterations = 2000        # Outer training loops
    steps_per_batch = 4096       # Transitions collected per iteration
    ppo_epochs = 10              # Inner SGD epochs per batch
    ppo_minibatch_size = 128
    ppo_clip = 0.2               # Clipping parameter ε
    gamma = 0.99                 # Discount factor
    lambda_gae = 0.95            # GAE coefficient
    vf_coef = 0.5                # Value loss weight
    ent_coef = 0.02              # Entropy bonus weight
    
    # MCTS guidance
    use_mcts = True
    mcts_simulations = 64        # Tree search iterations
    mcts_exploration = 1.4       # UCB constant
    mcts_policy_mix = 0.35       # Probability of MCTS action (0-1)
    
    # Curriculum learning
    complexity_threshold = 0.6   # Success rate to advance
    complexity_window = 200      # Rolling window for success rate
    
    # Data
    use_interesting_polynomials = True
    interesting_data_dir = "Game-Board-Generation/pre-training-data"
    interesting_prefix = "game_board_C1"
    max_interesting_samples = 5000
```

---

## Training

### Running the Full Pipeline

```bash
cd src/PPO\ RL/
python PPO.py
```

**What happens:**

1. **Supervised Phase** (~30-60 min, single GPU)
   - Loads interesting polynomial data from pre-training files
   - Generates 10k training + 2k test examples
   - Trains 20 epochs with cosine annealing
   - Checkpoints best model to `best_supervised_model_n3_C1.pt`

2. **PPO Phase** (~2-8 hours, single GPU)
   - Initializes from supervised checkpoint
   - For 2000 iterations:
     - Collect 4096 environment steps with current policy
     - Optional MCTS lookahead (35% of decisions)
     - Compute GAE advantages
     - Run 10 PPO epochs with minibatch SGD
     - Advance curriculum when success > 60%
   - Saves final model to `ppo_model_n3_C1_curriculum.pt`

### Key Metrics During Training

**Supervised Phase Output:**
```
Epoch 1: LR: 0.000300, Train Acc: 45.32%, Loss: 1.2345, VLoss: 0.5678
...
Epoch 20: LR: 0.000003, Train Acc: 82.15%, Loss: 0.3456, VLoss: 0.1234
*** New Best Test Accuracy (81.23%)! Model saved to best_supervised_model_n3_C1.pt ***
```

**PPO Phase Output:**
```
PPO Iter 1 (C=1) - Collecting: [████████] 4096/4096
Success rate: 12/50 games, Mean reward: 23.45
Policy loss: 0.234, Value loss: 0.567, Entropy: 1.234

PPO Iter 2 (C=2) - Collecting: [████████] 4096/4096
Success rate: 28/50 games, Mean reward: 45.67
...

PPO Iter 2000 (C=5) - Collecting: [████████] 4096/4096
Success rate: 89/100 games, Mean reward: 89.12
```

### Checkpointing

Models are saved at:
- `best_supervised_model_n{n_variables}_C{max_complexity}.pt` — Supervised pretrain
- `ppo_model_n{n_variables}_C{max_complexity}_curriculum.pt` — PPO final

To resume from checkpoint:
```python
model = CircuitBuilder(config, config.compact_size)
model.load_state_dict(torch.load("best_supervised_model_n3_C1.pt"))
# Continue with PPO or evaluation
```

---

## Experimental Features

### Interesting Polynomial Sampling

Instead of random polynomial generation, the system can use curated "interesting" polynomials:

```python
# Load from pre-computed data
interesting_entries = load_interesting_circuit_data(config)

# Properties
entry["expr"]              # SymPy expression
entry["actions"]           # Tuple of (op, node_i, node_j) steps
entry["encoding"]          # Compact one-hot vector
entry["shortest_length"]   # Minimum operations needed
```

This enables:
- **Multi-path learning**: Polynomials with multiple optimal constructions
- **Curriculum pacing**: Sort by complexity for easier scheduling
- **Realistic targets**: Real polynomials from polynomial games/graphs

### Temperature-Scaled Action Sampling

```python
logits = action_logits / temperature

# temperature > 1: More uniform (higher exploration)
# temperature = 1: Uniform scaling (standard)
# temperature < 1: Sharper peaks (lower exploration)

dist = Categorical(logits=logits)
action = dist.sample()
```

Allows trading off exploitation vs. exploration during data collection.

---

## Summary

This pipeline combines three complementary techniques:

1. **Supervised Pretraining** — Fast initialization from curated data
2. **PPO Reinforcement Learning** — Adaptation to novel polynomials, curriculum learning
3. **MCTS Planning** — Lookahead to improve policy quality

The **encoding system** efficiently represents circuit structure and enables gradient flow. The **architecture** leverages GNNs for graph understanding and Transformers for goal-conditioned reasoning.

The result is a learnable, generalizable system for discovering arithmetic circuits—a fundamental problem in computational mathematics and optimal algorithm synthesis.

---

## References

- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **GAE**: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
- **MCTS**: Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012)
- **GNN**: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (2017)
- **Transformers**: Vaswani et al., "Attention Is All You Need" (2017)
