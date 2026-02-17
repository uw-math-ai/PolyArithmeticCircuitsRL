# SAC RL: End-to-end walkthrough (from data generation to training loop)

This document explains how `SAC.py` works, starting from data generation and ending at program exit. It is written directly from the current code in [src/SAC RL/SAC.py](../src/SAC%20RL/SAC.py).

## 1) High-level overview

`SAC.py` implements a discrete Soft Actor-Critic (SAC) agent that builds arithmetic circuits to match a target polynomial. Each episode is a circuit-construction game. The agent chooses actions that add a node (either add or multiply two existing nodes). The code optionally mixes in MCTS guidance and supports curriculum learning over circuit complexity.

Core phases:

1. **Configuration**: sets hyperparameters for SAC, curriculum, and data generation.
2. **Data generation**: builds target polynomials (random, mixed patterns, or from a dataset) and optionally pre-fills the replay buffer with synthetic transitions.
3. **Model**: GNN + Transformer decoder to produce policy logits and twin Q-values per action.
4. **Experience collection**: run episodes to populate replay buffer.
5. **SAC update**: off-policy updates using the replay buffer, plus an optional cross-entropy loss to match MCTS priors.
6. **Training loop**: repeats for 10,000 iterations, saves checkpoints, supports early stop via Ctrl+C.

## 2) Inputs and outputs (what SAC.py consumes and produces)

Inputs:

- Configuration values from the Config class (problem size, model sizes, SAC hyperparameters, curriculum, and logging flags).
- A target polynomial for each episode, generated in one of the target modes (random, mixed patterns, pool, or dataset).
- The current Game state: graph of the partial circuit, action mask for legal actions, and the target’s compact encoding.
- Optional MCTS priors (policy distributions) when MCTS guidance is enabled.

Outputs:

- Trained model weights saved periodically as files named like sac_model_n{n_variables}_C{max_complexity}.pt.
- A final or interrupt checkpoint if training stops with Ctrl+C.
- Console logs per iteration (success rate, losses, buffer size, example episodes).
- Optional Weights & Biases logs if enabled.

## 3) Configuration and setup

The `Config` class defines all hyperparameters and constants:

- **Problem size**: `n_variables`, `max_complexity`, `max_degree`, and modulus `mod`.
- **Model**: GNN sizes, transformer layers/heads, embedding sizes.
- **SAC**: `gamma`, `alpha`, `tau`, learning rate, batch size, replay buffer size, `steps_per_iter`, `updates_per_iter`.
- **Action space**: determined by `max_nodes` and `max_actions`.
- **MCTS**: mix-in probability and temperature, cross-entropy weight.
- **Curriculum**: `complexity_threshold`, `complexity_window`.
- **Target generation**: random/mixed/pool/dataset settings.
- **Synthetic dataset**: prefill size and complexity range.
- **Logging**: `show_progress_bars`, `use_wandb` (disabled by default).

The script also sets up `sys.path` so it can import from sibling folders, and initializes the device (CUDA if available).

## 4) Target data sources (data generation)

There are four ways targets can be produced:

### A) Random circuits (`generate_random_circuit`)

- Imported from `generator`.
- Produces a random circuit and its resulting polynomials.
- Used in several places as a fallback or main mode.

### B) Pattern-based circuits (`build_pattern_actions`)

- Generates structured circuits at specific complexities (2, 3, 4, >=5) using templates like “sum times sum” or “square of sum”.
- Ensures the final polynomial’s degree is within `max_degree`.
- Returns actions and the target polynomial.

### C) Mixed circuits (`generate_mixed_circuit`)

- First tries pattern-based generation.
- Falls back to random generation if needed.
- Deduplicates targets when a `seen_polynomials` set is provided.
- Produces the target plus a compact encoder representation.

### D) “Interesting polynomial” dataset (`load_interesting_circuit_data`)

- Optionally loads precomputed circuits from JSONL files in `Game-Board-Generation/pre-training-data`.
- Reconstructs actions by graph traversal of stored nodes and edges.
- Filters by `max_complexity` and “multi-path” constraints.
- Caches results in memory for reuse.

## 5) Compact encoding of circuits

The CompactOneHotGraphEncoder encodes a circuit’s action history into a fixed-size vector. It is used for the target polynomial encoding that conditions the policy.

How the encoder works (from the encoder implementation):

- The encoding is a fixed-length vector split into three parts:
   - Operation type: a one-hot for add vs multiply at each step.
   - Edge slots: a one-hot placement for the two operands chosen at each step.
   - Last-node marker: a one-hot indicating which step is currently being filled.
- Each update appends a single operation (add or multiply) and the two operand node ids.
- The size is determined by max complexity (N), modulus (P), and number of variables (D).

`encode_actions_with_compact_encoder`:

- Iterates over actions and updates the encoder for each add/multiply operation.
- Produces a vector used as the “target polynomial encoding”.

## 6) Synthetic dataset prefill

`generate_synthetic_transitions` pre-fills the replay buffer with synthetic state-action transitions:

1. Randomly generate a circuit (within a complexity range).
2. Build a `Game` with the final polynomial as target.
3. Re-play the same action sequence to generate transitions.
4. For each step, compute reward, done flag, and next state.

These transitions are added to the replay buffer before training begins, if `use_synthetic_dataset` is enabled.

## 7) Model architecture

### A) ArithmeticCircuitGNN

- A GCN-based encoder over the current circuit graph.
- Uses residual connections and layer norms.
- Produces node embeddings that are pooled into a graph embedding.

### B) SACCircuitBuilder

- Shared encoder for both policy and Q-values.
- Components:
  - GNN embedding of the current circuit.
  - Linear embedding of the target polynomial encoding.
   - Transformer decoder that fuses the two embeddings into a single fused state vector.
  - Heads:
    - `action_head`: policy logits over all actions.
    - `q1_head`, `q2_head`: twin Q-values for SAC.

Roles of the GNN and Transformer:

- The GNN captures the structure of the partial circuit (how nodes connect and combine).
- The target encoding tells the model what polynomial it should reach.
- The Transformer decoder fuses “what we have built so far” (graph embedding) with “what we want” (target encoding). It uses a learned output token to query the memory formed by those two embeddings, producing a single representation that the policy and Q heads share.

### C) Action masking

- `available_actions_masks` can mask invalid actions.
- The action logits are set to $-\infty$ for invalid actions.

## 8) Replay buffer

A simple ring buffer `ReplayBuffer` stores tuples:

```
(state, action, reward, next_state, done, mcts_pi, has_mcts)
```

Sampling is uniform random for SAC updates.

## 9) Interaction with the Game environment

- `Game` is imported from `State.py` (SAC uses the RL environment in the project).
- The agent observes `(circuit_graph, target_encoding, actions, action_mask)`.
- `extract_state` keeps only `(graph, target_encoding, mask)` in a CPU-friendly format.
- Actions are integer indices (encoded add/multiply operations).

## 10) Experience collection (collect_experience)

This is the data collection step per iteration:

1. **Pick a target** based on `training_target_mode`:
   - `pool`: precomputed pool of targets.
   - `dataset`: draw from interesting polynomial dataset.
   - `mixed`: pattern + random mix.
   - `random`: random circuit only.
2. **Build a Game** with that target.
3. **Roll out** until done or until `steps_per_iter` are collected:
   - With probability `mcts_policy_mix`, ask MCTS for an action and policy prior.
   - Otherwise sample from the SAC policy.
   - Step the game, compute reward, and add transition to replay buffer.
4. **Track success** and keep up to 5 example episodes for logging.

A success is defined as the game being done with reward > 5.0. The success rate is computed per iteration.

## 11) SAC update (sac_update)

After collecting experience, the code performs `updates_per_iter` gradient steps:

1. Sample a batch from the replay buffer.
2. Compute Q-values and policy logits for current states.
3. Compute target values using the target network:
   - Sample next-action distribution from policy.
   - Compute target Q with entropy term:
     $$V(s') = \sum_a \pi(a|s')\big(\min(Q_1, Q_2) - \alpha \log \pi(a|s')\big)$$
   - Compute TD target:
     $$Q_{target} = r + (1 - done)\,\gamma V(s')$$
4. Losses:
   - **Q loss**: MSE for both Q heads.
   - **Policy loss**: expected $\alpha \log \pi - Q$.
   - **MCTS CE loss**: if MCTS guidance exists, match policy to MCTS prior.
5. Backprop, gradient clipping, and soft update of the target network.

## 12) Training loop

Inside `main()`:

1. Create models and optimizer.
2. Optionally prefill replay buffer with synthetic transitions.
3. Set `current_complexity = 1`.
4. For `iteration` from 1 to 10,000:
   - Collect experience.
   - Update curriculum if success rate exceeds the threshold.
   - Run SAC updates.
   - Log iteration stats and a few example episodes.
   - Save the model every 50 iterations.

### One training iteration, step by step

1. Choose or sample a target polynomial for the current complexity.
2. Build a Game instance with that target and its compact encoding.
3. Roll out actions until the per-iteration step budget is met:
   - With probability mcts_policy_mix, take an MCTS-guided action and store its prior.
   - Otherwise, sample from the SAC policy (action logits masked by legal actions).
4. Store transitions in the replay buffer.
5. Run updates_per_iter gradient steps (Q loss + policy loss + optional MCTS CE loss).
6. Update curriculum if the recent success rate exceeds the threshold.
7. Log metrics and show example episodes.

### Curriculum logic

- Maintains a sliding window of recent successes.
- When the success rate in that window exceeds `complexity_threshold`, and `current_complexity < max_complexity`, complexity increments by 1.

## 13) Stopping and saving

The training loop is wrapped in a `try/except KeyboardInterrupt`:

- If you press Ctrl+C, it saves a checkpoint named:
  `sac_model_n{n_variables}_C{max_complexity}_interrupt.pt`.

## 14) Last lines of code

The script ends with:

```python
if __name__ == "__main__":
    main()
```

This runs the training when you execute the file directly.

---

If you want this doc to include diagrams or inline examples for each function, say the word and I’ll expand it.