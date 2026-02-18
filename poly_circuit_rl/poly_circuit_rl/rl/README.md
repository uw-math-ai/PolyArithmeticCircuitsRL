# rl/ — DQN Agent, Transformer Network, and Training

This module contains the reinforcement learning components: the transformer Q-network, the Double DQN agent with action masking, the HER replay buffer, and the curriculum training loop.

## Files

### network.py — CircuitTransformerQ

A transformer-based Q-network that takes a flat observation and produces Q-values for all actions.

**Architecture:**
1. **Parse** flat obs into per-node features (continuous) + index fields (discrete)
2. **Embed** parent indices and position via `nn.Embedding(L+1, d_pos)` — learned structural encoding of the circuit DAG
3. **Encode target** polynomial via swappable `TargetEncoder` (currently `Linear(m, d_model)`, future: coefficient encoder for 20+ variables)
4. **Concatenate** continuous features + embeddings + target + steps_left → project to `d_model`
5. **Transformer encoder** with causal mask (node i can only attend to j ≤ i) and padding mask (empty slots ignored)
6. **Bilinear action scoring**:
   - `Q_add[i,j] = h_i^T W_add h_j` for all pairs i ≤ j
   - `Q_mul[i,j] = h_i^T W_mul h_j`
   - `Q_set_output[i] = w_out^T h_i`
   - `Q_stop = w_stop^T mean_pool(H)`

**Why bilinear scoring?** The action is structured (choose op + two parents), but we flatten to a single Q-value vector for standard DQN. Bilinear scoring over node embeddings captures pairwise interactions without autoregressive overhead.

**Parameter count:** ~110K with default config (d_model=64, 3 layers).

### agent.py — DQNAgent

Standard Double DQN with:
- **Action masking**: invalid actions get Q-value = -1e9 before argmax
- **Epsilon-greedy exploration**: linear decay from `eps_start` to `eps_end` over `eps_decay_steps`
- **Soft target update**: `tau=0.005` per training step
- **Huber loss** (smooth L1) with gradient clipping (max_norm=10.0)
- Save/load checkpoint support

### replay_buffer.py — HERReplayBuffer

Circular replay buffer with **Hindsight Experience Replay** (HER):

The key insight: most episodes fail to reach the target polynomial, producing zero reward. HER retroactively relabels the goal to a polynomial the agent *did* build, creating positive reward signals.

**Strategy: "future"**
- For each transition at time t, sample `k=4` future timesteps t' > t
- Pick a random node's eval vector from timestep t' as the new goal
- Relabel: if any node at t+1 matches the new goal → reward = +1.0, done = True
- Store both original and relabeled transitions

This turns sparse reward (only on solve) into dense reward, dramatically accelerating learning.

### trainer.py — Training Loop

**`train(config, interesting_jsonl)`** — main entry point:
1. Creates environment and agent
2. Optionally loads interesting polynomial data for mixed sampling
3. Runs curriculum training:
   - Level 0: max_ops=1 (simplest circuits)
   - Level 1: max_ops=2 (after 80% success rate)
   - Level 2: max_ops=3, Level 3: max_ops=4
4. At curriculum level >= 1, uses mixed sampling: 70% interesting polynomials + 30% random
5. Interleaved training: trains every 4 environment steps
6. Periodic evaluation with deterministic policy
7. Saves best checkpoint per curriculum level

**`collect_episode(env, agent, ...)`** — runs one episode, stores transitions with HER

**`evaluate(env, agent, max_ops, num_episodes)`** — evaluates with deterministic policy, returns success rate

## Training Tips

- Start with default config for 2-variable polynomials — should reach >50% on level 0 within ~5K steps
- The agent needs ~50K steps to stabilize on level 1
- If using interesting polynomials, the agent focuses on multi-path targets at higher levels
- Monitor `SR` (success rate) and `Loss` in the log output
- Checkpoints are saved to `log_dir` when eval success rate improves
