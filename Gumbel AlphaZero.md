# Implement Gumbel AlphaZero for Polynomial Arithmetic Circuit Discovery

This README is an implementation brief for Codex. The goal is to add a Gumbel AlphaZero / Gumbel MuZero style search option to the existing polynomial arithmetic circuit RL project.

The current project already has:

- a deterministic circuit-building environment,
- a flat masked action space of arithmetic operations `(op, i, j)`,
- PPO,
- PPO+MCTS / Expert Iteration,
- AlphaZero-style neural MCTS,
- a policy-value network,
- curriculum learning,
- factor-library shaping and completion bonuses.

The implementation should add a **Gumbel search mode** that can replace or augment the existing PUCT MCTS search in PPO+MCTS and AlphaZero.

Primary references:

- Danihelka et al., *Policy Improvement by Planning with Gumbel*, ICLR 2022: https://openreview.net/forum?id=bERaNdoegnO
- Paper PDF: https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf
- DeepMind Mctx JAX library: https://github.com/google-deepmind/mctx
- Mctx policy improvement demo: https://github.com/google-deepmind/mctx/blob/main/examples/policy_improvement_demo.py

---

## 0. High-level goal

Implement a new search algorithm:

```text
Current:   Neural MCTS with PUCT root selection and visit-count targets
New:       Gumbel root search with Sequential Halving and completed-Q policy targets
```

The minimal integration should support:

```bash
python -m src.main --algorithm ppo-mcts --search gumbel
python -m src.main --algorithm alphazero --search gumbel
python -m src.main --eval-only --checkpoint checkpoint.pt --algorithm alphazero --search gumbel
```

The implementation should not remove the existing PUCT MCTS. It should add Gumbel as a configurable alternative.

---

## 1. Why Gumbel is expected to help this project

In polynomial circuit discovery, the root action space is large and sparse. An action is a triple

```text
(op, i, j), where op in {ADD, MUL} and i <= j.
```

With default parameters, the action space is around 90 masked actions. Most actions are valid but useless. Ordinary PUCT can over-exploit the current policy prior. If the policy network initially assigns low prior to a crucial algebraic action, such as squaring an already-built factor, ordinary PUCT may never test it under a small simulation budget.

Gumbel AlphaZero changes the root search behavior:

1. Sample a diverse candidate set of root actions without replacement using Gumbel-Top-k.
2. Use Sequential Halving to allocate simulations fairly across candidates.
3. Pick the action maximizing

   ```text
   gumbel_noise[action] + prior_logits[action] + transformed_Q[action]
   ```

4. Train the policy toward a completed-Q improved policy rather than raw visit counts.

This is particularly suitable for polynomial circuits, where globally useful actions often look locally unremarkable:

```text
x0 + 1                    useful factor
(x0 + 1) * (x0 + 1)        useful square
x0^2 * x0^2                repeated squaring
(x0 + x1 + 1)              hidden additive factor
```

The point is not generic exploration. The point is **diverse root candidate testing under a small simulation budget**.

---

## 2. Preferred implementation path

There are two possible paths.

### Path A: Use `mctx` if this branch is truly JAX-native

If the current PPO/MCTS branch uses JAX environment states and JAX model application, use DeepMind's `mctx.gumbel_muzero_policy`.

This is preferred because Mctx already implements batched JIT-compatible Gumbel MuZero search with:

- `RootFnOutput`,
- `RecurrentFnOutput`,
- `gumbel_muzero_policy`,
- completed-Q transforms,
- action weights suitable as policy targets.

Install:

```bash
pip install mctx
```

### Path B: Hand-write a root-only Gumbel searcher

If the current code is mostly PyTorch or has a non-JAX environment that is hard to plug into Mctx, implement a root-only Gumbel wrapper around the existing `mcts.py`.

This is lower risk for the current codebase if the MCTS implementation already works. It gives most of the practical benefit:

```text
Root:      Gumbel-Top-k + Sequential Halving
Interior:  existing PUCT MCTS
Target:    completed-Q improved policy
```

Start with Path B if Mctx integration creates too much refactoring.

---

## 3. Files to add or edit

Add:

```text
src/algorithms/gumbel_mcts.py
```

Edit:

```text
src/config.py
src/main.py
src/algorithms/ppo_mcts.py
src/algorithms/alphazero.py
src/evaluation/evaluate.py
tests/test_gumbel_mcts.py
```

Optional if using Mctx/JAX:

```text
src/algorithms/mctx_gumbel.py
```

---

## 4. Configuration additions

Add to `Config`:

```python
# Search selection
search: str = "puct"  # choices: "puct", "gumbel"

# Gumbel search parameters
gumbel_num_simulations: int = 32
gumbel_max_num_considered_actions: int = 16
gumbel_scale: float = 1.0
gumbel_c_visit: float = 50.0
gumbel_c_scale: float = 0.1
gumbel_use_completed_q: bool = True
gumbel_use_mixed_value: bool = True
gumbel_q_normalize: bool = True
gumbel_root_only: bool = True

# Training loss integration
gumbel_policy_target: str = "completed_q"  # choices: "completed_q", "visits"
gumbel_distill_coef: float = 0.5
```

Add CLI flags:

```bash
--search {puct,gumbel}
--gumbel-num-simulations 32
--gumbel-max-num-considered-actions 16
--gumbel-scale 1.0
--gumbel-c-visit 50.0
--gumbel-c-scale 0.1
--gumbel-distill-coef 0.5
--no-gumbel-q-normalize
--gumbel-root-only
```

Recommended defaults:

```python
gumbel_num_simulations = 32
gumbel_max_num_considered_actions = 16
gumbel_scale = 1.0
gumbel_c_visit = 50.0
gumbel_c_scale = 0.1
```

For quick CPU tests:

```python
gumbel_num_simulations = 8
gumbel_max_num_considered_actions = 8
```

For evaluation:

```python
gumbel_num_simulations = 64
gumbel_max_num_considered_actions = 16
```

---

## 5. Core API for `gumbel_mcts.py`

Create a return container:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class GumbelMCTSOutput:
    action: int
    action_weights: np.ndarray  # shape [max_actions], completed-Q target
    root_q: np.ndarray          # shape [max_actions]
    root_visits: np.ndarray     # shape [max_actions]
    root_logits: np.ndarray     # shape [max_actions]
    considered_actions: np.ndarray
```

Main entry point:

```python
def run_gumbel_mcts(
    game,
    model,
    config,
    device=None,
    rng=None,
    num_simulations=None,
    temperature=1.0,
) -> GumbelMCTSOutput:
    """
    Run Gumbel root search from the current game state.

    The returned action is the action to execute.
    The returned action_weights are the policy target for training.
    """
```

The function must:

1. Evaluate the root network policy/value.
2. Mask invalid actions.
3. Sample Gumbel noise.
4. Select the top-k candidate actions by `root_logits + gumbel_noise`.
5. Run Sequential Halving over the candidate actions.
6. Fill unvisited Q-values using the root value.
7. Construct completed-Q action weights.
8. Return the selected action and training targets.

---

## 6. Root network evaluation

Use the same method currently used by `mcts.py`.

Expected shape:

```python
logits: np.ndarray  # [max_actions]
value: float
mask: np.ndarray    # [max_actions], True for valid actions
```

Pseudocode:

```python
def evaluate_root(game, model, config, device):
    obs = game.get_observation()
    logits, value = model_forward(model, obs, device)
    mask = obs["mask"]
    logits = np.asarray(logits, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    masked_logits = np.where(mask, logits, -np.inf)
    return masked_logits, float(value), mask
```

Be careful: invalid actions must remain impossible at every stage.

---

## 7. Gumbel-Top-k candidate selection

Implement:

```python
def sample_gumbel(shape, rng):
    u = rng.uniform(low=1e-8, high=1.0 - 1e-8, size=shape)
    return -np.log(-np.log(u))


def gumbel_top_k(masked_logits, valid_mask, k, rng, gumbel_scale=1.0):
    g = sample_gumbel(masked_logits.shape, rng) * gumbel_scale
    scores = masked_logits + g
    scores = np.where(valid_mask, scores, -np.inf)
    k = min(k, int(valid_mask.sum()))
    actions = np.argpartition(-scores, kth=k - 1)[:k]
    actions = actions[np.argsort(-scores[actions])]
    return actions, g
```

This samples a candidate slate without replacement, biased by the policy prior.

---

## 8. Sequential Halving root search

Implement root Sequential Halving as a loop over candidate actions.

Pseudocode:

```python
def sequential_halving_root(
    game,
    model,
    config,
    candidate_actions,
    gumbel_noise,
    root_logits,
    root_value,
    valid_mask,
    rng,
):
    q_sum = np.zeros_like(root_logits, dtype=np.float32)
    visits = np.zeros_like(root_logits, dtype=np.int32)

    remaining = list(candidate_actions)
    num_phases = int(np.ceil(np.log2(max(2, len(remaining)))))
    sims_used = 0
    total_sims = config.gumbel_num_simulations

    for phase in range(num_phases):
        if len(remaining) <= 1:
            break

        sims_left = total_sims - sims_used
        phases_left = num_phases - phase
        sims_per_action = max(1, sims_left // (phases_left * len(remaining)))

        for action in remaining:
            for _ in range(sims_per_action):
                q = simulate_after_root_action(
                    game=game,
                    action=action,
                    model=model,
                    config=config,
                    rng=rng,
                )
                q_sum[action] += q
                visits[action] += 1
                sims_used += 1

        q_mean = np.where(visits > 0, q_sum / np.maximum(visits, 1), root_value)
        sigma_q = transform_q(q_mean, visits, valid_mask, config)

        scores = {
            a: gumbel_noise[a] + root_logits[a] + sigma_q[a]
            for a in remaining
        }

        remaining = sorted(remaining, key=lambda a: scores[a], reverse=True)
        remaining = remaining[:max(1, len(remaining) // 2)]

    q_mean = np.where(visits > 0, q_sum / np.maximum(visits, 1), root_value)
    sigma_q = transform_q(q_mean, visits, valid_mask, config)

    final_scores = np.full_like(root_logits, -np.inf, dtype=np.float32)
    for action in remaining:
        final_scores[action] = gumbel_noise[action] + root_logits[action] + sigma_q[action]

    selected_action = int(np.argmax(final_scores))
    return selected_action, q_mean, visits
```

Important:

- Do not rank invalid actions.
- Do not let NaNs from `-inf` logits enter the final softmax.
- If only one valid action exists, return it directly.

---

## 9. Simulation after a root action

For the first implementation, use the current environment clone mechanism and current MCTS rollout machinery.

Pseudocode:

```python
def simulate_after_root_action(game, action, model, config, rng):
    sim_game = game.clone()
    obs, reward, done, info = sim_game.step(action)

    if done:
        return terminal_value(reward, done, info, sim_game, config)

    # Simple first version:
    # bootstrap from neural value after one root action.
    logits, value = evaluate_model(model, sim_game.get_observation())
    return reward + config.gamma * float(value)
```

This one-step bootstrap version is already useful.

Second version:

```python
def simulate_after_root_action(...):
    sim_game = game.clone()
    obs, reward, done, info = sim_game.step(action)

    if done:
        return terminal_value(...)

    # Continue with existing PUCT MCTS or a depth-limited greedy/value rollout.
    continuation_value = run_existing_puct_value_estimate(
        sim_game,
        model,
        config,
        num_simulations=config.gumbel_inner_simulations,
    )
    return reward + config.gamma * continuation_value
```

Start with the one-step bootstrap. Add inner PUCT only if performance needs it.

---

## 10. Terminal value for circuit discovery

Use a normalized value target. Recommended:

```python
def terminal_value(reward, done, info, game, config):
    if info.get("success", False):
        # Favor shorter circuits mildly.
        length_penalty = 0.05 * game.num_steps / max(1, config.max_steps)
        return 1.0 - length_penalty
    else:
        return -1.0
```

Avoid feeding large shaped reward sums directly into Gumbel Q at first. The environment has step penalty, success bonus, potential shaping, factor reward, library bonus, and completion bonus. These are useful for PPO, but the root Q transform can become unstable if the scale varies too much.

Recommended staged behavior:

```text
Stage 1: Gumbel Q uses sparse success/shortness value.
Stage 2: Add normalized shaped reward into one-step bootstrap.
Stage 3: Use full reward only after Q normalization and clipping are stable.
```

---

## 11. Q transform

Gumbel root scores use:

```text
gumbel_noise[action] + prior_logits[action] + sigma(Q[action])
```

Implement a stable transform:

```python
def transform_q(q, visits, valid_mask, config):
    q = np.asarray(q, dtype=np.float32)
    visits = np.asarray(visits, dtype=np.float32)

    if config.gumbel_q_normalize:
        valid_q = q[valid_mask]
        q_min = np.min(valid_q)
        q_max = np.max(valid_q)
        q_norm = (q - q_min) / (q_max - q_min + 1e-8)
    else:
        q_norm = q

    max_visit = np.max(visits)
    sigma_q = (config.gumbel_c_visit + max_visit) * config.gumbel_c_scale * q_norm
    sigma_q = np.where(valid_mask, sigma_q, -np.inf)
    return sigma_q
```

Start with:

```python
gumbel_c_visit = 50.0
gumbel_c_scale = 0.1
```

Keep `c_scale` conservative because polynomial-circuit rewards are shaped and not naturally in a fixed board-game value scale.

---

## 12. Completed-Q improved policy target

After search, fill unvisited actions with the root value:

```python
def completed_q_policy(root_logits, root_value, q, visits, valid_mask, config):
    completed_q = np.where(visits > 0, q, root_value)
    sigma_q = transform_q(completed_q, visits, valid_mask, config)
    target_logits = root_logits + sigma_q
    target_logits = np.where(valid_mask, target_logits, -np.inf)
    return masked_softmax(target_logits, valid_mask)
```

Masked softmax:

```python
def masked_softmax(logits, mask):
    logits = np.where(mask, logits, -np.inf)
    max_logit = np.max(logits[mask])
    exp_logits = np.zeros_like(logits, dtype=np.float32)
    exp_logits[mask] = np.exp(logits[mask] - max_logit)
    return exp_logits / (np.sum(exp_logits) + 1e-8)
```

This `action_weights` vector should replace visit-count targets whenever `--search gumbel` is active.

---

## 13. PPO+MCTS integration

Current PPO+MCTS behavior policy is approximately the MCTS visit-count distribution. For Gumbel, use `GumbelMCTSOutput.action_weights`.

During rollout collection:

```python
if config.search == "gumbel":
    search_out = run_gumbel_mcts(game, model, config, device, rng)
    action = search_out.action
    behavior_policy = search_out.action_weights
    behavior_log_prob = np.log(behavior_policy[action] + 1e-8)
else:
    search_out = run_puct_mcts(...)
    action = sample_from_visit_counts(...)
    behavior_log_prob = np.log(mcts_policy[action] + 1e-8)
```

Store in rollout buffer:

```python
transition = {
    "obs": obs,
    "action": action,
    "reward": reward,
    "done": done,
    "value": value,
    "behavior_log_prob": behavior_log_prob,
    "search_policy": behavior_policy,
}
```

PPO loss remains:

```python
ratio = exp(new_log_prob - behavior_log_prob)
policy_loss = clipped_surrogate_loss(ratio, advantages)
```

Add an auxiliary Gumbel distillation loss:

```python
log_probs = log_softmax(masked_logits)
distill_loss = -torch.sum(search_policy * log_probs, dim=-1).mean()

loss = (
    ppo_policy_loss
    + config.value_coef * value_loss
    - config.entropy_coef * entropy
    + config.gumbel_distill_coef * distill_loss
)
```

If this is a JAX branch, use the JAX equivalent:

```python
log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
distill_loss = -jnp.sum(search_policy * log_probs, axis=-1).mean()
```

Important: if PPO becomes unstable, temporarily set the PPO policy loss aside and train the policy by pure cross-entropy to the completed-Q search policy.

---

## 14. AlphaZero integration

This is cleaner than PPO+MCTS.

Current AlphaZero stores:

```python
(state, mcts_visit_policy, outcome)
```

Gumbel version should store:

```python
(state, gumbel_action_weights, return_target)
```

During self-play:

```python
search_out = run_gumbel_mcts(game, model, config, device, rng)
action = search_out.action
policy_target = search_out.action_weights
trajectory.append((obs, policy_target))
```

After episode:

```python
z = compute_episode_return_or_success_value(...)
for obs, policy_target in trajectory:
    replay_buffer.add(obs, policy_target, z)
```

Training loss:

```python
logits, value = model(obs)
log_probs = log_softmax(masked_logits)

policy_loss = -sum(policy_target * log_probs)
value_loss = mse(value, z)
loss = policy_loss + value_coef * value_loss
```

This should be the first serious benchmark because it matches the Gumbel AlphaZero training recipe more directly than PPO+MCTS.

---

## 15. Mctx/JAX implementation sketch

If using Mctx, implement a wrapper in:

```text
src/algorithms/mctx_gumbel.py
```

Expected structure:

```python
import functools
import jax
import jax.numpy as jnp
import mctx


def run_mctx_gumbel_policy(params, model_apply, env_state, config, rng_key):
    obs = observe(env_state)
    logits, value = model_apply(params, obs)
    invalid_actions = ~obs["mask"]
    masked_logits = jnp.where(obs["mask"], logits, -jnp.inf)

    root = mctx.RootFnOutput(
        prior_logits=masked_logits,
        value=value,
        embedding=env_state,
    )

    def recurrent_fn(params, rng_key, action, state):
        next_state, reward, done, info = env_step(state, action)
        next_obs = observe(next_state)
        next_logits, next_value = model_apply(params, next_obs)
        next_masked_logits = jnp.where(next_obs["mask"], next_logits, -jnp.inf)
        discount = jnp.where(done, 0.0, config.gamma)

        recurrent_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=next_masked_logits,
            value=next_value,
        )
        return recurrent_output, next_state

    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=config.gumbel_num_simulations,
        invalid_actions=invalid_actions,
        max_depth=config.max_steps,
        qtransform=functools.partial(
            mctx.qtransform_completed_by_mix_value,
            use_mixed_value=config.gumbel_use_mixed_value,
        ),
        max_num_considered_actions=config.gumbel_max_num_considered_actions,
        gumbel_scale=config.gumbel_scale,
    )

    return policy_output.action, policy_output.action_weights, policy_output
```

Notes:

- Mctx is batched. If the current code is single-environment, add a batch dimension of size 1.
- `embedding` must be a JAX pytree with static-compatible shapes.
- The environment transition must be pure/JAX-compatible for full JIT. If the current environment is Python-object based, start with the hand-written root-only implementation instead.
- The `invalid_actions` argument should be `True` for invalid actions.

---

## 16. Evaluation behavior

For evaluation, do not sample stochastically from a visit distribution.

Recommended:

```python
search_out = run_gumbel_mcts(..., gumbel_scale=0.0)
action = search_out.action
```

Alternative:

```python
fixed_rng = np.random.default_rng(seed)
search_out = run_gumbel_mcts(..., rng=fixed_rng)
action = search_out.action
```

The first option is deterministic. The second option preserves the Gumbel procedure but is reproducible.

---

## 17. Concrete examples to test manually

### Example A: square of a factor

Target:

```text
T = (x0 + 1)^2
```

Suppose the circuit already has:

```text
v0 = x0
v1 = x1
v2 = 1
v3 = x0 + 1
```

The best action is:

```text
MUL(v3, v3)
```

Test that Gumbel search can select `MUL(v3, v3)` even if its prior is lower than several addition actions.

### Example B: repeated squaring

Target:

```text
T = x0^4
```

Suppose the circuit has:

```text
v3 = x0^2
```

The best action is:

```text
MUL(v3, v3)
```

Create a fake prior where `MUL(v0, v3)` has higher prior than `MUL(v3, v3)`. The search should still pick `MUL(v3, v3)` after simulation because it immediately solves.

### Example C: hidden factor

Target:

```text
T = (x0 + x1 + 1)(x0 + 1)
```

Test whether Gumbel search increases the probability of choosing factor-building actions such as:

```text
ADD(x0, 1)
ADD(x0 + x1, 1)
```

compared to ordinary PUCT under the same simulation budget.

---

## 18. Unit tests

Add `tests/test_gumbel_mcts.py`.

### Test 1: invalid actions are never selected

```python
def test_gumbel_never_selects_invalid_actions():
    ...
```

Construct a state with only the initial nodes. Run Gumbel search many times. Assert:

```python
assert mask[output.action]
assert np.all(output.action_weights[~mask] == 0)
```

### Test 2: action weights sum to one

```python
def test_gumbel_action_weights_are_probability_distribution():
    ...
```

Assert:

```python
np.testing.assert_allclose(output.action_weights.sum(), 1.0, atol=1e-5)
assert np.all(output.action_weights >= 0)
```

### Test 3: terminal winning action gets high target weight

Make a tiny environment state where one valid action immediately solves the target. Stub the model to give that action mediocre prior. Assert that after Gumbel search:

```python
assert output.action == winning_action
assert output.action_weights[winning_action] > prior_prob[winning_action]
```

### Test 4: lower-prior winning action can beat higher-prior distractor

Stub priors:

```text
high prior distractor: 0.5
low prior winner:     0.05
```

Make the winner immediately terminal. With enough simulations and deterministic Gumbel off for evaluation, assert that the winner is selected.

### Test 5: reproducibility with fixed seed

Run the same search twice with the same RNG seed and same model. Assert identical action and close action weights.

### Test 6: PPO+MCTS buffer includes search policy

Run one collection step with `--search gumbel`. Assert the transition has:

```python
"behavior_log_prob"
"search_policy"
```

and `search_policy` is a valid probability distribution.

### Test 7: AlphaZero buffer uses completed-Q target

Run one self-play game with `--search gumbel`. Assert replay entries contain `policy_target` that is not merely raw visit counts unless configured otherwise.

---

## 19. Benchmark matrix

Run these four baselines on the same seeds and curriculum:

```bash
python -m src.main --algorithm ppo --iterations 100 --seed 0

python -m src.main --algorithm ppo-mcts --search puct --iterations 100 --seed 0 \
  --mcts-simulations 32

python -m src.main --algorithm ppo-mcts --search gumbel --iterations 100 --seed 0 \
  --gumbel-num-simulations 32 --gumbel-max-num-considered-actions 16

python -m src.main --algorithm alphazero --search gumbel --iterations 100 --seed 0 \
  --gumbel-num-simulations 32 --gumbel-max-num-considered-actions 16
```

Track:

```text
success_rate_by_complexity
average_steps_to_solution
average_circuit_length_among_successes
wall_clock_seconds_per_iteration
number_of_distinct_targets_solved
factor_library_hit_rate
completion_bonus_rate
policy_entropy
root_search_entropy
fraction_of_search_budget_spent_on winning/final action
```

Recommended seeds:

```text
0, 1, 2, 3, 4
```

---

## 20. Logging additions

Log the following per training iteration:

```python
log_dict.update({
    "search/type": config.search,
    "gumbel/num_simulations": config.gumbel_num_simulations,
    "gumbel/max_num_considered_actions": config.gumbel_max_num_considered_actions,
    "gumbel/root_policy_entropy": entropy(output.action_weights),
    "gumbel/considered_actions": len(output.considered_actions),
    "gumbel/selected_action_visit_count": output.root_visits[output.action],
    "gumbel/max_root_q": np.max(output.root_q[mask]),
    "gumbel/min_root_q": np.min(output.root_q[mask]),
})
```

Also log examples of selected actions after decoding:

```text
Gumbel selected: MUL(v3, v3)
Target: (x0 + 1)^2
Solved: yes
```

This will make debugging much easier.

---

## 21. Common pitfalls

### Pitfall 1: Treating Gumbel as Dirichlet noise

Do not just add Gumbel noise to the prior and keep PUCT unchanged. The key algorithmic pieces are:

```text
Gumbel-Top-k without replacement
Sequential Halving
completed-Q improved policy target
```

### Pitfall 2: Training on raw visits

For Gumbel mode, train on completed-Q action weights. Raw visit counts are mainly useful for logging.

### Pitfall 3: Letting shaped rewards destabilize Q

The environment reward has several components. Start with normalized sparse value for root Q. Add shaped reward only after the implementation is stable.

### Pitfall 4: Invalid action leakage

All logits, candidate scores, Q transforms, and policy targets must preserve the action mask.

### Pitfall 5: Non-deterministic tests

For unit tests, set either:

```python
gumbel_scale = 0.0
```

or a fixed RNG seed.

### Pitfall 6: JAX shape instability

If using Mctx, the environment state must have static-compatible shapes. The circuit graph should be padded to `max_nodes`, and action masks should always have shape `[max_actions]`.

---

## 22. Suggested implementation order

Follow this order exactly.

### Step 1: Add config and CLI flags

- Add `search` and Gumbel hyperparameters.
- Ensure old behavior is unchanged when `search="puct"`.

### Step 2: Implement standalone root Gumbel helper functions

- `sample_gumbel`
- `gumbel_top_k`
- `masked_softmax`
- `transform_q`
- `completed_q_policy`

Write unit tests for these pure functions first.

### Step 3: Implement `run_gumbel_mcts`

Use one-step bootstrap after root action.

Do not integrate with PPO yet.

Manually test on tiny targets.

### Step 4: Add evaluation support

Allow:

```bash
python -m src.main --eval-only --algorithm alphazero --search gumbel
```

Use deterministic Gumbel or fixed seed.

### Step 5: Integrate with AlphaZero

Replace visit-count target with completed-Q target in replay entries.

Benchmark AlphaZero+Gumbel first.

### Step 6: Integrate with PPO+MCTS

Store behavior log-prob from `action_weights[action]`.

Add auxiliary distillation loss to `search_policy`.

### Step 7: Optional Mctx migration

If the branch is fully JAX, replace the hand-written root-only implementation with `mctx.gumbel_muzero_policy` or keep both modes:

```bash
--search gumbel-root
--search mctx-gumbel
```

---

## 23. Acceptance criteria

The PR is acceptable when:

1. Existing PPO, PPO+MCTS, AlphaZero, and SAC tests still pass.
2. `--search puct` reproduces old behavior.
3. `--search gumbel` never selects invalid actions.
4. Gumbel action weights are valid probability distributions.
5. A tiny terminal-action test shows that a lower-prior winning action can be selected after search.
6. AlphaZero+Gumbel trains without NaNs for at least 5 iterations.
7. PPO+Gumbel trains without NaNs for at least 5 iterations.
8. Evaluation runs with Gumbel search from a checkpoint.
9. Logs include root Q range, selected action visit count, and search policy entropy.
10. The README documents the new flags.

---

## 24. Minimal code skeleton for `gumbel_mcts.py`

```python
"""Gumbel root search for polynomial arithmetic circuit discovery.

This module implements a Gumbel AlphaZero-style root search:

1. sample root candidate actions by Gumbel-Top-k;
2. compare them using Sequential Halving;
3. return the selected action and a completed-Q improved policy target.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GumbelMCTSOutput:
    action: int
    action_weights: np.ndarray
    root_q: np.ndarray
    root_visits: np.ndarray
    root_logits: np.ndarray
    considered_actions: np.ndarray


def sample_gumbel(shape, rng):
    u = rng.uniform(low=1e-8, high=1.0 - 1e-8, size=shape)
    return -np.log(-np.log(u))


def masked_softmax(logits, mask):
    logits = np.asarray(logits, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    out = np.zeros_like(logits, dtype=np.float32)
    if not np.any(mask):
        raise ValueError("masked_softmax received no valid actions")

    valid_logits = logits[mask]
    max_logit = np.max(valid_logits)
    exp_logits = np.exp(valid_logits - max_logit)
    out[mask] = exp_logits / (np.sum(exp_logits) + 1e-8)
    return out


def gumbel_top_k(masked_logits, valid_mask, k, rng, gumbel_scale=1.0):
    valid_count = int(np.sum(valid_mask))
    if valid_count == 0:
        raise ValueError("No valid actions available")

    k = min(int(k), valid_count)
    gumbels = sample_gumbel(masked_logits.shape, rng) * gumbel_scale
    scores = np.where(valid_mask, masked_logits + gumbels, -np.inf)

    # Stable enough for small action spaces.
    actions = np.argsort(-scores)[:k]
    return actions.astype(np.int64), gumbels.astype(np.float32)


def transform_q(q, visits, valid_mask, config):
    q = np.asarray(q, dtype=np.float32)
    visits = np.asarray(visits, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if getattr(config, "gumbel_q_normalize", True):
        valid_q = q[valid_mask]
        q_min = np.min(valid_q)
        q_max = np.max(valid_q)
        q_used = (q - q_min) / (q_max - q_min + 1e-8)
    else:
        q_used = q

    c_visit = getattr(config, "gumbel_c_visit", 50.0)
    c_scale = getattr(config, "gumbel_c_scale", 0.1)
    max_visit = np.max(visits) if visits.size else 0.0
    sigma_q = (c_visit + max_visit) * c_scale * q_used
    return np.where(valid_mask, sigma_q, -np.inf).astype(np.float32)


def completed_q_policy(root_logits, root_value, q, visits, valid_mask, config):
    completed_q = np.where(visits > 0, q, root_value)
    sigma_q = transform_q(completed_q, visits, valid_mask, config)
    target_logits = root_logits + sigma_q
    target_logits = np.where(valid_mask, target_logits, -np.inf)
    return masked_softmax(target_logits, valid_mask)


def terminal_value(reward, done, info, game, config):
    success = bool(info.get("success", False)) if isinstance(info, dict) else False
    if success:
        max_steps = max(1, getattr(config, "max_steps", getattr(config, "max_operations", 1)))
        steps = getattr(game, "steps", getattr(game, "num_steps", 0))
        return float(np.clip(1.0 - 0.05 * steps / max_steps, -1.0, 1.0))
    return -1.0


def evaluate_model_for_search(model, obs, device=None):
    """Adapt this helper to the existing model API.

    Must return:
        logits: np.ndarray of shape [max_actions]
        value: float
    """
    raise NotImplementedError("Wire this to the existing policy-value network")


def simulate_after_root_action(game, action, model, config, device=None, rng=None):
    sim_game = game.clone()
    obs, reward, done, info = sim_game.step(int(action))

    if done:
        return terminal_value(reward, done, info, sim_game, config)

    _, value = evaluate_model_for_search(model, sim_game.get_observation(), device=device)
    gamma = getattr(config, "gamma", 0.99)
    q = float(reward) + gamma * float(value)
    return float(np.clip(q, -1.0, 1.0))


def run_gumbel_mcts(game, model, config, device=None, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng()

    obs = game.get_observation()
    root_logits, root_value = evaluate_model_for_search(model, obs, device=device)
    valid_mask = np.asarray(obs["mask"], dtype=bool)

    root_logits = np.asarray(root_logits, dtype=np.float32)
    root_logits = np.where(valid_mask, root_logits, -np.inf)

    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        raise RuntimeError("Gumbel MCTS found no valid actions")

    if valid_count == 1:
        action = int(np.flatnonzero(valid_mask)[0])
        action_weights = np.zeros_like(root_logits, dtype=np.float32)
        action_weights[action] = 1.0
        return GumbelMCTSOutput(
            action=action,
            action_weights=action_weights,
            root_q=np.full_like(root_logits, float(root_value), dtype=np.float32),
            root_visits=np.zeros_like(root_logits, dtype=np.int32),
            root_logits=root_logits,
            considered_actions=np.array([action], dtype=np.int64),
        )

    k = min(
        getattr(config, "gumbel_max_num_considered_actions", 16),
        getattr(config, "gumbel_num_simulations", 32),
        valid_count,
    )

    candidate_actions, gumbel_noise = gumbel_top_k(
        masked_logits=root_logits,
        valid_mask=valid_mask,
        k=k,
        rng=rng,
        gumbel_scale=getattr(config, "gumbel_scale", 1.0),
    )

    q_sum = np.zeros_like(root_logits, dtype=np.float32)
    visits = np.zeros_like(root_logits, dtype=np.int32)

    remaining = list(map(int, candidate_actions))
    total_sims = int(getattr(config, "gumbel_num_simulations", 32))
    num_phases = int(np.ceil(np.log2(max(2, len(remaining)))))
    sims_used = 0

    for phase in range(num_phases):
        if len(remaining) <= 1:
            break

        sims_left = max(0, total_sims - sims_used)
        if sims_left <= 0:
            break

        phases_left = max(1, num_phases - phase)
        sims_per_action = max(1, sims_left // (phases_left * len(remaining)))

        for action in remaining:
            for _ in range(sims_per_action):
                if sims_used >= total_sims:
                    break
                q = simulate_after_root_action(
                    game=game,
                    action=action,
                    model=model,
                    config=config,
                    device=device,
                    rng=rng,
                )
                q_sum[action] += q
                visits[action] += 1
                sims_used += 1

        q_mean = np.where(visits > 0, q_sum / np.maximum(visits, 1), float(root_value))
        sigma_q = transform_q(q_mean, visits, valid_mask, config)

        scores = [(a, gumbel_noise[a] + root_logits[a] + sigma_q[a]) for a in remaining]
        scores.sort(key=lambda x: x[1], reverse=True)
        remaining = [a for a, _ in scores[:max(1, len(scores) // 2)]]

    q_mean = np.where(visits > 0, q_sum / np.maximum(visits, 1), float(root_value))
    sigma_q = transform_q(q_mean, visits, valid_mask, config)

    final_scores = np.full_like(root_logits, -np.inf, dtype=np.float32)
    for action in remaining:
        final_scores[action] = gumbel_noise[action] + root_logits[action] + sigma_q[action]

    selected_action = int(np.argmax(final_scores))
    action_weights = completed_q_policy(root_logits, float(root_value), q_mean, visits, valid_mask, config)

    return GumbelMCTSOutput(
        action=selected_action,
        action_weights=action_weights,
        root_q=q_mean.astype(np.float32),
        root_visits=visits.astype(np.int32),
        root_logits=root_logits.astype(np.float32),
        considered_actions=np.asarray(candidate_actions, dtype=np.int64),
    )
```

---

## 25. Final implementation advice

Do the simplest thing first:

```text
Root-only Gumbel search
One-step bootstrap value
Completed-Q action target
AlphaZero integration
Then PPO+MCTS integration
```

Only after this works should you attempt a fully JIT-compiled Mctx migration.

The expected first win is not necessarily higher final asymptotic performance. The expected first win is better **sample efficiency under small search budgets**, especially on targets whose shortest circuits require a low-prior multiplication, squaring, or factor-building action.
