## MCTS Guidance in PPO

1. **State cloning** – `mcts.CircuitGameState` mirrors the active `Game` object by copying its actions, generated SymPy polynomials, and curriculum limits. This clone exposes `available_actions`, `apply_action`, and a reward heuristic (`success_reward` for a match, otherwise `step_penalty + similarity_score`) so rollouts line up with PPO’s objective.
2. **Tree search** – `MCTSPlanner` wraps a vanilla UCT loop: selection travels the tree with exploitation/exploration scoring, expansion adds one new action, rollouts sample random continuations until hitting the max steps or achieving the target, and backpropagation accumulates visit counts and values.
3. **RL loop usage** – In `train_ppo`, we instantiate one planner (config flags `use_mcts`, `mcts_simulations`, `mcts_exploration`). During each environment step we:
   - Build the current observation via `Game.observe()`.
   - Ask the planner for its preferred action. If it returns `None`, PPO samples normally.
   - Pass the guided action into `model.get_action_and_value`. If the policy mask rejects it we immediately resample, preserving valid log probabilities for PPO’s loss.
4. **Outcome** – The guided action is executed in the environment, but PPO still records the policy’s log-prob and value estimate for that move, so gradients keep flowing even while MCTS nudges exploration toward promising algebraic compositions.
