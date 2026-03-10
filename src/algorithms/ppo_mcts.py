"""PPO + MCTS hybrid training loop (Expert Iteration with PPO updates).

Combines Monte Carlo Tree Search for action selection during rollouts with
Proximal Policy Optimization for network updates.  MCTS provides higher-quality
action selection (via look-ahead search guided by the policy-value network),
while PPO's clipped surrogate objective provides stable policy improvement.

This implements an *Expert Iteration* loop:
  1. MCTS (the "expert") uses the current network to search for good actions.
  2. Rollout data is collected using MCTS action selection.
  3. PPO updates train the network to approximate the MCTS-improved policy.
  4. The improved network makes MCTS even stronger → virtuous cycle.

The importance sampling ratio in the PPO update is:

    r(θ) = π_θ(a|s) / π_MCTS(a|s)

where π_MCTS is the normalised visit-count distribution from MCTS search and
π_θ is the network's own policy.  Clipping this ratio prevents the network from
overshooting the MCTS target in any single update, maintaining training stability.

Compared to standard AlphaZero (which uses cross-entropy loss on MCTS visit
counts), the PPO formulation adds:
  - Generalised Advantage Estimation (GAE) for lower-variance advantage targets
  - A clipped surrogate objective for conservative policy steps
  - An entropy bonus to prevent premature convergence
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..config import Config
from ..models.policy_value_net import PolicyValueNet
from ..environment.circuit_game import CircuitGame
from ..environment.factor_library import FactorLibrary
from ..game_board.generator import sample_target, build_game_board
from .mcts import MCTS


@dataclass
class MCTSRolloutStep:
    """Single (s, a, r, log π_MCTS, V, done) transition from MCTS rollout.

    Attributes:
        obs: Observation dict at time t (graph, target, mask).
        action: Integer action index selected by MCTS.
        reward: Scalar reward received after taking the action.
        mcts_log_prob: Log probability of the action under the MCTS visit
                       distribution (the behaviour policy).
        value: Baseline value estimate V(s_t) from the network's value head.
        done: True if the episode ended at this step.
    """
    obs: dict
    action: int
    reward: float
    mcts_log_prob: float
    value: float
    done: bool


class MCTSRolloutBuffer:
    """Ordered buffer of MCTSRolloutStep objects from MCTS-guided rollouts.

    Filled by collect_rollouts() and consumed once during update().  The buffer
    is cleared implicitly each iteration (a new one is created).
    """

    def __init__(self) -> None:
        """Initialise an empty buffer."""
        self.steps: List[MCTSRolloutStep] = []

    def add(
        self,
        obs: dict,
        action: int,
        reward: float,
        mcts_log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        """Append a single MCTS-collected transition.

        Args:
            obs: Observation dict from the environment.
            action: Action selected by MCTS.
            reward: Immediate scalar reward.
            mcts_log_prob: Log prob of action under MCTS visit distribution.
            value: Value estimate V(s) from the network.
            done: Whether the episode ended.
        """
        self.steps.append(MCTSRolloutStep(obs, action, reward, mcts_log_prob, value, done))

    def clear(self) -> None:
        """Empty the buffer."""
        self.steps = []

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return len(self.steps)


class PPOMCTSTrainer:
    """PPO + MCTS hybrid trainer with curriculum learning and factor library.

    Orchestrates the Expert Iteration cycle:
      collect (MCTS) → GAE → PPO update → curriculum adjustment.

    The MCTS object holds a reference to the same model being trained, so each
    iteration's MCTS search automatically benefits from the latest network
    weights.

    Attributes:
        config (Config): Shared hyperparameter configuration.
        model (PolicyValueNet): The policy-value network being trained.
        device (str): PyTorch device string ('cpu', 'cuda', 'mps').
        mcts (MCTS): MCTS engine sharing the model reference.
        env (CircuitGame): The circuit construction environment.
        current_complexity (int): Current curriculum complexity level.
        success_history (List[bool]): Rolling history of episode outcomes.
    """

    def __init__(
        self, config: Config, model: PolicyValueNet, device: str = "cpu"
    ) -> None:
        """Initialise the PPO+MCTS trainer.

        Creates the environment (with optional FactorLibrary), the MCTS
        engine, the Adam optimiser, and the curriculum state.

        Args:
            config: Configuration dataclass with all hyperparameters.
            model: Shared policy-value network to train.
            device: PyTorch device ('cpu', 'cuda', or 'mps').
        """
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=config.ppo_lr)

        # MCTS engine — shares the model reference so it always uses
        # the latest network weights.
        self.mcts = MCTS(model, config, device)

        # Factor library (session-level, shared across episodes).
        factor_library: Optional[FactorLibrary] = None
        if config.factor_library_enabled:
            factor_library = FactorLibrary(
                mod=config.mod,
                n_vars=config.n_variables,
                max_degree=config.effective_max_degree,
            )

        self.env = CircuitGame(config, factor_library=factor_library)
        self.factor_library = factor_library

        # Curriculum state.
        self.current_complexity = (
            config.starting_complexity if config.curriculum_enabled
            else config.max_complexity
        )
        self.success_history: List[bool] = []

        # Lazily-built BFS game boards keyed by complexity.
        self._boards: dict = {}

    def _get_board(self, complexity: int) -> dict:
        """Return (or build) the BFS game board for a given complexity.

        Args:
            complexity: Number of operations for polynomial enumeration.

        Returns:
            Game board dict as returned by build_game_board().
        """
        if complexity not in self._boards:
            self._boards[complexity] = build_game_board(self.config, complexity)
        return self._boards[complexity]

    def _get_temperature(self, step: int) -> float:
        """Compute MCTS temperature for action selection.

        Decays from temperature_init to temperature_final over the first
        temperature_decay_steps steps within an episode.  Higher temperature
        encourages exploration early in the episode.

        Args:
            step: Current step index within the episode (0-based).

        Returns:
            Temperature scalar (>= temperature_final).
        """
        decay_frac = min(step / max(self.config.temperature_decay_steps, 1), 1.0)
        return self.config.temperature_init + (
            self.config.temperature_final - self.config.temperature_init
        ) * decay_frac

    def collect_rollouts(self):
        """Collect trajectory data using MCTS for action selection.

        At each step the MCTS engine runs a full search from the current game
        state, producing a visit-count distribution π_MCTS.  An action is
        sampled from this distribution (with temperature), and the log
        probability under π_MCTS is recorded as the behaviour-policy log prob.

        The network's value head provides V(s) for GAE computation.

        Returns:
            Tuple (buffer, rollout_info) where:
              buffer: MCTSRolloutBuffer with collected transitions.
              rollout_info: Dict with 'episodes', 'success_rate', 'avg_reward',
                            'complexity', 'factor_hits', 'library_hits'.
        """
        buffer = MCTSRolloutBuffer()
        episodes_done = 0
        successes = 0
        total_rewards = 0.0
        factor_hits = 0
        library_hits = 0

        self.model.eval()

        while len(buffer) < self.config.steps_per_update:
            board = self._get_board(self.current_complexity)
            target_poly, _ = sample_target(self.config, self.current_complexity, board)
            obs = self.env.reset(target_poly)
            episode_reward = 0.0
            step = 0

            while not self.env.done:
                # Value estimate from the network (for GAE).
                obs_device = self._obs_to_device(obs)
                with torch.no_grad():
                    _, value = self.model(obs_device)

                # MCTS search → visit-count action distribution.
                temp = self._get_temperature(step)
                action, mcts_probs = self.mcts.get_action_probs(
                    self.env, temperature=temp
                )

                # Log prob of chosen action under MCTS behaviour policy.
                mcts_prob = mcts_probs[action]
                mcts_log_prob = float(np.log(max(mcts_prob, 1e-8)))

                next_obs, reward, done, info = self.env.step(action)

                buffer.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    mcts_log_prob=mcts_log_prob,
                    value=value.item(),
                    done=done,
                )

                episode_reward += reward
                obs = next_obs
                step += 1

                if info.get("factor_hit", False):
                    factor_hits += 1
                if info.get("library_hit", False):
                    library_hits += 1

            episodes_done += 1
            total_rewards += episode_reward
            if info.get("is_success", False):
                successes += 1
            self.success_history.append(info.get("is_success", False))

        self.model.train()

        success_rate = successes / max(episodes_done, 1)
        avg_reward = total_rewards / max(episodes_done, 1)

        rollout_info = {
            "episodes": episodes_done,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "complexity": self.current_complexity,
            "factor_hits": factor_hits,
            "library_hits": library_hits,
            "library_size": len(self.factor_library) if self.factor_library else 0,
        }
        return buffer, rollout_info

    def compute_gae(self, buffer: MCTSRolloutBuffer):
        """Compute Generalised Advantage Estimation (GAE).

        Identical to the PPO trainer's GAE computation.  Uses the network's
        value estimates (not MCTS values) for bootstrapping.

        Args:
            buffer: Filled MCTSRolloutBuffer.

        Returns:
            Tuple (advantages, returns) as float32 numpy arrays of shape (N,).
        """
        steps = buffer.steps
        n = len(steps)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        last_value = 0.0

        for t in reversed(range(n)):
            if steps[t].done:
                next_value = 0.0
                last_gae = 0.0
            elif t + 1 < n:
                next_value = steps[t + 1].value
            else:
                next_value = last_value

            delta = steps[t].reward + self.config.gamma * next_value - steps[t].value
            last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + steps[t].value

        return advantages, returns

    def update(
        self,
        buffer: MCTSRolloutBuffer,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict:
        """Run the PPO clipped surrogate update over MCTS-collected data.

        The key difference from standard PPO: the "old" log probabilities come
        from the MCTS visit-count distribution (the behaviour policy), not from
        the network's own policy at collection time.  The importance ratio

            r(θ) = π_θ(a|s) / π_MCTS(a|s)

        drives the network toward the MCTS-improved policy while the clipping
        ensures conservative steps.

        Args:
            buffer: MCTSRolloutBuffer with collected transitions.
            advantages: GAE advantage estimates, shape (N,).
            returns: Target returns, shape (N,).

        Returns:
            Dict with mean 'pg_loss', 'vf_loss', 'entropy' over all updates.
        """
        steps = buffer.steps
        n = len(steps)

        # Normalise advantages.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(
            [s.mcts_log_prob for s in steps], dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [s.action for s in steps], dtype=torch.long, device=self.device
        )

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(self.config.ppo_epochs):
            indices = np.random.permutation(n)

            for start in range(0, n, self.config.batch_size):
                end = min(start + self.config.batch_size, n)
                batch_idx = indices[start:end]

                batch_logits = []
                batch_values = []
                for idx in batch_idx:
                    obs_device = self._obs_to_device(steps[idx].obs)
                    logits, value = self.model(obs_device)
                    batch_logits.append(logits.squeeze(0))
                    batch_values.append(value.squeeze(0))

                batch_logits = torch.stack(batch_logits)
                batch_values = torch.stack(batch_values)
                batch_actions = actions[batch_idx]
                batch_adv = adv_tensor[batch_idx]
                batch_ret = ret_tensor[batch_idx]
                batch_old_lp = old_log_probs[batch_idx]

                # New log probs under the current network policy π_θ.
                dist = torch.distributions.Categorical(logits=batch_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Importance ratio: π_θ(a|s) / π_MCTS(a|s).
                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = (
                    torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip)
                    * batch_adv
                )
                pg_loss = -torch.min(surr1, surr2).mean()

                vf_loss = nn.functional.mse_loss(batch_values, batch_ret)

                loss = (
                    pg_loss
                    + self.config.vf_coef * vf_loss
                    - self.config.ent_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        return {
            "pg_loss": total_pg_loss / max(num_updates, 1),
            "vf_loss": total_vf_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def _maybe_advance_curriculum(self) -> None:
        """Check the recent success rate and adjust curriculum complexity.

        Uses a sliding window of the last 50 episode outcomes.  If the success
        rate exceeds the advance threshold, complexity is incremented.  If it
        falls below the backoff threshold, complexity is decremented.
        """
        if not self.config.curriculum_enabled:
            return

        window = 50
        if len(self.success_history) < window:
            return

        recent = self.success_history[-window:]
        rate = sum(recent) / len(recent)

        if (
            rate >= self.config.advance_threshold
            and self.current_complexity < self.config.max_complexity
        ):
            self.current_complexity += 1
            self.success_history.clear()
            print(f"[Curriculum] Advanced to complexity {self.current_complexity}")
        elif (
            rate <= self.config.backoff_threshold
            and self.current_complexity > self.config.starting_complexity
        ):
            self.current_complexity -= 1
            self.success_history.clear()
            print(f"[Curriculum] Backed off to complexity {self.current_complexity}")

    def train(self, num_iterations: int) -> dict:
        """Run the full PPO+MCTS training loop.

        Each iteration consists of:
          1. Collecting steps_per_update transitions using MCTS action selection.
          2. Computing GAE advantages and returns.
          3. Running ppo_epochs mini-batch PPO updates.
          4. Adjusting curriculum complexity.
          5. Logging a summary every log_interval iterations.

        Note: each collected step requires a full MCTS search (mcts_simulations
        forward passes), so this is significantly slower than plain PPO.
        Consider reducing steps_per_update or mcts_simulations for faster
        iteration on CPU.

        Args:
            num_iterations: Total number of collect + update cycles to run.

        Returns:
            Dict of metric lists keyed by name, each of length num_iterations.
            Keys: 'pg_loss', 'vf_loss', 'entropy', 'success_rate',
                  'avg_reward', 'complexity'.
        """
        history = {
            "pg_loss": [], "vf_loss": [], "entropy": [],
            "success_rate": [], "avg_reward": [], "complexity": [],
        }

        for iteration in range(1, num_iterations + 1):
            buffer, rollout_info = self.collect_rollouts()
            advantages, returns = self.compute_gae(buffer)
            loss_info = self.update(buffer, advantages, returns)
            self._maybe_advance_curriculum()

            history["pg_loss"].append(loss_info["pg_loss"])
            history["vf_loss"].append(loss_info["vf_loss"])
            history["entropy"].append(loss_info["entropy"])
            history["success_rate"].append(rollout_info["success_rate"])
            history["avg_reward"].append(rollout_info["avg_reward"])
            history["complexity"].append(rollout_info["complexity"])

            if iteration % self.config.log_interval == 0:
                lib_str = (
                    f"lib={rollout_info['library_size']} "
                    f"fhits={rollout_info['factor_hits']} "
                    f"lhits={rollout_info['library_hits']} "
                    if self.config.factor_library_enabled else ""
                )
                print(
                    f"[PPO+MCTS iter {iteration}] "
                    f"complexity={rollout_info['complexity']} "
                    f"episodes={rollout_info['episodes']} "
                    f"success={rollout_info['success_rate']:.2%} "
                    f"reward={rollout_info['avg_reward']:.3f} "
                    f"{lib_str}"
                    f"pg_loss={loss_info['pg_loss']:.4f} "
                    f"vf_loss={loss_info['vf_loss']:.4f} "
                    f"entropy={loss_info['entropy']:.4f}"
                )

        return history

    def _obs_to_device(self, obs: dict) -> dict:
        """Move all tensor values in an observation dict to the training device.

        Handles plain tensors, nested dicts of tensors, and PyG Data objects.

        Args:
            obs: Observation dict with keys 'graph', 'target', 'mask'.

        Returns:
            New dict with all tensors on self.device.
        """
        result = {}
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                result[key] = val.to(self.device)
            elif isinstance(val, dict):
                result[key] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in val.items()
                }
            else:
                if hasattr(val, "to"):
                    result[key] = val.to(self.device)
                else:
                    result[key] = val
        return result
