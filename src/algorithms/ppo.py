"""PPO (Proximal Policy Optimization) training loop.

Implements on-policy training with:
  - Generalised Advantage Estimation (GAE, Schulman et al. 2016)
  - Clipped surrogate objective
  - Adaptive curriculum learning
  - Optional factor library and subgoal rewards (see FactorLibrary)

All three algorithms (PPO, SAC, AlphaZero) share the same CircuitGame
environment and Config dataclass; only the training loop differs.
"""

from dataclasses import dataclass, field
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


@dataclass
class RolloutStep:
    """Container for a single (s, a, r, log_pi, V, done) transition.

    Attributes:
        obs: Observation dict at time t (graph, target, mask).
        action: Integer action index selected at time t.
        reward: Scalar reward received after taking the action.
        log_prob: Log probability of the selected action under the behaviour policy.
        value: Baseline value estimate V(s_t) from the value head.
        done: True if the episode ended at this step (success or timeout).
    """
    obs: dict
    action: int
    reward: float
    log_prob: float
    value: float
    done: bool


class RolloutBuffer:
    """Ordered buffer of RolloutStep objects collected during policy rollout.

    Filled by collect_rollouts() and consumed once during update(). The buffer
    is cleared after each PPO update.
    """

    def __init__(self) -> None:
        """Initialise an empty rollout buffer."""
        self.steps: List[RolloutStep] = []

    def add(
        self,
        obs: dict,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        """Append a single transition to the buffer.

        Args:
            obs: Observation dict from the environment at this timestep.
            action: Integer action index that was selected.
            reward: Immediate scalar reward received.
            log_prob: Log probability of the action under the behaviour policy.
            value: Value estimate V(s) from the critic at this state.
            done: Whether the episode terminated at this step.
        """
        self.steps.append(RolloutStep(obs, action, reward, log_prob, value, done))

    def clear(self) -> None:
        """Empty the buffer (called after each PPO update)."""
        self.steps = []

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return len(self.steps)


class PPOTrainer:
    """PPO training loop with curriculum learning and factor library support.

    Orchestrates the collect → advantage-estimate → update cycle. An adaptive
    curriculum adjusts the target polynomial complexity based on the recent
    success rate.

    The optional FactorLibrary is created here (once per training run) and
    shared with the CircuitGame environment. It accumulates knowledge across
    episodes within the session without persisting to disk.

    Attributes:
        config (Config): Shared hyperparameter configuration.
        model (PolicyValueNet): The policy-value network being trained.
        device (str): PyTorch device string ('cpu', 'cuda', 'mps').
        env (CircuitGame): The circuit construction environment instance.
        current_complexity (int): Current curriculum complexity level.
        success_history (List[bool]): Rolling history of episode outcomes.
    """

    def __init__(
        self, config: Config, model: PolicyValueNet, device: str = "cpu"
    ) -> None:
        """Initialise the PPO trainer.

        Creates the environment (with an optional FactorLibrary if enabled),
        the Adam optimiser, and the curriculum state.

        Args:
            config: Configuration dataclass with all hyperparameters.
            model: Shared policy-value network to train.
            device: PyTorch device ('cpu', 'cuda', or 'mps').
        """
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=config.ppo_lr)

        # Create the factor library once per training session if the feature is
        # enabled. The library lives here and is passed to CircuitGame so that
        # it persists across all episodes within this run.
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

        # Lazily-built game boards keyed by complexity level (BFS DAG of reachable
        # polynomials). Built on first access to avoid paying the BFS cost upfront.
        self._boards = {}

    def _get_board(self, complexity: int) -> dict:
        """Return (or build) the BFS game board for a given complexity level.

        The board is a dict mapping canonical polynomial key -> entry dict
        with keys 'poly', 'step', 'parents', 'paths'. It is cached after
        the first build to avoid redundant BFS traversals.

        Args:
            complexity: Number of operations up to which to enumerate polynomials.

        Returns:
            Game board dict as returned by build_game_board().
        """
        if complexity not in self._boards:
            self._boards[complexity] = build_game_board(self.config, complexity)
        return self._boards[complexity]

    def collect_rollouts(self):
        """Run the current policy in the environment and collect trajectory data.

        Runs episodes until at least steps_per_update transitions have been
        collected. Each complete episode uses a freshly sampled target polynomial
        from the BFS board at the current curriculum complexity.

        Returns:
            Tuple (buffer, rollout_info) where:
              buffer: RolloutBuffer containing all collected transitions.
              rollout_info: Dict with 'episodes', 'success_rate', 'avg_reward',
                            'complexity', 'factor_hits', 'library_hits'.
        """
        buffer = RolloutBuffer()
        episodes_done = 0
        successes = 0
        total_rewards = 0.0
        factor_hits = 0    # Number of steps where a factor subgoal was hit.
        library_hits = 0   # Subset of factor_hits where the factor was library-known.

        while len(buffer) < self.config.steps_per_update:
            # Sample a target polynomial from the board at the current complexity.
            board = self._get_board(self.current_complexity)
            target_poly, _ = sample_target(self.config, self.current_complexity, board)
            obs = self.env.reset(target_poly)
            episode_reward = 0.0

            while not self.env.done:
                obs_device = self._obs_to_device(obs)

                with torch.no_grad():
                    action, log_prob, _, value = self.model.get_action_and_value(
                        obs_device
                    )

                action_int = action.item()
                next_obs, reward, done, info = self.env.step(action_int)

                buffer.add(
                    obs=obs,
                    action=action_int,
                    reward=reward,
                    log_prob=log_prob.item(),
                    value=value.item(),
                    done=done,
                )

                episode_reward += reward
                obs = next_obs

                # Track factor subgoal statistics for logging.
                if info.get("factor_hit", False):
                    factor_hits += 1
                if info.get("library_hit", False):
                    library_hits += 1

            episodes_done += 1
            total_rewards += episode_reward
            if info.get("is_success", False):
                successes += 1
            self.success_history.append(info.get("is_success", False))

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

    def compute_gae(self, buffer: RolloutBuffer):
        """Compute Generalised Advantage Estimation (GAE) for all buffered steps.

        GAE smoothly interpolates between Monte Carlo returns (lambda=1) and
        TD(0) (lambda=0). The backward recursion is:

            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t = delta_t + (gamma * lambda) * A_{t+1}   (A_T = 0 at episode end)

        Returns are computed as: R_t = A_t + V(s_t).

        Args:
            buffer: Filled RolloutBuffer with at least one step.

        Returns:
            Tuple (advantages, returns) as float32 numpy arrays of shape (N,).
        """
        steps = buffer.steps
        n = len(steps)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        last_value = 0.0  # Bootstrap value is 0 for terminal steps.

        for t in reversed(range(n)):
            if steps[t].done:
                next_value = 0.0
                last_gae = 0.0  # Reset GAE accumulator at episode boundaries.
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
        buffer: RolloutBuffer,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict:
        """Run the PPO clipped surrogate update over the collected buffer.

        Normalises advantages, then runs ppo_epochs passes over mini-batches of
        size batch_size. Each mini-batch computes new log probabilities, the
        clipped policy gradient loss, value function loss, and entropy bonus.

        PPO clipped objective:
            L^CLIP = -E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
        where r_t = pi_theta(a|s) / pi_old(a|s).

        Total loss:
            L = L^CLIP + c_v * L^VF - c_e * H[pi]

        Args:
            buffer: Filled RolloutBuffer (consumed but not cleared here).
            advantages: GAE advantage estimates, shape (N,).
            returns: Target returns R_t = A_t + V(s_t), shape (N,).

        Returns:
            Dict with mean 'pg_loss', 'vf_loss', 'entropy' over all update steps.
        """
        steps = buffer.steps
        n = len(steps)

        # Normalise advantages to unit variance for stable policy gradient updates.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(
            [s.log_prob for s in steps], dtype=torch.float32, device=self.device
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

                # Forward pass for each observation in the mini-batch.
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

                # New log probabilities and entropy under updated policy.
                dist = torch.distributions.Categorical(logits=batch_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO clipped surrogate policy loss.
                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = (
                    torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip)
                    * batch_adv
                )
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value function MSE loss.
                vf_loss = nn.functional.mse_loss(batch_values, batch_ret)

                # Combined loss: policy gradient - entropy bonus + value loss.
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

        Uses a sliding window of the last `window` episode outcomes. If the
        success rate exceeds the advance threshold, complexity is incremented.
        If it falls below the backoff threshold, complexity is decremented.
        History is cleared after each transition to allow the agent to adapt.

        Does nothing if curriculum_enabled is False in the config.
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

    def train(self, num_iterations: int) -> None:
        """Run the full PPO training loop for the specified number of iterations.

        Each iteration consists of:
          1. Collecting steps_per_update environment transitions.
          2. Computing GAE advantages and returns.
          3. Running ppo_epochs mini-batch updates.
          4. Adjusting curriculum complexity.
          5. Logging a summary every log_interval iterations.

        Args:
            num_iterations: Total number of collect + update cycles to run.
        """
        for iteration in range(1, num_iterations + 1):
            buffer, rollout_info = self.collect_rollouts()
            advantages, returns = self.compute_gae(buffer)
            loss_info = self.update(buffer, advantages, returns)
            self._maybe_advance_curriculum()

            if iteration % self.config.log_interval == 0:
                lib_str = (
                    f"lib={rollout_info['library_size']} "
                    f"fhits={rollout_info['factor_hits']} "
                    f"lhits={rollout_info['library_hits']} "
                    if self.config.factor_library_enabled else ""
                )
                print(
                    f"[PPO iter {iteration}] "
                    f"complexity={rollout_info['complexity']} "
                    f"success={rollout_info['success_rate']:.2%} "
                    f"reward={rollout_info['avg_reward']:.3f} "
                    f"{lib_str}"
                    f"pg_loss={loss_info['pg_loss']:.4f} "
                    f"vf_loss={loss_info['vf_loss']:.4f} "
                    f"entropy={loss_info['entropy']:.4f}"
                )

    def _obs_to_device(self, obs: dict) -> dict:
        """Move all tensor values in an observation dict to the training device.

        Handles three cases: plain tensors, nested dicts of tensors, and PyG
        Data objects (which have a .to() method).

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
                # PyG Data object or similar — use .to() if available.
                if hasattr(val, "to"):
                    result[key] = val.to(self.device)
                else:
                    result[key] = val
        return result
