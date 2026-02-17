"""PPO (Proximal Policy Optimization) training loop."""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..config import Config
from ..models.policy_value_net import PolicyValueNet
from ..environment.circuit_game import CircuitGame
from ..game_board.generator import sample_target, build_game_board


@dataclass
class RolloutStep:
    """Single step in a rollout."""
    obs: dict
    action: int
    reward: float
    log_prob: float
    value: float
    done: bool


class RolloutBuffer:
    """Buffer for collecting rollout data."""

    def __init__(self):
        self.steps: List[RolloutStep] = []

    def add(self, obs, action, reward, log_prob, value, done):
        self.steps.append(RolloutStep(obs, action, reward, log_prob, value, done))

    def clear(self):
        self.steps = []

    def __len__(self):
        return len(self.steps)


class PPOTrainer:
    """PPO training with curriculum learning."""

    def __init__(self, config: Config, model: PolicyValueNet, device: str = "cpu"):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=config.ppo_lr)
        self.env = CircuitGame(config)

        # Curriculum state
        self.current_complexity = config.starting_complexity if config.curriculum_enabled else config.max_complexity
        self.success_history: List[bool] = []

        # Prebuilt game boards per complexity (lazy)
        self._boards = {}

    def _get_board(self, complexity: int):
        if complexity not in self._boards:
            self._boards[complexity] = build_game_board(self.config, complexity)
        return self._boards[complexity]

    def collect_rollouts(self) -> RolloutBuffer:
        """Run policy in environment, collect trajectory data."""
        buffer = RolloutBuffer()
        episodes_done = 0
        successes = 0
        total_rewards = 0.0

        while len(buffer) < self.config.steps_per_update:
            # Sample a target
            board = self._get_board(self.current_complexity)
            target_poly, _ = sample_target(self.config, self.current_complexity, board)
            obs = self.env.reset(target_poly)
            episode_reward = 0.0

            while not self.env.done:
                # Move obs to device
                obs_device = self._obs_to_device(obs)

                with torch.no_grad():
                    action, log_prob, _, value = self.model.get_action_and_value(obs_device)

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

            episodes_done += 1
            total_rewards += episode_reward
            if info.get("is_success", False):
                successes += 1
            self.success_history.append(info.get("is_success", False))

        success_rate = successes / max(episodes_done, 1)
        avg_reward = total_rewards / max(episodes_done, 1)
        return buffer, {
            "episodes": episodes_done,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "complexity": self.current_complexity,
        }

    def compute_gae(self, buffer: RolloutBuffer):
        """Compute Generalized Advantage Estimation.

        Returns:
            (advantages, returns) as numpy arrays
        """
        steps = buffer.steps
        n = len(steps)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        last_value = 0.0  # Bootstrap value for terminal states

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

    def update(self, buffer: RolloutBuffer, advantages: np.ndarray, returns: np.ndarray):
        """PPO clipped surrogate update.

        Returns:
            dict of loss components
        """
        steps = buffer.steps
        n = len(steps)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
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
            # Shuffle indices
            indices = np.random.permutation(n)

            for start in range(0, n, self.config.batch_size):
                end = min(start + self.config.batch_size, n)
                batch_idx = indices[start:end]

                # Get current policy outputs for batch
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

                # New log probs and entropy
                dist = torch.distributions.Categorical(logits=batch_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO clipped surrogate loss
                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * batch_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                vf_loss = nn.functional.mse_loss(batch_values, batch_ret)

                # Total loss
                loss = pg_loss + self.config.vf_coef * vf_loss - self.config.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
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

    def _maybe_advance_curriculum(self):
        """Check and adjust curriculum complexity."""
        if not self.config.curriculum_enabled:
            return

        window = 50
        if len(self.success_history) < window:
            return

        recent = self.success_history[-window:]
        rate = sum(recent) / len(recent)

        if rate >= self.config.advance_threshold and self.current_complexity < self.config.max_complexity:
            self.current_complexity += 1
            self.success_history.clear()
            print(f"[Curriculum] Advanced to complexity {self.current_complexity}")
        elif rate <= self.config.backoff_threshold and self.current_complexity > self.config.starting_complexity:
            self.current_complexity -= 1
            self.success_history.clear()
            print(f"[Curriculum] Backed off to complexity {self.current_complexity}")

    def train(self, num_iterations: int):
        """Main PPO training loop.

        Args:
            num_iterations: number of collect+update cycles
        """
        for iteration in range(1, num_iterations + 1):
            # Collect rollouts
            buffer, rollout_info = self.collect_rollouts()

            # Compute GAE
            advantages, returns = self.compute_gae(buffer)

            # Update policy
            loss_info = self.update(buffer, advantages, returns)

            # Curriculum
            self._maybe_advance_curriculum()

            # Logging
            if iteration % self.config.log_interval == 0:
                print(
                    f"[PPO iter {iteration}] "
                    f"complexity={rollout_info['complexity']} "
                    f"success={rollout_info['success_rate']:.2%} "
                    f"reward={rollout_info['avg_reward']:.3f} "
                    f"pg_loss={loss_info['pg_loss']:.4f} "
                    f"vf_loss={loss_info['vf_loss']:.4f} "
                    f"entropy={loss_info['entropy']:.4f}"
                )

    def _obs_to_device(self, obs: dict) -> dict:
        """Move observation tensors to device."""
        result = {}
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                result[key] = val.to(self.device)
            elif isinstance(val, dict):
                result[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                               for k, v in val.items()}
            else:
                # PyG Data or other
                if hasattr(val, 'to'):
                    result[key] = val.to(self.device)
                else:
                    result[key] = val
        return result
