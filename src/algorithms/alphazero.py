"""AlphaZero self-play training loop."""

import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..config import Config
from ..models.policy_value_net import PolicyValueNet
from ..environment.circuit_game import CircuitGame
from ..game_board.generator import sample_target, build_game_board
from .mcts import MCTS


class ReplayBuffer:
    """Fixed-size replay buffer for self-play data."""

    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, obs: dict, policy_target: np.ndarray, value_target: float):
        self.buffer.append((obs, policy_target, value_target))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class AlphaZeroTrainer:
    """AlphaZero training: self-play with MCTS + neural network training."""

    def __init__(self, config: Config, model: PolicyValueNet, device: str = "cpu"):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=config.az_lr, weight_decay=1e-4)
        self.replay_buffer = ReplayBuffer(config.az_buffer_size)
        self.mcts = MCTS(model, config, device)
        self.env = CircuitGame(config)

        # Curriculum state
        self.current_complexity = config.starting_complexity if config.curriculum_enabled else config.max_complexity
        self.success_history: List[bool] = []

        # Game boards (lazy)
        self._boards = {}

    def _get_board(self, complexity: int):
        if complexity not in self._boards:
            self._boards[complexity] = build_game_board(self.config, complexity)
        return self._boards[complexity]

    def _get_temperature(self, step: int, total_steps: int) -> float:
        """Temperature schedule: decay from init to final."""
        decay_frac = min(step / max(self.config.temperature_decay_steps, 1), 1.0)
        return self.config.temperature_init + (
            self.config.temperature_final - self.config.temperature_init
        ) * decay_frac

    def self_play_game(self, target_poly) -> Tuple[List[Tuple], bool]:
        """Play one game using MCTS.

        Returns:
            (trajectory, success) where trajectory is list of (obs, policy, None)
            The value targets are filled in after the game ends.
        """
        obs = self.env.reset(target_poly)
        trajectory = []
        step = 0

        while not self.env.done:
            temp = self._get_temperature(step, self.config.max_steps)
            action, probs = self.mcts.get_action_probs(self.env, temperature=temp)

            trajectory.append((obs, probs))

            obs, reward, done, info = self.env.step(action)
            step += 1

        success = info.get("is_success", False)
        return trajectory, success

    def generate_self_play_data(self) -> dict:
        """Run multiple self-play games and add data to replay buffer.

        Returns:
            Stats dict
        """
        successes = 0
        total_games = self.config.az_games_per_iter

        for game_idx in range(total_games):
            board = self._get_board(self.current_complexity)
            target_poly, _ = sample_target(self.config, self.current_complexity, board)

            trajectory, success = self.self_play_game(target_poly)
            successes += int(success)
            self.success_history.append(success)

            # Assign value targets: +1 for success, -1 for failure
            value_target = 1.0 if success else -1.0

            for obs, policy in trajectory:
                self.replay_buffer.add(obs, policy, value_target)

        success_rate = successes / max(total_games, 1)
        return {
            "games": total_games,
            "success_rate": success_rate,
            "buffer_size": len(self.replay_buffer),
            "complexity": self.current_complexity,
        }

    def train_on_buffer(self) -> dict:
        """Train network on replay buffer data.

        Loss = MSE(v, z) + CrossEntropy(p, pi)

        Returns:
            Loss stats dict
        """
        if len(self.replay_buffer) < self.config.az_batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_updates = 0

        for epoch in range(self.config.az_training_epochs):
            batch = self.replay_buffer.sample(self.config.az_batch_size)

            policy_losses = []
            value_losses = []

            for obs, policy_target, value_target in batch:
                obs_device = self._obs_to_device(obs)

                logits, value = self.model(obs_device)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                # Policy loss: cross-entropy with MCTS policy
                policy_target_tensor = torch.tensor(
                    policy_target, dtype=torch.float32, device=self.device
                )
                policy_loss = -(policy_target_tensor * log_probs).sum()
                policy_losses.append(policy_loss)

                # Value loss: MSE
                value_target_tensor = torch.tensor(
                    value_target, dtype=torch.float32, device=self.device
                )
                value_loss = (value.squeeze() - value_target_tensor) ** 2
                value_losses.append(value_loss)

            # Average losses over batch
            batch_policy_loss = torch.stack(policy_losses).mean()
            batch_value_loss = torch.stack(value_losses).mean()
            loss = batch_policy_loss + batch_value_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += batch_policy_loss.item()
            total_value_loss += batch_value_loss.item()
            num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "total_loss": (total_policy_loss + total_value_loss) / max(num_updates, 1),
        }

    def _maybe_advance_curriculum(self):
        """Check and adjust curriculum complexity."""
        if not self.config.curriculum_enabled:
            return

        window = self.config.az_games_per_iter
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
        """AlphaZero main training loop.

        Each iteration:
        1. Self-play N games with MCTS
        2. Train network on replay buffer
        3. Adjust curriculum
        """
        for iteration in range(1, num_iterations + 1):
            # Self-play phase
            self.model.eval()
            play_info = self.generate_self_play_data()

            # Training phase
            self.model.train()
            loss_info = self.train_on_buffer()

            # Curriculum
            self._maybe_advance_curriculum()

            # Logging
            if iteration % self.config.log_interval == 0:
                print(
                    f"[AZ iter {iteration}] "
                    f"complexity={play_info['complexity']} "
                    f"success={play_info['success_rate']:.2%} "
                    f"buffer={play_info['buffer_size']} "
                    f"p_loss={loss_info['policy_loss']:.4f} "
                    f"v_loss={loss_info['value_loss']:.4f}"
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
                if hasattr(val, 'to'):
                    result[key] = val.to(self.device)
                else:
                    result[key] = val
        return result
