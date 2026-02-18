from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..config import Config
from .network import CircuitTransformerQ
from .replay_buffer import HERReplayBuffer


class DQNAgent:
    """
    Double DQN agent with action masking and epsilon-greedy exploration.
    """

    def __init__(self, config: Config, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)

        self.q_network = CircuitTransformerQ(config).to(self.device)
        self.target_network = CircuitTransformerQ(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.buffer = HERReplayBuffer(config)

        self.total_steps = 0
        self.training_losses: list[float] = []

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Select an action using epsilon-greedy policy with action masking.

        Invalid actions (mask == 0) are pushed to -1e9 before argmax so the
        agent always picks from the valid set.

        Args:
            obs:          Flat observation vector, shape (obs_dim,).
            action_mask:  Binary mask, shape (action_dim,); 1 = valid action.
            deterministic: If True, use greedy policy (eps=0).

        Returns:
            Integer action index in [0, action_dim).
        """
        eps = self._epsilon() if not deterministic else 0.0

        if np.random.random() < eps:
            # Random exploration: sample uniformly from valid actions
            valid = np.where(action_mask > 0)[0]
            return int(np.random.choice(valid))

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_network(obs_t).squeeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
            q = q + (mask_t - 1.0) * 1e9  # push invalid actions to -1e9
            return int(q.argmax().item())

    def train_step(self) -> float:
        """Sample a batch from the buffer and perform one Double DQN update.

        Double DQN: the online network selects the next action (argmax),
        the target network evaluates it.  This decouples action selection
        from value estimation, reducing overestimation bias.

        Uses Huber loss (smooth_l1) and gradient clipping (norm <= 10).

        Returns:
            The training loss (0.0 if buffer is not yet large enough).
        """
        if len(self.buffer) < self.config.batch_size:
            return 0.0

        batch = self.buffer.sample(self.config.batch_size)

        obs = torch.tensor(batch["obs"], device=self.device)
        actions = torch.tensor(batch["actions"], device=self.device)
        rewards = torch.tensor(batch["rewards"], device=self.device)
        next_obs = torch.tensor(batch["next_obs"], device=self.device)
        dones = torch.tensor(batch["dones"], device=self.device)
        next_masks = torch.tensor(batch["next_action_masks"], device=self.device)

        # Q-value for the action actually taken in each transition
        q_all = self.q_network(obs)
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online network picks best action, target evaluates it
            next_q_online = self.q_network(next_obs)
            next_q_online = next_q_online + (next_masks - 1.0) * 1e9  # mask invalid
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)

            next_q_target = self.target_network(next_obs)
            next_q = next_q_target.gather(1, best_next_actions).squeeze(1)

            targets = rewards + self.config.gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        val = loss.item()
        self.training_losses.append(val)
        return val

    def soft_update_target(self) -> None:
        """Polyak-average the target network toward the online network.

        target_params <- tau * online_params + (1 - tau) * target_params

        A small tau (e.g. 0.005) keeps the target network stable, which
        reduces oscillation during training.
        """
        tau = self.config.target_update_tau
        for tp, op in zip(self.target_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)

    def _epsilon(self) -> float:
        """Return the current exploration rate, linearly annealed from eps_start to eps_end."""
        frac = min(1.0, self.total_steps / max(self.config.eps_decay_steps, 1))
        return self.config.eps_start + frac * (self.config.eps_end - self.config.eps_start)

    def save(self, path: str) -> None:
        """Save a full checkpoint (both networks + optimizer + step count) to path."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str) -> None:
        """Restore a checkpoint saved by save().  Also restores total_steps."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]
