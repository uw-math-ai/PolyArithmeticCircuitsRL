from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..config import Config
from .network import CircuitTransformerQ
from .replay_buffer import HERReplayBuffer


def load_checkpoint_payload(
    path: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a saved checkpoint payload without constructing an agent."""
    return torch.load(path, map_location=map_location, weights_only=False)


def config_from_checkpoint_payload(payload: dict[str, Any]) -> Optional[Config]:
    """Reconstruct Config from a checkpoint payload when present."""
    config_data = payload.get("config")
    if config_data is None:
        return None
    return Config(**config_data)


class DQNAgent:
    """
    Double DQN agent with action masking and epsilon-greedy exploration.
    """

    def __init__(self, config: Config, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(config.seed + 17)

        # Ensure reproducible parameter initialization for a given config seed.
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.q_network = CircuitTransformerQ(config).to(self.device)
        self.target_network = CircuitTransformerQ(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.buffer = HERReplayBuffer(config)

        self.total_steps = 0
        self.training_losses: list[float] = []
        # When non-zero, shifts eps-schedule backwards so epsilon starts higher
        # and re-decays from that level. Set by bump_epsilon_floor() at curriculum
        # advance. Decays naturally as total_steps grows — no permanent floor.
        self._eps_step_offset: int = 0

    def predict_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Return Q-values for one observation using eval-mode inference."""
        was_training = self.q_network.training
        self.q_network.eval()
        try:
            with torch.no_grad():
                obs_t = torch.tensor(
                    obs,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                q = self.q_network(obs_t).squeeze(0).cpu().numpy()
        finally:
            if was_training:
                self.q_network.train()
        return q

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

        if self.rng.random() < eps:
            # Random exploration: sample uniformly from valid actions
            valid = np.where(action_mask > 0)[0]
            return int(self.rng.choice(valid))

        q = self.predict_q_values(obs)
        q = q + (action_mask.astype(np.float32) - 1.0) * 1e9  # push invalid actions to -1e9
        return int(np.argmax(q))

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

        self.q_network.train()
        self.target_network.eval()

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
        """Return the current exploration rate, linearly annealed from eps_start to eps_end.

        Respects `_eps_step_offset` so bump_epsilon_floor() effectively rewinds
        the schedule — eps jumps up and then re-decays naturally.
        """
        effective_steps = max(0, self.total_steps - self._eps_step_offset)
        frac = min(1.0, effective_steps / max(self.config.eps_decay_steps, 1))
        eps = self.config.eps_start + frac * (self.config.eps_end - self.config.eps_start)
        return max(eps, self.config.eps_end)

    def bump_epsilon_floor(self, new_floor: float) -> None:
        """Rewind the epsilon schedule so current eps becomes `new_floor`.

        After this call, epsilon decays from `new_floor` toward eps_end over the
        normal `eps_decay_steps` budget. If current eps is already >= new_floor,
        this is a no-op.

        Called at curriculum advance to force re-exploration on a new target
        distribution.
        """
        new_floor = float(max(self.config.eps_end, min(self.config.eps_start, new_floor)))
        if new_floor <= self._epsilon():
            return
        # Solve for effective_steps such that scheduled eps == new_floor:
        #   eps = eps_start + (effective/decay) * (eps_end - eps_start)
        #   => effective = decay * (new_floor - eps_start) / (eps_end - eps_start)
        denom = self.config.eps_end - self.config.eps_start
        if denom == 0:
            return
        frac = (new_floor - self.config.eps_start) / denom
        frac = max(0.0, min(1.0, frac))
        effective_steps = int(frac * self.config.eps_decay_steps)
        self._eps_step_offset = self.total_steps - effective_steps

    def save(self, path: str) -> None:
        """Save a full checkpoint (both networks + optimizer + step count) to path."""
        torch.save({
            "config": asdict(self.config),
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str) -> None:
        """Restore a checkpoint saved by save().  Also restores total_steps."""
        ckpt = load_checkpoint_payload(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]
