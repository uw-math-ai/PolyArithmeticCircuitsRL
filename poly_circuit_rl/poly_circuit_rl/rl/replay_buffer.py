from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..config import Config
from ..env.obs import replace_goal


@dataclass
class Transition:
    obs: np.ndarray            # (obs_dim,)
    action: int
    reward: float
    next_obs: np.ndarray       # (obs_dim,)
    done: bool
    action_mask: np.ndarray    # (action_dim,)
    next_action_mask: np.ndarray


class HERReplayBuffer:
    """
    Circular replay buffer with Hindsight Experience Replay.

    Relabeling uses the "future" strategy and is applied at insertion time.
    """

    def __init__(self, config: Config):
        self.config = config
        self.capacity = config.buffer_size
        self.buffer: List[Transition] = []
        self.pos = 0
        self.rng = random.Random(config.seed + 1)

    def add(self, t: Transition) -> None:
        """Insert one transition.  Overwrites oldest entry when buffer is full."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(t)
        else:
            self.buffer[self.pos] = t
        self.pos = (self.pos + 1) % self.capacity

    def add_episode_with_her(
        self,
        ep_obs: List[np.ndarray],
        ep_actions: List[int],
        ep_rewards: List[float],
        ep_next_obs: List[np.ndarray],
        ep_dones: List[bool],
        ep_masks: List[np.ndarray],
        ep_next_masks: List[np.ndarray],
        ep_node_evals: List[List[np.ndarray]],
    ) -> None:
        """
        Store an episode with HER relabeling.

        For each transition t:
          1. Store original transition.
          2. Sample k future timesteps t' > t.
          3. Pick a random node eval from t' as new goal.
          4. Store relabeled transition with reward=+1 if any node
             at t+1 matches the new goal.
        """
        T = len(ep_obs)

        for t in range(T):
            # Original transition
            self.add(Transition(
                obs=ep_obs[t],
                action=ep_actions[t],
                reward=ep_rewards[t],
                next_obs=ep_next_obs[t],
                done=ep_dones[t],
                action_mask=ep_masks[t],
                next_action_mask=ep_next_masks[t],
            ))

            # HER relabeling
            future_indices = list(range(t + 1, T))
            if not future_indices:
                continue

            k = min(self.config.her_k, len(future_indices))
            selected = self.rng.sample(future_indices, k)

            for ft in selected:
                future_evals = ep_node_evals[ft]
                if not future_evals:
                    continue

                new_goal = self.rng.choice(future_evals)

                new_obs = replace_goal(ep_obs[t], new_goal, self.config)
                new_next_obs = replace_goal(ep_next_obs[t], new_goal, self.config)

                # Check if any node at t+1 matches the new goal
                next_evals = ep_node_evals[min(t + 1, T - 1)]
                new_reward = 0.0
                new_done = ep_dones[t]
                for ne in next_evals:
                    if np.allclose(ne, new_goal, rtol=0.0, atol=1e-6):
                        new_reward = 1.0
                        new_done = True
                        break

                self.add(Transition(
                    obs=new_obs,
                    action=ep_actions[t],
                    reward=new_reward,
                    next_obs=new_next_obs,
                    done=new_done,
                    action_mask=ep_masks[t],
                    next_action_mask=ep_next_masks[t],
                ))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a random batch of transitions.

        Returns a dict of stacked numpy arrays ready for training:
          obs, next_obs: (B, obs_dim)
          actions:       (B,) int64
          rewards:       (B,) float32
          dones:         (B,) float32 (1.0 = terminal)
          action_masks, next_action_masks: (B, action_dim) float32
        """
        n = min(batch_size, len(self.buffer))
        indices = self.rng.sample(range(len(self.buffer)), n)
        return {
            "obs": np.array([self.buffer[i].obs for i in indices], dtype=np.float32),
            "actions": np.array([self.buffer[i].action for i in indices], dtype=np.int64),
            "rewards": np.array([self.buffer[i].reward for i in indices], dtype=np.float32),
            "next_obs": np.array([self.buffer[i].next_obs for i in indices], dtype=np.float32),
            "dones": np.array([self.buffer[i].done for i in indices], dtype=np.float32),
            "action_masks": np.array([self.buffer[i].action_mask for i in indices], dtype=np.float32),
            "next_action_masks": np.array([self.buffer[i].next_action_mask for i in indices], dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.buffer)
