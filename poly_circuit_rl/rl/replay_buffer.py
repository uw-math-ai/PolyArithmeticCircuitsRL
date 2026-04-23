from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import Config
from ..core.poly import PolyKey
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
    is_demo: bool = False
    base_reward: float = 0.0
    shaping_reward: float = 0.0
    solve_bonus: float = 0.0


class HERReplayBuffer:
    """
    Circular replay buffer with Hindsight Experience Replay.

    Relabeling uses the "future" strategy and is applied at insertion time.

    Factor library shaping rewards are STRIPPED from HER-relabeled transitions
    since they depend on the original target polynomial.  Only the solve bonus
    is relabeled.  This makes HER fully compatible with factor library rewards.
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
        ep_solved: List[bool],
        ep_truncated: List[bool],
        ep_masks: List[np.ndarray],
        ep_next_masks: List[np.ndarray],
        ep_achieved_goals: List[Optional[np.ndarray]],
        ep_achieved_goal_keys: Optional[List[Optional[PolyKey]]] = None,
        ep_base_rewards: Optional[List[float]] = None,
        ep_shaping_rewards: Optional[List[float]] = None,
        ep_solve_bonuses: Optional[List[float]] = None,
    ) -> None:
        """
        Store an episode with HER relabeling.

        For each transition t:
          1. Store original transition (with full reward: base + shaping + solve).
          2. Sample k future timesteps t' > t.
          3. Pick a future achieved goal from t' as the new goal.
          4. Rewrite ONLY the solve bonus under that goal (goal-vector match
             + exact achieved-goal key equality).
             Factor shaping rewards are stripped (set to 0) in relabeled transitions.
        """
        T = len(ep_obs)

        # Default decomposed rewards for backward compat
        if ep_base_rewards is None:
            ep_base_rewards = ep_rewards
        if ep_shaping_rewards is None:
            ep_shaping_rewards = [0.0] * T
        if ep_solve_bonuses is None:
            ep_solve_bonuses = [1.0 if ep_solved[t] else 0.0 for t in range(T)]
        if ep_achieved_goal_keys is None:
            ep_achieved_goal_keys = [None] * T

        for t in range(T):
            # Original transition (full reward)
            self.add(Transition(
                obs=ep_obs[t],
                action=ep_actions[t],
                reward=ep_rewards[t],
                next_obs=ep_next_obs[t],
                done=ep_dones[t],
                action_mask=ep_masks[t],
                next_action_mask=ep_next_masks[t],
                base_reward=ep_base_rewards[t],
                shaping_reward=ep_shaping_rewards[t],
                solve_bonus=ep_solve_bonuses[t],
            ))

        # HER relabeling: always enabled (no shaping_coeff guard)
        for t in range(T):
            future_indices = [
                ft for ft in range(t + 1, T)
                if ep_achieved_goals[ft] is not None
            ]
            if not future_indices:
                continue

            k = min(self.config.her_k, len(future_indices))
            selected = self.rng.sample(future_indices, k)

            for ft in selected:
                new_goal = ep_achieved_goals[ft]
                if new_goal is None:
                    continue

                new_obs = replace_goal(ep_obs[t], new_goal, self.config)
                new_next_obs = replace_goal(ep_next_obs[t], new_goal, self.config)

                next_achieved_goal_key = ep_achieved_goal_keys[t]
                new_goal_key = ep_achieved_goal_keys[ft]
                relabeled_solved = (
                    next_achieved_goal_key is not None
                    and new_goal_key is not None
                    and next_achieved_goal_key == new_goal_key
                )
                relabeled_solve_bonus = 1.0 if relabeled_solved else 0.0
                # Relabeled reward: base_reward + relabeled_solve_bonus
                # Factor shaping is STRIPPED (depends on original target)
                new_reward = ep_base_rewards[t] + relabeled_solve_bonus
                new_done = ep_truncated[t] or relabeled_solved

                self.add(Transition(
                    obs=new_obs,
                    action=ep_actions[t],
                    reward=new_reward,
                    next_obs=new_next_obs,
                    done=new_done,
                    action_mask=ep_masks[t],
                    next_action_mask=ep_next_masks[t],
                    base_reward=ep_base_rewards[t],
                    shaping_reward=0.0,  # stripped
                    solve_bonus=relabeled_solve_bonus,
                ))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a random batch of transitions.

        When expert demos exist, reserves 20% of the batch for demo transitions
        to ensure they remain represented during training.
        """
        n = min(batch_size, len(self.buffer))

        demo_indices = [i for i in range(len(self.buffer)) if self.buffer[i].is_demo]
        if demo_indices and len(demo_indices) >= n // 5:
            n_demo = n // 5
            n_regular = n - n_demo
            demo_sample = self.rng.sample(demo_indices, min(n_demo, len(demo_indices)))
            regular_indices = [i for i in range(len(self.buffer)) if not self.buffer[i].is_demo]
            if regular_indices:
                regular_sample = self.rng.sample(regular_indices, min(n_regular, len(regular_indices)))
            else:
                regular_sample = []
            indices = demo_sample + regular_sample
        else:
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

    def clear(self) -> None:
        """Drop all stored transitions (useful on curriculum advance)."""
        self.buffer.clear()
        self.pos = 0

    def prune_non_demos(self, keep_recent: int) -> int:
        """Keep all is_demo=True transitions + the last `keep_recent` non-demo.

        Returns the number of transitions dropped. Preserves insertion order so
        future pushes continue FIFO overwrite semantics.
        """
        demos = [t for t in self.buffer if t.is_demo]
        non_demos = [t for t in self.buffer if not t.is_demo]
        kept_non_demos = non_demos[-keep_recent:] if keep_recent > 0 else []
        dropped = len(self.buffer) - len(demos) - len(kept_non_demos)
        self.buffer = demos + kept_non_demos
        self.pos = len(self.buffer) % self.capacity
        return dropped

    def count_demos(self) -> int:
        return sum(1 for t in self.buffer if t.is_demo)

    def __len__(self) -> int:
        return len(self.buffer)
