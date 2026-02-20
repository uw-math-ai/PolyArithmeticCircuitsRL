"""Monte Carlo Tree Search with Q-network priors (AlphaZero style).

Uses the DQN Q-network both as a policy prior (softmax over Q-values)
and as a leaf evaluator (max Q-value among valid actions).  No rollouts
are performed — the Q-network provides the value estimate directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from ..config import Config
    from ..env.circuit_env import PolyCircuitEnv
    from .agent import DQNAgent


class MCTSNode:
    """A node in the MCTS search tree."""

    __slots__ = (
        "state", "obs", "action_mask", "parent", "parent_action",
        "children", "visit_count", "value_sum", "prior",
        "is_terminal", "terminal_value",
    )

    def __init__(
        self,
        state: Dict,
        obs: np.ndarray,
        action_mask: np.ndarray,
        parent: Optional[MCTSNode] = None,
        parent_action: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.obs = obs
        self.action_mask = action_mask
        self.parent = parent
        self.parent_action = parent_action
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_terminal = False
        self.terminal_value = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """MCTS search using Q-network for priors and leaf evaluation."""

    def __init__(self, agent: DQNAgent, env: PolyCircuitEnv, config: Config):
        self.agent = agent
        self.env = env
        self.num_simulations = config.mcts_simulations
        self.c_puct = config.mcts_c_puct
        self.temperature = config.mcts_temperature

    def search(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Run MCTS from the current env state and return the best action."""
        root_state = self.env.get_state()
        root = MCTSNode(state=root_state, obs=obs, action_mask=action_mask)

        # Expand root
        self._expand(root)

        for _ in range(self.num_simulations):
            node = root

            # SELECT: walk tree using PUCT until we find an unexpanded node
            while node.is_expanded() and not node.is_terminal:
                node = self._select_child(node)

            # EVALUATE
            if node.is_terminal:
                value = node.terminal_value
            elif not node.is_expanded():
                # EXPAND and EVALUATE leaf
                value = self._expand(node)
            else:
                value = node.q_value

            # BACKUP
            self._backup(node, value)

        return self._select_action(root)

    def _get_priors(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        """Compute action priors from Q-network via softmax over valid Q-values."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
            q = self.agent.q_network(obs_t).squeeze(0).cpu().numpy()

        valid = action_mask > 0
        priors = np.zeros_like(q)
        if valid.any():
            q_valid = q[valid]
            # Numerically stable softmax
            q_shifted = q_valid - q_valid.max()
            exp_q = np.exp(q_shifted)
            priors[valid] = exp_q / exp_q.sum()
        return priors

    def _evaluate_leaf(self, obs: np.ndarray, action_mask: np.ndarray) -> float:
        """Evaluate a leaf node using max Q-value among valid actions."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
            q = self.agent.q_network(obs_t).squeeze(0).cpu().numpy()

        valid = action_mask > 0
        if not valid.any():
            return 0.0
        return float(q[valid].max())

    def _expand(self, node: MCTSNode) -> float:
        """Expand a node: create children for all valid actions, return leaf value."""
        priors = self._get_priors(node.obs, node.action_mask)
        valid_actions = np.where(node.action_mask > 0)[0]

        for action in valid_actions:
            # Simulate action
            self.env.set_state(node.state)
            obs_dict, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            child_state = self.env.get_state()
            child = MCTSNode(
                state=child_state,
                obs=obs_dict["obs"],
                action_mask=obs_dict["action_mask"],
                parent=node,
                parent_action=action,
                prior=priors[action],
            )

            if done:
                child.is_terminal = True
                # Terminal value: reward from this step
                # For solved: reward includes +1.0 bonus
                child.terminal_value = reward
            node.children[action] = child

        # Return leaf value estimate for this node
        return self._evaluate_leaf(node.obs, node.action_mask)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child using PUCT formula."""
        best_score = -float("inf")
        best_child = None
        sqrt_parent = math.sqrt(node.visit_count)

        for child in node.children.values():
            if child.is_terminal:
                score = child.terminal_value + self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            else:
                score = child.q_value + self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Propagate value up the tree."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def _select_action(self, root: MCTSNode) -> int:
        """Select action from root based on visit counts."""
        visits = np.array([
            root.children[a].visit_count if a in root.children else 0
            for a in range(len(root.action_mask))
        ], dtype=np.float64)

        if self.temperature == 0:
            # Greedy
            return int(np.argmax(visits))

        # Temperature-scaled sampling
        valid = visits > 0
        if not valid.any():
            # Fallback: pick random valid action
            valid_actions = np.where(root.action_mask > 0)[0]
            return int(np.random.choice(valid_actions))

        visits_temp = np.zeros_like(visits)
        visits_temp[valid] = visits[valid] ** (1.0 / self.temperature)
        probs = visits_temp / visits_temp.sum()
        return int(np.random.choice(len(probs), p=probs))
