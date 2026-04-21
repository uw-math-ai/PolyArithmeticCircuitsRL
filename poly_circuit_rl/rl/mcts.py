"""Monte Carlo Tree Search with Q-network priors (AlphaZero style).

Uses the DQN Q-network both as a policy prior (softmax over Q-values)
and as a leaf evaluator (max Q-value among valid actions).  No rollouts
are performed — the Q-network provides the value estimate directly.

Lazy expansion: only one child is expanded per simulation (the most
promising unexpanded action by prior), avoiding the cost of simulating
all valid actions at every node.
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
        "children", "visit_count", "value_sum", "priors",
        "reward_from_parent",
        "is_terminal", "terminal_value", "valid_actions",
        "unexpanded_actions",
    )

    def __init__(
        self,
        state: Dict,
        obs: np.ndarray,
        action_mask: np.ndarray,
        parent: Optional[MCTSNode] = None,
        parent_action: Optional[int] = None,
        reward_from_parent: float = 0.0,
    ):
        self.state = state
        self.obs = obs
        self.action_mask = action_mask
        self.parent = parent
        self.parent_action = parent_action
        self.reward_from_parent = reward_from_parent
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.priors: Optional[np.ndarray] = None  # set on first visit
        self.is_terminal = False
        self.terminal_value = 0.0
        self.valid_actions: Optional[np.ndarray] = None
        self.unexpanded_actions: Optional[List[int]] = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_fully_expanded(self) -> bool:
        return self.unexpanded_actions is not None and len(self.unexpanded_actions) == 0


class MCTS:
    """MCTS search using Q-network for priors and leaf evaluation.

    Uses lazy expansion: each simulation expands at most one child node.
    """

    def __init__(self, agent: DQNAgent, env: PolyCircuitEnv, config: Config):
        self.agent = agent
        self.env = env
        self.gamma = config.gamma
        self.num_simulations = config.mcts_simulations
        self.c_puct = config.mcts_c_puct
        self.temperature = config.mcts_temperature
        self.rng = np.random.default_rng(config.seed + 31)

    def search(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Run MCTS from the current env state and return the best action."""
        root_state = self.env.get_state()
        root = MCTSNode(state=root_state, obs=obs, action_mask=action_mask)

        # Initialize root priors
        self._init_node(root)
        try:
            for _ in range(self.num_simulations):
                node = root

                # SELECT: walk tree using PUCT until we find a node to expand
                while node.is_fully_expanded() and not node.is_terminal:
                    node = self._select_child(node)

                # If terminal, just backup
                if node.is_terminal:
                    self._backup(node, 0.0)
                    continue

                # EXPAND: expand one child (best unexpanded action by prior)
                if node.unexpanded_actions:
                    child, leaf_value = self._expand_one(node)
                    self._backup(child, leaf_value)
                else:
                    # No valid actions (shouldn't happen, but guard)
                    self._backup(node, self._evaluate_leaf(node.obs, node.action_mask))

            return self._select_action(root)
        finally:
            # Restore env to root state so the caller can execute the real step.
            self.env.set_state(root_state)
            self.env._simulation = False

    def _init_node(self, node: MCTSNode) -> None:
        """Compute priors and set up unexpanded action list for a node."""
        node.priors = self._get_priors(node.obs, node.action_mask)
        node.valid_actions = np.where(node.action_mask > 0)[0]

        if len(node.valid_actions) == 0:
            node.is_terminal = True
            node.unexpanded_actions = []
            return

        # Sort by prior descending so we expand best actions first
        prior_order = np.argsort(-node.priors[node.valid_actions])
        node.unexpanded_actions = list(node.valid_actions[prior_order])

    def _predict_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Agent Q-value helper with compatibility fallback for test doubles."""
        if hasattr(self.agent, "predict_q_values"):
            return self.agent.predict_q_values(obs)
        with torch.no_grad():
            obs_t = torch.tensor(
                obs,
                dtype=torch.float32,
                device=self.agent.device,
            ).unsqueeze(0)
            return self.agent.q_network(obs_t).squeeze(0).cpu().numpy()

    def _get_priors(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        """Compute action priors from Q-network via softmax over valid Q-values."""
        q = self._predict_q_values(obs)

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
        q = self._predict_q_values(obs)

        valid = action_mask > 0
        if not valid.any():
            return 0.0
        return float(q[valid].max())

    def _expand_one(self, node: MCTSNode) -> tuple[MCTSNode, float]:
        """Expand one action and return the child plus its state value."""
        action = node.unexpanded_actions.pop(0)

        # Simulate action from this node's state
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
            reward_from_parent=reward,
        )

        if done:
            child.is_terminal = True
            child.terminal_value = reward
            child.unexpanded_actions = []
            leaf_value = 0.0
        else:
            # Initialize child and evaluate
            self._init_node(child)
            leaf_value = self._evaluate_leaf(child.obs, child.action_mask)

        node.children[action] = child
        return child, leaf_value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child using PUCT formula."""
        best_score = -float("inf")
        best_child = None
        sqrt_parent = math.sqrt(node.visit_count)

        for action, child in node.children.items():
            prior = node.priors[action] if node.priors is not None else 0.0
            q = child.q_value
            score = q + self.c_puct * prior * sqrt_parent / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _backup(self, node: MCTSNode, state_value: float) -> None:
        """Propagate discounted action returns up the tree."""
        value = state_value
        current = node
        while current.parent is not None:
            value = current.reward_from_parent + self.gamma * value
            current.visit_count += 1
            current.value_sum += value
            current = current.parent

        current.visit_count += 1
        current.value_sum += value

    def _select_action(self, root: MCTSNode) -> int:
        """Select action from root based on visit counts."""
        visits = np.array([
            root.children[a].visit_count if a in root.children else 0
            for a in range(len(root.action_mask))
        ], dtype=np.float64)

        if self.temperature == 0:
            return int(np.argmax(visits))

        # Temperature-scaled sampling
        valid = visits > 0
        if not valid.any():
            valid_actions = np.where(root.action_mask > 0)[0]
            return int(self.rng.choice(valid_actions))

        visits_temp = np.zeros_like(visits)
        visits_temp[valid] = visits[valid] ** (1.0 / self.temperature)
        probs = visits_temp / visits_temp.sum()
        return int(self.rng.choice(len(probs), p=probs))
