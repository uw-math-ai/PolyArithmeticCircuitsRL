"""Monte Carlo Tree Search with neural network prior and value estimates."""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..config import Config
from ..environment.circuit_game import CircuitGame


class MCTSNode:
    """Tree node storing visit counts, values, and prior probabilities."""

    def __init__(self, prior: float = 0.0):
        self.prior = prior  # P(s, a) from neural network
        self.visit_count = 0  # N(s, a)
        self.total_value = 0.0  # W(s, a)
        self.children: Dict[int, "MCTSNode"] = {}
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        """Mean value Q(s, a) = W(s, a) / N(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MCTS:
    """Neural MCTS (AlphaZero-style).

    Uses neural network for both prior probabilities (policy) and
    leaf evaluation (value), replacing random rollouts.
    """

    def __init__(self, model, config: Config, device: str = "cpu"):
        self.model = model
        self.config = config
        self.device = device

    def _to_device(self, value: Any):
        """Recursively move observation values to the configured device."""
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self._to_device(v) for k, v in value.items()}
        # PyG Data and similar containers implement .to(device).
        if hasattr(value, "to"):
            return value.to(self.device)
        return value

    @torch.no_grad()
    def search(self, game: CircuitGame) -> Dict[int, int]:
        """Run MCTS simulations from current game state.

        Args:
            game: current game state (will be cloned for each simulation)

        Returns:
            Dictionary mapping action_idx -> visit_count
        """
        root = MCTSNode()
        # Expand root
        self._expand(root, game)

        for _ in range(self.config.mcts_simulations):
            node = root
            sim_game = game.clone()
            search_path = [node]

            # Selection: traverse tree using PUCT until we reach an unexpanded node
            while node.is_expanded and not sim_game.done:
                action, child = self._select_child(node)
                obs, reward, done, info = sim_game.step(action)
                node = child
                search_path.append(node)

            # Evaluate leaf
            if sim_game.done:
                # Terminal state: use actual outcome
                value = 1.0 if info.get("is_success", False) else -1.0
            else:
                # Non-terminal: expand and evaluate with neural network
                value = self._expand(node, sim_game)

            # Backpropagate
            for node in reversed(search_path):
                node.visit_count += 1
                node.total_value += value

        # Return visit counts for root's children
        return {action: child.visit_count for action, child in root.children.items()}

    def get_action_probs(self, game: CircuitGame,
                          temperature: float = 1.0) -> Tuple[int, np.ndarray]:
        """Run MCTS search and return action + probability distribution.

        Args:
            game: current game state
            temperature: controls exploration (higher = more random)

        Returns:
            (selected_action, action_probability_vector)
        """
        visit_counts = self.search(game)

        if not visit_counts:
            # No valid actions — shouldn't happen if game isn't done
            max_actions = self.config.max_actions
            return 0, np.zeros(max_actions)

        max_actions = self.config.max_actions
        counts = np.zeros(max_actions)
        for action, count in visit_counts.items():
            counts[action] = count

        if temperature == 0:
            # Deterministic: pick best action
            action = max(visit_counts, key=visit_counts.get)
            probs = np.zeros(max_actions)
            probs[action] = 1.0
        else:
            # Sample from visit counts with temperature
            counts_temp = counts ** (1.0 / temperature)
            total = counts_temp.sum()
            if total > 0:
                probs = counts_temp / total
            else:
                probs = np.zeros(max_actions)
            action = np.random.choice(max_actions, p=probs)

        return action, probs

    def _expand(self, node: MCTSNode, game: CircuitGame) -> float:
        """Expand a node using the neural network.

        Args:
            node: node to expand
            game: game state at this node

        Returns:
            Value estimate for this state
        """
        obs = game.get_observation()
        obs_device = {k: self._to_device(v) for k, v in obs.items()}

        action_probs, value = self.model.get_policy_and_value(obs_device)
        action_probs = action_probs.cpu().numpy()
        value = value.item()

        # Create children for all valid actions
        mask = obs["mask"].detach().cpu().numpy()
        for action_idx in range(len(mask)):
            if mask[action_idx]:
                node.children[action_idx] = MCTSNode(prior=action_probs[action_idx])

        node.is_expanded = True
        return value

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select child using PUCT formula.

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

        Returns:
            (action, child_node)
        """
        sqrt_total = math.sqrt(node.visit_count)
        c_puct = self.config.mcts_c_puct

        best_score = float("-inf")
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            puct = child.q_value + c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            if puct > best_score:
                best_score = puct
                best_action = action
                best_child = child

        return best_action, best_child
