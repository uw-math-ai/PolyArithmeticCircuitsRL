"""
Monte Carlo Tree Search implementation for polynomial circuit construction.
Based on AlphaZero methodology with UCB selection, expansion, simulation, and backpropagation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
import math
from typing import Dict, List, Optional, Tuple

from State import Game
from torch_geometric.data import Batch


class MCTSNode:
    """
    A node in the Monte Carlo Tree Search tree.

    Each node represents a state in the circuit construction process,
    storing visit counts, value estimates, and prior probabilities.
    """

    def __init__(
        self,
        game_state: Game,
        parent: Optional["MCTSNode"] = None,
        action: Optional[int] = None,
        prior_prob: float = 0.0,
    ):
        self.game_state = game_state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.prior_prob = prior_prob  # P(s,a) from neural network

        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, MCTSNode] = {}

        # Caching
        self._is_expanded = False
        self._is_terminal = None
        self._legal_actions = None

    def is_expanded(self) -> bool:
        """Check if this node has been expanded (children created)."""
        return self._is_expanded

    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state."""
        if self._is_terminal is None:
            self._is_terminal = self.game_state.is_done()
        return self._is_terminal

    def get_legal_actions(self) -> List[int]:
        """Get list of legal actions from this state."""
        if self._legal_actions is None:
            _, _, _, mask = self.game_state.observe()
            self._legal_actions = torch.where(mask[0])[0].tolist()
        return self._legal_actions

    def get_value(self) -> float:
        """Get the average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float = 1.0) -> float:
        """
        Calculate Upper Confidence Bound score for node selection.

        UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            c_puct: Exploration constant that balances exploration vs exploitation
        """
        if self.visit_count == 0:
            return float("inf")  # Unvisited nodes have highest priority

        # Q(s,a): Average action value
        q_value = self.get_value()

        # Exploration term: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = (
            c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visit_count)
        )

        return q_value + exploration

    def select_child(self, c_puct: float = 1.0) -> "MCTSNode":
        """Select best child using UCB scores."""
        if not self.children:
            raise ValueError("Cannot select child from unexpanded node")

        return max(self.children.values(), key=lambda child: child.ucb_score(c_puct))

    def expand(self, action_probs: Dict[int, float]) -> None:
        """
        Expand this node by creating children for all legal actions.

        Args:
            action_probs: Dictionary mapping actions to prior probabilities
        """
        if self.is_expanded():
            return

        legal_actions = self.get_legal_actions()

        for action in legal_actions:
            # Create new game state by taking the action
            new_game = copy.deepcopy(self.game_state)
            new_game.take_action(action)

            # Get prior probability for this action
            prior_prob = action_probs.get(action, 0.0)

            # Create child node
            child = MCTSNode(
                new_game, parent=self, action=action, prior_prob=prior_prob
            )
            self.children[action] = child

        self._is_expanded = True

    def backup(self, value: float) -> None:
        """
        Backpropagate value up the tree, updating visit counts and value sums.

        Args:
            value: The value to backpropagate (from current player's perspective)
        """
        self.visit_count += 1
        self.value_sum += value

        if self.parent is not None:
            self.parent.backup(value)

    def get_action_probabilities(self, temperature: float = 1.0) -> Dict[int, float]:
        """
        Get action probabilities based on visit counts.

        Args:
            temperature: Temperature parameter for action selection
                        - temperature=0: Select action with highest visit count
                        - temperature=1: Proportional to visit count
                        - temperature>1: More uniform distribution
        """
        if not self.children:
            return {}

        actions = list(self.children.keys())
        visit_counts = [self.children[action].visit_count for action in actions]

        if temperature == 0:
            # Select action with highest visit count
            best_action = actions[np.argmax(visit_counts)]
            probs = {action: 0.0 for action in actions}
            probs[best_action] = 1.0
            return probs

        # Apply temperature
        visit_counts = np.array(visit_counts, dtype=np.float64)
        if temperature != 1.0:
            visit_counts = visit_counts ** (1.0 / temperature)

        # Normalize to probabilities
        total = np.sum(visit_counts)
        if total == 0:
            # If no visits, return uniform distribution
            prob_value = 1.0 / len(actions)
            return {action: prob_value for action in actions}

        probs = visit_counts / total
        return {action: prob for action, prob in zip(actions, probs)}


class MCTS:
    """
    Monte Carlo Tree Search for polynomial circuit construction.

    Implements the four phases of MCTS:
    1. Selection: Navigate down tree using UCB
    2. Expansion: Add new nodes to tree
    3. Simulation: Evaluate leaf nodes using neural network
    4. Backpropagation: Update values up the tree
    """

    def __init__(self, model, config, c_puct: float = 1.0, num_simulations: int = 800):
        """
        Initialize MCTS.

        Args:
            model: Neural network model for policy and value prediction
            config: Configuration object
            c_puct: Exploration constant for UCB
            num_simulations: Number of MCTS simulations to run
        """
        self.model = model
        self.config = config
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def search(self, root_game: Game) -> MCTSNode:
        """
        Run MCTS search from the given root state.

        Args:
            root_game: Initial game state to search from

        Returns:
            Root node with expanded tree
        """
        # Create root node
        root = MCTSNode(root_game)

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root)

        return root

    def _simulate(self, root: MCTSNode) -> None:
        """
        Run a single MCTS simulation from root to leaf.

        This implements the four phases:
        1. Selection: Navigate to leaf using UCB
        2. Expansion: Expand leaf if not terminal
        3. Evaluation: Get value estimate from neural network
        4. Backpropagation: Update values up the path
        """
        # Phase 1: Selection - navigate to leaf node
        path = []
        current = root

        while current.is_expanded() and not current.is_terminal():
            current = current.select_child(self.c_puct)
            path.append(current)

        # Phase 2: Expansion (if not terminal)
        if not current.is_terminal():
            action_probs, value = self._evaluate_leaf(current)
            current.expand(action_probs)

            # If we expanded, select one child for evaluation
            if current.children:
                current = current.select_child(self.c_puct)
                path.append(current)
                _, value = self._evaluate_leaf(current)
        else:
            # Terminal node - compute exact value
            value = self._compute_terminal_value(current)

        # Phase 4: Backpropagation
        for node in reversed(path):
            node.backup(value)
        root.backup(value)

    def _evaluate_leaf(self, node: MCTSNode) -> Tuple[Dict[int, float], float]:
        """
        Evaluate a leaf node using the neural network.

        Args:
            node: Node to evaluate

        Returns:
            Tuple of (action_probabilities, value_estimate)
        """
        self.model.eval()

        with torch.no_grad():
            # Get state representation
            graph, target_vec, actions, mask = node.game_state.observe()

            # Move to device
            graph_batch = Batch.from_data_list([graph.to(self.device)])
            target_vec = target_vec.to(self.device)
            mask = mask.to(self.device)

            # Get policy and value from model
            action_logits, value = self.model(graph_batch, target_vec, actions, mask)

            # Convert to probabilities
            action_probs_tensor = F.softmax(action_logits[0], dim=0)

            # Extract probabilities for legal actions
            legal_actions = node.get_legal_actions()
            action_probs = {}

            for action in legal_actions:
                if action < len(action_probs_tensor):
                    action_probs[action] = action_probs_tensor[action].item()
                else:
                    action_probs[action] = 0.0

            # Normalize probabilities
            total_prob = sum(action_probs.values())
            if total_prob > 0:
                action_probs = {
                    action: prob / total_prob for action, prob in action_probs.items()
                }
            else:
                # Uniform distribution if all probabilities are zero
                uniform_prob = 1.0 / len(legal_actions) if legal_actions else 0.0
                action_probs = {action: uniform_prob for action in legal_actions}

            return action_probs, value.item()

    def _compute_terminal_value(self, node: MCTSNode) -> float:
        """
        Compute the exact value for a terminal node.

        Args:
            node: Terminal node to evaluate

        Returns:
            Value of the terminal state
        """
        game = node.game_state

        # Check if we found a correct solution
        if game.exprs:
            # Use symbolic verification
            import sympy as sp

            try:
                if sp.expand(game.exprs[-1] - game.target_sp) == 0:
                    # Success! Give high reward based on efficiency
                    circuit_length = len(game.actions_taken)
                    efficiency_bonus = max(
                        0, game.config.max_complexity - circuit_length
                    )
                    return 10.0 + efficiency_bonus
                else:
                    # Incorrect solution
                    return -5.0
            except:
                # Symbolic computation failed
                return -5.0

        # No solution found within complexity limit
        return -1.0

    def get_action_probabilities(
        self, root: MCTSNode, temperature: float = 1.0
    ) -> Dict[int, float]:
        """
        Get action probabilities from MCTS search results.

        Args:
            root: Root node after MCTS search
            temperature: Temperature for action selection

        Returns:
            Dictionary mapping actions to probabilities
        """
        return root.get_action_probabilities(temperature)

    def select_action(self, root: MCTSNode, temperature: float = 1.0) -> int:
        """
        Select an action based on MCTS search results.

        Args:
            root: Root node after MCTS search
            temperature: Temperature for action selection

        Returns:
            Selected action index
        """
        action_probs = self.get_action_probabilities(root, temperature)

        if not action_probs:
            # No legal actions
            raise ValueError("No legal actions available")

        actions = list(action_probs.keys())
        probs = list(action_probs.values())

        # Sample action according to probabilities
        action = np.random.choice(actions, p=probs)
        return action


def mcts_self_play_game(
    model,
    config,
    target_poly_sp,
    target_poly_vec,
    index_to_monomial,
    monomial_to_index,
    mcts_simulations: int = 800,
    temperature: float = 1.0,
) -> Tuple[List, List, float]:
    """
    Play a single self-play game using MCTS.

    Args:
        model: Neural network model
        config: Configuration object
        target_poly_sp: Target polynomial (SymPy)
        target_poly_vec: Target polynomial (vector)
        index_to_monomial: Monomial index mapping
        monomial_to_index: Monomial to index mapping
        mcts_simulations: Number of MCTS simulations per move
        temperature: Temperature for action selection

    Returns:
        Tuple of (states, action_probabilities, game_result)
    """
    # Initialize game
    game = Game(
        target_poly_sp,
        target_poly_vec.unsqueeze(0),
        config,
        index_to_monomial,
        monomial_to_index,
    )

    # Initialize MCTS
    mcts = MCTS(model, config, num_simulations=mcts_simulations)

    # Store game trajectory
    states = []
    action_probs = []

    while not game.is_done():
        # Store current state
        state_tuple = game.observe()
        states.append(state_tuple)

        # Run MCTS search
        root = mcts.search(game)

        # Get action probabilities
        action_probabilities = mcts.get_action_probabilities(root, temperature)
        action_probs.append(action_probabilities)

        # Select and take action
        action = mcts.select_action(root, temperature)
        game.take_action(action)

    # Compute game result
    game_result = mcts._compute_terminal_value(MCTSNode(game))

    return states, action_probs, game_result


if __name__ == "__main__":
    # Simple test
    print("MCTS implementation completed.")
    print("Key features:")
    print("- MCTSNode with UCB selection")
    print("- Four-phase MCTS: Selection, Expansion, Evaluation, Backpropagation")
    print("- Integration with neural network policy and value functions")
    print("- Self-play game generation")
