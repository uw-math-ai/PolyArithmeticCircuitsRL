from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import sympy as sp

from utils import decode_action, encode_action

if TYPE_CHECKING:  # pragma: no cover
    from State import Game


class CircuitGameState:
    """Lightweight, cloneable view of the Game state for MCTS planning."""

    def __init__(
        self,
        target_poly: sp.Expr,
        config,
        actions: List[Tuple[str, int, int]],
        polynomials: List[sp.Expr],
        current_step: int,
        max_nodes: int,
        max_steps: int,
    ) -> None:
        self.target_poly = sp.expand(target_poly)
        self.config = config
        self.actions = list(actions)
        self.polynomials = list(polynomials)
        self.current_step = current_step
        self.max_nodes = max_nodes
        self.max_steps = max_steps
        self._target_terms = self.target_poly.as_coefficients_dict()

    @classmethod
    def from_game(cls, game: "Game") -> "CircuitGameState":
        return cls(
            target_poly=game.target_poly_expr,
            config=game.config,
            actions=game.actions,
            polynomials=game.polynomials,
            current_step=game.current_step,
            max_nodes=game.max_nodes,
            max_steps=game.max_steps,
        )

    def clone(self) -> "CircuitGameState":
        return CircuitGameState(
            target_poly=self.target_poly,
            config=self.config,
            actions=self.actions,
            polynomials=self.polynomials,
            current_step=self.current_step,
            max_nodes=self.max_nodes,
            max_steps=self.max_steps,
        )

    def available_actions(self) -> List[int]:
        if self.current_step >= self.max_steps or len(self.polynomials) >= self.max_nodes:
            return []
        actions: List[int] = []
        n_nodes = len(self.polynomials)
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                actions.append(encode_action("add", i, j, self.max_nodes))
                actions.append(encode_action("multiply", i, j, self.max_nodes))
        return actions

    def apply_action(self, action_idx: int) -> None:
        if self.current_step >= self.max_steps:
            return
        operation, node1_id, node2_id = decode_action(action_idx, self.max_nodes)
        if node1_id >= len(self.polynomials) or node2_id >= len(self.polynomials):
            self.current_step = self.max_steps
            return
        poly1 = self.polynomials[node1_id]
        poly2 = self.polynomials[node2_id]
        if operation == "add":
            new_poly = sp.expand(poly1 + poly2)
        else:
            new_poly = sp.expand(poly1 * poly2)
        self.actions.append((operation, node1_id, node2_id))
        self.polynomials.append(new_poly)
        self.current_step += 1

    def matches_target(self) -> bool:
        if not self.polynomials:
            return False
        return sp.expand(self.polynomials[-1] - self.target_poly) == 0

    def is_terminal(self) -> bool:
        if self.current_step >= self.max_steps:
            return True
        return self.matches_target()

    def similarity_score(self) -> float:
        if not self.polynomials:
            return 0.0
        try:
            current_terms = self.polynomials[-1].as_coefficients_dict()
        except Exception:
            return 0.0
        total = max(len(self._target_terms), 1)
        matches = sum(
            1 for term, coeff in self._target_terms.items() if current_terms.get(term) == coeff
        )
        return matches / total

    def evaluate_reward(self) -> float:
        if self.matches_target():
            return float(getattr(self.config, "success_reward", 1.0))
        base_penalty = float(getattr(self.config, "step_penalty", -1.0))
        return base_penalty + float(self.similarity_score())


class MCTSNode:
    """Tree node storing visit statistics for UCT selection."""

    def __init__(
        self,
        state: CircuitGameState,
        parent: Optional["MCTSNode"],
        action_taken: Optional[int],
    ) -> None:
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: Dict[int, MCTSNode] = {}
        self.untried_actions = state.available_actions()
        random.shuffle(self.untried_actions)
        self.visit_count = 0
        self.total_value = 0.0

    def is_fully_expanded(self) -> bool:
        return not self.untried_actions

    def best_child(self, exploration_constant: float) -> "MCTSNode":
        best_score = -float("inf")
        best_child = None
        for child in self.children.values():
            if child.visit_count == 0:
                score = float("inf")
            else:
                exploit = child.total_value / child.visit_count
                explore = math.sqrt(math.log(max(self.visit_count, 1)) / child.visit_count)
                score = exploit + exploration_constant * explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child if best_child is not None else self

    def add_child(self, action_idx: int, child_state: CircuitGameState) -> "MCTSNode":
        child = MCTSNode(state=child_state, parent=self, action_taken=action_idx)
        self.children[action_idx] = child
        return child

    def update(self, reward: float) -> None:
        self.visit_count += 1
        self.total_value += reward


class MCTS:
    """Basic Monte Carlo Tree Search with random rollouts."""

    def __init__(self, root_state: CircuitGameState, exploration_constant: float = 1.4) -> None:
        self.root = MCTSNode(root_state, parent=None, action_taken=None)
        self.exploration_constant = exploration_constant

    def run(self, iterations: int) -> None:
        for _ in range(iterations):
            node = self._select(self.root)
            reward = self._rollout_from(node)
            self._backpropagate(node, reward)

    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while not current.state.is_terminal():
            if current.untried_actions:
                return self._expand(current)
            if not current.children:
                break
            current = current.best_child(self.exploration_constant)
        return current

    def _expand(self, node: MCTSNode) -> MCTSNode:
        action_idx = node.untried_actions.pop()
        next_state = node.state.clone()
        next_state.apply_action(action_idx)
        return node.add_child(action_idx, next_state)

    def _rollout_from(self, node: MCTSNode) -> float:
        rollout_state = node.state.clone()
        while not rollout_state.is_terminal():
            actions = rollout_state.available_actions()
            if not actions:
                break
            action_idx = random.choice(actions)
            rollout_state.apply_action(action_idx)
        return rollout_state.evaluate_reward()

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        current: Optional[MCTSNode] = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def best_root_action(self) -> Optional[int]:
        if not self.root.children:
            return None
        best_action, _ = max(
            self.root.children.items(), key=lambda item: item[1].visit_count
        )
        return best_action


class MCTSPlanner:
    """Helper that wires up CircuitGameState + MCTS for PPO data collection."""

    def __init__(self, config) -> None:
        self.config = config
        self.num_simulations = getattr(config, "mcts_simulations", 0)
        self.exploration_constant = getattr(config, "mcts_exploration", 1.4)

    def select_action(self, game: "Game") -> Optional[int]:
        if self.num_simulations <= 0:
            return None
        root_state = CircuitGameState.from_game(game)
        if root_state.is_terminal():
            return None
        search = MCTS(root_state, exploration_constant=self.exploration_constant)
        search.run(self.num_simulations)
        return search.best_root_action()
