"""Core circuit construction game environment with Gymnasium-style API.

Uses FastPoly (numpy-based) for all polynomial arithmetic instead of SymPy,
giving 10-100x speedup per environment step.
"""

from typing import Dict, List, Optional, Tuple

import torch

from ..config import Config
from .fast_polynomial import FastPoly
from .action_space import decode_action, get_valid_actions_mask

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None


class CircuitGame:
    """Circuit construction game.

    The agent builds an arithmetic circuit by selecting operations (add/multiply)
    on pairs of existing nodes. The goal is to construct a target polynomial.

    Nodes start as [x0, x1, ..., x_{n-1}, 1] and grow as operations are applied.
    All polynomial arithmetic uses FastPoly (dense numpy coefficient arrays mod p).
    """

    def __init__(self, config: Config):
        self.config = config
        self.n_vars = config.n_variables
        self.mod = config.mod
        self.max_deg = config.effective_max_degree
        self.max_nodes = config.max_nodes
        self.max_actions = config.max_actions

        # Precompute initial node polynomials (reused every reset)
        self._init_polys: List[FastPoly] = []
        for i in range(self.n_vars):
            self._init_polys.append(FastPoly.variable(i, self.n_vars, self.max_deg, self.mod))
        self._init_polys.append(FastPoly.constant(1, self.n_vars, self.max_deg, self.mod))

        # State (initialized in reset)
        self.nodes: List[FastPoly] = []
        self.node_types: List[Tuple[int, int, int, float]] = []
        self.edges: List[Tuple[int, int]] = []
        self.target_poly: Optional[FastPoly] = None
        self.steps_taken: int = 0
        self.done: bool = True

    def reset(self, target_poly: FastPoly) -> Dict[str, torch.Tensor]:
        """Reset the game with a new target polynomial.

        Args:
            target_poly: FastPoly target (must share same n_vars, max_degree, mod)

        Returns:
            Initial observation dict
        """
        self.target_poly = target_poly
        self.steps_taken = 0
        self.done = False

        # Initialize nodes: variables + constant 1
        self.nodes = [p.copy() for p in self._init_polys]
        self.node_types = []
        self.edges = []

        for _ in range(self.n_vars):
            self.node_types.append((1, 0, 0, 0.0))
        self.node_types.append((0, 1, 0, 0.0))  # constant

        return self.get_observation()

    def step(self, action_idx: int) -> Tuple[Dict[str, torch.Tensor], float, bool, dict]:
        """Apply an action and return (obs, reward, done, info)."""
        assert not self.done, "Game is already done. Call reset()."

        op, i, j = decode_action(action_idx, self.max_nodes)
        num_nodes = len(self.nodes)

        assert i < num_nodes and j < num_nodes, (
            f"Invalid action: nodes {i},{j} but only {num_nodes} nodes exist"
        )

        # Compute new polynomial (fast numpy arithmetic)
        if op == 0:
            new_poly = self.nodes[i] + self.nodes[j]
        else:
            new_poly = self.nodes[i] * self.nodes[j]

        # Compute potential before (for reward shaping)
        if self.config.use_reward_shaping:
            phi_before = self._best_similarity()

        # Add new node
        new_idx = len(self.nodes)
        self.nodes.append(new_poly)
        op_value = 0.5 if op == 0 else 1.0
        self.node_types.append((0, 0, 1, op_value))

        # Add edges
        self.edges.append((i, new_idx))
        self.edges.append((j, new_idx))

        self.steps_taken += 1

        # Check success (numpy array equality — very fast)
        is_success = (new_poly == self.target_poly)

        # Check termination
        at_max_steps = self.steps_taken >= self.config.max_steps
        at_max_nodes = len(self.nodes) >= self.max_nodes
        self.done = is_success or at_max_steps or at_max_nodes

        # Compute reward
        reward = self.config.step_penalty

        if is_success:
            reward += self.config.success_reward
        elif self.config.use_reward_shaping:
            phi_after = self._best_similarity()
            reward += self.config.gamma * phi_after - phi_before

        info = {
            "is_success": is_success,
            "steps_taken": self.steps_taken,
            "num_nodes": len(self.nodes),
            "new_poly": new_poly,
            "op": "add" if op == 0 else "mul",
            "operands": (i, j),
        }

        return self.get_observation(), reward, self.done, info

    def get_observation(self) -> Dict[str, torch.Tensor]:
        """Get current observation."""
        graph = self._build_graph()
        target = self._encode_target(self.target_poly)
        mask = get_valid_actions_mask(len(self.nodes), self.max_nodes)

        return {
            "graph": graph,
            "target": target,
            "mask": mask,
        }

    def _build_graph(self):
        """Build a PyG Data object from the current circuit state."""
        x = torch.zeros(self.max_nodes, self.config.node_feature_dim)
        for idx, features in enumerate(self.node_types):
            x[idx] = torch.tensor(features, dtype=torch.float32)

        if self.edges:
            src = [e[0] for e in self.edges]
            dst = [e[1] for e in self.edges]
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        num_nodes_actual = len(self.nodes)

        if Data is not None:
            return Data(
                x=x,
                edge_index=edge_index,
                num_nodes=self.max_nodes,
                num_nodes_actual=num_nodes_actual,
            )
        else:
            return {
                "x": x,
                "edge_index": edge_index,
                "num_nodes": self.max_nodes,
                "num_nodes_actual": num_nodes_actual,
            }

    def _encode_target(self, poly: FastPoly) -> torch.Tensor:
        """Encode target polynomial as a flattened coefficient vector."""
        vec = poly.to_vector()  # numpy float64 flat array
        return torch.tensor(vec, dtype=torch.float32) / self.mod

    def _best_similarity(self) -> float:
        """Find the best term similarity between any current node and the target."""
        best = 0.0
        for node_poly in self.nodes:
            sim = node_poly.term_similarity(self.target_poly)
            if sim > best:
                best = sim
                if best == 1.0:
                    break
        return best

    def clone(self) -> "CircuitGame":
        """Create a deep copy of this game state (for MCTS)."""
        new_game = CircuitGame.__new__(CircuitGame)
        new_game.config = self.config
        new_game.n_vars = self.n_vars
        new_game.mod = self.mod
        new_game.max_deg = self.max_deg
        new_game.max_nodes = self.max_nodes
        new_game.max_actions = self.max_actions
        new_game._init_polys = self._init_polys  # shared, immutable after __init__

        new_game.target_poly = self.target_poly  # targets don't change
        new_game.steps_taken = self.steps_taken
        new_game.done = self.done
        # Deep copy mutable state: nodes need array copies
        new_game.nodes = [p.copy() for p in self.nodes]
        new_game.node_types = list(self.node_types)
        new_game.edges = list(self.edges)
        return new_game
