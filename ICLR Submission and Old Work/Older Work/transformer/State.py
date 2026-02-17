import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import sympy
from utils import decode_action, encode_action


class Game:
    def __init__(self, target_poly_expr, target_poly_tensor, config):
        self.target_poly_expr = target_poly_expr
        self.target_poly_tensor = target_poly_tensor
        self.config = config
        self.symbols = [sympy.Symbol(f"x{i}") for i in range(config.n_variables)]

        self.actions = []
        self.polynomials = []
        self.max_nodes = self.config.n_variables + self.config.max_complexity + 1

        # Initialize with base nodes (variables and a constant)
        for i in range(self.config.n_variables):
            self.actions.append(("input", i, -1))
            self.polynomials.append(self.symbols[i])

        self.actions.append(("constant", -1, -1))
        self.polynomials.append(sympy.Integer(1))

        self.current_step = 0
        self.max_steps = self.config.max_complexity

    def to(self, device):
        """Moves internal tensors to the specified device (currently only CPU)."""
        self.device = device
        # If any tensors were stored, move them here.
        return self

    def observe(self):
        circuit_graph = self.get_graph_representation()

        # The target polynomial tensor is already computed, just needs to be on the right device
        target_poly_tensor_dev = self.target_poly_tensor.to(
            next(torch.zeros(1)).device if torch.cuda.is_available() else "cpu"
        )

        # The circuit actions are just the list of actions taken
        circuit_actions = [self.actions]

        # The mask of available actions
        available_actions_mask = self.get_available_actions_mask()

        return circuit_graph, target_poly_tensor_dev, circuit_actions, available_actions_mask.unsqueeze(0)

    def get_graph_representation(self):
        n_nodes = len(self.actions)
        node_features, edges = [], []

        for i, (action_type, input1_idx, input2_idx) in enumerate(self.actions):
            if action_type == "input":
                type_encoding, value = [1, 0, 0], i / max(1, self.config.n_variables)
            elif action_type == "constant":
                type_encoding, value = [0, 1, 0], 1.0
            else:  # "add" or "multiply"
                type_encoding, value = [0, 0, 1], 1.0 if action_type == "multiply" else 0.0
                edges.append((input1_idx, i))
                edges.append((input2_idx, i))

            node_features.append(type_encoding + [value])

        x = torch.tensor(node_features, dtype=torch.float) if node_features else torch.empty((0, 4), dtype=torch.float)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

        edge_index, _ = add_self_loops(edge_index, num_nodes=n_nodes)

        return Data(x=x, edge_index=edge_index)

    def get_available_actions_mask(self):
        n_nodes = len(self.actions)
        total_max_pairs = (self.max_nodes * (self.max_nodes + 1)) // 2
        max_possible_actions = total_max_pairs * 2
        mask = torch.zeros(max_possible_actions, dtype=torch.bool)

        for i in range(n_nodes):
            for j in range(i, n_nodes):
                # Add operation
                action_idx_add = encode_action("add", i, j, self.max_nodes)
                if action_idx_add < max_possible_actions:
                    mask[action_idx_add] = True

                # Multiply operation
                action_idx_mul = encode_action("multiply", i, j, self.max_nodes)
                if action_idx_mul < max_possible_actions:
                    mask[action_idx_mul] = True

        return mask

    def take_action(self, action_idx):
        op, node1_id, node2_id = decode_action(action_idx, self.max_nodes)

        if node1_id >= len(self.polynomials) or node2_id >= len(self.polynomials):
            # Invalid action, though the mask should prevent this.
            # We can handle this by adding a large penalty or ending the game.
            self.current_step = self.max_steps  # End the game
            return

        poly1 = self.polynomials[node1_id]
        poly2 = self.polynomials[node2_id]

        if op == "add":
            new_poly = sympy.expand(poly1 + poly2)
        elif op == "multiply":
            new_poly = sympy.expand(poly1 * poly2)
        else:
            # Should not happen
            self.current_step = self.max_steps
            return

        self.actions.append((op, node1_id, node2_id))
        self.polynomials.append(new_poly)
        self.current_step += 1

    def is_done(self):
        if self.current_step >= self.max_steps:
            return True

        # Check if the current polynomial matches the target
        if self.polynomials:
            last_poly = self.polynomials[-1]
            if sympy.expand(last_poly - self.target_poly_expr) == 0:
                return True

        return False

    def compute_rewards(self):
        if not self.polynomials:
            return [0.0]

        last_poly = self.polynomials[-1]

        # Reward for finding the exact polynomial
        if sympy.expand(last_poly - self.target_poly_expr) == 0:
            return [10.0]  # High reward for success

        # Penalty for exceeding complexity
        if len(self.actions) > self.config.n_variables + 1 + self.config.max_complexity:
            return [-1.0]

        # Intermediate reward based on similarity (e.g., number of matching terms)
        try:
            target_terms = self.target_poly_expr.as_coefficients_dict()
            current_terms = last_poly.as_coefficients_dict()

            matching_terms = 0
            for term, coeff in target_terms.items():
                if term in current_terms and current_terms[term] == coeff:
                    matching_terms += 1

            total_target_terms = len(target_terms)
            reward = (matching_terms / total_target_terms) * 0.5 - 0.1  # Small reward for progress, penalty for each step
        except Exception:
            reward = -0.1

        return [reward]
