import sympy as sp
import torch
from torch_geometric.data import Data
import torch_geometric.utils # Import this
from utils import encode_action
import torch
import math

class Game:
    def __init__(self, sympy_poly, vec_poly, config):
        self.config = config
        self.actions_taken = []
        self.exprs = []
        self.used_actions = []
        # Corrected line: Use list [sp.S.One] instead of tuple
        self.symbols = sp.symbols([f"x{i}" for i in range(config.n_variables)]) + [sp.S.One]
        self.target_sp = sympy_poly
        self.target_vec = vec_poly
        self.device = 'cpu'

    def _decode_action(self, action_idx):
        operation = "multiply" if action_idx % 2 == 1 else "add"
        pair_idx = action_idx // 2

        node1_id = int((math.sqrt(1 + 8 * pair_idx) - 1) / 2)
        node2_id = pair_idx - ((node1_id * (node1_id + 1)) // 2)
        return node1_id, operation, node2_id

    def _expr_used(self, node):
        num_base_nodes = self.config.n_variables + 1
        if node >= num_base_nodes:
            index = node - num_base_nodes
            return self.exprs[index], self.used_actions[index]
        else:
            return self.symbols[node], {node}

    def take_action(self, action_idx):
        mul = lambda a, b : a * b
        add = lambda a, b : a + b

        node1, op, node2 = self._decode_action(action_idx)
        expr1, used1 = self._expr_used(node1)
        expr2, used2 = self._expr_used(node2)
        op_fn = mul if op == "multiply" else add

        new_expr = sp.expand(op_fn(expr1, expr2))
        self.actions_taken.append((op, node1, node2))
        self.exprs.append(new_expr)
        self.used_actions.append(used1.union(used2).union({len(self.actions_taken) - 1 + self.config.n_variables + 1}))

    def is_done(self):
        return len(self.actions_taken) >= self.config.max_complexity or \
               (len(self.exprs) > 0 and sp.expand(self.exprs[-1] - self.target_sp) == 0)

    def compute_rewards(self):
        """Computes rewards for the game, shaping them for better RL."""
        if not self.actions_taken:
            return []

        success = sp.expand(self.exprs[-1] - self.target_sp) == 0

        rewards = [-0.1] * len(self.actions_taken)

        if success:
            rewards[-1] = 100.0
        elif len(self.actions_taken) >= self.config.max_complexity:
            rewards[-1] = -10.0

        return rewards

    def is_valid_action(self, action_idx):
        max_nodes = self.config.n_variables + self.config.max_complexity + 1
        total_pairs = (max_nodes * (max_nodes + 1)) // 2
        max_possible_actions = total_pairs * 2
        if action_idx >= max_possible_actions: return False

        left, _, right = self._decode_action(action_idx)
        current_num_nodes = len(self.actions_taken) + self.config.n_variables + 1
        return left < current_num_nodes and right < current_num_nodes

    def _get_graph(self, actions):
        n_nodes = len(actions)
        node_features = []
        edges = []

        for i, action in enumerate(actions):
            action_type, input1_idx, input2_idx = action

            if action_type == "input":
                type_encoding = [1, 0, 0]
                value = i / max(1, self.config.n_variables)
            elif action_type == "constant":
                type_encoding = [0, 1, 0]
                value = 1.0
            else:
                type_encoding = [0, 0, 1]
                value = 1.0 if action_type == "multiply" else 0.0
                edges.append((input1_idx, i))
                edges.append((input2_idx, i))
            node_features.append(type_encoding + [value])

        x = torch.tensor(node_features, dtype=torch.float, device=self.device)

        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()

        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=n_nodes)

        data = Data(x=x, edge_index=edge_index)
        return data

    def _action_mask(self, actions):
        n_nodes = len(actions)
        max_nodes = self.config.n_variables + self.config.max_complexity + 1
        total_max_pairs = (max_nodes * (max_nodes + 1)) // 2
        max_possible_actions = total_max_pairs * 2

        mask = torch.zeros(1, max_possible_actions, dtype=torch.bool, device=self.device)

        for i in range(n_nodes):
            for j in range(i, n_nodes):
                for op in ["add", "multiply"]:
                    action_idx = encode_action(op, i, j, max_nodes)
                    if action_idx < max_possible_actions:
                        mask[0][action_idx] = True
        return mask

    def to(self, device):
        self.device = device
        return self

    def observe(self):
        variables = [("input", None, None) for _ in range(self.config.n_variables)]
        constants = variables + [("constant", None, None)]
        history = constants + self.actions_taken

        graph = self._get_graph(history)
        mask = self._action_mask(history)

        target_vec_observed = self.target_vec
        if target_vec_observed.dim() == 1:
           target_vec_observed = target_vec_observed.unsqueeze(0)

        return graph, target_vec_observed, [history], mask