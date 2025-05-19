import sympy as sp
import torch
from torch_geometric.data import Data
from utils import encode_action
import torch
import math

class Game:
    def __init__(self, sympy_poly, vec_poly, config):
        self.config = config 

        # list of actions where each action is a tuple of (node index, op, node index)
        self.actions_taken = []

        # list of sympy expressions where the i-th entry represents the polynomial 
        # if actions_taken[i] were to be evaluated 
        self.exprs = []

        # list of sets where the i-th set represents all the nodes used by
        # actions_taken[i] and its predecessors. Can't use DSU here because 
        # actions may not be disjoint across trees. 
        self.used_actions = []

        self.symbols = sp.symbols([f"x{i}" for i in range(config.n_variables)]) + [1]

        # target polynomial   
        self.target_sp = sympy_poly
        self.target_vec = vec_poly

        self.device = 'cpu'

    def _decode_action(self, action_idx):
        operation = "multiply" if action_idx % 2 == 1 else 0
        pair_idx = (action_idx - operation) // 2
        node1_id = (math.sqrt(1 + 8 * pair_idx) - 1) // 2
        node2_id = pair_idx - ((node1_id * (node1_id + 1)) // 2)
        return node1_id, operation, node2_id
    
    def _expr_used(self, node):
        # check if the node was a past action
        if node > self.config.n_variables + 1:
            index = node - 1 - self.config.n_variables
            return self.exprs[index], self.used_actions[index]
        else:
            return self.symbols[node], set()

    def take_action(self, action_idx):
        mul = lambda a, b : a * b
        add = lambda a, b : a + b

        node1, op, node2 = self._decode_action(action_idx)
        expr1, used1 = self._expr_used(node1)
        expr2, used2 = self._expr_used(node2)
        op_fn = mul if op == "multiply" else add 

        self.actions_taken.append((op, node1, node2))
        self.exprs.append(op_fn(expr1, expr2))
        self.used_actions(used1.union(used2).union({ len(self.actions_taken) }))

    def is_done(self):
        # TODO: more efficient polynomial comparison with sampling?
        return len(self.actions_taken) >= self.config.max_complexity or \
               (False if len(self.exprs) == 0 else self.exprs[-1] - self.target_sp == 0)
    
    def compute_rewards(self):
        # TODO: add term to prevent +1 -1 reward hacking
        success = self.exprs[-1] - self.target_sp == 0
        actions_used = self.used_actions[-1]
        rewards = list(map(lambda i: (0.5 if success else 0.2) if i in actions_used else -1), len(self.actions_taken))

        # additional reward for successful computation of polynomial
        if success:
            rewards[-1] = 1

        return rewards
    
    def is_valid_action(self, action_idx):
        left, _, right = self._decode_action(action_idx)
        return left < len(self.actions_taken) and right < len(self.actions_taken)
    
    def _get_graph(self, actions):
        """Convert actions to a PyTorch Geometric graph"""
        # Create node features
        n_nodes = len(actions)
        node_features = []
        edges = []
        
        for i, action in enumerate(actions):
            action_type, input1_idx, input2_idx = action
            
            # Create node feature
            if action_type == "input":
                # Input node
                type_encoding = [1, 0, 0]  # One-hot for input
                value = i / max(1, self.config.n_variables)  # Normalize index
            elif action_type == "constant":
                # Constant node
                type_encoding = [0, 1, 0]  # One-hot for constant
                value = 1.0  # Constant value
            else:  # operation
                # Operation node
                type_encoding = [0, 0, 1]  # One-hot for operation
                value = 1.0 if action_type == "multiply" else 0.0  # 1 for multiply, 0 for add
                
                # Add edges from inputs to this node
                edges.append((input1_idx, i))
                edges.append((input2_idx, i))
            
            # Combine features
            node_features.append(type_encoding + [value])
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float, device=self.device)
        
        # Handle the case with no edges
        if len(edges) == 0:
            edge_index = torch.tensor([[i, i] for i in range(n_nodes)], dtype=torch.long, device=self.device).t().contiguous()
        else:
            edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
        
        # Create PyTorch Geometric data object
        data = Data(x=x, edge_index=edge_index)
        
        return data
    
    def _action_mask(self, actions):
        """Create a mask for available actions with a fixed size"""
        n_nodes = len(actions)
        max_nodes = self.config.n_variables + self.config.max_complexity + 1
        
        # Calculate the maximum possible number of actions
        total_max_pairs = (max_nodes * (max_nodes + 1)) // 2  # Max possible combinations
        max_possible_actions = total_max_pairs * 2
        
        # Create mask with the maximum possible size
        mask = torch.zeros(1, max_possible_actions, dtype=torch.bool, device=self.device)
        
        # Set available actions to True
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                for _, op in enumerate(["add", "multiply"]):
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
        return self._get_graph(history), self.target_vec, [history], self._action_mask(history)