import sympy as sp
import torch
from torch_geometric.data import Data
import torch_geometric.utils
from utils import encode_action, vector_to_sympy # Keep vector_to_sympy for debugging
from generator import add_polynomials_vector, multiply_polynomials_vector, create_polynomial_vector
import math

class Game:
    def __init__(self, sympy_poly, vec_poly, config, index_to_monomial, monomial_to_index):
        """
        Initializes the Game state for RL.

        Args:
            sympy_poly: The target polynomial as a SymPy expression.
            vec_poly: The target polynomial as a vector.
            config: The configuration object.
            index_to_monomial: Mapping from index to monomial tuple.
            monomial_to_index: Mapping from monomial tuple to index.
        """
        self.config = config
        self.index_to_monomial = index_to_monomial
        self.monomial_to_index = monomial_to_index
        self.actions_taken = [] # List to store (op, node1, node2) tuples
        self.exprs = []         # List to store SymPy expressions (for checking & debugging)
        self.poly_vectors = []  # List to store polynomial vectors (for rewards & potential input)
        self.used_actions = []  # Tracks which nodes are used (for potential pruning/reward)

        # Base nodes: n variables + 1 constant
        self.symbols = sp.symbols([f"x{i}" for i in range(config.n_variables)]) + [sp.S.One]
        self.target_sp = sympy_poly
        self.target_vec = vec_poly.squeeze(0) # Ensure it's 1D

        self.device = 'cpu' # Keep game logic on CPU, only model needs GPU

        # Initialize base vectors with proper size
        n, d = config.n_variables, config.max_complexity * 2
        vector_size = len(index_to_monomial)
        
        # Create variable vectors
        for i in range(config.n_variables):
            var_vector = [0] * vector_size
            # Find the index for variable i (exponent tuple with 1 at position i)
            exponents = [0] * config.n_variables
            exponents[i] = 1
            mono_tuple = tuple(exponents)
            if mono_tuple in monomial_to_index:
                idx = monomial_to_index[mono_tuple]
                var_vector[idx] = 1
            self.poly_vectors.append(var_vector)
        
        # Create constant vector
        const_vector = [0] * vector_size
        zero_tuple = tuple([0] * config.n_variables)
        if zero_tuple in monomial_to_index:
            idx = monomial_to_index[zero_tuple]
            const_vector[idx] = 1
        self.poly_vectors.append(const_vector)

        # Initial L1 norm distance - ensure target vector is correct size
        target_size = len(self.target_vec)
        if target_size != vector_size:
            if target_size < vector_size:
                # Pad target vector
                self.target_vec = torch.cat([self.target_vec, torch.zeros(vector_size - target_size)])
            else:
                # Truncate target vector
                self.target_vec = self.target_vec[:vector_size]
        
        self.current_l1_dist = torch.linalg.norm(torch.tensor(self.poly_vectors[-1], dtype=torch.float) - self.target_vec, 1).item()

    def _decode_action(self, action_idx):
        """Decodes an action index into (node1, operation, node2)."""
        operation = "multiply" if action_idx % 2 == 1 else "add"
        pair_idx = action_idx // 2

        # Inverse of triangular number to find node1_id
        # pair_idx = (n*(n+1))/2 + m  => n = floor((-1 + sqrt(1 + 8*pair_idx))/2)
        node1_id = math.floor((math.sqrt(1 + 8 * pair_idx) - 1) / 2)
        node2_id = pair_idx - ((node1_id * (node1_id + 1)) // 2)

        # Ensure node1_id <= node2_id as per encoding
        return node1_id, operation, node2_id

    def _get_node_data(self, node_id):
        """Helper to get SymPy expression and vector for a node."""
        num_base_nodes = self.config.n_variables + 1
        if node_id < num_base_nodes:
            return self.symbols[node_id], self.poly_vectors[node_id]
        else:
            index = node_id - num_base_nodes
            return self.exprs[index], self.poly_vectors[num_base_nodes + index]

    def take_action(self, action_idx):
        """Applies an action, updating the state (SymPy, vector, history)."""
        node1_id, op, node2_id = self._decode_action(action_idx)

        expr1, vec1 = self._get_node_data(node1_id)
        expr2, vec2 = self._get_node_data(node2_id)

        n, d = self.config.n_variables, self.config.max_complexity * 2

        if op == "multiply":
            op_fn = lambda a, b: a * b
            new_vec = multiply_polynomials_vector(vec1, vec2, self.config.mod, self.index_to_monomial, n, d)
        else: # add
            op_fn = lambda a, b: a + b
            new_vec = add_polynomials_vector(vec1, vec2, self.config.mod)

        # Update SymPy (can be slow, use mainly for final check/debug)
        new_expr = sp.expand(op_fn(expr1, expr2))
        self.exprs.append(new_expr)

        # Update vectors
        self.poly_vectors.append(new_vec)

        # Update history
        self.actions_taken.append((op, node1_id, node2_id))
        # self.used_actions.append(used1.union(used2).union({len(self.actions_taken) - 1 + self.config.n_variables + 1})) # If needed

    def is_done(self):
        """Checks if the game reached a terminal state (success or max steps)."""
        success = False
        if self.exprs: # Check only if actions have been taken
             # Use SymPy for exact check (can be slow)
             # success = sp.expand(self.exprs[-1] - self.target_sp) == 0
             # Use Vector for faster check (might have mod issues if not careful)
             current_vec_t = torch.tensor(self.poly_vectors[-1], dtype=torch.float)
             target_vec_t = self.target_vec.float()
             # Ensure same length before comparing
             max_len = max(len(current_vec_t), len(target_vec_t))
             current_vec_t = torch.cat([current_vec_t, torch.zeros(max_len - len(current_vec_t))])
             target_vec_t = torch.cat([target_vec_t, torch.zeros(max_len - len(target_vec_t))])
             success = torch.equal(current_vec_t, target_vec_t)


        return len(self.actions_taken) >= self.config.max_complexity or success

    def compute_rewards(self):
        """
        Computes rewards, including shaping based on L1 distance change.
        """
        if not self.actions_taken:
            return []

        rewards = []
        prev_l1_dist = self.current_l1_dist

        # Calculate new L1 distance
        current_vec_t = torch.tensor(self.poly_vectors[-1], dtype=torch.float)
        target_vec_t = self.target_vec.float()
        max_len = max(len(current_vec_t), len(target_vec_t))
        current_vec_t = torch.cat([current_vec_t, torch.zeros(max_len - len(current_vec_t))])
        target_vec_t = torch.cat([target_vec_t, torch.zeros(max_len - len(target_vec_t))])
        new_l1_dist = torch.linalg.norm(current_vec_t - target_vec_t, 1).item()

        # Reward shaping: reward for reducing distance (scaled)
        # Scale factor can be tuned, e.g., 0.1
        distance_reward = (prev_l1_dist - new_l1_dist) * 0.05

        # Small step penalty to encourage shorter solutions
        step_penalty = -0.01

        reward = distance_reward + step_penalty
        self.current_l1_dist = new_l1_dist # Update for next step

        # Check for terminal state
        success = torch.equal(current_vec_t, target_vec_t)
        max_steps_reached = len(self.actions_taken) >= self.config.max_complexity

        if success:
            reward += 10.0 # Large reward for success
        elif max_steps_reached:
            reward -= 1.0  # Penalty for failing at max steps

        # Return a list containing only the reward for the last action
        # PPO usually works with rewards per step.
        # We need a reward for *each* step in the trajectory.
        # For simplicity here, we apply the main reward at the end,
        # but a better way is to provide shaped rewards at *each* step.
        # Let's try returning a list for the whole trajectory.
        # We need to store rewards per step. Let's make this simple:
        # All previous steps get step_penalty + distance_reward, last gets terminal.

        # For now, let's return just the *last* reward, PPO needs to handle this.
        # A better PPO would collect (s, a, r, s') and calculate rewards per step.
        # The current PPO loop seems to expect rewards *after* the action.
        # Let's return just the reward for the *current* action.

        # We return a list of rewards for all actions taken *so far*.
        # The PPO loop will take the last one.
        # This isn't ideal. Let's return just the last reward.
        # The PPO `train_ppo` loop *collects* rewards. So `compute_rewards`
        # should return the reward for the *last* action.
        all_rewards = [-0.01] * (len(self.actions_taken) -1) # Base penalty
        all_rewards.append(reward) # Add the last reward

        return all_rewards # PPO loop will take rewards[-1]


    def _get_graph(self, actions):
        """Builds the Torch Geometric graph from actions."""
        n_nodes = len(actions)
        node_features = []
        edges = []

        for i, (action_type, input1_idx, input2_idx) in enumerate(actions):
            if action_type == "input":
                type_encoding, value = [1, 0, 0], i / max(1, self.config.n_variables)
            elif action_type == "constant":
                type_encoding, value = [0, 1, 0], 1.0
            else:
                type_encoding, value = [0, 0, 1], 1.0 if action_type == "multiply" else 0.0
                edges.append((input1_idx, i)); edges.append((input2_idx, i))
            node_features.append(type_encoding + [value])

        x = torch.tensor(node_features, dtype=torch.float, device=self.device)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=n_nodes)
        return Data(x=x, edge_index=edge_index)

    def _action_mask(self, actions):
        """Creates a mask of available actions."""
        n_nodes = len(actions)
        max_nodes = self.config.n_variables + self.config.max_complexity + 1
        total_max_pairs = (max_nodes * (max_nodes + 1)) // 2
        max_possible_actions = total_max_pairs * 2
        mask = torch.zeros(1, max_possible_actions, dtype=torch.bool, device=self.device)

        # Allow actions using any currently available node (0 to n_nodes-1)
        for i in range(n_nodes):
            for j in range(i, n_nodes): # Ensure i <= j for canonical pairs
                for op in ["add", "multiply"]:
                    action_idx = encode_action(op, i, j, max_nodes)
                    if action_idx < max_possible_actions:
                        mask[0][action_idx] = True

                        # *** MODIFICATION: Removed restriction on multiplying by constant ***
                        # No need to explicitly disable any action here unless
                        # there's a strong reason (e.g., avoid 0*X if 0 isn't a node)
        return mask

    def to(self, device):
        """Moves internal tensors to the specified device (currently only CPU)."""
        self.device = device
        # If any tensors were stored, move them here.
        return self

    def observe(self):
        """Returns the current state observation for the model."""
        # Build the list of nodes (inputs + constant + operations)
        variables = [("input", None, None) for _ in range(self.config.n_variables)]
        constants = [("constant", None, None)]
        history = variables + constants + self.actions_taken

        # Get the graph representation
        graph = self._get_graph(history)
        # Get the action mask
        mask = self._action_mask(history)

        # Get the target vector (ensure it's 2D for batching)
        target_vec_observed = self.target_vec.unsqueeze(0) if self.target_vec.dim() == 1 else self.target_vec

        # Return: graph, target_poly_vector, list_of_actions, mask
        return graph, target_vec_observed, [history], mask