import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import random
from generator import generate_random_circuit, generate_monomials_with_additive_indices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper function to encode actions with commutative operations
def encode_action(operation, node1_id, node2_id, max_nodes):
    """
    Encode an action so that commutative operations share the same action ID
    regardless of the order of input nodes.
    """
    # Ensure node IDs are valid
    if node1_id < 0 or node2_id < 0 or node1_id >= max_nodes or node2_id >= max_nodes:
        raise ValueError(f"Invalid node IDs: {node1_id}, {node2_id}. Must be between 0 and {max_nodes-1}")
        
    # For commutative operations, sort the node IDs
    if operation in ["add", "multiply"]:
        node1_id, node2_id = sorted([node1_id, node2_id])
    
    # Calculate the pair index safely
    # This formula is for unordered pairs (i,j) where i <= j
    pair_idx = (node1_id * (2 * max_nodes - node1_id - 1)) // 2 + (node2_id - node1_id)
    
    # Final action index: pair_index * num_operations + op_index
    action_idx = pair_idx * 2 + (1 if operation == "multiply" else 0)
    
    # Ensure the action index is valid
    if action_idx < 0:
        raise ValueError(f"Invalid action index: {action_idx}")
    
    return action_idx

def decode_action(action_idx, max_nodes):
    """
    Decode action index back to operation, node1_id, node2_id
    """
    # Safety check
    if action_idx < 0:
        raise ValueError(f"Invalid action index: {action_idx}")
        
    # Extract operation type (0 for add, 1 for multiply)
    op_type = action_idx % 2
    operation = "multiply" if op_type == 1 else "add"
    
    # Extract pair index
    pair_idx = action_idx // 2
    
    # Calculate the discriminant for the quadratic formula
    discriminant = (2 * max_nodes - 1)**2 - 8 * pair_idx
    
    # Safety check for negative discriminant
    if discriminant < 0:
        # Fallback to a valid node pair
        return operation, 0, 0
    
    # Find node indices from pair index using quadratic formula
    node1_id = int((2 * max_nodes - 1 - np.sqrt(discriminant)) / 2)
    
    # Ensure node1_id is valid
    node1_id = max(0, min(node1_id, max_nodes - 1))
    
    # Calculate node2_id
    node2_offset = pair_idx - (node1_id * (2 * max_nodes - node1_id - 1)) // 2
    node2_id = node1_id + node2_offset
    
    # Ensure node2_id is valid
    node2_id = max(0, min(node2_id, max_nodes - 1))
    
    return operation, node1_id, node2_id

class CircuitDataset(Dataset):
    def __init__(self, index_to_monomial, monomial_to_index, max_vector_size, config, size=10000):
        self.index_to_monomial = index_to_monomial
        self.monomial_to_index = monomial_to_index
        self.max_vector_size = max_vector_size
        self.config = config
        self.data = self.generate_data(size)

    def compute_value_score(self, current_poly, target_poly):
        current = torch.tensor(current_poly, dtype=torch.float)
        target = torch.tensor(target_poly, dtype=torch.float)

        # L1 distance
        l1_dist = torch.sum(torch.abs(current - target))
        dist_score = -l1_dist.item()  # Lower distance is better

        # Monomial overlap score
        overlap = torch.sum(torch.minimum(current, target))
        overlap_score = overlap.item() / max(torch.sum(target).item(), 1.0)

        # Weighted combination
        return 0.7 * overlap_score + 0.3 * dist_score
    
    def generate_data(self, size):
        """Generate training data: (intermediate_circuit, target_poly, circuit_history) -> next_action"""
        print("Generating supervised learning dataset")
        dataset = []
        
        n = self.config.n_variables
        d = self.config.max_complexity
        
        # Safety check for empty dataset
        if size <= 0:
            print("Warning: Requested dataset size is 0 or negative")
            return []
            
        num_circuits = max(1, size // self.config.max_complexity)
        
        # Generate random circuits
        for _ in range(num_circuits):
            try:
                # Generate a random circuit
                actions, polynomials, _, _ = generate_random_circuit(n, d, self.config.max_complexity, mod=self.config.mod, trim=self.config.trim_circuit)
                
                # Target polynomial is the final polynomial
                target_poly = torch.tensor(polynomials[-1], dtype=torch.float, device=device)
                
                # Go through all steps except the last one (which we want to predict)
                for i in range(n + 1, len(actions) - 1):  # Skip variable and constant nodes
                    # Current circuit is all actions up to i
                    current_actions = actions[:i]
                    
                    # Next action (ground truth)
                    next_action = actions[i]
                    next_op, next_node1_id, next_node2_id = next_action
                    
                    # Calculate available actions mask
                    max_nodes = self.config.n_variables + self.config.max_complexity + 1
                    available_mask = self.get_available_actions_mask(current_actions, max_nodes)
                    
                    try:
                        # Encode the action
                        action_idx = encode_action(next_op, next_node1_id, next_node2_id, max_nodes)
                        
                        # Compute value score for current circuit state
                        current_poly = polynomials[i]
                        value_score = self.compute_value_score(current_poly, target_poly)
    
                        # Store example
                        dataset.append({
                            'actions': current_actions,
                            'target_poly': target_poly,
                            'mask': available_mask,
                            'action': action_idx,
                            'value': value_score
                        })
                        
                        # If we have enough examples, return
                        if len(dataset) >= size:
                            print(f"Generated {len(dataset)} training examples")
                            return dataset
                    except Exception as e:
                        print(f"Error encoding action: {e}")
                        continue
            except Exception as e:
                print(f"Error generating circuit: {e}")
                continue
        
        print(f"Generated {len(dataset)} training examples")
        return dataset
    
    def get_available_actions_mask(self, actions, max_nodes):
        """Create a mask for available actions with a fixed size"""
        n_nodes = len(actions)
        
        # Calculate the maximum possible number of actions
        total_max_pairs = (max_nodes * (max_nodes + 1)) // 2  # Max possible combinations
        max_possible_actions = total_max_pairs * 2
        
        # Create mask with the maximum possible size
        mask = torch.zeros(max_possible_actions, dtype=torch.bool, device=device)
        
        # Set available actions to True - only for existing nodes
        for i in range(min(n_nodes, max_nodes)):
            for j in range(i, min(n_nodes, max_nodes)):
                for op_idx, op in enumerate(["add", "multiply"]):
                    try:
                        action_idx = encode_action(op, i, j, max_nodes)
                        if 0 <= action_idx < max_possible_actions:
                            mask[action_idx] = True
                    except Exception as e:
                        # Skip invalid actions without warning
                        pass
        
        return mask
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert current circuit to a graph representation for GNN
        circuit_graph = self.actions_to_graph(item['actions'])
        
        # Make sure tensors are on the correct device
        target_poly = item['target_poly'].to(device)
        mask = item['mask'].to(device)
        action = torch.tensor(item['action'], device=device)
        value = item['value']
        
        return circuit_graph, target_poly, item['actions'], mask, action, value
    
    def actions_to_graph(self, actions):
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
        x = torch.tensor(node_features, dtype=torch.float, device=device)
        
        # Handle the case with no edges
        if len(edges) == 0:
            edge_index = torch.tensor([[i, i] for i in range(n_nodes)], dtype=torch.long, device=device).t().contiguous()
        else:
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        
        # Create PyTorch Geometric data object
        data = Data(x=x, edge_index=edge_index)
        
        return data

# collate function for batching
def circuit_collate(batch):
    graphs = [item[0] for item in batch]
    target_polys = torch.stack([item[1] for item in batch])
    circuit_actions = [item[2] for item in batch]
    masks = torch.stack([item[3] for item in batch])
    actions = torch.stack([item[4] for item in batch])
    values = torch.tensor([item[5] for item in batch], dtype=torch.float, device=device)
    
    # Create a batched graph using PyTorch Geometric's Batch functionality
    batched_graph = Batch.from_data_list(graphs)
    
    return batched_graph, target_polys, circuit_actions, masks, actions, values