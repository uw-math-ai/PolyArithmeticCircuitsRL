import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import deque
import copy
import os
import tqdm
import traceback
from torch.utils.data import Dataset, DataLoader

# Add this right after imports, around line 40-50
class SimpleData:
    def __init__(self, x, edge_index, batch=None):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom implementation of global mean pooling to avoid torch-scatter dependency
def custom_global_mean_pool(x, batch):
    """
    Custom implementation of global_mean_pool without using torch-scatter
    """
    batch_size = batch.max().item() + 1
    result = torch.zeros(batch_size, x.size(1), device=x.device)
    
    # Count nodes per graph
    ones = torch.ones(batch.size(0), device=batch.device)
    count = torch.zeros(batch_size, device=batch.device)
    count.scatter_add_(0, batch, ones)
    
    # Sum node features per graph
    for i in range(batch_size):
        mask = (batch == i)
        if mask.any():
            result[i] = x[mask].mean(dim=0)
    
    return result

class Config:
    def __init__(self):
        # Circuit parameters
        self.n_variables = 3           # Number of variables
        self.max_complexity = 10       # Maximum complexity of circuits
        self.mod = 50                  # Modulo for coefficients
        self.trim_circuit = True       # Whether to trim unused actions
        
        # Model architecture
        self.hidden_dim = 256          # Hidden dimension for neural networks
        self.embedding_dim = 128       # Embedding dimension for nodes
        self.num_gnn_layers = 4        # Number of GNN layers
        self.num_transformer_layers = 4 # Number of transformer layers
        self.transformer_heads = 8     # Number of attention heads
        self.dropout = 0.1             # Dropout rate
        
        # Training parameters
        self.learning_rate = 0.0005    # Learning rate
        self.batch_size = 64           # Batch size for training
        self.train_size = 5000         # Number of training examples
        self.epochs = 300              # Number of training epochs
        self.max_circuit_length = 100  # Maximum length for circuit sequence
        self.weight_decay = 1e-5       # L2 regularization
        
        # RL parameters
        self.rl_episodes = 2000        # Number of RL episodes
        self.num_simulations = 200     # Number of MCTS simulations per step
        self.c_puct = 1.0              # Exploration constant
        self.dirichlet_alpha = 0.3     # Dirichlet noise parameter
        self.dirichlet_epsilon = 0.25  # Weight of Dirichlet noise
        self.value_loss_weight = 1.0   # Weight of value loss in total loss
        self.policy_loss_weight = 1.0  # Weight of policy loss in total loss
        self.use_gumbel = True         # Whether to use Gumbel sampling for policy
        
        # Polynomial parameters
        self.polynomial_degree = 5     # Maximum degree of polynomials

# Create a basic config
config = Config()

class CircuitHistoryEncoder(nn.Module):
    """Encodes the history of circuit actions into embeddings"""
    def __init__(self, embedding_dim, dropout=0.1):
        super(CircuitHistoryEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Embeddings for operations and node indices
        self.operation_embedding = nn.Embedding(3, embedding_dim)  # add, multiply, none
        self.node_embedding = nn.Linear(1, embedding_dim)  # Continuous embedding for node indices
        
        # Combine embeddings with attention
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=4, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def encode_circuit_actions(self, actions):
        """Convert actions to token format for embedding"""
        tokens = []
        for i, (op, node1, node2) in enumerate(actions):
            if op == "input":
                op_id = 2  # None operation
                node1_id = i
                node2_id = 0  # Placeholder
            elif op == "constant":
                op_id = 2  # None operation
                node1_id = 0  # Placeholder
                node2_id = 0  # Placeholder
            else:
                op_id = 0 if op == "add" else 1  # 0 for add, 1 for multiply
                node1_id = node1
                node2_id = node2
            
            tokens.append((op_id, node1_id, node2_id))
        
        return tokens
    
    def forward(self, tokens):
        """
        Args:
            tokens: List of (operation, node1, node2) tuples
        """
        if not tokens:
            return torch.zeros(0, self.embedding_dim, device=device)
            
        embeddings = []
        for op_id, node1_id, node2_id in tokens:
            # Get operation embedding
            op_embed = self.operation_embedding(torch.tensor(op_id, device=device))
            
            # Get node embeddings (using continuous representation)
            node1_embed = self.node_embedding(torch.tensor([[node1_id]], dtype=torch.float, device=device)).squeeze(0)
            node2_embed = self.node_embedding(torch.tensor([[node2_id]], dtype=torch.float, device=device)).squeeze(0)
            
            # Combine embeddings (using addition and layer norm)
            combined = op_embed + node1_embed + node2_embed
            embeddings.append(combined)
        
        # Apply transformer-like self-attention
        if len(embeddings) > 1:
            embeddings_tensor = torch.stack(embeddings)
            
            # Self-attention
            attended, _ = self.attention(
                embeddings_tensor, embeddings_tensor, embeddings_tensor
            )
            
            # Add & Norm
            attended = self.layer_norm1(embeddings_tensor + self.dropout(attended))
            
            # Feed-forward
            ff_output = self.feed_forward(attended)
            
            # Add & Norm
            embeddings_tensor = self.layer_norm2(attended + self.dropout(ff_output))
            
            return embeddings_tensor
        else:
            return torch.stack(embeddings)

# Simple GCN layer implementation that doesn't rely on specialized PyTorch Geometric operations
class SimpleGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # Compute adjacency matrix from edge_index
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        
        if edge_index.numel() > 0:
            # Add self-loops
            self_loops = torch.arange(num_nodes, device=edge_index.device)
            self_loops = torch.stack([self_loops, self_loops], dim=0)
            
            # Combine edge_index with self-loops
            edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)
            
            # Fill adjacency matrix
            adj[edge_index_with_loops[0], edge_index_with_loops[1]] = 1
            
            # Normalize adjacency matrix (D^(-1/2) A D^(-1/2))
            degrees = adj.sum(dim=1)
            degree_matrix_inv_sqrt = torch.diag(torch.pow(degrees + 1e-8, -0.5))
            adj_normalized = torch.mm(torch.mm(degree_matrix_inv_sqrt, adj), degree_matrix_inv_sqrt)
            
            # Message passing
            x = self.linear(torch.mm(adj_normalized, x))
        else:
            # If no edges, just apply linear transformation
            x = self.linear(x)
            
        return x

class EnhancedGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers=3, dropout=0.1):
        super(EnhancedGNN, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Use custom GCN layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(SimpleGCNLayer(hidden_dim, hidden_dim))
        
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, data):
        try:
            x, edge_index = data.x, data.edge_index
            
            # Project input to hidden dimension
            x = self.activation(self.input_projection(x))
            
            # Apply GNN layers with residual connections and layer normalization
            for i, conv in enumerate(self.conv_layers):
                identity = x
                x = conv(x, edge_index)
                x = self.activation(x)
                x = self.dropout(x)
                
                # Only add residual connection if dimensions match
                if identity.shape == x.shape:
                    x = x + identity
                
                x = self.layer_norms[i](x)
            
            # Project to output dimension
            x = self.output_projection(x)
            
            return x
        except Exception as e:
            print(f"GNN forward pass error: {e}")
            return torch.zeros(data.x.size(0), self.output_projection.out_features, device=device)

class AdvancedCircuitBuilder(nn.Module):
    def __init__(self, config, max_poly_vector_size):
        super(AdvancedCircuitBuilder, self).__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # Enhanced GNN component
        self.gnn = EnhancedGNN(
            input_dim=4,  # [node_type_one_hot (3), value_or_operation]
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_layers=config.num_gnn_layers,
            dropout=config.dropout
        )
        
        # Circuit history encoder
        self.circuit_encoder = CircuitHistoryEncoder(
            config.embedding_dim, 
            dropout=config.dropout
        )
        
        # Polynomial embedding with deeper network
        self.polynomial_embedding = nn.Sequential(
            nn.Linear(max_poly_vector_size, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )
        
        # Transformer components
        from positional_encoding import PositionalEncoding
        self.positional_encoding = PositionalEncoding(
            config.embedding_dim, 
            config.max_circuit_length
        )
        
        # Create transformer decoder layers with improved architecture
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embedding_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.hidden_dim * 4,  # Increased capacity
            dropout=config.dropout,
            activation="relu"  # Changed from "gelu" for compatibility
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_transformer_layers
        )
        
        # Define maximum actions
        max_nodes = config.n_variables + config.max_complexity + 1
        total_pairs = (max_nodes * (max_nodes + 1)) // 2  # Combinations with replacement
        max_actions = total_pairs * 2  # Unordered pairs * 2 operations
        
        # Action head with deeper network
        self.action_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, max_actions)
        )
        
        # Value head with deeper network
        self.value_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
        )
        
        # Special tokens
        self.output_token = nn.Parameter(torch.randn(1, 1, config.embedding_dim))
        
        # Auxiliary heads for better training
        self.aux_polynomial_head = nn.Linear(config.embedding_dim, max_poly_vector_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters for better training"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'output_token' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
    
    def forward(self, batched_graph, target_polynomials, circuit_actions, available_actions_masks=None):
        batch_size = target_polynomials.size(0)
        
        # Process circuit with GNN
        node_embeddings = self.gnn(batched_graph)
        # Use custom global pooling
        graph_embeddings = custom_global_mean_pool(node_embeddings, batched_graph.batch)
        
        # Process target polynomials
        poly_embeddings = self.polynomial_embedding(target_polynomials)
        
        # Process circuit histories for each example in the batch
        circuit_embeddings_list = []
        max_seq_len = 0
        
        for i in range(batch_size):
            tokens = self.circuit_encoder.encode_circuit_actions(circuit_actions[i])
            embeddings = self.circuit_encoder(tokens)
            circuit_embeddings_list.append(embeddings)
            max_seq_len = max(max_seq_len, embeddings.size(0))
        
        # Pad sequences to same length
        padded_circuit_embeddings = []
        for emb in circuit_embeddings_list:
            if emb.size(0) < max_seq_len:
                padding = torch.zeros(max_seq_len - emb.size(0), self.embedding_dim, device=device)
                emb = torch.cat([emb, padding], dim=0)
            padded_circuit_embeddings.append(emb)
        
        # Stack circuit embeddings (seq_len, batch_size, embedding_dim)
        circuit_embeddings = torch.stack(padded_circuit_embeddings, dim=1)
        
        # Add positional encoding
        circuit_embeddings = self.positional_encoding(circuit_embeddings)
        
        # Create memory tensor for transformer (combine graph, polynomial and circuit embeddings)
        memory = torch.cat([
            graph_embeddings.unsqueeze(0),  # (1, batch_size, embedding_dim)
            poly_embeddings.unsqueeze(0),   # (1, batch_size, embedding_dim)
            circuit_embeddings              # (seq_len, batch_size, embedding_dim)
        ], dim=0)  
        
        # Query tensor (output token for each example in batch)
        query = self.output_token.expand(-1, batch_size, -1)  # (1, batch_size, embedding_dim)
        
        # Apply transformer decoder
        output = self.transformer_decoder(
            tgt=query,
            memory=memory,
            memory_key_padding_mask=None,
            tgt_key_padding_mask=None
        )
        
        # Get action logits
        output_squeezed = output.squeeze(0)
        action_logits = self.action_head(output_squeezed)  # (batch_size, max_actions)

        # Value prediction
        value_pred = self.value_head(output_squeezed).squeeze(-1)
        
        # Auxiliary polynomial prediction
        aux_poly_pred = self.aux_polynomial_head(output_squeezed)
        
        # Apply masks to invalid actions
        if available_actions_masks is not None:
            action_logits = action_logits.masked_fill(~available_actions_masks, float('-inf'))
        
        return action_logits, value_pred, aux_poly_pred

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
        """Compute a sophisticated value score for circuit state"""
        current = torch.tensor(current_poly, dtype=torch.float)
        target = torch.tensor(target_poly, dtype=torch.float)
        
        # Get non-zero coefficients in both polynomials
        nonzero_current = (current != 0).nonzero().flatten()
        nonzero_target = (target != 0).nonzero().flatten()
        
        # Calculate coefficient matches
        matches = 0
        for idx in nonzero_target:
            if idx < len(current) and current[idx] == target[idx]:
                matches += 1
                
        # Calculate exact match ratio
        match_ratio = matches / max(1, len(nonzero_target))
        
        # Calculate normalized L1 distance
        max_val = max(current.abs().max().item(), target.abs().max().item())
        if max_val > 0:
            l1_dist = torch.norm(current - target, p=1).item() / max_val
        else:
            l1_dist = 0
            
        # Calculate coefficient similarity (reward partial progress)
        coef_sim = 0
        for idx in nonzero_target:
            if idx < len(current) and current[idx] != 0:
                # How close is the coefficient to the target?
                closeness = 1.0 - min(1.0, abs(current[idx] - target[idx]) / max(1, abs(target[idx])))
                coef_sim += closeness
        
        coef_sim = coef_sim / max(1, len(nonzero_target))
        
        # Calculate distance to completion (decreases as we get closer)
        steps_remaining = 1.0 - match_ratio
        
        # Combine metrics into value score (-1 to 1 range, higher is better)
        # Give more weight to exact matches
        value_score = 0.6 * match_ratio + 0.2 * coef_sim - 0.2 * l1_dist
        
        return value_score
    
    def generate_data(self, size):
        """Generate training data: (intermediate_circuit, target_poly, circuit_history) -> next_action"""
        print("Generating supervised learning dataset")
        dataset = []
        
        n = self.config.n_variables
        d = min(self.config.max_complexity, self.config.polynomial_degree)
        
        # Safety check for empty dataset
        if size <= 0:
            print("Warning: Requested dataset size is 0 or negative")
            return []
            
        num_circuits = max(1, size // (self.config.max_complexity // 2))
        
        # Generate random circuits
        successful_examples = 0
        for _ in range(num_circuits):
            try:
                # Generate a random circuit with varying complexity
                from generator import generate_random_circuit
                circuit_complexity = random.randint(d // 2, d)
                actions, polynomials, _, _ = generate_random_circuit(
                    n, d, circuit_complexity, 
                    mod=self.config.mod, 
                    trim=self.config.trim_circuit
                )
                
                if len(actions) <= n + 1:  # Only variables and constant
                    continue
                    
                # Target polynomial is the final polynomial
                target_poly = torch.tensor(polynomials[-1], dtype=torch.float, device=device)
                
                # Go through steps to predict actions
                for i in range(n + 1, len(actions) - 1):
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
                        current_poly = polynomials[i-1]  # Current polynomial
                        value_score = self.compute_value_score(current_poly, target_poly)
                        
                        # Store example
                        dataset.append({
                            'actions': current_actions,
                            'target_poly': target_poly,
                            'mask': available_mask,
                            'action': action_idx,
                            'value': value_score,
                            'current_poly': torch.tensor(current_poly, dtype=torch.float, device=device)
                        })
                        
                        successful_examples += 1
                        
                        # If we have enough examples, return
                        if successful_examples >= size:
                            print(f"Generated {len(dataset)} training examples")
                            return dataset
                    except Exception as e:
                        # Skip problematic examples
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
                    except Exception:
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
        current_poly = item.get('current_poly', torch.zeros_like(target_poly)).to(device)
        
        return circuit_graph, target_poly, item['actions'], mask, action, value, current_poly
    
    def actions_to_graph(self, actions):
        """Convert actions to a PyTorch Geometric graph data structure"""
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
                edges.append([input1_idx, i])
                edges.append([input2_idx, i])
            
            # Combine features
            node_features.append(type_encoding + [value])
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float, device=device)
        
        # Handle the case with no edges
        if len(edges) == 0:
            edge_index = torch.tensor([[], []], dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        
        # Create a simple data class to mimic PyTorch Geometric Data
        
                
        data = SimpleData(x, edge_index)
        return data

# Custom Batch class to avoid PyTorch Geometric dependency
class Batch:
    @staticmethod
    def from_data_list(data_list):
        """Create a batch from a list of SimpleData objects"""
        batch_size = len(data_list)
        if batch_size == 0:
            return None
            
        # Get total number of nodes
        total_nodes = sum(data.x.size(0) for data in data_list)
        
        # Create batch index tensor
        batch = torch.zeros(total_nodes, dtype=torch.long, device=device)
        
        # Concatenate node features
        x_list = []
        edge_index_list = []
        node_offset = 0
        
        for i, data in enumerate(data_list):
            num_nodes = data.x.size(0)
            
            # Add node features
            x_list.append(data.x)
            
            # Add batch indices
            batch[node_offset:node_offset + num_nodes] = i
            
            # Add edges with offset
            if data.edge_index.numel() > 0:
                edges = data.edge_index.clone()
                edges += node_offset
                edge_index_list.append(edges)
            
            # Update offset
            node_offset += num_nodes
        
        # Combine all tensors
        x = torch.cat(x_list, dim=0)
        
        if edge_index_list:
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # Create output object
        result = SimpleData(x, edge_index, batch)
        return result

def circuit_collate(batch):
    """Collate function for DataLoader"""
    graphs = [item[0] for item in batch]
    target_polys = torch.stack([item[1] for item in batch])
    circuit_actions = [item[2] for item in batch]
    masks = torch.stack([item[3] for item in batch])
    actions = torch.stack([item[4] for item in batch])
    values = torch.tensor([item[5] for item in batch], dtype=torch.float, device=device)
    current_polys = torch.stack([item[6] for item in batch])
    
    # Create a batched graph
    batched_graph = Batch.from_data_list(graphs)
    
    return batched_graph, target_polys, circuit_actions, masks, actions, values, current_polys

class Node:
    """Monte Carlo Tree Search Node for AlphaZero"""
    def __init__(self, prior=0, parent=None):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.parent = parent
        self.is_expanded = False
    
    def expanded(self):
        return self.is_expanded
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_action(self, temperature=1.0):
        """Select an action based on visit counts"""
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())
        
        if temperature == 0:
            # Choose the action with the highest visit count
            action_idx = np.argmax(visit_counts)
            return actions[action_idx]
        
        # Apply temperature
        visit_counts = visit_counts ** (1 / temperature)
        # Add small epsilon to avoid division by zero
        visit_counts = visit_counts / (np.sum(visit_counts) + 1e-8)
        
        # Sample action based on the probability distribution
        action_idx = np.random.choice(len(actions), p=visit_counts)
        return actions[action_idx]

class EnhancedMCTS:
    """Enhanced Monte Carlo Tree Search for arithmetic circuit construction"""
    def __init__(self, model, config, c_puct=1.0):
        self.model = model
        self.config = config
        self.c_puct = c_puct
        self.max_depth = config.max_complexity
        self.root = None
        self.dirichlet_alpha = config.dirichlet_alpha
        self.dirichlet_epsilon = config.dirichlet_epsilon
        self.use_gumbel = config.use_gumbel
    
    def ucb_score(self, parent, child):
        """Calculate the UCB score for a node"""
        # Exploitation term
        if child.visit_count > 0:
            # Use negative value since we want to minimize distance to target
            q_value = -child.value()
        else:
            q_value = 0
        
        # Exploration bonus (AlphaZero style)
        u_value = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        
        return q_value + u_value
    
    def add_dirichlet_noise(self, priors, valid_actions):
        """Add Dirichlet noise to root node priors for exploration"""
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_actions))
        
        # Create new dictionary with noise added
        noisy_priors = {}
        for i, action in enumerate(valid_actions):
            noisy_priors[action] = (1 - self.dirichlet_epsilon) * priors[action] + self.dirichlet_epsilon * noise[i]
        
        return noisy_priors

    def search(self, state, num_simulations=100):
        """Perform MCTS search"""
        circuit_graph, target_poly, current_actions, available_mask, polynomials = state
        
        if self.root is None:
            self.root = Node()
        
        for _ in range(num_simulations):
            node = self.root
            
            # Make a copy of the current state
            sim_actions = current_actions.copy()
            sim_polys = polynomials.copy()
            sim_mask = available_mask.clone()
            
            # Selection phase - traverse tree until we reach a leaf node
            search_path = [node]
            
            while node.expanded() and node.children:
                # Select best child according to UCB score
                max_ucb = float('-inf')
                best_action = None
                best_child = None
                
                for action, child in node.children.items():
                    ucb = self.ucb_score(node, child)
                    if ucb > max_ucb:
                        max_ucb = ucb
                        best_action = action
                        best_child = child
                
                node = best_child
                search_path.append(node)
                
                # Apply action to simulation state
                sim_actions, sim_polys, sim_mask = self.apply_action(
                    best_action, sim_actions, sim_polys, sim_mask, self.config.mod
                )
            
            # Expansion phase - expand the leaf node if it hasn't been expanded
            if not node.expanded():
                batched_graph = Batch.from_data_list([circuit_graph])
                target_poly_tensor = target_poly.unsqueeze(0)
                sim_actions_list = [sim_actions]
                sim_mask_tensor = sim_mask.unsqueeze(0)
                
                with torch.no_grad():
                    action_probs, value, _ = self.model(
                        batched_graph, target_poly_tensor, sim_actions_list, sim_mask_tensor
                    )
                
                # Get valid actions and their probabilities
                valid_actions = sim_mask.nonzero().squeeze(-1)
                
                # Safety check - ensure we have valid actions
                if valid_actions.numel() > 0:
                    # Get probabilities for valid actions only
                    valid_indices = valid_actions.cpu().numpy()
                    valid_probs = action_probs[0][valid_indices]
                    
                    # Option to use Gumbel sampling for better exploration
                    if self.use_gumbel and node == self.root:
                        # Add Gumbel noise
                        gumbel_noise = -torch.log(-torch.log(torch.rand_like(valid_probs) + 1e-8) + 1e-8)
                        valid_probs = valid_probs + gumbel_noise
                    
                    probs = torch.softmax(valid_probs, dim=0).cpu().numpy()
                    
                    # Create dict mapping actions to priors
                    priors = {int(valid_indices[i]): probs[i] for i in range(len(valid_indices))}
                    
                    # Add Dirichlet noise at root for more exploration
                    if node == self.root:
                        priors = self.add_dirichlet_noise(priors, list(valid_indices))
                    
                    # Create children nodes
                    for action, prior in priors.items():
                        node.children[action] = Node(prior=prior, parent=node)
                
                node.is_expanded = True
                
                # Use the value prediction from the model
                value = value.item()
            else:
                # If node is already expanded but has no children (terminal), assign value
                if len(node.children) == 0:
                    # Calculate similarity to target
                    final_poly = sim_polys[-1] if sim_polys else None
                    if final_poly:
                        sim_score = self.compute_similarity(final_poly, target_poly.cpu().numpy())
                        value = sim_score
                    else:
                        value = -1.0  # Bad state, no polynomial
            
            # Backpropagation - update values for all nodes traversed
            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
        
        return self.root

    def apply_action(self, action_idx, actions, polynomials, mask, mod):
        """Apply an action to the current state"""
        # Decode action
        max_nodes = self.config.n_variables + self.config.max_complexity + 1
        operation, node1_idx, node2_idx = decode_action(action_idx, max_nodes)
        
        # Apply operation
        from generator import add_polynomials_vector, multiply_polynomials_vector
        if operation == "add":
            new_poly = add_polynomials_vector(polynomials[node1_idx], polynomials[node2_idx], mod)
        else:  # multiply
            new_poly = multiply_polynomials_vector(polynomials[node1_idx], polynomials[node2_idx], mod)
        
        # Update actions and polynomials
        new_actions = actions + [(operation, node1_idx, node2_idx)]
        new_polynomials = polynomials + [new_poly]
        
        # Update mask
        new_mask = self.update_mask(new_actions, max_nodes)
        
        return new_actions, new_polynomials, new_mask
    
    def compute_similarity(self, poly1, poly2):
        """Compute similarity between two polynomials"""
        poly1_tensor = torch.tensor(poly1, dtype=torch.float)
        poly2_tensor = torch.tensor(poly2, dtype=torch.float)
        
        # Debugging output - truncate long vectors for readability
        def truncate_vector(vec, max_len=20):
            if len(vec) <= max_len:
                return vec
            else:
                nonzero_indices = (vec != 0).nonzero().flatten().tolist()
                if not nonzero_indices:
                    return vec[:max_len].tolist() + ["..."]
                
                important_elements = []
                for idx in nonzero_indices:
                    if idx < len(vec):
                        important_elements.append(f"idx {idx}: {vec[idx]}")
                
                return f"nonzero elements: {important_elements}"
        
        # Check if either polynomial is empty
        if torch.sum(poly1_tensor) == 0 or torch.sum(poly2_tensor) == 0:
            return 0.0
            
        # Get non-zero coefficients in both polynomials
        nonzero1 = set((poly1_tensor != 0).nonzero().flatten().cpu().numpy())
        nonzero2 = set((poly2_tensor != 0).nonzero().flatten().cpu().numpy())
        
        # Calculate coefficient matches
        coef_matches = 0
        for idx in nonzero1.intersection(nonzero2):
            if poly1_tensor[idx] == poly2_tensor[idx]:
                coef_matches += 1
                
        # Calculate Jaccard similarity of non-zero positions
        jaccard = 0.0
        if nonzero1 or nonzero2:  # Avoid division by zero
            jaccard = len(nonzero1.intersection(nonzero2)) / max(1, len(nonzero1.union(nonzero2)))
            
        # Calculate normalized L1 similarity (coefficient-based)
        max_val = max(torch.max(torch.abs(poly1_tensor)).item(), torch.max(torch.abs(poly2_tensor)).item())
        if max_val > 0:
            l1_similarity = 1.0 - torch.mean(torch.abs(poly1_tensor - poly2_tensor)).item() / max_val
        else:
            l1_similarity = 1.0
        
        # Calculate structural similarity (cosine)
        # Normalize
        poly1_norm = poly1_tensor / (torch.norm(poly1_tensor, p=2) + 1e-8)
        poly2_norm = poly2_tensor / (torch.norm(poly2_tensor, p=2) + 1e-8)
        
        # Cosine similarity
        cosine_similarity = torch.sum(poly1_norm * poly2_norm).item()
        
        # Combine the metrics with weights
        coefficient_match_ratio = coef_matches / max(1, len(nonzero2))
        
        # Final similarity score - prioritize exact coefficient matches
        combined_similarity = (0.4 * coefficient_match_ratio + 
                              0.3 * jaccard + 
                              0.2 * cosine_similarity + 
                              0.1 * l1_similarity)
        
        return combined_similarity
    
    def update_mask(self, actions, max_nodes):
        """Update action mask based on current actions"""
        n_nodes = len(actions)
        
        # Calculate the maximum possible number of actions
        total_max_pairs = (max_nodes * (max_nodes + 1)) // 2
        max_possible_actions = total_max_pairs * 2
        
        # Create mask
        mask = torch.zeros(max_possible_actions, dtype=torch.bool, device=device)
        
        # Set available actions to True - only for existing nodes
        for i in range(min(n_nodes, max_nodes)):
            for j in range(i, min(n_nodes, max_nodes)):
                # Skip if node indices would be out of bounds
                if i >= max_nodes or j >= max_nodes:
                    continue
                    
                for op_idx, op in enumerate(["add", "multiply"]):
                    try:
                        action_idx = encode_action(op, i, j, max_nodes)
                        if 0 <= action_idx < max_possible_actions:
                            mask[action_idx] = True
                    except Exception:
                        # Skip invalid actions without printing warnings
                        pass
        
        return mask

class ReplayBuffer:
    """Advanced experience replay buffer for reinforcement learning"""
    def __init__(self, capacity=10000, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Prioritized experience replay parameters
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta_start  # Importance sampling correction
        self.beta_increment = (beta_end - beta_start) / beta_frames
        self.max_priority = 1.0
    
    def add(self, experience, priority=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Set priority
        if priority is None:
            priority = self.max_priority
            
        if self.position < len(self.priorities):
            self.priorities[self.position] = priority
            
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < self.size:
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size):
        """Sample a batch of experiences with prioritization"""
        if self.size == 0:
            return None
            
        if batch_size > self.size:
            batch_size = self.size
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Sample indices based on priorities
        if self.alpha > 0:
            priorities = self.priorities[:self.size] ** self.alpha
            prob_sum = np.sum(priorities)
            if prob_sum > 0:
                probs = priorities / prob_sum
            else:
                probs = np.ones(self.size) / self.size
                
            indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
            
            # Calculate importance sampling weights
            weights = (self.size * probs[indices]) ** (-self.beta)
            weights = weights / weights.max()
        else:
            # Uniform sampling
            indices = np.random.choice(self.size, batch_size, replace=False)
            weights = np.ones(batch_size)
        
        # Extract experiences
        batch = [self.buffer[idx] for idx in indices]
        
        # Unpack the batch
        graphs, target_polys, circuit_actions, masks, policy_targets, value_targets, current_polys = zip(*batch)
        
        # Create batched data
        batched_graph = Batch.from_data_list(graphs)
        target_polys_tensor = torch.stack(target_polys)
        masks_tensor = torch.stack(masks)
        policy_targets_tensor = torch.stack(policy_targets)
        value_targets_tensor = torch.tensor(value_targets, dtype=torch.float, device=device)
        current_polys_tensor = torch.stack(current_polys)
        weights_tensor = torch.tensor(weights, dtype=torch.float, device=device)
        
        return (
            batched_graph, 
            target_polys_tensor, 
            circuit_actions, 
            masks_tensor, 
            policy_targets_tensor, 
            value_targets_tensor,
            current_polys_tensor,
            weights_tensor,
            indices
        )
    
    def __len__(self):
        return self.size

def train_supervised(model, dataset, config):
    """Train model with supervised learning"""
    # Create data loader with custom collate function
    data_loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=circuit_collate
    )
    
    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(config.epochs):
        total_action_loss = 0
        total_value_loss = 0
        total_aux_loss = 0
        action_correct = 0
        total = 0
        value_mse = 0
        
        # Process batches
        model.train()
        progress_bar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}")
        for batched_graph, target_polys, circuit_actions, masks, actions, values, current_polys in progress_bar:
            optimizer.zero_grad()
            
            # Forward pass batch
            action_logits, value_preds, aux_poly_preds = model(
                batched_graph, target_polys, circuit_actions, masks
            )
            
            # Calculate action loss with label smoothing for better generalization
            action_loss = F.cross_entropy(
                action_logits, 
                actions, 
                label_smoothing=0.1
            )
            
            # Calculate value loss
            value_loss = F.mse_loss(value_preds, values)
            
            # Calculate auxiliary polynomial prediction loss
            aux_loss = F.mse_loss(aux_poly_preds, target_polys)
            
            # Combined loss with weighting
            loss = action_loss + value_loss + 0.1 * aux_loss
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            pred_actions = torch.argmax(action_logits, dim=1)
            action_correct += (pred_actions == actions).sum().item()
            value_mse += F.mse_loss(value_preds, values, reduction='sum').item()
            total += actions.size(0)
            
            total_action_loss += action_loss.item() * actions.size(0)
            total_value_loss += value_loss.item() * actions.size(0)
            total_aux_loss += aux_loss.item() * actions.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'action_loss': action_loss.item(),
                'value_loss': value_loss.item(),
                'action_acc': (pred_actions == actions).float().mean().item()
            })
        
        # Step the scheduler
        scheduler.step()
        
        # Calculate epoch metrics
        avg_action_loss = total_action_loss / max(1, len(dataset))
        avg_value_loss = total_value_loss / max(1, len(dataset))
        avg_aux_loss = total_aux_loss / max(1, len(dataset))
        action_accuracy = 100 * action_correct / max(1, total)
        avg_value_mse = value_mse / max(1, total)
        
        print(f"Epoch {epoch+1}/{config.epochs}:")
        print(f"  Action Loss: {avg_action_loss:.4f}, Accuracy: {action_accuracy:.2f}%")
        print(f"  Value Loss: {avg_value_loss:.4f}, MSE: {avg_value_mse:.4f}")
        print(f"  Aux Loss: {avg_aux_loss:.4f}")
        
        # Save model if it's the best so far
        current_loss = avg_action_loss + avg_value_loss
        if current_loss < best_loss:
            best_loss = current_loss
            print(f"  Saving best model with loss {current_loss:.4f}")
            torch.save(model.state_dict(), f"best_supervised_model_n{config.n_variables}_C{config.max_complexity}.pt")
    
    # Load the best model
    try:
        best_model_path = f"best_supervised_model_n{config.n_variables}_C{config.max_complexity}.pt"
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
    except Exception as e:
        print(f"Error loading best model: {e}")
    
    return model

def train_reinforcement(model, index_to_monomial, monomial_to_index, config, max_vector_size):
    """Train model with reinforcement learning"""
    print("Starting reinforcement learning...")
    
    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate * 0.5,  # Lower learning rate for RL
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.rl_episodes
    )
    
    # Initialize replay buffer with prioritization
    replay_buffer = ReplayBuffer(capacity=50000)  # Larger buffer for better training
    
    # Initialize MCTS with dynamic exploration parameter
    mcts = EnhancedMCTS(model, config, c_puct=config.c_puct)
    
    # Create a dummy dataset for helper functions
    dummy_dataset = CircuitDataset(index_to_monomial, monomial_to_index, max_vector_size, config, size=0)
    
    # Training loop
    best_reward = -float('inf')
    running_rewards = []
    
    # Create progress bar for RL episodes
    pbar = tqdm.tqdm(range(config.rl_episodes), desc="RL Training")
    for episode in pbar:
        try:
            # Generate a random polynomial to simplify
            actions = None
            polynomials = None
            
            # Dynamic complexity based on training progress
            progress_ratio = episode / max(1, config.rl_episodes - 1)
            dynamic_complexity = max(
                config.max_complexity // 2,
                int(config.max_complexity * (0.5 + 0.5 * progress_ratio))
            )
            
            # Try to generate a valid circuit
            for _ in range(5):  # Try a few times if generation fails
                try:
                    from generator import generate_random_circuit
                    actions, polynomials, _, _ = generate_random_circuit(
                        config.n_variables, 
                        config.polynomial_degree,
                        dynamic_complexity,
                        mod=config.mod,
                        trim=config.trim_circuit
                    )
                    if actions and polynomials and len(actions) > config.n_variables + 1:
                        break
                except Exception as e:
                    continue
            
            # Skip this episode if generation failed
            if not actions or not polynomials or len(actions) <= config.n_variables + 1:
                continue
                
            # Target polynomial is the final polynomial
            target_poly = torch.tensor(polynomials[-1], dtype=torch.float, device=device)
            
            # Initialize circuit with variable and constant nodes
            current_actions = actions[:config.n_variables + 1]  # Variables + constant
            current_polys = polynomials[:config.n_variables + 1]
            
            # Calculate available actions mask
            max_nodes = config.n_variables + config.max_complexity + 1
            mask = dummy_dataset.get_available_actions_mask(current_actions, max_nodes)
            
            # Create initial graph
            circuit_graph = dummy_dataset.actions_to_graph(current_actions)
            
            # Self-play until we reach max complexity or solve the polynomial
            episode_states = []
            episode_policies = []
            episode_values = []
            episode_rewards = []
            
            # Track whether we've successfully completed the episode
            episode_completed = False
            
            # Dynamic temperature for MCTS action selection
            temperature = max(0.1, 1.0 - progress_ratio)
            
            # Dynamic number of simulations based on training progress
            dynamic_simulations = max(
                50,
                int(config.num_simulations * (0.5 + 0.5 * progress_ratio))
            )
            
            # Build polynomial step by step
            for step in range(dynamic_complexity):
                # Current state
                state = (circuit_graph, target_poly, current_actions, mask, current_polys)
                
                # Reset MCTS for new search
                mcts.root = None
                
                try:
                    # Run MCTS
                    root = mcts.search(state, num_simulations=dynamic_simulations)
                
                    # Extract policy from visit counts
                    policy = torch.zeros(mask.size(0), device=device)
                    
                    # Check if we have any valid children
                    if not root.children:
                        break
                        
                    # Calculate policy distribution from visit counts
                    visit_sum = sum(child.visit_count for child in root.children.values())
                    if visit_sum > 0:
                        for action, child in root.children.items():
                            policy[action] = child.visit_count / visit_sum
                    
                    # Store state, policy, and intermediate polynomial
                    current_poly = current_polys[-1] if current_polys else torch.zeros_like(target_poly)
                    episode_states.append((
                        circuit_graph, 
                        target_poly, 
                        current_actions, 
                        mask, 
                        current_poly
                    ))
                    episode_policies.append(policy)
                    
                    # Calculate reward for current state
                    similarity = mcts.compute_similarity(current_poly, target_poly.cpu().numpy())
                    episode_rewards.append(similarity)
                    
                    # Select action using temperature parameter
                    action = root.select_action(temperature=temperature)
                    
                    # Apply action
                    try:
                        new_actions, new_polys, new_mask = mcts.apply_action(
                            action, current_actions, current_polys, mask, config.mod
                        )
                        
                        # Update state
                        current_actions = new_actions
                        current_polys = new_polys
                        mask = new_mask
                        circuit_graph = dummy_dataset.actions_to_graph(current_actions)
                        
                        # Check if we've solved the polynomial
                        final_poly = current_polys[-1]
                        similarity = mcts.compute_similarity(final_poly, target_poly.cpu().numpy())
                        
                        if similarity > 0.95 or step == dynamic_complexity - 1:
                            # We've either solved it or reached max complexity
                            episode_completed = True
                            
                            # Final reward
                            final_reward = similarity
                            running_rewards.append(final_reward)
                            if len(running_rewards) > 10:
                                running_rewards.pop(0)
                            
                            # Calculate discounted rewards
                            rewards = episode_rewards + [final_reward]
                            discounted_rewards = []
                            
                            # Use a discount factor to propagate rewards backward
                            gamma = 0.99
                            R = final_reward
                            for r in reversed(rewards[:-1]):
                                R = r + gamma * R
                                discounted_rewards.insert(0, R)
                            
                            # Add experiences to replay buffer
                            for i in range(len(episode_states)):
                                state_graph, state_target, state_actions, state_mask, state_poly = episode_states[i]
                                
                                # Priority is proportional to the reward (higher reward = higher priority)
                                priority = max(0.01, discounted_rewards[i])
                                
                                replay_buffer.add(
                                    (
                                        state_graph,
                                        state_target,
                                        state_actions,
                                        state_mask,
                                        episode_policies[i],
                                        discounted_rewards[i],
                                        state_poly
                                    ),
                                    priority=priority
                                )
                            
                            break
                            
                    except Exception as e:
                        print(f"Error applying action: {e}")
                        break
                        
                except Exception as e:
                    print(f"Error during MCTS search: {e}")
                    break
            
            # Skip training if episode wasn't completed
            if not episode_completed:
                continue
                
            # Train on replay buffer
            if len(replay_buffer) >= config.batch_size:
                try:
                    # Sample batch from replay buffer with prioritization
                    batch = replay_buffer.sample(config.batch_size)
                    if batch:
                        batched_graph, target_polys, circuit_actions, masks, policy_targets, value_targets, current_polys, weights, indices = batch
                        
                        # Set model to training mode
                        model.train()
                        
                        # Forward pass
                        optimizer.zero_grad()
                        policy_logits, value_preds, aux_poly_preds = model(
                            batched_graph, target_polys, circuit_actions, masks
                        )
                        
                        # Calculate losses with importance sampling weights
                        # Policy loss (KL divergence for better numerical stability)
                        policy_logits_masked = policy_logits.clone()
                        policy_logits_masked = policy_logits_masked.masked_fill(~masks, float('-inf'))
                        policy_log_probs = F.log_softmax(policy_logits_masked, dim=1)
                        
                        # Use weighted KL divergence loss
                        policy_loss = -torch.sum(
                            weights.unsqueeze(1) * policy_targets * policy_log_probs
                        ) / weights.sum()
                        
                        # Value loss with importance sampling
                        value_loss = torch.sum(
                            weights * F.mse_loss(value_preds, value_targets, reduction='none')
                        ) / weights.sum()
                        
                        # Auxiliary polynomial prediction loss
                        aux_loss = F.mse_loss(aux_poly_preds, target_polys)
                        
                        # L2 regularization is handled by AdamW
                        
                        # Combined loss with weighting
                        loss = (
                            config.policy_loss_weight * policy_loss + 
                            config.value_loss_weight * value_loss + 
                            0.1 * aux_loss
                        )
                        
                        # Backpropagation
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        # Calculate new priorities based on TD error
                        with torch.no_grad():
                            td_errors = torch.abs(value_preds - value_targets).cpu().numpy()
                            new_priorities = td_errors + 1e-6  # Small constant to avoid zero priority
                            replay_buffer.update_priorities(indices, new_priorities)
                        
                except Exception as e:
                    print(f"Error during training step: {e}")
                    traceback.print_exc()
                    continue
            
            # Step learning rate scheduler
            scheduler.step()
            
            # Save model periodically
            if episode % 100 == 0 and episode > 0:
                torch.save(
                    model.state_dict(), 
                    f"rl_model_n{config.n_variables}_C{config.max_complexity}_episode{episode}.pt"
                )
            
            # Check if this is the best model so far
            avg_reward = sum(running_rewards) / max(1, len(running_rewards))
            if avg_reward > best_reward and len(running_rewards) >= 5:
                best_reward = avg_reward
                torch.save(
                    model.state_dict(), 
                    f"best_rl_model_n{config.n_variables}_C{config.max_complexity}.pt"
                )
                print(f"\nSaved best model with reward {avg_reward:.4f}")
            
            # Update progress bar
            pbar.set_postfix({
                'reward': final_reward if episode_completed else 0,
                'avg_reward': avg_reward,
                'buffer': len(replay_buffer),
                'lr': optimizer.param_groups[0]['lr']
            })
            
        except Exception as e:
            print(f"Error in episode {episode+1}: {e}")
            traceback.print_exc()
            continue
    
    # Load best model at the end
    try:
        best_model_path = f"best_rl_model_n{config.n_variables}_C{config.max_complexity}.pt"
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
    except Exception as e:
        print(f"Error loading best model: {e}")
    
    return model

def evaluate_model(model, dataset, config, index_to_monomial, monomial_to_index, num_tests=100):
    """Evaluate the model on a test set"""
    if len(dataset) == 0:
        print("No data in dataset for evaluation")
        return 0, 0, 0
        
    model.eval()
    test_indices = random.sample(range(len(dataset)), min(num_tests, len(dataset)))
    action_correct = 0
    value_mse = 0
    similarities = []
    
    for idx in test_indices:
        circuit_graph, target_poly, circuit_actions, mask, action, value, _ = dataset[idx]
        
        # Create a "batch" of size 1
        batched_graph = Batch.from_data_list([circuit_graph])
        target_poly = target_poly.unsqueeze(0)
        circuit_actions = [circuit_actions]
        mask = mask.unsqueeze(0)
        value = torch.tensor([value], dtype=torch.float, device=device)
        
        # Get prediction
        with torch.no_grad():
            action_logits, value_pred, _ = model(batched_graph, target_poly, circuit_actions, mask)
            pred_action = torch.argmax(action_logits[0]).item()
        
        # Check accuracy
        if pred_action == action.item():
            action_correct += 1
        
        # Calculate value error
        value_mse += F.mse_loss(value_pred, value, reduction='sum').item()
        
        # Run MCTS to get full solution for similarity calculation
        max_nodes = config.n_variables + config.max_complexity + 1
        
        # Try to build full circuit
        operation, node1_idx, node2_idx = decode_action(pred_action, max_nodes)
        actions = circuit_actions[0]
        polynomials = []
        mcts = EnhancedMCTS(model, config)
        
        try:
            # Reconstruct polynomials
            from generator import create_polynomial_vector
            
            # Create polynomials for variables and constant
            for i in range(config.n_variables + 1):
                if i < config.n_variables:
                    # Variable
                    poly = create_polynomial_vector(
                        index_to_monomial, monomial_to_index, config.n_variables, var_idx=i
                    )
                else:
                    # Constant
                    poly = create_polynomial_vector(
                        index_to_monomial, monomial_to_index, config.n_variables, constant_val=1
                    )
                polynomials.append(poly)
            
            # Add intermediate polynomials from circuit actions
            for i in range(config.n_variables + 1, len(actions)):
                op, in1, in2 = actions[i]
                if op == "add":
                    from generator import add_polynomials_vector
                    poly = add_polynomials_vector(polynomials[in1], polynomials[in2], config.mod)
                else:  # multiply
                    from generator import multiply_polynomials_vector
                    poly = multiply_polynomials_vector(polynomials[in1], polynomials[in2], config.mod)
                polynomials.append(poly)
            
            # Calculate similarity of final polynomial
            if polynomials:
                similarity = mcts.compute_similarity(polynomials[-1], target_poly.squeeze().cpu().numpy())
                similarities.append(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
    
    action_accuracy = 100 * action_correct / max(1, len(test_indices))
    avg_value_mse = value_mse / max(1, len(test_indices))
    avg_similarity = sum(similarities) / max(1, len(similarities)) if similarities else 0
    
    print(f"Test Results:")
    print(f"  Action Accuracy: {action_accuracy:.2f}%")
    print(f"  Value MSE: {avg_value_mse:.4f}")
    print(f"  Average Similarity: {avg_similarity:.4f}")
    
    return action_accuracy, avg_value_mse, avg_similarity

def simplify_polynomial(poly_vector, model, config, index_to_monomial, monomial_to_index):
    """Simplify a given polynomial using the trained model"""
    print("Searching for an efficient arithmetic circuit...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize MCTS with more simulations
    mcts = EnhancedMCTS(model, config, c_puct=config.c_puct * 1.5)  # Higher exploration for testing
    
    # Convert polynomial vector to tensor
    target_poly = torch.tensor(poly_vector, dtype=torch.float, device=device)
    
    # Initialize circuit with variables and constant
    from generator import create_polynomial_vector
    
    # Create initial polynomials for variables
    polynomials = []
    actions = []
    
    # Add variables
    for i in range(config.n_variables):
        actions.append(("input", None, None))
        # Create polynomial for variable x_i
        poly = create_polynomial_vector(index_to_monomial, monomial_to_index, config.n_variables, var_idx=i)
        polynomials.append(poly)
        print(f"Variable {i} (x_{i}) polynomial: {poly[:20]}...")
    
    # Add constant
    actions.append(("constant", None, None))
    poly = create_polynomial_vector(index_to_monomial, monomial_to_index, config.n_variables, constant_val=1)
    polynomials.append(poly)
    print(f"Constant polynomial: {poly[:20]}...")
    
    # Create dataset for helper functions
    dummy_dataset = CircuitDataset(index_to_monomial, monomial_to_index, len(poly_vector), config, size=0)
    
    # Calculate available actions mask
    max_nodes = config.n_variables + config.max_complexity + 1
    mask = dummy_dataset.get_available_actions_mask(actions, max_nodes)
    
    # Create initial graph
    circuit_graph = dummy_dataset.actions_to_graph(actions)
    
    # Track best circuit found
    best_similarity = 0.0
    best_actions = actions.copy()
    best_poly = None
    
    # Use more simulations for polynomial simplification
    num_simulations = 500  # Many more simulations for better results
    
    # Build circuit step by step with MCTS
    print("\nStarting MCTS search...")
    max_search_complexity = min(config.max_complexity, 15)  # Limit to reasonable size
    
    for step in range(max_search_complexity):
        # Current state
        state = (circuit_graph, target_poly, actions, mask, polynomials)
        
        # Reset MCTS for new search
        mcts.root = None
        
        try:
            # Run MCTS with progress indicator
            print(f"Step {step+1}/{max_search_complexity}: Running {num_simulations} simulations...")
            root = mcts.search(state, num_simulations=num_simulations)
            
            # Print top actions being considered by MCTS
            print("Top actions being considered:")
            top_actions = sorted(
                [(action, child.visit_count, child.value()) 
                 for action, child in root.children.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for action_idx, visits, value in top_actions:
                operation, node1, node2 = decode_action(action_idx, max_nodes)
                print(f"  {operation}({node1}, {node2}): visits={visits}, value={value:.4f}")
            
            # Select best action with low temperature (mostly greedy)
            action = root.select_action(temperature=0.01)
            
            # Apply action
            operation, node1_idx, node2_idx = decode_action(action, max_nodes)
            print(f"Selected action: {operation}({node1_idx}, {node2_idx})")
            
            # Create new polynomial for this operation
            if operation == "add":
                from generator import add_polynomials_vector
                new_poly = add_polynomials_vector(polynomials[node1_idx], polynomials[node2_idx], config.mod)
                print(f"New polynomial after addition (sample): {new_poly[:10]}...")
            else:  # multiply
                from generator import multiply_polynomials_vector
                new_poly = multiply_polynomials_vector(polynomials[node1_idx], polynomials[node2_idx], config.mod)
                print(f"New polynomial after multiplication (sample): {new_poly[:10]}...")
            
            # Update actions and polynomials
            actions.append((operation, node1_idx, node2_idx))
            polynomials.append(new_poly)
            
            # Update graph and mask
            circuit_graph = dummy_dataset.actions_to_graph(actions)
            mask = dummy_dataset.get_available_actions_mask(actions, max_nodes)
            
            # Check if current circuit is better
            similarity = mcts.compute_similarity(new_poly, target_poly.cpu().numpy())
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_actions = actions.copy()
                best_poly = new_poly
                
                print(f"  Found better circuit with similarity {similarity:.4f}")
            
            if similarity > 0.95:
                print(f"  Found excellent solution!")
                break
                
        except Exception as e:
            print(f"  Error during search: {e}")
            traceback.print_exc()
            break
    
    # Use the best circuit found
    actions = best_actions
    
    # Print the resulting circuit
    print("\nBest Arithmetic Circuit Found:")
    variable_names = ["x", "y", "z", "w", "v", "u"]  # For readability
    variable_names = []
    for i in range(config.n_variables):
        if i == 0:
            variable_names.append("x")
        elif i == 1:
            variable_names.append("y")
        elif i == 2:
            variable_names.append("z")
        elif i == 3:
            variable_names.append("w")
        else:
            variable_names.append(f"x_{i}")
    
    for i, (op, in1, in2) in enumerate(actions):
        if op == "input" and i < len(variable_names):
            print(f"{i+1}: {variable_names[i]}")
        elif op == "input":
            print(f"{i+1}: x_{i}")
        elif op == "constant":
            print(f"{i+1}: 1")
        else:
            # For operations, show the operation and inputs in a more readable way
            in1_name = variable_names[in1] if in1 < len(variable_names) else f"{in1+1}"
            in2_name = variable_names[in2] if in2 < len(variable_names) else f"{in2+1}"
            
            print(f"{i+1}: {op}({in1+1}, {in2+1})  # {op}({in1_name}, {in2_name})")
    
    # Print human-readable form of the circuit
    if len(actions) > config.n_variables + 1:  # If we have operations beyond variables and constant
        print("\nHuman-readable form:")
        try:
            readable_form = build_readable_circuit(actions, variable_names)
            print(readable_form)
        except Exception as e:
            print(f"Error creating readable form: {e}")
    
    print(f"\nFinal similarity score: {best_similarity:.4f}")
    if best_similarity < 0.5:
        print("WARNING: Low similarity score indicates the circuit may not correctly represent the polynomial.")
    
    return actions, best_poly

def build_readable_circuit(actions, variable_names):
    """Build a human-readable representation of the circuit"""
    expressions = []
    
    for i, (op, in1, in2) in enumerate(actions):
        if op == "input" and i < len(variable_names):
            expressions.append(variable_names[i])
        elif op == "input":
            expressions.append(f"x_{i}")
        elif op == "constant":
            expressions.append("1")
        else:
            # Get expressions for inputs
            in1_expr = expressions[in1]
            in2_expr = expressions[in2]
            
            # Build new expression
            if op == "add":
                # Parenthesize only if needed
                expr = f"{in1_expr} + {in2_expr}"
            else:  # multiply
                # Add parentheses if inputs are additions
                if "+" in in1_expr and not (in1_expr.startswith("(") and in1_expr.endswith(")")):
                    in1_expr = f"({in1_expr})"
                if "+" in in2_expr and not (in2_expr.startswith("(") and in2_expr.endswith(")")):
                    in2_expr = f"({in2_expr})"
                expr = f"{in1_expr} × {in2_expr}"
            
            expressions.append(expr)
    
    # Return the final expression
    return expressions[-1] if expressions else ""

def parse_polynomial(poly_str, index_to_monomial, monomial_to_index, n_variables):
    """Parse a polynomial string into its vector representation"""
    # Calculate max vector size
    max_vector_size = max(monomial_to_index.values()) + 1
    poly_vector = [0] * max_vector_size
    
    print(f"Parsing polynomial: {poly_str}")
    print(f"Number of variables: {n_variables}")
    
    # Print out some key monomials for reference
    print("Available key monomials in the indexing scheme:")
    key_monomials = []
    for idx, monomial in index_to_monomial.items():
        # Look for common monomials like x, y, x^2, y^2, xy, etc.
        is_key = False
        if sum(monomial) <= 3:  # Focus on lower-degree terms
            is_key = True
        
        if is_key:
            key_monomials.append((idx, monomial))
            
    # Sort by degree then index
    key_monomials.sort(key=lambda x: (sum(x[1]), x[0]))
    
    # Print first 20 key monomials
    for idx, monomial in key_monomials[:20]:
        # Format monomial as a readable string
        monomial_str = ""
        for var_idx, power in enumerate(monomial):
            if power > 0:
                var_name = ["x", "y", "z"][var_idx] if var_idx < 3 else f"x_{var_idx}"
                if power == 1:
                    monomial_str += var_name
                else:
                    monomial_str += f"{var_name}^{power}"
        print(f"Monomial {idx}: {monomial} -> {monomial_str}")
    
    # Special case handling for known polynomials
    known_polynomials = {
        "x^2+2xy+y^2": [(2, 0, 0), (1, 1, 0), (0, 2, 0)],  # (x+y)^2
        "x^3": [(3, 0, 0)],  # x^3
        "x*y": [(1, 1, 0)],  # xy
        "x+y": [(1, 0, 0), (0, 1, 0)],  # x+y
        "x^2": [(2, 0, 0)],  # x^2
        "y^2": [(0, 2, 0)],  # y^2
    }
    
    # Check if the polynomial is a known one
    if poly_str in known_polynomials:
        terms = known_polynomials[poly_str]
        coeffs = [1] * len(terms)
        
        # For x^2+2xy+y^2, the middle term has coefficient 2
        if poly_str == "x^2+2xy+y^2":
            coeffs[1] = 2
            
        for i, (term, coeff) in enumerate(zip(terms, coeffs)):
            if term in monomial_to_index:
                idx = monomial_to_index[term]
                poly_vector[idx] = coeff
                print(f"Set coefficient {coeff} for term {term} at index {idx}")
            else:
                print(f"Warning: Term {term} not found in monomial index")
        
        print(f"Created polynomial vector for {poly_str}: {poly_vector[:20]}...")
        return poly_vector
    
    # Parse general polynomial string
    try:
        # Split the string into terms
        poly_str = poly_str.replace(" ", "").replace("-", "+-")
        if poly_str.startswith("+"):
            poly_str = poly_str[1:]
            
        terms = poly_str.split("+")
        
        for term in terms:
            if not term:
                continue
                
            # Handle negative terms
            coeff = 1
            if term.startswith("-"):
                coeff = -1
                term = term[1:]
            
            # Extract coefficient if present
            if term[0].isdigit():
                i = 0
                while i < len(term) and (term[i].isdigit() or term[i] == '.'):
                    i += 1
                try:
                    coeff *= int(term[:i])
                    term = term[i:]
                except ValueError:
                    pass
            
            # Parse the variables and powers
            exponents = [0] * n_variables
            
            # Simple parsing for terms like x^2, y^3, etc.
            i = 0
            while i < len(term):
                if term[i] == 'x' and (i+1 == len(term) or term[i+1] != '_'):
                    var_idx = 0
                elif term[i] == 'y':
                    var_idx = 1
                elif term[i] == 'z':
                    var_idx = 2
                elif term[i] == 'x' and i+1 < len(term) and term[i+1] == '_':
                    # Handle x_n notation
                    i += 2
                    var_idx_str = ""
                    while i < len(term) and term[i].isdigit():
                        var_idx_str += term[i]
                        i += 1
                    var_idx = int(var_idx_str)
                    i -= 1  # Adjust for the loop increment
                else:
                    i += 1
                    continue
                
                # Check for power
                power = 1
                if i+1 < len(term) and term[i+1] == '^':
                    i += 2
                    power_str = ""
                    while i < len(term) and term[i].isdigit():
                        power_str += term[i]
                        i += 1
                    power = int(power_str)
                    i -= 1  # Adjust for the loop increment
                
                if var_idx < n_variables:
                    exponents[var_idx] += power
                
                i += 1
            
            # Find the corresponding monomial index
            monomial_tuple = tuple(exponents)
            if monomial_tuple in monomial_to_index:
                idx = monomial_to_index[monomial_tuple]
                poly_vector[idx] = coeff
                print(f"Set coefficient {coeff} for term {monomial_tuple} at index {idx}")
            else:
                print(f"Warning: Term {monomial_tuple} not found in monomial index")
                
        print(f"Parsed polynomial vector: {poly_vector[:20]}...")
        return poly_vector
                
    except Exception as e:
        print(f"Error parsing polynomial: {e}")
        traceback.print_exc()
    
    # If parsing fails, try to match with known monomials
    print("Searching for matching monomials...")
    for idx, monomial in index_to_monomial.items():
        # Convert monomial to string representation
        monomial_str = ""
        for var_idx, power in enumerate(monomial):
            if power > 0:
                var_name = ["x", "y", "z"][var_idx] if var_idx < 3 else f"x_{var_idx}"
                if power == 1:
                    monomial_str += var_name
                else:
                    monomial_str += f"{var_name}^{power}"
        
        # Check if this monomial appears in the polynomial string
        if monomial_str and monomial_str in poly_str:
            coeff = 1
            # Check for coefficients
            if poly_str.find(monomial_str) > 0:
                prefix = poly_str[:poly_str.find(monomial_str)].strip()
                if prefix.endswith("+"):
                    coeff = 1
                elif prefix.endswith("-"):
                    coeff = -1
                else:
                    try:
                        # Extract numeric coefficient
                        coeff_str = ""
                        for c in reversed(prefix):
                            if c.isdigit() or c == '.':
                                coeff_str = c + coeff_str
                            elif c in '+-':
                                if c == '-':
                                    coeff_str = '-' + coeff_str
                                break
                            else:
                                break
                                
                        if coeff_str:
                            coeff = int(coeff_str)
                    except:
                        coeff = 1
            
            if idx < len(poly_vector):
                poly_vector[idx] = coeff
                print(f"Matched monomial {monomial_str} with coefficient {coeff} at index {idx}")
    
    # If no matches found, create a fallback for known polynomials
    if all(c == 0 for c in poly_vector):
        if "x^2" in poly_str or "x*x" in poly_str:
            for idx, monomial in index_to_monomial.items():
                if monomial == (2, 0, 0) and idx < len(poly_vector):
                    poly_vector[idx] = 1
                    print(f"Using fallback: x^2 at index {idx}")
        
        elif "x^3" in poly_str:
            for idx, monomial in index_to_monomial.items():
                if monomial == (3, 0, 0) and idx < len(poly_vector):
                    poly_vector[idx] = 1
                    print(f"Using fallback: x^3 at index {idx}")
    
    print(f"Final polynomial vector: {poly_vector[:20]}...")
    return poly_vector