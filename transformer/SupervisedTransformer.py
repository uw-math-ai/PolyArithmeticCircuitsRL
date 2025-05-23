import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils import add_self_loops
import numpy as np
import random
from collections import deque
import copy
import os
import tqdm
import math
from torch.utils.data import Dataset, DataLoader, Subset
from generator import *
from PositionalEncoding import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Config:
    def __init__(self):
        self.n_variables = 3           # Number of variables
        self.max_complexity = 5        # Maximum complexity of circuits
        self.hidden_dim = 128          # Hidden dimension for neural networks
        self.embedding_dim = 64        # Embedding dimension for nodes
        self.num_gnn_layers = 12       # Number of GNN layers
        self.num_transformer_layers = 12 # Number of transformer layers
        self.transformer_heads = 16     # Number of attention heads
        self.learning_rate = 0.001     # Learning rate
        self.batch_size = 128          # Batch size for training
        self.mod = 50                  # Modulo for coefficients
        self.train_size = 500         # Number of training examples
        self.epochs = 200              # Number of training epochs
        self.max_circuit_length = 100  # Maximum length for circuit sequence
        self.trim_circuit = False

config = Config()


class ArithmeticCircuitGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(ArithmeticCircuitGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # ensure edge_index is not empty
        if edge_index.numel() == 0:
            # Return zero embeddings if no edges
            return torch.zeros(x.size(0), self.conv3.out_channels, device=device)
        
        # GNN layers with error handling
        try:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
        except Exception as e:
            print(f"GNN forward pass error: {e}")
        
        return x

class CircuitBuilder(nn.Module):
    def __init__(self, config, max_poly_vector_size):
        super(CircuitBuilder, self).__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # GNN component
        self.gnn = ArithmeticCircuitGNN(
            input_dim=4,  # [node_type_one_hot (3), value_or_operation]
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim
        )
        
        # Circuit history encoder
        self.circuit_encoder = CircuitHistoryEncoder(config.embedding_dim)
        
        # Polynomial embedding
        self.polynomial_embedding = nn.Linear(
            max_poly_vector_size,  # Size of polynomial vector
            config.embedding_dim
        )
        
        # Transformer components
        self.positional_encoding = PositionalEncoding(config.embedding_dim, config.max_circuit_length)
        
        # Create transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embedding_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.hidden_dim,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_transformer_layers
        )
        
        # Action head
        max_nodes = config.n_variables + config.max_complexity + 1
        total_pairs = (max_nodes * (max_nodes + 1)) // 2  # Combinations with replacement
        max_actions = total_pairs * 2  # Unordered pairs * 2 operations
        
        self.action_head = nn.Linear(config.embedding_dim, max_actions)
        
        # Value head
        self.value_head = nn.Linear(config.embedding_dim, 1)

        # Special tokens
        self.output_token = nn.Parameter(torch.randn(1, 1, config.embedding_dim))
    
    def forward(self, batched_graph, target_polynomials, circuit_actions, available_actions_masks=None):
        batch_size = target_polynomials.size(0)
        
        # Process circuit with GNN
        node_embeddings = self.gnn(batched_graph)
        graph_embeddings = global_mean_pool(node_embeddings, batched_graph.batch)
        
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

        # Value logits
        value_pred = self.value_head(output_squeezed).squeeze(-1)
        
        # Apply masks to invalid actions
        if available_actions_masks is not None:
            action_logits = action_logits.masked_fill(~available_actions_masks, float('-inf'))
        
        return action_logits, value_pred

# Helper function to encode actions with commutative operations
def encode_action(operation, node1_id, node2_id, max_nodes):
    """
    Encode an action so that commutative operations share the same action ID
    regardless of the order of input nodes.
    """
    # For commutative operations, sort the node IDs
    if operation in ["add", "multiply"]:
        node1_id, node2_id = sorted([node1_id, node2_id])
    
    # Compute unique pair index using triangular numbers
    # This maps each unordered pair (i,j) to a unique index
    if node1_id > node2_id:
        node1_id, node2_id = node2_id, node1_id
    
    pair_idx = (node1_id * (2 * max_nodes - node1_id - 1)) // 2 + node2_id - node1_id
    
    # Final action index: pair_index * num_operations + op_index
    return pair_idx * 2 + (1 if operation == "multiply" else 0)


class CircuitDataset(Dataset):
    def __init__(self, index_to_monomial, monomial_to_index, max_vector_size, size=10000):
        self.index_to_monomial = index_to_monomial
        self.monomial_to_index = monomial_to_index
        self.max_vector_size = max_vector_size
        self.config = config
        self.data = self.generate_data(size)

    def train_test_split(self, test_ratio=0.2, seed=42):
        """Split the dataset into training and testing subsets."""
        np.random.seed(seed)
        indices = np.random.permutation(len(self.data))
        test_size = int(len(self.data) * test_ratio)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        train_subset = Subset(self, train_indices)
        test_subset = Subset(self, test_indices)
        return train_subset, test_subset

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
        
        num_circuits = size // self.config.max_complexity
        
        # Generate random circuits
        for _ in range(num_circuits):
            # Generate a random circuit
            actions, polynomials, _, _ = generate_random_circuit(n, d, self.config.max_complexity, mod=self.config.mod, trim = config.trim_circuit)
            
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
                
                # Encode the action
                action_idx = encode_action(next_op, next_node1_id, next_node2_id, max_nodes)
                
                # Store example
                # Compute value score for current circuit state
                current_poly = polynomials[i]
                value_score = self.compute_value_score(current_poly, target_poly)

                dataset.append({
                    'actions': current_actions,
                    'target_poly': target_poly,
                    'mask': available_mask,
                    'action': action_idx,
                    'value': value_score  # ← this is now a heuristic score
                })
                
                # If we have enough examples, return
                if len(dataset) >= size:
                    print(f"Generated {len(dataset)} training examples")
                    return dataset
        
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
        
        # Set available actions to True
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                for op_idx, op in enumerate(["add", "multiply"]):
                    action_idx = encode_action(op, i, j, max_nodes)
                    if action_idx < max_possible_actions:
                        mask[action_idx] = True
        
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

def train_supervised(model, dataset, config):
    # Create data loader with custom collate function
    data_loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=circuit_collate
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    for epoch in range(config.epochs):
        total_action_loss = 0
        total_value_loss = 0
        action_correct = 0
        total = 0
        value_mse = 0
        
        # Process batches
        for batched_graph, target_polys, circuit_actions, masks, actions, values in tqdm.tqdm(data_loader):
            optimizer.zero_grad()
            
            # Forward pass batch
            action_logits, value_preds = model(batched_graph, target_polys, circuit_actions, masks)
            
            # Calculate action loss
            action_loss = F.cross_entropy(action_logits, actions)
            
            # Calculate value loss
            value_loss = F.mse_loss(value_preds, values)
            
            # Combined loss 
            loss = action_loss + value_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Track metrics
            pred_actions = torch.argmax(action_logits, dim=1)
            action_correct += (pred_actions == actions).sum().item()
            value_mse += F.mse_loss(value_preds, values, reduction='sum').item()
            total += actions.size(0)
            
            total_action_loss += action_loss.item() * actions.size(0)
            total_value_loss += value_loss.item() * actions.size(0)
        
        # Print epoch results
        avg_action_loss = total_action_loss / len(dataset)
        avg_value_loss = total_value_loss / len(dataset)
        action_accuracy = 100 * action_correct / total
        avg_value_mse = value_mse / total
        
        print(f"Epoch {epoch+1}:")
        print(f"  Action Loss: {avg_action_loss}, Accuracy: {action_accuracy}%")
        print(f"  Value Loss: {avg_value_loss}, MSE: {avg_value_mse}")
    
    return model

def evaluate_model(model, dataset, config, num_tests=100):
    """Evaluate the model on a test set"""
    test_indices = random.sample(range(len(dataset)), min(num_tests, len(dataset)))
    action_correct = 0
    value_mse = 0
    
    for idx in test_indices:
        circuit_graph, target_poly, circuit_actions, mask, action, value = dataset[idx]
        
        # Create a "batch" of size 1
        batched_graph = Batch.from_data_list([circuit_graph])
        target_poly = target_poly.unsqueeze(0)
        circuit_actions = [circuit_actions]
        mask = mask.unsqueeze(0)
        value = torch.tensor([value], dtype=torch.float, device=device)
        
        # Get prediction
        with torch.no_grad():
            action_logits, value_pred = model(batched_graph, target_poly, circuit_actions, mask)
            pred_action = torch.argmax(action_logits[0]).item()
        
        # Check accuracy
        if pred_action == action.item():
            action_correct += 1
        
        # Calculate value error
        value_mse += F.mse_loss(value_pred, value, reduction='sum').item()
    
    action_accuracy = 100 * action_correct / len(test_indices)
    avg_value_mse = value_mse / len(test_indices)
    
    print(f"Test Results:")
    print(f"  Action Accuracy: {action_accuracy}%")
    print(f"  Value MSE: {avg_value_mse}")
    
    return action_accuracy, avg_value_mse

def main():
    # Initialize configuration
    config = Config()
    
    # Generate monomial indexing
    n = config.n_variables
    d = config.max_complexity
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(n, d)

    # Calculate maximum vector size
    max_vector_size = max(monomial_to_index.values()) + 1
    
    # Create dataset
    dataset = CircuitDataset(index_to_monomial, monomial_to_index, max_vector_size, size=config.train_size)
    train, test = dataset.train_test_split()
    
    # Initialize model
    model = CircuitBuilder(config, max_vector_size).to(device)
    model_path = f"transformer_model_n{config.n_variables}_C{config.max_complexity}_mod{config.mod}_GNN{config.num_gnn_layers}_TF{config.num_transformer_layers}.pt"
    
    # Check if there's a saved model
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Evaluate the loaded model first
        print("Evaluating loaded model:")
        evaluate_model(model, test, config)
    else:
        print("No existing model found. Starting training from scratch.")
    
    # Train the model
    print("Training model...")
    model = train_supervised(model, train, config)
    
    # Save the improved model
    print(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)
    
    # Evaluate the final model
    print("Evaluating final model:")
    evaluate_model(model, test, config)

if __name__ == "__main__":
    main()
