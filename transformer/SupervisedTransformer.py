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
from gen import generate_monomials_with_additive_indices, generate_random_polynomials
from torch.utils.data import Dataset, DataLoader
from attention import PositionalEncoding, CircuitHistoryEncoder
from gen import CircuitNode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
class Config:
    def __init__(self):
        self.n_variables = 3           # Number of variables
        self.max_complexity = 5        # Maximum complexity of circuits
        self.hidden_dim = 128          # Hidden dimension for neural networks
        self.embedding_dim = 64        # Embedding dimension for nodes
        self.num_gnn_layers = 12        # Number of GNN layers
        self.num_transformer_layers = 12 # Number of transformer layers
        self.transformer_heads = 16     # Number of attention heads
        self.learning_rate = 0.001     # Learning rate
        self.batch_size = 128           # Batch size for training
        self.mod = 50                  # Modulo for coefficients
        self.train_size = 5000        # Number of training examples
        self.epochs = 200               # Number of training epochs
        self.max_circuit_length = 100  # Maximum length for circuit sequence


config = Config()

# general circuit representation
class ArithmeticCircuit:
    def __init__(self, n_variables, max_vector_size=None):
        # Initialize circuit with variables
        self.nodes = []
        self.edges = []
        self.n_variables = n_variables
        self.max_vector_size = max_vector_size
        
        # Add variable nodes
        for i in range(n_variables):
            self.add_node("input", value=i)
        
        # Add constant node (1)
        self.add_node("constant", value=1)
    
    def add_node(self, node_type, value=None, operation=None, inputs=None):
        node_id = len(self.nodes)
        self.nodes.append({
            "id": node_id,
            "type": node_type,
            "value": value,
            "operation": operation
        })
        
        # Add edges if this is an operation node
        if inputs is not None:
            for input_id in inputs:
                self.edges.append((input_id, node_id))
        
        return node_id
    
    def apply_operation(self, operation, node1_id, node2_id):
        return self.add_node("operation", operation=operation, inputs=[node1_id, node2_id])
    
    def to_graph(self):
        """Convert circuit to PyTorch Geometric graph"""
        # Node features: [type_embedding, value_or_operation]
        node_features = []
        for node in self.nodes:
            # One-hot encoding for node type
            type_embedding = [0, 0, 0]  # [input, constant, operation]
            if node["type"] == "input":
                type_embedding[0] = 1
                value_embedding = node["value"] / max(1, self.n_variables)
            elif node["type"] == "constant":
                type_embedding[1] = 1
                value_embedding = node["value"]
            else:  # operation
                type_embedding[2] = 1
                # One-hot for operation
                value_embedding = 1 if node["operation"] == "multiply" else 0
            
            # Combine embeddings
            node_features.append(type_embedding + [value_embedding])
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float).to(device)
        
        # Ensure we have at least some edges (add self-loops if necessary)
        if not self.edges:
            # Add self-loops for each node
            self.edges = [(i, i) for i in range(len(self.nodes))]
        
        # Convert edges to tensor
        edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous().to(device)
        
        # Always add self-loops to ensure GCN can operate properly
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(self.nodes))
        
        # Create PyTorch Geometric data object
        data = Data(x=x, edge_index=edge_index)
        
        return data
    
    def to_polynomial_vector(self, index_to_monomial, monomial_to_index):
        """Convert circuit to polynomial vector representation directly"""
        # Determine the maximum size of the vector
        if self.max_vector_size is None:
            max_idx = max(monomial_to_index.values())
            vector_size = max_idx + 1
        else:
            vector_size = self.max_vector_size
        
        # Initialize node polynomials as vectors
        node_polynomials = {}
        
        # Process nodes in order
        for i, node in enumerate(self.nodes):
            if node["type"] == "input":
                # Variable node
                var_idx = node["value"]
                vector = [0] * vector_size
                
                # Find the monomial corresponding to this variable
                exponents = [0] * self.n_variables
                exponents[var_idx] = 1
                mono_tuple = tuple(exponents)
                
                if mono_tuple in monomial_to_index:
                    idx = monomial_to_index[mono_tuple]
                    vector[idx] = 1
                
                node_polynomials[i] = vector
            
            elif node["type"] == "constant":
                # Constant node
                vector = [0] * vector_size
                
                # Find the monomial corresponding to the constant term
                zero_tuple = tuple([0] * self.n_variables)
                
                if zero_tuple in monomial_to_index:
                    idx = monomial_to_index[zero_tuple]
                    vector[idx] = node["value"]
                
                node_polynomials[i] = vector
            
            else:  # Operation node
                # Get input node indices
                input_ids = [j for j, k in self.edges if k == i]
                
                if len(input_ids) < 2:
                    # Error case
                    node_polynomials[i] = [0] * vector_size
                    continue
                
                poly1 = node_polynomials[input_ids[0]]
                poly2 = node_polynomials[input_ids[1]]
                
                # Apply operation using vector representation
                if node["operation"] == "add":
                    node_polynomials[i] = add_polynomials_vector(poly1, poly2, config.mod)  
                else:  # multiply
                    node_polynomials[i] = multiply_polynomials_vector(poly1, poly2, config.mod)  
        
        # Return polynomial vector for the last node
        if len(self.nodes) > 0:
            return torch.tensor(node_polynomials[len(self.nodes) - 1], dtype=torch.float).to(device)
        else:
            return torch.tensor([0] * vector_size, dtype=torch.float).to(device)
    
    def circuit_complexity(self):
        """Calculate the complexity of the circuit"""
        # Count operation nodes
        return sum(1 for node in self.nodes if node["type"] == "operation")
    
    def copy(self):
        """Create a deep copy of the circuit"""
        new_circuit = ArithmeticCircuit(self.n_variables, self.max_vector_size)
        new_circuit.nodes = copy.deepcopy(self.nodes)
        new_circuit.edges = copy.deepcopy(self.edges)
        return new_circuit
    
    # def __str__(self):
    #     """String representation of the circuit"""
    #     result = []
    #     for i, node in enumerate(self.nodes):
    #         if node["type"] == "input":
    #             result.append(f"{i}: Input(x_{node['value']})")
    #         elif node["type"] == "constant":
    #             result.append(f"{i}: Constant({node['value']})")
    #         else:
    #             inputs = [j for j, k in self.edges if k == i]
    #             if len(inputs) >= 2:
    #                 result.append(f"{i}: {node['operation']}({inputs[0]}, {inputs[1]})")
    #             else:
    #                 result.append(f"{i}: {node['operation']}(invalid inputs)")
    #     return "\n".join(result)

# Neural Network Architecture
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
    
    def forward(self, batched_graph, target_polynomials, circuit_histories, available_actions_masks=None):
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
            tokens = self.circuit_encoder.encode_circuit_node(circuit_histories[i])
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
        
        # Create memory tensor for transformer (combine graph and polynomial embeddings)
        memory = torch.cat([
            graph_embeddings.unsqueeze(0),  # (1, batch_size, embedding_dim)
            poly_embeddings.unsqueeze(0)    # (1, batch_size, embedding_dim)
        ], dim=0)  # (2, batch_size, embedding_dim)
        
        # Query tensor (output token for each example in batch)
        query = self.output_token.expand(-1, batch_size, -1)  # (1, batch_size, embedding_dim)
        
        # Apply transformer decoder
        output = self.transformer_decoder(
            tgt=query,
            memory=memory,
            memory_key_padding_mask=None,  # Could add masking here if needed
            tgt_key_padding_mask=None
        )
        
        # Get action logits
        output_squeezed = output.squeeze(0)
        action_logits = self.action_head(output_squeezed)  # (batch_size, max_actions)

        # value logits
        value_pred = self.value_head(output_squeezed).squeeze(-1)

        
        # Apply masks to invalid actions
        if available_actions_masks is not None:
            action_logits = action_logits.masked_fill(~available_actions_masks, float('-inf'))
        
        return action_logits, value_pred


# Vector-based polynomial operations
def add_polynomials_vector(poly1, poly2, mod):
    """Add two polynomial vectors"""
    if len(poly1) != len(poly2):
        # Ensure equal length
        max_len = max(len(poly1), len(poly2))
        if len(poly1) < max_len:
            poly1 = poly1 + [0] * (max_len - len(poly1))
        if len(poly2) < max_len:
            poly2 = poly2 + [0] * (max_len - len(poly2))

    result = [0] * len(poly1)
    
    # Add coefficients
    for i in range(len(poly1)):
        result[i] = (poly1[i] + poly2[i]) % mod
    
    return result

def multiply_polynomials_vector(poly1, poly2, mod):
    """Multiply two polynomial vectors using additive indexing scheme"""
    if len(poly1) != len(poly2):
        return "error"
    result = [0] * len(poly1)
    
    # Multiply using the additive indexing property
    for i in range(len(poly1)):
        if poly1[i] == 0:  # Skip zero coefficients
            continue
        for j in range(len(poly2)):
            if poly2[j] == 0:  # Skip zero coefficients
                continue
            # With additive indexing, i+j represents the product monomial's index
            if i+j < len(result):
                result[i+j] = (result[i+j] + (poly1[i] * poly2[j])) % mod
    
    return result

# Dataset class
class CircuitDataset(Dataset):
    def __init__(self, index_to_monomial, monomial_to_index, max_vector_size, size=10000):
        self.index_to_monomial = index_to_monomial
        self.monomial_to_index = monomial_to_index
        self.max_vector_size = max_vector_size
        self.config = config
        self.data = self.generate_data(size)
        
    def generate_data(self, size):
        """Generate training data: (intermediate_circuit, target_poly, circuit_history) -> next_action"""
        print("Generating supervised learning dataset")
        dataset = []
        
        n = self.config.n_variables
        d = self.config.max_complexity
        
        num_circuits = size // self.config.max_complexity
        
        # Generate circuits for this complexity
        _, _, polys, circuits = generate_random_polynomials(n, d, self.config.max_complexity, 
                                                            num_polynomials=num_circuits, 
                                                            mod=self.config.mod)
        
        for polynomial, circuit_node in zip(polys, circuits):
            # Convert the circuit node to a sequence of operations
            operation_sequence = self.convert_circuit_to_operations(circuit_node)
            
            # Convert polynomial to tensor
            target_poly = torch.tensor(polynomial, dtype=torch.float, device=device)
            
            # Create examples for each step
            current_circuit = ArithmeticCircuit(n, self.max_vector_size)
            current_circuit_node = None  # This will build up the circuit history
            
            # Go through all steps except the last one (which we want to predict)
            for i in range(len(operation_sequence) - 1):
                # Apply current step
                op, node1_id, node2_id = operation_sequence[i]
                current_circuit.apply_operation(op, node1_id, node2_id)
                
                # Build circuit history node
                current_circuit_node = self.build_circuit_node_history(
                    operation_sequence[:i+1], n
                )
                
                # Next step (ground truth)
                next_op, next_node1_id, next_node2_id = operation_sequence[i + 1]

                max_nodes = self.config.n_variables + self.config.max_complexity + 1
                action_idx = encode_action(next_op, next_node1_id, next_node2_id, max_nodes)

                # Get available actions mask
                available_mask = self.get_available_actions_mask(current_circuit, max_nodes)
                
                
                # Store example: (current_circuit, target_poly, circuit_history, available_mask, next_action)
                dataset.append({
                    'circuit': current_circuit.copy(),
                    'polynomial': target_poly,
                    'circuit_history': current_circuit_node,
                    'mask': available_mask,
                    'action': action_idx,
                    'value': i/len(operation_sequence)
                })
                
                # If we have enough examples, return
                if len(dataset) >= size:
                    print(f"Generated {len(dataset)} training examples")
                    return dataset
        
        print(f"Generated {len(dataset)} training examples")
        return dataset
    
    def build_circuit_node_history(self, operation_sequence, n_vars):
        """Build a CircuitNode representation from operation sequence"""

        
        # Create basic nodes
        nodes = {}
        
        # Add variables
        for i in range(n_vars):
            nodes[i] = CircuitNode(node_type="input", value=i)
        
        # Add constant
        nodes[n_vars] = CircuitNode(node_type="constant", value=1)
        
        current_idx = n_vars + 1
        
        # Build the circuit from operations
        for op, input1_idx, input2_idx in operation_sequence:
            node1 = nodes[input1_idx]
            node2 = nodes[input2_idx]
            
            operation_node = CircuitNode(
                node_type="operation",
                inputs=[node1, node2],
                operation=op
            )
            
            nodes[current_idx] = operation_node
            current_idx += 1
        
        # Return the last node (root of the circuit)
        return nodes[current_idx - 1] if current_idx > n_vars + 1 else None
                    
    def convert_circuit_to_operations(self, circuit_node):
        """Convert a circuit node to a sequence of operations"""
        operations = []
        node_map = {}  # Maps circuit nodes to sequential indices
        
        # Add initial nodes (variables and constants)
        n_vars = self.config.n_variables
        
        # Variables
        for i in range(n_vars):
            node_map[f"var_{i}"] = i
        
        # Constant
        node_map["const"] = n_vars
        
        # Recursive function to process the circuit
        def process_node(node):
            if node.node_type == "input":
                return node_map[f"var_{node.value}"]
            elif node.node_type == "constant":
                return node_map["const"]
            elif node.node_type == "operation":
                # Process inputs first
                input1_idx = process_node(node.inputs[0])
                input2_idx = process_node(node.inputs[1])
                
                # Create a new operation node
                new_idx = len(node_map)
                node_map[f"op_{new_idx}"] = new_idx
                
                # Add operation to sequence
                operations.append((node.operation, input1_idx, input2_idx))
                
                return new_idx
        
        # Process the root node
        process_node(circuit_node)
        return operations
    
    def get_available_actions_mask(self, circuit, max_nodes):
        """Create a mask for available actions with a fixed size"""
        n_nodes = len(circuit.nodes)
        
        # Calculate the maximum possible number of actions (based on max_nodes)
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
    
    # This is so we can use dataset[idx]
    def __getitem__(self, idx):
        item = self.data[idx]
        # Make sure polynomial tensor is on the correct device
        polynomial = item['polynomial'].to(device)
        mask = item['mask'].to(device)
        action = torch.tensor(item['action'], device=device)
        circuit_history = item['circuit_history']  # This is a CircuitNode
        value = item['value']
        return item['circuit'], polynomial, circuit_history, mask, action, value


def circuit_collate(batch):
    circuits = [item[0] for item in batch]
    polynomials = torch.stack([item[1] for item in batch])
    circuit_histories = [item[2] for item in batch]
    masks = torch.stack([item[3] for item in batch])
    actions = torch.stack([item[4] for item in batch])
    values = torch.tensor([item[5] for item in batch], dtype=torch.float, device=device)
    
    # Create a batched graph using PyTorch Geometric's Batch functionality
    graphs = [circuit.to_graph() for circuit in circuits]
    batched_graph = Batch.from_data_list(graphs)
    
    return batched_graph, polynomials, circuit_histories, masks, actions, values

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


# Training function
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
        for batched_graph, polynomials, circuit_histories, masks, actions, values in tqdm.tqdm(data_loader):
            optimizer.zero_grad()
            
            # Forward pass batch
            action_logits, value_preds = model(batched_graph, polynomials, circuit_histories, masks)
            
            # Calculate action loss
            action_loss = F.cross_entropy(action_logits, actions)
            
            # Calculate value loss
            value_loss = F.mse_loss(value_preds, values)
            
            # Combined loss 
            loss = action_loss + value_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Track action accuracy
            pred_actions = torch.argmax(action_logits, dim=1)
            action_correct += (pred_actions == actions).sum().item()
            
            # Track value prediction error
            value_mse += F.mse_loss(value_preds, values, reduction='sum').item()
            
            total += actions.size(0)
            
            total_action_loss += action_loss.item() * actions.size(0)
            total_value_loss += value_loss.item() * actions.size(0)
        
        avg_action_loss = total_action_loss / len(dataset)
        avg_value_loss = total_value_loss / len(dataset)
        action_accuracy = 100 * action_correct / total
        avg_value_mse = value_mse / total
        
        print(f"Epoch {epoch+1}:")
        print(f"  Action Loss: {avg_action_loss}, Accuracy: {action_accuracy}%")
        print(f"  Value Loss: {avg_value_loss}, MSE: {avg_value_mse}")
    
    return model

# evaluation function
def evaluate_model(model, dataset, config, num_tests=100):
    """Evaluate the model on a test set"""
    test_indices = random.sample(range(len(dataset)), min(num_tests, len(dataset)))
    action_correct = 0
    value_mse = 0
    
    for idx in test_indices:
        circuit, polynomial, circuit_history, mask, action, value = dataset[idx]
        
        # Convert the circuit to a PyTorch Geometric graph
        graph = circuit.to_graph()
        
        # Create a "batch" of size 1 for a single graph
        batched_graph = Batch.from_data_list([graph])
        
        # Make sure tensor dimensions are correct (add batch dimension)
        polynomial = polynomial.unsqueeze(0)
        circuit_histories = [circuit_history]
        mask = mask.unsqueeze(0)
        value = torch.tensor([value], dtype=torch.float, device=device)
        
        # Get prediction
        with torch.no_grad():
            action_logits, value_pred = model(batched_graph, polynomial, circuit_histories, mask)
            pred_action = torch.argmax(action_logits[0]).item()
        
        # Check action prediction
        if pred_action == action.item():
            action_correct += 1
        
        # Calculate value prediction error
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
    
    # Initialize model
    model = CircuitBuilder(config, max_vector_size).to(device)
    model_path = f"transformer_model_n{config.n_variables}_C{config.max_complexity}_mod{config.mod}_GNN{config.num_gnn_layers}_TF{config.num_transformer_layers}.pt"
    
    # Check if there's a saved model
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Evaluate the loaded model first
        print("Evaluating loaded model:")
        evaluate_model(model, dataset, config)
    else:
        print("No existing model found. Starting training from scratch.")
    
    # Continue training the model (whether loaded or new)
    print("Training model...")
    model = train_supervised(model, dataset, config)
    
    # Save the improved model
    print(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)
    
    # Evaluate the final model
    print("Evaluating final model:")
    evaluate_model(model, dataset, config)

if __name__ == "__main__":
    main()
