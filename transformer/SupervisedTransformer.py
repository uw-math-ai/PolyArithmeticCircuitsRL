import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils import add_self_loops
import random
import os
import tqdm
import math
from torch.utils.data import Dataset, DataLoader
from generator import *
from PositionalEncoding import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import encode_action, vector_to_sympy
from State import *
from State import Game
from torch.distributions import Categorical
from utils import vector_to_sympy, encode_action


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Config:
    def __init__(self):
        self.n_variables = 3         
        self.max_complexity = 5      
        self.hidden_dim = 512         
        self.embedding_dim = 512       
        self.num_gnn_layers = 6        
        self.num_transformer_layers = 48 
        self.transformer_heads = 8    
        self.learning_rate = 0.0003    # Reduced learning rate for stability
        self.batch_size = 64           
        self.mod = 50                  
        self.train_size = 10000
        self.epochs = 300             
        self.max_circuit_length = 100  # Maximum length for circuit sequence
        self.warmup_steps = 1000       # learning rate warmup
        self.weight_decay = 0.01       # L2 regularization

        self.rl_learning_rate = 1e-3            # learning rate
        self.rl_reward_decay = 0.99             # reward decay 
        self.rl_episodes = 10000                  # number of games to play
        self.rl_eps = np.finfo(np.float32).eps  # epsilon for numerical stability
config = Config()


class ArithmeticCircuitGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(ArithmeticCircuitGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for i in range(config.num_gnn_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        self.convs.append(GCNConv(hidden_dim, embedding_dim))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(config.num_gnn_layers - 1)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if edge_index.numel() == 0:
            return torch.zeros(x.size(0), self.convs[-1].out_channels, device=device)
        
        # First layer
        x = F.relu(self.convs[0](x, edge_index))
        
        # Middle layers with residual connections
        for i in range(1, len(self.convs) - 1):
            identity = x
            x = self.layer_norms[i-1](x)
            x = F.relu(self.convs[i](x, edge_index))
            x = x + identity  # Residual connection
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        x = self.final_norm(x)
        
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
        poly_embeddings = poly_embeddings + torch.randn_like(poly_embeddings) * 1e-6 # adding some random noise so the attention weights aren't uniform
        graph_embeddings = graph_embeddings + torch.randn_like(graph_embeddings) * 1e-6

        
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
            # graph_embeddings.unsqueeze(0),  # (1, batch_size, embedding_dim)
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
        d = self.config.max_complexity*2
        
        num_circuits = size // self.config.max_complexity
        
        # Generate random circuits
        for _ in range(num_circuits):
            # Generate a random circuit
            actions, polynomials, _, _ = generate_random_circuit(n, d, self.config.max_complexity, mod=self.config.mod)
            
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
                dataset.append({
                    'actions': current_actions,
                    'target_poly': target_poly,
                    'mask': available_mask,
                    'action': action_idx,
                    'value': (i - n - 1) / (len(actions) - n - 1)  # Progress through operations
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
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=len(data_loader) * config.epochs,
        eta_min=config.learning_rate/100
    )
    
    # Training loop
    for epoch in range(config.epochs):

        total_action_loss = 0
        total_value_loss = 0
        action_correct = 0
        action_incorrect = 0
        total = 0
        value_mse = 0
        
        # Process batches
        for batched_graph, target_polys, circuit_actions, masks, actions, values in tqdm.tqdm(data_loader):
            optimizer.zero_grad()
            
            # Forward pass batch
            action_logits, value_preds = model(batched_graph, target_polys, circuit_actions, masks)
            
            # Calculate action loss
            action_loss = F.cross_entropy(action_logits, actions) # test with label_smoothing=0.1
            
            # Calculate value loss
            value_loss = F.mse_loss(value_preds, values)
            
            # Combined loss 
            loss = action_loss + value_loss
            loss= action_loss # first let's see if we can get good action accuracy
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            
            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            
            # Track metrics
            pred_actions = torch.argmax(action_logits, dim=1)
            action_correct += (pred_actions == actions).sum().item()
            action_incorrect += (pred_actions != actions).sum().item()

            value_mse += F.mse_loss(value_preds, values, reduction='sum').item()
            total += actions.size(0)
            
            total_action_loss += action_loss.item() * actions.size(0)
            total_value_loss += value_loss.item() * actions.size(0)
                # Update learning rate
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current learning rate: {current_lr:.6f}")
        # Print epoch results
        avg_action_loss = total_action_loss / len(dataset)
        avg_value_loss = total_value_loss / len(dataset)
        action_accuracy = 100 * action_correct / (action_correct+action_incorrect)
        avg_value_mse = value_mse / total
        
        print(f"Epoch {epoch+1}:")
        print(f"  Action Loss: {avg_action_loss}, Accuracy: {action_accuracy}%")
        print(f"  Value Loss: {avg_value_loss}, MSE: {avg_value_mse}")
        model_path = f"transformer_model_n{config.n_variables}_C{config.max_complexity}_mod{config.mod}_GNN{config.num_gnn_layers}_TF{config.num_transformer_layers}.pt"

        if epoch%20==0:
            torch.save(model.state_dict(), model_path)
    
    return model


def train_reinforce(model, dataset, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.rl_learning_rate)
    index_to_monomial = dataset.index_to_monomial
    correct_count=0
    incorrect_count=0
    for i in range(config.rl_episodes):
        circuit_graph_dataset, target_poly, circuit_actions_dataset, mask_dataset, action_dataset, value = dataset[random.randint(1, len(dataset)-2)]
        sp_target_poly = vector_to_sympy(target_poly, index_to_monomial)
        game = Game(sp_target_poly, target_poly.unsqueeze(0), config).to(device)
        log_probs = []
        while not game.is_done():
            circuit_graph, target_poly, circuit_actions, mask = game.observe()
            # print(circuit_actions_dataset)
            # print(circuit_actions)
            # for i in range(50):
            #     _,tar,cir,_,_,_ = dataset[i]
            #     print(cir)
            #     print(vector_to_sympy(tar, index_to_monomial))
            # print(mask)
            action_logits, _ = model(circuit_graph, target_poly, circuit_actions, mask)
            
            # Find valid actions (where mask is True)
            valid_indices = torch.where(mask[0])[0]
            
            # Extract only the logits for valid actions
            valid_logits = action_logits[0, valid_indices]
            
            # Create categorical distribution only on valid actions
            dist = Categorical(logits=valid_logits)
            
            # Sample from the valid actions only
            local_action = dist.sample()
            log_prob = dist.log_prob(local_action)
            
            # Convert back to global action index
            action = valid_indices[local_action]

            
            while not game.is_valid_action(action):
                print("error with mask")
                # action, log_prob = dist.sample()
                # log_prob = dist.log_prob(action)
                # action = valid_indices[action]

            game.take_action(action)
            log_probs.append(log_prob)

        rewards = game.compute_rewards()
        if rewards[-1]==100:
            correct_count = correct_count+1
        else:
            incorrect_count = incorrect_count+1

        # Process rewards and compute loss
        R, loss, returns = 0, 0, []

        for r in rewards[::-1]:
            R = r + config.rl_reward_decay * R
            returns.append(R)

        returns = torch.tensor(returns[::-1], device=device)
        
        # if len(returns) > 1:  
        #     returns = (returns - returns.mean()) / (returns.std() + config.rl_eps)

        for lp, R in zip(log_probs, returns):
            loss += -lp * R

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress every 100 episodes
        if (i + 1) % 10 == 0:
            print(f"correct count: {correct_count}")
            print(f"Episode {i+1}/{config.rl_episodes}, Last reward: {rewards[-1]:.4f}")

def create_example_data(max_vector_size, model):
    # Graph with 4 nodes (3 inputs + 1 constant), 4 features each
    x = torch.tensor([
        [1, 0, 0, 0],    # Input 0
        [1, 0, 0, 1/3],  # Input 1
        [1, 0, 0, 2/3],  # Input 2
        [0, 1, 0, 1.0]   # Constant
    ], dtype=torch.float, device=device)
    
    # Create self-loops for each node (initial circuit has no computation edges)
    n_nodes = x.size(0)
    edge_index = torch.tensor([
        [i for i in range(n_nodes)],  # Source nodes
        [i for i in range(n_nodes)]   # Target nodes (self-loops)
    ], dtype=torch.long, device=device)
    
    graph_data = Data(x=x, edge_index=edge_index)
    
    # Target polynomial (using the example from original code)
    target_poly = torch.zeros(max_vector_size, dtype=torch.float, device=device)
    target_poly[36] = 2.0
    
    # Initial circuit actions - just the base nodes
    actions = [
        ('input', None, None),    # x0
        ('input', None, None),    # x1
        ('input', None, None),    # x2
        ('constant', None, None)  # Constant value
    ]
    
    # Get available actions mask
    max_nodes = config.n_variables + config.max_complexity + 1
    total_max_pairs = (max_nodes * (max_nodes + 1)) // 2
    max_possible_actions = total_max_pairs * 2
    
    mask = torch.zeros(max_possible_actions, dtype=torch.bool, device=device)
    
    # Set available actions to True for all pairs of initial nodes
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            for op_idx, op in enumerate(["add", "multiply"]):
                action_idx = encode_action(op, i, j, max_nodes)
                if action_idx < max_possible_actions:
                    mask[action_idx] = True
    
    # Prepare data for model
    batched_graph = Batch.from_data_list([graph_data])
    target_poly = target_poly.unsqueeze(0)
    mask = mask.unsqueeze(0)
    circuit_actions = [actions]
    
    # Forward pass through the model
    print("Predicting first operation in circuit construction...")
    action_logits, value_pred = model(batched_graph, target_poly, circuit_actions, mask)
    print(action_logits)
    # Get the top predicted actions
    top_action_values, top_action_indices = torch.topk(action_logits[0], 5)
    
    print("\nTop 5 predicted actions:")
    for i, (idx, val) in enumerate(zip(top_action_indices, top_action_values)):
        idx = idx.item()
        val = val.item()
        
        # Decode the action
        op_idx = idx % 2
        pair_idx = idx // 2
        operation = "multiply" if op_idx == 1 else "add"
        
        # Find node indices from pair index
        node1 = int(math.floor((-1 + math.sqrt(1 + 8 * pair_idx)) / 2))
        node2 = pair_idx - (node1 * (node1 + 1)) // 2
        
        print(f"  {i+1}. {operation}({node1}, {node2}) - score: {val:.4f}")
    
    return action_logits, value_pred
    
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
    d = config.max_complexity*2
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(n, d)
    
    # Calculate maximum vector size, +1 because max index is 0-based
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
        create_example_data(max_vector_size, model)
        train_reinforce(model, dataset, config)
        evaluate_model(model, dataset, config)
    else:
        print("No existing model found. Starting training from scratch.")
    
    # Train the model
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
