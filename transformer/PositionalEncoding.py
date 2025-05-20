import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CircuitHistoryEncoder(nn.Module):
    """Encodes the circuit history as a sequence for the transformer"""
    def __init__(self, embedding_dim):
        super(CircuitHistoryEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Token types: 0=pad, 1=input, 2=constant, 3=add, 4=multiply
        self.token_embedding = nn.Embedding(5, embedding_dim)
        self.value_embedding = nn.Linear(1, embedding_dim)
        self.node_idx_embedding = nn.Embedding(100, embedding_dim)  # Support up to 100 node indices
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def encode_circuit_actions(self, actions):
        """Convert a list of actions to a sequence of tokens"""
        if not actions:
            return []
        
        tokens = []
        
        # Process each action
        for i, action in enumerate(actions):
            action_type, input1_idx, input2_idx = action
            
            if action_type == "input":
                # Token for input variable
                tokens.append({
                    'type': 1,  # input type
                    'value': i,  # Use the node index as its value
                    'node_idx': i
                })
            elif action_type == "constant":
                # Token for constant
                tokens.append({
                    'type': 2,  # constant type
                    'value': 1,  # Constant value is 1
                    'node_idx': i
                })
            else:  # operation
                # Add operation token
                op_type = 3 if action_type == "add" else 4  # 3=add, 4=multiply
                tokens.append({
                    'type': op_type,
                    'value': 0,  
                    'node_idx': i,
                    'input1': input1_idx,
                    'input2': input2_idx
                })
        
        return tokens
    
    def forward(self, circuit_tokens):
        """Convert tokens to embeddings"""
        # circuit_tokens is a list of token dictionaries
        if not circuit_tokens:
            return torch.zeros(1, self.embedding_dim, device=self.device)
        
        # Convert to tensors
        token_types = torch.tensor([t['type'] for t in circuit_tokens], device=self.device)
        token_values = torch.tensor([[t['value']] for t in circuit_tokens], dtype=torch.float, device=self.device)
        node_indices = torch.tensor([t['node_idx'] for t in circuit_tokens], device=self.device)
        
        # Get embeddings
        type_embeddings = self.token_embedding(token_types)
        value_embeddings = self.value_embedding(token_values)
        node_idx_embeddings = self.node_idx_embedding(node_indices)
        
        # Combine embeddings
        combined_embeddings = type_embeddings + value_embeddings + node_idx_embeddings
        
        return combined_embeddings

# PositionalEncoding remains unchanged
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
