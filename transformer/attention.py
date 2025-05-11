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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def encode_circuit_node(self, node):
        """Convert a CircuitNode to a sequence of tokens"""
        if node is None:
            return []
        
        tokens = []
        
        if node.node_type == "input":
            # Token for input variable
            tokens.append({
                'type': 1,  # input type
                'value': node.value
            })
        elif node.node_type == "constant":
            # Token for constant
            tokens.append({
                'type': 2,  # constant type
                'value': node.value
            })
        elif node.node_type == "operation":
            # First encode the inputs recursively
            for input_node in node.inputs:
                tokens.extend(self.encode_circuit_node(input_node))
            
            # Then add the operation token
            op_type = 3 if node.operation == "add" else 4  # 3=add, 4=multiply
            tokens.append({
                'type': op_type,
                'value': 0  # Operations don't have values
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
        
        # Get embeddings
        type_embeddings = self.token_embedding(token_types)
        value_embeddings = self.value_embedding(token_values)
        
        # Combine embeddings
        combined_embeddings = type_embeddings + value_embeddings
        
        return combined_embeddings

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
