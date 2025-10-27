import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnhancedCircuitHistoryEncoder(nn.Module):
    """
    Enhanced circuit history encoder that:
    1. Uses input edge information (input1_idx, input2_idx)
    2. Replaces naive addition with attention-based combination
    3. Works with existing pipeline
    """
    
    def __init__(self, embedding_dim, max_nodes=100):
        super(EnhancedCircuitHistoryEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Core embeddings (same as original)
        self.token_embedding = nn.Embedding(5, embedding_dim)  # 0=pad, 1=input, 2=constant, 3=add, 4=multiply
        self.value_embedding = nn.Linear(1, embedding_dim)
        self.node_idx_embedding = nn.Embedding(max_nodes, embedding_dim)
        
        # NEW: Input relationship embeddings
        self.input1_embedding = nn.Embedding(max_nodes, embedding_dim)
        self.input2_embedding = nn.Embedding(max_nodes, embedding_dim)
        
        # NEW: Attention-based combination instead of naive addition
        self.embedding_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=False
        )
        
        # NEW: Layer normalization for stability
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # NEW: Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def encode_circuit_actions(self, actions):
        """
        Enhanced token encoding that preserves input relationship information
        """
        if not actions:
            return []
        
        tokens = []
        
        for i, action in enumerate(actions):
            action_type, input1_idx, input2_idx = action
            
            if action_type == "input":
                tokens.append({
                    "type": 1,  # input type
                    "value": i,  # Use the node index as its value
                    "node_idx": i,
                    "input1": -1,  # No inputs for base nodes
                    "input2": -1,
                })
            elif action_type == "constant":
                tokens.append({
                    "type": 2,  # constant type
                    "value": 1,  # Constant value is 1
                    "node_idx": i,
                    "input1": -1,  # No inputs for base nodes
                    "input2": -1,
                })
            else:  # operation
                # Add operation token
                op_type = 3 if action_type == "add" else 4  # 3=add, 4=multiply
                tokens.append({
                    "type": op_type,
                    "value": 0,
                    "node_idx": i,
                    "input1": input1_idx if input1_idx is not None else -1,
                    "input2": input2_idx if input2_idx is not None else -1,
                })
        
        return tokens
    
    def forward(self, circuit_tokens):
        """
        Enhanced forward pass with attention-based embedding combination
        """
        if not circuit_tokens:
            return torch.zeros(1, self.embedding_dim, device=self.device)
        
        # Convert tokens to tensors
        token_types = torch.tensor([t["type"] for t in circuit_tokens], device=self.device)
        token_values = torch.tensor(
            [[t["value"]] for t in circuit_tokens],
            dtype=torch.float,
            device=self.device,
        )
        node_indices = torch.tensor([t["node_idx"] for t in circuit_tokens], device=self.device)
        input1_indices = torch.tensor([max(0, t["input1"]) for t in circuit_tokens], device=self.device)
        input2_indices = torch.tensor([max(0, t["input2"]) for t in circuit_tokens], device=self.device)
        
        # Get individual embeddings
        type_embeddings = self.token_embedding(token_types)
        value_embeddings = self.value_embedding(token_values)
        node_idx_embeddings = self.node_idx_embedding(node_indices)
        
        # NEW: Get input relationship embeddings
        input1_embeddings = self.input1_embedding(input1_indices)
        input2_embeddings = self.input2_embedding(input2_indices)
        
        # NEW: Use attention-based combination instead of naive addition
        # Stack all embeddings: (seq_len, num_embeddings, embedding_dim)
        all_embeddings = torch.stack([
            type_embeddings,
            value_embeddings, 
            node_idx_embeddings,
            input1_embeddings,
            input2_embeddings
        ], dim=1)  # (seq_len, 5, embedding_dim)
        
        # Apply attention to combine embeddings
        seq_len, num_embeddings, emb_dim = all_embeddings.shape
        
        # Reshape for attention: (seq_len * num_embeddings, embedding_dim)
        reshaped_embeddings = all_embeddings.view(seq_len * num_embeddings, emb_dim)
        
        # Self-attention to combine embeddings
        attended_embeddings, _ = self.embedding_attention(
            reshaped_embeddings.unsqueeze(0),  # Add batch dimension
            reshaped_embeddings.unsqueeze(0),
            reshaped_embeddings.unsqueeze(0)
        )
        
        # Reshape back and pool across embedding types
        attended_embeddings = attended_embeddings.squeeze(0)  # Remove batch dimension
        attended_embeddings = attended_embeddings.view(seq_len, num_embeddings, emb_dim)
        
        # Pool across the different embedding types (mean pooling)
        combined_embeddings = torch.mean(attended_embeddings, dim=1)  # (seq_len, embedding_dim)
        
        # Apply layer normalization and dropout
        combined_embeddings = self.layer_norm(combined_embeddings)
        combined_embeddings = self.dropout(combined_embeddings)
        
        return combined_embeddings


# Keep the original PositionalEncoding for compatibility
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class SimpleImprovedEmbedding(nn.Module):
    """
    Simpler alternative that just improves the embedding combination
    without changing the overall architecture too much
    """
    
    def __init__(self, embedding_dim, max_nodes=100):
        super(SimpleImprovedEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Original embeddings
        self.token_embedding = nn.Embedding(5, embedding_dim)
        self.value_embedding = nn.Linear(1, embedding_dim)
        self.node_idx_embedding = nn.Embedding(max_nodes, embedding_dim)
        
        # NEW: Input embeddings
        self.input1_embedding = nn.Embedding(max_nodes, embedding_dim)
        self.input2_embedding = nn.Embedding(max_nodes, embedding_dim)
        
        # NEW: Learnable combination weights instead of simple addition
        self.combination_weights = nn.Parameter(torch.randn(5, embedding_dim))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def encode_circuit_actions(self, actions):
        """Same as original but preserves input information"""
        if not actions:
            return []
        
        tokens = []
        for i, action in enumerate(actions):
            action_type, input1_idx, input2_idx = action
            
            if action_type == "input":
                tokens.append({
                    "type": 1,
                    "value": i,
                    "node_idx": i,
                    "input1": -1,
                    "input2": -1,
                })
            elif action_type == "constant":
                tokens.append({
                    "type": 2,
                    "value": 1,
                    "node_idx": i,
                    "input1": -1,
                    "input2": -1,
                })
            else:  # operation
                op_type = 3 if action_type == "add" else 4
                tokens.append({
                    "type": op_type,
                    "value": 0,
                    "node_idx": i,
                    "input1": input1_idx if input1_idx is not None else -1,
                    "input2": input2_idx if input2_idx is not None else -1,
                })
        
        return tokens
    
    def forward(self, circuit_tokens):
        """Improved forward pass with learnable combination weights"""
        if not circuit_tokens:
            return torch.zeros(1, self.embedding_dim, device=next(self.parameters()).device)
        
        # Convert to tensors
        token_types = torch.tensor([t["type"] for t in circuit_tokens], device=next(self.parameters()).device)
        token_values = torch.tensor(
            [[t["value"]] for t in circuit_tokens],
            dtype=torch.float,
            device=next(self.parameters()).device,
        )
        node_indices = torch.tensor([t["node_idx"] for t in circuit_tokens], device=next(self.parameters()).device)
        input1_indices = torch.tensor([max(0, t["input1"]) for t in circuit_tokens], device=next(self.parameters()).device)
        input2_indices = torch.tensor([max(0, t["input2"]) for t in circuit_tokens], device=next(self.parameters()).device)
        
        # Get embeddings
        type_embeddings = self.token_embedding(token_types)
        value_embeddings = self.value_embedding(token_values)
        node_idx_embeddings = self.node_idx_embedding(node_indices)
        input1_embeddings = self.input1_embedding(input1_indices)
        input2_embeddings = self.input2_embedding(input2_indices)
        
        # NEW: Use learnable combination weights instead of simple addition
        embeddings = torch.stack([
            type_embeddings,
            value_embeddings,
            node_idx_embeddings,
            input1_embeddings,
            input2_embeddings
        ], dim=1)  # (seq_len, 5, embedding_dim)
        
        # Apply learnable weights
        weighted_embeddings = embeddings * self.combination_weights.unsqueeze(0)  # Broadcast weights
        combined_embeddings = torch.sum(weighted_embeddings, dim=1)  # Sum across embedding types
        
        # Apply layer normalization
        combined_embeddings = self.layer_norm(combined_embeddings)
        
        return combined_embeddings
