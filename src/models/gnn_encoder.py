"""GCN encoder for circuit graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class CircuitGNN(nn.Module):
    """3-layer GCN with residual connections and LayerNorm.

    Takes node features and edge indices, produces a single graph-level embedding
    via global mean pooling over actual (non-padding) nodes.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if HAS_PYG:
            self.convs = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
        else:
            # Fallback: simple message passing via adjacency matrix
            self.linears = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                num_nodes_actual: int = None, batch: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: node features [num_nodes, input_dim]
            edge_index: edge indices [2, num_edges]
            num_nodes_actual: actual number of nodes (rest are padding)
            batch: batch assignment vector for PyG batching

        Returns:
            Graph-level embedding [batch_size, output_dim]
        """
        h = self.input_proj(x)

        for i in range(self.num_layers):
            residual = h
            if HAS_PYG:
                h = self.convs[i](h, edge_index)
            else:
                h = self._simple_message_passing(h, edge_index, i)
            h = self.norms[i](h)
            h = F.relu(h)
            h = h + residual  # Skip connection

        h = self.output_proj(h)

        # Global mean pooling over actual nodes only
        if batch is not None and HAS_PYG:
            return global_mean_pool(h, batch)
        else:
            # Single graph: mean pool over actual nodes
            if num_nodes_actual is not None:
                h = h[:num_nodes_actual]
            return h.mean(dim=0, keepdim=True)

    def _simple_message_passing(self, h: torch.Tensor, edge_index: torch.Tensor,
                                 layer_idx: int) -> torch.Tensor:
        """Simple fallback message passing without PyG."""
        num_nodes = h.size(0)
        # Aggregate neighbor features
        agg = torch.zeros_like(h)
        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            agg.index_add_(0, dst, h[src])
            # Normalize by degree + 1
            degree = torch.zeros(num_nodes, device=h.device)
            degree.index_add_(0, dst, torch.ones(src.size(0), device=h.device))
            degree = degree.clamp(min=1).unsqueeze(1)
            agg = agg / degree

        # Combine self + neighbors
        return self.linears[layer_idx](h + agg)
