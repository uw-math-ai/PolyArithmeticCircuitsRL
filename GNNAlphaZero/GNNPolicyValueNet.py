import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNPolicyValueNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3):
        """
        Args:
            in_dim (int): Input feature dimension (size of polynomial vectors).
            hidden_dim (int): Hidden layer dimension.
            num_layers (int): Number of GNN layers.
        """
        super(GNNPolicyValueNet, self).__init__()

        # Node embedding
        self.node_encoder = nn.Linear(in_dim, hidden_dim)

        # Edge embedding (operation type: add/multiply → embedding)
        self.edge_encoder = nn.Embedding(2, hidden_dim)

        # GNN layers (simple GCN for now)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Pooling: mean of all nodes
        self.pool = global_mean_pool

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # output logit for each possible action
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # output scalar value
            nn.Tanh()  # value between -1 and 1
        )

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Data): Batch of graphs.
        Returns:
            policy_logits (torch.Tensor): (num_nodes,) predicted logits for choosing actions.
            value (torch.Tensor): (batch_size, 1) estimated value of the position.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr.squeeze(), data.batch

        # Encode nodes and edges
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Message passing
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # Pool over graph to get a fixed-size vector
        pooled = self.pool(x, batch)  # (batch_size, hidden_dim)

        # Heads
        policy_logits = self.policy_head(x)  # node-wise policy logits
        value = self.value_head(pooled)  # graph-level value

        return policy_logits, value