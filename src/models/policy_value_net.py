"""Policy-value network for both PPO and AlphaZero."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .gnn_encoder import CircuitGNN


class PolicyValueNet(nn.Module):
    """Shared network for PPO and AlphaZero.

    Components:
    - CircuitGNN: encodes current circuit graph -> embedding
    - target_encoder: encodes target polynomial coefficient vector
    - fusion: concatenate [graph_emb, target_emb] -> fused embedding
    - policy_head: outputs action logits
    - value_head: outputs state value estimate
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        emb = config.embedding_dim

        # Graph encoder
        self.gnn = CircuitGNN(
            input_dim=config.node_feature_dim,
            hidden_dim=hidden,
            output_dim=emb,
            num_layers=config.num_gnn_layers,
        )

        # Target polynomial encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(config.target_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb),
        )

        # Fusion: concat [graph_emb, target_emb] -> fused
        self.fusion = nn.Sequential(
            nn.Linear(2 * emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.max_actions),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, obs: dict) -> tuple:
        """Forward pass.

        Args:
            obs: dict with 'graph', 'target', 'mask' tensors

        Returns:
            (policy_logits, value) where policy_logits are masked
        """
        graph = obs["graph"]
        target = obs["target"]
        mask = obs["mask"]

        # Encode graph
        if isinstance(graph, dict):
            graph_emb = self.gnn(
                graph["x"], graph["edge_index"],
                num_nodes_actual=graph.get("num_nodes_actual"),
            )
        else:
            # PyG Data object
            graph_emb = self.gnn(
                graph.x, graph.edge_index,
                num_nodes_actual=getattr(graph, "num_nodes_actual", None),
                batch=getattr(graph, "batch", None),
            )

        # Encode target
        if target.dim() == 1:
            target = target.unsqueeze(0)
        target_emb = self.target_encoder(target)

        # Ensure matching batch dimensions
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)

        # Fuse
        fused = self.fusion(torch.cat([graph_emb, target_emb], dim=-1))

        # Policy
        logits = self.policy_head(fused)

        # Mask invalid actions
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        logits = logits.masked_fill(~mask, float("-inf"))

        # Value
        value = self.value_head(fused).squeeze(-1)

        return logits, value

    def get_action_and_value(self, obs: dict, action: torch.Tensor = None):
        """Get action, log probability, entropy, and value from observation.

        Used by PPO for rollout collection and policy update.

        Args:
            obs: observation dict
            action: if provided, compute log_prob for this action instead of sampling

        Returns:
            (action, log_prob, entropy, value)
        """
        logits, value = self.forward(obs)
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        return action, log_prob, entropy, value

    def get_policy_and_value(self, obs: dict):
        """Get policy distribution and value (for MCTS).

        Returns:
            (action_probs, value) where action_probs is a probability distribution
        """
        logits, value = self.forward(obs)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs.squeeze(0), value.squeeze(0)
