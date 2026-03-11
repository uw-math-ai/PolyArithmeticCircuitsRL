"""Flax policy-value network for JAX-based parallel MCTS.

Architecture mirrors the PyTorch PolicyValueNet:
  GNN encoder (message-passing) + target MLP → fusion → policy head + value head.

Uses simple message-passing (sum-aggregate + linear) rather than PyG GCNConv,
implemented entirely in JAX for jit/vmap compatibility.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


class GraphEncoder(nn.Module):
    """GCN-style message-passing encoder in pure JAX.

    Performs num_layers rounds of:
        h = LayerNorm(ReLU(Linear(h + AggNeighbors(h)))) + residual
    then mean-pools over actual (non-padding) nodes.
    """
    hidden_dim: int
    output_dim: int
    num_layers: int = 4

    @nn.compact
    def __call__(self, x, edge_src, edge_dst, num_nodes, num_edges):
        """Forward pass.

        Args:
            x: (max_nodes, input_dim) node features.
            edge_src: (max_edges,) source node indices (-1 = padding).
            edge_dst: (max_edges,) destination node indices.
            num_nodes: scalar int, number of actual nodes.
            num_edges: scalar int, number of actual edges.

        Returns:
            (output_dim,) graph-level embedding.
        """
        h = nn.Dense(self.hidden_dim, name='input_proj')(x)

        for i in range(self.num_layers):
            residual = h

            # Message passing: aggregate neighbor features to each node.
            agg = _message_pass(h, edge_src, edge_dst, num_edges)

            h = nn.Dense(self.hidden_dim, name=f'conv_{i}')(h + agg)
            h = nn.LayerNorm(name=f'norm_{i}')(h)
            h = nn.relu(h)
            h = h + residual

        h = nn.Dense(self.output_dim, name='output_proj')(h)

        # Mean pool over actual nodes only.
        mask = jnp.arange(h.shape[0]) < num_nodes
        h_masked = h * mask[:, None]
        return h_masked.sum(axis=0) / jnp.maximum(num_nodes.astype(jnp.float32), 1.0)


def _message_pass(h, edge_src, edge_dst, num_edges):
    """Sum-aggregate neighbor features. Pure JAX, no scatter.

    Args:
        h: (max_nodes, hidden_dim) node embeddings.
        edge_src: (max_edges,) source indices.
        edge_dst: (max_edges,) destination indices.
        num_edges: scalar int.

    Returns:
        (max_nodes, hidden_dim) aggregated messages.
    """
    max_nodes, hidden_dim = h.shape

    # Gather source features for each edge.
    # Clamp indices to 0 for invalid edges (will be masked out).
    safe_src = jnp.clip(edge_src, 0, max_nodes - 1)
    safe_dst = jnp.clip(edge_dst, 0, max_nodes - 1)
    src_features = h[safe_src]  # (max_edges, hidden_dim)

    # Mask out padding edges.
    edge_mask = (jnp.arange(edge_src.shape[0]) < num_edges).astype(jnp.float32)
    src_features = src_features * edge_mask[:, None]

    # Scatter-add to destination nodes.
    agg = jnp.zeros((max_nodes, hidden_dim), dtype=h.dtype)
    agg = agg.at[safe_dst].add(src_features)

    # Degree normalization.
    degree = jnp.zeros(max_nodes, dtype=jnp.float32)
    degree = degree.at[safe_dst].add(edge_mask)
    degree = jnp.maximum(degree, 1.0)

    return agg / degree[:, None]


class PolicyValueNet(nn.Module):
    """Policy-value network matching the PyTorch architecture.

    Architecture:
        graph -> GraphEncoder -> graph_emb
        target -> MLP -> target_emb
        [graph_emb, target_emb] -> fusion MLP -> fused
        fused -> policy_head -> logits (masked)
        fused -> value_head -> scalar value
    """
    hidden_dim: int = 256
    embedding_dim: int = 256
    num_gnn_layers: int = 4
    target_size: int = 49      # (max_degree+1)^n_variables
    max_actions: int = 90      # max_nodes * (max_nodes + 1)

    @nn.compact
    def __call__(self, obs):
        """Forward pass.

        Args:
            obs: Dict with 'node_features', 'edge_src', 'edge_dst',
                 'num_nodes', 'num_edges', 'target', 'mask'.

        Returns:
            (logits, value) where logits has shape (max_actions,) with
            invalid actions set to -inf, and value is a scalar.
        """
        # Graph encoding.
        graph_emb = GraphEncoder(
            hidden_dim=self.hidden_dim,
            output_dim=self.embedding_dim,
            num_layers=self.num_gnn_layers,
            name='gnn',
        )(
            obs['node_features'],
            obs['edge_src'],
            obs['edge_dst'],
            obs['num_nodes'],
            obs['num_edges'],
        )

        # Target encoding.
        target = obs['target']
        target_emb = nn.Dense(self.hidden_dim, name='target_enc_0')(target)
        target_emb = nn.relu(target_emb)
        target_emb = nn.Dense(self.embedding_dim, name='target_enc_1')(target_emb)

        # Fusion.
        fused = jnp.concatenate([graph_emb, target_emb], axis=-1)
        fused = nn.Dense(self.hidden_dim, name='fusion_0')(fused)
        fused = nn.relu(fused)
        fused = nn.Dense(self.embedding_dim, name='fusion_1')(fused)
        fused = nn.relu(fused)

        # Policy head.
        logits = nn.Dense(self.hidden_dim, name='policy_0')(fused)
        logits = nn.relu(logits)
        logits = nn.Dense(self.max_actions, name='policy_1')(logits)

        # Mask invalid actions.
        mask = obs['mask']
        logits = jnp.where(mask, logits, -1e9)

        # Value head.
        value = nn.Dense(self.hidden_dim, name='value_0')(fused)
        value = nn.relu(value)
        value = nn.Dense(1, name='value_1')(value)
        value = value.squeeze(-1)

        return logits, value


def create_network(config) -> PolicyValueNet:
    """Create a PolicyValueNet with dimensions from Config."""
    return PolicyValueNet(
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_gnn_layers=config.num_gnn_layers,
        target_size=config.target_size,
        max_actions=config.max_actions,
    )


def init_params(network: PolicyValueNet, env_config, rng_key):
    """Initialize network parameters with a dummy observation."""
    max_nodes = env_config.max_nodes
    max_edges = max_nodes * 4
    target_size = env_config.target_size
    max_actions = env_config.max_actions

    dummy_obs = {
        'node_features': jnp.zeros((max_nodes, 4), dtype=jnp.float32),
        'edge_src': jnp.zeros((max_edges,), dtype=jnp.int32),
        'edge_dst': jnp.zeros((max_edges,), dtype=jnp.int32),
        'num_nodes': jnp.int32(3),
        'num_edges': jnp.int32(0),
        'target': jnp.zeros((target_size,), dtype=jnp.float32),
        'mask': jnp.ones((max_actions,), dtype=jnp.bool_),
    }
    return network.init(rng_key, dummy_obs)
