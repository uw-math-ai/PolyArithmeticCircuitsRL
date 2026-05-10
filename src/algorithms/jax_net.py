"""Flax policy-value network for JAX-based parallel MCTS.

Architecture:
  node features + node coeffs + node ids -> GNN encoder
  target -> MLP
  per-action (op, node_i, node_j) features -> pairwise policy head
  graph + target embedding -> value head

Uses simple message-passing (sum-aggregate + linear) rather than PyG GCNConv,
implemented entirely in JAX for jit/vmap compatibility.
"""

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
            (node_embeddings, graph_embedding), where node_embeddings has shape
            (max_nodes, output_dim) and graph_embedding has shape (output_dim,).
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

        mask = jnp.arange(h.shape[0]) < num_nodes
        h = nn.Dense(self.output_dim, name='output_proj')(h)
        h = h * mask[:, None]

        # Mean pool over actual nodes only.
        h_masked = h * mask[:, None]
        graph_emb = h_masked.sum(axis=0) / jnp.maximum(
            num_nodes.astype(jnp.float32), 1.0
        )
        return h, graph_emb


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
    """Policy-value network with semantic node encoding and pairwise actions.

    Architecture:
        node_features + node_coeffs + node ids -> GraphEncoder
        target -> MLP -> target_emb
        [h_i, h_j, op, target_emb, graph_emb] -> pairwise policy logits
        [graph_emb, target_emb] -> fusion MLP -> value
        fused -> value_head -> scalar value
    """
    hidden_dim: int = 256
    embedding_dim: int = 256
    num_gnn_layers: int = 4
    target_size: int = 49      # (max_degree+1)^n_variables
    max_actions: int = 90      # max_nodes * (max_nodes + 1)
    poly_embedding_dim: int = 128
    node_id_embedding_dim: int = 32
    op_embedding_dim: int = 16
    pair_hidden_dim: int = 384

    @nn.compact
    def __call__(self, obs):
        """Forward pass.

        Args:
            obs: Dict with 'node_features', 'node_coeffs', 'edge_src', 'edge_dst',
                 'num_nodes', 'num_edges', 'target', 'mask'.

        Returns:
            (logits, value) where logits has shape (max_actions,) with
            invalid actions set to -inf, and value is a scalar.
        """
        max_nodes = obs['node_features'].shape[0]
        node_mask = (jnp.arange(max_nodes) < obs['num_nodes'])[:, None]

        # Per-node polynomial identity encoding.
        poly_emb = nn.Dense(self.hidden_dim, name='poly_enc_0')(obs['node_coeffs'])
        poly_emb = nn.relu(poly_emb)
        poly_emb = nn.Dense(self.poly_embedding_dim, name='poly_enc_1')(poly_emb)
        poly_emb = nn.relu(poly_emb)

        node_ids = jnp.arange(max_nodes, dtype=jnp.int32)
        node_id_emb = nn.Embed(
            num_embeddings=max_nodes,
            features=self.node_id_embedding_dim,
            name='node_id_embed',
        )(node_ids)

        node_inputs = jnp.concatenate(
            [obs['node_features'], poly_emb, node_id_emb],
            axis=-1,
        )
        node_inputs = node_inputs * node_mask.astype(node_inputs.dtype)

        # Graph encoding.
        node_emb, graph_emb = GraphEncoder(
            hidden_dim=self.hidden_dim,
            output_dim=self.embedding_dim,
            num_layers=self.num_gnn_layers,
            name='gnn',
        )(
            node_inputs,
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

        # Pairwise policy head.
        action_indices = jnp.arange(self.max_actions, dtype=jnp.int32)
        op_arr, i_arr, j_arr = _decode_action_batch(action_indices, max_nodes)
        h_i = node_emb[i_arr]
        h_j = node_emb[j_arr]
        op_emb = nn.Embed(
            num_embeddings=2,
            features=self.op_embedding_dim,
            name='op_embed',
        )(op_arr)
        target_rep = jnp.broadcast_to(
            target_emb[None, :], (self.max_actions, self.embedding_dim)
        )
        graph_rep = jnp.broadcast_to(
            graph_emb[None, :], (self.max_actions, self.embedding_dim)
        )
        pair_feat = jnp.concatenate(
            [
                h_i,
                h_j,
                h_i * h_j,
                jnp.abs(h_i - h_j),
                op_emb,
                target_rep,
                graph_rep,
            ],
            axis=-1,
        )
        logits = nn.Dense(self.pair_hidden_dim, name='pair_policy_0')(pair_feat)
        logits = nn.relu(logits)
        logits = nn.Dense(self.pair_hidden_dim, name='pair_policy_1')(logits)
        logits = nn.relu(logits)
        logits = nn.Dense(1, name='pair_policy_out')(logits).squeeze(-1)

        # Mask invalid actions.
        mask = obs['mask']
        logits = jnp.where(mask, logits, -1e9)

        # Value fusion.
        fused = jnp.concatenate([graph_emb, target_emb], axis=-1)
        fused = nn.Dense(self.hidden_dim, name='fusion_0')(fused)
        fused = nn.relu(fused)
        fused = nn.Dense(self.embedding_dim, name='fusion_1')(fused)
        fused = nn.relu(fused)

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
        poly_embedding_dim=getattr(config, 'poly_embedding_dim', 128),
        node_id_embedding_dim=getattr(config, 'node_id_embedding_dim', 32),
        op_embedding_dim=getattr(config, 'op_embedding_dim', 16),
        pair_hidden_dim=getattr(config, 'pair_hidden_dim', config.hidden_dim),
    )


def init_params(network: PolicyValueNet, env_config, rng_key):
    """Initialize network parameters with a dummy observation."""
    max_nodes = env_config.max_nodes
    max_edges = max_nodes * 4
    target_size = env_config.target_size
    max_actions = env_config.max_actions

    dummy_obs = {
        'node_features': jnp.zeros((max_nodes, 4), dtype=jnp.float32),
        'node_coeffs': jnp.zeros((max_nodes, target_size), dtype=jnp.float32),
        'edge_src': jnp.zeros((max_edges,), dtype=jnp.int32),
        'edge_dst': jnp.zeros((max_edges,), dtype=jnp.int32),
        'num_nodes': jnp.int32(3),
        'num_edges': jnp.int32(0),
        'target': jnp.zeros((target_size,), dtype=jnp.float32),
        'mask': jnp.ones((max_actions,), dtype=jnp.bool_),
    }
    return network.init(rng_key, dummy_obs)


def _decode_action_batch(action_indices: jnp.ndarray, max_nodes: int):
    """Decode flat action IDs into vectorized (op, i, j) arrays."""
    op = action_indices % 2
    pair_idx = action_indices // 2

    discriminant = (2 * max_nodes + 1) ** 2 - 8 * pair_idx
    sqrt_disc = jnp.floor(jnp.sqrt(discriminant.astype(jnp.float32))).astype(jnp.int32)
    i = (2 * max_nodes + 1 - sqrt_disc) // 2

    i = jnp.clip(i, 0, max_nodes - 1)
    row_start = i * max_nodes - i * (i - 1) // 2
    i = jnp.where(row_start > pair_idx, i - 1, i)
    row_start = i * max_nodes - i * (i - 1) // 2
    j = i + (pair_idx - row_start)
    return op, i, j
