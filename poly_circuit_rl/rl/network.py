"""CircuitTransformerQ: Transformer-based Q-network with parent pointer embeddings
and bilinear action scoring.

Architecture:
  1. Parse flat obs -> continuous node feats + index fields
  2. Embed parent/position indices via nn.Embedding
  3. Concat with target encoding + steps_left, project to d_model
  4. Transformer encoder with causal mask (construction order)
  5. Bilinear action scoring: Q_add[i,j], Q_mul[i,j], Q_out[i], Q_stop
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from ..env.obs import _TYPE_OFFSET, _OP_OFFSET, _PARENT_OFFSET, _POS_OFFSET, _leaf_offset, _eval_offset


class TargetEncoder(nn.Module):
    """Swappable target encoder. Current: Linear(m, D).

    Future: replace with CoefficientEncoder for 20+ variable polynomials.
    """

    def __init__(self, m: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(m, d_model)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """(B, m) -> (B, d_model)"""
        return self.proj(target)


class CircuitTransformerQ(nn.Module):
    """Transformer Q-network for polynomial circuit construction.

    Takes a flat observation (B, obs_dim), internally parses it into
    node features + index fields, embeds indices, runs a causal
    transformer, and produces Q-values via bilinear action scoring.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        L = config.L
        m = config.m
        d_model = config.d_model
        d_pos = config.d_pos

        # --- Embeddings for discrete index fields ---
        # parent indices: 0..L-1 for real nodes, L for "no parent" sentinel
        self.parent_emb = nn.Embedding(L + 1, d_pos, padding_idx=L)
        # position indices: 0..L-1
        self.pos_emb = nn.Embedding(L, d_pos)

        # --- Target encoder (swappable) ---
        self.target_enc = TargetEncoder(m, d_model)

        # --- Input projection ---
        # continuous node feats: type_oh(3) + op_oh(2) + leaf_id(n_leaf) + eval(m)
        d_continuous = config.d_node_continuous
        # after embedding: + parent1(d_pos) + parent2(d_pos) + pos(d_pos) + target(d_model) + steps(1)
        d_input = d_continuous + 3 * d_pos + d_model + 1
        self.input_proj = nn.Linear(d_input, d_model)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers,
        )

        # --- Action scoring heads ---
        # Bilinear for ADD/MUL pair scoring
        self.W_add = nn.Bilinear(d_model, d_model, 1, bias=False)
        self.W_mul = nn.Bilinear(d_model, d_model, 1, bias=False)
        # Linear for SET_OUTPUT per-node scoring
        self.w_out = nn.Linear(d_model, 1)
        # Linear for STOP on mean-pooled embedding
        self.w_stop = nn.Linear(d_model, 1)

        # Pre-compute pair indices for action scoring (i <= j)
        idx_i, idx_j = [], []
        for j in range(L):
            for i in range(j + 1):
                idx_i.append(i)
                idx_j.append(j)
        self.register_buffer("pair_idx_i", torch.tensor(idx_i, dtype=torch.long))
        self.register_buffer("pair_idx_j", torch.tensor(idx_j, dtype=torch.long))

        # Causal mask: node i can only attend to nodes j <= i
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim) flat observation vector
        Returns:
            q_values: (B, action_dim)
        """
        B = obs.shape[0]
        config = self.config
        L = config.L
        d = config.d_node_raw

        # --- 1. Parse flat obs into structured fields ---
        node_section = obs[:, :L * d].reshape(B, L, d)

        # Continuous features: type_oh + op_oh + leaf_id + eval_vec
        lo = _leaf_offset(config)
        eo = _eval_offset(config)
        type_oh = node_section[:, :, _TYPE_OFFSET:_TYPE_OFFSET + 3]           # (B,L,3)
        op_oh = node_section[:, :, _OP_OFFSET:_OP_OFFSET + 2]                 # (B,L,2)
        leaf_id = node_section[:, :, lo:lo + config.n_leaf_types]              # (B,L,n_leaf)
        eval_vec = node_section[:, :, eo:eo + config.m]                        # (B,L,m)
        continuous = torch.cat([type_oh, op_oh, leaf_id, eval_vec], dim=-1)    # (B,L,d_cont)

        # Index fields (convert to long for embedding lookup)
        parent_idx = node_section[:, :, _PARENT_OFFSET:_PARENT_OFFSET + 2]    # (B,L,2)
        parent_idx = parent_idx.long().clamp(0, L)  # sentinel -1 -> mapped via clamp? No, stored as L already
        pos_idx = node_section[:, :, _POS_OFFSET].long().clamp(0, L - 1)      # (B,L)

        # Padding mask: empty nodes have type_oh = [0,0,1]
        padding_mask = type_oh[:, :, 2] > 0.5  # (B, L) True = pad

        # --- 2. Embed discrete indices ---
        p1_emb = self.parent_emb(parent_idx[:, :, 0])   # (B, L, d_pos)
        p2_emb = self.parent_emb(parent_idx[:, :, 1])   # (B, L, d_pos)
        pos_e = self.pos_emb(pos_idx)                     # (B, L, d_pos)

        # --- 3. Target encoding ---
        target_start = L * d
        target = obs[:, target_start:target_start + config.m]  # (B, m)
        target_emb = self.target_enc(target)                    # (B, d_model)
        target_exp = target_emb.unsqueeze(1).expand(-1, L, -1)  # (B, L, d_model)

        # Steps left
        steps_left = obs[:, -1:]                                # (B, 1)
        steps_exp = steps_left.unsqueeze(1).expand(-1, L, -1)   # (B, L, 1)

        # --- 4. Concatenate and project ---
        tokens = torch.cat([
            continuous, p1_emb, p2_emb, pos_e, target_exp, steps_exp,
        ], dim=-1)  # (B, L, d_input)
        tokens = self.input_proj(tokens)  # (B, L, d_model)

        # --- 5. Transformer with causal + padding mask ---
        H = self.transformer(
            tokens,
            mask=self.causal_mask,
            src_key_padding_mask=padding_mask,
        )  # (B, L, d_model)

        # --- 6. Bilinear action scoring ---
        q_values = self._score_actions(H, padding_mask)
        return q_values

    def _score_actions(
        self, H: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-values for all actions.

        Returns: (B, action_dim) where action_dim = 2*pairs + L + 1
        """
        B, L, D = H.shape

        # Gather node embeddings for all (i,j) pairs with i <= j
        hi = H[:, self.pair_idx_i]  # (B, pairs, D)
        hj = H[:, self.pair_idx_j]  # (B, pairs, D)

        q_add = self.W_add(hi, hj).squeeze(-1)  # (B, pairs)
        q_mul = self.W_mul(hi, hj).squeeze(-1)  # (B, pairs)

        # SET_OUTPUT scores per node
        q_out = self.w_out(H).squeeze(-1)  # (B, L)

        # STOP score on mean-pooled (non-padded) embeddings
        mask_float = (~padding_mask).float().unsqueeze(-1)  # (B, L, 1)
        n_real = mask_float.sum(dim=1).clamp(min=1)          # (B, 1)
        pooled = (H * mask_float).sum(dim=1) / n_real         # (B, D)
        q_stop = self.w_stop(pooled)                           # (B, 1)

        return torch.cat([q_add, q_mul, q_out, q_stop], dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
