"""Observation encoding for the polynomial circuit environment.

Flat observation layout (per node, d_node_raw floats):
  [type_onehot(3) | op_onehot(2) | parent1_idx(1) | parent2_idx(1)
   | pos_idx(1) | leaf_id(n_leaf_types) | eval_vector(m)]

Full obs:
  [L nodes x d_node_raw | target_eval(m) | steps_left(1)]

Parent/position indices are stored as raw ints; the network embeds them.
Sentinel value for "no parent" is stored as L (mapped to padding_idx in embedding).
"""

from __future__ import annotations

import math
import numpy as np
from fractions import Fraction
from typing import List, Tuple

from ..core.node import Node, OP_VAR, OP_CONST, OP_ADD, OP_MUL
from ..core.fingerprints import eval_poly_points, EvalPoint
from ..config import Config


# Offsets within each node's feature vector
_TYPE_OFFSET = 0     # 3 floats: [input, op, empty]
_OP_OFFSET = 3       # 2 floats: [add, mul]
_PARENT_OFFSET = 5   # 2 floats: parent1_idx, parent2_idx
_POS_OFFSET = 7      # 1 float: pos_idx


def _leaf_offset(config: Config) -> int:
    return 8  # after type(3) + op(2) + parents(2) + pos(1)


def _eval_offset(config: Config) -> int:
    return 8 + config.n_leaf_types


def node_eval_vector(node: Node, eval_points: List[EvalPoint]) -> List[float]:
    """Return the node's polynomial evaluated at each eval point as floats."""
    if node.evals is not None:
        return [float(e) for e in node.evals]
    return [float(e) for e in eval_poly_points(node.poly, eval_points)]


def encode_node(
    node: Node,
    node_idx: int,
    eval_points: List[EvalPoint],
    config: Config,
) -> np.ndarray:
    """Encode one node as a d_node_raw-dim vector.

    Layout: [type_oh(3), op_oh(2), parent1_idx, parent2_idx,
             pos_idx, leaf_id(n_leaf_types), eval_vec(m)]
    """
    d = config.d_node_raw
    feat = np.zeros(d, dtype=np.float32)

    # type_onehot(3): [input=0, op=1, empty=2]
    if node.op in (OP_VAR, OP_CONST):
        feat[_TYPE_OFFSET] = 1.0       # input
    elif node.op in (OP_ADD, OP_MUL):
        feat[_TYPE_OFFSET + 1] = 1.0   # op
    # else: empty (stays zero -- but real nodes won't hit this)

    # op_onehot(2): [add, mul]
    if node.op == OP_ADD:
        feat[_OP_OFFSET] = 1.0
    elif node.op == OP_MUL:
        feat[_OP_OFFSET + 1] = 1.0

    # parent indices (raw ints for embedding lookup)
    if node.op in (OP_ADD, OP_MUL):
        feat[_PARENT_OFFSET] = float(node.args[0])
        feat[_PARENT_OFFSET + 1] = float(node.args[1])
    else:
        # Sentinel: L means "no parent" (padding_idx in embedding)
        feat[_PARENT_OFFSET] = float(config.L)
        feat[_PARENT_OFFSET + 1] = float(config.L)

    # position index
    feat[_POS_OFFSET] = float(node_idx)

    # leaf_id one-hot
    lo = _leaf_offset(config)
    if node.op == OP_VAR:
        var_idx = node.args[0]
        feat[lo + var_idx] = 1.0
    elif node.op == OP_CONST:
        feat[lo + config.n_vars] = 1.0

    # eval vector — tanh-normalised to prevent float32 overflow at high degree
    eo = _eval_offset(config)
    evals = node_eval_vector(node, eval_points)
    scale = config.eval_norm_scale
    for k, v in enumerate(evals[:config.m]):
        feat[eo + k] = math.tanh(v / scale) if scale > 0.0 else v

    return feat


def encode_obs(
    nodes: List[Node],
    target_evals: Tuple[Fraction, ...],
    eval_points: List[EvalPoint],
    steps_left: int,
    max_ops: int,
    config: Config,
) -> np.ndarray:
    """Build flat observation vector of size config.obs_dim.

    Layout:
      [node_0_feats ... node_{L-1}_feats | target_eval_vector | steps_left_norm]

    Unused node slots are zero-padded (type_onehot = [0,0,1] "empty" is implicit).
    """
    d = config.d_node_raw
    obs = np.zeros(config.obs_dim, dtype=np.float32)

    visible = min(len(nodes), config.L)
    for i in range(visible):
        obs[i * d:(i + 1) * d] = encode_node(nodes[i], i, eval_points, config)

    # Empty slots: parent indices should also be sentinel L
    for i in range(visible, config.L):
        obs[i * d + _PARENT_OFFSET] = float(config.L)
        obs[i * d + _PARENT_OFFSET + 1] = float(config.L)
        obs[i * d + _POS_OFFSET] = float(i)
        obs[i * d + _TYPE_OFFSET + 2] = 1.0  # empty type

    # Target eval vector — same tanh normalisation as node eval vectors
    target_start = config.L * d
    scale = config.eval_norm_scale
    for k, e in enumerate(target_evals[:config.m]):
        v = float(e)
        obs[target_start + k] = math.tanh(v / scale) if scale > 0.0 else v

    # Steps left (normalized)
    obs[-1] = steps_left / max(max_ops, 1)

    return obs


def extract_goal(obs: np.ndarray, config: Config) -> np.ndarray:
    """Extract the target eval vector from a flat observation."""
    start = config.L * config.d_node_raw
    return obs[start:start + config.m].copy()


def replace_goal(
    obs: np.ndarray, new_goal: np.ndarray, config: Config,
) -> np.ndarray:
    """Return a copy of obs with the target eval vector replaced."""
    new_obs = obs.copy()
    start = config.L * config.d_node_raw
    new_obs[start:start + config.m] = new_goal
    return new_obs


def get_num_real_nodes(obs: np.ndarray, config: Config) -> int:
    """Count the number of non-empty nodes by checking type_onehot."""
    d = config.d_node_raw
    count = 0
    for i in range(config.L):
        # If empty type bit is NOT set, it's a real node
        if obs[i * d + _TYPE_OFFSET + 2] < 0.5:
            count += 1
        else:
            break  # nodes are contiguous
    return count
