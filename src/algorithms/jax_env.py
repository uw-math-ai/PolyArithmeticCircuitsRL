"""Vectorized circuit game environment in pure JAX for parallel MCTS.

Represents the full game state as fixed-size JAX arrays so that mctx can
vmap over batches of environments. Polynomial arithmetic (add/mul mod p)
is implemented directly in JAX, and the action space uses the same
upper-triangular encoding as the PyTorch version.

Key differences from the PyTorch CircuitGame:
  - No Python objects (FastPoly, lists, sets) — everything is jnp arrays.
  - Fixed-size buffers: node coefficients are pre-allocated to max_nodes.
  - Factor library / subgoal rewards are NOT ported (they require SymPy).
    Reward shaping is simplified to success_reward + step_penalty.
  - The environment is purely functional: step() takes state in, returns
    state out, with no mutation.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class EnvState(NamedTuple):
    """Immutable environment state represented as JAX arrays.

    All arrays have fixed shapes for jit/vmap compatibility.

    Attributes:
        node_coeffs: (max_nodes, target_size) — flattened polynomial coeffs
            for each circuit node, mod p.
        node_features: (max_nodes, 4) — [is_input, is_constant, is_op, op_value].
        edge_src: (max_edges,) — source indices of edges (int32, -1 = unused).
        edge_dst: (max_edges,) — destination indices of edges.
        num_nodes: int32 scalar — number of active nodes.
        num_edges: int32 scalar — number of active edges (each op adds 4: 2 bidir).
        target_coeffs: (target_size,) — flattened target polynomial, mod p.
        steps_taken: int32 scalar.
        done: bool scalar.
        is_success: bool scalar.
    """
    node_coeffs: jnp.ndarray
    node_features: jnp.ndarray
    edge_src: jnp.ndarray
    edge_dst: jnp.ndarray
    num_nodes: jnp.ndarray
    num_edges: jnp.ndarray
    target_coeffs: jnp.ndarray
    steps_taken: jnp.ndarray
    done: jnp.ndarray
    is_success: jnp.ndarray


class EnvConfig(NamedTuple):
    """Static environment configuration (not traced by JAX)."""
    n_variables: int
    mod: int
    max_degree: int
    max_complexity: int
    max_steps: int
    max_nodes: int        # n_variables + 1 + max_complexity
    max_actions: int      # max_nodes * (max_nodes + 1)
    target_size: int      # (max_degree + 1) ** n_variables
    node_feature_dim: int
    success_reward: float
    step_penalty: float
    gamma: float


def make_env_config(config) -> EnvConfig:
    """Build an EnvConfig from the main Config dataclass."""
    max_deg = config.effective_max_degree
    max_nodes = config.max_nodes
    return EnvConfig(
        n_variables=config.n_variables,
        mod=config.mod,
        max_degree=max_deg,
        max_complexity=config.max_complexity,
        max_steps=config.max_steps,
        max_nodes=max_nodes,
        max_actions=config.max_actions,
        target_size=config.target_size,
        node_feature_dim=config.node_feature_dim,
        success_reward=config.success_reward,
        step_penalty=config.step_penalty,
        gamma=config.gamma,
    )


# ---------------------------------------------------------------------------
# Action encoding / decoding  (mirrors action_space.py)
# ---------------------------------------------------------------------------

def encode_action(op: jnp.ndarray, i: jnp.ndarray, j: jnp.ndarray,
                  max_nodes: int) -> jnp.ndarray:
    """Encode (op, i, j) into a single action index (JAX-compatible)."""
    # Ensure i <= j
    i_new = jnp.minimum(i, j)
    j_new = jnp.maximum(i, j)
    pair_idx = i_new * max_nodes - i_new * (i_new - 1) // 2 + (j_new - i_new)
    return 2 * pair_idx + op


def decode_action(action_idx: jnp.ndarray, max_nodes: int):
    """Decode action index into (op, i, j).  Pure JAX, no loops."""
    op = action_idx % 2
    pair_idx = action_idx // 2

    discriminant = (2 * max_nodes + 1) ** 2 - 8 * pair_idx
    # Integer square root via floor of float sqrt (safe for our sizes < 1000).
    sqrt_disc = jnp.floor(jnp.sqrt(discriminant.astype(jnp.float32))).astype(jnp.int32)
    i = (2 * max_nodes + 1 - sqrt_disc) // 2

    # Clamp to valid range.
    i = jnp.clip(i, 0, max_nodes - 1)
    row_start = i * max_nodes - i * (i - 1) // 2
    # Adjust i if needed (off-by-one from floor).
    i = jnp.where(row_start > pair_idx, i - 1, i)
    row_start = i * max_nodes - i * (i - 1) // 2
    j = i + (pair_idx - row_start)
    return op, i, j


def get_valid_actions_mask(num_nodes: jnp.ndarray, max_nodes: int,
                           max_actions: int) -> jnp.ndarray:
    """Boolean mask of valid actions given current node count. Pure JAX."""
    action_indices = jnp.arange(max_actions, dtype=jnp.int32)
    _, i_arr, j_arr = jax.vmap(
        lambda a: decode_action(a, max_nodes)
    )(action_indices)
    return (i_arr < num_nodes) & (j_arr < num_nodes)


# ---------------------------------------------------------------------------
# Polynomial arithmetic in JAX (flat coefficient vectors, mod p)
# ---------------------------------------------------------------------------

def poly_add(a: jnp.ndarray, b: jnp.ndarray, mod: int) -> jnp.ndarray:
    """Add two polynomials (flat coefficient vectors) mod p."""
    return (a + b) % mod


def poly_mul(a: jnp.ndarray, b: jnp.ndarray, mod: int,
             n_variables: int, max_degree: int) -> jnp.ndarray:
    """Multiply two polynomials mod p using reshape + n-d convolution + truncate.

    For 2 variables with max_degree d, coeffs are (d+1)x(d+1) arrays.
    We reshape flat vectors, convolve, truncate, flatten, and reduce mod p.
    """
    shape = (max_degree + 1,) * n_variables
    a_nd = a.reshape(shape)
    b_nd = b.reshape(shape)

    if n_variables == 1:
        # 1D convolution: jnp.convolve, truncate to max_degree+1.
        result = jnp.convolve(a_nd, b_nd)
        result = result[:max_degree + 1]
    elif n_variables == 2:
        # 2D convolution via jax.scipy.signal.
        from jax.scipy.signal import correlate2d
        # correlate2d with flipped kernel = convolve
        b_flip = b_nd[::-1, ::-1]
        # Pad a to get full convolution result, then use 'same' size trick.
        # Actually, use direct approach: convolve = correlate with flipped kernel.
        # Full convolution output shape = (2*d+1, 2*d+1); we only need (d+1, d+1).
        full_size = 2 * max_degree + 1
        a_pad = jnp.zeros((full_size, full_size), dtype=a.dtype)
        a_pad = a_pad.at[:max_degree + 1, :max_degree + 1].set(a_nd)
        # Convolve via correlate with flipped b.
        result_2d = jnp.zeros((max_degree + 1, max_degree + 1), dtype=a.dtype)
        # Direct nested loop replaced by: slide b_flip across a_pad.
        # For small sizes (7x7), this is fine to express as:
        result_full = _convolve_2d_jax(a_nd, b_nd, max_degree)
        result = result_full[:max_degree + 1, :max_degree + 1]
    else:
        # General n-d: fall back to nested-loop approach via scan.
        # For now, only support n_variables <= 2 in the JAX path.
        raise NotImplementedError(
            f"JAX poly_mul not implemented for n_variables={n_variables}"
        )

    return result.flatten() % mod


def _convolve_2d_jax(a: jnp.ndarray, b: jnp.ndarray,
                     max_degree: int) -> jnp.ndarray:
    """2D polynomial multiplication truncated to (max_degree+1)^2.

    Uses jax.lax.conv_general_dilated for GPU-friendly 2D convolution.
    """
    d = max_degree + 1
    # Reshape for conv: (batch=1, channels=1, H, W)
    a4d = a.reshape(1, 1, d, d).astype(jnp.float32)
    b4d = b.reshape(1, 1, d, d).astype(jnp.float32)
    # conv_general_dilated computes cross-correlation. To get polynomial
    # convolution (a*b)[i,j] = sum a[m,n]*b[i-m,j-n], we flip b and pad a
    # on the LEFT/TOP so that result[h,w] maps to product index (h,w).
    b_flip = b4d[:, :, ::-1, ::-1]
    a_padded = jnp.pad(a4d, ((0, 0), (0, 0), (d - 1, 0), (d - 1, 0)))
    result = jax.lax.conv_general_dilated(
        a_padded, b_flip,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
    )
    # Result shape: (1, 1, d, d). First d×d entries are the truncated product.
    return result[0, 0, :d, :d].astype(jnp.int32)


# ---------------------------------------------------------------------------
# Environment init / step  (pure functions)
# ---------------------------------------------------------------------------

def make_initial_node_coeffs(env_config: EnvConfig) -> jnp.ndarray:
    """Build the initial node coefficient matrix for [x0, ..., x_{n-1}, 1].

    Returns:
        (n_variables + 1, target_size) array of int32 coefficients.
    """
    n_vars = env_config.n_variables
    d = env_config.max_degree
    mod = env_config.mod
    shape = (d + 1,) * n_vars
    target_size = env_config.target_size

    rows = []
    for var_idx in range(n_vars):
        arr = jnp.zeros(shape, dtype=jnp.int32)
        idx = tuple(1 if i == var_idx else 0 for i in range(n_vars))
        arr = arr.at[idx].set(1)
        rows.append(arr.flatten())

    # Constant 1.
    arr = jnp.zeros(shape, dtype=jnp.int32)
    idx = (0,) * n_vars
    arr = arr.at[idx].set(1)
    rows.append(arr.flatten())

    return jnp.stack(rows, axis=0)  # (n_vars + 1, target_size)


def reset(env_config: EnvConfig, target_coeffs: jnp.ndarray) -> EnvState:
    """Reset the environment with a new target.

    Args:
        env_config: Static environment configuration.
        target_coeffs: Flat int32 array of shape (target_size,), mod p coefficients.

    Returns:
        Initial EnvState.
    """
    max_nodes = env_config.max_nodes
    target_size = env_config.target_size
    max_edges = max_nodes * 4  # Conservative upper bound on edge slots.

    n_initial = env_config.n_variables + 1
    init_coeffs = make_initial_node_coeffs(env_config)  # (n_initial, target_size)

    node_coeffs = jnp.zeros((max_nodes, target_size), dtype=jnp.int32)
    node_coeffs = node_coeffs.at[:n_initial].set(init_coeffs)

    node_features = jnp.zeros((max_nodes, 4), dtype=jnp.float32)
    for i in range(env_config.n_variables):
        node_features = node_features.at[i].set(
            jnp.array([1.0, 0.0, 0.0, 0.0])
        )
    node_features = node_features.at[env_config.n_variables].set(
        jnp.array([0.0, 1.0, 0.0, 0.0])
    )

    edge_src = jnp.full((max_edges,), -1, dtype=jnp.int32)
    edge_dst = jnp.full((max_edges,), -1, dtype=jnp.int32)

    return EnvState(
        node_coeffs=node_coeffs,
        node_features=node_features,
        edge_src=edge_src,
        edge_dst=edge_dst,
        num_nodes=jnp.int32(n_initial),
        num_edges=jnp.int32(0),
        target_coeffs=target_coeffs.astype(jnp.int32),
        steps_taken=jnp.int32(0),
        done=jnp.bool_(False),
        is_success=jnp.bool_(False),
    )


def step(env_config: EnvConfig, state: EnvState,
         action: jnp.ndarray) -> tuple:
    """Take one step in the environment. Pure function.

    Args:
        env_config: Static config.
        state: Current EnvState.
        action: Scalar int32 action index.

    Returns:
        (next_state, reward, done, is_success)
    """
    mod = env_config.mod
    max_nodes = env_config.max_nodes

    op, i, j = decode_action(action, max_nodes)

    poly_i = state.node_coeffs[i]
    poly_j = state.node_coeffs[j]

    # Compute new polynomial.
    new_coeffs_add = poly_add(poly_i, poly_j, mod)
    new_coeffs_mul = poly_mul(poly_i, poly_j, mod,
                              env_config.n_variables, env_config.max_degree)
    new_coeffs = jnp.where(op == 0, new_coeffs_add, new_coeffs_mul)

    # Append new node.
    new_idx = state.num_nodes
    node_coeffs = state.node_coeffs.at[new_idx].set(new_coeffs)

    op_value = jnp.where(op == 0, 0.5, 1.0)
    new_feature = jnp.array([0.0, 0.0, 1.0, op_value])
    node_features = state.node_features.at[new_idx].set(new_feature)

    # Add bidirectional edges (4 new edges: i->new, new->i, j->new, new->j).
    ne = state.num_edges
    edge_src = state.edge_src.at[ne].set(i)
    edge_dst = state.edge_dst.at[ne].set(new_idx)
    edge_src = edge_src.at[ne + 1].set(new_idx)
    edge_dst = edge_dst.at[ne + 1].set(i)
    edge_src = edge_src.at[ne + 2].set(j)
    edge_dst = edge_dst.at[ne + 2].set(new_idx)
    edge_src = edge_src.at[ne + 3].set(new_idx)
    edge_dst = edge_dst.at[ne + 3].set(j)
    num_edges = ne + 4

    num_nodes = new_idx + 1
    steps_taken = state.steps_taken + 1

    # Success check.
    is_success = jnp.all(new_coeffs == state.target_coeffs)

    # Termination.
    at_max_steps = steps_taken >= env_config.max_steps
    at_max_nodes = num_nodes >= max_nodes
    done = is_success | at_max_steps | at_max_nodes

    # Reward: step_penalty + success_reward if success.
    reward = env_config.step_penalty + jnp.where(
        is_success, env_config.success_reward, 0.0
    )

    next_state = EnvState(
        node_coeffs=node_coeffs,
        node_features=node_features,
        edge_src=edge_src,
        edge_dst=edge_dst,
        num_nodes=num_nodes.astype(jnp.int32),
        num_edges=num_edges.astype(jnp.int32),
        target_coeffs=state.target_coeffs,
        steps_taken=steps_taken.astype(jnp.int32),
        done=done,
        is_success=is_success,
    )
    return next_state, reward.astype(jnp.float32), done, is_success


def get_observation(env_config: EnvConfig, state: EnvState) -> dict:
    """Extract observation arrays from state for the policy network.

    Returns:
        Dict with:
          'node_features': (max_nodes, 4) float32
          'edge_src': (max_edges,) int32
          'edge_dst': (max_edges,) int32
          'num_nodes': int32 scalar
          'num_edges': int32 scalar
          'target': (target_size,) float32, normalized to [0, 1]
          'mask': (max_actions,) bool
    """
    target_norm = state.target_coeffs.astype(jnp.float32) / env_config.mod
    mask = get_valid_actions_mask(
        state.num_nodes, env_config.max_nodes, env_config.max_actions
    )
    return {
        'node_features': state.node_features,
        'edge_src': state.edge_src,
        'edge_dst': state.edge_dst,
        'num_nodes': state.num_nodes,
        'num_edges': state.num_edges,
        'target': target_norm,
        'mask': mask,
    }
