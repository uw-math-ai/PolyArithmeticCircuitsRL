"""Vectorized circuit game environment in pure JAX for parallel MCTS.

Represents the full game state as fixed-size JAX arrays so that mctx can
vmap over batches of environments. Polynomial arithmetic (add/mul mod p)
is implemented directly in JAX, and the action space uses the same
upper-triangular encoding as the PyTorch version.

The factor-library reward path is implemented in a JAX-friendly form:
  - initial factor subgoals are precomputed on the host at reset time;
  - subgoal hits, library bonus, additive completion, and exact
    multiplicative completion are handled in JAX;
  - dynamic discovery adds direct residual and direct quotient subgoals;
  - the cross-episode library itself stays on the host and is exported to
    JAX as a dense cache of known polynomial coefficient vectors.

This preserves the exact success condition while keeping the hot path
GPU/JIT-friendly.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


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
        subgoal_coeffs: (max_subgoals, target_size) active per-episode subgoals.
        subgoal_active: (max_subgoals,) bool mask for valid subgoal slots.
        subgoal_library_known: (max_subgoals,) bool mask for subgoals that were
            already in the cross-episode library at the time they were added.
        subgoal_hit: (max_subgoals,) bool mask for subgoals already rewarded in
            this episode.
        additive_complete_hit: bool scalar.
        mult_complete_hit: bool scalar.
        on_path_coeffs: (on_path_max_size, target_size) cached oracle nodes.
        on_path_hashes: (on_path_max_size,) uint32 hash prefilter values.
        on_path_steps: (on_path_max_size,) board-step values for oracle nodes.
        on_path_route_masks: (on_path_max_size,) uint32 coherent-route masks.
        on_path_active: (on_path_max_size,) bool mask for valid oracle slots.
        on_path_hit: (on_path_max_size,) bool mask for already rewarded nodes.
        on_path_count: int32 scalar.
        on_path_total: int32 scalar.
        on_path_deepest_step: int32 scalar.
        on_path_active_route_mask: uint32 scalar for currently compatible routes.
        target_board_step: int32 scalar.
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
    subgoal_coeffs: jnp.ndarray
    subgoal_active: jnp.ndarray
    subgoal_library_known: jnp.ndarray
    subgoal_hit: jnp.ndarray
    additive_complete_hit: jnp.ndarray
    mult_complete_hit: jnp.ndarray
    on_path_coeffs: jnp.ndarray
    on_path_hashes: jnp.ndarray
    on_path_steps: jnp.ndarray
    on_path_route_masks: jnp.ndarray
    on_path_active: jnp.ndarray
    on_path_hit: jnp.ndarray
    on_path_count: jnp.ndarray
    on_path_total: jnp.ndarray
    on_path_deepest_step: jnp.ndarray
    on_path_active_route_mask: jnp.ndarray
    target_board_step: jnp.ndarray
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
    terminal_success_reward: float
    step_penalty: float
    gamma: float
    reward_mode: str
    use_reward_shaping: bool
    factor_library_enabled: bool
    factor_subgoal_reward: float
    factor_library_bonus: float
    completion_bonus: float
    max_subgoals: int
    graph_onpath_shaping_coeff: float
    on_path_terminal_zero: bool
    on_path_phi_mode: str
    on_path_route_consistency: bool
    on_path_route_consistency_mode: str
    on_path_max_size: int
    initial_node_coeffs: jnp.ndarray
    base_node_coeffs: jnp.ndarray
    on_path_hash_weights: jnp.ndarray
    monomial_exponents: jnp.ndarray
    monomial_order_desc: jnp.ndarray
    monomial_strides: jnp.ndarray
    inv_mod_table: jnp.ndarray


def make_env_config(config) -> EnvConfig:
    """Build an EnvConfig from the main Config dataclass."""
    max_deg = config.effective_max_degree
    max_nodes = config.max_nodes
    on_path_max_size = (
        max(1, int(config.on_path_max_size))
        if config.reward_mode == "clean_onpath"
        else 1
    )
    initial_node_coeffs = make_initial_node_coeffs_raw(
        config.n_variables, max_deg, config.mod
    )
    base_node_coeffs = initial_node_coeffs[: config.n_variables + 1]
    monomial_exponents, monomial_order_desc, monomial_strides = (
        make_monomial_metadata(config.n_variables, max_deg)
    )
    inv_mod_table = make_inv_mod_table(config.mod)
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
        terminal_success_reward=config.terminal_success_reward,
        step_penalty=config.step_penalty,
        gamma=config.gamma,
        reward_mode=config.reward_mode,
        use_reward_shaping=config.use_reward_shaping,
        factor_library_enabled=(
            config.reward_mode == "legacy" and config.factor_library_enabled
        ),
        factor_subgoal_reward=config.factor_subgoal_reward,
        factor_library_bonus=config.factor_library_bonus,
        completion_bonus=config.completion_bonus,
        max_subgoals=max(16, 4 * max_nodes),
        graph_onpath_shaping_coeff=config.graph_onpath_shaping_coeff,
        on_path_terminal_zero=config.on_path_terminal_zero,
        on_path_phi_mode=config.on_path_phi_mode,
        on_path_route_consistency=config.on_path_route_consistency,
        on_path_route_consistency_mode=(
            "off"
            if not config.on_path_route_consistency
            else config.on_path_route_consistency_mode
        ),
        on_path_max_size=on_path_max_size,
        initial_node_coeffs=initial_node_coeffs,
        base_node_coeffs=base_node_coeffs,
        on_path_hash_weights=make_on_path_hash_weights(config.target_size),
        monomial_exponents=monomial_exponents,
        monomial_order_desc=monomial_order_desc,
        monomial_strides=monomial_strides,
        inv_mod_table=inv_mod_table,
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
    """Multiply two polynomials mod p using n-d convolution + truncate.

    Supports n_variables = 1, 2, or 3 via jax.lax.conv_general_dilated.
    Coefficients are reshaped to (d+1)^n arrays, convolved, truncated back
    to (d+1)^n, flattened, and reduced mod p.
    """
    shape = (max_degree + 1,) * n_variables
    a_nd = a.reshape(shape)
    b_nd = b.reshape(shape)
    result = _convolve_nd_jax(a_nd, b_nd, n_variables, max_degree)
    return result.flatten() % mod


# Dimension number strings for jax.lax.conv_general_dilated by spatial rank.
_CONV_DIM_NUMBERS = {
    1: ('NCH', 'OIH', 'NCH'),
    2: ('NCHW', 'OIHW', 'NCHW'),
    3: ('NCDHW', 'OIDHW', 'NCDHW'),
}


def _convolve_nd_jax(a: jnp.ndarray, b: jnp.ndarray,
                     n_variables: int, max_degree: int) -> jnp.ndarray:
    """N-dimensional polynomial multiplication truncated to (max_degree+1)^n.

    Uses jax.lax.conv_general_dilated for GPU-friendly n-D convolution.
    Supports n_variables = 1, 2, or 3 (matching JAX's supported spatial dims).

    Args:
        a: n-d array of shape (d+1,)*n_variables (int32 coefficients).
        b: n-d array of shape (d+1,)*n_variables (int32 coefficients).
        n_variables: Number of polynomial variables (1, 2, or 3).
        max_degree: Maximum degree per variable.

    Returns:
        n-d int32 array of shape (d+1,)*n_variables — the truncated product.
    """
    d = max_degree + 1
    shape = (d,) * n_variables

    # Reshape for conv: (batch=1, channels=1, *spatial_dims)
    a_conv = a.reshape((1, 1) + shape).astype(jnp.float32)
    b_conv = b.reshape((1, 1) + shape).astype(jnp.float32)

    # Flip kernel along all spatial axes for convolution (correlate w/ flipped = convolve).
    flip_slices = (slice(None), slice(None)) + tuple(
        slice(None, None, -1) for _ in range(n_variables)
    )
    b_flip = b_conv[flip_slices]

    # Pad input on the left/front of each spatial dimension by (d-1).
    pad_config = [(0, 0), (0, 0)] + [(d - 1, 0)] * n_variables
    a_padded = jnp.pad(a_conv, pad_config)

    result = jax.lax.conv_general_dilated(
        a_padded, b_flip,
        window_strides=(1,) * n_variables,
        padding='VALID',
        dimension_numbers=_CONV_DIM_NUMBERS[n_variables],
    )

    # Extract the first d entries along each spatial dimension.
    out_slices = (0, 0) + tuple(slice(None, d) for _ in range(n_variables))
    return result[out_slices].astype(jnp.int32)


def make_inv_mod_table(mod: int) -> jnp.ndarray:
    """Multiplicative inverses in F_p for 0..p-1 (0 maps to 0)."""
    inv = np.zeros((mod,), dtype=np.int32)
    for a in range(1, mod):
        inv[a] = pow(a, -1, mod)
    return jnp.array(inv, dtype=jnp.int32)


def make_on_path_hash_weights(target_size: int) -> jnp.ndarray:
    """Deterministic uint32 weights matching game_board.on_path.hash_weights_np."""
    idx = np.arange(target_size, dtype=np.uint32)
    weights = (
        idx * np.uint32(2654435761) + np.uint32(2246822519)
    ).astype(np.uint32)
    return jnp.array(weights, dtype=jnp.uint32)


def make_monomial_metadata(n_variables: int, max_degree: int):
    """Precompute exponent tuples, descending lex order, and flat strides."""
    d = max_degree + 1
    shape = (d,) * n_variables
    exponents = np.indices(shape, dtype=np.int32).reshape(n_variables, -1).T
    order_desc = np.lexsort(tuple(exponents[:, i] for i in reversed(range(n_variables))))[::-1]
    strides = np.array([d ** (n_variables - 1 - i) for i in range(n_variables)], dtype=np.int32)
    return (
        jnp.array(exponents, dtype=jnp.int32),
        jnp.array(order_desc, dtype=jnp.int32),
        jnp.array(strides, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Reward shaping helpers (potential-based, Ng et al. 1999)
# ---------------------------------------------------------------------------

def _term_similarity(node_coeffs: jnp.ndarray,
                     target_coeffs: jnp.ndarray) -> jnp.ndarray:
    """Fraction of matching nonzero target coefficients. Pure JAX scalar.

    Mirrors FastPoly.term_similarity: counts positions where both the target
    has a nonzero coefficient and the node matches it, divided by the total
    number of nonzero target terms.
    """
    target_nonzero = target_coeffs != 0
    total = jnp.maximum(target_nonzero.sum(), 1).astype(jnp.float32)
    matching = ((node_coeffs == target_coeffs) & target_nonzero).sum().astype(jnp.float32)
    return matching / total


def _best_similarity(node_coeffs: jnp.ndarray, num_nodes: jnp.ndarray,
                     target_coeffs: jnp.ndarray,
                     max_nodes: int) -> jnp.ndarray:
    """Max term_similarity across all active circuit nodes.

    This is the shaping potential phi(s). A value of 1.0 means some node
    already matches the target exactly.
    """
    sims = jax.vmap(lambda nc: _term_similarity(nc, target_coeffs))(node_coeffs)
    mask = jnp.arange(max_nodes) < num_nodes
    sims = jnp.where(mask, sims, 0.0)
    return sims.max()


def _poly_is_zero(poly: jnp.ndarray) -> jnp.ndarray:
    return ~jnp.any(poly != 0)


def _poly_is_scalar(poly: jnp.ndarray) -> jnp.ndarray:
    return (poly[0] != 0) & ~jnp.any(poly[1:] != 0)


def _poly_equal_any(poly: jnp.ndarray, polys: jnp.ndarray,
                    mask: jnp.ndarray) -> jnp.ndarray:
    eq = jnp.all(polys == poly[None, :], axis=1)
    return jnp.any(eq & mask)


def _poly_match_index(poly: jnp.ndarray, polys: jnp.ndarray,
                      mask: jnp.ndarray) -> tuple:
    matches = jnp.all(polys == poly[None, :], axis=1) & mask
    idx = jnp.argmax(matches.astype(jnp.int32))
    return idx, jnp.any(matches)


def _poly_hash(poly: jnp.ndarray, env_config: EnvConfig) -> jnp.ndarray:
    vals = poly.astype(jnp.uint32) + jnp.uint32(1)
    return jnp.bitwise_xor.reduce(vals * env_config.on_path_hash_weights)


def _route_bits() -> jnp.ndarray:
    return (jnp.uint32(1) << jnp.arange(32, dtype=jnp.uint32))


def _best_route_count_phi(state: EnvState) -> jnp.ndarray:
    bits = _route_bits()
    in_route = (
        (state.on_path_route_masks[None, :] & bits[:, None]) != 0
    ) & state.on_path_active[None, :]
    route_totals = jnp.sum(in_route.astype(jnp.float32), axis=1)
    route_hits = jnp.sum(
        (in_route & state.on_path_hit[None, :]).astype(jnp.float32),
        axis=1,
    )
    route_phi = jnp.where(route_totals > 0.0, route_hits / route_totals, 0.0)
    return jnp.max(route_phi)


def _best_route_max_step_phi(state: EnvState) -> jnp.ndarray:
    bits = _route_bits()
    in_route = (
        (state.on_path_route_masks[None, :] & bits[:, None]) != 0
    ) & state.on_path_active[None, :]
    route_has_nodes = jnp.any(in_route, axis=1)
    route_hit_steps = jnp.where(
        in_route & state.on_path_hit[None, :],
        state.on_path_steps[None, :],
        jnp.int32(0),
    )
    deepest = jnp.max(route_hit_steps, axis=1).astype(jnp.float32)
    target_step = state.target_board_step.astype(jnp.float32)
    route_phi = jnp.where(
        route_has_nodes & (state.target_board_step > 0),
        deepest / jnp.maximum(target_step, 1.0),
        0.0,
    )
    return jnp.max(route_phi)


def _on_path_phi(state: EnvState, env_config: EnvConfig) -> jnp.ndarray:
    if env_config.on_path_route_consistency_mode == "best_route_phi":
        if env_config.on_path_phi_mode == "count":
            return _best_route_count_phi(state)
        if env_config.on_path_phi_mode == "max_step":
            return _best_route_max_step_phi(state)
        return jnp.float32(0.0)

    if env_config.on_path_phi_mode == "count":
        total = state.on_path_total.astype(jnp.float32)
        phi = state.on_path_count.astype(jnp.float32) / jnp.maximum(
            total, 1.0
        )
        return jnp.where(state.on_path_total <= 0, 0.0, phi)
    if env_config.on_path_phi_mode == "max_step":
        target_step = state.target_board_step.astype(jnp.float32)
        phi = state.on_path_deepest_step.astype(jnp.float32) / jnp.maximum(
            target_step, 1.0
        )
        return jnp.where(state.target_board_step <= 0, 0.0, phi)
    return jnp.float32(0.0)


def _on_path_match_index(poly: jnp.ndarray, state: EnvState,
                         env_config: EnvConfig) -> tuple:
    """Hash-prefilter then coefficient-verify a clean_onpath hit."""
    poly_hash = _poly_hash(poly, env_config)
    route_ok = jnp.ones_like(state.on_path_active, dtype=jnp.bool_)
    if env_config.on_path_route_consistency_mode == "lock_on_first_hit":
        route_ok = (state.on_path_route_masks & state.on_path_active_route_mask) != 0
    candidate = (
        state.on_path_active
        & (~state.on_path_hit)
        & (state.on_path_hashes == poly_hash)
        & route_ok
    )
    coeff_match = jnp.all(state.on_path_coeffs == poly[None, :], axis=1)
    matches = candidate & coeff_match
    idx = jnp.argmax(matches.astype(jnp.int32))
    return idx, jnp.any(matches)


def _is_base_poly(poly: jnp.ndarray, env_config: EnvConfig) -> jnp.ndarray:
    return _poly_equal_any(
        poly, env_config.base_node_coeffs,
        jnp.ones((env_config.base_node_coeffs.shape[0],), dtype=jnp.bool_),
    )


def _library_contains(poly: jnp.ndarray, library_coeffs: jnp.ndarray,
                      library_mask: jnp.ndarray) -> jnp.ndarray:
    if library_coeffs is None:
        return jnp.bool_(False)
    return _poly_equal_any(poly, library_coeffs, library_mask)


def _existing_node_contains(poly: jnp.ndarray, node_coeffs: jnp.ndarray,
                            num_nodes: jnp.ndarray) -> jnp.ndarray:
    mask = jnp.arange(node_coeffs.shape[0]) < num_nodes
    return _poly_equal_any(poly, node_coeffs, mask)


def _first_free_index(mask: jnp.ndarray) -> jnp.ndarray:
    free = ~mask
    return jnp.argmax(free.astype(jnp.int32))


def _add_subgoal_if_new(subgoal_coeffs: jnp.ndarray,
                        subgoal_active: jnp.ndarray,
                        subgoal_library_known: jnp.ndarray,
                        subgoal_hit: jnp.ndarray,
                        poly: jnp.ndarray,
                        poly_library_known: jnp.ndarray,
                        valid_to_add: jnp.ndarray):
    already_present = _poly_equal_any(poly, subgoal_coeffs, subgoal_active)
    free_exists = jnp.any(~subgoal_active)
    should_add = valid_to_add & (~already_present) & free_exists
    idx = _first_free_index(subgoal_active)
    subgoal_coeffs = jax.lax.cond(
        should_add,
        lambda arr: arr.at[idx].set(poly),
        lambda arr: arr,
        subgoal_coeffs,
    )
    subgoal_active = jax.lax.cond(
        should_add,
        lambda arr: arr.at[idx].set(True),
        lambda arr: arr,
        subgoal_active,
    )
    subgoal_library_known = jax.lax.cond(
        should_add,
        lambda arr: arr.at[idx].set(poly_library_known),
        lambda arr: arr,
        subgoal_library_known,
    )
    subgoal_hit = jax.lax.cond(
        should_add,
        lambda arr: arr.at[idx].set(False),
        lambda arr: arr,
        subgoal_hit,
    )
    return subgoal_coeffs, subgoal_active, subgoal_library_known, subgoal_hit


def _leading_term(poly: jnp.ndarray, env_config: EnvConfig):
    order = env_config.monomial_order_desc
    ordered = poly[order]
    mask = ordered != 0
    idx_in_order = jnp.argmax(mask.astype(jnp.int32))
    flat_idx = order[idx_in_order]
    coeff = poly[flat_idx]
    exp = env_config.monomial_exponents[flat_idx]
    return flat_idx, coeff, exp


def _monomial_term(coeff: jnp.ndarray, exp: jnp.ndarray,
                   env_config: EnvConfig) -> jnp.ndarray:
    flat_idx = jnp.sum(exp * env_config.monomial_strides)
    return jax.nn.one_hot(
        flat_idx, env_config.target_size, dtype=jnp.int32
    ) * coeff


def exact_quotient(dividend: jnp.ndarray, divisor: jnp.ndarray,
                   env_config: EnvConfig):
    """Exact quotient over F_p via dense multivariate long division."""
    divisor_zero = _poly_is_zero(divisor)

    def _do_division(_):
        _, lead_div_coeff, lead_div_exp = _leading_term(divisor, env_config)
        inv_lead = env_config.inv_mod_table[lead_div_coeff]
        zero = jnp.zeros_like(dividend)

        def cond(carry):
            rem, _quot, exact, iters = carry
            return (iters < env_config.target_size) & exact & jnp.any(rem != 0)

        def body(carry):
            rem, quot, exact, iters = carry
            _, lead_rem_coeff, lead_rem_exp = _leading_term(rem, env_config)
            divisible = jnp.all(lead_rem_exp >= lead_div_exp)
            exp_diff = lead_rem_exp - lead_div_exp
            term_coeff = (lead_rem_coeff * inv_lead) % env_config.mod
            term = _monomial_term(term_coeff, exp_diff, env_config)
            subtract_poly = poly_mul(
                term, divisor, env_config.mod,
                env_config.n_variables, env_config.max_degree,
            )
            new_rem = jnp.where(divisible, (rem - subtract_poly) % env_config.mod, rem)
            new_quot = jnp.where(divisible, (quot + term) % env_config.mod, quot)
            return new_rem, new_quot, exact & divisible, iters + 1

        rem, quot, exact, _ = jax.lax.while_loop(
            cond, body, (dividend, zero, jnp.bool_(True), jnp.int32(0))
        )
        is_exact = exact & _poly_is_zero(rem) & (~_poly_is_zero(quot))
        return quot, is_exact

    return jax.lax.cond(
        divisor_zero,
        lambda _: (jnp.zeros_like(dividend), jnp.bool_(False)),
        _do_division,
        operand=None,
    )


# ---------------------------------------------------------------------------
# Environment init / step  (pure functions)
# ---------------------------------------------------------------------------

def make_initial_node_coeffs_raw(n_variables: int, max_degree: int,
                                 mod: int) -> jnp.ndarray:
    """Build the initial node coefficient matrix for [x0, ..., x_{n-1}, 1].

    Returns:
        (n_variables + 1, target_size) array of int32 coefficients.
    """
    n_vars = n_variables
    d = max_degree
    shape = (d + 1,) * n_vars

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


def make_initial_node_coeffs(env_config: EnvConfig) -> jnp.ndarray:
    """Compatibility wrapper for tests/callers."""
    return env_config.initial_node_coeffs


def make_empty_subgoal_arrays(env_config: EnvConfig):
    """Zero-valued per-episode subgoal arrays."""
    return (
        jnp.zeros((env_config.max_subgoals, env_config.target_size), dtype=jnp.int32),
        jnp.zeros((env_config.max_subgoals,), dtype=jnp.bool_),
        jnp.zeros((env_config.max_subgoals,), dtype=jnp.bool_),
        jnp.zeros((env_config.max_subgoals,), dtype=jnp.bool_),
    )


def make_empty_on_path_arrays(env_config: EnvConfig):
    """Zero-valued cached OnPath arrays."""
    return (
        jnp.zeros(
            (env_config.on_path_max_size, env_config.target_size), dtype=jnp.int32
        ),
        jnp.zeros((env_config.on_path_max_size,), dtype=jnp.uint32),
        jnp.zeros((env_config.on_path_max_size,), dtype=jnp.int32),
        jnp.zeros((env_config.on_path_max_size,), dtype=jnp.uint32),
        jnp.zeros((env_config.on_path_max_size,), dtype=jnp.bool_),
        jnp.int32(0),
    )


def reset(env_config: EnvConfig, target_coeffs: jnp.ndarray,
          subgoal_coeffs: jnp.ndarray | None = None,
          subgoal_active: jnp.ndarray | None = None,
          subgoal_library_known: jnp.ndarray | None = None,
          on_path_coeffs: jnp.ndarray | None = None,
          on_path_hashes: jnp.ndarray | None = None,
          on_path_steps: jnp.ndarray | None = None,
          on_path_route_masks: jnp.ndarray | None = None,
          on_path_active: jnp.ndarray | None = None,
          target_board_step: jnp.ndarray | None = None) -> EnvState:
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
    init_coeffs = env_config.initial_node_coeffs  # (n_initial, target_size)

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

    if subgoal_coeffs is None:
        subgoal_coeffs, subgoal_active, subgoal_library_known, subgoal_hit = (
            make_empty_subgoal_arrays(env_config)
        )
    else:
        if subgoal_active is None or subgoal_library_known is None:
            raise ValueError("subgoal_active and subgoal_library_known must be provided")
        subgoal_hit = jnp.zeros_like(subgoal_active, dtype=jnp.bool_)

    if on_path_coeffs is None:
        (
            on_path_coeffs,
            on_path_hashes,
            on_path_steps,
            on_path_route_masks,
            on_path_active,
            target_board_step,
        ) = make_empty_on_path_arrays(env_config)
    else:
        if (
            on_path_hashes is None
            or on_path_steps is None
            or on_path_active is None
            or target_board_step is None
        ):
            raise ValueError("all on_path arrays must be provided together")
        if on_path_route_masks is None:
            on_path_route_masks = jnp.where(
                on_path_active,
                jnp.full_like(on_path_hashes, jnp.uint32(0xFFFFFFFF)),
                jnp.zeros_like(on_path_hashes, dtype=jnp.uint32),
            )
    on_path_active_route_mask = jnp.bitwise_or.reduce(
        on_path_route_masks.astype(jnp.uint32)
    )
    on_path_active_route_mask = jnp.where(
        on_path_active_route_mask == jnp.uint32(0),
        jnp.uint32(0xFFFFFFFF),
        on_path_active_route_mask,
    )

    return EnvState(
        node_coeffs=node_coeffs,
        node_features=node_features,
        edge_src=edge_src,
        edge_dst=edge_dst,
        num_nodes=jnp.int32(n_initial),
        num_edges=jnp.int32(0),
        target_coeffs=target_coeffs.astype(jnp.int32),
        subgoal_coeffs=subgoal_coeffs,
        subgoal_active=subgoal_active,
        subgoal_library_known=subgoal_library_known,
        subgoal_hit=subgoal_hit,
        additive_complete_hit=jnp.bool_(False),
        mult_complete_hit=jnp.bool_(False),
        on_path_coeffs=on_path_coeffs.astype(jnp.int32),
        on_path_hashes=on_path_hashes.astype(jnp.uint32),
        on_path_steps=on_path_steps.astype(jnp.int32),
        on_path_route_masks=on_path_route_masks.astype(jnp.uint32),
        on_path_active=on_path_active,
        on_path_hit=jnp.zeros_like(on_path_active, dtype=jnp.bool_),
        on_path_count=jnp.int32(0),
        on_path_total=jnp.sum(on_path_active.astype(jnp.int32)),
        on_path_deepest_step=jnp.int32(0),
        on_path_active_route_mask=on_path_active_route_mask,
        target_board_step=jnp.asarray(target_board_step, dtype=jnp.int32),
        steps_taken=jnp.int32(0),
        done=jnp.bool_(False),
        is_success=jnp.bool_(False),
    )


def step(env_config: EnvConfig, state: EnvState,
         action: jnp.ndarray,
         library_coeffs: jnp.ndarray | None = None,
         library_mask: jnp.ndarray | None = None) -> tuple:
    """Take one step in the environment. Pure function.

    Args:
        env_config: Static config.
        state: Current EnvState.
        action: Scalar int32 action index.

    Returns:
        (next_state, reward, done, is_success, factor_hit, library_hit,
         additive_complete, mult_complete)
    """
    mod = env_config.mod
    max_nodes = env_config.max_nodes

    op, i, j = decode_action(action, max_nodes)

    poly_i = state.node_coeffs[i]
    poly_j = state.node_coeffs[j]

    # Snapshot shaping potential before adding the new node.
    if env_config.reward_mode == "legacy" and env_config.use_reward_shaping:
        phi_before = _best_similarity(
            state.node_coeffs, state.num_nodes,
            state.target_coeffs, max_nodes,
        )
    elif env_config.reward_mode == "clean_onpath":
        phi_before = _on_path_phi(state, env_config)

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

    # Base reward.
    if env_config.reward_mode == "legacy" and env_config.use_reward_shaping:
        phi_after = _best_similarity(
            node_coeffs, num_nodes, state.target_coeffs, max_nodes,
        )
        shaping = env_config.gamma * phi_after - phi_before
        reward = env_config.step_penalty + jnp.where(
            is_success, env_config.success_reward, shaping
        )
    elif env_config.reward_mode == "legacy":
        reward = env_config.step_penalty + jnp.where(
            is_success, env_config.success_reward, 0.0
        )
    else:
        reward = env_config.step_penalty + jnp.where(
            is_success, env_config.terminal_success_reward, 0.0
        )

    factor_hit = jnp.bool_(False)
    library_hit = jnp.bool_(False)
    additive_complete = jnp.bool_(False)
    mult_complete = jnp.bool_(False)
    on_path_hit_flag = jnp.bool_(False)
    on_path_phi = jnp.float32(0.0)

    subgoal_coeffs = state.subgoal_coeffs
    subgoal_active = state.subgoal_active
    subgoal_library_known = state.subgoal_library_known
    subgoal_hit = state.subgoal_hit
    additive_complete_hit = state.additive_complete_hit
    mult_complete_hit = state.mult_complete_hit

    on_path_hit = state.on_path_hit
    on_path_count = state.on_path_count
    on_path_deepest_step = state.on_path_deepest_step
    on_path_active_route_mask = state.on_path_active_route_mask

    if env_config.reward_mode == "clean_onpath":
        match_idx, has_on_path_match = _on_path_match_index(
            new_coeffs, state, env_config
        )
        hit_step = jnp.where(has_on_path_match, state.on_path_steps[match_idx], 0)
        hit_route_mask = jnp.where(
            has_on_path_match,
            state.on_path_route_masks[match_idx],
            state.on_path_active_route_mask,
        )
        on_path_hit = jax.lax.cond(
            has_on_path_match,
            lambda arr: arr.at[match_idx].set(True),
            lambda arr: arr,
            on_path_hit,
        )
        if env_config.on_path_route_consistency_mode == "lock_on_first_hit":
            on_path_active_route_mask = on_path_active_route_mask & hit_route_mask
        on_path_count = on_path_count + has_on_path_match.astype(jnp.int32)
        on_path_deepest_step = jnp.maximum(on_path_deepest_step, hit_step)
        phi_state = state._replace(
            on_path_hit=on_path_hit,
            on_path_count=on_path_count,
            on_path_deepest_step=on_path_deepest_step,
        )
        phi_after = _on_path_phi(phi_state, env_config)
        if env_config.on_path_terminal_zero:
            phi_after_for_reward = jnp.where(done, 0.0, phi_after)
        else:
            phi_after_for_reward = phi_after
        reward = reward + env_config.graph_onpath_shaping_coeff * (
            env_config.gamma * phi_after_for_reward - phi_before
        )
        on_path_hit_flag = has_on_path_match
        on_path_phi = phi_after

    if env_config.factor_library_enabled:
        match_idx, has_subgoal_match = _poly_match_index(
            new_coeffs, subgoal_coeffs, subgoal_active & (~subgoal_hit)
        )
        was_library_known = jnp.where(
            has_subgoal_match, subgoal_library_known[match_idx], False
        )
        reward = reward + jnp.where(
            has_subgoal_match, env_config.factor_subgoal_reward, 0.0
        )
        reward = reward + jnp.where(
            has_subgoal_match & was_library_known,
            env_config.factor_library_bonus,
            0.0,
        )
        subgoal_hit = jax.lax.cond(
            has_subgoal_match,
            lambda arr: arr.at[match_idx].set(True),
            lambda arr: arr,
            subgoal_hit,
        )
        factor_hit = has_subgoal_match
        library_hit = has_subgoal_match & was_library_known

        residual = (state.target_coeffs - new_coeffs) % env_config.mod
        residual_nonzero = ~_poly_is_zero(residual)

        can_additive_complete = (
            (~is_success)
            & (~additive_complete_hit)
            & residual_nonzero
            & _existing_node_contains(residual, state.node_coeffs, state.num_nodes)
        )
        reward = reward + jnp.where(
            can_additive_complete, env_config.completion_bonus, 0.0
        )
        additive_complete = can_additive_complete
        additive_complete_hit = additive_complete_hit | can_additive_complete

        new_is_library_known = _library_contains(
            new_coeffs, library_coeffs, library_mask
        )

        def _dynamic_discovery(args):
            sg_coeffs, sg_active, sg_known, sg_hit, reward_val, mult_hit = args

            residual_library_known = _library_contains(
                residual, library_coeffs, library_mask
            )
            valid_residual_subgoal = residual_nonzero & (~_is_base_poly(residual, env_config))
            sg_coeffs, sg_active, sg_known, sg_hit = _add_subgoal_if_new(
                sg_coeffs, sg_active, sg_known, sg_hit,
                residual, residual_library_known, valid_residual_subgoal,
            )

            quotient, quotient_exact = exact_quotient(
                state.target_coeffs, new_coeffs, env_config
            )
            quotient_in_existing = _existing_node_contains(
                quotient, state.node_coeffs, state.num_nodes
            )
            can_mult_complete = (
                (~mult_hit) & quotient_exact & quotient_in_existing
            )
            reward_val = reward_val + jnp.where(
                can_mult_complete, env_config.completion_bonus, 0.0
            )
            quotient_library_known = _library_contains(
                quotient, library_coeffs, library_mask
            )
            valid_quotient_subgoal = (
                quotient_exact
                & (~_poly_is_scalar(quotient))
                & (~_is_base_poly(quotient, env_config))
            )
            sg_coeffs, sg_active, sg_known, sg_hit = _add_subgoal_if_new(
                sg_coeffs, sg_active, sg_known, sg_hit,
                quotient, quotient_library_known, valid_quotient_subgoal,
            )
            return (
                sg_coeffs,
                sg_active,
                sg_known,
                sg_hit,
                reward_val,
                mult_hit | can_mult_complete,
                can_mult_complete,
            )

        (
            subgoal_coeffs,
            subgoal_active,
            subgoal_library_known,
            subgoal_hit,
            reward,
            mult_complete_hit,
            mult_complete,
        ) = jax.lax.cond(
            (~is_success) & new_is_library_known,
            _dynamic_discovery,
            lambda args: (*args[:5], args[5], jnp.bool_(False)),
            (
                subgoal_coeffs,
                subgoal_active,
                subgoal_library_known,
                subgoal_hit,
                reward,
                mult_complete_hit,
            ),
        )

    next_state = EnvState(
        node_coeffs=node_coeffs,
        node_features=node_features,
        edge_src=edge_src,
        edge_dst=edge_dst,
        num_nodes=num_nodes.astype(jnp.int32),
        num_edges=num_edges.astype(jnp.int32),
        target_coeffs=state.target_coeffs,
        subgoal_coeffs=subgoal_coeffs,
        subgoal_active=subgoal_active,
        subgoal_library_known=subgoal_library_known,
        subgoal_hit=subgoal_hit,
        additive_complete_hit=additive_complete_hit,
        mult_complete_hit=mult_complete_hit,
        on_path_coeffs=state.on_path_coeffs,
        on_path_hashes=state.on_path_hashes,
        on_path_steps=state.on_path_steps,
        on_path_route_masks=state.on_path_route_masks,
        on_path_active=state.on_path_active,
        on_path_hit=on_path_hit,
        on_path_count=on_path_count.astype(jnp.int32),
        on_path_total=state.on_path_total,
        on_path_deepest_step=on_path_deepest_step.astype(jnp.int32),
        on_path_active_route_mask=on_path_active_route_mask.astype(jnp.uint32),
        target_board_step=state.target_board_step,
        steps_taken=steps_taken.astype(jnp.int32),
        done=done,
        is_success=is_success,
    )
    return (
        next_state,
        reward.astype(jnp.float32),
        done,
        is_success,
        factor_hit,
        library_hit,
        additive_complete,
        mult_complete,
        on_path_hit_flag,
        on_path_phi,
    )


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
