"""Tests for JAX environment, network, and MCTS compilation.

Verifies that jax_env, jax_net, and ppo_mcts_jax compile and produce
correct shapes / values without requiring a GPU.
"""

import pytest
import numpy as np

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
flax = pytest.importorskip("flax")
optax = pytest.importorskip("optax")
mctx = pytest.importorskip("mctx")

from src.config import Config
from src.algorithms.jax_env import (
    EnvConfig, EnvState, make_env_config,
    reset, step, get_observation, get_valid_actions_mask,
    encode_action, decode_action,
    poly_add, poly_mul, make_initial_node_coeffs,
    _on_path_phi,
)
from src.algorithms.jax_net import (
    PolicyValueNet, create_network, init_params, GraphEncoder,
)
from src.game_board.on_path import hash_coeff_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Small config for fast tests."""
    return Config(
        n_variables=2,
        mod=5,
        max_complexity=4,
        max_steps=6,
        max_degree=-1,  # auto = max_complexity = 4
        hidden_dim=32,
        embedding_dim=32,
        num_gnn_layers=2,
        seed=0,
    )


@pytest.fixture
def env_config(config):
    return make_env_config(config)


@pytest.fixture
def initial_state(env_config):
    """Reset env with target = x0 + x1 (coeffs [0,1,1,0,...])."""
    target_size = env_config.target_size
    target = jnp.zeros(target_size, dtype=jnp.int32)
    # For 2 vars, degree 4: shape is (5,5). x0 -> index (1,0), x1 -> index (0,1)
    shape = (env_config.max_degree + 1,) * env_config.n_variables
    t_nd = np.zeros(shape, dtype=np.int32)
    t_nd[1, 0] = 1  # x0
    t_nd[0, 1] = 1  # x1
    target = jnp.array(t_nd.flatten(), dtype=jnp.int32)
    return reset(env_config, target)


# ---------------------------------------------------------------------------
# jax_env: action encoding / decoding
# ---------------------------------------------------------------------------

class TestActionEncoding:
    def test_roundtrip(self, env_config):
        """encode(decode(a)) == a for all valid action indices."""
        max_nodes = env_config.max_nodes
        max_actions = env_config.max_actions
        for a in range(max_actions):
            op, i, j = decode_action(jnp.int32(a), max_nodes)
            a2 = encode_action(op, i, j, max_nodes)
            assert int(a2) == a, f"Roundtrip failed for action {a}"

    def test_decode_values(self, env_config):
        """Spot-check: action 0 -> op=0, i=0, j=0; action 1 -> op=1, i=0, j=0."""
        mn = env_config.max_nodes
        op0, i0, j0 = decode_action(jnp.int32(0), mn)
        assert (int(op0), int(i0), int(j0)) == (0, 0, 0)
        op1, i1, j1 = decode_action(jnp.int32(1), mn)
        assert (int(op1), int(i1), int(j1)) == (1, 0, 0)

    def test_jit_compatible(self, env_config):
        """decode_action should work inside jax.jit."""
        mn = env_config.max_nodes
        jit_decode = jax.jit(lambda a: decode_action(a, mn))
        op, i, j = jit_decode(jnp.int32(5))
        assert op.shape == ()
        assert i.shape == ()
        assert j.shape == ()


# ---------------------------------------------------------------------------
# jax_env: polynomial arithmetic
# ---------------------------------------------------------------------------

class TestPolyArithmetic:
    def test_poly_add_basic(self):
        """(1 + 0) mod 5 = 1; (3 + 4) mod 5 = 2."""
        a = jnp.array([1, 3, 0], dtype=jnp.int32)
        b = jnp.array([0, 4, 2], dtype=jnp.int32)
        result = poly_add(a, b, mod=5)
        np.testing.assert_array_equal(result, [1, 2, 2])

    def test_poly_mul_1d(self):
        """(1 + x) * (1 + x) = 1 + 2x + x^2, mod 5."""
        # 1D: n_variables=1, max_degree=4. Coeffs: [1, 1, 0, 0, 0]
        a = jnp.array([1, 1, 0, 0, 0], dtype=jnp.int32)
        b = jnp.array([1, 1, 0, 0, 0], dtype=jnp.int32)
        result = poly_mul(a, b, mod=5, n_variables=1, max_degree=4)
        # Expected: 1 + 2x + x^2 = [1, 2, 1, 0, 0]
        np.testing.assert_array_equal(result, [1, 2, 1, 0, 0])

    def test_poly_mul_2d(self):
        """x0 * x1 in 2 variables, mod 5. Result should have 1 at (1,1)."""
        # 2 variables, max_degree=4. Shape (5,5), flattened size 25.
        d = 5
        a = np.zeros((d, d), dtype=np.int32)
        a[1, 0] = 1  # x0
        b = np.zeros((d, d), dtype=np.int32)
        b[0, 1] = 1  # x1
        result = poly_mul(
            jnp.array(a.flatten()), jnp.array(b.flatten()),
            mod=5, n_variables=2, max_degree=4,
        )
        expected = np.zeros((d, d), dtype=np.int32)
        expected[1, 1] = 1  # x0 * x1
        np.testing.assert_array_equal(result, expected.flatten())

    def test_poly_mul_3d(self):
        """x0 * x1 and x1 * x2 in 3 variables, mod 5."""
        # 3 variables, max_degree=4. Shape (5,5,5), flattened size 125.
        d = 5
        # x0 * x1: coefficient at (1,0,0) * (0,1,0) -> (1,1,0)
        a = np.zeros((d, d, d), dtype=np.int32)
        a[1, 0, 0] = 1  # x0
        b = np.zeros((d, d, d), dtype=np.int32)
        b[0, 1, 0] = 1  # x1
        result = poly_mul(
            jnp.array(a.flatten()), jnp.array(b.flatten()),
            mod=5, n_variables=3, max_degree=4,
        )
        expected = np.zeros((d, d, d), dtype=np.int32)
        expected[1, 1, 0] = 1  # x0 * x1
        np.testing.assert_array_equal(result, expected.flatten())

        # x1 * x2: coefficient at (0,1,0) * (0,0,1) -> (0,1,1)
        c = np.zeros((d, d, d), dtype=np.int32)
        c[0, 1, 0] = 1  # x1
        e = np.zeros((d, d, d), dtype=np.int32)
        e[0, 0, 1] = 1  # x2
        result2 = poly_mul(
            jnp.array(c.flatten()), jnp.array(e.flatten()),
            mod=5, n_variables=3, max_degree=4,
        )
        expected2 = np.zeros((d, d, d), dtype=np.int32)
        expected2[0, 1, 1] = 1  # x1 * x2
        np.testing.assert_array_equal(result2, expected2.flatten())

    def test_poly_mul_3d_higher_degree(self):
        """(x0 + x1) * (x0 + x2) in 3 variables, mod 5."""
        d = 5
        a = np.zeros((d, d, d), dtype=np.int32)
        a[1, 0, 0] = 1  # x0
        a[0, 1, 0] = 1  # x1
        b = np.zeros((d, d, d), dtype=np.int32)
        b[1, 0, 0] = 1  # x0
        b[0, 0, 1] = 1  # x2
        result = poly_mul(
            jnp.array(a.flatten()), jnp.array(b.flatten()),
            mod=5, n_variables=3, max_degree=4,
        )
        # (x0+x1)(x0+x2) = x0^2 + x0*x2 + x0*x1 + x1*x2
        expected = np.zeros((d, d, d), dtype=np.int32)
        expected[2, 0, 0] = 1  # x0^2
        expected[1, 0, 1] = 1  # x0*x2
        expected[1, 1, 0] = 1  # x0*x1
        expected[0, 1, 1] = 1  # x1*x2
        np.testing.assert_array_equal(result, expected.flatten())

    def test_poly_mul_mod(self):
        """Multiplication result is reduced mod p."""
        # (3) * (2) = 6 mod 5 = 1 in 1 variable, degree 2.
        a = jnp.array([3, 0, 0], dtype=jnp.int32)
        b = jnp.array([2, 0, 0], dtype=jnp.int32)
        result = poly_mul(a, b, mod=5, n_variables=1, max_degree=2)
        assert int(result[0]) == 1  # 6 mod 5


# ---------------------------------------------------------------------------
# jax_env: environment reset / step
# ---------------------------------------------------------------------------

class TestEnvResetStep:
    def test_reset_shapes(self, env_config, initial_state):
        """Check that reset returns correctly shaped arrays."""
        s = initial_state
        max_nodes = env_config.max_nodes
        max_edges = max_nodes * 4
        target_size = env_config.target_size

        assert s.node_coeffs.shape == (max_nodes, target_size)
        assert s.node_features.shape == (max_nodes, 4)
        assert s.edge_src.shape == (max_edges,)
        assert s.edge_dst.shape == (max_edges,)
        assert int(s.num_nodes) == env_config.n_variables + 1  # x0, x1, 1
        assert int(s.num_edges) == 0
        assert int(s.steps_taken) == 0
        assert not bool(s.done)

    def test_initial_node_coeffs(self, env_config):
        """Initial nodes are x0, x1, and constant 1."""
        coeffs = make_initial_node_coeffs(env_config)
        d = env_config.max_degree + 1
        # x0: coeff at (1,0) = 1
        x0 = coeffs[0].reshape(d, d)
        assert int(x0[1, 0]) == 1
        assert int(x0.sum()) == 1
        # x1: coeff at (0,1) = 1
        x1 = coeffs[1].reshape(d, d)
        assert int(x1[0, 1]) == 1
        assert int(x1.sum()) == 1
        # const 1: coeff at (0,0) = 1
        c = coeffs[2].reshape(d, d)
        assert int(c[0, 0]) == 1
        assert int(c.sum()) == 1

    def test_step_add(self, env_config, initial_state):
        """Adding x0 + x1 should produce x0+x1 node and increment counts."""
        # action: add node 0 (x0) and node 1 (x1) -> op=0, i=0, j=1
        action = encode_action(jnp.int32(0), jnp.int32(0), jnp.int32(1),
                               env_config.max_nodes)
        next_state, reward, done, is_success, *_ = step(env_config, initial_state, action)

        assert int(next_state.num_nodes) == int(initial_state.num_nodes) + 1
        assert int(next_state.num_edges) == 4  # bidirectional to both operands
        assert int(next_state.steps_taken) == 1

    def test_step_success(self, env_config, initial_state):
        """Adding x0 + x1 when target is x0+x1 should yield success."""
        action = encode_action(jnp.int32(0), jnp.int32(0), jnp.int32(1),
                               env_config.max_nodes)
        next_state, reward, done, is_success, *_ = step(env_config, initial_state, action)

        assert bool(is_success)
        assert bool(done)
        assert float(reward) > 0  # success_reward + step_penalty > 0

    def test_step_jit(self, env_config, initial_state):
        """env step is JIT-compilable."""
        import functools
        jit_step = jax.jit(functools.partial(step, env_config))
        action = encode_action(jnp.int32(0), jnp.int32(0), jnp.int32(1),
                               env_config.max_nodes)
        next_state, reward, done, is_success, *_ = jit_step(initial_state, action)
        assert next_state.num_nodes.shape == ()

    def test_fl_additive_completion(self, config):
        """JAX env awards additive completion when one add-away from target."""
        ec = make_env_config(config)
        d = ec.max_degree + 1
        target_nd = np.zeros((d, d), dtype=np.int32)
        target_nd[1, 0] = 1  # x0
        target_nd[0, 1] = 1  # x1
        target_nd[0, 0] = 1  # 1
        state = reset(ec, jnp.array(target_nd.flatten(), dtype=jnp.int32))

        action = encode_action(jnp.int32(0), jnp.int32(0), jnp.int32(1), ec.max_nodes)
        next_state, reward, done, is_success, factor_hit, library_hit, add_complete, mult_complete, *_ = step(
            ec, state, action
        )

        assert bool(add_complete)
        assert not bool(mult_complete)
        assert float(reward) >= ec.completion_bonus + ec.step_penalty

    def test_fl_initial_subgoal_reward(self, config):
        """A preloaded initial subgoal should trigger factor and library bonus rewards."""
        ec = make_env_config(config)
        d = ec.max_degree + 1

        target_nd = np.zeros((d, d), dtype=np.int32)
        target_nd[2, 0] = 1  # x0^2
        target_nd[1, 0] = 2  # 2*x0
        target_nd[0, 0] = 1  # 1

        subgoal_nd = np.zeros((d, d), dtype=np.int32)
        subgoal_nd[1, 0] = 1  # x0
        subgoal_nd[0, 0] = 1  # 1

        max_subgoals = ec.max_subgoals
        subgoal_coeffs = np.zeros((max_subgoals, ec.target_size), dtype=np.int32)
        subgoal_active = np.zeros((max_subgoals,), dtype=bool)
        subgoal_library_known = np.zeros((max_subgoals,), dtype=bool)
        subgoal_coeffs[0] = subgoal_nd.flatten()
        subgoal_active[0] = True
        subgoal_library_known[0] = True

        state = reset(
            ec,
            jnp.array(target_nd.flatten(), dtype=jnp.int32),
            jnp.array(subgoal_coeffs, dtype=jnp.int32),
            jnp.array(subgoal_active),
            jnp.array(subgoal_library_known),
        )

        action = encode_action(jnp.int32(0), jnp.int32(0), jnp.int32(2), ec.max_nodes)
        _next_state, reward, _done, _is_success, factor_hit, library_hit, _add_complete, _mult_complete, *_ = step(
            ec, state, action
        )

        assert bool(factor_hit)
        assert bool(library_hit)
        assert float(reward) >= (
            ec.step_penalty + ec.factor_subgoal_reward + ec.factor_library_bonus
        )

    def test_clean_onpath_reward_jit(self):
        """Clean on-path mode counts verified coefficient hits once."""
        cfg = Config(
            n_variables=2,
            mod=5,
            max_complexity=4,
            max_steps=6,
            reward_mode="clean_onpath",
            on_path_max_size=4,
            on_path_phi_mode="count",
            on_path_route_consistency_mode="lock_on_first_hit",
            graph_onpath_shaping_coeff=1.0,
        )
        ec = make_env_config(cfg)
        d = ec.max_degree + 1

        target_nd = np.zeros((d, d), dtype=np.int32)
        target_nd[1, 0] = 1
        target_nd[0, 1] = 2
        target = target_nd.flatten()

        inter_nd = np.zeros((d, d), dtype=np.int32)
        inter_nd[1, 0] = 1
        inter_nd[0, 1] = 1

        on_path = np.zeros((ec.on_path_max_size, ec.target_size), dtype=np.int32)
        on_path[0] = inter_nd.flatten()
        on_path[1] = target
        hashes = np.zeros((ec.on_path_max_size,), dtype=np.uint32)
        hashes[:2] = hash_coeff_matrix(on_path[:2])
        steps = np.array([1, 2, 0, 0], dtype=np.int32)
        active = np.array([True, True, False, False])

        state = reset(
            ec,
            jnp.array(target, dtype=jnp.int32),
            on_path_coeffs=jnp.array(on_path, dtype=jnp.int32),
            on_path_hashes=jnp.array(hashes, dtype=jnp.uint32),
            on_path_steps=jnp.array(steps, dtype=jnp.int32),
            on_path_active=jnp.array(active),
            target_board_step=jnp.int32(2),
        )
        action = encode_action(jnp.int32(0), jnp.int32(0), jnp.int32(1), ec.max_nodes)

        import functools
        jit_step = jax.jit(functools.partial(step, ec))
        next_state, reward, _done, _success, *_rest, on_path_hit, on_path_phi = (
            jit_step(state, action)
        )

        assert bool(on_path_hit)
        assert float(on_path_phi) == pytest.approx(0.5)
        assert int(next_state.on_path_count) == 1
        assert float(reward) == pytest.approx(ec.step_penalty + ec.gamma * 0.5)

    def test_clean_onpath_route_consistency_blocks_incompatible_jax_hits(self):
        cfg = Config(
            n_variables=2,
            mod=5,
            max_complexity=4,
            max_steps=6,
            reward_mode="clean_onpath",
            on_path_max_size=4,
            on_path_phi_mode="count",
            graph_onpath_shaping_coeff=1.0,
        )
        ec = make_env_config(cfg)
        d = ec.max_degree + 1

        route_a_nd = np.zeros((d, d), dtype=np.int32)
        route_a_nd[1, 0] = 1
        route_a_nd[0, 1] = 1
        route_b_nd = np.zeros((d, d), dtype=np.int32)
        route_b_nd[1, 1] = 1
        target_nd = (route_a_nd + route_b_nd) % cfg.mod

        on_path = np.zeros((ec.on_path_max_size, ec.target_size), dtype=np.int32)
        on_path[0] = route_a_nd.flatten()
        on_path[1] = route_b_nd.flatten()
        on_path[2] = target_nd.flatten()
        hashes = np.zeros((ec.on_path_max_size,), dtype=np.uint32)
        hashes[:3] = hash_coeff_matrix(on_path[:3])
        steps = np.array([1, 1, 2, 0], dtype=np.int32)
        route_masks = np.array([0b01, 0b10, 0b11, 0], dtype=np.uint32)
        active = np.array([True, True, True, False])

        state = reset(
            ec,
            jnp.array(target_nd.flatten(), dtype=jnp.int32),
            on_path_coeffs=jnp.array(on_path, dtype=jnp.int32),
            on_path_hashes=jnp.array(hashes, dtype=jnp.uint32),
            on_path_steps=jnp.array(steps, dtype=jnp.int32),
            on_path_route_masks=jnp.array(route_masks, dtype=jnp.uint32),
            on_path_active=jnp.array(active),
            target_board_step=jnp.int32(2),
        )

        import functools
        jit_step = jax.jit(functools.partial(step, ec))
        add_action = encode_action(jnp.int32(0), jnp.int32(0), jnp.int32(1), ec.max_nodes)
        mul_action = encode_action(jnp.int32(1), jnp.int32(0), jnp.int32(1), ec.max_nodes)

        next_state, _reward, _done, _success, *_rest, on_path_hit, on_path_phi = (
            jit_step(state, add_action)
        )
        assert bool(on_path_hit)
        assert float(on_path_phi) == pytest.approx(1 / 3)
        assert int(next_state.on_path_count) == 1

        next_state, reward2, _done, _success, *_rest, on_path_hit2, on_path_phi2 = (
            jit_step(next_state, mul_action)
        )
        assert not bool(on_path_hit2)
        assert float(on_path_phi2) == pytest.approx(1 / 3)
        assert int(next_state.on_path_count) == 1
        assert float(reward2) == pytest.approx(
            ec.step_penalty + ec.gamma * (1 / 3) - (1 / 3)
        )

    def test_clean_onpath_best_route_phi_allows_switch_without_frankenstein_jax(self):
        cfg = Config(
            n_variables=2,
            mod=5,
            max_complexity=4,
            max_steps=6,
            reward_mode="clean_onpath",
            on_path_max_size=4,
            on_path_phi_mode="count",
            on_path_route_consistency_mode="best_route_phi",
            graph_onpath_shaping_coeff=1.0,
        )
        ec = make_env_config(cfg)
        d = ec.max_degree + 1

        a1_nd = np.zeros((d, d), dtype=np.int32)
        a1_nd[1, 0] = 1
        a1_nd[0, 1] = 1
        b1_nd = np.zeros((d, d), dtype=np.int32)
        b1_nd[1, 1] = 1
        b2_nd = b1_nd.copy()
        b2_nd[0, 1] = 1
        target_nd = b2_nd.copy()
        target_nd[0, 1] = (target_nd[0, 1] + 1) % cfg.mod

        on_path = np.zeros((ec.on_path_max_size, ec.target_size), dtype=np.int32)
        on_path[0] = a1_nd.flatten()
        on_path[1] = b1_nd.flatten()
        on_path[2] = b2_nd.flatten()
        on_path[3] = target_nd.flatten()
        hashes = hash_coeff_matrix(on_path)
        steps = np.array([1, 1, 2, 3], dtype=np.int32)
        route_masks = np.array([0b01, 0b10, 0b10, 0b11], dtype=np.uint32)
        active = np.array([True, True, True, True])

        state = reset(
            ec,
            jnp.array(target_nd.flatten(), dtype=jnp.int32),
            on_path_coeffs=jnp.array(on_path, dtype=jnp.int32),
            on_path_hashes=jnp.array(hashes, dtype=jnp.uint32),
            on_path_steps=jnp.array(steps, dtype=jnp.int32),
            on_path_route_masks=jnp.array(route_masks, dtype=jnp.uint32),
            on_path_active=jnp.array(active),
            target_board_step=jnp.int32(3),
        )

        import functools
        jit_step = jax.jit(functools.partial(step, ec))
        a1_action = encode_action(jnp.int32(0), jnp.int32(0), jnp.int32(1), ec.max_nodes)
        b1_action = encode_action(jnp.int32(1), jnp.int32(0), jnp.int32(1), ec.max_nodes)
        b2_action = encode_action(jnp.int32(0), jnp.int32(4), jnp.int32(1), ec.max_nodes)

        state, _reward1, _done, _success, *_rest, hit1, phi1 = jit_step(
            state, a1_action
        )
        assert bool(hit1)
        assert float(phi1) == pytest.approx(1 / 2)

        state, _reward2, _done, _success, *_rest, hit2, phi2 = jit_step(
            state, b1_action
        )
        assert bool(hit2)
        assert int(state.on_path_count) == 2
        assert float(phi2) == pytest.approx(1 / 2)

        state, _reward3, _done, _success, *_rest, hit3, phi3 = jit_step(
            state, b2_action
        )
        assert bool(hit3)
        assert int(state.on_path_count) == 3
        assert float(phi3) == pytest.approx(2 / 3)

    def test_clean_onpath_depth_weighted_phi_values_jax(self):
        cfg = Config(
            n_variables=2,
            mod=5,
            max_complexity=4,
            max_steps=6,
            reward_mode="clean_onpath",
            on_path_max_size=3,
            on_path_phi_mode="depth_weighted",
            on_path_depth_weight_power=2.0,
            on_path_route_consistency_mode="best_route_phi",
        )
        ec = make_env_config(cfg)
        on_path = np.zeros((ec.on_path_max_size, ec.target_size), dtype=np.int32)
        hashes = np.zeros((ec.on_path_max_size,), dtype=np.uint32)
        steps = np.array([1, 2, 0], dtype=np.int32)
        route_masks = np.array([0b01, 0b01, 0], dtype=np.uint32)
        active = np.array([True, True, False])
        target = jnp.zeros(ec.target_size, dtype=jnp.int32)

        state = reset(
            ec,
            target,
            on_path_coeffs=jnp.array(on_path, dtype=jnp.int32),
            on_path_hashes=jnp.array(hashes, dtype=jnp.uint32),
            on_path_steps=jnp.array(steps, dtype=jnp.int32),
            on_path_route_masks=jnp.array(route_masks, dtype=jnp.uint32),
            on_path_active=jnp.array(active),
            target_board_step=jnp.int32(2),
        )

        shallow = state._replace(
            on_path_hit=jnp.array([True, False, False])
        )
        deep = state._replace(
            on_path_hit=jnp.array([False, True, False])
        )
        both = state._replace(
            on_path_hit=jnp.array([True, True, False])
        )

        assert float(_on_path_phi(shallow, ec)) == pytest.approx(1 / 5)
        assert float(_on_path_phi(deep, ec)) == pytest.approx(4 / 5)
        assert float(_on_path_phi(both, ec)) == pytest.approx(1.0)

    def test_clean_onpath_depth_weighted_prevents_frankenstein_jax(self):
        cfg = Config(
            n_variables=2,
            mod=5,
            max_complexity=4,
            max_steps=6,
            reward_mode="clean_onpath",
            on_path_max_size=4,
            on_path_phi_mode="depth_weighted",
            on_path_depth_weight_power=1.0,
            on_path_route_consistency_mode="best_route_phi",
        )
        ec = make_env_config(cfg)
        on_path = np.zeros((ec.on_path_max_size, ec.target_size), dtype=np.int32)
        hashes = np.zeros((ec.on_path_max_size,), dtype=np.uint32)
        steps = np.array([1, 1, 2, 3], dtype=np.int32)
        route_masks = np.array([0b01, 0b10, 0b10, 0b11], dtype=np.uint32)
        active = np.array([True, True, True, True])
        target = jnp.zeros(ec.target_size, dtype=jnp.int32)

        state = reset(
            ec,
            target,
            on_path_coeffs=jnp.array(on_path, dtype=jnp.int32),
            on_path_hashes=jnp.array(hashes, dtype=jnp.uint32),
            on_path_steps=jnp.array(steps, dtype=jnp.int32),
            on_path_route_masks=jnp.array(route_masks, dtype=jnp.uint32),
            on_path_active=jnp.array(active),
            target_board_step=jnp.int32(3),
        )
        mixed = state._replace(
            on_path_hit=jnp.array([True, True, False, False])
        )
        route_b = state._replace(
            on_path_hit=jnp.array([True, True, True, False])
        )

        assert float(_on_path_phi(mixed, ec)) == pytest.approx(1 / 4)
        assert float(_on_path_phi(route_b, ec)) == pytest.approx(3 / 6)

    def test_clean_onpath_terminal_zero_reward_jit(self):
        """Clean on-path logs terminal phi but zeroes reward-side terminal phi."""
        cfg = Config(
            n_variables=2,
            mod=5,
            max_complexity=4,
            max_steps=6,
            reward_mode="clean_onpath",
            on_path_max_size=4,
            on_path_phi_mode="count",
            graph_onpath_shaping_coeff=1.0,
        )
        ec = make_env_config(cfg)
        d = ec.max_degree + 1

        target_nd = np.zeros((d, d), dtype=np.int32)
        target_nd[1, 0] = 1
        target_nd[0, 1] = 1
        target = target_nd.flatten()

        on_path = np.zeros((ec.on_path_max_size, ec.target_size), dtype=np.int32)
        on_path[0] = target
        hashes = np.zeros((ec.on_path_max_size,), dtype=np.uint32)
        hashes[:1] = hash_coeff_matrix(on_path[:1])
        steps = np.array([1, 0, 0, 0], dtype=np.int32)
        active = np.array([True, False, False, False])

        state = reset(
            ec,
            jnp.array(target, dtype=jnp.int32),
            on_path_coeffs=jnp.array(on_path, dtype=jnp.int32),
            on_path_hashes=jnp.array(hashes, dtype=jnp.uint32),
            on_path_steps=jnp.array(steps, dtype=jnp.int32),
            on_path_active=jnp.array(active),
            target_board_step=jnp.int32(1),
        )
        action = encode_action(jnp.int32(0), jnp.int32(0), jnp.int32(1), ec.max_nodes)

        import functools
        jit_step = jax.jit(functools.partial(step, ec))
        next_state, reward, done, success, *_rest, on_path_hit, on_path_phi = (
            jit_step(state, action)
        )

        assert bool(done)
        assert bool(success)
        assert bool(on_path_hit)
        assert float(on_path_phi) == pytest.approx(1.0)
        assert int(next_state.on_path_count) == 1
        assert float(reward) == pytest.approx(
            ec.step_penalty + ec.terminal_success_reward
        )

    def test_clean_onpath_phi_zero_denominator_guards(self):
        """Degenerate packed contexts should produce zero phi, not divide by one."""
        cfg = Config(
            n_variables=2,
            mod=5,
            max_complexity=4,
            max_steps=6,
            reward_mode="clean_onpath",
            on_path_max_size=4,
            on_path_phi_mode="count",
            graph_onpath_shaping_coeff=1.0,
        )
        ec = make_env_config(cfg)
        d = ec.max_degree + 1
        target_nd = np.zeros((d, d), dtype=np.int32)
        target_nd[2, 0] = 1
        target = target_nd.flatten()
        on_path = np.zeros((ec.on_path_max_size, ec.target_size), dtype=np.int32)
        hashes = np.zeros((ec.on_path_max_size,), dtype=np.uint32)
        steps = np.zeros((ec.on_path_max_size,), dtype=np.int32)
        active = np.zeros((4,), dtype=bool)

        state = reset(
            ec,
            jnp.array(target, dtype=jnp.int32),
            on_path_coeffs=jnp.array(on_path, dtype=jnp.int32),
            on_path_hashes=jnp.array(hashes, dtype=jnp.uint32),
            on_path_steps=jnp.array(steps, dtype=jnp.int32),
            on_path_active=jnp.array(active),
            target_board_step=jnp.int32(0),
        )
        assert float(_on_path_phi(state, ec)) == pytest.approx(0.0)

        max_step_cfg = Config(**{**cfg.__dict__, "on_path_phi_mode": "max_step"})
        max_step_ec = make_env_config(max_step_cfg)
        max_step_state = state._replace(
            on_path_deepest_step=jnp.int32(1),
            target_board_step=jnp.int32(0),
        )
        assert float(_on_path_phi(max_step_state, max_step_ec)) == pytest.approx(0.0)

        depth_cfg = Config(**{**cfg.__dict__, "on_path_phi_mode": "depth_weighted"})
        depth_ec = make_env_config(depth_cfg)
        assert float(_on_path_phi(state, depth_ec)) == pytest.approx(0.0)

    def test_get_observation(self, env_config, initial_state):
        """Observation dict has expected keys and shapes."""
        obs = get_observation(env_config, initial_state)
        max_nodes = env_config.max_nodes
        max_edges = max_nodes * 4

        assert obs['node_features'].shape == (max_nodes, 4)
        assert obs['edge_src'].shape == (max_edges,)
        assert obs['edge_dst'].shape == (max_edges,)
        assert obs['num_nodes'].shape == ()
        assert obs['num_edges'].shape == ()
        assert obs['target'].shape == (env_config.target_size,)
        assert obs['mask'].shape == (env_config.max_actions,)
        # Target should be normalized to [0, 1]
        assert float(obs['target'].max()) <= 1.0

    def test_valid_actions_mask(self, env_config):
        """Mask should be True only for pairs (i, j) with i, j < num_nodes."""
        mask = get_valid_actions_mask(
            jnp.int32(3), env_config.max_nodes, env_config.max_actions,
        )
        assert mask.shape == (env_config.max_actions,)
        # With 3 nodes, valid pairs: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2) = 6 pairs
        # times 2 ops = 12 valid actions.
        assert int(mask.sum()) == 12


# ---------------------------------------------------------------------------
# jax_net: network forward pass
# ---------------------------------------------------------------------------

class TestJAXNet:
    def test_create_and_init(self, config, env_config):
        """Network can be created and parameters initialized."""
        net = create_network(config)
        rng = jax.random.PRNGKey(0)
        params = init_params(net, env_config, rng)
        # params should be a FrozenDict-like structure
        assert 'params' in params

    def test_forward_shapes(self, config, env_config):
        """Forward pass produces (max_actions,) logits and scalar value."""
        net = create_network(config)
        rng = jax.random.PRNGKey(0)
        params = init_params(net, env_config, rng)

        # Build a dummy observation.
        target = jnp.zeros(env_config.target_size, dtype=jnp.int32)
        state = reset(env_config, target)
        obs = get_observation(env_config, state)

        logits, value = net.apply(params, obs)
        assert logits.shape == (env_config.max_actions,)
        assert value.shape == ()

    def test_masked_logits(self, config, env_config):
        """Invalid actions should have very negative logits."""
        net = create_network(config)
        rng = jax.random.PRNGKey(0)
        params = init_params(net, env_config, rng)

        target = jnp.zeros(env_config.target_size, dtype=jnp.int32)
        state = reset(env_config, target)
        obs = get_observation(env_config, state)
        logits, _ = net.apply(params, obs)

        # Invalid action logits should be ~ -1e9
        invalid_mask = ~obs['mask']
        if invalid_mask.any():
            max_invalid = float(logits[invalid_mask].max())
            assert max_invalid < -1e8

    def test_jit_forward(self, config, env_config):
        """Network forward pass is JIT-compilable."""
        net = create_network(config)
        rng = jax.random.PRNGKey(0)
        params = init_params(net, env_config, rng)

        target = jnp.zeros(env_config.target_size, dtype=jnp.int32)
        state = reset(env_config, target)
        obs = get_observation(env_config, state)

        jit_apply = jax.jit(net.apply)
        logits, value = jit_apply(params, obs)
        assert logits.shape == (env_config.max_actions,)

    def test_vmap_forward(self, config, env_config):
        """Network forward pass works under vmap (batched inference)."""
        net = create_network(config)
        rng = jax.random.PRNGKey(0)
        params = init_params(net, env_config, rng)

        # Create batch of 4 observations.
        target = jnp.zeros(env_config.target_size, dtype=jnp.int32)
        state = reset(env_config, target)
        obs = get_observation(env_config, state)
        batch_obs = jax.tree.map(lambda x: jnp.stack([x] * 4), obs)

        logits, values = jax.vmap(lambda o: net.apply(params, o))(batch_obs)
        assert logits.shape == (4, env_config.max_actions)
        assert values.shape == (4,)


# ---------------------------------------------------------------------------
# jax_net: graph encoder
# ---------------------------------------------------------------------------

class TestGraphEncoder:
    def test_forward(self):
        """GraphEncoder produces correct output shape."""
        enc = GraphEncoder(hidden_dim=16, output_dim=8, num_layers=2)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((5, 4))
        edge_src = jnp.array([0, 1], dtype=jnp.int32)
        edge_dst = jnp.array([1, 2], dtype=jnp.int32)
        params = enc.init(rng, x, edge_src, edge_dst,
                          jnp.int32(3), jnp.int32(2))
        out = enc.apply(params, x, edge_src, edge_dst,
                        jnp.int32(3), jnp.int32(2))
        assert out.shape == (8,)

    def test_no_edges(self):
        """GraphEncoder handles zero edges gracefully."""
        enc = GraphEncoder(hidden_dim=16, output_dim=8, num_layers=2)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((5, 4))
        edge_src = jnp.array([-1, -1], dtype=jnp.int32)
        edge_dst = jnp.array([-1, -1], dtype=jnp.int32)
        params = enc.init(rng, x, edge_src, edge_dst,
                          jnp.int32(3), jnp.int32(0))
        out = enc.apply(params, x, edge_src, edge_dst,
                        jnp.int32(3), jnp.int32(0))
        assert out.shape == (8,)
        assert jnp.isfinite(out).all()


# ---------------------------------------------------------------------------
# Integration: MCTS search compiles
# ---------------------------------------------------------------------------

class TestMCTXIntegration:
    def test_mctx_muzero_compiles(self, config, env_config):
        """A minimal mctx.muzero_policy call compiles without error."""
        config_small = Config(
            n_variables=2, mod=5, max_complexity=4, max_steps=4,
            hidden_dim=16, embedding_dim=16, num_gnn_layers=1,
            mcts_simulations=4, seed=0,
        )
        ec = make_env_config(config_small)
        net = create_network(config_small)
        rng = jax.random.PRNGKey(0)
        params = init_params(net, ec, rng)

        # Create a batch of 2 environments.
        B = 2
        target = jnp.zeros(ec.target_size, dtype=jnp.int32)
        states = jax.vmap(lambda tc: reset(ec, tc))(
            jnp.stack([target] * B)
        )
        obs_batch = jax.vmap(lambda s: get_observation(ec, s))(states)

        # Root function.
        def root_fn(params, rng_key, obs_batch):
            logits, values = jax.vmap(
                lambda obs: net.apply(params, obs)
            )(obs_batch)
            embedding = {**obs_batch, '_env_state': states}
            return mctx.RootFnOutput(
                prior_logits=logits, value=values, embedding=embedding,
            )

        # Recurrent function.
        def recurrent_fn(params, rng_key, action, embedding):
            env_states = embedding['_env_state']
            import functools
            next_states, rewards, dones, _, *_ = jax.vmap(
                lambda s, a: step(ec, s, a)
            )(env_states, action)
            next_obs = jax.vmap(lambda s: get_observation(ec, s))(next_states)
            logits, values = jax.vmap(
                lambda obs: net.apply(params, obs)
            )(next_obs)
            discount = jnp.where(dones, 0.0, 0.99)
            new_emb = {**next_obs, '_env_state': next_states}
            return mctx.RecurrentFnOutput(
                reward=rewards, discount=discount,
                prior_logits=logits, value=values,
            ), new_emb

        root = root_fn(params, rng, obs_batch)
        rng, search_rng = jax.random.split(rng)

        policy_output = mctx.muzero_policy(
            params=params,
            rng_key=search_rng,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config_small.mcts_simulations,
            max_depth=config_small.max_steps,
            invalid_actions=~obs_batch['mask'],
        )

        # action_weights should have shape (B, max_actions)
        assert policy_output.action_weights.shape == (B, ec.max_actions)
        # action_weights should sum to ~1 per env
        sums = policy_output.action_weights.sum(axis=-1)
        np.testing.assert_allclose(np.array(sums), 1.0, atol=1e-5)

    def test_ppo_step_compiles(self, config, env_config):
        """A single PPO gradient step compiles and runs."""
        config_small = Config(
            n_variables=2, mod=5, max_complexity=4, max_steps=4,
            hidden_dim=16, embedding_dim=16, num_gnn_layers=1,
            batch_size=4, ppo_epochs=1, seed=0,
        )
        ec = make_env_config(config_small)
        net = create_network(config_small)
        rng = jax.random.PRNGKey(0)
        params = init_params(net, ec, rng)

        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(3e-4),
        )
        from flax.training import train_state
        ts = train_state.TrainState.create(
            apply_fn=net.apply, params=params, tx=tx,
        )

        # Dummy batch of 4 transitions.
        target = jnp.zeros(ec.target_size, dtype=jnp.int32)
        state = reset(ec, target)
        obs = get_observation(ec, state)
        batch_obs = jax.tree.map(lambda x: jnp.stack([x] * 4), obs)
        actions = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        advantages = jnp.array([0.1, -0.2, 0.3, -0.1])
        returns = jnp.array([1.0, 0.5, 1.5, 0.2])
        old_log_probs = jnp.array([-2.0, -3.0, -2.5, -1.5])

        @jax.jit
        def ppo_step(ts, obs, actions, advantages, returns, old_log_probs):
            def loss_fn(params):
                logits, values = jax.vmap(
                    lambda o: net.apply(params, o)
                )(obs)
                log_probs = jax.nn.log_softmax(logits)
                new_lp = log_probs[jnp.arange(actions.shape[0]), actions]
                probs = jax.nn.softmax(logits)
                entropy = -(probs * log_probs).sum(axis=-1).mean()
                ratio = jnp.exp(new_lp - old_log_probs)
                surr1 = ratio * advantages
                surr2 = jnp.clip(ratio, 0.8, 1.2) * advantages
                pg_loss = -jnp.minimum(surr1, surr2).mean()
                vf_loss = jnp.mean((values - returns) ** 2)
                total = pg_loss + 0.5 * vf_loss - 0.01 * entropy
                return total, {'pg_loss': pg_loss, 'vf_loss': vf_loss,
                               'entropy': entropy}

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (_, info), grads = grad_fn(ts.params)
            ts = ts.apply_gradients(grads=grads)
            return ts, info

        new_ts, info = ppo_step(
            ts, batch_obs, actions, advantages, returns, old_log_probs,
        )
        assert jnp.isfinite(info['pg_loss'])
        assert jnp.isfinite(info['vf_loss'])
        assert jnp.isfinite(info['entropy'])
        # Params should have been updated.
        old_flat = jax.tree.leaves(ts.params)
        new_flat = jax.tree.leaves(new_ts.params)
        any_changed = any(
            not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat)
        )
        assert any_changed, "PPO step should update at least some parameters"

    def test_ppo_update_logs_kl_and_early_stops(self):
        from src.algorithms.ppo_mcts_jax import PPOMCTSJAXTrainer, Transition

        cfg = Config(
            n_variables=2,
            mod=5,
            max_complexity=2,
            max_steps=2,
            hidden_dim=16,
            embedding_dim=16,
            num_gnn_layers=1,
            batch_size=2,
            ppo_epochs=3,
            target_kl=0.01,
            seed=0,
        )
        trainer = PPOMCTSJAXTrainer(cfg, batch_size=2)
        ec = trainer.env_config
        state = reset(ec, jnp.zeros(ec.target_size, dtype=jnp.int32))
        obs = get_observation(ec, state)
        transitions = [
            Transition(
                obs=obs,
                action=i % ec.max_actions,
                reward=0.0,
                network_log_prob=10.0,
                value=0.0,
                done=False,
            )
            for i in range(4)
        ]
        advantages = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        returns = np.array([1.0, 0.0, 0.5, -0.5], dtype=np.float32)

        info = trainer.update(transitions, advantages, returns)

        assert info["approx_kl"] > cfg.target_kl
        assert info["kl_early_stop_count"] == 1
        assert info["applied_minibatch_updates"] < cfg.ppo_epochs * 2
        assert info["skipped_minibatch_updates"] == 0
        assert info["skipped_outer_iteration"] == 0
