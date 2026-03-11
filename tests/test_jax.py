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
)
from src.algorithms.jax_net import (
    PolicyValueNet, create_network, init_params, GraphEncoder,
)


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
        next_state, reward, done, is_success = step(env_config, initial_state, action)

        assert int(next_state.num_nodes) == int(initial_state.num_nodes) + 1
        assert int(next_state.num_edges) == 4  # bidirectional to both operands
        assert int(next_state.steps_taken) == 1

    def test_step_success(self, env_config, initial_state):
        """Adding x0 + x1 when target is x0+x1 should yield success."""
        action = encode_action(jnp.int32(0), jnp.int32(0), jnp.int32(1),
                               env_config.max_nodes)
        next_state, reward, done, is_success = step(env_config, initial_state, action)

        assert bool(is_success)
        assert bool(done)
        assert float(reward) > 0  # success_reward + step_penalty > 0

    def test_step_jit(self, env_config, initial_state):
        """env step is JIT-compilable."""
        import functools
        jit_step = jax.jit(functools.partial(step, env_config))
        action = encode_action(jnp.int32(0), jnp.int32(0), jnp.int32(1),
                               env_config.max_nodes)
        next_state, reward, done, is_success = jit_step(initial_state, action)
        assert next_state.num_nodes.shape == ()

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
            next_states, rewards, dones, _ = jax.vmap(
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
