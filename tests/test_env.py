import unittest
import warnings
import numpy as np

from poly_circuit_rl.core.action_codec import (
    ACTION_ADD, ACTION_MUL, ACTION_SET_OUTPUT, ACTION_STOP,
    encode_action, decode_action, action_space_size,
)
from poly_circuit_rl.core.poly import make_var, add, make_const, mul
from poly_circuit_rl.core.factor_library import FactorLibrary

from poly_circuit_rl.config import Config
from poly_circuit_rl.env.circuit_env import PolyCircuitEnv


class TestPolyCircuitEnv(unittest.TestCase):

    def setUp(self):
        self.config = Config(n_vars=2, m=8, L=8, max_nodes=8, max_ops=4, seed=123)
        self.env = PolyCircuitEnv(self.config)

    def test_reset_shapes(self):
        obs_dict, info = self.env.reset(options={"max_ops": 2})
        self.assertEqual(obs_dict["obs"].shape, (self.config.obs_dim,))
        self.assertEqual(obs_dict["action_mask"].shape, (self.config.action_dim,))
        self.assertEqual(obs_dict["obs"].dtype, np.float32)
        self.assertEqual(obs_dict["action_mask"].dtype, np.int8)

    def test_mask_all_valid_actions_executable(self):
        obs_dict, _ = self.env.reset(options={"max_ops": 2})
        mask = obs_dict["action_mask"]
        valid_actions = np.where(mask > 0)[0]
        self.assertGreater(len(valid_actions), 0)
        for a in valid_actions[:5]:
            env2 = PolyCircuitEnv(self.config)
            env2.reset(seed=123, options={"max_ops": 2})
            obs_d, r, term, trunc, info = env2.step(int(a))
            self.assertFalse(info.get("invalid", False))

    def test_steps_left_decrements(self):
        obs_dict, _ = self.env.reset(options={"max_ops": 3})
        self.assertEqual(self.env.steps_left, 3)
        L = self.config.L
        action = encode_action(ACTION_ADD, 0, 1, L)
        self.env.step(action)
        self.assertEqual(self.env.steps_left, 2)

    def test_step_cost_reward(self):
        self.env.reset(options={"max_ops": 3})
        L = self.config.L
        _, r, _, _, _ = self.env.step(encode_action(ACTION_ADD, 0, 1, L))
        self.assertAlmostEqual(r, -self.config.step_cost)

    def test_solve_detection(self):
        target = make_var(2, 0)
        obs_dict, _ = self.env.reset(options={"max_ops": 2, "target_poly": target})
        L = self.config.L
        self.env.step(encode_action(ACTION_SET_OUTPUT, 0, None, L))
        _, r, term, trunc, info = self.env.step(encode_action(ACTION_STOP, None, None, L))
        self.assertTrue(info["solved"] or term)

    def test_solve_with_addition(self):
        a = make_var(2, 0)
        b = make_var(2, 1)
        target = add(a, b)
        obs_dict, _ = self.env.reset(options={"max_ops": 2, "target_poly": target})
        L = self.config.L
        self.env.step(encode_action(ACTION_ADD, 0, 1, L))
        new_node_idx = len(self.env.builder.nodes) - 1
        self.env.step(encode_action(ACTION_SET_OUTPUT, new_node_idx, None, L))
        _, r, term, trunc, info = self.env.step(encode_action(ACTION_STOP, None, None, L))
        self.assertTrue(info["solved"] or term)

    def test_truncation_on_budget(self):
        self.env.reset(options={"max_ops": 1})
        L = self.config.L
        self.env.step(encode_action(ACTION_ADD, 0, 1, L))
        _, r, term, trunc, info = self.env.step(encode_action(ACTION_STOP, None, None, L))
        self.assertTrue(trunc or term)

    def test_trajectory_length(self):
        self.env.reset(options={"max_ops": 2})
        L = self.config.L
        self.env.step(encode_action(ACTION_ADD, 0, 1, L))
        self.env.step(encode_action(ACTION_STOP, None, None, L))
        traj = self.env.get_trajectory()
        self.assertEqual(len(traj), 2)

    def test_determinism(self):
        config = Config(n_vars=2, m=8, L=8, max_nodes=8, max_ops=2, seed=999)
        env1 = PolyCircuitEnv(config)
        env2 = PolyCircuitEnv(config)
        obs1, _ = env1.reset(options={"max_ops": 2})
        obs2, _ = env2.reset(options={"max_ops": 2})
        np.testing.assert_array_equal(obs1["obs"], obs2["obs"])

    def test_no_hidden_node_expansion_beyond_visible_slots(self):
        config = Config(n_vars=2, m=8, L=4, max_nodes=4, max_ops=4, seed=123)
        env = PolyCircuitEnv(config)
        obs_dict, _ = env.reset(options={"max_ops": 4})
        L = config.L

        # Build one new visible op node (initial leaves are x0, x1, const_1 => 3 nodes).
        obs_dict, _, _, _, _ = env.step(encode_action(ACTION_ADD, 0, 1, L))
        self.assertEqual(len(env.builder.nodes), L)

        # Once visible slots are full, ADD/MUL should be masked out.
        pairs = L * (L + 1) // 2
        mask = obs_dict["action_mask"]
        self.assertEqual(int(mask[: 2 * pairs].sum()), 0)

    def test_reset_restores_sampler_max_steps_on_exception(self):
        class _RaisingSampler:
            def __init__(self):
                self.max_steps = 99

            def sample(self, rng):
                raise RuntimeError("sampler failure")

        sampler = _RaisingSampler()
        env = PolyCircuitEnv(self.config, target_sampler=sampler)
        with self.assertRaises(RuntimeError):
            env.reset(options={"max_ops": 2})
        self.assertEqual(sampler.max_steps, 99)

    def test_oracle_empty_actions_warns_once_per_target(self):
        class _EmptyOracle:
            def get_optimal_actions(self, target_poly, current_nodes):
                _ = (target_poly, current_nodes)
                return []

        env = PolyCircuitEnv(self.config)
        env._oracle_helper = _EmptyOracle()
        target = add(make_var(2, 0), make_var(2, 1))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            env.reset(options={"max_ops": 2, "target_poly": target})
            _ = env._build_mask()
            _ = env._build_mask()

        oracle_warnings = [w for w in caught if "Oracle mask returned no actions" in str(w.message)]
        self.assertEqual(len(oracle_warnings), 1)

    def test_sparse_reward_mode_only_base_and_solve_bonus(self):
        config = Config(
            n_vars=2,
            m=8,
            L=8,
            max_nodes=8,
            max_ops=2,
            seed=123,
            reward_mode="sparse",
            shaping_coeff=1.0,
            factor_shaping_coeff=1.0,
            factor_library_enabled=True,
        )
        env = PolyCircuitEnv(config, factor_library=FactorLibrary(n_vars=2))
        target = add(make_var(2, 0), make_var(2, 1))
        env.reset(options={"max_ops": 2, "target_poly": target})
        L = config.L
        _, r_add, _, _, _ = env.step(encode_action(ACTION_ADD, 0, 1, L))
        new_node_idx = len(env.builder.nodes) - 1
        _, r_out, term, _, _ = env.step(encode_action(ACTION_SET_OUTPUT, new_node_idx, None, L))

        self.assertAlmostEqual(r_add, -config.step_cost)
        self.assertAlmostEqual(r_out, 1.0)
        self.assertTrue(term)

    def test_completion_bonus_not_double_counted(self):
        config = Config(
            n_vars=2,
            m=8,
            L=8,
            max_nodes=8,
            max_ops=2,
            seed=123,
            shaping_coeff=0.0,
            factor_shaping_coeff=0.0,
            factor_library_enabled=True,
            factor_subgoal_reward=0.0,
            factor_library_bonus=0.0,
            completion_bonus=0.5,
            reward_mode="full",
        )
        x0 = make_var(2, 0)
        one = make_const(2, 1)
        two = make_const(2, 2)
        x0_plus_one = add(x0, one)
        target = mul(two, x0_plus_one)  # 2 * (x0 + 1)

        factor_library = FactorLibrary(n_vars=2)
        factor_library.register(x0_plus_one, step_num=1)
        env = PolyCircuitEnv(config, factor_library=factor_library)
        env.reset(options={"max_ops": 2, "target_poly": target})

        add_action = encode_action(ACTION_ADD, 0, 2, config.L)
        _, reward, _, _, _ = env.step(add_action)
        expected = -config.step_cost + config.completion_bonus
        self.assertAlmostEqual(reward, expected)


if __name__ == "__main__":
    unittest.main()
