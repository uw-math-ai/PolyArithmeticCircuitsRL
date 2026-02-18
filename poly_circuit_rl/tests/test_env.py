import unittest
import numpy as np

from poly_circuit_rl.core.action_codec import (
    ACTION_ADD, ACTION_MUL, ACTION_SET_OUTPUT, ACTION_STOP,
    encode_action, decode_action, action_space_size,
)
from poly_circuit_rl.core.poly import make_var, add

from poly_circuit_rl.config import Config
from poly_circuit_rl.env.circuit_env import PolyCircuitEnv


class TestPolyCircuitEnv(unittest.TestCase):

    def setUp(self):
        self.config = Config(n_vars=2, m=8, L=8, max_ops=4, seed=123)
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
        config = Config(n_vars=2, m=8, L=8, max_ops=2, seed=999)
        env1 = PolyCircuitEnv(config)
        env2 = PolyCircuitEnv(config)
        obs1, _ = env1.reset(options={"max_ops": 2})
        obs2, _ = env2.reset(options={"max_ops": 2})
        np.testing.assert_array_equal(obs1["obs"], obs2["obs"])


if __name__ == "__main__":
    unittest.main()
