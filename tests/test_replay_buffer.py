import unittest
import math
import numpy as np

from poly_circuit_rl.config import Config
from poly_circuit_rl.core.action_codec import ACTION_ADD, ACTION_SET_OUTPUT, encode_action
from poly_circuit_rl.core.fingerprints import eval_poly_points
from poly_circuit_rl.core.poly import add, make_var, mul, poly_hashkey
from poly_circuit_rl.env.circuit_env import PolyCircuitEnv
from poly_circuit_rl.env.obs import extract_goal
from poly_circuit_rl.rl.replay_buffer import HERReplayBuffer, Transition


class TestHERReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.config = Config(
            n_vars=2, m=8, L=8, max_nodes=8,
            d_model=32, n_heads=2, n_layers=1, d_pos=4,
            buffer_size=100, batch_size=4, learning_starts=5,
        )
        self.buf = HERReplayBuffer(self.config)

    def test_add_and_sample(self):
        for _ in range(10):
            t = Transition(
                obs=np.random.randn(self.config.obs_dim).astype(np.float32),
                action=0,
                reward=0.0,
                next_obs=np.random.randn(self.config.obs_dim).astype(np.float32),
                done=False,
                action_mask=np.ones(self.config.action_dim, dtype=np.int8),
                next_action_mask=np.ones(self.config.action_dim, dtype=np.int8),
            )
            self.buf.add(t)
        self.assertEqual(len(self.buf), 10)
        batch = self.buf.sample(4)
        self.assertEqual(batch["obs"].shape, (4, self.config.obs_dim))

    def test_her_increases_buffer_size(self):
        T = 5
        ep_obs = [np.random.randn(self.config.obs_dim).astype(np.float32) for _ in range(T)]
        ep_actions = [0] * T
        ep_rewards = [0.0] * T
        ep_next_obs = [np.random.randn(self.config.obs_dim).astype(np.float32) for _ in range(T)]
        ep_dones = [False] * (T - 1) + [True]
        ep_solved = [False] * T
        ep_truncated = [False] * (T - 1) + [True]
        ep_masks = [np.ones(self.config.action_dim, dtype=np.int8) for _ in range(T)]
        ep_next_masks = [np.ones(self.config.action_dim, dtype=np.int8) for _ in range(T)]
        ep_achieved_goals = [np.random.randn(self.config.m).astype(np.float32) for _ in range(T)]

        self.buf.add_episode_with_her(
            ep_obs, ep_actions, ep_rewards, ep_next_obs,
            ep_dones, ep_solved, ep_truncated, ep_masks, ep_next_masks, ep_achieved_goals,
        )
        self.assertGreater(len(self.buf), T)

    def test_her_requires_selected_output_goal(self):
        env = PolyCircuitEnv(Config(
            n_vars=2, m=8, L=8, max_nodes=8, max_ops=2, seed=123,
            buffer_size=100, her_k=1, shaping_coeff=0.0, factor_shaping_coeff=0.0,
        ))
        target = mul(make_var(2, 0), make_var(2, 1))
        obs0, _ = env.reset(options={"max_ops": 2, "target_poly": target})

        add_action = encode_action(ACTION_ADD, 0, 1, env.config.L)
        obs1, r1, term1, trunc1, _ = env.step(add_action)
        out_action = encode_action(ACTION_SET_OUTPUT, len(env.builder.nodes) - 1, None, env.config.L)
        obs2, r2, term2, trunc2, _ = env.step(out_action)
        traj = env.get_trajectory()

        buf = HERReplayBuffer(env.config)
        buf.add_episode_with_her(
            ep_obs=[obs0["obs"], obs1["obs"]],
            ep_actions=[add_action, out_action],
            ep_rewards=[r1, r2],
            ep_next_obs=[obs1["obs"], obs2["obs"]],
            ep_dones=[term1 or trunc1, term2 or trunc2],
            ep_solved=[term1, term2],
            ep_truncated=[trunc1, trunc2],
            ep_masks=[obs0["action_mask"], obs1["action_mask"]],
            ep_next_masks=[obs1["action_mask"], obs2["action_mask"]],
            ep_achieved_goals=[traj[0]["achieved_goal"], traj[1]["achieved_goal"]],
        )

        relabeled = [
            t for t in buf.buffer
            if t.action == add_action
            and np.allclose(extract_goal(t.obs, env.config), traj[1]["achieved_goal"], atol=1e-6)
        ]
        self.assertEqual(len(relabeled), 1)
        self.assertAlmostEqual(relabeled[0].reward, r1)
        self.assertFalse(relabeled[0].done)

    def test_her_rewrites_only_solve_bonus_and_termination(self):
        goal_now = np.ones(self.config.m, dtype=np.float32)
        future_goal = np.full(self.config.m, 2.0, dtype=np.float32)
        ep_obs = [np.zeros(self.config.obs_dim, dtype=np.float32) for _ in range(2)]
        ep_next_obs = [np.zeros(self.config.obs_dim, dtype=np.float32) for _ in range(2)]
        ep_masks = [np.ones(self.config.action_dim, dtype=np.int8) for _ in range(2)]
        ep_next_masks = [np.ones(self.config.action_dim, dtype=np.int8) for _ in range(2)]

        # Decomposed: base=0.25, shaping=0.0, solve=1.0 -> total=1.25
        self.buf.add_episode_with_her(
            ep_obs=ep_obs,
            ep_actions=[7, 9],
            ep_rewards=[1.25, 0.0],
            ep_next_obs=ep_next_obs,
            ep_dones=[True, True],
            ep_solved=[True, False],
            ep_truncated=[False, True],
            ep_masks=ep_masks,
            ep_next_masks=ep_next_masks,
            ep_achieved_goals=[goal_now, future_goal],
            ep_base_rewards=[0.25, 0.0],
            ep_shaping_rewards=[0.0, 0.0],
            ep_solve_bonuses=[1.0, 0.0],
        )

        relabeled = [
            t for t in self.buf.buffer
            if t.action == 7 and np.allclose(extract_goal(t.obs, self.config), future_goal, atol=1e-6)
        ]
        self.assertEqual(len(relabeled), 1)
        # Relabeled reward = base_reward(0.25) + relabeled_solve_bonus(0.0) = 0.25
        self.assertAlmostEqual(relabeled[0].reward, 0.25)
        self.assertFalse(relabeled[0].done)

    def test_her_works_with_shaping_via_decomposition(self):
        """HER relabeling strips shaping rewards but keeps base_reward."""
        config = Config(
            n_vars=2, m=8, L=8, max_nodes=8,
            d_model=32, n_heads=2, n_layers=1, d_pos=4,
            buffer_size=100, batch_size=4, learning_starts=5,
            shaping_coeff=0.3,
        )
        buf = HERReplayBuffer(config)
        ep_obs = [np.random.randn(config.obs_dim).astype(np.float32) for _ in range(2)]
        ep_next_obs = [np.random.randn(config.obs_dim).astype(np.float32) for _ in range(2)]
        ep_masks = [np.ones(config.action_dim, dtype=np.int8) for _ in range(2)]
        ep_next_masks = [np.ones(config.action_dim, dtype=np.int8) for _ in range(2)]

        achieved = np.zeros(config.m, dtype=np.float32)
        buf.add_episode_with_her(
            ep_obs=ep_obs,
            ep_actions=[0, 1],
            ep_rewards=[-0.01, 1.3],  # total rewards
            ep_next_obs=ep_next_obs,
            ep_dones=[False, True],
            ep_solved=[False, True],
            ep_truncated=[False, False],
            ep_masks=ep_masks,
            ep_next_masks=ep_next_masks,
            ep_achieved_goals=[None, achieved],
            ep_base_rewards=[-0.01, -0.01],
            ep_shaping_rewards=[0.0, 0.31],
            ep_solve_bonuses=[0.0, 1.0],
        )

        # 2 original + HER relabeled (only step 0 can be relabeled with step 1's goal,
        # but step 0's achieved_goal is None so relabeled_solved = False)
        self.assertGreaterEqual(len(buf), 2)
        # Check that relabeled transitions (if any) have shaping_reward=0.0
        for t in buf.buffer:
            if t.action == 0 and t.shaping_reward == 0.0:
                # Relabeled: reward = base_reward + relabeled_solve_bonus
                self.assertAlmostEqual(t.reward, t.base_reward + t.solve_bonus)

    def test_circular_buffer(self):
        small_config = Config(
            n_vars=2, m=8, L=8, max_nodes=8,
            d_model=32, n_heads=2, n_layers=1, d_pos=4,
            buffer_size=5,
        )
        buf = HERReplayBuffer(small_config)
        for i in range(10):
            t = Transition(
                obs=np.full(small_config.obs_dim, float(i), dtype=np.float32),
                action=0, reward=0.0,
                next_obs=np.zeros(small_config.obs_dim, dtype=np.float32),
                done=False,
                action_mask=np.ones(small_config.action_dim, dtype=np.int8),
                next_action_mask=np.ones(small_config.action_dim, dtype=np.int8),
            )
            buf.add(t)
        self.assertEqual(len(buf), 5)

    def test_her_goal_collision_requires_exact_poly_key(self):
        config = Config(
            n_vars=1,
            m=4,
            L=4,
            max_nodes=4,
            her_k=1,
            buffer_size=100,
            eval_norm_scale=0.01,
        )
        buf = HERReplayBuffer(config)

        x = make_var(1, 0)
        x2 = mul(x, x)
        x4 = mul(x2, x2)
        x8 = mul(x4, x4)
        self.assertNotEqual(poly_hashkey(x4), poly_hashkey(x8))

        points = [(2,), (3,), (4,), (5,)]
        goal_x4 = np.array(
            [math.tanh(float(v) / config.eval_norm_scale) for v in eval_poly_points(x4, points)],
            dtype=np.float32,
        )
        goal_x8 = np.array(
            [math.tanh(float(v) / config.eval_norm_scale) for v in eval_poly_points(x8, points)],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(goal_x4, goal_x8, atol=1e-6))

        ep_obs = [np.zeros(config.obs_dim, dtype=np.float32) for _ in range(2)]
        ep_next_obs = [np.zeros(config.obs_dim, dtype=np.float32) for _ in range(2)]
        ep_masks = [np.ones(config.action_dim, dtype=np.int8) for _ in range(2)]
        ep_next_masks = [np.ones(config.action_dim, dtype=np.int8) for _ in range(2)]

        buf.add_episode_with_her(
            ep_obs=ep_obs,
            ep_actions=[0, 1],
            ep_rewards=[0.0, 0.0],
            ep_next_obs=ep_next_obs,
            ep_dones=[False, True],
            ep_solved=[False, False],
            ep_truncated=[False, True],
            ep_masks=ep_masks,
            ep_next_masks=ep_next_masks,
            ep_achieved_goals=[goal_x4, goal_x8],
            ep_achieved_goal_keys=[poly_hashkey(x4), poly_hashkey(x8)],
            ep_base_rewards=[0.0, 0.0],
            ep_shaping_rewards=[0.0, 0.0],
            ep_solve_bonuses=[0.0, 0.0],
        )

        relabeled = [
            t for t in buf.buffer
            if t.action == 0 and np.allclose(extract_goal(t.obs, config), goal_x8, atol=1e-6)
        ]
        self.assertEqual(len(relabeled), 1)
        self.assertEqual(relabeled[0].solve_bonus, 0.0)
        self.assertFalse(relabeled[0].done)


if __name__ == "__main__":
    unittest.main()
