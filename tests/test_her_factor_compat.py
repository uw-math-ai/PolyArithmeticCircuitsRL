"""Tests for HER + Factor Library compatibility.

Verifies that:
- Original transitions retain factor shaping rewards
- HER-relabeled transitions have shaping stripped (only solve bonus changes)
- Reward decomposition is consistent: reward == base + shaping + solve
"""

import unittest
from fractions import Fraction

import numpy as np

from poly_circuit_rl.config import Config
from poly_circuit_rl.core.poly import add, mul, make_var, make_const
from poly_circuit_rl.core.factor_library import FactorLibrary
from poly_circuit_rl.env.circuit_env import PolyCircuitEnv
from poly_circuit_rl.rl.replay_buffer import HERReplayBuffer, Transition


class TestHERFactorCompatibility(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            n_vars=2,
            max_ops=4,
            L=16,
            m=16,
            seed=42,
            factor_library_enabled=True,
            factor_subgoal_reward=0.3,
            factor_library_bonus=0.15,
            completion_bonus=0.5,
            her_k=4,
            buffer_size=10000,
        )

    def test_decomposed_rewards_sum_correctly(self):
        """Verify reward = base_reward + shaping_reward + solve_bonus in trajectory."""
        fl = FactorLibrary(n_vars=2)
        env = PolyCircuitEnv(self.config, factor_library=fl)

        # Reset with a specific target
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        c1 = make_const(2, 1)
        target = add(mul(x0, x1), c1)  # x0*x1 + 1

        env.reset(options={"target_poly": target, "max_ops": 4})

        # Take a few actions and check decomposition
        mask = env._build_mask()
        valid = np.where(mask > 0)[0]
        if len(valid) == 0:
            return

        # Take a valid action
        action = int(valid[0])
        _, reward, _, _, _ = env.step(action)

        traj = env.get_trajectory()
        self.assertTrue(len(traj) > 0)

        entry = traj[-1]
        expected = entry["base_reward"] + entry["shaping_reward"] + entry["solve_bonus"]
        self.assertAlmostEqual(entry["reward"], expected, places=10)

    def test_her_relabeled_no_shaping(self):
        """HER-relabeled transitions should have shaping_reward = 0."""
        buffer = HERReplayBuffer(self.config)

        obs_dim = self.config.obs_dim
        act_dim = self.config.action_dim

        # Create a simple episode with non-zero shaping
        T = 3
        ep_obs = [np.random.randn(obs_dim).astype(np.float32) for _ in range(T)]
        ep_actions = [0, 1, 2]
        ep_rewards = [-0.05 + 0.3, -0.05, -0.05]  # first has shaping
        ep_next_obs = [np.random.randn(obs_dim).astype(np.float32) for _ in range(T)]
        ep_dones = [False, False, True]
        ep_solved = [False, False, False]
        ep_truncated = [False, False, True]
        ep_masks = [np.ones(act_dim, dtype=np.int8) for _ in range(T)]
        ep_next_masks = [np.ones(act_dim, dtype=np.int8) for _ in range(T)]
        # Achieved goals: only the last step has one
        goal = np.random.randn(self.config.m).astype(np.float32)
        ep_achieved_goals = [None, None, goal]

        ep_base_rewards = [-0.05, -0.05, -0.05]
        ep_shaping_rewards = [0.3, 0.0, 0.0]
        ep_solve_bonuses = [0.0, 0.0, 0.0]

        buffer.add_episode_with_her(
            ep_obs=ep_obs,
            ep_actions=ep_actions,
            ep_rewards=ep_rewards,
            ep_next_obs=ep_next_obs,
            ep_dones=ep_dones,
            ep_solved=ep_solved,
            ep_truncated=ep_truncated,
            ep_masks=ep_masks,
            ep_next_masks=ep_next_masks,
            ep_achieved_goals=ep_achieved_goals,
            ep_achieved_goal_keys=[None, None, ("goal", 1)],
            ep_base_rewards=ep_base_rewards,
            ep_shaping_rewards=ep_shaping_rewards,
            ep_solve_bonuses=ep_solve_bonuses,
        )

        # Buffer should have original transitions (3) + HER relabeled
        # The original transitions should have shaping_reward as recorded
        # HER relabeled transitions should have shaping_reward = 0

        originals = buffer.buffer[:T]
        relabeled = buffer.buffer[T:]

        # Check original has shaping
        self.assertAlmostEqual(originals[0].shaping_reward, 0.3)

        # Check all relabeled have zero shaping
        for t in relabeled:
            self.assertAlmostEqual(t.shaping_reward, 0.0,
                                   msg="HER relabeled transition should have no shaping")

    def test_her_relabeled_solve_bonus_correct(self):
        """When relabeled goal matches achieved, solve_bonus should be 1.0."""
        buffer = HERReplayBuffer(self.config)

        obs_dim = self.config.obs_dim
        act_dim = self.config.action_dim

        # Episode where step 1 achieves a goal that step 2 uses for relabeling
        T = 2
        goal_vec = np.ones(self.config.m, dtype=np.float32) * 0.5
        ep_obs = [np.random.randn(obs_dim).astype(np.float32) for _ in range(T)]
        ep_actions = [0, 1]
        ep_rewards = [-0.05, -0.05]
        ep_next_obs = [np.random.randn(obs_dim).astype(np.float32) for _ in range(T)]
        ep_dones = [False, True]
        ep_solved = [False, False]
        ep_truncated = [False, True]
        ep_masks = [np.ones(act_dim, dtype=np.int8) for _ in range(T)]
        ep_next_masks = [np.ones(act_dim, dtype=np.int8) for _ in range(T)]
        # Both steps achieve the same goal (so relabeling step 0 with step 1's goal
        # should detect a match at step 0's achieved_goal)
        ep_achieved_goals = [goal_vec.copy(), goal_vec.copy()]

        ep_base_rewards = [-0.05, -0.05]
        ep_shaping_rewards = [0.0, 0.0]
        ep_solve_bonuses = [0.0, 0.0]

        buffer.add_episode_with_her(
            ep_obs=ep_obs,
            ep_actions=ep_actions,
            ep_rewards=ep_rewards,
            ep_next_obs=ep_next_obs,
            ep_dones=ep_dones,
            ep_solved=ep_solved,
            ep_truncated=ep_truncated,
            ep_masks=ep_masks,
            ep_next_masks=ep_next_masks,
            ep_achieved_goals=ep_achieved_goals,
            ep_achieved_goal_keys=[("goal", 7), ("goal", 7)],
            ep_base_rewards=ep_base_rewards,
            ep_shaping_rewards=ep_shaping_rewards,
            ep_solve_bonuses=ep_solve_bonuses,
        )

        # Check that relabeled transition for step 0 has solve_bonus = 1.0
        # (because achieved_goal[0] matches the relabeled goal from step 1)
        relabeled = buffer.buffer[T:]
        if relabeled:
            # At least one relabeled should have solve_bonus = 1.0
            has_solve = any(t.solve_bonus > 0.5 for t in relabeled)
            self.assertTrue(has_solve,
                            "Relabeled transition should get solve_bonus=1.0 "
                            "when achieved goal matches relabeled goal")

    def test_original_total_reward_is_full(self):
        """Original transitions should have reward = base + shaping + solve."""
        buffer = HERReplayBuffer(self.config)

        obs_dim = self.config.obs_dim
        act_dim = self.config.action_dim

        T = 2
        ep_obs = [np.random.randn(obs_dim).astype(np.float32) for _ in range(T)]
        ep_actions = [0, 1]
        ep_rewards = [0.25, 0.95]  # base + shaping + solve
        ep_next_obs = [np.random.randn(obs_dim).astype(np.float32) for _ in range(T)]
        ep_dones = [False, True]
        ep_solved = [False, True]
        ep_truncated = [False, False]
        ep_masks = [np.ones(act_dim, dtype=np.int8) for _ in range(T)]
        ep_next_masks = [np.ones(act_dim, dtype=np.int8) for _ in range(T)]
        ep_achieved_goals = [None, np.ones(self.config.m, dtype=np.float32)]

        ep_base_rewards = [-0.05, -0.05]
        ep_shaping_rewards = [0.3, 0.0]
        ep_solve_bonuses = [0.0, 1.0]

        buffer.add_episode_with_her(
            ep_obs=ep_obs,
            ep_actions=ep_actions,
            ep_rewards=ep_rewards,
            ep_next_obs=ep_next_obs,
            ep_dones=ep_dones,
            ep_solved=ep_solved,
            ep_truncated=ep_truncated,
            ep_masks=ep_masks,
            ep_next_masks=ep_next_masks,
            ep_achieved_goals=ep_achieved_goals,
            ep_achieved_goal_keys=[None, ("goal", 2)],
            ep_base_rewards=ep_base_rewards,
            ep_shaping_rewards=ep_shaping_rewards,
            ep_solve_bonuses=ep_solve_bonuses,
        )

        # Check original transitions
        t0 = buffer.buffer[0]
        self.assertAlmostEqual(t0.reward, 0.25)
        self.assertAlmostEqual(t0.base_reward, -0.05)
        self.assertAlmostEqual(t0.shaping_reward, 0.3)
        self.assertAlmostEqual(t0.solve_bonus, 0.0)

        t1 = buffer.buffer[1]
        self.assertAlmostEqual(t1.reward, 0.95)
        self.assertAlmostEqual(t1.base_reward, -0.05)
        self.assertAlmostEqual(t1.shaping_reward, 0.0)
        self.assertAlmostEqual(t1.solve_bonus, 1.0)


if __name__ == "__main__":
    unittest.main()
