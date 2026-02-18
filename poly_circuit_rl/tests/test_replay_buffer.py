import unittest
import numpy as np

from poly_circuit_rl.config import Config
from poly_circuit_rl.rl.replay_buffer import HERReplayBuffer, Transition


class TestHERReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.config = Config(
            n_vars=2, m=8, L=8,
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
        ep_masks = [np.ones(self.config.action_dim, dtype=np.int8) for _ in range(T)]
        ep_next_masks = [np.ones(self.config.action_dim, dtype=np.int8) for _ in range(T)]
        ep_node_evals = [
            [np.random.randn(self.config.m).astype(np.float32) for _ in range(3)]
            for _ in range(T)
        ]

        self.buf.add_episode_with_her(
            ep_obs, ep_actions, ep_rewards, ep_next_obs,
            ep_dones, ep_masks, ep_next_masks, ep_node_evals,
        )
        self.assertGreater(len(self.buf), T)

    def test_circular_buffer(self):
        small_config = Config(
            n_vars=2, m=8, L=8,
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


if __name__ == "__main__":
    unittest.main()
