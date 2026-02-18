import os
import tempfile
import unittest
import numpy as np

from poly_circuit_rl.config import Config
from poly_circuit_rl.rl.agent import DQNAgent


class TestDQNAgent(unittest.TestCase):

    def setUp(self):
        self.config = Config(
            n_vars=2, m=8, L=8,
            d_model=32, n_heads=2, n_layers=1, d_pos=4,
            buffer_size=100, batch_size=4, learning_starts=5,
        )
        self.agent = DQNAgent(self.config)

    def test_select_action_valid(self):
        obs = np.random.randn(self.config.obs_dim).astype(np.float32)
        mask = np.zeros(self.config.action_dim, dtype=np.int8)
        mask[0] = 1
        mask[5] = 1
        mask[10] = 1

        action = self.agent.select_action(obs, mask, deterministic=True)
        self.assertIn(action, [0, 5, 10])

    def test_select_action_explores(self):
        """Epsilon=1 should always pick a random valid action."""
        self.agent.total_steps = 0  # eps = 1.0
        obs = np.random.randn(self.config.obs_dim).astype(np.float32)
        mask = np.zeros(self.config.action_dim, dtype=np.int8)
        mask[0] = 1
        mask[5] = 1

        actions = set()
        for _ in range(50):
            a = self.agent.select_action(obs, mask)
            self.assertIn(a, [0, 5])
            actions.add(a)

        # With 50 trials and eps=1, should hit both actions
        self.assertEqual(actions, {0, 5})

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_ckpt.pt")
            self.agent.total_steps = 42
            self.agent.save(path)

            agent2 = DQNAgent(self.config)
            agent2.load(path)
            self.assertEqual(agent2.total_steps, 42)

            # Forward pass should give same results (use fixed obs)
            rng = np.random.RandomState(123)
            obs = rng.randn(self.config.obs_dim).astype(np.float32)
            mask = np.ones(self.config.action_dim, dtype=np.int8)

            # Both agents should have identical weights after load
            import torch
            self.agent.q_network.eval()
            agent2.q_network.eval()
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q1 = self.agent.q_network(obs_t)
            q2 = agent2.q_network(obs_t)
            torch.testing.assert_close(q1, q2)


if __name__ == "__main__":
    unittest.main()
