import unittest
import numpy as np
import torch

from poly_circuit_rl.config import Config
from poly_circuit_rl.rl.network import CircuitTransformerQ


class TestCircuitTransformerQ(unittest.TestCase):

    def setUp(self):
        self.config = Config(
            n_vars=2, m=8, L=8,
            d_model=32, n_heads=2, n_layers=1, d_pos=4,
        )
        self.net = CircuitTransformerQ(self.config)
        self.net.eval()

    def test_forward_shape(self):
        B = 4
        obs = torch.randn(B, self.config.obs_dim)
        q = self.net(obs)
        self.assertEqual(q.shape, (B, self.config.action_dim))

    def test_single_batch(self):
        obs = torch.randn(1, self.config.obs_dim)
        q = self.net(obs)
        self.assertEqual(q.shape, (1, self.config.action_dim))

    def test_parameter_count(self):
        count = self.net.count_parameters()
        self.assertGreater(count, 0)
        self.assertLess(count, 500_000)

    def test_output_changes_with_input(self):
        """Different observations should produce different Q-values."""
        obs1 = torch.randn(1, self.config.obs_dim)
        obs2 = torch.randn(1, self.config.obs_dim) + 5.0
        q1 = self.net(obs1)
        q2 = self.net(obs2)
        self.assertFalse(torch.allclose(q1, q2, atol=1e-5))

    def test_deterministic_forward(self):
        """Same input should produce same output (eval mode)."""
        obs = torch.randn(1, self.config.obs_dim)
        q1 = self.net(obs)
        q2 = self.net(obs)
        torch.testing.assert_close(q1, q2)

    def test_gradients_flow(self):
        """Verify gradients flow through the network."""
        self.net.train()
        obs = torch.randn(2, self.config.obs_dim)
        q = self.net(obs)
        loss = q.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.net.parameters()
        )
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
