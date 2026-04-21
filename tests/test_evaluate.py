import os
import tempfile
import unittest
import warnings

import torch

from poly_circuit_rl.config import Config
from poly_circuit_rl.rl.agent import DQNAgent
from scripts.evaluate import config_for_evaluation


class TestEvaluateScript(unittest.TestCase):
    def test_config_for_evaluation_uses_checkpoint_config(self):
        config = Config(
            n_vars=2,
            max_ops=3,
            L=10,
            max_nodes=10,
            m=6,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_pos=4,
        )
        agent = DQNAgent(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            agent.save(path)

            loaded = config_for_evaluation(path)
            self.assertEqual(loaded, config)

            overridden = config_for_evaluation(path, max_ops_override=5)
            self.assertEqual(overridden.max_ops, 5)
            self.assertEqual(overridden.L, config.L)
            self.assertEqual(overridden.m, config.m)

    def test_config_for_evaluation_warns_for_legacy_checkpoint(self):
        config = Config(
            n_vars=2,
            m=8,
            L=8,
            max_nodes=8,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_pos=4,
        )
        agent = DQNAgent(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.pt")
            torch.save(
                {
                    "q_network": agent.q_network.state_dict(),
                    "target_network": agent.target_network.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "total_steps": 0,
                },
                path,
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = config_for_evaluation(path)

            self.assertEqual(loaded, Config())
            self.assertTrue(any("saved config" in str(w.message) for w in caught))

    def test_config_for_evaluation_preserves_checkpoint_architecture(self):
        config = Config(
            n_vars=2,
            max_ops=3,
            L=10,
            max_nodes=10,
            m=6,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_pos=4,
        )
        agent = DQNAgent(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            agent.save(path)

            loaded_cfg = config_for_evaluation(path)
            eval_agent = DQNAgent(loaded_cfg)
            # Should not raise due to architecture mismatch.
            eval_agent.load(path)


if __name__ == "__main__":
    unittest.main()
