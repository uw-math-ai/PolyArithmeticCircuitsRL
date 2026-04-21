import unittest

from poly_circuit_rl.config import Config
from poly_circuit_rl.rl.trainer import train


class TestConfigValidation(unittest.TestCase):
    def test_default_config_is_valid(self):
        cfg = Config()
        self.assertIsInstance(cfg, Config)

    def test_l_and_max_nodes_must_be_positive(self):
        with self.assertRaises(AssertionError):
            Config(L=0)
        with self.assertRaises(AssertionError):
            Config(max_nodes=0)

    def test_eval_norm_scale_must_be_positive(self):
        with self.assertRaises(AssertionError):
            Config(eval_norm_scale=0.0)

    def test_train_requires_positive_eval_every(self):
        cfg = Config(
            eval_every=0,
            total_steps=1,
            expert_demo_count=0,
            auto_interesting=False,
            factor_library_enabled=False,
            use_mcts=False,
        )
        with self.assertRaises(AssertionError):
            train(cfg)


if __name__ == "__main__":
    unittest.main()
