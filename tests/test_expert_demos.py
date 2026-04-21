"""Tests for expert demonstration generation."""

import unittest

from poly_circuit_rl.config import Config
from poly_circuit_rl.core.poly import add, mul, make_var, make_const
from poly_circuit_rl.core.factor_library import FactorLibrary
from poly_circuit_rl.env.circuit_env import PolyCircuitEnv


class TestExpertDemoGenerator(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            n_vars=2,
            max_ops=2,
            L=16,
            m=16,
            seed=42,
            factor_library_enabled=False,
            gen_max_graph_nodes=1000,
            gen_max_successors=20,
        )

    def test_demo_generation_simple(self):
        """Generate demo for a 1-op target (x0 + x1)."""
        from poly_circuit_rl.env.expert_demos import ExpertDemoGenerator

        env = PolyCircuitEnv(self.config)
        gen = ExpertDemoGenerator(self.config)
        gen.build_graph(max_steps=2)

        # x0 + x1 should be reachable in 1 op
        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        target = add(x0, x1)

        demo = gen.generate_demo(env, target, max_ops=2)
        self.assertIsNotNone(demo, "Should generate demo for simple target")
        self.assertTrue(len(demo) > 0, "Demo should have at least one transition")

    def test_demo_transitions_valid(self):
        """Demo transitions should have correct shapes and types."""
        from poly_circuit_rl.env.expert_demos import ExpertDemoGenerator

        env = PolyCircuitEnv(self.config)
        gen = ExpertDemoGenerator(self.config)
        gen.build_graph(max_steps=2)

        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        target = add(x0, x1)

        demo = gen.generate_demo(env, target, max_ops=2)
        if demo is None:
            self.skipTest("Demo generation returned None")

        for t in demo:
            self.assertEqual(t.obs.shape, (self.config.obs_dim,))
            self.assertEqual(t.next_obs.shape, (self.config.obs_dim,))
            self.assertEqual(t.action_mask.shape, (self.config.action_dim,))
            self.assertEqual(t.next_action_mask.shape, (self.config.action_dim,))
            self.assertIsInstance(t.action, int)
            self.assertIsInstance(t.reward, float)

    def test_demo_flag_set(self):
        """All demo transitions should have is_demo=True."""
        from poly_circuit_rl.env.expert_demos import ExpertDemoGenerator

        env = PolyCircuitEnv(self.config)
        gen = ExpertDemoGenerator(self.config)
        gen.build_graph(max_steps=2)

        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        target = add(x0, x1)

        demo = gen.generate_demo(env, target, max_ops=2)
        if demo is None:
            self.skipTest("Demo generation returned None")

        for t in demo:
            self.assertTrue(t.is_demo, "Demo transitions must have is_demo=True")

    def test_generate_demos_batch(self):
        """Batch demo generation should produce transitions."""
        from poly_circuit_rl.env.expert_demos import ExpertDemoGenerator

        env = PolyCircuitEnv(self.config)
        gen = ExpertDemoGenerator(self.config)

        demos = gen.generate_demos(
            env,
            num_demos=10,
            curriculum_levels=(1, 2),
        )
        # Should generate at least some transitions
        self.assertGreater(len(demos), 0, "Should generate at least some demo transitions")

    def test_demo_buffer_integration(self):
        """Demo transitions should be addable to replay buffer."""
        from poly_circuit_rl.env.expert_demos import ExpertDemoGenerator
        from poly_circuit_rl.rl.replay_buffer import HERReplayBuffer

        env = PolyCircuitEnv(self.config)
        gen = ExpertDemoGenerator(self.config)
        gen.build_graph(max_steps=2)

        x0 = make_var(2, 0)
        x1 = make_var(2, 1)
        target = add(x0, x1)

        demo = gen.generate_demo(env, target, max_ops=2)
        if demo is None:
            self.skipTest("Demo generation returned None")

        buffer = HERReplayBuffer(self.config)
        for t in demo:
            buffer.add(t)

        self.assertEqual(len(buffer), len(demo))

        # Sampling should work
        if len(buffer) >= self.config.batch_size:
            batch = buffer.sample(self.config.batch_size)
            self.assertEqual(batch["obs"].shape[0], self.config.batch_size)


if __name__ == "__main__":
    unittest.main()
