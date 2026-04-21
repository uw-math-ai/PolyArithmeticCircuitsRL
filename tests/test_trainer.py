import random
import unittest

import numpy as np

from poly_circuit_rl.config import Config
from poly_circuit_rl.core.action_codec import ACTION_ADD, ACTION_SET_OUTPUT, encode_action
from poly_circuit_rl.core.factor_library import FactorLibrary
from poly_circuit_rl.core.poly import add, make_var, poly_hashkey
from poly_circuit_rl.env.circuit_env import PolyCircuitEnv
from poly_circuit_rl.rl.trainer import (
    evaluate,
    maybe_sample_target_poly,
    should_mix_interesting_targets,
)


class _DummySampler:
    def __init__(self):
        self.calls = []

    def sample(self, rng, max_ops):
        goal = f"interesting-{len(self.calls)}"
        self.calls.append((goal, max_ops))
        return goal, {"source": "interesting"}


class _FakeAgent:
    def __init__(self):
        self.total_steps = 123

    def select_action(self, obs, mask, deterministic=False):
        valid = np.where(mask > 0)[0]
        return int(valid[0])


class _FakeEnv:
    def __init__(self, config: Config):
        self.config = config
        self.reset_options = []

    def reset(self, options=None):
        options = dict(options or {})
        self.reset_options.append(options)
        return {
            "obs": np.zeros(self.config.obs_dim, dtype=np.float32),
            "action_mask": np.ones(self.config.action_dim, dtype=np.int8),
        }, {}

    def step(self, action):
        return {
            "obs": np.zeros(self.config.obs_dim, dtype=np.float32),
            "action_mask": np.ones(self.config.action_dim, dtype=np.int8),
        }, 0.0, False, True, {"solved": False}


class _FixedSampler:
    def __init__(self, target):
        self.target = target

    def sample(self, rng, max_ops):
        return self.target, {"source": "fixed"}


class _ScriptedSolveAgent:
    def __init__(self, config: Config):
        self.config = config
        self.total_steps = 0

    def select_action(self, obs, mask, deterministic=False):
        set_action = encode_action(ACTION_SET_OUTPUT, 3, None, self.config.L)
        if set_action < len(mask) and mask[set_action] == 1:
            return set_action
        return encode_action(ACTION_ADD, 0, 1, self.config.L)


class TestTrainerSampling(unittest.TestCase):
    def test_should_mix_interesting_targets_only_from_level_one(self):
        config = Config(interesting_ratio=0.7)
        self.assertFalse(should_mix_interesting_targets(0, config))
        self.assertTrue(should_mix_interesting_targets(1, config))
        self.assertFalse(should_mix_interesting_targets(2, Config(interesting_ratio=0.0)))

    def test_maybe_sample_target_poly_respects_level_gate(self):
        config = Config(interesting_ratio=1.0)
        sampler = _DummySampler()
        rng = random.Random(0)

        target = maybe_sample_target_poly(0, 2, config, rng, sampler)
        self.assertIsNone(target)
        self.assertEqual(sampler.calls, [])

        target = maybe_sample_target_poly(1, 2, config, rng, sampler)
        self.assertEqual(target, "interesting-0")
        self.assertEqual(sampler.calls, [("interesting-0", 2)])

    def test_evaluate_uses_same_interesting_mix_from_level_one(self):
        config = Config(n_vars=1, m=4, L=4, max_nodes=4, interesting_ratio=1.0)
        env = _FakeEnv(config)
        agent = _FakeAgent()
        sampler = _DummySampler()

        evaluate(
            env,
            agent,
            max_ops=2,
            num_episodes=3,
            level_idx=1,
            interesting_sampler=sampler,
            target_rng=random.Random(0),
        )

        self.assertEqual(len(env.reset_options), 3)
        self.assertTrue(all(opts["target_poly"].startswith("interesting-") for opts in env.reset_options))
        self.assertEqual(agent.total_steps, 123)

    def test_evaluate_keeps_level_zero_random_only(self):
        config = Config(n_vars=1, m=4, L=4, max_nodes=4, interesting_ratio=1.0)
        env = _FakeEnv(config)
        agent = _FakeAgent()
        sampler = _DummySampler()

        evaluate(
            env,
            agent,
            max_ops=1,
            num_episodes=2,
            level_idx=0,
            interesting_sampler=sampler,
            target_rng=random.Random(0),
        )

        self.assertEqual(len(env.reset_options), 2)
        self.assertTrue(all("target_poly" not in opts for opts in env.reset_options))
        self.assertEqual(sampler.calls, [])

    def test_evaluate_does_not_mutate_factor_library_state(self):
        config = Config(
            n_vars=2,
            m=4,
            L=8,
            max_nodes=8,
            max_ops=2,
            interesting_ratio=1.0,
            factor_library_enabled=True,
            seed=42,
        )
        factor_library = FactorLibrary(n_vars=2)
        env = PolyCircuitEnv(config, factor_library=factor_library.frozen_view())
        agent = _ScriptedSolveAgent(config)
        target = add(make_var(2, 0), make_var(2, 1))
        sampler = _FixedSampler(target)

        size_before = len(factor_library)
        evaluate(
            env,
            agent,
            max_ops=2,
            num_episodes=3,
            level_idx=1,
            interesting_sampler=sampler,
            target_rng=random.Random(0),
        )
        self.assertEqual(len(factor_library), size_before)

    def test_evaluate_with_held_out_targets_is_deterministic(self):
        config = Config(n_vars=2, m=4, L=8, max_nodes=8, max_ops=2, seed=42)
        env = PolyCircuitEnv(config)
        agent = _ScriptedSolveAgent(config)
        target = add(make_var(2, 0), make_var(2, 1))
        held_out = [target, target, target]

        ev1 = evaluate(
            env,
            agent,
            max_ops=2,
            num_episodes=3,
            level_idx=0,
            eval_targets=held_out,
        )
        ev2 = evaluate(
            env,
            agent,
            max_ops=2,
            num_episodes=3,
            level_idx=0,
            eval_targets=held_out,
        )
        self.assertEqual(ev1["success_rate"], ev2["success_rate"])

    def test_evaluate_reports_nonnegative_gap_to_optimal(self):
        config = Config(n_vars=2, m=4, L=8, max_nodes=8, max_ops=2, seed=42)
        env = PolyCircuitEnv(config)
        agent = _ScriptedSolveAgent(config)
        target = add(make_var(2, 0), make_var(2, 1))
        optimal_ops = {poly_hashkey(target): 1}

        ev = evaluate(
            env,
            agent,
            max_ops=2,
            num_episodes=3,
            level_idx=0,
            eval_targets=[target, target, target],
            optimal_ops=optimal_ops,
        )
        self.assertIsNotNone(ev["mean_gap_to_optimal"])
        self.assertGreaterEqual(ev["mean_gap_to_optimal"], 0.0)


if __name__ == "__main__":
    unittest.main()
