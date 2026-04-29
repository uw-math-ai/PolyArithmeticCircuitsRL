"""Tests for baseline scripts and shared baseline reward helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.environment.fast_polynomial import FastPoly
from src.game_board.on_path import OnPathTargetContext

from scripts.baseline_beam_search import solve_beam_search
from scripts.baseline_greedy import solve_greedy
from scripts.baseline_reward import BaselineRewardEvaluator, initial_nodes
from scripts.baseline_onpath_mcts import solve_onpath_mcts
from scripts.baseline_uniform_mcts import solve_uniform_mcts


@pytest.fixture
def small_config():
    """2-var, mod 5, max_complexity 4, max_degree 4, max_steps 6."""
    return Config(
        n_variables=2,
        mod=5,
        max_complexity=4,
        max_degree=4,
        max_steps=6,
        reward_mode="clean_onpath",
        on_path_phi_mode="depth_weighted",
        on_path_route_consistency_mode="best_route_phi",
        graph_onpath_shaping_coeff=3.0,
        on_route_bonus_coeff=3.0,
    )


def _flatten(poly: FastPoly) -> np.ndarray:
    return poly.coeffs.flatten().astype(np.int32)


def _context_for(
    target: FastPoly,
    route_polys: list[FastPoly],
    route_steps: list[int],
) -> OnPathTargetContext:
    route_masks = np.ones(len(route_polys), dtype=np.uint32)
    return OnPathTargetContext(
        target_id=0,
        target_poly=target,
        target_board_step=max(route_steps) if route_steps else 0,
        on_path_ids=np.arange(len(route_polys), dtype=np.int32),
        on_path_coeffs=np.stack([_flatten(p) for p in route_polys], axis=0),
        on_path_hashes=np.zeros(len(route_polys), dtype=np.uint64),
        on_path_steps=np.asarray(route_steps, dtype=np.int16),
        on_path_route_masks=route_masks,
    )


class TestGreedy:
    def test_greedy_solves_one_step_target(self, small_config):
        """Target = x0 + x1. Greedy must succeed in exactly 1 action."""
        x0 = FastPoly.variable(0, 2, small_config.effective_max_degree, 5)
        x1 = FastPoly.variable(1, 2, small_config.effective_max_degree, 5)
        target = x0 + x1
        r = solve_greedy(_flatten(target), small_config)
        assert r["success"] is True
        assert r["num_steps"] == 1

    def test_greedy_solves_two_step_target(self, small_config):
        """Target = (x0 + x1) * x0. Greedy might or might not solve via this
        exact path, but it should finish in <= max_steps."""
        x0 = FastPoly.variable(0, 2, small_config.effective_max_degree, 5)
        x1 = FastPoly.variable(1, 2, small_config.effective_max_degree, 5)
        target = (x0 + x1) * x0
        r = solve_greedy(_flatten(target), small_config)
        # Don't assert success — greedy can mispredict on harder targets. Just
        # ensure it terminates without crashing and reports valid metrics.
        assert isinstance(r["success"], bool)
        assert 0 <= r["num_steps"] <= small_config.max_steps
        assert isinstance(r["env_reward"], float)


class TestUniformMCTS:
    def test_solves_c1_at_default_budget(self, small_config):
        """At sims=32 (deployment default), uniform MCTS solves x0+x1 reliably.

        With value-aware tie-break in action selection, the success child's
        reward (+9.9) dominates other children's rollout returns even when all
        children have 1 visit, so the success action gets picked.
        """
        x0 = FastPoly.variable(0, 2, small_config.effective_max_degree, 5)
        x1 = FastPoly.variable(1, 2, small_config.effective_max_degree, 5)
        target = x0 + x1
        succ = 0
        for seed in range(10):
            r = solve_uniform_mcts(
                _flatten(target), small_config,
                mcts_simulations=32, seed=seed,
            )
            succ += int(r["success"])
        assert succ >= 8, f"Expected >=8/10 successes, got {succ}"

    def test_reports_both_returns(self, small_config):
        """Successful rollout populates both env_reward and success_only_return."""
        x0 = FastPoly.variable(0, 2, small_config.effective_max_degree, 5)
        x1 = FastPoly.variable(1, 2, small_config.effective_max_degree, 5)
        target = x0 + x1
        r = solve_uniform_mcts(
            _flatten(target), small_config, mcts_simulations=16, seed=0,
        )
        assert "env_reward" in r
        assert "success_only_return" in r
        if r["success"]:
            # Successful trajectories should land near +success_reward,
            # minus a small step_penalty.
            assert r["success_only_return"] == pytest.approx(small_config.success_reward)


class TestOnPathBaselineReward:
    def test_on_path_reward_fires_once(self, small_config):
        """Building a cached on-path node increases phi only once."""
        x0 = FastPoly.variable(0, 2, small_config.effective_max_degree, 5)
        one = FastPoly.constant(1, 2, small_config.effective_max_degree, 5)
        target = (x0 + one) * x0
        intermediate = x0 + one
        context = _context_for(
            target,
            [intermediate, target],
            [1, 2],
        )

        evaluator = BaselineRewardEvaluator(small_config, context)
        nodes = initial_nodes(small_config)
        state = evaluator.initial_state()

        first = evaluator.step_reward(nodes, intermediate, state)
        second = evaluator.step_reward(
            nodes + [intermediate],
            intermediate,
            first.next_state,
        )

        assert first.on_path_hit is True
        assert second.on_path_hit is False
        assert first.phi_after > first.phi_before
        assert second.phi_after == pytest.approx(first.phi_after)
        assert first.reward > second.reward

    def test_best_route_phi_uses_coherent_route(self, small_config):
        x0 = FastPoly.variable(0, 2, small_config.effective_max_degree, 5)
        x1 = FastPoly.variable(1, 2, small_config.effective_max_degree, 5)
        target = x0 + x1
        context = OnPathTargetContext(
            target_id=0,
            target_poly=target,
            target_board_step=2,
            on_path_ids=np.array([10, 11], dtype=np.int32),
            on_path_coeffs=np.stack([_flatten(x0), _flatten(x1)], axis=0),
            on_path_hashes=np.zeros(2, dtype=np.uint64),
            on_path_steps=np.array([1, 2], dtype=np.int16),
            on_path_route_masks=np.array([1, 2], dtype=np.uint32),
        )
        evaluator = BaselineRewardEvaluator(small_config, context)
        state = evaluator.initial_state()
        hit_x0, state = evaluator._record_on_path_hit(x0, state)
        hit_x1, state = evaluator._record_on_path_hit(x1, state)

        assert hit_x0 is True
        assert hit_x1 is True
        assert evaluator.on_path_phi(state) == pytest.approx(1.0)


class TestOnPathMCTS:
    def test_solves_c1_at_default_budget(self, small_config):
        """On-path MCTS should preserve one-step strength."""
        x0 = FastPoly.variable(0, 2, small_config.effective_max_degree, 5)
        x1 = FastPoly.variable(1, 2, small_config.effective_max_degree, 5)
        target = x0 + x1
        context = _context_for(target, [target], [1])
        succ = 0
        for seed in range(10):
            r = solve_onpath_mcts(
                context, small_config,
                mcts_simulations=32, seed=seed,
            )
            succ += int(r["success"])
        assert succ >= 8, f"Expected >=8/10 successes, got {succ}"


class TestBeamSearch:
    def test_terminates_with_valid_metrics(self, small_config):
        x0 = FastPoly.variable(0, 2, small_config.effective_max_degree, 5)
        x1 = FastPoly.variable(1, 2, small_config.effective_max_degree, 5)
        one = FastPoly.constant(1, 2, small_config.effective_max_degree, 5)
        target = x0 + x1 + one
        context = _context_for(target, [x0 + x1, target], [1, 2])

        r = solve_beam_search(
            _flatten(target),
            small_config,
            beam_width=8,
            context=context,
        )

        assert isinstance(r["success"], bool)
        assert 0 <= r["num_steps"] <= small_config.max_steps
        assert len(r["actions"]) == r["num_steps"]
        assert isinstance(r["env_reward"], float)


class TestTargetCacheBuilder:
    def test_seed_deterministic(self, tmp_path):
        """Same seed produces identical target_cache content."""
        from scripts import build_baseline_target_cache as bcache

        out1 = tmp_path / "cache1.npz"
        out2 = tmp_path / "cache2.npz"

        # Mock argv twice with the same seed.
        argv = [
            "build_baseline_target_cache.py",
            "--complexities", "2", "3",
            "--num-trials", "5",
            "--seed", "123",
            "--n-variables", "2",
            "--mod", "5",
            "--max-degree", "4",
            "--max-steps", "8",
            "--out", str(out1),
        ]
        sys_argv_orig = sys.argv
        try:
            sys.argv = argv
            bcache.main()
            sys.argv = argv[:-1] + [str(out2)]
            bcache.main()
        finally:
            sys.argv = sys_argv_orig

        d1 = np.load(out1, allow_pickle=True)
        d2 = np.load(out2, allow_pickle=True)
        np.testing.assert_array_equal(d1["target_coeffs"], d2["target_coeffs"])
        np.testing.assert_array_equal(
            d1["generated_complexity"], d2["generated_complexity"]
        )

    def test_cache_has_required_fields(self, tmp_path):
        from scripts import build_baseline_target_cache as bcache

        out = tmp_path / "cache.npz"
        argv = [
            "build_baseline_target_cache.py",
            "--complexities", "2",
            "--num-trials", "3",
            "--seed", "0",
            "--n-variables", "2",
            "--mod", "5",
            "--max-degree", "4",
            "--max-steps", "8",
            "--out", str(out),
        ]
        sys_argv_orig = sys.argv
        try:
            sys.argv = argv
            bcache.main()
        finally:
            sys.argv = sys_argv_orig

        d = np.load(out, allow_pickle=True)
        for key in (
            "target_coeffs", "canonical_keys", "generated_complexity",
            "true_min_complexity", "config_n_variables", "config_mod",
            "config_max_degree", "config_max_steps", "config_target_size", "seed",
        ):
            assert key in d.files, f"missing field {key}"
        assert d["target_coeffs"].shape[0] >= 1
        # No JSON provided => true_min_complexity all -1.
        assert int(np.min(d["true_min_complexity"])) == -1
