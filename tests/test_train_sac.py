"""Smoke tests for the discrete SAC trainer on the decomposition environment."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.config import DecompEnvConfig, FactorizerConfig
from decomp_rl.decomp_env import DecompEnv
from decomp_rl.factor_fp import FiniteFieldFactorizer
from decomp_rl.factor_library import FactorizableLibrary
from decomp_rl.model import TorchPolicyValueNetwork
from decomp_rl.polynomial import SparsePolynomial
from decomp_rl.train_sac import (
    ReplayBuffer,
    SACConfig,
    SACTransition,
    collect_episode,
    train_sac,
)

PRIME = 3
VARS = ("x", "y")


def _build_env_and_actor():
    library = FactorizableLibrary(prime=PRIME, variables=VARS)
    factorizer = FiniteFieldFactorizer(FactorizerConfig(), library=library)
    env = DecompEnv(
        config=DecompEnvConfig(),
        factorizer=factorizer,
        baseline_model=BaselineCostModel(),
        library=library,
    )
    return env, TorchPolicyValueNetwork()


def _xy_plus_x_plus_y() -> SparsePolynomial:
    return SparsePolynomial(PRIME, VARS, ((1, (1, 1)), (1, (1, 0)), (1, (0, 1))))


def _square_target() -> SparsePolynomial:
    return SparsePolynomial(PRIME, VARS, ((1, (2, 0)), (2, (1, 1)), (1, (0, 2))))


def test_collect_episode_builds_linked_transitions():
    env, actor = _build_env_and_actor()
    transitions, stats = collect_episode(
        env, actor, _xy_plus_x_plus_y(),
        SACConfig(max_episode_steps=8, candidates_per_step=8),
    )
    assert stats["length"] == len(transitions)
    for i, tr in enumerate(transitions):
        assert 0 <= tr.chosen_index < len(tr.candidates)
        # Only the final transition is terminal; earlier ones link to a next state.
        if i < len(transitions) - 1:
            assert tr.done is False
            assert tr.next_target is not None
            assert len(tr.next_candidates) > 0
    if transitions:
        assert transitions[-1].done is True
        assert transitions[-1].next_target is None


def test_replay_buffer_capacity_and_sample():
    import random
    buf = ReplayBuffer(capacity=3, rng=random.Random(0))
    poly = _xy_plus_x_plus_y()
    for _ in range(5):
        buf.add(SACTransition(poly, (), 0, 1.0, True, None, ()))
    assert len(buf) == 3  # bounded by capacity
    assert len(buf.sample(10)) == 3  # never more than available


def test_train_sac_smoke_keeps_parameters_finite():
    env, actor = _build_env_and_actor()
    config = SACConfig(
        rollouts_per_update=2,
        candidates_per_step=8,
        max_episode_steps=6,
        gradient_steps=2,
        batch_size=8,
        learning_starts=1,
        seed=42,
    )
    metrics = train_sac([_xy_plus_x_plus_y(), _square_target()], actor, env, config, iterations=2)
    assert len(metrics) == 2
    for param in actor.parameters():
        assert torch.isfinite(param).all(), "SAC produced non-finite actor parameters"
    assert metrics[-1].buffer_size > 0


def test_train_sac_fixed_alpha_runs():
    env, actor = _build_env_and_actor()
    config = SACConfig(
        rollouts_per_update=1,
        candidates_per_step=6,
        max_episode_steps=4,
        gradient_steps=1,
        batch_size=4,
        learning_starts=1,
        autotune_alpha=False,
        alpha_init=0.1,
        seed=7,
    )
    metrics = train_sac([_xy_plus_x_plus_y()], actor, env, config, iterations=1)
    assert len(metrics) == 1
    assert metrics[-1].alpha == pytest.approx(0.1, abs=1e-6)
