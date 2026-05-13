"""Smoke tests for the PPO trainer on the decomposition environment."""

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
from decomp_rl.train_optional_rl import OptionalRLConfig, run_optional_rl
from decomp_rl.train_ppo import (
    PPOConfig,
    collect_episode,
    compute_gae,
    train_ppo,
)


PRIME = 3
VARS = ("x", "y")


def _build_env_and_model():
    library = FactorizableLibrary(prime=PRIME, variables=VARS)
    factorizer = FiniteFieldFactorizer(FactorizerConfig(), library=library)
    env = DecompEnv(
        config=DecompEnvConfig(),
        factorizer=factorizer,
        baseline_model=BaselineCostModel(),
        library=library,
    )
    network = TorchPolicyValueNetwork()
    return env, network


def _xy_plus_x_plus_y() -> SparsePolynomial:
    return SparsePolynomial(
        PRIME, VARS, ((1, (1, 1)), (1, (1, 0)), (1, (0, 1)))
    )


def _square_target() -> SparsePolynomial:
    # x^2 + 2xy + y^2 = (x+y)^2 over F_3
    return SparsePolynomial(
        PRIME, VARS, ((1, (2, 0)), (2, (1, 1)), (1, (0, 2)))
    )


def test_collect_episode_runs_and_records_log_probs():
    env, network = _build_env_and_model()
    transitions = collect_episode(
        env, network, _xy_plus_x_plus_y(), PPOConfig(max_episode_steps=8, candidates_per_step=8)
    )
    for tr in transitions:
        assert 0 <= tr.chosen_index < len(tr.candidates)
        assert tr.log_prob <= 0.0  # log of a probability
        assert isinstance(tr.value, float)


def test_compute_gae_shapes_and_finite():
    env, network = _build_env_and_model()
    transitions = collect_episode(
        env, network, _square_target(), PPOConfig(max_episode_steps=8, candidates_per_step=8)
    )
    if not transitions:
        pytest.skip("environment produced no policy steps for this target")
    advs, rets = compute_gae(transitions, gamma=0.99, lam=0.95)
    assert len(advs) == len(transitions) == len(rets)
    for a, r in zip(advs, rets):
        assert a == a  # not NaN
        assert r == r


def test_train_ppo_smoke_keeps_parameters_finite():
    env, network = _build_env_and_model()
    config = PPOConfig(
        rollouts_per_update=2,
        candidates_per_step=8,
        update_epochs=1,
        minibatch_size=8,
        max_episode_steps=6,
        seed=42,
    )
    metrics = train_ppo([_xy_plus_x_plus_y(), _square_target()], network, env, config, iterations=2)
    assert len(metrics) == 2
    for param in network.parameters():
        assert torch.isfinite(param).all(), "PPO produced non-finite parameters"


def test_run_optional_rl_delegates_to_ppo():
    env, network = _build_env_and_model()
    metrics = run_optional_rl(
        targets=[_xy_plus_x_plus_y()],
        env=env,
        model=network,
        config=OptionalRLConfig(iterations=1, rollouts_per_update=1, seed=7),
    )
    assert len(metrics) == 1


def test_run_optional_rl_rejects_non_ppo_algorithm():
    env, network = _build_env_and_model()
    with pytest.raises(NotImplementedError):
        run_optional_rl(
            targets=[_xy_plus_x_plus_y()],
            env=env,
            model=network,
            config=OptionalRLConfig(algorithm="sac"),
        )


def test_train_ppo_mcts_smoke_keeps_parameters_finite():
    env, network = _build_env_and_model()
    config = PPOConfig(
        rollouts_per_update=1,
        candidates_per_step=4,
        update_epochs=1,
        minibatch_size=4,
        max_episode_steps=4,
        use_mcts=True,
        mcts_simulations=4,
        mcts_max_depth=3,
        mcts_distill_coef=1.0,
        seed=11,
    )
    metrics = train_ppo([_xy_plus_x_plus_y()], network, env, config, iterations=1)
    assert len(metrics) == 1
    for param in network.parameters():
        assert torch.isfinite(param).all(), "PPO+MCTS produced non-finite parameters"


def test_collect_episode_mcts_records_visit_distribution():
    env, network = _build_env_and_model()
    transitions = collect_episode(
        env,
        network,
        _xy_plus_x_plus_y(),
        PPOConfig(
            max_episode_steps=4,
            candidates_per_step=4,
            use_mcts=True,
            mcts_simulations=4,
            mcts_max_depth=3,
        ),
    )
    if not transitions:
        pytest.skip("MCTS produced no policy steps for this target")
    for tr in transitions:
        assert tr.mcts_policy is not None
        assert len(tr.mcts_policy) == len(tr.candidates)
        s = sum(tr.mcts_policy)
        assert abs(s - 1.0) < 1e-5 or s == 0.0  # normalized or all-zero (no visits)
        assert 0 <= tr.chosen_index < len(tr.candidates)
