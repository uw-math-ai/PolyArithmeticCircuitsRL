"""Tests for SAC components."""

from collections import deque

import torch

from src.config import Config
from src.algorithms.sac import (
    RawStep,
    SACActor,
    StratifiedReplayBuffer,
    build_n_step_transition,
)
from src.environment.circuit_game import CircuitGame
from src.environment.fast_polynomial import FastPoly


class TestSACActor:
    def test_masked_logits(self):
        config = Config(
            n_variables=2,
            mod=5,
            max_complexity=4,
            hidden_dim=32,
            embedding_dim=32,
            num_gnn_layers=2,
        )
        actor = SACActor(config)
        game = CircuitGame(config)
        target = FastPoly.variable(0, 2, config.effective_max_degree, config.mod)
        obs = game.reset(target)

        logits = actor(obs)
        assert logits.shape[-1] == config.max_actions
        invalid = logits[0, ~obs["mask"]]
        if invalid.numel() > 0:
            assert torch.isinf(invalid).all()


class TestNStep:
    def test_build_n_step_transition(self):
        obs0 = {"target": torch.tensor([0.0]), "mask": torch.tensor([True]), "graph": {}}
        obs1 = {"target": torch.tensor([1.0]), "mask": torch.tensor([True]), "graph": {}}
        obs2 = {"target": torch.tensor([2.0]), "mask": torch.tensor([True]), "graph": {}}
        obs3 = {"target": torch.tensor([3.0]), "mask": torch.tensor([True]), "graph": {}}

        gamma = 0.9
        raw = deque(
            [
                RawStep(obs=obs0, action=1, reward=1.0, next_obs=obs1, done=False),
                RawStep(obs=obs1, action=2, reward=2.0, next_obs=obs2, done=False),
                RawStep(obs=obs2, action=3, reward=3.0, next_obs=obs3, done=False),
            ]
        )

        obs, action, reward, next_obs, done, discount = build_n_step_transition(
            raw, n_step=3, gamma=gamma
        )
        expected_reward = 1.0 + gamma * 2.0 + (gamma**2) * 3.0
        assert action == 1
        assert abs(reward - expected_reward) < 1e-6
        assert next_obs["target"].item() == 3.0
        assert not done
        assert abs(discount - gamma**3) < 1e-6


class TestStratifiedReplay:
    def _dummy_obs(self):
        return {
            "graph": {},
            "target": torch.tensor([0.0]),
            "mask": torch.tensor([True, False, True]),
        }

    def test_sampling_returns_batch(self):
        buffer = StratifiedReplayBuffer(capacity=64, recent_window=16)
        obs = self._dummy_obs()
        for i in range(40):
            buffer.add(
                obs=obs,
                action=0,
                reward=1.0,
                next_obs=obs,
                done=False,
                discount=0.99,
                complexity=3 if i % 2 == 0 else 2,
                episode_success=(i % 3 == 0),
            )

        batch = buffer.sample(
            batch_size=16,
            current_complexity=3,
            current_fraction=0.5,
            success_fraction=0.2,
            recent_fraction=0.2,
        )
        assert len(batch) == 16
        assert all(item is not None for item in batch)
        assert any(item.complexity == 3 for item in batch)
        assert any(item.episode_success for item in batch)
