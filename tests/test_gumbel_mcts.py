import numpy as np
import pytest
import torch
import torch.nn as nn

from src.algorithms.alphazero import AlphaZeroTrainer
from src.algorithms.gumbel_mcts import masked_softmax, run_gumbel_mcts, visit_count_policy
from src.algorithms.ppo_mcts import PPOMCTSTrainer
from src.config import Config
from src.environment.action_space import encode_action
from src.environment.circuit_game import CircuitGame
from src.environment.fast_polynomial import FastPoly


def make_test_config(**overrides) -> Config:
    config = Config(
        n_variables=2,
        mod=5,
        max_complexity=2,
        max_steps=1,
        hidden_dim=16,
        embedding_dim=16,
        num_gnn_layers=1,
        search="gumbel",
        gumbel_num_simulations=8,
        gumbel_max_num_considered_actions=4,
        gumbel_scale=0.0,
        gumbel_c_visit=50.0,
        gumbel_c_scale=0.2,
        steps_per_update=2,
        ppo_epochs=1,
        ppo_minibatch_size=2,
        az_games_per_iter=1,
        az_batch_size=1,
        az_training_epochs=1,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def target_x0_plus_one(config: Config) -> FastPoly:
    return (
        FastPoly.variable(0, config.n_variables, config.effective_max_degree, config.mod)
        + FastPoly.constant(1, config.n_variables, config.effective_max_degree, config.mod)
    )


def winning_action(config: Config) -> int:
    constant_idx = config.n_variables
    return encode_action(0, 0, constant_idx, config.max_nodes)


def distractor_action(config: Config) -> int:
    return encode_action(1, 0, 1, config.max_nodes)


class FixedLogitModel(nn.Module):
    def __init__(self, config: Config, logits_by_action=None, value: float = 0.0):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        base_logits = torch.full((config.max_actions,), -4.0, dtype=torch.float32)
        for action, logit in (logits_by_action or {}).items():
            base_logits[action] = logit
        self.register_buffer("base_logits", base_logits)
        self.value = float(value)

    def forward(self, obs):
        mask = obs["mask"]
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        logits = self.base_logits.to(mask.device).unsqueeze(0).expand(mask.shape[0], -1).clone()
        logits = logits + self.anchor * 0.0
        logits = logits.masked_fill(~mask, float("-inf"))
        value = torch.full(
            (mask.shape[0],),
            self.value,
            dtype=torch.float32,
            device=mask.device,
        )
        value = value + self.anchor * 0.0
        return logits, value


def test_gumbel_never_selects_invalid_actions_and_weights_sum_to_one():
    config = make_test_config(gumbel_scale=1.0)
    game = CircuitGame(config)
    model = FixedLogitModel(config)
    obs = game.reset(target_x0_plus_one(config))
    invalid_mask = ~obs["mask"].cpu().numpy()

    for seed in range(8):
        output = run_gumbel_mcts(
            game,
            model,
            config,
            rng=np.random.default_rng(seed),
        )
        assert bool(obs["mask"][output.action])
        assert np.all(output.action_weights[invalid_mask] == 0.0)
        np.testing.assert_allclose(output.action_weights.sum(), 1.0, atol=1e-6)


def test_gumbel_can_choose_lower_prior_winning_action_over_distractor():
    config = make_test_config()
    game = CircuitGame(config)
    target = target_x0_plus_one(config)
    obs = game.reset(target)

    winner = winning_action(config)
    distractor = distractor_action(config)
    model = FixedLogitModel(
        config,
        logits_by_action={
            distractor: 4.0,
            winner: -1.0,
        },
    )

    output = run_gumbel_mcts(
        game,
        model,
        config,
        rng=np.random.default_rng(0),
    )
    prior = masked_softmax(model.base_logits.cpu().numpy(), obs["mask"].cpu().numpy())

    assert prior[winner] < prior[distractor]
    assert output.action == winner
    assert output.action_weights[winner] > output.action_weights[distractor]
    assert winner in output.considered_actions


def test_gumbel_search_is_reproducible_with_fixed_rng_seed():
    config = make_test_config(gumbel_scale=1.0)
    target = target_x0_plus_one(config)
    logits = {winning_action(config): 1.0, distractor_action(config): 1.0}
    model = FixedLogitModel(config, logits_by_action=logits)

    game_a = CircuitGame(config)
    game_a.reset(target)
    game_b = CircuitGame(config)
    game_b.reset(target)

    output_a = run_gumbel_mcts(
        game_a,
        model,
        config,
        rng=np.random.default_rng(123),
    )
    output_b = run_gumbel_mcts(
        game_b,
        model,
        config,
        rng=np.random.default_rng(123),
    )

    assert output_a.action == output_b.action
    np.testing.assert_allclose(output_a.action_weights, output_b.action_weights)
    np.testing.assert_array_equal(output_a.considered_actions, output_b.considered_actions)


def test_ppo_mcts_rollout_buffer_stores_gumbel_search_policy():
    config = make_test_config()
    model = FixedLogitModel(
        config,
        logits_by_action={
            distractor_action(config): 4.0,
            winning_action(config): -1.0,
        },
    )
    trainer = PPOMCTSTrainer(config, model)
    trainer._sample_target = lambda complexity: target_x0_plus_one(config)

    buffer, rollout_info = trainer.collect_rollouts()

    assert len(buffer) == config.steps_per_update
    assert buffer.steps[0].search_policy.shape == (config.max_actions,)
    np.testing.assert_allclose(buffer.steps[0].search_policy.sum(), 1.0, atol=1e-6)
    assert rollout_info["search_type"] == "gumbel"
    assert "gumbel_root_policy_entropy" in rollout_info


def test_alphazero_replay_buffer_uses_completed_q_policy_target():
    config = make_test_config()
    model = FixedLogitModel(
        config,
        logits_by_action={
            distractor_action(config): 4.0,
            winning_action(config): -1.0,
        },
    )
    trainer = AlphaZeroTrainer(config, model)
    trainer._sample_target = lambda complexity: target_x0_plus_one(config)

    play_info = trainer.generate_self_play_data()
    obs, policy_target, _ = trainer.replay_buffer.buffer[0]
    visit_target = visit_count_policy(
        trainer.mcts.last_search_output.root_visits,
        obs["mask"].cpu().numpy(),
    )

    np.testing.assert_allclose(policy_target.sum(), 1.0, atol=1e-6)
    assert not np.allclose(policy_target, visit_target)
    assert play_info["search_type"] == "gumbel"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])