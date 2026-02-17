"""Tests for neural network models."""

import pytest
import torch

from src.config import Config
from src.models.gnn_encoder import CircuitGNN
from src.models.policy_value_net import PolicyValueNet
from src.environment.circuit_game import CircuitGame
from src.environment.fast_polynomial import FastPoly


class TestCircuitGNN:
    def test_forward_shape(self):
        gnn = CircuitGNN(input_dim=4, hidden_dim=32, output_dim=16, num_layers=3)
        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = gnn(x, edge_index, num_nodes_actual=4)
        assert out.shape == (1, 16)

    def test_no_edges(self):
        gnn = CircuitGNN(input_dim=4, hidden_dim=32, output_dim=16)
        x = torch.randn(3, 4)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        out = gnn(x, edge_index, num_nodes_actual=3)
        assert out.shape == (1, 16)

    def test_gradient_flow(self):
        gnn = CircuitGNN(input_dim=4, hidden_dim=32, output_dim=16)
        x = torch.randn(5, 4, requires_grad=True)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = gnn(x, edge_index, num_nodes_actual=3)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


def _make_target(config):
    """Helper: create x0 + x1 as a FastPoly."""
    n = config.n_variables
    d = config.effective_max_degree
    m = config.mod
    return FastPoly.variable(0, n, d, m) + FastPoly.variable(1, n, d, m)


class TestPolicyValueNet:
    def setup_method(self):
        self.config = Config(
            n_variables=2, mod=5, max_complexity=4, max_steps=6,
            hidden_dim=32, embedding_dim=32, num_gnn_layers=2
        )
        self.model = PolicyValueNet(self.config)
        self.game = CircuitGame(self.config)

    def test_forward(self):
        target = _make_target(self.config)
        obs = self.game.reset(target)
        logits, value = self.model(obs)
        assert logits.shape[-1] == self.config.max_actions
        assert value.shape[-1] == 1 or value.dim() == 1

    def test_masked_logits(self):
        target = _make_target(self.config)
        obs = self.game.reset(target)
        logits, _ = self.model(obs)
        mask = obs["mask"]
        invalid_logits = logits[0, ~mask]
        if invalid_logits.numel() > 0:
            assert (invalid_logits == float("-inf")).all()

    def test_get_action_and_value(self):
        target = _make_target(self.config)
        obs = self.game.reset(target)
        action, log_prob, entropy, value = self.model.get_action_and_value(obs)
        assert action.dim() == 1 or action.dim() == 0
        assert log_prob.dim() <= 1
        assert obs["mask"][action.item()]

    def test_get_policy_and_value(self):
        target = _make_target(self.config)
        obs = self.game.reset(target)
        probs, value = self.model.get_policy_and_value(obs)
        assert probs.shape[0] == self.config.max_actions
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_gradient_flow_through_game(self):
        target = _make_target(self.config)
        obs = self.game.reset(target)
        logits, value = self.model(obs)
        loss = logits.sum() + value.sum()
        loss.backward()
        for param in self.model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
