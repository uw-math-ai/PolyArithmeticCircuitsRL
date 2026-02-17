"""Tests for the game board generator."""

import pytest

from src.config import Config
from src.environment.fast_polynomial import FastPoly
from src.game_board.generator import (
    build_game_board,
    find_interesting_targets,
    sample_target,
    generate_random_circuit,
)


class TestBuildGameBoard:
    def setup_method(self):
        self.config = Config(n_variables=2, mod=5, max_complexity=3)
        self.n_vars = 2
        self.mod = 5
        self.max_deg = self.config.effective_max_degree

    def test_base_nodes(self):
        board = build_game_board(self.config, complexity=0)
        # Should contain x0, x1, 1
        assert len(board) == 3
        for entry in board.values():
            assert entry["step"] == 0

    def test_complexity_1(self):
        board = build_game_board(self.config, complexity=1)
        step_0 = [e for e in board.values() if e["step"] == 0]
        step_1 = [e for e in board.values() if e["step"] == 1]
        assert len(step_0) == 3
        assert len(step_1) > 0

    def test_no_duplicate_keys(self):
        board = build_game_board(self.config, complexity=2)
        keys = list(board.keys())
        assert len(keys) == len(set(keys))

    def test_known_polynomial_reachable(self):
        board = build_game_board(self.config, complexity=1)
        # x0 + x1 should be reachable at step 1
        x0 = FastPoly.variable(0, self.n_vars, self.max_deg, self.mod)
        x1 = FastPoly.variable(1, self.n_vars, self.max_deg, self.mod)
        target = x0 + x1
        key = target.canonical_key()
        assert key in board
        assert board[key]["step"] == 1

    def test_board_grows_with_complexity(self):
        b1 = build_game_board(self.config, complexity=1)
        b2 = build_game_board(self.config, complexity=2)
        assert len(b2) >= len(b1)


class TestInterestingTargets:
    def test_finds_multi_path_targets(self):
        config = Config(n_variables=2, mod=5, max_complexity=3)
        board = build_game_board(config, complexity=3)
        interesting = find_interesting_targets(board, min_paths=2)
        assert len(interesting) > 0
        for entry in interesting:
            assert entry["paths"] >= 2

    def test_sorted_by_step(self):
        config = Config(n_variables=2, mod=5, max_complexity=3)
        board = build_game_board(config, complexity=3)
        interesting = find_interesting_targets(board)
        for i in range(1, len(interesting)):
            assert interesting[i]["step"] >= interesting[i - 1]["step"]


class TestSampleTarget:
    def test_returns_valid_polynomial(self):
        config = Config(n_variables=2, mod=5, max_complexity=3)
        poly, steps = sample_target(config, complexity=2)
        assert poly is not None
        assert isinstance(poly, FastPoly)
        assert 0 < steps <= 2

    def test_deterministic_with_board(self):
        config = Config(n_variables=2, mod=5, max_complexity=3)
        board = build_game_board(config, complexity=2)
        poly, steps = sample_target(config, complexity=2, board=board)
        assert poly is not None
        assert isinstance(poly, FastPoly)


class TestGenerateRandomCircuit:
    def test_returns_polynomial_and_actions(self):
        config = Config(n_variables=2, mod=5, max_complexity=4)
        poly, actions = generate_random_circuit(config, complexity=3)
        assert poly is not None
        assert isinstance(poly, FastPoly)
        assert len(actions) == 3

    def test_actions_are_valid(self):
        config = Config(n_variables=2, mod=5, max_complexity=4)
        poly, actions = generate_random_circuit(config, complexity=3)
        n_base = config.n_variables + 1
        for step, (op, i, j) in enumerate(actions):
            assert op in (0, 1)
            num_nodes = n_base + step
            assert 0 <= i < num_nodes
            assert i <= j < num_nodes

    def test_multiple_random_circuits(self):
        config = Config(n_variables=2, mod=5, max_complexity=6)
        for _ in range(50):
            poly, actions = generate_random_circuit(config, complexity=4)
            assert poly is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
