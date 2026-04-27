"""Tests for the environment: polynomial utils, action space, and circuit game."""

import pytest
import torch
from sympy import symbols, expand, Integer

from src.config import Config
from src.environment.polynomial_utils import (
    create_variables,
    mod_reduce,
    canonical_key,
    poly_equal,
    poly_to_coefficient_vector,
    get_monomial_list,
    term_similarity,
    sympy_to_fast,
    fast_to_sympy,
)
from src.environment.fast_polynomial import FastPoly
from src.environment.action_space import (
    encode_action,
    decode_action,
    compute_max_actions,
    get_valid_actions_mask,
)
from src.environment.circuit_game import CircuitGame
from src.environment.factor_library import FactorLibrary


# ========== Polynomial Utils Tests (SymPy — kept for backward compat) ==========

class TestModReduce:
    def setup_method(self):
        self.syms = create_variables(2)
        self.x0, self.x1 = self.syms

    def test_no_reduction_needed(self):
        expr = 3 * self.x0 + 4 * self.x1
        result = mod_reduce(expr, self.syms, 5)
        assert poly_equal(result, 3 * self.x0 + 4 * self.x1, self.syms, 5)

    def test_coefficient_reduction(self):
        expr = 6 * self.x0
        result = mod_reduce(expr, self.syms, 5)
        assert poly_equal(result, self.x0, self.syms, 5)

    def test_zero_coefficient_removed(self):
        expr = 5 * self.x0 + 3 * self.x1
        result = mod_reduce(expr, self.syms, 5)
        assert poly_equal(result, 3 * self.x1, self.syms, 5)

    def test_constant_mod(self):
        result = mod_reduce(Integer(7), self.syms, 5)
        assert result == Integer(2)

    def test_zero_result(self):
        result = mod_reduce(Integer(0), self.syms, 5)
        assert result == Integer(0)

    def test_negative_coefficients(self):
        expr = -1 * self.x0
        result = mod_reduce(expr, self.syms, 5)
        assert poly_equal(result, 4 * self.x0, self.syms, 5)

    def test_multiplication_then_reduce(self):
        expr = (3 * self.x0) * (2 * self.x0)
        result = mod_reduce(expr, self.syms, 5)
        assert poly_equal(result, self.x0 ** 2, self.syms, 5)

    def test_complex_expression(self):
        expr = (self.x0 + self.x1) ** 2
        result = mod_reduce(expr, self.syms, 5)
        expected = self.x0 ** 2 + 2 * self.x0 * self.x1 + self.x1 ** 2
        assert poly_equal(result, expected, self.syms, 5)


class TestCanonicalKey:
    def setup_method(self):
        self.syms = create_variables(2)
        self.x0, self.x1 = self.syms

    def test_same_expression_same_key(self):
        e1 = self.x0 + self.x1
        e2 = self.x1 + self.x0
        assert canonical_key(e1, self.syms, 5) == canonical_key(e2, self.syms, 5)

    def test_equivalent_after_mod(self):
        e1 = 6 * self.x0
        e2 = self.x0
        assert canonical_key(e1, self.syms, 5) == canonical_key(e2, self.syms, 5)

    def test_different_expressions_different_keys(self):
        e1 = self.x0
        e2 = self.x1
        assert canonical_key(e1, self.syms, 5) != canonical_key(e2, self.syms, 5)


class TestPolyEqual:
    def setup_method(self):
        self.syms = create_variables(2)
        self.x0, self.x1 = self.syms

    def test_equal(self):
        assert poly_equal(self.x0 + self.x1, self.x1 + self.x0, self.syms, 5)

    def test_equal_mod(self):
        assert poly_equal(6 * self.x0, self.x0, self.syms, 5)

    def test_not_equal(self):
        assert not poly_equal(self.x0, self.x1, self.syms, 5)


class TestMonomialList:
    def test_two_vars_degree_2(self):
        monoms = get_monomial_list(2, 2)
        assert len(monoms) == 6
        assert monoms[0] == (0, 0)
        assert (2, 0) in monoms
        assert (1, 1) in monoms
        assert (0, 2) in monoms

    def test_one_var_degree_3(self):
        monoms = get_monomial_list(1, 3)
        assert len(monoms) == 4
        assert monoms == [(0,), (1,), (2,), (3,)]


class TestCoefficientVector:
    def setup_method(self):
        self.syms = create_variables(2)
        self.x0, self.x1 = self.syms

    def test_simple_polynomial(self):
        expr = 3 * self.x0 + 2 * self.x1 + 1
        coeffs = poly_to_coefficient_vector(expr, self.syms, 5, max_degree=2)
        monoms = get_monomial_list(2, 2)
        const_idx = monoms.index((0, 0))
        x0_idx = monoms.index((1, 0))
        x1_idx = monoms.index((0, 1))
        assert coeffs[const_idx] == 1
        assert coeffs[x0_idx] == 3
        assert coeffs[x1_idx] == 2

    def test_zero_polynomial(self):
        expr = Integer(0)
        coeffs = poly_to_coefficient_vector(expr, self.syms, 5, max_degree=2)
        assert all(c == 0 for c in coeffs)


class TestTermSimilarity:
    def setup_method(self):
        self.syms = create_variables(2)
        self.x0, self.x1 = self.syms

    def test_identical(self):
        expr = self.x0 + self.x1
        sim = term_similarity(expr, expr, self.syms, 5, max_degree=2)
        assert sim == 1.0

    def test_completely_different(self):
        current = self.x0
        target = self.x1
        sim = term_similarity(current, target, self.syms, 5, max_degree=2)
        assert sim == 0.0

    def test_partial_match(self):
        current = self.x0 + self.x1
        target = self.x0 + 2 * self.x1
        sim = term_similarity(current, target, self.syms, 5, max_degree=2)
        assert sim == 0.5


# ========== SymPy <-> FastPoly Conversion Tests ==========

class TestSymPyFastPolyConversion:
    def setup_method(self):
        self.syms = create_variables(2)
        self.x0, self.x1 = self.syms

    def test_variable_roundtrip(self):
        fp = sympy_to_fast(self.x0, self.syms, 5, 6)
        expected = FastPoly.variable(0, 2, 6, 5)
        assert fp == expected

    def test_expression_roundtrip(self):
        expr = 3 * self.x0 + 2 * self.x1 + 1
        fp = sympy_to_fast(expr, self.syms, 5, 6)
        back = fast_to_sympy(fp, self.syms)
        assert poly_equal(expr, back, self.syms, 5)

    def test_mod_reduction_in_conversion(self):
        expr = 6 * self.x0  # 6 mod 5 = 1
        fp = sympy_to_fast(expr, self.syms, 5, 6)
        expected = FastPoly.variable(0, 2, 6, 5)
        assert fp == expected

    def test_zero_conversion(self):
        expr = Integer(0)
        fp = sympy_to_fast(expr, self.syms, 5, 6)
        assert fp.is_zero()


# ========== Action Space Tests ==========

class TestActionEncoding:
    def test_roundtrip(self):
        max_nodes = 5
        for op in (0, 1):
            for i in range(max_nodes):
                for j in range(i, max_nodes):
                    idx = encode_action(op, i, j, max_nodes)
                    decoded_op, decoded_i, decoded_j = decode_action(idx, max_nodes)
                    assert (decoded_op, decoded_i, decoded_j) == (op, i, j), \
                        f"Failed: encode({op},{i},{j})={idx} -> decode={decoded_op},{decoded_i},{decoded_j}"

    def test_roundtrip_large(self):
        max_nodes = 9
        for op in (0, 1):
            for i in range(max_nodes):
                for j in range(i, max_nodes):
                    idx = encode_action(op, i, j, max_nodes)
                    decoded = decode_action(idx, max_nodes)
                    assert decoded == (op, i, j)

    def test_swap_normalization(self):
        idx1 = encode_action(0, 1, 3, 5)
        idx2 = encode_action(0, 3, 1, 5)
        assert idx1 == idx2

    def test_max_actions(self):
        assert compute_max_actions(5) == 30
        assert compute_max_actions(9) == 90

    def test_no_duplicate_indices(self):
        max_nodes = 9
        seen = set()
        for op in (0, 1):
            for i in range(max_nodes):
                for j in range(i, max_nodes):
                    idx = encode_action(op, i, j, max_nodes)
                    assert idx not in seen, f"Duplicate index {idx} for ({op},{i},{j})"
                    seen.add(idx)
        assert len(seen) == compute_max_actions(max_nodes)


class TestActionMask:
    def test_initial_mask(self):
        mask = get_valid_actions_mask(3, 9)
        assert mask.shape[0] == compute_max_actions(9)
        assert mask.sum().item() == 12

    def test_single_node(self):
        mask = get_valid_actions_mask(1, 5)
        assert mask.sum().item() == 2

    def test_all_nodes(self):
        max_nodes = 5
        mask = get_valid_actions_mask(max_nodes, max_nodes)
        assert mask.sum().item() == compute_max_actions(max_nodes)


# ========== Circuit Game Tests (now using FastPoly) ==========

class TestCircuitGame:
    def setup_method(self):
        self.config = Config(n_variables=2, mod=5, max_complexity=4, max_steps=6)
        self.game = CircuitGame(self.config)
        self.mod = 5
        self.n_vars = 2
        self.max_deg = self.config.effective_max_degree
        # FastPoly helpers
        self.x0 = FastPoly.variable(0, self.n_vars, self.max_deg, self.mod)
        self.x1 = FastPoly.variable(1, self.n_vars, self.max_deg, self.mod)
        self.one = FastPoly.constant(1, self.n_vars, self.max_deg, self.mod)

    def test_reset(self):
        target = self.x0 + self.x1
        obs = self.game.reset(target)
        assert not self.game.done
        assert self.game.steps_taken == 0
        assert len(self.game.nodes) == 3  # x0, x1, 1
        assert "graph" in obs
        assert "target" in obs
        assert "mask" in obs

    def test_step_add(self):
        target = self.x0 + self.x1
        self.game.reset(target)
        action = encode_action(0, 0, 1, self.config.max_nodes)
        obs, reward, done, info = self.game.step(action)
        assert info["is_success"]
        assert done
        assert info["op"] == "add"

    def test_step_multiply(self):
        target = self.x0 * self.x1
        self.game.reset(target)
        action = encode_action(1, 0, 1, self.config.max_nodes)
        obs, reward, done, info = self.game.step(action)
        assert info["is_success"]

    def test_multi_step(self):
        """Build x0^2 + x1 in two steps."""
        target = (self.x0 * self.x0) + self.x1
        self.game.reset(target)

        # Step 1: x0 * x0
        action1 = encode_action(1, 0, 0, self.config.max_nodes)
        obs, reward, done, info = self.game.step(action1)
        assert not info["is_success"]
        assert not done

        # Step 2: x0^2 + x1 (node 3 is x0^2, node 1 is x1)
        action2 = encode_action(0, 3, 1, self.config.max_nodes)
        obs, reward, done, info = self.game.step(action2)
        assert info["is_success"]

    def test_max_steps_terminates(self):
        # Hard target unlikely to be hit by repeated x0+x0
        target = (self.x0 * self.x0) * (self.x0 * self.x0) + self.x1
        self.game.reset(target)

        for _ in range(self.config.max_steps):
            if self.game.done:
                break
            action = encode_action(0, 0, 0, self.config.max_nodes)
            obs, reward, done, info = self.game.step(action)

        assert self.game.done

    def test_observation_shapes(self):
        target = self.x0 + self.x1
        obs = self.game.reset(target)
        graph = obs["graph"]
        if isinstance(graph, dict):
            assert graph["x"].shape == (self.config.max_nodes, self.config.node_feature_dim)
        else:
            assert graph.x.shape == (self.config.max_nodes, self.config.node_feature_dim)
        assert obs["target"].shape[0] == self.config.target_size
        assert obs["mask"].shape[0] == self.config.max_actions

    def test_success_reward(self):
        target = self.x0 + self.x1
        self.game.reset(target)
        action = encode_action(0, 0, 1, self.config.max_nodes)
        _, reward, _, info = self.game.step(action)
        assert info["is_success"]
        assert reward > self.config.success_reward / 2

    def test_clone(self):
        target = self.x0 + self.x1
        self.game.reset(target)
        action = encode_action(0, 0, 0, self.config.max_nodes)
        self.game.step(action)

        cloned = self.game.clone()
        assert len(cloned.nodes) == len(self.game.nodes)
        assert cloned.steps_taken == self.game.steps_taken
        assert cloned.done == self.game.done

        # Modifying clone shouldn't affect original
        action2 = encode_action(0, 0, 1, self.config.max_nodes)
        cloned.step(action2)
        assert len(cloned.nodes) != len(self.game.nodes)

    def test_constant_node_usage(self):
        """Test using the constant node (value 1)."""
        target = self.x0 + self.one
        self.game.reset(target)
        action = encode_action(0, 0, 2, self.config.max_nodes)
        _, _, _, info = self.game.step(action)
        assert info["is_success"]


class TestFactorLibraryRewards:
    def setup_method(self):
        self.config = Config(n_variables=2, mod=5, max_complexity=4, max_steps=6)
        self.mod = 5
        self.n_vars = 2
        self.max_deg = self.config.effective_max_degree
        self.x0 = FastPoly.variable(0, self.n_vars, self.max_deg, self.mod)
        self.x1 = FastPoly.variable(1, self.n_vars, self.max_deg, self.mod)
        self.one = FastPoly.constant(1, self.n_vars, self.max_deg, self.mod)

    def _make_env(self) -> CircuitGame:
        lib = FactorLibrary(
            mod=self.config.mod,
            n_vars=self.config.n_variables,
            max_degree=self.config.effective_max_degree,
        )
        return CircuitGame(self.config, factor_library=lib)

    def test_additive_completion_bonus_is_exact(self):
        env = self._make_env()
        target = self.x0 + self.x1 + self.one
        env.reset(target)

        action = encode_action(0, 0, 1, self.config.max_nodes)  # x0 + x1
        _, reward, _, info = env.step(action)

        assert info["additive_complete"]
        assert reward >= self.config.completion_bonus + self.config.step_penalty

    def test_scalar_exact_quotient_does_not_trigger_mult_completion(self):
        env = self._make_env()
        syms = create_variables(self.n_vars)
        target = sympy_to_fast(3 * (syms[0] + 1), syms, self.mod, self.max_deg)

        known_factor = self.x0 + self.one
        env.factor_library.register(known_factor, step_num=1)
        env.reset(target)

        action = encode_action(0, 0, 2, self.config.max_nodes)  # x0 + 1
        _, _, _, info = env.step(action)

        assert info["factor_hit"]
        assert info["library_hit"]
        assert not info["mult_complete"]


class _OnPathContext:
    def __init__(self, mapping, target_board_step, route_mapping=None):
        self.on_path_keys = mapping
        self.on_path_route_keys = (
            route_mapping
            if route_mapping is not None
            else {key: (1 << 32) - 1 for key in mapping}
        )
        self.target_board_step = target_board_step


class TestRewardModes:
    def setup_method(self):
        self.config = Config(n_variables=2, mod=5, max_complexity=4, max_steps=6)
        self.mod = 5
        self.n_vars = 2
        self.max_deg = self.config.effective_max_degree
        self.x0 = FastPoly.variable(0, self.n_vars, self.max_deg, self.mod)
        self.x1 = FastPoly.variable(1, self.n_vars, self.max_deg, self.mod)

    def test_clean_sparse_has_only_terminal_and_step_reward(self):
        self.config.reward_mode = "clean_sparse"
        self.config.success_reward = 99.0
        self.config.terminal_success_reward = 7.0
        self.config.factor_subgoal_reward = 99.0
        self.config.completion_bonus = 99.0
        env = CircuitGame(self.config)

        target = self.x0 + self.x1
        env.reset(target)
        action = encode_action(0, 0, 1, self.config.max_nodes)
        _, reward, _, info = env.step(action)

        assert info["is_success"]
        assert reward == pytest.approx(self.config.step_penalty + 7.0)
        assert not info["factor_hit"]
        assert not info["additive_complete"]

    def test_legacy_and_clean_sparse_fixed_action_rewards_are_stable(self):
        target = self.x0 + self.x1
        action = encode_action(0, 0, 1, self.config.max_nodes)

        self.config.reward_mode = "legacy"
        legacy_env = CircuitGame(self.config)
        legacy_env.reset(target)
        _, legacy_reward, _, _ = legacy_env.step(action)
        assert [legacy_reward] == pytest.approx([9.9])

        self.config.reward_mode = "clean_sparse"
        self.config.terminal_success_reward = 7.0
        clean_env = CircuitGame(self.config)
        clean_env.reset(target)
        _, clean_reward, _, _ = clean_env.step(action)
        assert [clean_reward] == pytest.approx([6.9])

    def test_clean_onpath_count_phi_and_duplicate_hits(self):
        self.config.reward_mode = "clean_onpath"
        self.config.on_path_phi_mode = "count"
        self.config.graph_onpath_shaping_coeff = 1.0
        env = CircuitGame(self.config)

        intermediate = self.x0 + self.x1
        target = intermediate + self.x1
        ctx = _OnPathContext(
            {
                intermediate.canonical_key(): 1,
                target.canonical_key(): 2,
            },
            target_board_step=2,
        )
        env.reset(target, on_path_context=ctx)

        action = encode_action(0, 0, 1, self.config.max_nodes)
        _, reward, _, info = env.step(action)

        assert info["on_path_hit"]
        assert info["on_path_hits"] == 1
        assert info["on_path_phi"] == pytest.approx(0.5)
        assert reward == pytest.approx(
            self.config.step_penalty + self.config.gamma * 0.5
        )

        _, reward2, _, info2 = env.step(action)
        assert not info2["on_path_hit"]
        assert info2["on_path_hits"] == 1
        assert info2["on_path_phi"] == pytest.approx(0.5)
        assert reward2 == pytest.approx(
            self.config.step_penalty
            + self.config.gamma * 0.5
            - 0.5
        )

    def test_clean_onpath_route_consistency_blocks_incompatible_hits(self):
        self.config.reward_mode = "clean_onpath"
        self.config.on_path_phi_mode = "count"
        self.config.graph_onpath_shaping_coeff = 1.0
        env = CircuitGame(self.config)

        route_a = self.x0 + self.x1
        route_b = self.x0 * self.x1
        target = route_a + route_b
        ctx = _OnPathContext(
            {
                route_a.canonical_key(): 1,
                route_b.canonical_key(): 1,
                target.canonical_key(): 2,
            },
            target_board_step=2,
            route_mapping={
                route_a.canonical_key(): 0b01,
                route_b.canonical_key(): 0b10,
                target.canonical_key(): 0b11,
            },
        )
        env.reset(target, on_path_context=ctx)

        add_action = encode_action(0, 0, 1, self.config.max_nodes)
        mul_action = encode_action(1, 0, 1, self.config.max_nodes)

        _, reward, _, info = env.step(add_action)
        assert info["on_path_hit"]
        assert info["on_path_hits"] == 1
        assert info["on_path_phi"] == pytest.approx(1 / 3)
        assert reward == pytest.approx(
            self.config.step_penalty + self.config.gamma * (1 / 3)
        )

        _, reward2, _, info2 = env.step(mul_action)
        assert not info2["on_path_hit"]
        assert info2["on_path_hits"] == 1
        assert info2["on_path_phi"] == pytest.approx(1 / 3)
        assert reward2 == pytest.approx(
            self.config.step_penalty
            + self.config.gamma * (1 / 3)
            - (1 / 3)
        )

    def test_clean_onpath_success_logs_raw_phi_but_zeros_terminal_reward_phi(self):
        self.config.reward_mode = "clean_onpath"
        self.config.on_path_phi_mode = "count"
        self.config.graph_onpath_shaping_coeff = 1.0
        env = CircuitGame(self.config)

        target = self.x0 + self.x1
        ctx = _OnPathContext({target.canonical_key(): 1}, target_board_step=1)
        env.reset(target, on_path_context=ctx)

        action = encode_action(0, 0, 1, self.config.max_nodes)
        _, reward, done, info = env.step(action)

        assert done
        assert info["is_success"]
        assert info["on_path_hit"]
        assert info["on_path_phi"] == pytest.approx(1.0)
        assert reward == pytest.approx(
            self.config.step_penalty + self.config.terminal_success_reward
        )

    def test_clean_onpath_truncation_zeros_terminal_reward_phi(self):
        self.config.reward_mode = "clean_onpath"
        self.config.on_path_phi_mode = "count"
        self.config.graph_onpath_shaping_coeff = 1.0
        self.config.max_steps = 1
        env = CircuitGame(self.config)

        intermediate = self.x0 + self.x1
        target = intermediate + self.x1
        ctx = _OnPathContext(
            {
                intermediate.canonical_key(): 1,
                target.canonical_key(): 2,
            },
            target_board_step=2,
        )
        env.reset(target, on_path_context=ctx)

        action = encode_action(0, 0, 1, self.config.max_nodes)
        _, reward, done, info = env.step(action)

        assert done
        assert not info["is_success"]
        assert info["on_path_hit"]
        assert info["on_path_phi"] == pytest.approx(0.5)
        assert reward == pytest.approx(self.config.step_penalty)

    def test_clean_onpath_discounted_shaping_telescopes_to_initial_phi(self):
        self.config.reward_mode = "clean_onpath"
        self.config.on_path_phi_mode = "count"
        self.config.graph_onpath_shaping_coeff = 1.0
        env = CircuitGame(self.config)

        intermediate = self.x0 + self.x1
        target = intermediate + self.x1
        ctx = _OnPathContext(
            {
                intermediate.canonical_key(): 1,
                target.canonical_key(): 2,
            },
            target_board_step=2,
        )
        env.reset(target, on_path_context=ctx)
        phi0 = env._on_path_phi()

        discounted_shaping_sum = 0.0
        actions = [
            encode_action(0, 0, 1, self.config.max_nodes),
            encode_action(0, 3, 1, self.config.max_nodes),
        ]
        for t, action in enumerate(actions):
            _, reward, done, info = env.step(action)
            base_reward = self.config.step_penalty
            if info["is_success"]:
                base_reward += self.config.terminal_success_reward
            shaping = reward - base_reward
            discounted_shaping_sum += (self.config.gamma ** t) * shaping
            if done:
                break

        assert done
        assert info["on_path_phi"] == pytest.approx(1.0)
        assert discounted_shaping_sum == pytest.approx(-phi0)

    def test_clean_onpath_clone_copies_hit_state_independently(self):
        self.config.reward_mode = "clean_onpath"
        env = CircuitGame(self.config)

        intermediate = self.x0 + self.x1
        target = intermediate + self.x1
        ctx = _OnPathContext(
            {
                intermediate.canonical_key(): 1,
                target.canonical_key(): 2,
            },
            target_board_step=2,
        )
        env.reset(target, on_path_context=ctx)
        cloned = env.clone()

        action = encode_action(0, 0, 1, self.config.max_nodes)
        _, _, _, info = env.step(action)
        _, _, _, clone_info = cloned.step(action)

        assert info["on_path_hit"]
        assert clone_info["on_path_hit"]
        assert env._on_path_count == 1
        assert cloned._on_path_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
