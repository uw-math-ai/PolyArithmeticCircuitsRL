"""Tests for the game board generator."""

import numpy as np
import pytest

from src.config import Config
from src.environment.fast_polynomial import FastPoly
from src.game_board.generator import (
    build_game_board,
    find_interesting_targets,
    sample_target,
    generate_random_circuit,
)
from src.game_board.on_path import (
    OnPathCache,
    _route_masks_from_minimal_routes,
    build_caches,
    compute_on_path_ids,
    compute_on_path_route_masks,
    compute_sequential_route_sets,
    validate_cache_metadata,
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


class TestOnPathCache:
    def _node_id_for_poly(self, comp, poly):
        coeffs = poly.coeffs.flatten().astype(np.int64)
        matches = np.nonzero(np.all(comp.node_coeffs == coeffs[None, :], axis=1))[0]
        assert matches.size > 0
        return int(matches[0])

    def test_minimal_sequential_route_ids_exclude_base_nodes(self):
        parents = [
            [],
            [],
            [(0, 1)],
            [],
            [(2, 1), (3, 1)],
        ]
        steps = [0, 0, 1, 2, 2]
        assert compute_on_path_ids(4, parents, steps) == {2, 4}

    def test_route_masks_separate_incompatible_optimal_branches(self):
        parents = [
            [],
            [],
            [(0, 1)],
            [(0, 1)],
            [(2, 1), (3, 1)],
        ]
        steps = [0, 0, 1, 1, 2]
        masks, route_cap_hit = compute_on_path_route_masks(
            4,
            parents,
            steps,
            base_ids={0, 1},
            max_routes=2,
        )

        assert masks[4] == 0b11
        assert masks[2] & masks[3] == 0
        assert not route_cap_hit

    def test_route_cap_hit_propagates_from_capped_ancestor(self):
        # Two minimal routes selected; both fit under max_routes; target itself
        # is not capped. But ancestor 2 was working-cap truncated upstream, so
        # the conservative cap-hit flag must still fire.
        minimal_routes = [frozenset({1, 2, 10}), frozenset({3, 4, 10})]
        masks, route_cap_hit = _route_masks_from_minimal_routes(
            minimal_routes,
            max_routes=8,
            capped_nodes={2},
            target_id=10,
        )
        assert masks[10] == 0b11
        assert route_cap_hit, "ancestor 2 was capped; target route_cap_hit should fire"

    def test_route_cap_hit_does_not_fire_for_unrelated_capped_node(self):
        minimal_routes = [frozenset({1, 2, 10})]
        _masks, route_cap_hit = _route_masks_from_minimal_routes(
            minimal_routes,
            max_routes=8,
            capped_nodes={99},
            target_id=10,
        )
        assert not route_cap_hit

    def test_route_cap_hit_fires_on_overflow_when_minimal_count_exceeds_max(self):
        minimal_routes = [
            frozenset({i, 100}) for i in range(5)
        ]
        _masks, route_cap_hit = _route_masks_from_minimal_routes(
            minimal_routes,
            max_routes=2,
            capped_nodes=set(),
            target_id=100,
        )
        assert route_cap_hit

    def test_working_cap_truncation_propagates_through_compute_routes(self):
        # Three distinct C1 ancestors (3, 4, 5) feed into node 6 via three
        # disjoint parent pairs. Node 6 has 3 minimal routes; with
        # working_route_cap=2 it gets truncated and joins capped_nodes.
        # Conservative cap-hit at the ancestor must surface even when a
        # downstream target itself has only one selected route.
        parents = [
            [],                 # 0 base
            [],                 # 1 base
            [],                 # 2 base
            [(0, 1)],           # 3 = f(0,1)
            [(0, 2)],           # 4 = f(0,2)
            [(1, 2)],           # 5 = f(1,2)
            [(3, 4), (3, 5), (4, 5)],  # 6: three distinct parent pairs
        ]
        base_ids = {0, 1, 2}
        _routes_by_id, capped = compute_sequential_route_sets(
            parents,
            base_ids,
            max_cost=4,
            working_route_cap=2,
        )
        assert 6 in capped
        # Imagine the target is node 6 itself, selected from a single
        # minimal route. The target isn't capped at "more routes than
        # max_routes", but enumeration at node 6 was truncated upstream.
        minimal_routes = [frozenset({3, 4, 6})]
        _masks, route_cap_hit = _route_masks_from_minimal_routes(
            minimal_routes,
            max_routes=8,
            capped_nodes=capped,
            target_id=6,
        )
        assert route_cap_hit

    def test_shared_square_is_sequential_c2_with_only_shared_intermediate(self, tmp_path):
        config = Config(
            n_variables=2,
            mod=5,
            max_complexity=2,
            max_degree=2,
            on_path_max_size=64,
        )
        build_caches(config, [2], tmp_path, split_seed=11, max_on_path_size=64)
        cache = OnPathCache.load(tmp_path, config, [2])
        comp = cache.by_complexity[2]

        x0 = FastPoly.variable(0, 2, config.effective_max_degree, config.mod)
        x1 = FastPoly.variable(1, 2, config.effective_max_degree, config.mod)
        shared = x0 + x1
        target = shared * shared
        shared_id = self._node_id_for_poly(comp, shared)
        target_id = self._node_id_for_poly(comp, target)
        ctx = comp.target_context(target_id)

        assert target_id in set(comp.target_ids.tolist())
        assert set(ctx.on_path_ids.tolist()) == {shared_id, target_id}
        assert int(comp.node_steps[shared_id]) == 1
        assert int(comp.node_steps[target_id]) == 2

    def test_independent_two_intermediates_are_sequential_c3_not_c2(self, tmp_path):
        config = Config(
            n_variables=2,
            mod=5,
            max_complexity=3,
            max_degree=2,
            on_path_max_size=64,
        )
        build_caches(config, [2, 3], tmp_path, split_seed=11, max_on_path_size=64)
        cache = OnPathCache.load(tmp_path, config, [2, 3])
        comp_c2 = cache.by_complexity[2]
        comp_c3 = cache.by_complexity[3]

        x0 = FastPoly.variable(0, 2, config.effective_max_degree, config.mod)
        x1 = FastPoly.variable(1, 2, config.effective_max_degree, config.mod)
        one = FastPoly.constant(1, 2, config.effective_max_degree, config.mod)
        target = (x0 + x1) * (x0 + one)

        c2_target_id = self._node_id_for_poly(comp_c2, target)
        c3_target_id = self._node_id_for_poly(comp_c3, target)
        assert c2_target_id not in set(comp_c2.target_ids.tolist())
        assert c3_target_id in set(comp_c3.target_ids.tolist())
        assert int(comp_c3.node_steps[c3_target_id]) == 3

    def test_c2_route_masks_have_two_nonbase_nodes(self, tmp_path):
        config = Config(
            n_variables=2,
            mod=5,
            max_complexity=2,
            max_degree=2,
            on_path_max_size=64,
        )
        build_caches(config, [2], tmp_path, split_seed=11, max_on_path_size=64)
        cache = OnPathCache.load(tmp_path, config, [2])
        comp = cache.by_complexity[2]

        for target_pos, _target_id in enumerate(comp.target_ids):
            start = int(comp.on_path_offsets[target_pos])
            end = int(comp.on_path_offsets[target_pos + 1])
            masks = comp.on_path_route_masks[start:end]
            route_sizes = [
                int(np.sum((masks & np.uint32(1 << route_idx)) != 0))
                for route_idx in range(32)
                if np.any((masks & np.uint32(1 << route_idx)) != 0)
            ]
            assert route_sizes
            assert all(size == 2 for size in route_sizes)

    def test_cache_excludes_base_nodes_and_includes_target(self, tmp_path):
        config = Config(
            n_variables=2,
            mod=5,
            max_complexity=2,
            max_degree=2,
            on_path_max_size=64,
        )
        build_caches(config, [2], tmp_path, split_seed=11, max_on_path_size=64)
        cache = OnPathCache.load(tmp_path, config, [2])
        comp = cache.by_complexity[2]
        target_id = int(comp.train_target_ids[0])
        ctx = comp.target_context(target_id)

        assert target_id in set(ctx.on_path_ids.tolist())
        assert all(int(comp.node_steps[i]) > 0 for i in ctx.on_path_ids)
        assert len(ctx.on_path_ids) > 0
        assert comp.train_target_ids.size > 0
        assert comp.metadata["split_seed"] == 11

    def test_cache_allows_requested_complexity_subset(self, tmp_path):
        config = Config(
            n_variables=2,
            mod=5,
            max_complexity=2,
            max_degree=2,
            on_path_max_size=64,
        )
        build_caches(config, [1, 2], tmp_path, split_seed=7, max_on_path_size=64)
        cache = OnPathCache.load(tmp_path, config, [2])
        assert sorted(cache.by_complexity) == [2]

    def test_cache_metadata_mismatch_fails(self, tmp_path):
        config = Config(
            n_variables=2,
            mod=5,
            max_complexity=1,
            max_degree=1,
            on_path_max_size=64,
        )
        build_caches(config, [1], tmp_path, split_seed=7, max_on_path_size=64)
        bad_config = Config(
            n_variables=2,
            mod=7,
            max_complexity=1,
            max_degree=1,
            on_path_max_size=64,
        )
        with pytest.raises(ValueError, match="metadata mismatch"):
            OnPathCache.load(tmp_path, bad_config, [1])

    def test_old_depth_metric_cache_metadata_fails(self):
        config = Config(
            n_variables=2,
            mod=5,
            max_complexity=1,
            max_degree=1,
            on_path_max_size=64,
        )
        metadata = {
            "cache_version": 3,
            "n_variables": 2,
            "mod": 5,
            "max_degree": config.effective_max_degree,
            "target_size": config.target_size,
            "op_set": ["add", "mul"],
            "canonicalizer_version": "FastPoly.coeffs.int64.tobytes.v1",
            "include_constant": True,
            "base_node_count": 3,
            "step_metric": "board_step_depth_max_parent_plus_one",
            "route_mask_mode": "coherent_optimal_route_masks_v1",
            "route_count_cap": config.on_path_num_routes,
            "complexity": 1,
        }
        with pytest.raises(ValueError, match="metadata mismatch for cache_version"):
            validate_cache_metadata(metadata, config, requested_complexity=1)

    def test_cache_route_truncation_threshold_fails(self, tmp_path):
        config = Config(
            n_variables=2,
            mod=5,
            max_complexity=2,
            max_degree=2,
            on_path_max_size=64,
            on_path_num_routes=1,
        )
        with pytest.raises(ValueError, match="route cap hit rate"):
            build_caches(
                config,
                [2],
                tmp_path,
                split_seed=7,
                max_on_path_size=64,
                max_routes=1,
                max_route_truncation_rate=0.0,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
