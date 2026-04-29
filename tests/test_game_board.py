"""Tests for the game board generator."""

import warnings

import numpy as np
import pytest

from src.config import Config
from src.environment.fast_polynomial import FastPoly
from src.environment.poly_batch_ops import PolyBatchOps
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


def _bs(*ids: int) -> int:
    """Build a bitset from node ids; route literals are Python ints now."""
    out = 0
    for i in ids:
        out |= 1 << i
    return out


def _build_game_board_scalar_reference(config: Config, complexity: int):
    """Legacy scalar-loop board builder kept for vectorization parity tests."""
    n_vars = config.n_variables
    mod = config.mod
    max_deg = config.effective_max_degree

    initial_polys = [
        FastPoly.variable(i, n_vars, max_deg, mod)
        for i in range(n_vars)
    ]
    initial_polys.append(FastPoly.constant(1, n_vars, max_deg, mod))

    board = {}
    all_polys = []
    for poly in initial_polys:
        key = poly.canonical_key()
        if key not in board:
            board[key] = {
                "poly": poly,
                "step": 0,
                "parents": [],
                "paths": 1,
            }
            all_polys.append(poly)

    for step in range(1, complexity + 1):
        new_polys = []
        n = len(all_polys)
        for i in range(n):
            for j in range(i, n):
                for op in (0, 1):
                    result = (
                        all_polys[i] + all_polys[j]
                        if op == 0
                        else all_polys[i] * all_polys[j]
                    )
                    key = result.canonical_key()
                    parent_info = {
                        "op": "add" if op == 0 else "mul",
                        "left": all_polys[i].canonical_key(),
                        "right": all_polys[j].canonical_key(),
                    }
                    if key not in board:
                        board[key] = {
                            "poly": result,
                            "step": step,
                            "parents": [parent_info],
                            "paths": 1,
                        }
                        new_polys.append(result)
                    else:
                        board[key]["parents"].append(parent_info)
                        board[key]["paths"] += 1
        all_polys.extend(new_polys)
    return board


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

    def test_vectorized_builder_matches_scalar_reference_c3(self):
        config = Config(n_variables=2, mod=5, max_complexity=3, max_degree=3)
        board = build_game_board(config, complexity=3)
        ref = _build_game_board_scalar_reference(config, complexity=3)

        assert set(board) == set(ref)
        for key in ref:
            assert board[key]["step"] == ref[key]["step"]
            assert board[key]["paths"] == ref[key]["paths"]
            assert np.array_equal(board[key]["poly"].coeffs, ref[key]["poly"].coeffs)
            assert len(board[key]["parents"]) == len(ref[key]["parents"])


class TestPolyBatchOpsBackends:
    @staticmethod
    def _sample_coeffs_and_pairs():
        config = Config(n_variables=2, mod=5, max_complexity=2, max_degree=2)
        n_vars = config.n_variables
        max_deg = config.effective_max_degree
        mod = config.mod
        polys = [
            FastPoly.variable(0, n_vars, max_deg, mod),
            FastPoly.variable(1, n_vars, max_deg, mod),
            FastPoly.constant(1, n_vars, max_deg, mod),
        ]
        polys.append(polys[0] + polys[1])
        coeffs = np.stack([p.coeffs.reshape(-1) for p in polys], axis=0)
        tri_i, tri_j = np.triu_indices(len(polys))
        pair_idx = np.stack((tri_i, tri_j), axis=1)
        return config, polys, coeffs, pair_idx

    def test_numpy_backend_matches_scalar_loop_c2(self):
        config, polys, coeffs, pair_idx = self._sample_coeffs_and_pairs()
        n_vars = config.n_variables
        max_deg = config.effective_max_degree
        mod = config.mod

        ops = PolyBatchOps(n_vars, max_deg, mod, backend="numpy")
        add_rows = ops.add_all_pairs(coeffs, pair_idx)
        mul_rows = ops.mul_all_pairs(coeffs, pair_idx)

        expected_add = []
        expected_mul = []
        for i, j in pair_idx:
            expected_add.append((polys[int(i)] + polys[int(j)]).coeffs.reshape(-1))
            expected_mul.append((polys[int(i)] * polys[int(j)]).coeffs.reshape(-1))

        assert np.array_equal(add_rows, np.stack(expected_add, axis=0))
        assert np.array_equal(mul_rows, np.stack(expected_mul, axis=0))

    def test_jax_backend_matches_numpy_when_available(self):
        pytest.importorskip("jax")
        config, _polys, coeffs, pair_idx = self._sample_coeffs_and_pairs()
        numpy_ops = PolyBatchOps(
            config.n_variables,
            config.effective_max_degree,
            config.mod,
            backend="numpy",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            jax_ops = PolyBatchOps(
                config.n_variables,
                config.effective_max_degree,
                config.mod,
                backend="jax",
            )
        if jax_ops.backend != "jax":
            pytest.skip("JAX is importable but no JAX GPU device is available")

        assert np.array_equal(
            jax_ops.add_all_pairs(coeffs, pair_idx),
            numpy_ops.add_all_pairs(coeffs, pair_idx),
        )
        assert np.array_equal(
            jax_ops.mul_all_pairs(coeffs, pair_idx),
            numpy_ops.mul_all_pairs(coeffs, pair_idx),
        )


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
        minimal_routes = [_bs(1, 2, 10), _bs(3, 4, 10)]
        masks, route_cap_hit = _route_masks_from_minimal_routes(
            minimal_routes,
            max_routes=8,
            capped_nodes={2},
            target_id=10,
        )
        assert masks[10] == 0b11
        assert route_cap_hit, "ancestor 2 was capped; target route_cap_hit should fire"

    def test_route_cap_hit_does_not_fire_for_unrelated_capped_node(self):
        minimal_routes = [_bs(1, 2, 10)]
        _masks, route_cap_hit = _route_masks_from_minimal_routes(
            minimal_routes,
            max_routes=8,
            capped_nodes={99},
            target_id=10,
        )
        assert not route_cap_hit

    def test_route_cap_hit_fires_on_overflow_when_minimal_count_exceeds_max(self):
        minimal_routes = [
            _bs(i, 100) for i in range(5)
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
        minimal_routes = [_bs(3, 4, 6)]
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
            "route_enumeration_mode": "bitset_minimal_routes_v1",
            "route_count_cap": config.on_path_num_routes,
            "board_backend": "numpy",
            "complexity": 1,
        }
        with pytest.raises(ValueError, match="metadata mismatch for cache_version"):
            validate_cache_metadata(metadata, config, requested_complexity=1)

        metadata_v4 = dict(metadata)
        metadata_v4.update({
            "cache_version": 4,
            "step_metric": "sequential_route_size_nonbase_nodes",
        })
        with pytest.raises(ValueError, match="metadata mismatch for cache_version"):
            validate_cache_metadata(metadata_v4, config, requested_complexity=1)

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


class TestRouteEnumerationEquivalence:
    """Bitset enumeration must match a frozenset reference implementation."""

    @staticmethod
    def _frozenset_reference(parents_by_id, base_ids, max_cost, working_route_cap):
        """Legacy frozenset enumeration kept inline for equivalence testing."""

        def sort_key(route):
            return (len(route), tuple(sorted(int(n) for n in route)))

        n_nodes = len(parents_by_id)
        routes_by_id = [set() for _ in range(n_nodes)]
        for base_id in base_ids:
            if 0 <= int(base_id) < n_nodes:
                routes_by_id[int(base_id)].add(frozenset())

        capped = set()
        for _ in range(max_cost):
            changed = False
            for child in range(n_nodes):
                if child in base_ids:
                    continue
                before = len(routes_by_id[child])
                for left, right in parents_by_id[child]:
                    lefts = tuple(routes_by_id[int(left)])
                    rights = tuple(routes_by_id[int(right)])
                    if not lefts or not rights:
                        continue
                    for lr in lefts:
                        if child in lr:
                            continue
                        for rr in rights:
                            if child in rr:
                                continue
                            r = frozenset({child}) | lr | rr
                            if len(r) <= max_cost:
                                routes_by_id[child].add(r)
                if len(routes_by_id[child]) > working_route_cap:
                    capped.add(child)
                    kept = sorted(routes_by_id[child], key=sort_key)[:working_route_cap]
                    routes_by_id[child] = set(kept)
                if len(routes_by_id[child]) != before:
                    changed = True
            if not changed:
                break

        return (
            tuple(tuple(sorted(rs, key=sort_key)) for rs in routes_by_id),
            capped,
        )

    @staticmethod
    def _bitset_to_frozenset(route: int) -> frozenset:
        out = set()
        i = 0
        while route:
            if route & 1:
                out.add(i)
            route >>= 1
            i += 1
        return frozenset(out)

    def _assert_equal(self, parents, base_ids, max_cost, cap):
        bs_routes, bs_capped = compute_sequential_route_sets(
            parents, base_ids, max_cost=max_cost, working_route_cap=cap
        )
        fs_routes, fs_capped = self._frozenset_reference(
            parents, base_ids, max_cost=max_cost, working_route_cap=cap
        )
        assert bs_capped == fs_capped
        for node_id, (bs_node, fs_node) in enumerate(zip(bs_routes, fs_routes)):
            bs_as_fs = sorted(
                (self._bitset_to_frozenset(r) for r in bs_node),
                key=lambda r: (len(r), tuple(sorted(r))),
            )
            fs_sorted = list(fs_node)
            assert bs_as_fs == fs_sorted, f"mismatch at node {node_id}"

    def test_bitset_matches_frozenset_toy_graph(self):
        parents = [
            [],
            [],
            [(0, 1)],
            [(0, 1)],
            [(2, 1), (3, 1)],
        ]
        self._assert_equal(parents, base_ids={0, 1}, max_cost=2, cap=128)

    def test_bitset_matches_frozenset_with_truncation(self):
        # Three disjoint parent pairs into node 6; cap=2 forces truncation.
        parents = [
            [],
            [],
            [],
            [(0, 1)],
            [(0, 2)],
            [(1, 2)],
            [(3, 4), (3, 5), (4, 5)],
        ]
        self._assert_equal(parents, base_ids={0, 1, 2}, max_cost=4, cap=2)

    def test_bitset_matches_frozenset_c3_board(self):
        from src.game_board.generator import build_game_board

        config = Config(n_variables=2, mod=5, max_complexity=3, max_degree=2)
        board = build_game_board(config, complexity=3)
        ordered_keys = sorted(board.keys(), key=lambda k: (board[k]["step"], k))
        key_to_id = {k: i for i, k in enumerate(ordered_keys)}
        parents = [[] for _ in ordered_keys]
        for k, entry in board.items():
            cid = key_to_id[k]
            for p in entry["parents"]:
                parents[cid].append((key_to_id[p["left"]], key_to_id[p["right"]]))
        base_ids = {i for i, k in enumerate(ordered_keys) if board[k]["step"] == 0}
        self._assert_equal(parents, base_ids=base_ids, max_cost=3, cap=128)


class TestParallelDeterminism:
    @staticmethod
    def _assert_npz_equal(left, right):
        with np.load(left, allow_pickle=False) as left_data:
            with np.load(right, allow_pickle=False) as right_data:
                assert set(left_data.files) == set(right_data.files)
                for name in left_data.files:
                    assert np.array_equal(left_data[name], right_data[name]), name

    def test_serial_and_parallel_builds_match(self, tmp_path):
        config = Config(
            n_variables=2,
            mod=5,
            max_complexity=4,
            max_degree=1,
            on_path_max_size=128,
        )
        serial_dir = tmp_path / "serial"
        parallel_dir = tmp_path / "parallel"
        complexities = [2, 3, 4]

        serial_paths = build_caches(
            config,
            complexities,
            serial_dir,
            split_seed=13,
            max_on_path_size=128,
            force_serial=True,
            max_route_truncation_rate=1.0,
        )
        parallel_paths = build_caches(
            config,
            complexities,
            parallel_dir,
            split_seed=13,
            max_on_path_size=128,
            num_processes=4,
            max_route_truncation_rate=1.0,
        )

        assert [p.name for p in serial_paths] == [p.name for p in parallel_paths]
        for serial_path, parallel_path in zip(serial_paths, parallel_paths):
            self._assert_npz_equal(serial_path, parallel_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
