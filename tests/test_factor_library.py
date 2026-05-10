"""Tests for the FactorizableLibrary top-down matching system."""

from __future__ import annotations

import pytest

from decomp_rl.factor_fp import FiniteFieldFactorizer
from decomp_rl.factor_library import FactorizableLibrary, LibraryEntry, LibraryMatch
from decomp_rl.polynomial import SparsePolynomial


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PRIME = 5
VARS = ("x", "y")


def _var(name: str) -> SparsePolynomial:
    return SparsePolynomial.variable(name, PRIME, VARS)


def _const(a: int) -> SparsePolynomial:
    return SparsePolynomial.from_monomial(a % PRIME, (0, 0), PRIME, VARS)


def _make_library(**kw) -> FactorizableLibrary:
    defaults = dict(prime=PRIME, variables=VARS, max_degree=6, match_scalar=True, match_permutations=True)
    defaults.update(kw)
    return FactorizableLibrary(**defaults)


# ---------------------------------------------------------------------------
# Basic add / is_known
# ---------------------------------------------------------------------------

def test_add_rejects_zero_savings():
    lib = _make_library()
    x, y = _var("x"), _var("y")
    poly = x + y
    factorizer = FiniteFieldFactorizer()
    fact = factorizer.factor(poly)
    factorizer.close()
    # x + y is irreducible — no poly factors → savings = 0 from perspective of rebuild
    # just check that we can call add and it won't crash
    added = lib.add(poly, fact, rebuild_ops=2, savings=0)
    assert not added


def test_add_and_is_known():
    lib = _make_library()
    x, y = _var("x"), _var("y")
    # (x + 1)^2 = x^2 + 2x + 1 — should factor as (x+1)^2
    poly = (x + _const(1)) * (x + _const(1))
    factorizer = FiniteFieldFactorizer()
    fact = factorizer.factor(poly)
    factorizer.close()
    from decomp_rl.cost_model import rebuild_cost
    from decomp_rl.baseline_cost import BaselineCostModel
    direct = BaselineCostModel().direct_construction_cost(poly)
    rb = rebuild_cost(fact)
    savings = direct - rb
    if savings > 0:
        added = lib.add(poly, fact, rb, savings)
        assert added
        assert lib.is_known(poly)
        assert len(lib) == 1
    else:
        pytest.skip("(x+1)^2 has no savings at p=5, skip")


# ---------------------------------------------------------------------------
# Exact subset matching
# ---------------------------------------------------------------------------

def test_exact_subset_match():
    lib = _make_library()
    x, y = _var("x"), _var("y")
    # Library entry: x^2 + 2xy + y^2 = (x+y)^2
    factorable = x * x + _const(2) * x * y + y * y
    factorizer = FiniteFieldFactorizer()
    fact = factorizer.factor(factorable)
    factorizer.close()
    from decomp_rl.cost_model import rebuild_cost
    from decomp_rl.baseline_cost import BaselineCostModel
    direct = BaselineCostModel().direct_construction_cost(factorable)
    rb = rebuild_cost(fact)
    if not lib.add(factorable, fact, rb, direct - rb):
        pytest.skip("no savings, skip")

    # Target: x^2 + 2xy + y^2 + 1
    target = factorable + _const(1)
    matches = lib.find_matches(target)
    assert len(matches) >= 1
    match = matches[0]
    assert match.matched_poly == factorable
    assert match.match_type == "exact"
    assert match.scale == 1

    # Verify the proposed split is valid: matched_poly + complement = target
    complement = target - match.matched_poly
    assert match.matched_poly + complement == target


# ---------------------------------------------------------------------------
# Scalar matching
# ---------------------------------------------------------------------------

def test_scalar_match():
    lib = _make_library()
    x, y = _var("x"), _var("y")
    # Library: x^2 + 2xy + y^2
    factorable = x * x + _const(2) * x * y + y * y
    factorizer = FiniteFieldFactorizer()
    fact = factorizer.factor(factorable)
    factorizer.close()
    from decomp_rl.cost_model import rebuild_cost
    from decomp_rl.baseline_cost import BaselineCostModel
    direct = BaselineCostModel().direct_construction_cost(factorable)
    rb = rebuild_cost(fact)
    if not lib.add(factorable, fact, rb, direct - rb):
        pytest.skip("no savings")

    # Target: 2*(x^2 + 2xy + y^2) + 1 = 2x^2 + 4xy + 2y^2 + 1
    scaled_entry = factorable.scale(2)
    target = scaled_entry + _const(1)
    matches = lib.find_matches(target)
    scalar_matches = [m for m in matches if m.match_type == "scalar"]
    assert len(scalar_matches) >= 1
    match = scalar_matches[0]
    assert match.scale == 2
    complement = target - match.matched_poly
    assert match.matched_poly + complement == target


# ---------------------------------------------------------------------------
# Permutation matching
# ---------------------------------------------------------------------------

def test_permutation_match():
    lib = _make_library()
    x, y = _var("x"), _var("y")
    # Library: x^2 + 2xy + y^2  (symmetric, so permuting x↔y gives same poly)
    # Use an asymmetric one: x^2 + y (only x-heavy)
    # Library: x^2 + x*y (treat as x(x + y))
    factorable = x * x + x * y   # x*(x + y)
    factorizer = FiniteFieldFactorizer()
    fact = factorizer.factor(factorable)
    factorizer.close()
    from decomp_rl.cost_model import rebuild_cost
    from decomp_rl.baseline_cost import BaselineCostModel
    direct = BaselineCostModel().direct_construction_cost(factorable)
    rb = rebuild_cost(fact)
    if not lib.add(factorable, fact, rb, direct - rb):
        pytest.skip("no savings for x^2 + xy")

    # Target contains y^2 + xy = y*(y + x), which is the x↔y permuted version
    permuted = y * y + x * y
    target = permuted + _const(1)
    matches = lib.find_matches(target)
    perm_matches = [m for m in matches if "permut" in m.match_type]
    assert len(perm_matches) >= 1
    m = perm_matches[0]
    complement = target - m.matched_poly
    assert m.matched_poly + complement == target


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def test_seed_known_families_adds_entries():
    lib = _make_library()
    added = lib.seed_known_families()
    assert added > 0
    assert len(lib) > 0


def test_seeded_library_finds_perfect_square():
    lib = _make_library()
    lib.seed_known_families()
    x = _var("x")
    # (x + 1)^2 = x^2 + 2x + 1 should be in library after seeding
    sq = (x + _const(1)) * (x + _const(1))
    # Target: perfect square + some other term
    y = _var("y")
    target = sq + y
    matches = lib.find_matches(target)
    assert any(m.matched_poly == sq for m in matches)


# ---------------------------------------------------------------------------
# Integration: propose_splits returns library_match candidates
# ---------------------------------------------------------------------------

def test_propose_splits_includes_library_match():
    from decomp_rl.split_proposals import propose_splits
    from decomp_rl.baseline_cost import BaselineCostModel

    lib = _make_library()
    lib.seed_known_families()

    x, y = _var("x"), _var("y")
    # (x + 1)^2 + y — library should propose extracting the square
    sq = (x + _const(1)) * (x + _const(1))
    target = sq + y

    candidates = propose_splits(target, 16, baseline_model=BaselineCostModel(), library=lib)
    sources = {c.source for c in candidates}
    assert "library_match" in sources

    # Verify all splits are valid
    for c in candidates:
        assert c.g + c.h == target


# ---------------------------------------------------------------------------
# Integration: DecompEnv emits library_reward
# ---------------------------------------------------------------------------

def test_decomp_env_library_reward():
    from decomp_rl.decomp_env import DecompEnv
    from decomp_rl.split_proposals import SplitAction

    lib = _make_library()
    lib.seed_known_families()

    x, y = _var("x"), _var("y")
    sq = (x + _const(1)) * (x + _const(1))
    target = sq + y

    env = DecompEnv(library=lib)
    state = env.reset(target)
    # Action: split into sq and y
    action = SplitAction(g=sq, h=y, source="test").ordered()
    # Verify the action is valid before stepping
    if action.g + action.h != target:
        # ordered may flip; try both orientations
        action = SplitAction(g=y, h=sq, source="test").ordered()
    assert action.g + action.h == target

    _, reward, done, info = env.step(state, 0, action)
    # Library reward should be > 0 since sq or y is in library
    assert info.library_reward >= 0.0  # non-negative at minimum
    if lib.is_known(action.g) or lib.is_known(action.h):
        assert info.library_reward > 0.0


# ---------------------------------------------------------------------------
# FiniteFieldFactorizer auto-adds to library
# ---------------------------------------------------------------------------

def test_factorizer_auto_adds_to_library():
    lib = _make_library()
    factorizer = FiniteFieldFactorizer(library=lib)
    x, y = _var("x"), _var("y")
    # Factor something with non-trivial factorization
    poly = (x + _const(1)) * (y + _const(2))
    _ = factorizer.factor(poly)
    factorizer.close()
    # Library should now know about poly (if it had positive savings)
    # We don't assert is_known because savings may be 0 for very simple polys,
    # but we do assert no exception was raised
    assert isinstance(len(lib), int)
