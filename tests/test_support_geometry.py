from math import isfinite

from lgs.env.action import Action
from lgs.env.circuit_state import CircuitState
from lgs.models.feature_encoder import CandidateFeatureEncoder
from lgs.poly.fast_poly import FastPoly
from lgs.poly.support_geometry import (
    support_affine_dimension,
    support_bounding_box_volume,
    support_minkowski_coverage,
    support_minkowski_sum,
)
from lgs.search.candidate_generator import generate_candidates
from lgs.data.target_generators import (
    make_product_of_sums_instance,
    make_square_instance,
)


def test_minkowski_support_covers_square_target_exactly():
    x = FastPoly.variable(0, 2, 4, 17)
    y = FastPoly.variable(1, 2, 4, 17)
    candidate = x + y
    target = candidate * candidate

    minkowski = support_minkowski_sum(candidate.support(), candidate.support())
    coverage, outside_fraction = support_minkowski_coverage(
        candidate.support(),
        candidate.support(),
        target.support(),
    )

    assert minkowski == target.support()
    assert coverage == 1.0
    assert outside_fraction == 0.0


def test_minkowski_support_covers_product_of_sums_target_exactly():
    a = FastPoly.variable(0, 4, 4, 17)
    b = FastPoly.variable(1, 4, 4, 17)
    c = FastPoly.variable(2, 4, 4, 17)
    d = FastPoly.variable(3, 4, 4, 17)
    left = a + b
    right = c + d
    target = left * right

    coverage, outside_fraction = support_minkowski_coverage(
        left.support(),
        right.support(),
        target.support(),
    )

    assert coverage == 1.0
    assert outside_fraction == 0.0


def test_bad_product_has_outside_support():
    a = FastPoly.variable(0, 4, 4, 17)
    b = FastPoly.variable(1, 4, 4, 17)
    c = FastPoly.variable(2, 4, 4, 17)
    d = FastPoly.variable(3, 4, 4, 17)
    target = (a + b) * (c + d)

    coverage, outside_fraction = support_minkowski_coverage(
        (a + b).support(),
        (a + c).support(),
        target.support(),
    )

    assert coverage > 0.0
    assert outside_fraction > 0.0


def test_support_shape_features_are_simple_and_deterministic():
    support = {(0, 0), (1, 0), (0, 1), (1, 1)}

    assert support_affine_dimension(support) == 2
    assert support_bounding_box_volume(support) == 4.0
    assert support_affine_dimension(set()) == 0
    assert support_bounding_box_volume(set()) == 0.0


def test_candidate_generator_populates_support_geometry_features():
    instance = make_square_instance(
        ("x", "y"),
        field_p=17,
        degree_cap=4,
        op_budget=2,
    )
    state = CircuitState.initial(instance)
    candidates = generate_candidates(instance, state, K=64, tier2_m=128)
    action = Action.make("add", 0, 1)
    candidate = next(item for item in candidates if item.action == action)

    assert candidate.features["max_mul_support_coverage"] == 1.0
    assert candidate.features["min_mul_outside_fraction"] == 0.0
    assert "support_mul_coverage" in candidate.source_tags


def test_feature_encoder_includes_support_geometry_features():
    instance = make_product_of_sums_instance(
        ("a", "b"),
        ("c", "d"),
        field_p=17,
        degree_cap=4,
        op_budget=3,
    )
    state = CircuitState.initial(instance)
    candidate = generate_candidates(instance, state, K=64, tier2_m=128)[0]
    encoder = CandidateFeatureEncoder()
    values = encoder.encode(instance, state, candidate)

    for name in (
        "max_mul_support_coverage",
        "min_mul_outside_fraction",
        "max_add_support_coverage",
        "min_add_outside_fraction",
        "support_affine_dim_result",
        "support_bbox_volume_result",
    ):
        assert f"feature_{name}" in encoder.feature_names
    assert len(values) == len(encoder.feature_names)
    assert all(isfinite(value) for value in values)
