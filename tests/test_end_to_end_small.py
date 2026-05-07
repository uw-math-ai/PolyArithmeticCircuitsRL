from random import Random

from decomp_rl.andor_search import AndOrSearch
from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.family_generators import planted_factorable_example


def test_end_to_end_small_structured_target():
    rng = Random(0)
    example = planted_factorable_example(rng, 3, ("x", "y"))
    search = AndOrSearch()
    baseline = BaselineCostModel().direct_construction_cost(example.target)
    result = search.search(example.target)

    assert result.best_trace.poly == example.target
    assert result.best_cost <= baseline
    assert len(result.root_candidates) == len(result.root_policy)
