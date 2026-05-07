from decomp_rl.andor_search import AndOrSearch
from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.polynomial import SparsePolynomial


def test_andor_search_returns_nonworse_than_baseline():
    variables = ("x", "y")
    p = 3
    x = SparsePolynomial.variable("x", p, variables)
    y = SparsePolynomial.variable("y", p, variables)
    target = x * y + y

    search = AndOrSearch()
    baseline = BaselineCostModel().direct_construction_cost(target)
    result = search.search(target)

    assert result.best_cost <= baseline
    assert result.best_trace.poly == target
    assert result.stats.node_expansions >= 1
    assert result.stats.factor_cache_requests >= result.stats.factor_cache_hits
