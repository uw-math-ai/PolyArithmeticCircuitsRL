from decomp_rl.polynomial import SparsePolynomial
from decomp_rl.split_proposals import SplitAction


def test_canonical_merges_duplicate_terms_and_hashes_stably():
    variables = ("x", "y")
    poly_a = SparsePolynomial.from_terms(((1, (1, 0)), (2, (1, 0))), 3, variables)
    poly_b = SparsePolynomial.zero(3, variables)
    assert poly_a == poly_b
    assert poly_a.to_key() == poly_b.to_key()


def test_split_action_orders_pairs():
    variables = ("x",)
    left = SparsePolynomial.from_monomial(1, (1,), 3, variables)
    right = SparsePolynomial.from_monomial(1, (0,), 3, variables)
    ordered = SplitAction(g=left, h=right, source="test").ordered()
    assert ordered.key()[0] <= ordered.key()[1]

