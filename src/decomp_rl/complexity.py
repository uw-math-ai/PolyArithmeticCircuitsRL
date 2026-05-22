"""Arithmetic-circuit complexity (Ck) of a sparse polynomial.

We define ``Ck`` as the number of operations (one polynomial add or one
polynomial multiply = 1 op; scalar multiplication and bare variables are free)
in the *shortest circuit* the search below can find. Two construction moves are
considered at every node, which is what lets a perfect square reach its true
optimum:

  1. **Whole-polynomial factorization** — factor ``f`` over F_p (FLINT backend)
     and pay ``rebuild_cost`` to recombine the irreducible factors, plus the
     cost of building each distinct factor. This is the move that makes
     ``x^2 + 2xy + y^2 = (x + y)^2`` cost 2 (one add to build ``x + y``, one
     multiply to square it) — i.e. C2.
  2. **Additive split** ``f = g + h`` — pay 1 for the add, factor each piece,
     and recurse on the unresolved factor children.

The recursion is memoized and every child is strictly smaller than its parent
(proper factors / split halves), so it terminates.

``circuit_complexity`` is the *hybrid* entry point: it runs the exact search
for small polynomials and falls back to the multi-baseline upper bound
(``BaselineBundle.min_cost``) for large ones, returning ``(ck, method)`` where
``method`` is ``"exact"`` or ``"heuristic"``.
"""

from __future__ import annotations

from .baseline_cost import BaselineCostModel
from .baselines import BaselineBundle
from .cost_model import rebuild_cost, unresolved_children
from .factor_fp import FiniteFieldFactorizer
from .polynomial import SparsePolynomial
from .split_proposals import propose_splits

# A polynomial is "small enough" for the exact search when both its support
# size and total degree are within these bounds. Beyond this the branching of
# the exact search is not worth it and we use the heuristic upper bound.
DEFAULT_EXACT_SUPPORT_LIMIT = 7
DEFAULT_EXACT_DEGREE_LIMIT = 6


def exact_circuit_complexity(
    poly: SparsePolynomial,
    factorizer: FiniteFieldFactorizer,
    baseline_model: BaselineCostModel,
    k_candidates: int = 12,
    memo: dict[str, int] | None = None,
) -> int:
    """Minimal op count over {direct build, whole-poly factor, additive split}.

    Memoized exhaustive search; optimal within these three moves (a strong
    upper bound on the true arithmetic-circuit complexity).
    """
    if memo is None:
        memo = {}
    key = poly.to_key()
    cached = memo.get(key)
    if cached is not None:
        return cached

    if baseline_model.is_base_case(poly):
        value = int(baseline_model.exact_base_cost(poly))
        memo[key] = value
        return value

    # Provisional upper bound (direct construction) breaks any defensive cycle
    # where a child canonicalizes back to the parent before we finish.
    best = int(baseline_model.sparse_direct_cost(poly))
    memo[key] = best

    # Move 1: factor the whole polynomial.
    factorization = factorizer.factor(poly)
    children = unresolved_children(factorization)
    # Non-trivial only if it is not "poly is its own sole irreducible factor".
    nontrivial = not (
        len(children) == 1
        and children[0].to_key() == key
        and rebuild_cost(factorization) == 0
    )
    if children and nontrivial and all(c.to_key() != key for c in children):
        cost = rebuild_cost(factorization) + sum(
            exact_circuit_complexity(c, factorizer, baseline_model, k_candidates, memo)
            for c in children
        )
        best = min(best, cost)

    # Move 2: additive splits f = g + h.
    for action in propose_splits(poly, k_candidates, baseline_model=baseline_model):
        g_factors = factorizer.factor(action.g)
        h_factors = factorizer.factor(action.h)
        child_map: dict[str, SparsePolynomial] = {}
        cyclic = False
        for child in unresolved_children(g_factors) + unresolved_children(h_factors):
            if child.to_key() == key:
                cyclic = True
                break
            child_map.setdefault(child.to_key(), child)
        if cyclic:
            continue
        cost = 1 + rebuild_cost(g_factors) + rebuild_cost(h_factors) + sum(
            exact_circuit_complexity(c, factorizer, baseline_model, k_candidates, memo)
            for c in child_map.values()
        )
        best = min(best, cost)

    memo[key] = best
    return best


def circuit_complexity(
    poly: SparsePolynomial,
    factorizer: FiniteFieldFactorizer,
    baseline_model: BaselineCostModel | None = None,
    bundle: BaselineBundle | None = None,
    exact_support_limit: int = DEFAULT_EXACT_SUPPORT_LIMIT,
    exact_degree_limit: int = DEFAULT_EXACT_DEGREE_LIMIT,
    k_candidates: int = 12,
) -> tuple[int, str]:
    """Hybrid Ck: exact search for small polys, heuristic bound for large.

    Returns ``(ck, method)`` with ``method in {"exact", "heuristic"}``.
    """
    baseline_model = baseline_model or BaselineCostModel()
    if (
        poly.support_size <= exact_support_limit
        and poly.total_degree <= exact_degree_limit
    ):
        ck = exact_circuit_complexity(poly, factorizer, baseline_model, k_candidates)
        return ck, "exact"
    bundle = bundle or BaselineBundle(baseline_model=baseline_model)
    return int(bundle.min_cost(poly)), "heuristic"
