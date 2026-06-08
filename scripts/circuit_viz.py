"""
Circuit-graph reconstruction for the inference demo.

Turns a *decomposition* into an explicit arithmetic-circuit tree of
add / mul / pow / leaf nodes, suitable for rendering in the web UI. Two
sources of decomposition are supported:

  * Agent circuits — the additive splits the policy/search chose for each
    sub-polynomial (a ``{poly_key -> SplitAction}`` map), with each split
    piece factored by the environment's CAS factorizer.
  * Optimal/baseline circuits — reconstructed directly from the closed-form
    baselines (sparse-direct, top-down power-pivot, and gap-aware Horner),
    picking whichever yields the cheapest tree.

Node schema (all JSON-serialisable):
    {"op": "add", "latex": str, "children": [...]}              # k-ary +, cost k-1
    {"op": "mul", "latex": str, "scalar": int|None,             # k-ary ×, cost k-1
     "children": [...]}
    {"op": "pow", "latex": str, "exp": int, "children": [node]} # (·)^e, cost rsc(e)
    {"op": "leaf", "latex": str}                                # variable / constant
    {"op": "direct", "latex": str, "cost": int}                 # opaque direct build

The ``cost`` of a node = additions (k-1 for a k-ary add) + multiplications
(k-1 for a k-ary mul; the scalar unit is free) + repeated-squaring ops for
powers. ``circuit_cost`` sums this over the tree and matches the environment /
baseline cost models (modulo DAG sharing, which a tree cannot express).
"""
from __future__ import annotations

from decomp_rl.cost_model import monomial_build_cost, repeated_squaring_cost
from decomp_rl.polynomial import SparsePolynomial


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

def poly_to_latex(poly: SparsePolynomial) -> str:
    if poly.is_zero:
        return "0"
    parts = []
    for coeff, exp in poly.terms:
        mono = []
        for var, p in zip(poly.variables, exp):
            if p == 0:
                continue
            mono.append(var if p == 1 else f"{var}^{{{p}}}")
        m = "".join(mono)
        if not m:
            parts.append(str(coeff))
        elif coeff == 1:
            parts.append(m)
        else:
            parts.append(f"{coeff}{m}")
    return " + ".join(parts)


# ---------------------------------------------------------------------------
# Node constructors / helpers
# ---------------------------------------------------------------------------

def _leaf(text: str) -> dict:
    return {"op": "leaf", "latex": text}


def _is_atomic(poly: SparsePolynomial) -> bool:
    return poly.is_zero or poly.is_constant or poly.is_variable() or poly.is_monomial


def _monomial_circuit(poly: SparsePolynomial) -> dict:
    """Decompose a single monomial c·x^a·y^b·… into pow/mul/leaf nodes."""
    if poly.is_zero or poly.is_constant:
        return _leaf(poly_to_latex(poly))
    coeff, exp = poly.terms[0]
    nodes = []
    for var, e in zip(poly.variables, exp):
        if e == 0:
            continue
        node = _leaf(var)
        if e > 1:
            node = {"op": "pow", "exp": e, "latex": f"{var}^{{{e}}}", "children": [node]}
        nodes.append(node)
    scalar = coeff if coeff != 1 else None
    if not nodes:
        return _leaf(str(coeff))
    if len(nodes) == 1 and scalar is None:
        return nodes[0]
    return {"op": "mul", "latex": poly_to_latex(poly), "scalar": scalar, "children": nodes}


def _atomic_circuit(poly: SparsePolynomial) -> dict:
    if poly.is_zero or poly.is_constant or poly.is_variable():
        return _leaf(poly_to_latex(poly))
    return _monomial_circuit(poly)


def circuit_cost(node: dict) -> int:
    op = node["op"]
    if op == "leaf":
        return 0
    if op == "direct":
        return int(node.get("cost", 0))
    if op == "pow":
        return repeated_squaring_cost(node["exp"]) + circuit_cost(node["children"][0])
    if op in ("add", "mul"):
        kids = node["children"]
        return max(0, len(kids) - 1) + sum(circuit_cost(c) for c in kids)
    return 0


# ---------------------------------------------------------------------------
# New-model gate count.
#
# This is the authoritative cost used everywhere the demo compares circuits:
# the ONLY free constant is 1; every other constant is built by adding 1s
# (reused), scalar multiplication k·E is realised as repeated addition, powers
# as repeated squaring, and subexpression reuse is free (a DAG).  The function
# mirrors scripts/ui/circuit.js exactly, so the number it returns equals the
# number of operation nodes the diagram renders.
# ---------------------------------------------------------------------------

import copy as _copy


def binarize(node: dict) -> dict:
    """Rewrite every k-ary (+, ×) node into nested binary gates.

    The gate model is binary + and ×, so a k-ary node really stands for (k-1)
    gates.  Binarising makes the rendered node count equal the gate count for
    every circuit (the solver's optimal circuits are already binary, so this is
    a no-op for them).  ``scalar`` on a ×-node applies to the whole product, so
    it is lifted onto a unary ×-node wrapping the binarised product.
    """
    if not isinstance(node, dict):
        return node
    op = node.get("op")
    if op in ("leaf", "ref", "direct"):
        return dict(node)
    if op == "pow":
        return {**node, "children": [binarize(node["children"][0])]}
    kids = [binarize(c) for c in node.get("children", [])]
    # Already binary (or unary): keep the node as-is so id / scalar / latex /
    # ref bookkeeping survives.  The solver's optimal circuits land here.
    if len(kids) <= 2:
        return {**node, "children": kids}
    # k-ary (>2): fold into a left-leaning binary chain of the same op.
    if op == "mul":
        scalar = node.get("scalar")
        acc = kids[0]
        for k in kids[1:]:
            acc = {"op": "mul", "latex": "", "children": [acc, k]}
        if scalar and scalar != 1:
            acc = {"op": "mul", "scalar": scalar, "children": [acc]}
    else:  # add
        acc = kids[0]
        for k in kids[1:]:
            acc = {"op": "add", "latex": "", "children": [acc, k]}
    acc["latex"] = node.get("latex", "")
    if "id" in node:
        acc["id"] = node["id"]
    return acc


def _gc_link_refs(root: dict) -> dict:
    defs: dict = {}

    def collect(n):
        if not isinstance(n, dict):
            return
        if "id" in n:
            defs[n["id"]] = n
        for c in n.get("children", []):
            collect(c)

    collect(root)
    seen: set = set()

    def relink(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        if "children" in n:
            n["children"] = [
                defs[c["ref"]] if isinstance(c, dict) and c.get("op") == "ref" else c
                for c in n["children"]
            ]
            for c in n["children"]:
                relink(c)

    relink(root)
    return root


def _gc_build_const(v: int, cc: dict) -> dict:
    if v <= 1:
        return {"op": "leaf", "latex": str(v)}
    if v in cc:
        return cc[v]
    a, b = v // 2, v - v // 2
    node = {"op": "add", "children": [_gc_build_const(a, cc), _gc_build_const(b, cc)]}
    cc[v] = node
    return node


def _gc_expand_pow(base: dict, exp: int) -> dict:
    if exp <= 1:
        return base
    if exp == 2:
        return {"op": "mul", "_share": True, "children": [base]}
    h = _gc_expand_pow(base, exp // 2)
    if exp % 2 == 0:
        return {"op": "mul", "_share": True, "children": [h]}
    return {"op": "mul", "children": [_gc_expand_pow(base, exp - 1), base]}


def _gc_expand_scalar(product: dict, k: int) -> dict:
    if k <= 1:
        return product
    if k == 2:
        return {"op": "add", "_share": True, "children": [product]}
    h = _gc_expand_scalar(product, k // 2)
    if k % 2 == 0:
        return {"op": "add", "_share": True, "children": [h]}
    return {"op": "add", "children": [_gc_expand_scalar(product, k - 1), product]}


def _gc_normalize(node: dict, memo: dict, cc: dict) -> dict:
    nid = id(node)
    if nid in memo:
        return memo[nid]
    op = node["op"]
    if op == "leaf":
        lx = node.get("latex", "")
        m = int(lx) if lx.isdigit() else None
        result = _gc_build_const(m, cc) if (m is not None and m >= 2) else node
        memo[nid] = result
        return result
    if op == "direct":
        memo[nid] = node
        return node
    kids = [_gc_normalize(c, memo, cc) for c in node.get("children", [])]
    if op == "pow":
        base = kids[0] if kids else {"op": "leaf", "latex": "1"}
        result = _gc_expand_pow(base, node.get("exp", 1))
    elif op == "mul":
        if not kids:
            product = {"op": "leaf", "latex": "1"}
        elif len(kids) == 1:
            product = kids[0]
        else:
            product = {"op": "mul", "children": kids}
        k = node.get("scalar")
        result = product if (not k or k == 1) else _gc_expand_scalar(product, k)
    else:  # add
        if not kids:
            result = {"op": "leaf", "latex": "0"}
        elif len(kids) == 1:
            result = kids[0]
        else:
            result = {"op": "add", "children": kids}
    memo[nid] = result
    return result


def gate_count(node: dict) -> int:
    """Number of + / × gates the circuit needs under the demo's gate model.

    Matches scripts/ui/circuit.js (constants built from 1s, scalar/power
    expansion, DAG reuse), so it equals the rendered operation-node count.
    A leftover ``direct`` node (should not occur once circuits are
    materialised) contributes its declared cost.
    """
    root = _gc_link_refs(binarize(_copy.deepcopy(node)))
    root = _gc_normalize(root, {}, {})
    seen: set = set()
    total = 0
    stack = [root]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if n["op"] in ("add", "mul"):
            total += 1
        elif n["op"] == "direct":
            total += int(n.get("cost", 0))
        for c in n.get("children", []):
            stack.append(c)
    return total


# ---------------------------------------------------------------------------
# Agent circuit (from a {poly_key -> SplitAction} decomposition)
# ---------------------------------------------------------------------------

def direct_circuit(poly, baseline_model, cache: dict | None = None) -> dict:
    """Materialise a *direct construction* of ``poly`` as a full + / × circuit.

    Mirrors ``BaselineCostModel.direct_construction_cost`` (the cheaper of the
    sparse build and a multivariate-Horner build), but emits the actual circuit
    rather than an opaque node, picking the variant with the lowest new-model
    ``gate_count`` so the agent is scored on a concrete, well-formed circuit.
    """
    if cache is None:
        cache = {}
    key = poly.to_key()
    if key in cache:
        return cache[key]
    if _is_atomic(poly):
        node = _atomic_circuit(poly)
        cache[key] = node
        return node

    # Provisional sparse entry guards against pathological recursion.
    cache[key] = _sparse_circuit(poly)
    candidates = [_sparse_circuit(poly)]

    # Multivariate Horner: poly = remainder + x_i · quotient, per variable.
    for vi, max_deg in enumerate(poly.max_degrees):
        if max_deg == 0:
            continue
        remainder, quotient = poly.split_by_variable(vi)
        if quotient.is_zero:
            continue
        pivot = poly.variable_factor(vi)
        xnode = _leaf(poly.variables[vi])
        if quotient.is_constant:
            cval = quotient.terms[0][0] if quotient.terms else 1
            xq = pivot * quotient
            mul_node = xnode if cval == 1 else {
                "op": "mul", "latex": poly_to_latex(xq),
                "scalar": cval, "children": [xnode],
            }
        else:
            xq = pivot * quotient
            mul_node = {
                "op": "mul", "latex": poly_to_latex(xq),
                "children": [xnode, direct_circuit(quotient, baseline_model, cache)],
            }
        if remainder.is_zero:
            cand = mul_node
        else:
            cand = {
                "op": "add", "latex": poly_to_latex(poly),
                "children": [direct_circuit(remainder, baseline_model, cache), mul_node],
            }
        candidates.append(cand)

    best = min(candidates, key=gate_count)
    cache[key] = best
    return best


def build_agent_circuit(poly, splits, factorizer, baseline_model,
                        depth: int = 0, max_depth: int = 24) -> dict:
    if _is_atomic(poly):
        return _atomic_circuit(poly)
    split = splits.get(poly.to_key())
    if split is None or depth >= max_depth:
        # Agent solved this sub-polynomial directly: materialise the actual
        # construction circuit (no opaque "direct" node) so it renders fully.
        return direct_circuit(poly, baseline_model)
    if split.kind == "factor":
        return _factor_product(poly, factorizer, splits, baseline_model, depth)
    g_node = _factor_product(split.g, factorizer, splits, baseline_model, depth)
    h_node = _factor_product(split.h, factorizer, splits, baseline_model, depth)
    return {"op": "add", "latex": poly_to_latex(poly), "children": [g_node, h_node]}


def _factor_product(poly, factorizer, splits, baseline_model, depth: int) -> dict:
    if _is_atomic(poly):
        return _atomic_circuit(poly)
    fr = factorizer.factor(poly)
    nodes = []
    for factor, exp in fr.factors:
        if factor.is_constant:
            continue
        node = build_agent_circuit(factor, splits, factorizer, baseline_model, depth + 1)
        if exp > 1:
            node = {
                "op": "pow", "exp": exp,
                "latex": poly_to_latex(factor.pow(exp)),
                "children": [node],
            }
        nodes.append(node)
    scalar = fr.unit if fr.unit != 1 else None
    if not nodes:
        return _atomic_circuit(poly)
    if len(nodes) == 1 and scalar is None:
        return nodes[0]
    return {"op": "mul", "latex": poly_to_latex(poly), "scalar": scalar, "children": nodes}


# ---------------------------------------------------------------------------
# Decomposition capture
# ---------------------------------------------------------------------------

def splits_from_trace(trace, out: dict | None = None) -> dict:
    """Collect {poly_key -> SplitAction} from an AndOrSearch DecompositionTrace."""
    if out is None:
        out = {}
    if getattr(trace, "chosen_action", None) is not None:
        out[trace.poly.to_key()] = trace.chosen_action
    for child in getattr(trace, "children", ()):
        splits_from_trace(child, out)
    return out


def greedy_splits(env, model, poly, k: int = 16, max_steps: int = 64):
    """Replay the greedy policy, returning ({poly_key -> SplitAction}, acc_cost)."""
    state = env.reset(poly)
    splits: dict = {
        info.active_poly.to_key(): info.split
        for info in state.history
        if info.split is not None
    }
    for _ in range(max_steps):
        if not state.frontier:
            break
        active = state.frontier[0]
        candidates = env.get_candidate_splits(state, 0, k=k)
        if not candidates:
            state, _, _, _ = env.solve_direct(state, 0)
        else:
            priors, _ = model.score_candidates(active, candidates)
            best_idx = max(range(len(priors)), key=lambda i: priors[i])
            chosen = candidates[best_idx]
            splits[active.to_key()] = chosen
            state, _, _, _ = env.step(state, 0, chosen)
    while state.frontier:
        state, _, _, _ = env.solve_direct(state, 0)
    return splits, state.acc_cost


# ---------------------------------------------------------------------------
# Optimal / baseline circuit (min over sparse-direct, top-down, gap-Horner)
# ---------------------------------------------------------------------------

def build_optimal_circuit(poly: SparsePolynomial, cache: dict | None = None,
                          factorizer=None) -> dict:
    """Best tree-structured circuit (lowest circuit_cost).

    Pass ``factorizer`` (e.g. ``env.factorizer``) to enable polynomial
    factorisation as an additional strategy, which finds optimal circuits for
    polynomials whose minimal cost arises from a product structure rather than
    from additive splits alone.
    """
    if cache is None:
        cache = {}
    return _opt(poly, cache, factorizer)


def _opt(poly: SparsePolynomial, cache: dict, fac=None) -> dict:
    key = poly.to_key()
    if key in cache:
        return cache[key]
    if _is_atomic(poly):
        node = _atomic_circuit(poly)
        cache[key] = node
        return node
    # Provisional (sparse) entry guards against pathological recursion.
    cache[key] = _sparse_circuit(poly)
    candidates = [_sparse_circuit(poly)]
    candidates.extend(_top_down_candidates(poly, cache, fac))
    candidates.extend(_gap_horner_candidates(poly, cache, fac))
    best = min(candidates, key=circuit_cost)
    cache[key] = best
    return best


def _sparse_circuit(poly: SparsePolynomial) -> dict:
    mono_nodes = [
        _monomial_circuit(SparsePolynomial.from_monomial(c, e, poly.p, poly.variables))
        for c, e in poly.terms
    ]
    if len(mono_nodes) == 1:
        return mono_nodes[0]
    return {"op": "add", "latex": poly_to_latex(poly), "children": mono_nodes}


def _var_power_node(var: str, power: int) -> dict:
    node = _leaf(var)
    if power > 1:
        node = {"op": "pow", "exp": power, "latex": f"{var}^{{{power}}}", "children": [node]}
    return node


def _top_down_candidates(poly: SparsePolynomial, cache: dict, fac=None) -> list[dict]:
    out: list[dict] = []
    for vi, max_deg in enumerate(poly.max_degrees):
        if max_deg == 0:
            continue
        for pivot in range(1, min(max_deg, 3) + 1):
            lower_terms, upper_terms = [], []
            for c, e in poly.terms:
                if e[vi] >= pivot:
                    red = list(e)
                    red[vi] -= pivot
                    upper_terms.append((c, tuple(red)))
                else:
                    lower_terms.append((c, e))
            if not upper_terms or not lower_terms:
                continue
            lower = SparsePolynomial(poly.p, poly.variables, tuple(lower_terms))
            upper = SparsePolynomial(poly.p, poly.variables, tuple(upper_terms))
            pivot_mono = SparsePolynomial.from_monomial(
                1, tuple(pivot if i == vi else 0 for i in range(len(poly.variables))),
                poly.p, poly.variables,
            )
            # Bug fix: when upper is a constant, avoid wrapping pow+leaf inside a
            # 2-child mul (which adds a spurious 1-op cost). Scalar multiples are
            # free, so build `scalar * x^pivot` as a scalar-annotated 1-child mul.
            if upper.is_constant:
                c_val = upper.terms[0][0] if upper.terms else 1
                xnode = _var_power_node(poly.variables[vi], pivot)
                if c_val == 1:
                    mul = xnode
                else:
                    mul = {"op": "mul", "latex": poly_to_latex(pivot_mono * upper),
                           "scalar": c_val, "children": [xnode]}
            else:
                mul = {
                    "op": "mul",
                    "latex": poly_to_latex(pivot_mono * upper),
                    "children": [_var_power_node(poly.variables[vi], pivot),
                                 _opt(upper, cache, fac)],
                }
            out.append({
                "op": "add",
                "latex": poly_to_latex(poly),
                "children": [_opt(lower, cache, fac), mul],
            })
    return out


def _embed(inner_poly: SparsePolynomial, full_vars: tuple[str, ...], vi: int) -> SparsePolynomial:
    """Lift a polynomial in the inner variables back into the full variable set."""
    terms = []
    for c, ie in inner_poly.terms:
        fe = list(ie[:vi]) + [0] + list(ie[vi:])
        terms.append((c, tuple(fe)))
    return SparsePolynomial(inner_poly.p, full_vars, tuple(terms))


def _gap_horner_candidates(poly: SparsePolynomial, cache: dict, fac=None) -> list[dict]:
    out: list[dict] = []
    for vi in range(len(poly.variables)):
        node = _gap_horner_for_var(poly, vi, cache, fac)
        if node is not None:
            out.append(node)
    return out


def _gap_horner_for_var(poly: SparsePolynomial, vi: int, cache: dict, fac=None) -> dict | None:
    groups: dict[int, list] = {}
    for c, e in poly.terms:
        oe = e[vi]
        inner = tuple(v for i, v in enumerate(e) if i != vi)
        groups.setdefault(oe, []).append((c, inner))
    if len(groups) <= 1:
        return None

    full_vars = poly.variables
    inner_vars = tuple(v for i, v in enumerate(full_vars) if i != vi)
    var = full_vars[vi]
    sorted_exps = sorted(groups)

    coeff_poly: dict[int, SparsePolynomial] = {}
    coeff_circ: dict[int, dict] = {}
    for oe in sorted_exps:
        cp = SparsePolynomial(poly.p, inner_vars, tuple(groups[oe]))
        coeff_poly[oe] = cp
        coeff_circ[oe] = _opt(cp, cache, fac)

    x_full = SparsePolynomial.from_monomial(
        1, tuple(1 if i == vi else 0 for i in range(len(full_vars))), poly.p, full_vars,
    )
    top = sorted_exps[-1]
    leading_scalar = coeff_poly[top].is_constant

    acc = coeff_circ[top]
    acc_poly = _embed(coeff_poly[top], full_vars, vi)

    for step_idx, j in enumerate(range(len(sorted_exps) - 1, 0, -1)):
        gap = sorted_exps[j] - sorted_exps[j - 1]
        acc_poly = acc_poly * x_full.pow(gap)
        if step_idx == 0 and leading_scalar:
            # acc is a bare scalar c: c·x^gap is a free monomial (only rsc(gap) charged).
            scalar = coeff_poly[top].terms[0][0] if not coeff_poly[top].is_zero else 1
            xnode = _var_power_node(var, gap)
            if scalar == 1:
                acc = xnode
            else:
                acc = {"op": "mul", "latex": poly_to_latex(acc_poly),
                       "scalar": scalar, "children": [xnode]}
        else:
            acc = {"op": "mul", "latex": poly_to_latex(acc_poly),
                   "children": [acc, _var_power_node(var, gap)]}
        lower_exp = sorted_exps[j - 1]
        acc_poly = acc_poly + _embed(coeff_poly[lower_exp], full_vars, vi)
        acc = {"op": "add", "latex": poly_to_latex(acc_poly),
               "children": [acc, coeff_circ[lower_exp]]}

    if sorted_exps[0] > 0:
        acc_poly = acc_poly * x_full.pow(sorted_exps[0])
        acc = {"op": "mul", "latex": poly_to_latex(acc_poly),
               "children": [acc, _var_power_node(var, sorted_exps[0])]}
    return acc


# ---------------------------------------------------------------------------
# Reference circuits for all 21 test polynomials (hard-coded).
#
# Most entries were generated by an exhaustive min-DAG-size search under the
# demo's gate model; larger rows may use a known compact construction:
#   * Free leaves are the variables and the constant 1 ONLY.
#   * Every other constant must be built by adding 1s (e.g. 2 = 1+1), reused.
#   * Gates are binary + and x; subexpression reuse is free (a DAG).
# The stored "cost" equals the number of operation nodes the diagram renders,
# and each circuit is verified to compute the target polynomial.  Shared nodes
# use id/ref so the renderer draws one node with fan-out edges.
# ---------------------------------------------------------------------------

OPTIMAL_CIRCUITS: dict[str, dict] = {
    'F3_xy  xy+x+y': {"cost": 3, "circuit": {"op":"add","latex":"xy + y + x","children":[{"op":"leaf","latex":"y"},{"op":"mul","latex":"xy + x","children":[{"op":"leaf","latex":"x"},{"op":"add","latex":"y + 1","children":[{"op":"leaf","latex":"1"},{"op":"leaf","latex":"y"}]}]}]}},
    'F3_xy  x2y+xy2 [=xy(x+y)]': {"cost": 3, "circuit": {"op":"mul","latex":"xy^{2} + x^{2}y","children":[{"op":"mul","latex":"xy","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"x"}]},{"op":"add","latex":"y + x","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"x"}]}]}},
    'F3_xy  x2+2xy+y2 [=(x+y)^2]': {"cost": 2, "circuit": {"op":"mul","latex":"y^{2} + 2xy + x^{2}","children":[{"op":"add","latex":"y + x","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"x"}],"id":"n0"},{"op":"ref","ref":"n0"}]}},
    'F3_xy  x2y2+xy+1': {"cost": 4, "circuit": {"op":"mul","latex":"x^{2}y^{2} + xy + 1","children":[{"op":"add","latex":"xy + 2","children":[{"op":"mul","latex":"xy","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"x"}]},{"op":"add","latex":"2","children":[{"op":"leaf","latex":"1"},{"op":"leaf","latex":"1"}]}],"id":"n0"},{"op":"ref","ref":"n0"}]}},
    'F3_xy  x2+xy+y2+x+y': {"cost": 4, "circuit": {"op":"add","latex":"y^{2} + xy + x^{2} + y + x","children":[{"op":"add","latex":"y + x","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"x"}],"id":"n0"},{"op":"mul","latex":"y^{2} + xy + x^{2}","children":[{"op":"add","latex":"2y + x","children":[{"op":"leaf","latex":"y"},{"op":"ref","ref":"n0"}],"id":"n1"},{"op":"ref","ref":"n1"}]}]}},
    'F3_xy  x3+y3 [sum of cubes]': {"cost": 3, "circuit": {"op":"mul","latex":"y^{3} + x^{3}","children":[{"op":"mul","latex":"y^{2} + 2xy + x^{2}","children":[{"op":"add","latex":"y + x","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"x"}],"id":"n0"},{"op":"ref","ref":"n0"}]},{"op":"ref","ref":"n0"}]}},
    'F3_xy  x2y+xy+x+y': {"cost": 5, "circuit": {"op":"add","latex":"x^{2}y + xy + y + x","children":[{"op":"leaf","latex":"x"},{"op":"mul","latex":"x^{2}y + xy + y","children":[{"op":"mul","latex":"x^{2} + x + 1","children":[{"op":"add","latex":"x + 2","children":[{"op":"leaf","latex":"x"},{"op":"add","latex":"2","children":[{"op":"leaf","latex":"1"},{"op":"leaf","latex":"1"}]}],"id":"n0"},{"op":"ref","ref":"n0"}]},{"op":"leaf","latex":"y"}]}]}},
    'F3_xy  x2y2+x2+y2': {"cost": 5, "circuit": {"op":"add","latex":"x^{2}y^{2} + y^{2} + x^{2}","children":[{"op":"mul","latex":"x^{2}y^{2} + x^{2}","children":[{"op":"leaf","latex":"x"},{"op":"mul","latex":"xy^{2} + x","children":[{"op":"add","latex":"y^{2} + 1","children":[{"op":"leaf","latex":"1"},{"op":"mul","latex":"y^{2}","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"y"}],"id":"n0"}]},{"op":"leaf","latex":"x"}]}]},{"op":"ref","ref":"n0"}]}},
    'F3_x   x4+x3+x2+x+1': {"cost": 5, "circuit": {"op":"add","latex":"x^{4} + x^{3} + x^{2} + x + 1","children":[{"op":"mul","latex":"x^{4} + x^{3} + x + 1","children":[{"op":"mul","latex":"x^{2} + 2x + 1","children":[{"op":"add","latex":"x + 1","children":[{"op":"leaf","latex":"1"},{"op":"leaf","latex":"x"}],"id":"n1"},{"op":"ref","ref":"n1"}],"id":"n0"},{"op":"ref","ref":"n0"}]},{"op":"mul","latex":"x^{2}","children":[{"op":"leaf","latex":"x"},{"op":"leaf","latex":"x"}]}]}},
    'F3_x   x4+2x2+1 [=(x2+1)^2]': {"cost": 3, "circuit": {"op":"mul","latex":"x^{4} + 2x^{2} + 1","children":[{"op":"add","latex":"x^{2} + 1","children":[{"op":"leaf","latex":"1"},{"op":"mul","latex":"x^{2}","children":[{"op":"leaf","latex":"x"},{"op":"leaf","latex":"x"}]}],"id":"n0"},{"op":"ref","ref":"n0"}]}},
    'F3_x   x3+2x2+2x+1': {"cost": 4, "circuit": {"op":"add","latex":"x^{3} + 2x^{2} + 2x + 1","children":[{"op":"add","latex":"x + 1","children":[{"op":"leaf","latex":"1"},{"op":"leaf","latex":"x"}],"id":"n0"},{"op":"mul","latex":"x^{3} + 2x^{2} + x","children":[{"op":"mul","latex":"x^{2} + 2x + 1","children":[{"op":"ref","ref":"n0"},{"op":"ref","ref":"n0"}]},{"op":"leaf","latex":"x"}]}]}},
    'F3_x   2x4+x3+x+2': {"cost": 5, "circuit": {"op":"mul","latex":"2x^{4} + x^{3} + x + 2","children":[{"op":"mul","latex":"2x^{2} + 2x + 2","children":[{"op":"add","latex":"2","children":[{"op":"leaf","latex":"1"},{"op":"leaf","latex":"1"}],"id":"n0"},{"op":"mul","latex":"x^{2} + x + 1","children":[{"op":"add","latex":"x + 2","children":[{"op":"ref","ref":"n0"},{"op":"leaf","latex":"x"}],"id":"n2"},{"op":"ref","ref":"n2"}],"id":"n1"}]},{"op":"ref","ref":"n1"}]}},
    'F3_xyz xy+xz+yz [e2]': {"cost": 4, "circuit": {"op":"add","latex":"yz + xz + xy","children":[{"op":"mul","latex":"yz + xz","children":[{"op":"add","latex":"y + x","children":[{"op":"leaf","latex":"x"},{"op":"leaf","latex":"y"}]},{"op":"leaf","latex":"z"}]},{"op":"mul","latex":"xy","children":[{"op":"leaf","latex":"x"},{"op":"leaf","latex":"y"}]}]}},
    'F3_xyz xyz+xy+xz+yz': {"cost": 5, "circuit": {"op":"add","latex":"xyz + yz + xz + xy","children":[{"op":"mul","latex":"xyz + yz + xz","children":[{"op":"add","latex":"xy + y + x","children":[{"op":"add","latex":"y + x","children":[{"op":"leaf","latex":"x"},{"op":"leaf","latex":"y"}]},{"op":"mul","latex":"xy","children":[{"op":"leaf","latex":"x"},{"op":"leaf","latex":"y"}],"id":"n0"}]},{"op":"leaf","latex":"z"}]},{"op":"ref","ref":"n0"}]}},
    'F3_xyz x2+y2+z2+xy+xz+yz': {"cost": 6, "circuit": {"op":"add","latex":"z^{2} + yz + y^{2} + xz + xy + x^{2}","children":[{"op":"mul","latex":"z^{2} + yz + xz","children":[{"op":"leaf","latex":"z"},{"op":"add","latex":"z + y + x","children":[{"op":"add","latex":"y + x","children":[{"op":"leaf","latex":"x"},{"op":"leaf","latex":"y"}],"id":"n0"},{"op":"leaf","latex":"z"}]}]},{"op":"mul","latex":"y^{2} + xy + x^{2}","children":[{"op":"add","latex":"y + 2x","children":[{"op":"leaf","latex":"x"},{"op":"ref","ref":"n0"}],"id":"n1"},{"op":"ref","ref":"n1"}]}]}},
    'F3_xyz xyz+x+y+z': {"cost": 5, "circuit": {"op":"add","latex":"xyz + z + y + x","children":[{"op":"mul","latex":"xyz","children":[{"op":"mul","latex":"xy","children":[{"op":"leaf","latex":"x"},{"op":"leaf","latex":"y"}]},{"op":"leaf","latex":"z"}]},{"op":"add","latex":"z + y + x","children":[{"op":"add","latex":"y + x","children":[{"op":"leaf","latex":"x"},{"op":"leaf","latex":"y"}]},{"op":"leaf","latex":"z"}]}]}},
    'F3_perm3  3x3 permanent': {"cost": 14, "circuit": {"op":"add","latex":"aei + afh + bdi + bfg + cdh + ceg","children":[{"op":"add","latex":"a(ei + fh) + b(di + fg)","children":[{"op":"mul","latex":"a(ei + fh)","children":[{"op":"leaf","latex":"a"},{"op":"add","latex":"ei + fh","children":[{"op":"mul","latex":"ei","children":[{"op":"leaf","latex":"e"},{"op":"leaf","latex":"i"}]},{"op":"mul","latex":"fh","children":[{"op":"leaf","latex":"f"},{"op":"leaf","latex":"h"}]}]}]},{"op":"mul","latex":"b(di + fg)","children":[{"op":"leaf","latex":"b"},{"op":"add","latex":"di + fg","children":[{"op":"mul","latex":"di","children":[{"op":"leaf","latex":"d"},{"op":"leaf","latex":"i"}]},{"op":"mul","latex":"fg","children":[{"op":"leaf","latex":"f"},{"op":"leaf","latex":"g"}]}]}]}]},{"op":"mul","latex":"c(dh + eg)","children":[{"op":"leaf","latex":"c"},{"op":"add","latex":"dh + eg","children":[{"op":"mul","latex":"dh","children":[{"op":"leaf","latex":"d"},{"op":"leaf","latex":"h"}]},{"op":"mul","latex":"eg","children":[{"op":"leaf","latex":"e"},{"op":"leaf","latex":"g"}]}]}]}]}},
    'F5_xy  x2+4y2 [=(x+y)(x+4y)]': {"cost": 4, "circuit": {"op":"add","latex":"4y^{2} + x^{2}","children":[{"op":"mul","latex":"x^{2}","children":[{"op":"leaf","latex":"x"},{"op":"leaf","latex":"x"}]},{"op":"mul","latex":"4y^{2}","children":[{"op":"add","latex":"2y","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"y"}],"id":"n0"},{"op":"ref","ref":"n0"}]}]}},
    'F5_xy  x2y+xy2+x+y [=(x+y)(xy+1)]': {"cost": 4, "circuit": {"op":"mul","latex":"xy^{2} + x^{2}y + y + x","children":[{"op":"add","latex":"xy + 1","children":[{"op":"mul","latex":"xy","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"x"}]},{"op":"leaf","latex":"1"}]},{"op":"add","latex":"y + x","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"x"}]}]}},
    'F5_xy  x2+xy+y2': {"cost": 4, "circuit": {"op":"add","latex":"y^{2} + xy + x^{2}","children":[{"op":"mul","latex":"xy + x^{2}","children":[{"op":"add","latex":"y + x","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"x"}]},{"op":"leaf","latex":"x"}]},{"op":"mul","latex":"y^{2}","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"y"}]}]}},
    'F5_xy  x3y+xy3 [=xy(x2+y2)]': {"cost": 5, "circuit": {"op":"mul","latex":"xy^{3} + x^{3}y","children":[{"op":"mul","latex":"xy","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"x"}]},{"op":"add","latex":"y^{2} + x^{2}","children":[{"op":"mul","latex":"y^{2}","children":[{"op":"leaf","latex":"y"},{"op":"leaf","latex":"y"}]},{"op":"mul","latex":"x^{2}","children":[{"op":"leaf","latex":"x"},{"op":"leaf","latex":"x"}]}]}]}},
}
