"""Per-complexity baseline table.

Targets are sampled from gumbel/src/game_board/generator.generate_random_circuit:
for each c in 1..MAX_DEPTH, run a length-c random walk over (variables + 1)
combining nodes with add/mul. The resulting polynomial is REACHABLE in c ops
by construction, so c is an upper bound on its true minimum complexity.
(BFS would give the exact minimum but is intractable past depth ~6 on a
laptop, so we deliberately skip it.)

For each target we run all 5 baselines and tabulate per-level means. The
"best ub" column = min over the five baselines = our tightest upper bound on
the polynomial's actual circuit cost. Comparing best_ub to c tells you
whether the random-walk label looks tight or loose.
"""

import sys
import statistics
import time
from random import Random, sample as random_sample

sys.path.insert(0, "/Users/kj/Documents/Poly/top-down/src")
sys.path.insert(0, "/Users/kj/Documents/Poly/gumbel")

import numpy as np

from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.baselines import (
    BivariateHornerBaseline,
    CSEBaseline,
    TopDownSearchBaseline,
)
from decomp_rl.polynomial import SparsePolynomial

from src.config import Config
from src.environment.fast_polynomial import FastPoly
from src.game_board.generator import generate_random_circuit


# ----------------------------- knobs -----------------------------------------

MOD = 3
N_VARS = 2
MAX_DEPTH = 10
N_PER_LEVEL = 200
MAX_ATTEMPTS_PER_LEVEL = 2000   # sampling cap per level (dedup may stall)

VARIABLES = ("x0", "x1")
SEED = 0


# ----------------------------- conversion ------------------------------------


def fastpoly_to_sparse(fp: FastPoly) -> SparsePolynomial:
    """Dense numpy coeff array -> SparsePolynomial in F_p."""
    terms = []
    coeffs = fp.coeffs
    for idx in zip(*np.nonzero(coeffs)):
        c = int(coeffs[idx])
        terms.append((c, tuple(int(e) for e in idx)))
    return SparsePolynomial(int(fp.mod), VARIABLES, tuple(terms))


# ----------------------------- main ------------------------------------------


def evaluate_target(sp: SparsePolynomial, base, mv, cse, td) -> dict:
    return {
        "support":         sp.support_size,
        "total_degree":    sp.total_degree,
        "sparse_direct":   base.sparse_direct_cost(sp),
        "horner_one_step": base.horner_upper_bound(sp),
        "mv_horner":       mv.cost(sp),
        "cse":             cse.cost(sp),
        "top_down":        td.cost(sp),
    }


def main() -> None:
    base = BaselineCostModel()
    mv = BivariateHornerBaseline()
    cse = CSEBaseline()
    td = TopDownSearchBaseline()

    print(
        f"Sampling random circuits: n_vars={N_VARS}, mod={MOD}, "
        f"levels c1..c{MAX_DEPTH}, target {N_PER_LEVEL} unique polys per level.",
        flush=True,
    )
    print(flush=True)

    # generate_random_circuit uses the global `random` module; seed it for
    # reproducibility.
    import random as _random_module
    _random_module.seed(SEED)

    by_level: dict[int, list[SparsePolynomial]] = {}
    for c in range(1, MAX_DEPTH + 1):
        cfg = Config(n_variables=N_VARS, mod=MOD, max_complexity=c)
        sps: list[SparsePolynomial] = []
        seen: set[str] = set()
        attempts = 0
        t0 = time.time()
        while len(sps) < N_PER_LEVEL and attempts < MAX_ATTEMPTS_PER_LEVEL:
            attempts += 1
            fp, _actions = generate_random_circuit(cfg, complexity=c)
            sp = fastpoly_to_sparse(fp)
            if sp.is_zero:
                continue
            key = sp.to_key()
            if key in seen:
                continue
            seen.add(key)
            sps.append(sp)
        by_level[c] = sps
        print(
            f"  c={c:<2}  collected {len(sps):>3} unique polys "
            f"in {attempts:>4} attempts  ({time.time()-t0:.1f}s)",
            flush=True,
        )

    # ---- Run baselines and tabulate ----
    print(flush=True)
    print(
        f"{'level':<6} {'n':>4}  {'support':>7} {'tot_deg':>7}  "
        f"{'sparse':>7} {'horner':>7} {'mvHorn':>7} {'cse':>7} {'topdn':>7}    "
        f"{'best ub':>7} {'gap':>6}",
        flush=True,
    )
    print("-" * 100, flush=True)

    for c in range(1, MAX_DEPTH + 1):
        pool = by_level.get(c, [])
        if not pool:
            print(f"c{c:<5} {0:>4}", flush=True)
            continue
        rows = [evaluate_target(sp, base, mv, cse, td) for sp in pool]

        def m(field):
            return statistics.mean(r[field] for r in rows)

        best = [
            min(r["sparse_direct"], r["horner_one_step"],
                r["mv_horner"], r["cse"], r["top_down"])
            for r in rows
        ]
        best_mean = statistics.mean(best)
        gap = best_mean - c
        print(
            f"c{c:<5} {len(rows):>4}  "
            f"{m('support'):>7.2f} {m('total_degree'):>7.2f}  "
            f"{m('sparse_direct'):>7.2f} {m('horner_one_step'):>7.2f} "
            f"{m('mv_horner'):>7.2f} {m('cse'):>7.2f} {m('top_down'):>7.2f}    "
            f"{best_mean:>7.2f} {gap:>+6.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
