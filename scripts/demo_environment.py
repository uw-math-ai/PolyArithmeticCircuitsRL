#!/usr/bin/env python3
"""Demo a few decomposition-environment rollouts with CAS-backed factorization."""

from __future__ import annotations

from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.config import FactorizerConfig
from decomp_rl.decomp_env import DecompEnv
from decomp_rl.factor_fp import FiniteFieldFactorizer
from decomp_rl.family_generators import (
    elementary_symmetric_example,
    horner_example,
)
from decomp_rl.frontier_policy import choose_frontier_index
from decomp_rl.polynomial import SparsePolynomial


def rollout_example(name, target, max_steps: int = 8) -> None:
    factorizer = FiniteFieldFactorizer(FactorizerConfig(backend_name="auto"))
    baseline = BaselineCostModel()
    env = DecompEnv(factorizer=factorizer, baseline_model=baseline)
    state = env.reset(target)

    print(f"\n=== {name} ===")
    print(f"target: {target}")
    print(f"baseline direct cost: {baseline.direct_construction_cost(target)}")

    step = 0
    while state.frontier and step < max_steps:
        poly_index = choose_frontier_index(state.frontier, baseline)
        active = state.frontier[poly_index]
        candidates = env.get_candidate_splits(state, poly_index, k=8)
        print(f"\nstep {step + 1}")
        print(f"active: {active}")
        print(f"frontier size: {len(state.frontier)}")
        if candidates:
            top_summaries = [
                f"{candidate.source}:{candidate.score_hint:.2f}"
                for candidate in candidates[:3]
            ]
            print(f"top candidates: {top_summaries}")

        if not candidates or candidates[0].score_hint <= 0:
            state, reward, done, info = env.solve_direct(state, poly_index)
            print(f"action: direct solve")
            print(f"direct cost: {info.direct_cost}")
        else:
            chosen = candidates[0]
            state, reward, done, info = env.step(state, poly_index, chosen)
            print(f"action: split via {chosen.source}")
            print(f"g: {chosen.g}")
            print(f"h: {chosen.h}")
            print(f"g backend: {info.g_factorization.backend if info.g_factorization else 'n/a'}")
            print(f"g factors: {info.g_factorization.factors if info.g_factorization else ()}")
            print(f"h factors: {info.h_factorization.factors if info.h_factorization else ()}")
            print(f"reward: {reward}")
            print(f"new children: {[str(child) for child in info.children]}")
        print(f"accumulated cost: {state.acc_cost}")
        if done:
            break
        step += 1

    print(f"final frontier size: {len(state.frontier)}")
    print(f"final accumulated cost: {state.acc_cost}")
    factorizer.close()


def main() -> None:
    variables = ("x", "y")
    p = 3
    x = SparsePolynomial.variable("x", p, variables)
    y = SparsePolynomial.variable("y", p, variables)
    examples = [
        ("Simple Factorable", x * y + y),
        ("Horner Trace", horner_example([1, 2, 0, 1], 3).target),
        ("Elementary Symmetric", elementary_symmetric_example(4, 2, 3).target),
    ]
    for name, target in examples:
        rollout_example(name, target)


if __name__ == "__main__":
    main()
