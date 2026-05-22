#!/usr/bin/env python3
"""Evaluate a trained policy on the *unfiltered* random-polynomial distribution.

The Ck curriculum is bucket-balanced and Ck-filtered, which biases it toward
small, solvable instances. This script instead samples random sparse
polynomials with NO Ck filtering, computes the exact shortest-circuit cost Ck
for each one where the exact search is tractable (support/degree within
limits), and reports the trained policy's greedy Ck-match rate vs a uniform
random split policy on that natural distribution.

It answers: does the success rate hold up away from the curated benchmark?

Example:
    python scripts/eval_unfiltered.py \
        --checkpoint artifacts/ck_curriculum/mcts-1000/checkpoints/final.pt \
        --n-samples 400 --max-support 7 --max-degree 4 --seed 123
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from random import Random

import torch

from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.complexity import (
    DEFAULT_EXACT_DEGREE_LIMIT,
    DEFAULT_EXACT_SUPPORT_LIMIT,
    exact_circuit_complexity,
)
from decomp_rl.config import DecompEnvConfig, FactorizerConfig
from decomp_rl.decomp_env import DecompEnv
from decomp_rl.evaluate import random_rollout_cost
from decomp_rl.factor_fp import FiniteFieldFactorizer
from decomp_rl.factor_library import FactorizableLibrary
from decomp_rl.family_generators import random_sparse_polynomial
from decomp_rl.model import (
    TorchPolicyValueNetwork,
    candidate_feature_vector,
    target_feature_vector,
)


@torch.no_grad()
def greedy_cost(env, net, poly, k, max_steps, device):
    st = env.reset(poly)
    for _ in range(max_steps):
        if not st.frontier:
            break
        active = st.frontier[0]
        cands = env.get_candidate_splits(st, 0, k)
        if not cands:
            st, _, done, _ = env.solve_direct(st, 0)
            if done:
                break
            continue
        tt = torch.tensor([target_feature_vector(active)], dtype=torch.float32, device=device)
        ct = torch.tensor([[candidate_feature_vector(active, a) for a in cands]],
                          dtype=torch.float32, device=device)
        st, _, done, _ = env.step(st, 0, cands[int(net(ct, tt)[0].squeeze(0).argmax())])
        if done:
            break
    while st.frontier:
        st, _, _, _ = env.solve_direct(st, 0)
    return int(st.acc_cost)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prime", type=int, default=3)
    p.add_argument("--variables", nargs="+", default=["x", "y", "z"])
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--max-support", type=int, default=7)
    p.add_argument("--max-degree", type=int, default=4)
    p.add_argument("--candidates-per-step", type=int, default=16)
    p.add_argument("--max-episode-steps", type=int, default=24)
    p.add_argument("--rollouts", type=int, default=25, help="Random rollouts per polynomial.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    P = args.prime
    VARS = tuple(args.variables)
    lib = FactorizableLibrary(prime=P, variables=VARS)
    fac = FiniteFieldFactorizer(FactorizerConfig(), library=lib)
    bm = BaselineCostModel()
    env = DecompEnv(config=DecompEnvConfig(), factorizer=fac, baseline_model=bm, library=lib)
    net = TorchPolicyValueNetwork().to(args.device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    net.eval()

    rng = Random(args.seed)
    ag = defaultdict(lambda: [0, 0])
    rd = defaultdict(lambda: [0, 0])
    n_ck_exact = 0
    n_skipped = 0
    seen: set[str] = set()

    attempts = 0
    while n_ck_exact < args.n_samples and attempts < args.n_samples * 20:
        attempts += 1
        support = rng.randint(2, args.max_support)
        max_deg = rng.randint(1, args.max_degree)
        poly = random_sparse_polynomial(rng, P, VARS, support, max_deg)
        if poly.is_zero or poly.is_constant or poly.support_size < 2:
            continue
        key = poly.to_key()
        if key in seen:
            continue
        seen.add(key)
        # Only keep polys where the exact shortest circuit is tractable.
        if poly.support_size > DEFAULT_EXACT_SUPPORT_LIMIT or poly.total_degree > DEFAULT_EXACT_DEGREE_LIMIT:
            n_skipped += 1
            continue
        ck = exact_circuit_complexity(poly, fac, bm, k_candidates=16, memo={})
        n_ck_exact += 1
        ag[ck][0] += int(greedy_cost(env, net, poly, args.candidates_per_step, args.max_episode_steps, args.device) <= ck)
        ag[ck][1] += 1
        for _ in range(args.rollouts):
            cost, _ = random_rollout_cost(poly, env, rng, k_candidates=args.candidates_per_step, max_steps=args.max_episode_steps)
            rd[ck][0] += int(cost <= ck)
            rd[ck][1] += 1

    fac.close()

    ta = na = tr = nr = 0
    print(f"Unfiltered random polynomials with exact Ck: {n_ck_exact} "
          f"(skipped {n_skipped} too-large for exact search)")
    print(f"{'Ck':>4}{'n':>5}{'agent':>8}{'random':>8}{'lift':>7}")
    for ck in sorted(ag):
        a = ag[ck][0] / ag[ck][1]
        r = rd[ck][0] / rd[ck][1]
        ta += ag[ck][0]; na += ag[ck][1]; tr += rd[ck][0]; nr += rd[ck][1]
        print(f"C{ck:>3}{ag[ck][1]:>5}{a:>8.2f}{r:>8.2f}{a - r:>+7.2f}")
    print(f"OVERALL  agent={ta / max(1,na):.3f}  random={tr / max(1,nr):.3f}  "
          f"lift={ta / max(1,na) - tr / max(1,nr):+.3f}")


if __name__ == "__main__":
    main()
