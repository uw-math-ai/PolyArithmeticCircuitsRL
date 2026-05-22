#!/usr/bin/env python3
"""Top-down PPO training graded by circuit complexity (Ck) with rich W&B dashboards.

This is ``run_ppo_finetune.py`` plus the metrics a human actually wants when
watching a run: a periodic *greedy* evaluation over the whole target set that
reports, overall and broken out by complexity bucket ``Ck``:

  * ck_match_rate    — fraction of targets whose greedy circuit cost reaches the
                       shortest-circuit complexity Ck (cost <= ck),
  * mean_gap_to_optimal — average (cost - ck), distance from the optimum,
  * mean discovered cost vs mean Ck.

A one-time uniform-random split policy is evaluated up front and logged as a
flat reference line so the learned policy is easy to read against "random".

Per-iteration training curves (reward, savings, policy/value loss, entropy,
approx-KL, terminal-bonus) are logged every iteration as before.

Targets come from ``generate_ck_curriculum.py`` (JSONL with prime, variables,
terms, ck, method). Note: the agent's action space is additive-split-only, so
on targets whose optimum needs whole-polynomial factorization (e.g. (x+y)^2)
the match rate honestly stays below 1 — that gap is the metric's point.

Example:
    python scripts/run_ppo_curriculum.py \
        --target-file artifacts/ck_curriculum/targets.jsonl \
        --iterations 500 --device cuda \
        --wandb-entity zengrf-university-of-washington --wandb-mode online
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from random import Random

import torch

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.config import DecompEnvConfig, FactorizerConfig
from decomp_rl.decomp_env import DecompEnv
from decomp_rl.evaluate import random_rollout_cost
from decomp_rl.factor_fp import FiniteFieldFactorizer
from decomp_rl.factor_library import FactorizableLibrary
from decomp_rl.model import (
    TorchPolicyValueNetwork,
    candidate_feature_vector,
    target_feature_vector,
)
from decomp_rl.polynomial import SparsePolynomial
from decomp_rl.train_ppo import PPOConfig, TrainingMetrics, train_ppo


# ─────────────────────────── targets ────────────────────────────
class Target:
    __slots__ = ("poly", "ck", "method")

    def __init__(self, poly: SparsePolynomial, ck: int, method: str) -> None:
        self.poly = poly
        self.ck = ck          # shortest-circuit op count (success threshold)
        self.method = method  # "exact" or "heuristic"


def load_curriculum(path: Path) -> list[Target]:
    targets: list[Target] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            poly = SparsePolynomial(
                obj["prime"],
                tuple(obj["variables"]),
                tuple((c, tuple(e)) for c, e in obj["terms"]),
            )
            ck = int(obj.get("ck", obj.get("level", 0)))
            targets.append(Target(poly, ck, str(obj.get("method", "?"))))
    if not targets:
        raise ValueError(f"--target-file {path} is empty")
    return targets


# ───────────────────────── greedy eval ──────────────────────────
@torch.no_grad()
def greedy_episode_cost(env: DecompEnv, model, target: SparsePolynomial,
                        candidates_per_step: int, max_steps: int, device) -> int:
    """Deterministic argmax rollout; solve any leftover frontier directly."""
    state = env.reset(target)
    for _ in range(max_steps):
        if not state.frontier:
            break
        active = state.frontier[0]
        candidates = env.get_candidate_splits(state, 0, candidates_per_step)
        if not candidates:
            state, _, done, _ = env.solve_direct(state, 0)
            if done:
                break
            continue
        target_t = torch.tensor([target_feature_vector(active)], dtype=torch.float32, device=device)
        cand_t = torch.tensor(
            [[candidate_feature_vector(active, a) for a in candidates]],
            dtype=torch.float32, device=device,
        )
        logits, _ = model(cand_t, target_t)
        idx = int(logits.squeeze(0).argmax().item())
        state, _, done, _ = env.step(state, 0, candidates[idx])
        if done:
            break
    while state.frontier:
        state, _, _, _ = env.solve_direct(state, 0)
    return int(state.acc_cost)


def evaluate_curriculum(env: DecompEnv, model, targets: list[Target],
                        candidates_per_step: int, max_steps: int, device) -> dict[str, float]:
    """Greedy eval against the shortest-circuit target Ck.

    Success on a target = the greedy circuit cost reaches its complexity Ck
    (``cost <= ck``). ``gap = cost - ck`` measures distance from the optimal.
    Reported overall and per Ck bucket.
    """
    was_training = model.training
    model.eval()

    overall_match = 0
    overall_gap = 0.0
    overall_cost = 0.0
    overall_ck = 0.0
    per_ck_match: dict[int, int] = defaultdict(int)
    per_ck_count: dict[int, int] = defaultdict(int)
    per_ck_gap: dict[int, float] = defaultdict(float)

    for tgt in targets:
        cost = float(greedy_episode_cost(env, model, tgt.poly, candidates_per_step, max_steps, device))
        gap = cost - tgt.ck
        matched = cost <= tgt.ck

        overall_match += int(matched)
        overall_gap += gap
        overall_cost += cost
        overall_ck += tgt.ck
        per_ck_match[tgt.ck] += int(matched)
        per_ck_count[tgt.ck] += 1
        per_ck_gap[tgt.ck] += gap

    if was_training:
        model.train()

    n = len(targets)
    metrics: dict[str, float] = {
        "eval/ck_match_rate": overall_match / n,
        "eval/mean_gap_to_optimal": overall_gap / n,
        "eval/mean_discovered_cost": overall_cost / n,
        "eval/mean_ck": overall_ck / n,
    }
    for ck in sorted(per_ck_count):
        c = per_ck_count[ck]
        metrics[f"eval/C{ck}/match_rate"] = per_ck_match[ck] / c
        metrics[f"eval/C{ck}/mean_gap"] = per_ck_gap[ck] / c
        metrics[f"eval/C{ck}/count"] = float(c)
    return metrics


def random_ck_match_rate(targets: list[Target], env: DecompEnv,
                         k_candidates: int, max_steps: int, seed: int, rollouts: int = 4) -> float:
    """Reference: uniform-random split policy's Ck-match rate (flat line)."""
    rng = Random(seed)
    matched = 0
    total = 0
    for tgt in targets:
        for _ in range(rollouts):
            cost, _ = random_rollout_cost(tgt.poly, env, rng, k_candidates=k_candidates, max_steps=max_steps)
            matched += int(cost <= tgt.ck)
            total += 1
    return matched / total if total else 0.0


# ───────────────────────────── wandb ────────────────────────────
def init_wandb(args, run_id: str, config_payload: dict):
    if args.wandb_mode == "disabled" or wandb is None:
        return None, ("disabled" if args.wandb_mode == "disabled" else "wandb not installed")
    if not args.wandb_entity:
        return None, "no_entity"
    init_kwargs = dict(
        entity=args.wandb_entity, project=args.wandb_project,
        id=run_id, name=run_id, group=args.wandb_group or None,
        resume="allow", config=config_payload, dir=str(args.metrics_out.parent),
    )
    modes = [args.wandb_mode] if args.wandb_mode != "auto" else ["online", "offline"]
    last_error = None
    for mode in modes:
        try:
            return wandb.init(mode=mode, **init_kwargs), mode
        except Exception as exc:  # pragma: no cover - depends on env auth
            last_error = exc
    (args.metrics_out.parent / "wandb_warning.txt").write_text(
        f"Failed to init wandb in {modes}: {last_error}\n", encoding="utf-8")
    return None, "disabled"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-file", type=Path, required=True)
    p.add_argument("--prime", type=int, default=3)
    p.add_argument("--variables", nargs="+", default=["x", "y", "z"])
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--rollouts-per-update", type=int, default=16)
    p.add_argument("--candidates-per-step", type=int, default=16)
    p.add_argument("--max-episode-steps", type=int, default=24)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--library-reward-weight", type=float, default=1.0)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10, help="Run greedy curriculum eval every N iters.")
    p.add_argument("--use-mcts", action="store_true",
                   help="Enable AlphaZero-style MCTS guidance (AndOrSearch per step + distillation).")
    p.add_argument("--mcts-simulations", type=int, default=48)
    p.add_argument("--mcts-max-depth", type=int, default=6)
    p.add_argument("--mcts-temperature", type=float, default=1.0)
    p.add_argument("--mcts-distill-coef", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/curriculum/checkpoints"))
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--checkpoint-out", type=Path, default=Path("artifacts/curriculum/checkpoints/final.pt"))
    p.add_argument("--metrics-out", type=Path, default=Path("artifacts/curriculum/metrics.jsonl"))
    p.add_argument("--checkpoint-in", type=Path, default=None)
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-project", default="PolyArithmeticCircuitsRL")
    p.add_argument("--wandb-group", default="top-down")
    p.add_argument("--wandb-run-id", default="")
    p.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.unlink(missing_ok=True)

    variables = tuple(args.variables)
    targets = load_curriculum(args.target_file)
    target_polys = [t.poly for t in targets]
    print(f"Loaded {len(targets)} targets from {args.target_file}", flush=True)

    ck_hist = defaultdict(int)
    for t in targets:
        ck_hist[t.ck] += 1
    print(f"Ck buckets: {dict(sorted(ck_hist.items()))}", flush=True)

    library = FactorizableLibrary(prime=args.prime, variables=variables)
    factorizer = FiniteFieldFactorizer(FactorizerConfig(), library=library)
    baseline_model = BaselineCostModel()
    env = DecompEnv(config=DecompEnvConfig(), factorizer=factorizer,
                    baseline_model=baseline_model, library=library)

    network = TorchPolicyValueNetwork().to(args.device)
    if args.checkpoint_in is not None and args.checkpoint_in.exists():
        network.load_state_dict(torch.load(args.checkpoint_in, map_location=args.device))
    network.train()

    run_id = args.wandb_run_id or f"top-down-ppo-curriculum-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    config_payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config_payload["n_targets"] = len(targets)
    wandb_run, wandb_mode = init_wandb(args, run_id, config_payload)
    print(f"wandb_run_id={run_id}  wandb_mode={wandb_mode}", flush=True)

    # Reference: uniform-random split policy Ck-match rate (flat line on graphs).
    random_match = random_ck_match_rate(
        targets, env, args.candidates_per_step, args.max_episode_steps, args.seed,
    )
    print(f"random-policy reference ck_match_rate={random_match:.3f}", flush=True)

    # Baseline (untrained net) greedy eval at iteration 0.
    init_eval = evaluate_curriculum(env, network, targets,
                                    args.candidates_per_step, args.max_episode_steps, args.device)
    print(f"init greedy ck_match_rate={init_eval['eval/ck_match_rate']:.3f}", flush=True)

    def log_callback(m: TrainingMetrics) -> None:
        payload = {
            "iteration": m.iteration,
            "train/mean_episode_reward": m.mean_episode_reward,
            "train/mean_episode_length": m.mean_episode_length,
            "train/mean_episode_savings": m.mean_episode_savings,
            "train/policy_loss": m.policy_loss,
            "train/value_loss": m.value_loss,
            "train/entropy": m.entropy,
            "train/approx_kl": m.approx_kl,
            "train/mean_terminal_bonus": m.mean_terminal_bonus,
            "baseline/random_ck_match_rate": random_match,
        }
        if args.eval_every > 0 and (m.iteration % args.eval_every == 0 or m.iteration + 1 == args.iterations):
            payload.update(
                evaluate_curriculum(env, network, targets,
                                    args.candidates_per_step, args.max_episode_steps, args.device)
            )

        with args.metrics_out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        if wandb_run is not None:
            wandb_run.log(payload, step=m.iteration)

        match = payload.get("eval/ck_match_rate")
        match_str = f"  ck_match={match:.2f}  gap={payload['eval/mean_gap_to_optimal']:+.2f}" if match is not None else ""
        print(
            f"[iter {m.iteration:4d}] reward={m.mean_episode_reward:+.3f}  "
            f"savings={m.mean_episode_savings:+.3f}  bonus={m.mean_terminal_bonus:+.3f}  "
            f"pl={m.policy_loss:+.4f}  vl={m.value_loss:.4f}  H={m.entropy:.3f}{match_str}",
            flush=True,
        )
        if args.checkpoint_every > 0 and (m.iteration + 1) % args.checkpoint_every == 0:
            ckpt = args.checkpoint_dir / f"iter_{m.iteration + 1:05d}.pt"
            torch.save(network.state_dict(), ckpt)

    config = PPOConfig(
        rollouts_per_update=args.rollouts_per_update,
        candidates_per_step=args.candidates_per_step,
        max_episode_steps=args.max_episode_steps,
        learning_rate=args.learning_rate,
        library_reward_weight=args.library_reward_weight,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        use_mcts=args.use_mcts,
        mcts_simulations=args.mcts_simulations,
        mcts_max_depth=args.mcts_max_depth,
        mcts_temperature=args.mcts_temperature,
        mcts_distill_coef=args.mcts_distill_coef,
    )
    train_ppo(target_polys, network, env, config, iterations=args.iterations, log_callback=log_callback)

    torch.save(network.state_dict(), args.checkpoint_out)
    print(f"Saved checkpoint to {args.checkpoint_out}", flush=True)

    if wandb_run is not None:
        try:
            artifact = wandb.Artifact(f"{run_id}-checkpoints", type="model")
            for path in (args.checkpoint_out, args.metrics_out, args.target_file):
                if path.exists():
                    artifact.add_file(str(path), name=path.name)
            wandb_run.log_artifact(artifact)
        except Exception as exc:  # pragma: no cover - depends on env auth
            print(f"wandb artifact upload failed: {exc}", flush=True)
        wandb_run.finish()
    factorizer.close()


if __name__ == "__main__":
    main()
