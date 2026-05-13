"""PPO fine-tuning on the split-based decomposition environment.

Implements the split-point-built circuits theory: the policy chooses additive
splits ``f = g + h`` for each frontier polynomial; the environment auto-factors
both pieces using the ``FactorizableLibrary``-backed ``FiniteFieldFactorizer``.

Examples:
    python scripts/run_ppo_finetune.py --iterations 10
    python scripts/run_ppo_finetune.py \
        --target-file targets.jsonl \
        --checkpoint-in artifacts/search_distill/best.pt \
        --checkpoint-out artifacts/ppo/finetuned.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.config import (
    DecompEnvConfig,
    FactorizerConfig,
)
from decomp_rl.decomp_env import DecompEnv
from decomp_rl.factor_fp import FiniteFieldFactorizer
from decomp_rl.factor_library import FactorizableLibrary
from decomp_rl.model import TorchPolicyValueNetwork
from decomp_rl.polynomial import SparsePolynomial
from decomp_rl.train_ppo import PPOConfig, TrainingMetrics, train_ppo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prime", type=int, default=3)
    p.add_argument("--variables", nargs="+", default=["x", "y"])
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--rollouts-per-update", type=int, default=8)
    p.add_argument("--candidates-per-step", type=int, default=16)
    p.add_argument("--max-episode-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--library-reward-weight", type=float, default=1.0)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use-mcts", action="store_true",
                   help="Enable AlphaZero-style MCTS guidance (AndOrSearch per step + distillation)")
    p.add_argument("--mcts-simulations", type=int, default=32)
    p.add_argument("--mcts-max-depth", type=int, default=4)
    p.add_argument("--mcts-temperature", type=float, default=1.0)
    p.add_argument("--mcts-distill-coef", type=float, default=1.0)
    p.add_argument("--target-file", type=Path, default=None,
                   help="Optional JSONL file; each line {prime, variables, terms}")
    p.add_argument("--checkpoint-in", type=Path, default=None,
                   help="Optional warm-start checkpoint (state_dict for TorchPolicyValueNetwork)")
    p.add_argument("--checkpoint-out", type=Path, default=Path("artifacts/ppo/finetuned.pt"))
    p.add_argument("--metrics-out", type=Path, default=Path("artifacts/ppo/finetuned.metrics.jsonl"))
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_targets(args: argparse.Namespace) -> list[SparsePolynomial]:
    if args.target_file is not None and args.target_file.exists():
        targets: list[SparsePolynomial] = []
        with args.target_file.open() as fh:
            for line in fh:
                obj = json.loads(line)
                targets.append(
                    SparsePolynomial(
                        obj["prime"],
                        tuple(obj["variables"]),
                        tuple((c, tuple(e)) for c, e in obj["terms"]),
                    )
                )
        if not targets:
            raise ValueError(f"--target-file {args.target_file} is empty")
        return targets

    p = args.prime
    vars_t = tuple(args.variables)
    if len(vars_t) < 2:
        raise ValueError("Built-in target set requires at least 2 variables")
    return [
        # xy + x + y
        SparsePolynomial(p, vars_t,
            ((1, _pad(vars_t, (1, 1))), (1, _pad(vars_t, (1, 0))), (1, _pad(vars_t, (0, 1))))),
        # x^2 + 2xy + y^2  (= (x+y)^2 over Z, exposed as a square in F_p when p != 2)
        SparsePolynomial(p, vars_t,
            ((1, _pad(vars_t, (2, 0))), (2 % p, _pad(vars_t, (1, 1))), (1, _pad(vars_t, (0, 2))))),
    ]


def _pad(variables: tuple[str, ...], head: tuple[int, ...]) -> tuple[int, ...]:
    if len(head) > len(variables):
        raise ValueError("monomial head longer than variable count")
    return head + (0,) * (len(variables) - len(head))


def main() -> None:
    args = parse_args()
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.unlink(missing_ok=True)

    library = FactorizableLibrary(prime=args.prime, variables=tuple(args.variables))
    factorizer = FiniteFieldFactorizer(FactorizerConfig(), library=library)
    env = DecompEnv(
        config=DecompEnvConfig(),
        factorizer=factorizer,
        baseline_model=BaselineCostModel(),
        library=library,
    )

    network = TorchPolicyValueNetwork().to(args.device)
    if args.checkpoint_in is not None:
        state = torch.load(args.checkpoint_in, map_location=args.device)
        network.load_state_dict(state)
    network.train()

    targets = load_targets(args)
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

    def log_callback(m: TrainingMetrics) -> None:
        with args.metrics_out.open("a") as fh:
            fh.write(
                json.dumps({
                    "iteration": m.iteration,
                    "mean_episode_reward": m.mean_episode_reward,
                    "mean_episode_length": m.mean_episode_length,
                    "mean_episode_savings": m.mean_episode_savings,
                    "policy_loss": m.policy_loss,
                    "value_loss": m.value_loss,
                    "entropy": m.entropy,
                    "approx_kl": m.approx_kl,
                    "distill_loss": m.distill_loss,
                }) + "\n"
            )
        distill_str = f"  dl={m.distill_loss:.4f}" if args.use_mcts else ""
        print(
            f"[iter {m.iteration:4d}] reward={m.mean_episode_reward:+.3f}  "
            f"len={m.mean_episode_length:.1f}  savings={m.mean_episode_savings:+.3f}  "
            f"pl={m.policy_loss:+.4f}  vl={m.value_loss:.4f}  H={m.entropy:.3f}  "
            f"kl={m.approx_kl:+.4f}{distill_str}"
        )

    train_ppo(targets, network, env, config, iterations=args.iterations, log_callback=log_callback)
    torch.save(network.state_dict(), args.checkpoint_out)
    print(f"Saved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
