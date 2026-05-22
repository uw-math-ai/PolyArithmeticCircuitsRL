"""Discrete SAC fine-tuning on the split-based decomposition environment.

Off-policy counterpart to ``run_ppo_finetune.py``. Implements the same top-down
decomposition theory (additive splits ``f = g + h`` plus the whole-polynomial
factor action; pieces auto-factored via the ``FactorizableLibrary``-backed
``FiniteFieldFactorizer``) but trains the policy with discrete Soft Actor-Critic
(``decomp_rl.train_sac``): twin Q-critics, target networks, a replay buffer, and
automatically tuned entropy temperature.

Examples:
    python scripts/run_sac_finetune.py --iterations 50
    python scripts/run_sac_finetune.py \
        --target-file artifacts/ck_curriculum/targets.jsonl \
        --checkpoint-in artifacts/curriculum/best.pt \
        --checkpoint-out artifacts/sac/finetuned.pt --device cuda

The actor shares the ``TorchPolicyValueNetwork`` architecture used by PPO, so
``--checkpoint-in`` can warm-start SAC from a PPO-trained policy, and the saved
actor checkpoint is loadable by the existing greedy/evaluation utilities.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.config import DecompEnvConfig, FactorizerConfig
from decomp_rl.decomp_env import DecompEnv
from decomp_rl.factor_fp import FiniteFieldFactorizer
from decomp_rl.factor_library import FactorizableLibrary
from decomp_rl.model import TorchPolicyValueNetwork
from decomp_rl.polynomial import SparsePolynomial
from decomp_rl.train_sac import SACConfig, SACMetrics, train_sac


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prime", type=int, default=3)
    p.add_argument("--variables", nargs="+", default=["x", "y"])
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--rollouts-per-update", type=int, default=8)
    p.add_argument("--gradient-steps", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-capacity", type=int, default=50_000)
    p.add_argument("--learning-starts", type=int, default=256)
    p.add_argument("--candidates-per-step", type=int, default=16)
    p.add_argument("--max-episode-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha-init", type=float, default=0.2)
    p.add_argument("--no-autotune-alpha", action="store_true", help="Use a fixed alpha instead of tuning it.")
    p.add_argument("--target-entropy-scale", type=float, default=0.7)
    p.add_argument("--library-reward-weight", type=float, default=1.0)
    p.add_argument("--terminal-bonus-weight", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--target-file", type=Path, default=None,
                   help="Optional JSONL; each line {prime, variables, terms, ...}.")
    p.add_argument("--checkpoint-in", type=Path, default=None,
                   help="Optional warm-start actor checkpoint (e.g. a PPO-trained policy).")
    p.add_argument("--checkpoint-out", type=Path, default=Path("artifacts/sac/finetuned.pt"))
    p.add_argument("--metrics-out", type=Path, default=Path("artifacts/sac/finetuned.metrics.jsonl"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-project", default="PolyArithmeticCircuitsRL")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--wandb-run-id", default="")
    p.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    return p.parse_args()


def init_wandb(args: argparse.Namespace, run_id: str):
    if args.wandb_mode == "disabled" or wandb is None:
        return None, ("disabled" if args.wandb_mode == "disabled" else "wandb not installed")
    if not args.wandb_entity:
        return None, "no_entity"
    config_payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
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


def _pad(variables: tuple[str, ...], head: tuple[int, ...]) -> tuple[int, ...]:
    return head + (0,) * (len(variables) - len(head))


def load_targets(args: argparse.Namespace) -> list[SparsePolynomial]:
    if args.target_file is not None and args.target_file.exists():
        targets: list[SparsePolynomial] = []
        with args.target_file.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                targets.append(SparsePolynomial(
                    obj["prime"], tuple(obj["variables"]),
                    tuple((c, tuple(e)) for c, e in obj["terms"]),
                ))
        if not targets:
            raise ValueError(f"--target-file {args.target_file} is empty")
        return targets

    p, vars_t = args.prime, tuple(args.variables)
    if len(vars_t) < 2:
        raise ValueError("Built-in target set requires at least 2 variables")
    return [
        SparsePolynomial(p, vars_t,
            ((1, _pad(vars_t, (1, 1))), (1, _pad(vars_t, (1, 0))), (1, _pad(vars_t, (0, 1))))),
        SparsePolynomial(p, vars_t,
            ((1, _pad(vars_t, (2, 0))), (2 % p, _pad(vars_t, (1, 1))), (1, _pad(vars_t, (0, 2))))),
    ]


def main() -> None:
    args = parse_args()
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.unlink(missing_ok=True)

    library = FactorizableLibrary(prime=args.prime, variables=tuple(args.variables))
    factorizer = FiniteFieldFactorizer(FactorizerConfig(), library=library)
    env = DecompEnv(config=DecompEnvConfig(), factorizer=factorizer,
                    baseline_model=BaselineCostModel(), library=library)

    actor = TorchPolicyValueNetwork().to(args.device)
    if args.checkpoint_in is not None and args.checkpoint_in.exists():
        actor.load_state_dict(torch.load(args.checkpoint_in, map_location=args.device))
        print(f"Warm-started actor from {args.checkpoint_in}", flush=True)
    actor.train()

    run_id = args.wandb_run_id or f"decomp-rl-sac-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    wandb_run, wandb_mode = init_wandb(args, run_id)
    print(f"wandb_run_id={run_id}  wandb_mode={wandb_mode}", flush=True)

    targets = load_targets(args)
    config = SACConfig(
        gamma=args.gamma, tau=args.tau, learning_rate=args.learning_rate,
        candidates_per_step=args.candidates_per_step, max_episode_steps=args.max_episode_steps,
        rollouts_per_update=args.rollouts_per_update, gradient_steps=args.gradient_steps,
        batch_size=args.batch_size, buffer_capacity=args.buffer_capacity,
        learning_starts=args.learning_starts, library_reward_weight=args.library_reward_weight,
        terminal_bonus_weight=args.terminal_bonus_weight, alpha_init=args.alpha_init,
        autotune_alpha=not args.no_autotune_alpha, target_entropy_scale=args.target_entropy_scale,
        seed=args.seed,
    )

    def log_callback(m: SACMetrics) -> None:
        payload = {
            "iteration": m.iteration,
            "mean_episode_reward": m.mean_episode_reward,
            "mean_episode_length": m.mean_episode_length,
            "mean_episode_savings": m.mean_episode_savings,
            "critic_loss": m.critic_loss,
            "actor_loss": m.actor_loss,
            "alpha_loss": m.alpha_loss,
            "alpha": m.alpha,
            "entropy": m.entropy,
            "mean_terminal_bonus": m.mean_terminal_bonus,
            "buffer_size": m.buffer_size,
        }
        with args.metrics_out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        if wandb_run is not None:
            wandb_run.log(payload, step=m.iteration)
        print(
            f"[iter {m.iteration:4d}] reward={m.mean_episode_reward:+.3f}  "
            f"len={m.mean_episode_length:.1f}  savings={m.mean_episode_savings:+.3f}  "
            f"cl={m.critic_loss:.4f}  al={m.actor_loss:+.4f}  alpha={m.alpha:.3f}  "
            f"H={m.entropy:.3f}  buf={m.buffer_size}",
            flush=True,
        )

    train_sac(targets, actor, env, config, iterations=args.iterations, log_callback=log_callback)
    torch.save(actor.state_dict(), args.checkpoint_out)
    print(f"Saved actor checkpoint to {args.checkpoint_out}", flush=True)

    if wandb_run is not None:
        try:
            artifact = wandb.Artifact(f"{run_id}-checkpoints", type="model")
            for path in (args.checkpoint_out, args.metrics_out):
                if path.exists():
                    artifact.add_file(str(path), name=path.name)
            wandb_run.log_artifact(artifact)
        except Exception as exc:  # pragma: no cover - depends on env auth
            print(f"wandb artifact upload failed: {exc}", flush=True)
        wandb_run.finish()
    factorizer.close()


if __name__ == "__main__":
    main()
