"""Discrete Soft Actor-Critic (SAC) on the split-based decomposition environment.

This is the off-policy counterpart to ``train_ppo.py``. It implements the
top-down decomposition theory identically — at each step the policy chooses an
action for ``frontier[0]`` (an additive split ``f = g + h`` or the
whole-polynomial factor action), the environment auto-factors the pieces via the
``FactorizableLibrary``-backed ``FiniteFieldFactorizer`` and pushes unresolved
factor children back onto the frontier — but learns it with SAC instead of PPO.

Action space note: each step exposes a *variable-sized* candidate set, so this
is **discrete SAC over a variable action set** (Christodoulou, 2019, "Soft
Actor-Critic for Discrete Action Settings"). The policy is a softmax over the
per-step candidates; two Q-networks score every candidate; the soft state value
is the policy-weighted ``min(Q1, Q2) - alpha * log pi`` expectation.

Components, matching the discrete-SAC recipe:
  * Actor: a ``TorchPolicyValueNetwork`` (only its candidate logits are used;
    reusing it lets SAC warm-start from a PPO checkpoint).
  * Twin critics Q1, Q2 (``TorchQNetwork``) + Polyak-averaged target critics.
  * Replay buffer of (s, a, r, s', done) transitions, where a "state" is an
    active polynomial together with its candidate set.
  * Automatic temperature (alpha) tuning toward a target entropy that scales
    with the per-state number of candidates.

Reward shaping mirrors PPO: the env cost-savings reward plus an optional
``FactorizableLibrary`` match bonus, plus an optional terminal bonus when the
finished circuit beats the five-baseline minimum.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None

from .baselines import BaselineBundle
from .decomp_env import DecompEnv
from .model import (
    TorchQNetwork,
    candidate_feature_vector,
    target_feature_vector,
)
from .polynomial import SparsePolynomial
from .split_proposals import SplitAction


@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005                  # Polyak coefficient for target critics
    learning_rate: float = 3e-4
    candidates_per_step: int = 16
    max_episode_steps: int = 32
    rollouts_per_update: int = 8        # episodes collected before each update phase
    gradient_steps: int = 16            # SAC updates per iteration
    batch_size: int = 64
    buffer_capacity: int = 50_000
    learning_starts: int = 256          # min transitions before updates begin
    library_reward_weight: float = 1.0
    terminal_bonus_weight: float = 10.0
    # Temperature (entropy weight). When ``autotune_alpha`` is True, ``alpha`` is
    # learned toward ``target_entropy_scale * log(num_candidates)`` per state.
    alpha_init: float = 0.2
    autotune_alpha: bool = True
    target_entropy_scale: float = 0.7
    grad_clip_norm: float = 1.0
    critic_hidden_dim: int = 128
    critic_layers: int = 3
    seed: int | None = None


@dataclass
class SACTransition:
    target: SparsePolynomial
    candidates: tuple[SplitAction, ...]
    chosen_index: int
    reward: float
    done: bool
    next_target: SparsePolynomial | None
    next_candidates: tuple[SplitAction, ...]
    library_reward: float = 0.0
    terminal_bonus: float = 0.0


@dataclass
class SACMetrics:
    iteration: int
    mean_episode_reward: float
    mean_episode_length: float
    mean_episode_savings: float
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    entropy: float
    mean_terminal_bonus: float = 0.0
    buffer_size: int = 0


class ReplayBuffer:
    """Uniform replay buffer of SAC transitions."""

    def __init__(self, capacity: int, rng: random.Random) -> None:
        self._buffer: deque[SACTransition] = deque(maxlen=capacity)
        self._rng = rng

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, transition: SACTransition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[SACTransition]:
        n = min(batch_size, len(self._buffer))
        return self._rng.sample(list(self._buffer), n)


# ──────────────────────────── feature helpers ───────────────────────────────
def _candidate_tensor(target, candidates, device):
    return torch.tensor(
        [[candidate_feature_vector(target, a) for a in candidates]],
        dtype=torch.float32, device=device,
    )  # (1, A, F)


def _target_tensor(target, device):
    return torch.tensor([target_feature_vector(target)], dtype=torch.float32, device=device)


# ──────────────────────────── rollout collection ────────────────────────────
def collect_episode(
    env: DecompEnv,
    actor,
    target: SparsePolynomial,
    config: SACConfig,
    rng=None,
    baselines: BaselineBundle | None = None,
) -> tuple[list[SACTransition], dict]:
    """Run one episode sampling from the actor policy; build SAC transitions.

    Returns the (s, a, r, s', done) transitions plus an episode-stats dict
    (reward, length, savings, terminal_bonus). A "state" is an active
    polynomial and its candidate set; consecutive policy decisions are linked
    so transition ``i``'s ``next`` is decision ``i+1`` (terminal at the end).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for SAC training")

    device = next(actor.parameters()).device
    state = env.reset(target)

    # Per-decision records: (active, candidates, idx, shaped_reward, env_done, lib_r).
    records: list[list] = []
    for _ in range(config.max_episode_steps):
        if not state.frontier:
            break
        active = state.frontier[0]
        candidates = env.get_candidate_splits(state, 0, config.candidates_per_step)
        if not candidates:
            state, _, done, _ = env.solve_direct(state, 0)
            if done:
                break
            continue

        with torch.no_grad():
            logits, _ = actor(_candidate_tensor(active, candidates, device),
                              _target_tensor(active, device))
            probs = F.softmax(logits.squeeze(0), dim=-1)
            if rng is not None:
                idx = int(torch.multinomial(probs, 1, generator=rng).item())
            else:
                idx = int(torch.multinomial(probs, 1).item())

        chosen = candidates[idx]
        state, reward, done, info = env.step(state, 0, chosen)
        lib_r = float(getattr(info, "library_reward", 0.0))
        shaped = float(reward) + config.library_reward_weight * lib_r
        records.append([active, tuple(candidates), idx, shaped, bool(done), lib_r])
        if done:
            break

    terminal_bonus = 0.0
    if (baselines is not None and config.terminal_bonus_weight > 0.0
            and records and not state.frontier):
        min_baseline = baselines.min_cost(target)
        if state.acc_cost < min_baseline:
            terminal_bonus = config.terminal_bonus_weight * float(min_baseline - state.acc_cost)
            records[-1][3] += terminal_bonus

    transitions: list[SACTransition] = []
    last = len(records) - 1
    for i, rec in enumerate(records):
        active, candidates, idx, shaped, env_done, lib_r = rec
        is_last = i == last
        if env_done or is_last:
            next_target, next_candidates, done_flag = None, (), True
        else:
            nxt = records[i + 1]
            next_target, next_candidates, done_flag = nxt[0], nxt[1], False
        transitions.append(SACTransition(
            target=active, candidates=candidates, chosen_index=idx,
            reward=shaped, done=done_flag,
            next_target=next_target, next_candidates=next_candidates,
            library_reward=lib_r,
            terminal_bonus=terminal_bonus if is_last else 0.0,
        ))

    stats = {
        "reward": sum(r[3] for r in records),
        "length": len(records),
        "savings": sum(r[3] - config.library_reward_weight * r[5] for r in records) - terminal_bonus,
        "terminal_bonus": terminal_bonus,
    }
    return transitions, stats


# ──────────────────────────── SAC update ────────────────────────────────────
def _soft_update(target_net, source_net, tau: float) -> None:
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * sp.data)


def sac_update(
    batch: list[SACTransition],
    actor,
    q1, q2, q1_target, q2_target,
    log_alpha,
    actor_opt, critic_opt, alpha_opt,
    config: SACConfig,
    device,
) -> dict[str, float]:
    """One discrete-SAC gradient step over a sampled batch."""
    if not batch:
        return {"critic_loss": 0.0, "actor_loss": 0.0, "alpha_loss": 0.0,
                "alpha": float(log_alpha.exp().item()), "entropy": 0.0}

    alpha = log_alpha.exp().detach()

    critic_loss = torch.zeros((), device=device)
    actor_loss = torch.zeros((), device=device)
    alpha_loss = torch.zeros((), device=device)
    entropy_sum = 0.0

    for tr in batch:
        # ---- TD target (no grad) ----
        with torch.no_grad():
            if tr.done or not tr.next_candidates:
                v_next = torch.zeros((), device=device)
            else:
                n_cand = _candidate_tensor(tr.next_target, tr.next_candidates, device)
                n_tgt = _target_tensor(tr.next_target, device)
                n_logits, _ = actor(n_cand, n_tgt)
                n_logp = F.log_softmax(n_logits.squeeze(0), dim=-1)
                n_p = n_logp.exp()
                nq = torch.min(q1_target(n_cand).squeeze(0), q2_target(n_cand).squeeze(0))
                v_next = (n_p * (nq - alpha * n_logp)).sum()
            y = tr.reward + config.gamma * v_next  # v_next already 0 when done

        # ---- critic loss on the chosen action ----
        cand = _candidate_tensor(tr.target, tr.candidates, device)
        q1_all = q1(cand).squeeze(0)
        q2_all = q2(cand).squeeze(0)
        q1_sa = q1_all[tr.chosen_index]
        q2_sa = q2_all[tr.chosen_index]
        critic_loss = critic_loss + F.mse_loss(q1_sa, y) + F.mse_loss(q2_sa, y)

        # ---- actor + temperature loss ----
        logits, _ = actor(cand, _target_tensor(tr.target, device))
        logp = F.log_softmax(logits.squeeze(0), dim=-1)
        p = logp.exp()
        with torch.no_grad():
            min_q = torch.min(q1_all, q2_all)
        actor_loss = actor_loss + (p * (alpha * logp - min_q)).sum()

        entropy = -(p * logp).sum()
        entropy_sum += float(entropy.item())
        if config.autotune_alpha:
            target_entropy = config.target_entropy_scale * torch.log(
                torch.tensor(float(max(1, len(tr.candidates))), device=device)
            )
            alpha_loss = alpha_loss + (log_alpha * (entropy.detach() - target_entropy))

    n = len(batch)
    critic_loss = critic_loss / n
    actor_loss = actor_loss / n
    alpha_loss = alpha_loss / n

    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), config.grad_clip_norm)
    critic_opt.step()

    actor_opt.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip_norm)
    actor_opt.step()

    if config.autotune_alpha:
        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()

    _soft_update(q1_target, q1, config.tau)
    _soft_update(q2_target, q2, config.tau)

    return {
        "critic_loss": float(critic_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "alpha_loss": float(alpha_loss.item()) if config.autotune_alpha else 0.0,
        "alpha": float(log_alpha.exp().item()),
        "entropy": entropy_sum / n,
    }


# ──────────────────────────── training loop ─────────────────────────────────
def train_sac(
    targets: Sequence[SparsePolynomial],
    actor,
    env: DecompEnv,
    config: SACConfig,
    iterations: int = 100,
    critics: tuple | None = None,
    log_callback: Callable[[SACMetrics], None] | None = None,
) -> list[SACMetrics]:
    """Run discrete SAC over a cycling list of target polynomials.

    ``actor`` is a ``TorchPolicyValueNetwork`` (its candidate logits are the
    policy). Twin critics are created internally unless supplied via ``critics``
    (used by tests). Returns the per-iteration metrics list.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for SAC training")
    if not targets:
        raise ValueError("targets must be non-empty")

    device = next(actor.parameters()).device
    py_rng = random.Random(config.seed)
    sample_rng = None
    if config.seed is not None:
        torch.manual_seed(config.seed)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(config.seed)

    if critics is not None:
        q1, q2, q1_target, q2_target = critics
    else:
        q1 = TorchQNetwork(hidden_dim=config.critic_hidden_dim, layers=config.critic_layers).to(device)
        q2 = TorchQNetwork(hidden_dim=config.critic_hidden_dim, layers=config.critic_layers).to(device)
        q1_target = TorchQNetwork(hidden_dim=config.critic_hidden_dim, layers=config.critic_layers).to(device)
        q2_target = TorchQNetwork(hidden_dim=config.critic_hidden_dim, layers=config.critic_layers).to(device)
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=config.learning_rate)
    import math
    log_alpha = torch.tensor(float(math.log(config.alpha_init)), device=device, requires_grad=config.autotune_alpha)
    alpha_opt = torch.optim.Adam([log_alpha], lr=config.learning_rate) if config.autotune_alpha else None

    buffer = ReplayBuffer(config.buffer_capacity, py_rng)
    baselines = BaselineBundle(baseline_model=env.baseline_model)

    metrics_log: list[SACMetrics] = []
    target_cycle = list(targets)
    cursor = 0

    for it in range(iterations):
        ep_rewards, ep_lengths, ep_savings, ep_bonuses = [], [], [], []
        for _ in range(config.rollouts_per_update):
            target = target_cycle[cursor % len(target_cycle)]
            cursor += 1
            transitions, stats = collect_episode(env, actor, target, config, rng=sample_rng, baselines=baselines)
            for tr in transitions:
                buffer.add(tr)
            if transitions:
                ep_rewards.append(stats["reward"])
                ep_lengths.append(stats["length"])
                ep_savings.append(stats["savings"])
                ep_bonuses.append(stats["terminal_bonus"])

        update_stats = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha_loss": 0.0,
                        "alpha": float(log_alpha.exp().item()), "entropy": 0.0}
        if len(buffer) >= max(config.batch_size, config.learning_starts):
            accum = {k: 0.0 for k in update_stats}
            for _ in range(config.gradient_steps):
                stats = sac_update(
                    buffer.sample(config.batch_size), actor,
                    q1, q2, q1_target, q2_target, log_alpha,
                    actor_opt, critic_opt, alpha_opt, config, device,
                )
                for k in accum:
                    accum[k] += stats[k]
            update_stats = {k: v / config.gradient_steps for k, v in accum.items()}

        metrics = SACMetrics(
            iteration=it,
            mean_episode_reward=_mean(ep_rewards),
            mean_episode_length=_mean(ep_lengths),
            mean_episode_savings=_mean(ep_savings),
            critic_loss=update_stats["critic_loss"],
            actor_loss=update_stats["actor_loss"],
            alpha_loss=update_stats["alpha_loss"],
            alpha=update_stats["alpha"],
            entropy=update_stats["entropy"],
            mean_terminal_bonus=_mean(ep_bonuses),
            buffer_size=len(buffer),
        )
        metrics_log.append(metrics)
        if log_callback is not None:
            log_callback(metrics)
    return metrics_log


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
