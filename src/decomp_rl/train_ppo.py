"""PPO fine-tuning on the split-based decomposition environment.

Implements the split-point-built circuits theory: at each step the policy
chooses an additive split ``f = g + h`` for ``frontier[0]``; the environment
auto-factors ``g`` and ``h`` via ``FiniteFieldFactorizer`` (backed by the
latest ``FactorizableLibrary``) and pushes their unresolved factor pieces back
onto the frontier. PPO learns a policy/value model over candidate splits using
the cost-savings reward emitted by ``DecompEnv.step`` plus an optional
library-match shaping term.

Action space note: each step exposes a *variable-sized* candidate set, so the
policy is a softmax over the per-step candidates rather than a fixed action
index. Old log-probs are cached on the rollout transition; the update epoch
re-encodes each step's candidates and recomputes new log-probs to form the
clipped surrogate.

Optional AlphaZero-style MCTS guidance: when ``PPOConfig.use_mcts`` is True,
each step runs ``AndOrSearch`` from ``frontier[0]`` using the current network
as prior/value model. The resulting visit-count distribution serves as
(i) the behavior policy for action selection and (ii) a distillation target,
added to the PPO loss as a cross-entropy term weighted by
``mcts_distill_coef``. The clipped surrogate, value MSE, and entropy bonus all
still apply; MCTS just produces a stronger behavior policy and a stronger
imitation target than direct softmax sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None

from .baselines import BaselineBundle
from .decomp_env import DecompEnv
from .model import (
    candidate_feature_vector,
    target_feature_vector,
)
from .polynomial import SparsePolynomial
from .split_proposals import SplitAction


@dataclass(frozen=True)
class PPOConfig:
    rollouts_per_update: int = 8
    candidates_per_step: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 32
    max_episode_steps: int = 32
    library_reward_weight: float = 1.0
    grad_clip_norm: float = 0.5
    # Terminal bonus shaping: if the completed episode's accumulated circuit
    # cost beats the min of all five baselines on the original target, add
    # ``weight * (min_baseline - acc_cost)`` to the final transition's reward.
    # Set to 0.0 to disable.
    terminal_bonus_weight: float = 10.0
    seed: int | None = None
    # MCTS guidance (AlphaZero-style). When use_mcts is True, action selection
    # is driven by AndOrSearch visit counts and the visit distribution is added
    # to the PPO loss as a distillation target.
    use_mcts: bool = False
    mcts_simulations: int = 32
    mcts_max_depth: int = 4
    mcts_temperature: float = 1.0
    mcts_distill_coef: float = 1.0


@dataclass
class Transition:
    target: SparsePolynomial
    candidates: tuple[SplitAction, ...]
    chosen_index: int
    log_prob: float
    value: float
    reward: float
    done: bool
    library_reward: float = 0.0
    terminal_bonus: float = 0.0
    mcts_policy: tuple[float, ...] | None = None


@dataclass
class TrainingMetrics:
    iteration: int
    mean_episode_reward: float
    mean_episode_length: float
    mean_episode_savings: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    distill_loss: float = 0.0
    mean_terminal_bonus: float = 0.0


def collect_episode(
    env: DecompEnv,
    model,
    target: SparsePolynomial,
    config: PPOConfig,
    rng=None,
    baselines: BaselineBundle | None = None,
) -> list[Transition]:
    """Run one episode on ``target``, sampling splits from the policy.

    Frontier handling: always expand ``frontier[0]`` (FIFO). When no candidate
    splits exist for an active polynomial we fall back to ``solve_direct`` —
    that step yields no policy decision and is not recorded as a transition.

    If ``config.use_mcts`` is True, action selection runs ``AndOrSearch`` at
    each step and samples from the resulting visit distribution; the visit
    distribution is stored on the transition for distillation in the update.

    If ``baselines`` is provided and ``config.terminal_bonus_weight > 0``, the
    final transition gets a proportional bonus reward when the completed
    episode beats the min cost across all five baselines on ``target``.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for PPO training")

    transitions: list[Transition] = []
    state = env.reset(target)
    device = next(model.parameters()).device
    search = _build_search(env, model, config) if config.use_mcts else None

    for _ in range(config.max_episode_steps):
        if not state.frontier:
            break
        active = state.frontier[0]

        if search is not None:
            step = _select_step_mcts(active, model, search, config, device, rng)
        else:
            step = _select_step_direct(active, env, state, model, config, device, rng)

        if step is None:
            state, _, done, _ = env.solve_direct(state, 0)
            if done:
                break
            continue

        candidates, idx, chosen_log_prob, value_scalar, mcts_policy = step
        chosen = candidates[idx]
        state, reward, done, info = env.step(state, 0, chosen)
        library_reward = float(getattr(info, "library_reward", 0.0))
        shaped_reward = float(reward) + config.library_reward_weight * library_reward

        transitions.append(
            Transition(
                target=active,
                candidates=tuple(candidates),
                chosen_index=idx,
                log_prob=chosen_log_prob,
                value=value_scalar,
                reward=shaped_reward,
                done=bool(done),
                library_reward=library_reward,
                mcts_policy=mcts_policy,
            )
        )
        if done:
            break

    # Terminal bonus: only when the circuit is fully constructed (frontier empty)
    # and the discovered cost beats the tightest baseline upper bound.
    if (
        baselines is not None
        and config.terminal_bonus_weight > 0.0
        and transitions
        and not state.frontier
    ):
        min_baseline = baselines.min_cost(target)
        if state.acc_cost < min_baseline:
            bonus = config.terminal_bonus_weight * float(
                min_baseline - state.acc_cost
            )
            last = transitions[-1]
            last.terminal_bonus = bonus
            last.reward += bonus

    return transitions


def _select_step_direct(
    active: SparsePolynomial,
    env: DecompEnv,
    state,
    model,
    config: PPOConfig,
    device,
    rng,
):
    """Direct softmax sampling. Returns None when no candidates exist."""
    candidates = env.get_candidate_splits(state, 0, config.candidates_per_step)
    if not candidates:
        return None
    target_t = torch.tensor(
        [target_feature_vector(active)], dtype=torch.float32, device=device
    )
    cand_t = torch.tensor(
        [[candidate_feature_vector(active, action) for action in candidates]],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        logits, value = model(cand_t, target_t)
        log_probs = F.log_softmax(logits.squeeze(0), dim=-1)
        probs = log_probs.exp()
        if rng is not None:
            idx = int(torch.multinomial(probs, num_samples=1, generator=rng).item())
        else:
            idx = int(torch.multinomial(probs, num_samples=1).item())
        chosen_log_prob = float(log_probs[idx].item())
        value_scalar = float(value.squeeze(0).item())
    return list(candidates), idx, chosen_log_prob, value_scalar, None


def _select_step_mcts(
    active: SparsePolynomial,
    model,
    search,
    config: PPOConfig,
    device,
    rng,
):
    """AlphaZero-style step: MCTS visit distribution drives sampling.

    The model is also forward-passed on the MCTS-returned candidate set so the
    PPO clipped surrogate has a well-defined ``old_logp`` and ``value`` over
    exactly the same candidate ordering used in the update.
    """
    result = search.search(active)
    candidates = list(result.root_candidates)
    if not candidates:
        return None
    visit_dist = list(result.root_policy)
    total = sum(visit_dist)
    if total <= 0.0:
        # No simulations reached children — fall back to uniform over MCTS candidates.
        visit_dist = [1.0 / len(candidates)] * len(candidates)
    else:
        visit_dist = [p / total for p in visit_dist]

    if config.mcts_temperature != 1.0 and config.mcts_temperature > 0.0:
        inv_t = 1.0 / config.mcts_temperature
        tempered = [p ** inv_t for p in visit_dist]
        z = sum(tempered)
        sample_dist = [p / z for p in tempered] if z > 0 else visit_dist
    else:
        sample_dist = visit_dist

    target_t = torch.tensor(
        [target_feature_vector(active)], dtype=torch.float32, device=device
    )
    cand_t = torch.tensor(
        [[candidate_feature_vector(active, action) for action in candidates]],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        logits, value = model(cand_t, target_t)
        log_probs = F.log_softmax(logits.squeeze(0), dim=-1)
        sample_t = torch.tensor(sample_dist, dtype=torch.float32, device=device)
        if rng is not None:
            idx = int(torch.multinomial(sample_t, num_samples=1, generator=rng).item())
        else:
            idx = int(torch.multinomial(sample_t, num_samples=1).item())
        chosen_log_prob = float(log_probs[idx].item())
        value_scalar = float(value.squeeze(0).item())

    return candidates, idx, chosen_log_prob, value_scalar, tuple(visit_dist)


def _build_search(env: DecompEnv, model, config: PPOConfig):
    """Construct an AndOrSearch that uses the current network as prior/value."""
    from .andor_search import AndOrSearch
    from .config import SearchConfig
    from .model import TorchPolicyValueWrapper

    wrapper = TorchPolicyValueWrapper(model)
    return AndOrSearch(
        factorizer=env.factorizer,
        baseline_model=env.baseline_model,
        model=wrapper,
        search_config=SearchConfig(
            simulations=config.mcts_simulations,
            max_depth=config.mcts_max_depth,
            expand_top_k=config.candidates_per_step,
        ),
        library=env.library,
    )


def compute_gae(
    transitions: list[Transition],
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    """Generalized advantage estimation for a single episode."""
    n = len(transitions)
    advantages = [0.0] * n
    last_adv = 0.0
    for t in reversed(range(n)):
        non_terminal = 0.0 if transitions[t].done else 1.0
        next_value = (
            0.0 if (t + 1 == n or transitions[t].done) else transitions[t + 1].value
        )
        delta = transitions[t].reward + gamma * next_value * non_terminal - transitions[t].value
        last_adv = delta + gamma * lam * non_terminal * last_adv
        advantages[t] = last_adv
    returns = [adv + tr.value for adv, tr in zip(advantages, transitions)]
    return advantages, returns


def ppo_update(
    model,
    optimizer,
    transitions: list[Transition],
    advantages: list[float],
    returns: list[float],
    config: PPOConfig,
) -> dict[str, float]:
    """Run ``config.update_epochs`` epochs of PPO on the rollout buffer."""
    if torch is None:
        raise RuntimeError("PyTorch is required for PPO training")
    n = len(transitions)
    sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    if n == 0:
        return sums

    device = next(model.parameters()).device
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    if adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
    old_logp = torch.tensor(
        [tr.log_prob for tr in transitions], dtype=torch.float32, device=device
    )
    has_mcts = any(tr.mcts_policy is not None for tr in transitions)
    sums["distill_loss"] = 0.0

    n_updates = 0
    for _ in range(config.update_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, config.minibatch_size):
            mb_idx = perm[start : start + config.minibatch_size].tolist()
            new_logp_list, value_list, entropy_list = [], [], []
            distill_terms: list[torch.Tensor] = []
            for i in mb_idx:
                tr = transitions[i]
                target_t = torch.tensor(
                    [target_feature_vector(tr.target)],
                    dtype=torch.float32,
                    device=device,
                )
                cand_t = torch.tensor(
                    [[candidate_feature_vector(tr.target, a) for a in tr.candidates]],
                    dtype=torch.float32,
                    device=device,
                )
                logits, value = model(cand_t, target_t)
                logits = logits.squeeze(0)
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum()
                new_logp_list.append(log_probs[tr.chosen_index])
                value_list.append(value.squeeze(0))
                entropy_list.append(entropy)
                if has_mcts and tr.mcts_policy is not None:
                    pi_mcts = torch.tensor(
                        tr.mcts_policy, dtype=torch.float32, device=device
                    )
                    # Cross-entropy CE(pi_mcts, pi_model) = -sum(pi_mcts * log_pi_model).
                    # Equivalent to KL(pi_mcts || pi_model) up to a target-only constant,
                    # so its gradient matches the KL distillation gradient.
                    distill_terms.append(-(pi_mcts * log_probs).sum())

            new_logp = torch.stack(new_logp_list)
            new_value = torch.stack(value_list)
            entropy_t = torch.stack(entropy_list)

            mb_old_logp = old_logp[torch.tensor(mb_idx, device=device)]
            mb_adv = adv_t[torch.tensor(mb_idx, device=device)]
            mb_ret = ret_t[torch.tensor(mb_idx, device=device)]

            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_value, mb_ret)
            entropy_bonus = entropy_t.mean()

            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy_bonus
            )
            distill_value = 0.0
            if distill_terms:
                distill_loss = torch.stack(distill_terms).mean()
                loss = loss + config.mcts_distill_coef * distill_loss
                distill_value = float(distill_loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = float((mb_old_logp - new_logp).mean().item())
            sums["policy_loss"] += float(policy_loss.item())
            sums["value_loss"] += float(value_loss.item())
            sums["entropy"] += float(entropy_bonus.item())
            sums["approx_kl"] += approx_kl
            sums["distill_loss"] += distill_value
            n_updates += 1

    return {k: v / max(1, n_updates) for k, v in sums.items()}


def train_ppo(
    targets: Sequence[SparsePolynomial],
    model,
    env: DecompEnv,
    config: PPOConfig,
    iterations: int = 100,
    log_callback: Callable[[TrainingMetrics], None] | None = None,
) -> list[TrainingMetrics]:
    """Run PPO training over a cycling list of target polynomials."""
    if torch is None:
        raise RuntimeError("PyTorch is required for PPO training")
    if not targets:
        raise ValueError("targets must be non-empty")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = None
    if config.seed is not None:
        rng = torch.Generator(device=next(model.parameters()).device)
        rng.manual_seed(config.seed)

    # Shared bundle so per-baseline memoisation accumulates across rollouts;
    # reuses the env's BaselineCostModel to keep sparse/horner caches aligned.
    baselines = BaselineBundle(baseline_model=env.baseline_model)

    target_iter = _cycle(targets)
    metrics_log: list[TrainingMetrics] = []

    for it in range(iterations):
        all_tr: list[Transition] = []
        all_adv: list[float] = []
        all_ret: list[float] = []
        ep_rewards, ep_lengths, ep_savings, ep_bonuses = [], [], [], []

        for _ in range(config.rollouts_per_update):
            target = next(target_iter)
            transitions = collect_episode(
                env, model, target, config, rng=rng, baselines=baselines
            )
            if not transitions:
                continue
            advs, rets = compute_gae(transitions, config.gamma, config.gae_lambda)
            all_tr.extend(transitions)
            all_adv.extend(advs)
            all_ret.extend(rets)
            ep_rewards.append(sum(t.reward for t in transitions))
            ep_lengths.append(len(transitions))
            ep_savings.append(
                sum(
                    t.reward
                    - config.library_reward_weight * t.library_reward
                    - t.terminal_bonus
                    for t in transitions
                )
            )
            ep_bonuses.append(sum(t.terminal_bonus for t in transitions))

        update_stats = ppo_update(model, optimizer, all_tr, all_adv, all_ret, config)
        metrics = TrainingMetrics(
            iteration=it,
            mean_episode_reward=_mean(ep_rewards),
            mean_episode_length=_mean(ep_lengths),
            mean_episode_savings=_mean(ep_savings),
            policy_loss=update_stats["policy_loss"],
            value_loss=update_stats["value_loss"],
            entropy=update_stats["entropy"],
            approx_kl=update_stats["approx_kl"],
            distill_loss=update_stats.get("distill_loss", 0.0),
            mean_terminal_bonus=_mean(ep_bonuses),
        )
        metrics_log.append(metrics)
        if log_callback is not None:
            log_callback(metrics)
    return metrics_log


def _cycle(items: Iterable[SparsePolynomial]) -> Iterator[SparsePolynomial]:
    seq = list(items)
    while True:
        for item in seq:
            yield item


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
