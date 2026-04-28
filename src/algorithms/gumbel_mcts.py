"""Gumbel root search for polynomial arithmetic circuit discovery.

This module implements a root-only Gumbel AlphaZero-style search:

1. sample root candidate actions by Gumbel-Top-k;
2. compare them using Sequential Halving;
3. return the selected action and a completed-Q improved policy target.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class GumbelMCTSOutput:
    action: int
    action_weights: np.ndarray
    root_q: np.ndarray
    root_visits: np.ndarray
    root_logits: np.ndarray
    considered_actions: np.ndarray


def sample_gumbel(shape, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(low=1e-8, high=1.0 - 1e-8, size=shape)
    return -np.log(-np.log(u))


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    out = np.zeros_like(logits, dtype=np.float32)
    if not np.any(mask):
        raise ValueError("masked_softmax received no valid actions")

    valid_logits = logits[mask]
    max_logit = np.max(valid_logits)
    exp_logits = np.exp(valid_logits - max_logit)
    out[mask] = exp_logits / (np.sum(exp_logits) + 1e-8)
    return out


def gumbel_top_k(
    masked_logits: np.ndarray,
    valid_mask: np.ndarray,
    k: int,
    rng: np.random.Generator,
    gumbel_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    valid_count = int(np.sum(valid_mask))
    if valid_count == 0:
        raise ValueError("No valid actions available")

    k = min(int(k), valid_count)
    gumbels = sample_gumbel(masked_logits.shape, rng) * gumbel_scale
    scores = np.where(valid_mask, masked_logits + gumbels, -np.inf)
    actions = np.argsort(-scores)[:k]
    return actions.astype(np.int64), gumbels.astype(np.float32)


def normalize_value_for_search(value: float, config) -> float:
    scale = max(1.0, float(getattr(config, "success_reward", 1.0)))
    return float(np.clip(float(value) / scale, -1.0, 1.0))


def transform_q(
    q: np.ndarray,
    visits: np.ndarray,
    valid_mask: np.ndarray,
    config,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    visits = np.asarray(visits, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if getattr(config, "gumbel_q_normalize", True):
        valid_q = q[valid_mask]
        q_min = np.min(valid_q)
        q_max = np.max(valid_q)
        q_used = (q - q_min) / (q_max - q_min + 1e-8)
    else:
        q_used = q

    c_visit = getattr(config, "gumbel_c_visit", 50.0)
    c_scale = getattr(config, "gumbel_c_scale", 0.1)
    max_visit = np.max(visits) if visits.size else 0.0
    sigma_q = (c_visit + max_visit) * c_scale * q_used
    return np.where(valid_mask, sigma_q, -np.inf).astype(np.float32)


def completed_q_policy(
    root_logits: np.ndarray,
    root_value: float,
    q: np.ndarray,
    visits: np.ndarray,
    valid_mask: np.ndarray,
    config,
) -> np.ndarray:
    completed_q = np.where(visits > 0, q, root_value)
    sigma_q = transform_q(completed_q, visits, valid_mask, config)
    target_logits = root_logits + sigma_q
    target_logits = np.where(valid_mask, target_logits, -np.inf)
    return masked_softmax(target_logits, valid_mask)


def visit_count_policy(visits: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    visits = np.asarray(visits, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    probs = np.zeros_like(visits, dtype=np.float32)

    valid_visits = visits[valid_mask]
    total = float(valid_visits.sum())
    if total > 0:
        probs[valid_mask] = valid_visits / total
        return probs

    probs[valid_mask] = 1.0 / max(int(valid_mask.sum()), 1)
    return probs


def terminal_value(reward, done, info, game, config) -> float:
    success = bool(info.get("is_success", False)) if isinstance(info, dict) else False
    if success:
        max_steps = max(1, getattr(config, "max_steps", 1))
        steps = getattr(game, "steps_taken", getattr(game, "num_steps", 0))
        return float(np.clip(1.0 - 0.05 * steps / max_steps, -1.0, 1.0))
    return -1.0


def _obs_to_device(obs: dict, device: Optional[str]) -> dict:
    if device is None:
        return obs

    out = {}
    for key, val in obs.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.to(device)
        elif isinstance(val, dict):
            out[key] = {
                sub_key: sub_val.to(device) if isinstance(sub_val, torch.Tensor) else sub_val
                for sub_key, sub_val in val.items()
            }
        elif hasattr(val, "to"):
            out[key] = val.to(device)
        else:
            out[key] = val
    return out


def evaluate_model_for_search(model, obs: dict, device: Optional[str] = None):
    obs_device = _obs_to_device(obs, device)
    logits, value = model(obs_device)
    logits_np = logits.squeeze(0).detach().cpu().numpy().astype(np.float32)
    value_f = float(value.squeeze(0).item())
    return logits_np, value_f


def simulate_after_root_action(
    game,
    action: int,
    model,
    config,
    device: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    del rng
    sim_game = game.clone()
    obs, reward, done, info = sim_game.step(int(action))

    if done:
        return terminal_value(reward, done, info, sim_game, config)

    _, value = evaluate_model_for_search(model, obs, device=device)
    gamma = getattr(config, "gamma", 0.99)
    q = (float(reward) + gamma * float(value)) / max(1.0, float(getattr(config, "success_reward", 1.0)))
    return float(np.clip(q, -1.0, 1.0))


def run_gumbel_mcts(
    game,
    model,
    config,
    device: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
    num_simulations: Optional[int] = None,
    gumbel_scale: Optional[float] = None,
) -> GumbelMCTSOutput:
    if rng is None:
        rng = np.random.default_rng()

    obs = game.get_observation()
    root_logits, root_value_raw = evaluate_model_for_search(model, obs, device=device)
    valid_mask = np.asarray(obs["mask"].cpu().numpy(), dtype=bool)

    root_logits = np.asarray(root_logits, dtype=np.float32)
    root_logits = np.where(valid_mask, root_logits, -np.inf)
    root_value = normalize_value_for_search(root_value_raw, config)

    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        raise RuntimeError("Gumbel MCTS found no valid actions")

    if valid_count == 1:
        action = int(np.flatnonzero(valid_mask)[0])
        action_weights = np.zeros_like(root_logits, dtype=np.float32)
        action_weights[action] = 1.0
        root_q = np.full_like(root_logits, root_value, dtype=np.float32)
        return GumbelMCTSOutput(
            action=action,
            action_weights=action_weights,
            root_q=root_q,
            root_visits=np.zeros_like(root_logits, dtype=np.int32),
            root_logits=root_logits,
            considered_actions=np.array([action], dtype=np.int64),
        )

    total_sims = int(num_simulations or getattr(config, "gumbel_num_simulations", 32))
    gumbel_scale = (
        getattr(config, "gumbel_scale", 1.0)
        if gumbel_scale is None else gumbel_scale
    )

    k = min(
        getattr(config, "gumbel_max_num_considered_actions", 16),
        total_sims,
        valid_count,
    )

    candidate_actions, gumbel_noise = gumbel_top_k(
        masked_logits=root_logits,
        valid_mask=valid_mask,
        k=k,
        rng=rng,
        gumbel_scale=gumbel_scale,
    )

    q_sum = np.zeros_like(root_logits, dtype=np.float32)
    visits = np.zeros_like(root_logits, dtype=np.int32)

    remaining = list(map(int, candidate_actions))
    num_phases = int(np.ceil(np.log2(max(2, len(remaining)))))
    sims_used = 0

    for phase in range(num_phases):
        if len(remaining) <= 1:
            break

        sims_left = max(0, total_sims - sims_used)
        if sims_left <= 0:
            break

        phases_left = max(1, num_phases - phase)
        sims_per_action = max(1, sims_left // (phases_left * len(remaining)))

        for action in remaining:
            for _ in range(sims_per_action):
                if sims_used >= total_sims:
                    break
                q = simulate_after_root_action(
                    game=game,
                    action=action,
                    model=model,
                    config=config,
                    device=device,
                    rng=rng,
                )
                q_sum[action] += q
                visits[action] += 1
                sims_used += 1

        q_mean = np.where(visits > 0, q_sum / np.maximum(visits, 1), root_value)
        sigma_q = transform_q(q_mean, visits, valid_mask, config)

        scores = [(a, gumbel_noise[a] + root_logits[a] + sigma_q[a]) for a in remaining]
        scores.sort(key=lambda item: item[1], reverse=True)
        remaining = [action for action, _ in scores[:max(1, len(scores) // 2)]]

    q_mean = np.where(visits > 0, q_sum / np.maximum(visits, 1), root_value).astype(np.float32)
    sigma_q = transform_q(q_mean, visits, valid_mask, config)

    final_scores = np.full_like(root_logits, -np.inf, dtype=np.float32)
    for action in remaining:
        final_scores[action] = gumbel_noise[action] + root_logits[action] + sigma_q[action]

    selected_action = int(np.argmax(final_scores))
    if getattr(config, "gumbel_policy_target", "completed_q") == "visits":
        action_weights = visit_count_policy(visits, valid_mask)
    else:
        action_weights = completed_q_policy(
            root_logits, root_value, q_mean, visits, valid_mask, config
        )

    return GumbelMCTSOutput(
        action=selected_action,
        action_weights=action_weights.astype(np.float32),
        root_q=q_mean.astype(np.float32),
        root_visits=visits.astype(np.int32),
        root_logits=root_logits.astype(np.float32),
        considered_actions=np.asarray(candidate_actions, dtype=np.int64),
    )