from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import Config
from ..env.circuit_env import PolyCircuitEnv
from ..env.samplers import (
    GenerativeInterestingPolynomialSampler,
    InterestingPolynomialSampler,
)
from .agent import DQNAgent


def collect_episode(
    env: PolyCircuitEnv,
    agent: DQNAgent,
    max_ops: int,
    deterministic: bool = False,
    target_poly=None,
) -> Dict:
    """Run one episode, store in buffer with HER, return stats."""
    options = {"max_ops": max_ops}
    if target_poly is not None:
        options["target_poly"] = target_poly
    obs_dict, _ = env.reset(options=options)

    ep_obs: List[np.ndarray] = []
    ep_actions: List[int] = []
    ep_rewards: List[float] = []
    ep_next_obs: List[np.ndarray] = []
    ep_dones: List[bool] = []
    ep_masks: List[np.ndarray] = []
    ep_next_masks: List[np.ndarray] = []
    ep_node_evals: List[List[np.ndarray]] = []

    total_reward = 0.0
    steps = 0
    done = False
    solved = False

    # Safety cap: belt-and-suspenders in case env step limit is misconfigured.
    # The env's own _episode_step_limit is the primary guard.
    _hard_cap = max_ops * 4 + env.config.L + 10

    while not done:
        if steps >= _hard_cap:
            # Should never be reached; indicates a bug in env termination logic.
            break

        obs = obs_dict["obs"]
        mask = obs_dict["action_mask"]

        action = agent.select_action(obs, mask, deterministic=deterministic)
        if not deterministic:
            agent.total_steps += 1

        next_obs_dict, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_obs.append(obs)
        ep_actions.append(action)
        ep_rewards.append(reward)
        ep_next_obs.append(next_obs_dict["obs"])
        ep_dones.append(done)
        ep_masks.append(mask)
        ep_next_masks.append(next_obs_dict["action_mask"])

        # Node evals from trajectory (env records them)
        traj = env.get_trajectory()
        ep_node_evals.append(traj[-1]["node_evals"] if traj else [])

        total_reward += reward
        steps += 1
        solved = info.get("solved", False)
        obs_dict = next_obs_dict

        # Interleaved training
        if not deterministic and agent.total_steps >= agent.config.learning_starts:
            if agent.total_steps % agent.config.train_freq == 0:
                agent.train_step()
                agent.soft_update_target()

    # Store with HER
    if not deterministic:
        agent.buffer.add_episode_with_her(
            ep_obs=ep_obs,
            ep_actions=ep_actions,
            ep_rewards=ep_rewards,
            ep_next_obs=ep_next_obs,
            ep_dones=ep_dones,
            ep_masks=ep_masks,
            ep_next_masks=ep_next_masks,
            ep_node_evals=ep_node_evals,
        )

    return {"solved": solved, "reward": total_reward, "length": steps}


def evaluate(
    env: PolyCircuitEnv,
    agent: DQNAgent,
    max_ops: int,
    num_episodes: int,
) -> Dict:
    """Evaluate with deterministic policy."""
    solved_count = 0
    total_reward = 0.0
    total_steps = 0

    saved = agent.total_steps
    for _ in range(num_episodes):
        obs_dict, _ = env.reset(options={"max_ops": max_ops})
        done = False
        ep_r = 0.0
        ep_s = 0
        info = {}
        while not done:
            obs = obs_dict["obs"]
            mask = obs_dict["action_mask"]
            action = agent.select_action(obs, mask, deterministic=True)
            obs_dict, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_r += r
            ep_s += 1
        if info.get("solved", False):
            solved_count += 1
        total_reward += ep_r
        total_steps += ep_s
    agent.total_steps = saved

    return {
        "success_rate": solved_count / max(num_episodes, 1),
        "avg_reward": total_reward / max(num_episodes, 1),
        "avg_steps": total_steps / max(num_episodes, 1),
    }


def train(
    config: Optional[Config] = None,
    interesting_jsonl: Optional[str] = None,
) -> DQNAgent:
    """Main training loop with curriculum and mixed sampling.

    Args:
        config: Training configuration.
        interesting_jsonl: Path to analysis JSONL for interesting polynomials.
            If provided, uses mixed sampling (random + interesting).
    """
    if config is None:
        config = Config()

    env = PolyCircuitEnv(config)
    agent = DQNAgent(config)

    # Optional interesting polynomial sampler
    interesting_sampler = None
    if interesting_jsonl and Path(interesting_jsonl).exists():
        # Explicit JSONL takes precedence
        try:
            interesting_sampler = InterestingPolynomialSampler(
                interesting_jsonl, n_vars=config.n_vars,
            )
            print(f"Loaded {len(interesting_sampler)} interesting polynomials")
        except Exception as e:
            print(f"Warning: could not load interesting polys: {e}")
    elif config.auto_interesting:
        # Auto-generate interesting polynomials from graph enumeration
        max_steps = max(config.curriculum_levels)
        try:
            interesting_sampler = GenerativeInterestingPolynomialSampler(
                n_vars=config.n_vars,
                max_steps=max_steps,
                only_multipath=True,
                max_graph_nodes=config.gen_max_graph_nodes,
                max_successors_per_node=config.gen_max_successors,
            )
            print(f"Auto-generating interesting polynomials (max_steps={max_steps})")
        except Exception as e:
            print(f"Warning: could not init auto interesting sampler: {e}")

    print(f"Parameters: {agent.q_network.count_parameters()}")
    print(f"Action dim: {config.action_dim}  |  Obs dim: {config.obs_dim}")

    cur_level = 0
    cur_max_ops = config.curriculum_levels[cur_level]
    window: deque = deque(maxlen=config.curriculum_window)
    best_sr = 0.0
    episode = 0

    import random as _random
    train_rng = _random.Random(config.seed + 100)

    while agent.total_steps < config.total_steps:
        # Mixed sampling: use interesting polys at higher curriculum levels
        target_poly = None
        if interesting_sampler:
            if train_rng.random() < config.interesting_ratio:
                target_poly, _ = interesting_sampler.sample(train_rng, max_ops=cur_max_ops)

        result = collect_episode(env, agent, max_ops=cur_max_ops, target_poly=target_poly)
        episode += 1
        window.append(1.0 if result["solved"] else 0.0)
        sr = sum(window) / len(window) if window else 0.0

        # Log
        if episode % 100 == 0:
            eps = agent._epsilon()
            avg_loss = (
                np.mean(agent.training_losses[-100:])
                if agent.training_losses else 0.0
            )
            print(
                f"Ep {episode} | Steps {agent.total_steps} | "
                f"Lvl {cur_level} (ops={cur_max_ops}) | "
                f"SR {sr:.2%} | Eps {eps:.3f} | "
                f"Loss {avg_loss:.4f} | Buf {len(agent.buffer)}"
            )

        # Curriculum advance
        if (
            len(window) >= config.curriculum_window
            and sr >= config.curriculum_threshold
            and cur_level < len(config.curriculum_levels) - 1
        ):
            cur_level += 1
            cur_max_ops = config.curriculum_levels[cur_level]
            window.clear()
            print(f"=== ADVANCE: Level {cur_level}, max_ops={cur_max_ops} ===")

        # Periodic eval
        if episode % 500 == 0:
            ev = evaluate(env, agent, cur_max_ops, config.eval_episodes)
            print(f"  [EVAL] SR={ev['success_rate']:.2%} ({config.eval_episodes} eps)")
            if ev["success_rate"] > best_sr:
                best_sr = ev["success_rate"]
                os.makedirs(config.log_dir, exist_ok=True)
                path = os.path.join(config.log_dir, f"best_lvl{cur_level}.pt")
                agent.save(path)
                print(f"  [EVAL] New best! Saved to {path}")

    os.makedirs(config.log_dir, exist_ok=True)
    agent.save(os.path.join(config.log_dir, "final.pt"))
    print("Training complete.")
    return agent
