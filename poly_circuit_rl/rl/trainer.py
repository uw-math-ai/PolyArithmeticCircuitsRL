from __future__ import annotations

import os
import random as _random
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..baselines.exhaustive import ExhaustiveSearch
from ..config import Config
from ..core.factor_library import FactorLibrary
from ..core.poly import Poly, PolyKey, poly_hashkey
from ..env.circuit_env import PolyCircuitEnv
from ..env.samplers import (
    GenerativeInterestingPolynomialSampler,
    InterestingPolynomialSampler,
    RandomCircuitSampler,
)
from .agent import DQNAgent
from .mcts import MCTS


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducible training runs."""
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def should_use_mcts_for_level(level_idx: int, config: Config) -> bool:
    """Use MCTS from curriculum level 1 onward when enabled."""
    return config.use_mcts and level_idx >= 1


def should_mix_interesting_targets(level_idx: int, config: Config) -> bool:
    """Only mix interesting targets from curriculum level 1 onward."""
    return config.interesting_ratio > 0.0 and level_idx >= 1


def maybe_sample_target_poly(
    level_idx: int,
    max_ops: int,
    config: Config,
    rng,
    interesting_sampler,
):
    """Sample from the interesting-target mixture for this level, if enabled."""
    if interesting_sampler is None or not should_mix_interesting_targets(level_idx, config):
        return None
    if rng.random() >= config.interesting_ratio:
        return None
    target_poly, _ = interesting_sampler.sample(rng, max_ops=max_ops)
    return target_poly


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a Bernoulli success rate."""
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)
    return max(0.0, center - margin), min(1.0, center + margin)


def _presample_eval_targets(
    config: Config,
    levels: List[int],
    interesting_sampler,
) -> Dict[int, List[Poly]]:
    """Deterministically pre-sample held-out eval targets per curriculum level."""
    rng = _random.Random(config.seed + 90_000)
    eval_targets: Dict[int, List[Poly]] = {}
    for level_idx, max_ops in enumerate(levels):
        random_sampler = RandomCircuitSampler(n_vars=config.n_vars, max_steps=max_ops)
        targets: List[Poly] = []
        for _ in range(config.eval_episodes):
            target_poly = maybe_sample_target_poly(
                level_idx,
                max_ops,
                config,
                rng,
                interesting_sampler,
            )
            if target_poly is None:
                target_poly, _ = random_sampler.sample(rng)
            targets.append(target_poly)
        eval_targets[level_idx] = targets
    return eval_targets


def collect_episode(
    env: PolyCircuitEnv,
    agent: DQNAgent,
    max_ops: int,
    deterministic: bool = False,
    target_poly=None,
    mcts: Optional[MCTS] = None,
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
    ep_solved: List[bool] = []
    ep_truncated: List[bool] = []
    ep_masks: List[np.ndarray] = []
    ep_next_masks: List[np.ndarray] = []
    ep_achieved_goals: List[Optional[np.ndarray]] = []
    ep_achieved_goal_keys: List[Optional[PolyKey]] = []
    ep_base_rewards: List[float] = []
    ep_shaping_rewards: List[float] = []
    ep_solve_bonuses: List[float] = []

    total_reward = 0.0
    steps = 0
    done = False
    solved = False

    # Safety cap
    _hard_cap = max_ops * 4 + env.config.L + 10

    while not done:
        if steps >= _hard_cap:
            break

        obs = obs_dict["obs"]
        mask = obs_dict["action_mask"]

        if mcts is not None:
            action = mcts.search(obs, mask)
        else:
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
        ep_solved.append(terminated)
        ep_truncated.append(truncated)
        ep_masks.append(mask)
        ep_next_masks.append(next_obs_dict["action_mask"])

        # Extract decomposed rewards from trajectory
        traj = env.get_trajectory()
        if traj:
            last_entry = traj[-1]
            ep_achieved_goals.append(last_entry["achieved_goal"])
            ep_achieved_goal_keys.append(last_entry.get("achieved_goal_key"))
            ep_base_rewards.append(last_entry.get("base_reward", reward))
            ep_shaping_rewards.append(last_entry.get("shaping_reward", 0.0))
            ep_solve_bonuses.append(last_entry.get("solve_bonus", 0.0))
        else:
            ep_achieved_goals.append(None)
            ep_achieved_goal_keys.append(None)
            ep_base_rewards.append(reward)
            ep_shaping_rewards.append(0.0)
            ep_solve_bonuses.append(0.0)

        total_reward += reward
        steps += 1
        solved = info.get("solved", False)
        obs_dict = next_obs_dict

        # Interleaved training
        if not deterministic and agent.total_steps >= agent.config.learning_starts:
            if agent.total_steps % agent.config.train_freq == 0:
                agent.train_step()
                agent.soft_update_target()

    # Store with HER (decomposed rewards enable proper relabeling)
    if not deterministic:
        agent.buffer.add_episode_with_her(
            ep_obs=ep_obs,
            ep_actions=ep_actions,
            ep_rewards=ep_rewards,
            ep_next_obs=ep_next_obs,
            ep_dones=ep_dones,
            ep_solved=ep_solved,
            ep_truncated=ep_truncated,
            ep_masks=ep_masks,
            ep_next_masks=ep_next_masks,
            ep_achieved_goals=ep_achieved_goals,
            ep_achieved_goal_keys=ep_achieved_goal_keys,
            ep_base_rewards=ep_base_rewards,
            ep_shaping_rewards=ep_shaping_rewards,
            ep_solve_bonuses=ep_solve_bonuses,
        )

    return {"solved": solved, "reward": total_reward, "length": steps}


def evaluate(
    env: PolyCircuitEnv,
    agent: DQNAgent,
    max_ops: int,
    num_episodes: int,
    mcts: Optional[MCTS] = None,
    level_idx: int = 0,
    interesting_sampler=None,
    target_rng=None,
    eval_targets: Optional[List[Poly]] = None,
    optimal_ops: Optional[Dict[PolyKey, Optional[int]]] = None,
) -> Dict:
    """Evaluate with deterministic policy (or MCTS if provided)."""
    solved_count = 0
    total_reward = 0.0
    total_steps = 0
    solved_op_gaps: List[float] = []

    if target_rng is None:
        target_rng = _random.Random(env.config.seed + 10_000 + level_idx)

    saved = agent.total_steps
    try:
        episode_targets = eval_targets
        if episode_targets is None:
            episode_targets = []
            for _ in range(num_episodes):
                target_poly = maybe_sample_target_poly(
                    level_idx,
                    max_ops,
                    env.config,
                    target_rng,
                    interesting_sampler,
                )
                episode_targets.append(target_poly)

        for target_poly in episode_targets:
            options = {"max_ops": max_ops}
            if target_poly is not None:
                options["target_poly"] = target_poly
            obs_dict, _ = env.reset(options=options)
            done = False
            ep_r = 0.0
            ep_s = 0
            info = {}
            while not done:
                obs = obs_dict["obs"]
                mask = obs_dict["action_mask"]
                if mcts is not None:
                    action = mcts.search(obs, mask)
                else:
                    action = agent.select_action(obs, mask, deterministic=True)
                obs_dict, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_r += r
                ep_s += 1
            if info.get("solved", False):
                solved_count += 1
                if target_poly is not None and optimal_ops is not None:
                    target_key = poly_hashkey(target_poly)
                    best = optimal_ops.get(target_key)
                    if best is not None:
                        actual_ops = max_ops - env.steps_left
                        solved_op_gaps.append(float(actual_ops - best))
            total_reward += ep_r
            total_steps += ep_s
    finally:
        agent.total_steps = saved

    n_eval = len(eval_targets) if eval_targets is not None else num_episodes
    wilson_low, wilson_high = _wilson_interval(solved_count, max(n_eval, 1))

    return {
        "success_rate": solved_count / max(n_eval, 1),
        "avg_reward": total_reward / max(n_eval, 1),
        "avg_steps": total_steps / max(n_eval, 1),
        "solved_count": solved_count,
        "n_eval": n_eval,
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "mean_gap_to_optimal": (
            float(np.mean(solved_op_gaps))
            if solved_op_gaps
            else None
        ),
    }


def train(
    config: Optional[Config] = None,
    interesting_jsonl: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> DQNAgent:
    """Main training loop with factor library, HER, curriculum, and expert demos.

    Key improvements over previous version:
    - Factor library provides intermediate subgoal rewards
    - HER relabeling is always enabled (shaping stripped from relabeled transitions)
    - Expert demonstrations pre-fill the replay buffer
    - Lower epsilon floor (0.02) and better curriculum thresholds
    """
    if config is None:
        config = Config()
    assert config.eval_every > 0, "eval_every must be > 0"
    _seed_everything(config.seed)

    # Optional Weights & Biases tracking
    wb = None
    if wandb_project is not None:
        try:
            import wandb

            wb = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=dict(
                    n_vars=config.n_vars,
                    max_ops=config.max_ops,
                    L=config.L,
                    m=config.m,
                    step_cost=config.step_cost,
                    shaping_coeff=config.shaping_coeff,
                    factor_shaping_coeff=config.factor_shaping_coeff,
                    reward_mode=config.reward_mode,
                    factor_library_enabled=config.factor_library_enabled,
                    factor_library_max_size=config.factor_library_max_size,
                    factor_subgoal_reward=config.factor_subgoal_reward,
                    factor_library_bonus=config.factor_library_bonus,
                    completion_bonus=config.completion_bonus,
                    expert_demo_count=config.expert_demo_count,
                    allow_partial_demos=config.allow_partial_demos,
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_layers=config.n_layers,
                    dropout=config.dropout,
                    lr=config.lr,
                    gamma=config.gamma,
                    batch_size=config.batch_size,
                    buffer_size=config.buffer_size,
                    eps_start=config.eps_start,
                    eps_end=config.eps_end,
                    eps_decay_steps=config.eps_decay_steps,
                    target_update_tau=config.target_update_tau,
                    train_freq=config.train_freq,
                    learning_starts=config.learning_starts,
                    her_k=config.her_k,
                    curriculum_levels=list(config.curriculum_levels),
                    curriculum_window=config.curriculum_window,
                    curriculum_train_threshold=config.curriculum_train_threshold,
                    curriculum_eval_threshold=config.curriculum_eval_threshold,
                    interesting_ratio=config.interesting_ratio,
                    auto_interesting=config.auto_interesting,
                    gen_max_seconds=config.gen_max_seconds,
                    use_mcts=config.use_mcts,
                    mcts_simulations=config.mcts_simulations,
                    wandb_artifact_min_interval_steps=config.wandb_artifact_min_interval_steps,
                    total_steps=config.total_steps,
                    seed=config.seed,
                ),
            )
            print(f"W&B run: {wb.url}")
        except ImportError:
            print("Warning: wandb not installed. Run: pip install wandb")
            wb = None
        except Exception as e:
            print(f"Warning: failed to initialize wandb: {e}")
            wb = None

    # --- Factor library ---
    factor_library = None
    if config.factor_library_enabled:
        factor_library = FactorLibrary(
            n_vars=config.n_vars,
            max_size=config.factor_library_max_size,
        )
        print(f"Factor library ENABLED (subgoal={config.factor_subgoal_reward}, "
              f"library_bonus={config.factor_library_bonus}, "
              f"completion={config.completion_bonus})")

    env = PolyCircuitEnv(config, factor_library=factor_library)
    eval_env = PolyCircuitEnv(
        config,
        factor_library=factor_library.frozen_view() if factor_library is not None else None,
    )
    eval_env.eval_points = env.eval_points
    eval_env.rng = _random.Random(config.seed + 200)
    agent = DQNAgent(config)

    # Optional interesting polynomial sampler
    interesting_sampler = None
    if interesting_jsonl and Path(interesting_jsonl).exists():
        try:
            interesting_sampler = InterestingPolynomialSampler(
                interesting_jsonl, n_vars=config.n_vars,
            )
            print(f"Loaded {len(interesting_sampler)} interesting polynomials")
        except Exception as e:
            print(f"Warning: could not load interesting polys: {e}")
    elif config.auto_interesting:
        max_steps = max(config.curriculum_levels)
        try:
            interesting_sampler = GenerativeInterestingPolynomialSampler(
                n_vars=config.n_vars,
                max_steps=max_steps,
                only_shortcut=True,
                max_graph_nodes=config.gen_max_graph_nodes,
                max_successors_per_node=config.gen_max_successors,
                max_seconds=config.gen_max_seconds,
            )
            print(f"Auto-generating interesting polynomials (max_steps={max_steps})")
        except Exception as e:
            print(f"Warning: could not init auto interesting sampler: {e}")

    # Oracle mask diagnostic
    if config.oracle_mask and interesting_sampler is not None:
        from ..env.oracle_mask import OracleMaskHelper
        max_level = max(config.curriculum_levels)
        if hasattr(interesting_sampler, '_ensure_built'):
            interesting_sampler._ensure_built(max_level)
        G, dist, roots = interesting_sampler.get_dag_data()
        if G is not None:
            var_names = interesting_sampler.var_names
            oracle_helper = OracleMaskHelper(G, dist, roots, config.n_vars, var_names)
            env._oracle_helper = oracle_helper
            eval_env._oracle_helper = oracle_helper
            print(f"Oracle mask ENABLED ({G.number_of_nodes()} DAG nodes)")
        else:
            print("Warning: oracle_mask requested but DAG not available")
    elif config.oracle_mask:
        print("Warning: oracle_mask requires an interesting polynomial sampler")

    print(f"Parameters: {agent.q_network.count_parameters()}")
    print(f"Action dim: {config.action_dim}  |  Obs dim: {config.obs_dim}")

    # --- Expert demo pre-fill ---
    if config.expert_demo_count > 0:
        try:
            from ..env.expert_demos import ExpertDemoGenerator

            demo_gen = ExpertDemoGenerator(config)
            demo_gen.build_graph(max_steps=max(config.curriculum_levels))
            demos = demo_gen.generate_demos(
                env,
                num_demos=config.expert_demo_count,
                curriculum_levels=config.curriculum_levels,
            )
            if len(demos) < 0.5 * config.expert_demo_count:
                msg = (
                    f"Expert demo prefill is low: requested {config.expert_demo_count}, "
                    f"generated {len(demos)} transitions."
                )
                warnings.warn(msg, stacklevel=2)
                assert config.allow_partial_demos, msg
            for t in demos:
                agent.buffer.add(t)
            print(f"Pre-filled {len(demos)} expert demo transitions "
                  f"({len(agent.buffer)} total in buffer)")
        except Exception as e:
            print(f"Warning: expert demo generation failed: {e}")

    cur_level = 0
    cur_max_ops = config.curriculum_levels[cur_level]
    window: deque = deque(maxlen=config.curriculum_window)
    last_eval_sr: Optional[float] = None
    best_sr = 0.0
    best_eval_avg_reward = float("-inf")
    episode = 0

    levels = list(config.curriculum_levels)
    eval_targets_by_level = _presample_eval_targets(config, levels, interesting_sampler)
    optimal_ops_by_level: Dict[int, Dict[PolyKey, Optional[int]]] = {}
    exhaustive_by_max_ops: Dict[int, ExhaustiveSearch] = {}
    for level_idx, level_max_ops in enumerate(levels):
        if level_max_ops > 4:
            continue
        search = exhaustive_by_max_ops.get(level_max_ops)
        if search is None:
            search = ExhaustiveSearch(config)
            search.build(level_max_ops)
            exhaustive_by_max_ops[level_max_ops] = search
        optimal_ops_by_level[level_idx] = {
            poly_hashkey(target): search.find_optimal(target)
            for target in eval_targets_by_level[level_idx]
        }

    train_rng = _random.Random(config.seed + 100)
    last_log_time = time.perf_counter()
    last_log_steps = agent.total_steps

    # MCTS setup
    train_mcts = MCTS(agent, env, config) if config.use_mcts else None
    last_artifact_step = -config.wandb_artifact_min_interval_steps
    next_eval_step = config.eval_every

    while agent.total_steps < config.total_steps:
        # Mixed sampling
        target_poly = maybe_sample_target_poly(
            cur_level,
            cur_max_ops,
            config,
            train_rng,
            interesting_sampler,
        )

        ep_mcts = train_mcts if should_use_mcts_for_level(cur_level, config) else None
        result = collect_episode(env, agent, max_ops=cur_max_ops, target_poly=target_poly, mcts=ep_mcts)
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
            now = time.perf_counter()
            dt = max(now - last_log_time, 1e-9)
            ds = max(agent.total_steps - last_log_steps, 0)
            steps_per_sec = ds / dt
            buffer_fill_ratio = len(agent.buffer) / max(config.buffer_size, 1)
            fl_size = len(factor_library) if factor_library else 0
            print(
                f"Ep {episode} | Steps {agent.total_steps} | "
                f"Lvl {cur_level} (ops={cur_max_ops}) | "
                f"SR {sr:.2%} | Eps {eps:.3f} | "
                f"Loss {avg_loss:.4f} | Buf {len(agent.buffer)} | "
                f"FL {fl_size}"
            )
            if wb is not None:
                wb.log(
                    {
                        "train/success_rate": sr,
                        "train/epsilon": eps,
                        "train/loss": avg_loss,
                        "train/buffer_size": len(agent.buffer),
                        "train/buffer_fill_ratio": buffer_fill_ratio,
                        "train/steps_per_sec": steps_per_sec,
                        "train/curriculum_level": cur_level,
                        "train/max_ops": cur_max_ops,
                        "train/episode": episode,
                        "train/episode_reward": result["reward"],
                        "train/episode_length": result["length"],
                        "factor_library/size": fl_size,
                    },
                    step=agent.total_steps,
                )
            last_log_time = now
            last_log_steps = agent.total_steps

        # Curriculum advance
        if (
            len(window) >= config.curriculum_window
            and sr >= config.curriculum_train_threshold
            and last_eval_sr is not None
            and last_eval_sr >= config.curriculum_eval_threshold
            and cur_level < len(config.curriculum_levels) - 1
        ):
            cur_level += 1
            cur_max_ops = config.curriculum_levels[cur_level]
            window.clear()
            last_eval_sr = None
            print(f"=== ADVANCE: Level {cur_level}, max_ops={cur_max_ops} (buffer retained) ===")
            if wb is not None:
                wb.log(
                    {
                        "curriculum/level": cur_level,
                        "curriculum/max_ops": cur_max_ops,
                    },
                    step=agent.total_steps,
                )

        # Periodic eval
        while next_eval_step is not None and agent.total_steps >= next_eval_step:
            eval_mcts = (
                MCTS(agent, eval_env, config)
                if should_use_mcts_for_level(cur_level, config)
                else None
            )
            ev = evaluate(
                eval_env,
                agent,
                cur_max_ops,
                config.eval_episodes,
                mcts=eval_mcts,
                level_idx=cur_level,
                eval_targets=eval_targets_by_level[cur_level],
                optimal_ops=optimal_ops_by_level.get(cur_level),
            )
            last_eval_sr = ev["success_rate"]
            eval_line = (
                f"  [EVAL] SR={ev['success_rate']:.2%} "
                f"[{ev['wilson_low']:.2%}, {ev['wilson_high']:.2%}] "
                f"({ev['n_eval']} eps)"
            )
            if ev["mean_gap_to_optimal"] is not None:
                eval_line += f" | gap-to-opt={ev['mean_gap_to_optimal']:.3f}"
            print(eval_line)
            prev_best_sr = best_sr
            best_sr = max(best_sr, ev["success_rate"])
            best_eval_avg_reward = max(best_eval_avg_reward, ev["avg_reward"])
            if wb is not None:
                wb.log(
                    {
                        "eval/success_rate": ev["success_rate"],
                        "eval/avg_reward": ev["avg_reward"],
                        "eval/avg_steps": ev["avg_steps"],
                        "eval/best_success_rate": best_sr,
                        "eval/best_avg_reward": best_eval_avg_reward,
                        "eval/wilson_low": ev["wilson_low"],
                        "eval/wilson_high": ev["wilson_high"],
                        "eval/mean_gap_to_optimal": (
                            ev["mean_gap_to_optimal"]
                            if ev["mean_gap_to_optimal"] is not None
                            else np.nan
                        ),
                    },
                    step=agent.total_steps,
                )
            if ev["success_rate"] > prev_best_sr:
                os.makedirs(config.log_dir, exist_ok=True)
                path = os.path.join(config.log_dir, f"best_lvl{cur_level}.pt")
                agent.save(path)
                print(f"  [EVAL] New best! Saved to {path}")
                if (
                    wb is not None
                    and (agent.total_steps - last_artifact_step) >= config.wandb_artifact_min_interval_steps
                ):
                    try:
                        artifact = wandb.Artifact(
                            name=f"best-model-{wb.id}",
                            type="model",
                            metadata={
                                "curriculum_level": cur_level,
                                "max_ops": cur_max_ops,
                                "success_rate": ev["success_rate"],
                                "avg_reward": ev["avg_reward"],
                                "step": agent.total_steps,
                            },
                        )
                        artifact.add_file(path)
                        wb.log_artifact(
                            artifact,
                            aliases=["best", "latest", f"lvl{cur_level}"],
                        )
                        last_artifact_step = agent.total_steps
                    except Exception as e:
                        print(f"Warning: failed to log model artifact: {e}")
            next_eval_step += config.eval_every

    os.makedirs(config.log_dir, exist_ok=True)
    final_path = os.path.join(config.log_dir, "final.pt")
    agent.save(final_path)
    print("Training complete.")
    if wb is not None:
        try:
            artifact = wandb.Artifact(
                name=f"final-model-{wb.id}",
                type="model",
                metadata={
                    "step": agent.total_steps,
                    "curriculum_level": cur_level,
                    "max_ops": cur_max_ops,
                },
            )
            artifact.add_file(final_path)
            wb.log_artifact(artifact, aliases=["final", "latest-final"])
        except Exception as e:
            print(f"Warning: failed to log final model artifact: {e}")
        wb.finish()
    return agent
