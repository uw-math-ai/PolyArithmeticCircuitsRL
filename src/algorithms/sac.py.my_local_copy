"""Discrete SAC trainer adapted to the rewritten circuit environment.

Design goals:
- Fit the same modular stack as PPO/AlphaZero (`Config`, `CircuitGame`, `main.py`)
- Respect action masks at every policy/value computation
- Use replay that stays balanced across complexity levels and success/failure
- Support optional warm-start from constructive trajectories
- Support optional MCTS policy distillation as a soft guidance signal
"""

from __future__ import annotations

import copy
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..config import Config
from ..environment.action_space import encode_action
from ..environment.circuit_game import CircuitGame
from ..environment.polynomial_utils import create_variables, fast_to_sympy
from ..game_board.generator import build_game_board, generate_random_circuit
from ..models.gnn_encoder import CircuitGNN
from .mcts import MCTS


@dataclass
class Transition:
    obs: dict
    action: int
    reward: float
    next_obs: dict
    done: bool
    complexity: int
    success: bool
    demo: bool
    mcts_policy: Optional[np.ndarray]


class StratifiedReplayBuffer:
    """Replay buffer balancing complexity and success strata."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._all = deque(maxlen=max_size)
        self._by_complexity = defaultdict(lambda: {True: deque(), False: deque()})

    def add(self, transition: Transition):
        # Remove evicted transition from side indexes
        if len(self._all) == self.max_size:
            evicted = self._all[0]
            bucket = self._by_complexity[evicted.complexity][evicted.success]
            try:
                bucket.remove(evicted)
            except ValueError:
                pass

        self._all.append(transition)
        self._by_complexity[transition.complexity][transition.success].append(transition)

    def __len__(self):
        return len(self._all)

    def sample(self, batch_size: int, success_ratio: float) -> List[Transition]:
        if not self._all:
            return []

        complexities = list(self._by_complexity.keys())
        batch = []
        for _ in range(batch_size):
            comp = random.choice(complexities)
            choose_success = random.random() < success_ratio
            preferred = self._by_complexity[comp][choose_success]
            fallback = self._by_complexity[comp][not choose_success]

            if preferred:
                batch.append(random.choice(preferred))
            elif fallback:
                batch.append(random.choice(fallback))
            else:
                batch.append(random.choice(self._all))
        return batch


class SACActor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        hidden = config.hidden_dim
        emb = config.embedding_dim

        self.gnn = CircuitGNN(
            input_dim=config.node_feature_dim,
            hidden_dim=hidden,
            output_dim=emb,
            num_layers=config.num_gnn_layers,
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(config.target_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.max_actions),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        graph = obs["graph"]
        target = obs["target"]
        mask = obs["mask"]

        if isinstance(graph, dict):
            graph_emb = self.gnn(
                graph["x"],
                graph["edge_index"],
                num_nodes_actual=graph.get("num_nodes_actual"),
            )
        else:
            graph_emb = self.gnn(
                graph.x,
                graph.edge_index,
                num_nodes_actual=getattr(graph, "num_nodes_actual", None),
                batch=getattr(graph, "batch", None),
            )

        if target.dim() == 1:
            target = target.unsqueeze(0)
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        target_emb = self.target_encoder(target)
        fused = self.fusion(torch.cat([graph_emb, target_emb], dim=-1))
        logits = self.policy_head(fused)
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class SACCritic(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        hidden = config.hidden_dim
        emb = config.embedding_dim

        self.gnn = CircuitGNN(
            input_dim=config.node_feature_dim,
            hidden_dim=hidden,
            output_dim=emb,
            num_layers=config.num_gnn_layers,
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(config.target_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.max_actions),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        graph = obs["graph"]
        target = obs["target"]
        mask = obs["mask"]

        if isinstance(graph, dict):
            graph_emb = self.gnn(
                graph["x"],
                graph["edge_index"],
                num_nodes_actual=graph.get("num_nodes_actual"),
            )
        else:
            graph_emb = self.gnn(
                graph.x,
                graph.edge_index,
                num_nodes_actual=getattr(graph, "num_nodes_actual", None),
                batch=getattr(graph, "batch", None),
            )

        if target.dim() == 1:
            target = target.unsqueeze(0)
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        target_emb = self.target_encoder(target)
        fused = self.fusion(torch.cat([graph_emb, target_emb], dim=-1))
        q_values = self.q_head(fused)
        # Keep invalid actions out of max/sum reductions.
        q_values = q_values.masked_fill(~mask, -1e9)
        return q_values


class _SACMCTSAdapter:
    """Adapter exposing `get_policy_and_value` expected by MCTS."""

    def __init__(self, actor: SACActor, critic1: SACCritic, critic2: SACCritic):
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2

    @torch.no_grad()
    def get_policy_and_value(self, obs: dict):
        logits = self.actor(obs)
        probs = torch.softmax(logits, dim=-1)
        q1 = self.critic1(obs)
        q2 = self.critic2(obs)
        q = torch.minimum(q1, q2)
        # Conservative scalar value for MCTS leaf eval.
        value = torch.tanh(q.max(dim=-1).values / 10.0)
        return probs.squeeze(0), value.squeeze(0)


class SACTrainer:
    """Discrete Soft Actor-Critic with mask-aware losses."""

    def __init__(self, config: Config, device: str = "cpu"):
        self.config = config
        self.device = device

        self.actor = SACActor(config).to(device)
        self.critic1 = SACCritic(config).to(device)
        self.critic2 = SACCritic(config).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.sac_actor_lr)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.sac_critic_lr,
        )

        self.auto_alpha = config.sac_auto_entropy_tuning
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(max(config.sac_init_alpha, 1e-6)),
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            self.alpha_opt = optim.Adam([self.log_alpha], lr=config.sac_alpha_lr)
        else:
            self.log_alpha = None
            self.alpha_opt = None
        self.alpha_value = float(config.sac_init_alpha)

        self.env = CircuitGame(config)
        self.replay = StratifiedReplayBuffer(config.sac_replay_size)

        self.current_complexity = (
            config.starting_complexity if config.curriculum_enabled else config.max_complexity
        )
        self.success_history: List[bool] = []
        self.iter_success_rates: List[float] = []
        self.assist_mode = False
        self.assist_cooldown = 0
        self._boards = {}
        self._sympy_vars = create_variables(config.n_variables)
        self._base_target_keys = {poly.canonical_key() for poly in self.env._init_polys}
        self._very_easy_target_keys = self._build_reachable_target_keys(max_depth=1)
        # Reachability cache for "too easy" targets (<=2 sequential ops).
        self._easy_target_keys = self._build_reachable_target_keys(max_depth=2)

        self.mcts = None
        if config.sac_use_mcts_distillation or config.sac_stuck_detection_enabled:
            adapter = _SACMCTSAdapter(self.actor, self.critic1, self.critic2)
            self.mcts = MCTS(adapter, config, device)

    def _get_board(self, complexity: int):
        if complexity not in self._boards:
            self._boards[complexity] = build_game_board(self.config, complexity)
        return self._boards[complexity]

    def _obs_to_device(self, obs: dict) -> dict:
        result = {}
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                result[key] = val.to(self.device)
            elif isinstance(val, dict):
                result[key] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in val.items()
                }
            else:
                result[key] = val.to(self.device) if hasattr(val, "to") else val
        return result

    def _format_target_poly(self, poly) -> str:
        """Render FastPoly in a readable SymPy expression form."""
        return str(fast_to_sympy(poly, self._sympy_vars))

    def _build_reachable_target_keys(self, max_depth: int) -> set[bytes]:
        """Enumerate target keys reachable within a small sequential-op budget."""
        if max_depth <= 0:
            return set(self._base_target_keys)

        key_to_poly = {poly.canonical_key(): poly for poly in self.env._init_polys}
        init_state = frozenset(key_to_poly.keys())
        visited_states = {init_state}
        frontier = [init_state]
        reachable = set(init_state)

        for _ in range(max_depth):
            next_frontier = []
            for state in frontier:
                polys = [key_to_poly[key] for key in state]
                for i in range(len(polys)):
                    for j in range(i, len(polys)):
                        for op in (0, 1):
                            result = polys[i] + polys[j] if op == 0 else polys[i] * polys[j]
                            key = result.canonical_key()
                            if key not in key_to_poly:
                                key_to_poly[key] = result
                            reachable.add(key)

                            if key in state:
                                continue
                            next_state = state.union((key,))
                            if next_state not in visited_states:
                                visited_states.add(next_state)
                                next_frontier.append(next_state)

            if not next_frontier:
                break
            frontier = next_frontier

        return reachable

    def _is_easy_target(self, target_poly, complexity: int) -> bool:
        key = target_poly.canonical_key()
        # For C>=2, filter targets solvable in <=1 sequential op.
        if complexity >= 2 and key in self._very_easy_target_keys:
            return True
        # For C>=3, filter targets solvable in <=2 sequential ops.
        if complexity >= 3 and key in self._easy_target_keys:
            return True
        # For higher complexities, reject single-term monomials/constants.
        if complexity >= 4 and int(np.count_nonzero(target_poly.coeffs)) <= 1:
            return True
        return False

    def _sample_training_target(self, complexity: int, seen_keys: set[bytes]):
        """Sample a nontrivial, non-repeated target for the current iteration."""
        fallback = None
        for _ in range(64):
            target_poly, _ = generate_random_circuit(self.config, complexity)
            key = target_poly.canonical_key()
            if fallback is None:
                fallback = target_poly
            if key in seen_keys:
                continue
            if self._is_easy_target(target_poly, complexity):
                continue
            seen_keys.add(key)
            return target_poly

        # If filtering is too strict for a corner case, keep training moving.
        if fallback is None:
            fallback, _ = generate_random_circuit(self.config, complexity)
        seen_keys.add(fallback.canonical_key())
        return fallback

    @staticmethod
    def _masked_probs_and_log_probs(logits: torch.Tensor, mask: torch.Tensor):
        masked_logits = logits.masked_fill(~mask, -1e9)
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        probs = torch.exp(log_probs) * mask.float()
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        return probs, log_probs

    @torch.no_grad()
    def _sample_action(self, obs: dict) -> int:
        obs_dev = self._obs_to_device(obs)
        logits = self.actor(obs_dev)
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def _maybe_mcts_action(self, obs: dict) -> tuple[int, Optional[np.ndarray]]:
        if self.mcts is None:
            return self._sample_action(obs), None
        distill_prob = self._effective_distill_prob()
        if distill_prob <= 0.0 or random.random() >= distill_prob:
            return self._sample_action(obs), None

        action, probs = self.mcts.get_action_probs(self.env, temperature=1.0)
        return int(action), probs

    def _effective_distill_prob(self) -> float:
        base_prob = self.config.sac_mcts_distill_prob if self.config.sac_use_mcts_distillation else 0.0
        if self.assist_mode:
            return max(base_prob, self.config.sac_assist_distill_prob)
        return base_prob

    def _effective_distill_coef(self) -> float:
        base_coef = self.config.sac_distill_coef if self.config.sac_use_mcts_distillation else 0.0
        if self.assist_mode:
            return max(base_coef, self.config.sac_assist_distill_coef)
        return base_coef

    def _effective_stuck_sr_ceiling(self) -> float:
        # If SR plateaus below curriculum advance target, allow assist to kick in.
        adaptive = self.config.advance_threshold - 0.08
        ceiling = max(self.config.sac_stuck_sr_ceiling, adaptive)
        return min(max(ceiling, 0.0), 1.0)

    def _stuck_stats(self) -> tuple[bool, float, float]:
        if not self.config.sac_stuck_detection_enabled:
            return False, 0.0, 0.0

        window = max(2, self.config.sac_stuck_window)
        min_iters = max(window, self.config.sac_stuck_min_iters)
        if len(self.iter_success_rates) < min_iters:
            return False, 0.0, 0.0

        recent = self.iter_success_rates[-window:]
        slope = (recent[-1] - recent[0]) / max(window - 1, 1)
        mean_sr = float(np.mean(recent))
        stuck_ceiling = self._effective_stuck_sr_ceiling()

        is_stuck = (
            abs(slope) <= self.config.sac_stuck_slope_threshold
            and mean_sr <= stuck_ceiling
        )
        return is_stuck, float(slope), mean_sr

    def _update_assist_mode(self, iteration: int):
        if not self.config.sac_stuck_detection_enabled:
            return

        is_stuck, slope, mean_sr = self._stuck_stats()
        stuck_ceiling = self._effective_stuck_sr_ceiling()
        if not self.assist_mode and is_stuck:
            self.assist_mode = True
            self.assist_cooldown = self.config.sac_assist_cooldown_iters
            print(
                f"*** Assist Mode ON at iter {iteration} "
                f"(mean SR {100.0 * mean_sr:.1f}%, slope {100.0 * slope:+.2f} pp/iter) ***"
            )
            return

        if self.assist_mode:
            recovered = mean_sr >= (
                stuck_ceiling + self.config.sac_stuck_recovery_margin
            )
            if is_stuck and not recovered:
                self.assist_cooldown = self.config.sac_assist_cooldown_iters
                return

            self.assist_cooldown -= 1
            if self.assist_cooldown <= 0:
                self.assist_mode = False
                self.assist_cooldown = 0
                print(
                    f"*** Assist Mode OFF at iter {iteration} "
                    f"(mean SR {100.0 * mean_sr:.1f}%, slope {100.0 * slope:+.2f} pp/iter) ***"
                )

    def warmstart_replay(self):
        if self.config.sac_warmstart_episodes <= 0:
            return

        max_comp = max(1, min(self.current_complexity, self.config.max_complexity))
        for _ in range(self.config.sac_warmstart_episodes):
            complexity = random.randint(1, max_comp)
            target_poly, actions = generate_random_circuit(self.config, complexity)
            obs = self.env.reset(target_poly)

            for op, i, j in actions:
                action_idx = encode_action(op, i, j, self.config.max_nodes)
                next_obs, reward, done, info = self.env.step(action_idx)
                transition = Transition(
                    obs=obs,
                    action=action_idx,
                    reward=float(reward),
                    next_obs=next_obs,
                    done=bool(done),
                    complexity=complexity,
                    success=bool(info.get("is_success", False)),
                    demo=True,
                    mcts_policy=None,
                )
                self.replay.add(transition)
                obs = next_obs
                if done:
                    break

    def collect_experience(self):
        steps_collected = 0
        episodes_done = 0
        successes = 0
        total_reward = 0.0
        examples = []
        seen_targets = set()

        while steps_collected < self.config.sac_steps_per_update:
            # Use sequential circuit generation so curriculum complexity matches
            # the number of environment actions needed to construct targets.
            target_poly = self._sample_training_target(self.current_complexity, seen_targets)
            obs = self.env.reset(target_poly)
            episode_reward = 0.0
            episode_transitions = []

            while not self.env.done and steps_collected < self.config.sac_steps_per_update:
                action, mcts_policy = self._maybe_mcts_action(obs)
                next_obs, reward, done, info = self.env.step(action)

                episode_transitions.append(
                    Transition(
                        obs=obs,
                        action=action,
                        reward=float(reward),
                        next_obs=next_obs,
                        done=bool(done),
                        complexity=self.current_complexity,
                        success=False,  # filled at episode end
                        demo=False,
                        mcts_policy=mcts_policy,
                    )
                )

                episode_reward += reward
                steps_collected += 1
                obs = next_obs

            is_success = bool(info.get("is_success", False))
            for tr in episode_transitions:
                tr.success = is_success
                self.replay.add(tr)

            episodes_done += 1
            successes += int(is_success)
            total_reward += episode_reward
            self.success_history.append(is_success)

            if len(examples) < 5:
                examples.append(
                    {
                        "target": self._format_target_poly(target_poly),
                        "success": is_success,
                        "reward": float(episode_reward),
                        "steps": int(info.get("steps_taken", 0)),
                    }
                )

        return {
            "episodes": episodes_done,
            "success_rate": successes / max(episodes_done, 1),
            "avg_reward": total_reward / max(episodes_done, 1),
            "complexity": self.current_complexity,
            "replay_size": len(self.replay),
            "examples": examples,
        }

    def _compute_alpha(self):
        if self.auto_alpha and self.log_alpha is not None:
            return self.log_alpha.exp()
        return torch.tensor(self.alpha_value, dtype=torch.float32, device=self.device)

    def update(self):
        if len(self.replay) < self.config.sac_min_replay_size:
            return {
                "q_loss": 0.0,
                "policy_loss": 0.0,
                "alpha": float(self._compute_alpha().item()),
                "alpha_loss": 0.0,
                "mcts_ce_loss": 0.0,
                "bc_loss": 0.0,
            }

        q_losses = []
        policy_losses = []
        alpha_losses = []
        mcts_ce_losses = []
        bc_losses = []

        for _ in range(self.config.sac_updates_per_iter):
            batch = self.replay.sample(
                self.config.sac_batch_size,
                success_ratio=self.config.sac_success_sample_ratio,
            )
            if not batch:
                continue

            obs_batch = [self._obs_to_device(t.obs) for t in batch]
            next_obs_batch = [self._obs_to_device(t.next_obs) for t in batch]

            actions = torch.tensor([t.action for t in batch], device=self.device, dtype=torch.long)
            rewards = torch.tensor([t.reward for t in batch], device=self.device, dtype=torch.float32)
            dones = torch.tensor([t.done for t in batch], device=self.device, dtype=torch.float32)

            # Forward current states.
            logits = torch.cat([self.actor(obs) for obs in obs_batch], dim=0)
            q1 = torch.cat([self.critic1(obs) for obs in obs_batch], dim=0)
            q2 = torch.cat([self.critic2(obs) for obs in obs_batch], dim=0)
            masks = torch.cat([obs["mask"].unsqueeze(0) if obs["mask"].dim() == 1 else obs["mask"] for obs in obs_batch], dim=0)

            # Forward next states for target.
            with torch.no_grad():
                next_logits = torch.cat([self.actor(obs) for obs in next_obs_batch], dim=0)
                next_masks = torch.cat(
                    [
                        obs["mask"].unsqueeze(0) if obs["mask"].dim() == 1 else obs["mask"]
                        for obs in next_obs_batch
                    ],
                    dim=0,
                )
                next_q1_t = torch.cat([self.target_critic1(obs) for obs in next_obs_batch], dim=0)
                next_q2_t = torch.cat([self.target_critic2(obs) for obs in next_obs_batch], dim=0)
                next_q_t = torch.minimum(next_q1_t, next_q2_t)

                next_probs, next_log_probs = self._masked_probs_and_log_probs(next_logits, next_masks)
                alpha = self._compute_alpha()
                next_v = (next_probs * (next_q_t - alpha * next_log_probs)).sum(dim=-1)
                target_q = rewards + self.config.sac_gamma * (1.0 - dones) * next_v

            q1_a = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_a = q2.gather(1, actions.unsqueeze(1)).squeeze(1)
            q_loss = nn.functional.mse_loss(q1_a, target_q) + nn.functional.mse_loss(q2_a, target_q)

            self.critic_opt.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.critic1.parameters()) + list(self.critic2.parameters()),
                self.config.max_grad_norm,
            )
            self.critic_opt.step()

            probs, log_probs = self._masked_probs_and_log_probs(logits, masks)
            q1_pi = torch.cat([self.critic1(obs) for obs in obs_batch], dim=0)
            q2_pi = torch.cat([self.critic2(obs) for obs in obs_batch], dim=0)
            q_pi = torch.minimum(q1_pi, q2_pi)
            alpha = self._compute_alpha()
            policy_loss = (probs * (alpha * log_probs - q_pi)).sum(dim=-1).mean()

            # Optional MCTS policy distillation (soft target).
            distill_mask = []
            distill_targets = []
            for t in batch:
                if t.mcts_policy is None:
                    distill_mask.append(False)
                    distill_targets.append(np.zeros(self.config.max_actions, dtype=np.float32))
                else:
                    distill_mask.append(True)
                    distill_targets.append(t.mcts_policy.astype(np.float32))
            distill_mask_t = torch.tensor(distill_mask, device=self.device, dtype=torch.bool)
            if distill_mask_t.any():
                target_pi = torch.tensor(np.stack(distill_targets), device=self.device, dtype=torch.float32)
                # Avoid NaNs for any all-zero guidance vectors.
                target_pi = target_pi / (target_pi.sum(dim=-1, keepdim=True) + 1e-8)
                ce = -(target_pi[distill_mask_t] * log_probs[distill_mask_t]).sum(dim=-1).mean()
                mcts_ce_loss = ce
                distill_coef = self._effective_distill_coef()
                if distill_coef > 0.0:
                    policy_loss = policy_loss + distill_coef * ce
            else:
                mcts_ce_loss = torch.tensor(0.0, device=self.device)

            # Warm-start behavior cloning on demonstration transitions.
            demo_mask = torch.tensor([t.demo for t in batch], device=self.device, dtype=torch.bool)
            if demo_mask.any():
                chosen_logp = log_probs[torch.arange(log_probs.size(0), device=self.device), actions]
                bc = -chosen_logp[demo_mask].mean()
                bc_loss = bc
                policy_loss = policy_loss + self.config.sac_bc_coef * bc
            else:
                bc_loss = torch.tensor(0.0, device=self.device)

            self.actor_opt.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_opt.step()

            alpha_loss = torch.tensor(0.0, device=self.device)
            if self.auto_alpha and self.log_alpha is not None and self.alpha_opt is not None:
                with torch.no_grad():
                    entropy = -(probs * log_probs).sum(dim=-1)
                    valid_counts = masks.float().sum(dim=-1).clamp(min=2.0)
                    target_entropy = self.config.sac_target_entropy_ratio * torch.log(valid_counts)

                alpha_loss = (self.log_alpha.exp() * (target_entropy - entropy)).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()

            # Soft update target critics.
            self._soft_update(self.target_critic1, self.critic1, self.config.sac_tau)
            self._soft_update(self.target_critic2, self.critic2, self.config.sac_tau)

            q_losses.append(float(q_loss.item()))
            policy_losses.append(float(policy_loss.item()))
            alpha_losses.append(float(alpha_loss.item()))
            mcts_ce_losses.append(float(mcts_ce_loss.item()))
            bc_losses.append(float(bc_loss.item()))

        return {
            "q_loss": float(np.mean(q_losses)) if q_losses else 0.0,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "alpha": float(self._compute_alpha().item()),
            "alpha_loss": float(np.mean(alpha_losses)) if alpha_losses else 0.0,
            "mcts_ce_loss": float(np.mean(mcts_ce_losses)) if mcts_ce_losses else 0.0,
            "bc_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
        }

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

    def _maybe_advance_curriculum(self):
        if not self.config.curriculum_enabled:
            return

        window = max(50, self.config.sac_steps_per_update // max(self.config.max_steps, 1))
        if len(self.success_history) < window:
            return

        recent = self.success_history[-window:]
        rate = sum(recent) / len(recent)
        if rate >= self.config.advance_threshold and self.current_complexity < self.config.max_complexity:
            self.current_complexity += 1
            self.success_history.clear()
            print(f"[Curriculum] Advanced to complexity {self.current_complexity}")
        elif rate <= self.config.backoff_threshold and self.current_complexity > self.config.starting_complexity:
            self.current_complexity -= 1
            self.success_history.clear()
            print(f"[Curriculum] Backed off to complexity {self.current_complexity}")

    @torch.no_grad()
    def evaluate(self, num_trials: int = 100, complexities: Optional[List[int]] = None):
        if complexities is None:
            complexities = list(range(2, self.config.max_complexity + 1))

        self.actor.eval()
        env = CircuitGame(self.config)
        results = {}

        for complexity in complexities:
            successes = 0
            total_steps = 0
            for _ in range(num_trials):
                target_poly, _ = generate_random_circuit(self.config, complexity)
                obs = env.reset(target_poly)
                success = False
                info = {}
                while not env.done:
                    obs_dev = self._obs_to_device(obs)
                    logits = self.actor(obs_dev)
                    action = int(logits.argmax(dim=-1).item())
                    obs, _, _, info = env.step(action)
                    if info.get("is_success", False):
                        success = True
                if success:
                    successes += 1
                    total_steps += int(info.get("steps_taken", 0))

            success_rate = successes / max(num_trials, 1)
            avg_steps = total_steps / max(successes, 1)
            results[complexity] = {
                "success_rate": success_rate,
                "avg_steps": avg_steps,
                "num_trials": num_trials,
                "successes": successes,
            }

        total_successes = sum(v["successes"] for v in results.values())
        total_trials = sum(v["num_trials"] for v in results.values())
        results["overall"] = {
            "success_rate": total_successes / max(total_trials, 1),
            "total_successes": total_successes,
            "total_trials": total_trials,
        }
        return results

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu() if self.log_alpha is not None else None,
                "alpha_value": float(self.alpha_value),
                "config": self.config,
                "algorithm": "sac",
            },
            path,
        )

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.target_critic1.load_state_dict(state.get("target_critic1", state["critic1"]))
        self.target_critic2.load_state_dict(state.get("target_critic2", state["critic2"]))

        if self.auto_alpha and self.log_alpha is not None and state.get("log_alpha") is not None:
            loaded_log_alpha = state["log_alpha"]
            if isinstance(loaded_log_alpha, torch.Tensor):
                self.log_alpha.data.copy_(loaded_log_alpha.to(self.device))
        elif state.get("alpha_value") is not None:
            self.alpha_value = float(state["alpha_value"])

    def train(self, num_iterations: int):
        checkpoint_interval = max(0, int(self.config.sac_checkpoint_interval))
        checkpoint_dir = self.config.sac_checkpoint_dir
        if checkpoint_interval > 0 and checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.warmstart_replay()
        if self.config.sac_warmstart_episodes > 0:
            print(f"Prefilled replay buffer with {len(self.replay)} synthetic steps")

        for iteration in range(1, num_iterations + 1):
            rollout_info = self.collect_experience()
            self.iter_success_rates.append(float(rollout_info["success_rate"]))
            self._update_assist_mode(iteration)
            loss_info = self.update()
            self._maybe_advance_curriculum()

            if iteration % self.config.log_interval == 0:
                print(
                    f"Iter {iteration}: SR {100.0 * rollout_info['success_rate']:.1f}%, "
                    f"Q {loss_info['q_loss']:.4f}, "
                    f"Pi {loss_info['policy_loss']:.4f}, "
                    f"CE {loss_info['mcts_ce_loss']:.4f}, "
                    f"Buffer {rollout_info['replay_size']}"
                )
                for i, ex in enumerate(rollout_info["examples"]):
                    print(
                        f"  Ex {i + 1}: {'Success' if ex['success'] else 'Fail'} "
                        f"(R: {ex['reward']:.2f}, Steps: {ex['steps']}) Target: {ex['target']}"
                    )

            if checkpoint_interval > 0 and iteration % checkpoint_interval == 0 and checkpoint_dir:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"sac_iter_{iteration:05d}.pt",
                )
                self.save_checkpoint(checkpoint_path)
                print(f"[Checkpoint] Saved iteration checkpoint: {checkpoint_path}")
