"""Discrete SAC trainer with masked actions for circuit construction.

Implements Soft Actor-Critic for discrete action spaces (Christodoulou 2019)
with:
  - Masked action logits (invalid actions set to -inf)
  - Twin-Q critics for reduced overestimation bias
  - Stratified replay buffer (current-complexity / success / recent mix)
  - Adaptive entropy temperature tuning
  - Optional CQL-lite conservative regularisation
  - Optional behaviour-cloning warm start from board demonstrations
  - Optional factor library and subgoal rewards (see FactorLibrary)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..config import Config
from ..environment.action_space import encode_action
from ..environment.circuit_game import CircuitGame
from ..environment.fast_polynomial import FastPoly
from ..environment.factor_library import FactorLibrary
from ..evaluation.evaluate import evaluate_model
from ..game_board.generator import build_game_board, sample_target
from ..models.gnn_encoder import CircuitGNN


def masked_categorical_stats(
    logits: torch.Tensor, mask: torch.BoolTensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (log_probs, probs, entropy) for a masked categorical policy."""
    mask = mask.bool()
    masked_logits = logits.masked_fill(~mask, -1e9)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    probs = torch.exp(log_probs) * mask.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_probs = torch.log(probs.clamp_min(1e-8))
    entropy = -(probs * log_probs).sum(dim=-1)
    return log_probs, probs, entropy


def masked_logsumexp(values: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    """Masked log-sum-exp over the action dimension."""
    masked_values = values.masked_fill(~mask, -1e9)
    return torch.logsumexp(masked_values, dim=-1)


@dataclass
class RawStep:
    """Single environment step before n-step aggregation."""

    obs: dict
    action: int
    reward: float
    next_obs: dict
    done: bool


@dataclass
class Transition:
    """Replay transition with metadata for stratified sampling."""

    uid: int
    obs: dict
    action: int
    reward: float
    next_obs: dict
    done: float
    discount: float
    complexity: int
    episode_success: bool


def build_n_step_transition(
    raw_steps: Deque[RawStep], n_step: int, gamma: float
) -> Tuple[dict, int, float, dict, bool, float]:
    """Build an n-step transition from the left side of a deque."""
    first = raw_steps[0]
    reward_sum = 0.0
    steps_used = 0
    done = False
    next_obs = first.next_obs

    for step in raw_steps:
        reward_sum += (gamma ** steps_used) * step.reward
        steps_used += 1
        next_obs = step.next_obs
        if step.done or steps_used >= n_step:
            done = step.done
            break

    discount = gamma ** steps_used
    return first.obs, first.action, reward_sum, next_obs, done, discount


class StratifiedReplayBuffer:
    """Replay buffer with complexity/success/recent stratified sampling."""

    def __init__(self, capacity: int, recent_window: int):
        self.capacity = capacity
        self.storage: List[Optional[Transition]] = [None] * capacity
        self.size = 0
        self.pos = 0
        self.next_uid = 0

        self.success_indices = set()
        self.failure_indices = set()
        self.indices_by_complexity = defaultdict(set)
        self.recent_indices: Deque[Tuple[int, int]] = deque(maxlen=recent_window)

    def __len__(self) -> int:
        return self.size

    def _remove_index_metadata(self, idx: int):
        transition = self.storage[idx]
        if transition is None:
            return
        self.success_indices.discard(idx)
        self.failure_indices.discard(idx)
        self.indices_by_complexity[transition.complexity].discard(idx)
        if not self.indices_by_complexity[transition.complexity]:
            del self.indices_by_complexity[transition.complexity]

    def add(
        self,
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        done: bool,
        discount: float,
        complexity: int,
        episode_success: bool,
    ):
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            idx = self.pos
            self._remove_index_metadata(idx)
            self.pos = (self.pos + 1) % self.capacity

        transition = Transition(
            uid=self.next_uid,
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=float(done),
            discount=discount,
            complexity=complexity,
            episode_success=episode_success,
        )
        self.next_uid += 1
        self.storage[idx] = transition

        if episode_success:
            self.success_indices.add(idx)
        else:
            self.failure_indices.add(idx)
        self.indices_by_complexity[complexity].add(idx)
        self.recent_indices.append((idx, transition.uid))

    def _take_sample(
        self, candidates: List[int], n: int, selected: set
    ) -> List[int]:
        if n <= 0:
            return []
        available = [idx for idx in candidates if idx not in selected]
        if not available:
            return []
        if len(available) <= n:
            return available
        return random.sample(available, n)

    def _valid_recent_candidates(self) -> List[int]:
        seen = set()
        candidates = []
        for idx, uid in self.recent_indices:
            if idx in seen:
                continue
            transition = self.storage[idx]
            if transition is None:
                continue
            if transition.uid != uid:
                continue
            seen.add(idx)
            candidates.append(idx)
        return candidates

    def sample(
        self,
        batch_size: int,
        current_complexity: int,
        current_fraction: float,
        success_fraction: float,
        recent_fraction: float,
    ) -> List[Transition]:
        if self.size == 0:
            return []

        batch_size = min(batch_size, self.size)
        n_current = int(batch_size * current_fraction)
        n_success = int(batch_size * success_fraction)
        n_recent = int(batch_size * recent_fraction)

        selected = []
        selected_set = set()

        current_candidates = list(
            self.indices_by_complexity.get(current_complexity, set())
        )
        take = self._take_sample(current_candidates, n_current, selected_set)
        selected.extend(take)
        selected_set.update(take)

        success_candidates = list(self.success_indices)
        take = self._take_sample(success_candidates, n_success, selected_set)
        selected.extend(take)
        selected_set.update(take)

        recent_candidates = self._valid_recent_candidates()
        take = self._take_sample(recent_candidates, n_recent, selected_set)
        selected.extend(take)
        selected_set.update(take)

        remaining = batch_size - len(selected)
        all_candidates = list(range(self.size))
        take = self._take_sample(all_candidates, remaining, selected_set)
        selected.extend(take)
        selected_set.update(take)

        # If unique indices are insufficient, fill the remainder with replacement.
        while len(selected) < batch_size:
            selected.append(random.choice(all_candidates))

        return [self.storage[idx] for idx in selected if self.storage[idx] is not None]


class StateEncoder(nn.Module):
    """Shared state encoder: graph + target polynomial -> fused embedding."""

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

    def forward(self, obs: dict) -> torch.Tensor:
        graph = obs["graph"]
        target = obs["target"]

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
        target_emb = self.target_encoder(target)

        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)

        return self.fusion(torch.cat([graph_emb, target_emb], dim=-1))


class SACActor(nn.Module):
    """Discrete masked policy network."""

    def __init__(self, config: Config):
        super().__init__()
        self.encoder = StateEncoder(config)
        hidden = config.hidden_dim
        emb = config.embedding_dim
        self.policy_head = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.max_actions),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        emb = self.encoder(obs)
        logits = self.policy_head(emb)
        mask = obs["mask"]
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        return logits.masked_fill(~mask, float("-inf"))

    def forward_batch(self, obs_batch: List[dict]) -> torch.Tensor:
        logits = []
        for obs in obs_batch:
            logits.append(self.forward(obs).squeeze(0))
        return torch.stack(logits, dim=0)


class SACCritic(nn.Module):
    """Twin-Q critic for discrete action SAC."""

    def __init__(self, config: Config):
        super().__init__()
        self.encoder = StateEncoder(config)
        hidden = config.hidden_dim
        emb = config.embedding_dim
        self.q1_head = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.max_actions),
        )
        self.q2_head = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.max_actions),
        )

    def forward(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(obs)
        return self.q1_head(emb), self.q2_head(emb)

    def forward_batch(self, obs_batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        q1_all = []
        q2_all = []
        for obs in obs_batch:
            q1, q2 = self.forward(obs)
            q1_all.append(q1.squeeze(0))
            q2_all.append(q2.squeeze(0))
        return torch.stack(q1_all, dim=0), torch.stack(q2_all, dim=0)


def reconstruct_actions_from_board(
    config: Config, board: Dict[bytes, dict], target_key: bytes
) -> Optional[List[int]]:
    """Reconstruct one valid action sequence to build target_key from a board."""
    n_vars = config.n_variables
    max_deg = config.effective_max_degree
    mod = config.mod

    base_polys = [FastPoly.variable(i, n_vars, max_deg, mod) for i in range(n_vars)]
    base_polys.append(FastPoly.constant(1, n_vars, max_deg, mod))
    node_index = {poly.canonical_key(): idx for idx, poly in enumerate(base_polys)}
    actions: List[int] = []
    visiting = set()

    def choose_parent(entry: dict) -> Optional[dict]:
        if not entry.get("parents"):
            return None
        step = entry["step"]
        for parent in entry["parents"]:
            left = board.get(parent["left"])
            right = board.get(parent["right"])
            if left is None or right is None:
                continue
            if max(left["step"], right["step"]) == step - 1:
                return parent
        return entry["parents"][0]

    def ensure_node(key: bytes) -> Optional[int]:
        if key in node_index:
            return node_index[key]
        if key in visiting:
            return None
        entry = board.get(key)
        if entry is None:
            return None
        if entry["step"] == 0:
            return node_index.get(key)

        parent = choose_parent(entry)
        if parent is None:
            return None

        visiting.add(key)
        left_idx = ensure_node(parent["left"])
        right_idx = ensure_node(parent["right"])
        visiting.remove(key)

        if left_idx is None or right_idx is None:
            return None
        if len(node_index) >= config.max_nodes:
            return None

        op = 0 if parent["op"] == "add" else 1
        action_idx = encode_action(op, left_idx, right_idx, config.max_nodes)
        actions.append(action_idx)
        new_idx = len(node_index)
        node_index[key] = new_idx
        return new_idx

    result = ensure_node(target_key)
    if result is None:
        return None
    return actions


class SACTrainer:
    """Discrete SAC training loop with curriculum, stratified replay, and factor library.

    Orchestrates the experience-collection → critic-update → actor-update cycle
    for discrete masked-action SAC. An adaptive curriculum adjusts target complexity
    and a stratified replay buffer over-samples successful and recently-seen episodes.

    The optional FactorLibrary is created here (once per training run) and shared
    with the CircuitGame environment. It grows as the agent succeeds, providing
    increasingly rich subgoal guidance to the policy.

    Attributes:
        config (Config): Shared hyperparameter configuration.
        device (str): PyTorch device string.
        actor (SACActor): Policy network.
        critic (SACCritic): Twin-Q critic network.
        target_critic (SACCritic): Exponentially-averaged target critic.
        env (CircuitGame): Circuit construction environment.
        replay (StratifiedReplayBuffer): Off-policy experience replay buffer.
        factor_library (FactorLibrary | None): Session-level factor cache.
    """

    def __init__(self, config: Config, device: str = "cpu") -> None:
        """Initialise the SAC trainer.

        Constructs actor, critic, and target-critic networks; creates optimisers for
        each network and for the entropy temperature log_alpha; initialises the
        environment (with FactorLibrary if enabled) and the replay buffer.

        Args:
            config: Configuration dataclass with all hyperparameters.
            device: PyTorch device to use ('cpu', 'cuda', or 'mps').
        """
        self.config = config
        self.device = device

        self.actor = SACActor(config).to(device)
        self.critic = SACCritic(config).to(device)
        self.target_critic = SACCritic(config).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.sac_actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.sac_critic_lr
        )
        # log_alpha is a learnable scalar; alpha = exp(log_alpha) is the entropy temperature.
        self.log_alpha = torch.tensor(
            math.log(config.sac_alpha_init),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.sac_alpha_lr)

        # Create the factor library once per training session if the feature is enabled.
        # The library persists across all episodes within this run (in-memory only).
        factor_library: Optional[FactorLibrary] = None
        if config.factor_library_enabled:
            factor_library = FactorLibrary(
                mod=config.mod,
                n_vars=config.n_variables,
                max_degree=config.effective_max_degree,
            )

        self.env = CircuitGame(config, factor_library=factor_library)
        self.factor_library = factor_library

        self.replay = StratifiedReplayBuffer(
            capacity=config.sac_replay_size,
            recent_window=config.sac_recent_window,
        )

        self.current_complexity = (
            config.starting_complexity
            if config.curriculum_enabled
            else config.max_complexity
        )
        self.success_history: List[bool] = []
        self.total_env_steps = 0
        # Lazily-built BFS game boards per complexity level.
        self._boards = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(
            min=self.config.sac_alpha_min, max=self.config.sac_alpha_max
        )

    def _get_board(self, complexity: int):
        if complexity not in self._boards:
            self._boards[complexity] = build_game_board(self.config, complexity)
        return self._boards[complexity]

    def _clone_obs(self, obs: dict) -> dict:
        graph = obs["graph"]
        if isinstance(graph, dict):
            graph_clone = {
                k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in graph.items()
            }
        else:
            graph_clone = graph.clone()

        return {
            "graph": graph_clone,
            "target": obs["target"].detach().cpu().clone(),
            "mask": obs["mask"].detach().cpu().clone(),
        }

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
                if hasattr(val, "to"):
                    result[key] = val.to(self.device)
                else:
                    result[key] = val
        return result

    def _sample_random_valid_action(self, mask: torch.BoolTensor) -> int:
        valid_indices = torch.where(mask)[0]
        choice = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
        return int(choice)

    def _select_action(self, obs: dict, deterministic: bool = False) -> int:
        if self.total_env_steps < self.config.sac_initial_random_steps and not deterministic:
            return self._sample_random_valid_action(obs["mask"])

        obs_device = self._obs_to_device(obs)
        with torch.no_grad():
            logits = self.actor(obs_device)
            mask = obs_device["mask"]
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)

            if deterministic:
                return int(logits.argmax(dim=-1).item())

            _, probs, _ = masked_categorical_stats(logits, mask)
            dist = torch.distributions.Categorical(probs=probs)
            return int(dist.sample().item())

    def _collect_bc_dataset(self, num_samples: int) -> List[Tuple[dict, int]]:
        """Collect a state-action dataset from board-derived demonstrations."""
        dataset = []
        attempts = 0
        max_attempts = max(50, num_samples * 10)
        min_c = max(1, self.config.starting_complexity)
        max_c = max(min_c, self.config.max_complexity)

        while len(dataset) < num_samples and attempts < max_attempts:
            attempts += 1
            complexity = random.randint(min_c, max_c)
            board = self._get_board(complexity)
            candidates = [
                (key, entry)
                for key, entry in board.items()
                if entry["step"] == complexity and entry["step"] > 0
            ]
            if not candidates:
                continue

            target_key, target_entry = random.choice(candidates)
            actions = reconstruct_actions_from_board(self.config, board, target_key)
            if not actions:
                continue

            try:
                obs = self.env.reset(target_entry["poly"])
                for action in actions:
                    dataset.append((self._clone_obs(obs), action))
                    obs, _, done, _ = self.env.step(action)
                    if done or len(dataset) >= num_samples:
                        break
            except AssertionError:
                # Skip rare reconstructed traces that become invalid due to dedup effects.
                continue

        return dataset

    def _behavior_clone_warmstart(self):
        dataset = self._collect_bc_dataset(self.config.sac_bc_samples)
        if not dataset:
            print("[SAC BC] No demonstrations collected; skipping warm start.")
            return

        self.actor.train()
        for step in range(1, self.config.sac_bc_steps + 1):
            batch = random.sample(
                dataset, min(self.config.sac_bc_batch_size, len(dataset))
            )
            obs_batch = [self._obs_to_device(item[0]) for item in batch]
            action_targets = torch.tensor(
                [item[1] for item in batch], dtype=torch.long, device=self.device
            )

            logits = self.actor.forward_batch(obs_batch)
            loss = F.cross_entropy(logits, action_targets)

            self.actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()

            if step % max(1, self.config.sac_bc_steps // 5) == 0:
                print(
                    f"[SAC BC step {step}] loss={loss.item():.4f} demos={len(dataset)}"
                )

    def _collect_experience(self) -> dict:
        """Run the current actor in the environment and add transitions to the replay buffer.

        Collects at least sac_steps_per_iter environment steps across potentially
        multiple episodes. Each episode uses a freshly sampled target polynomial.
        Transitions are aggregated into n-step returns before being stored.

        Returns:
            Dict with episode statistics: 'episodes', 'success_rate', 'avg_reward',
            'complexity', 'replay_size', 'factor_hits', 'library_hits', 'library_size'.
        """
        self.actor.eval()
        steps_collected = 0
        episodes = 0
        successes = 0
        total_reward = 0.0
        factor_hits = 0   # Steps where a factor subgoal was hit this iteration.
        library_hits = 0  # Subset of factor_hits where the factor was library-known.

        while steps_collected < self.config.sac_steps_per_iter:
            board = self._get_board(self.current_complexity)
            target_poly, _ = sample_target(self.config, self.current_complexity, board)
            obs = self.env.reset(target_poly)

            episode_reward = 0.0
            raw_steps: Deque[RawStep] = deque()
            episode_nstep = []
            info = {"is_success": False}

            while not self.env.done and steps_collected < self.config.sac_steps_per_iter:
                action = self._select_action(obs, deterministic=False)
                next_obs, reward, done, info = self.env.step(action)
                self.total_env_steps += 1
                steps_collected += 1
                episode_reward += reward

                # Track factor subgoal statistics for logging.
                if info.get("factor_hit", False):
                    factor_hits += 1
                if info.get("library_hit", False):
                    library_hits += 1

                raw_steps.append(
                    RawStep(
                        obs=self._clone_obs(obs),
                        action=action,
                        reward=reward,
                        next_obs=self._clone_obs(next_obs),
                        done=done,
                    )
                )

                # Flush completed n-step windows as they fill.
                if len(raw_steps) >= self.config.sac_n_step:
                    episode_nstep.append(
                        build_n_step_transition(
                            raw_steps, self.config.sac_n_step, self.config.gamma
                        )
                    )
                    raw_steps.popleft()

                obs = next_obs

            # Drain any remaining partial n-step windows at episode end.
            while raw_steps:
                episode_nstep.append(
                    build_n_step_transition(
                        raw_steps, self.config.sac_n_step, self.config.gamma
                    )
                )
                raw_steps.popleft()

            episode_success = bool(info.get("is_success", False))
            for transition in episode_nstep:
                self.replay.add(
                    obs=transition[0],
                    action=transition[1],
                    reward=transition[2],
                    next_obs=transition[3],
                    done=transition[4],
                    discount=transition[5],
                    complexity=self.current_complexity,
                    episode_success=episode_success,
                )

            episodes += 1
            total_reward += episode_reward
            successes += int(episode_success)
            self.success_history.append(episode_success)

        return {
            "episodes": episodes,
            "success_rate": successes / max(episodes, 1),
            "avg_reward": total_reward / max(episodes, 1),
            "complexity": self.current_complexity,
            "replay_size": len(self.replay),
            "factor_hits": factor_hits,
            "library_hits": library_hits,
            "library_size": len(self.factor_library) if self.factor_library else 0,
        }

    def _soft_update_target(self):
        tau = self.config.sac_tau
        for target_param, source_param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    def _update(self) -> dict:
        if len(self.replay) < self.config.sac_min_replay_size:
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "alpha_loss": 0.0,
                "alpha": self.alpha.item(),
                "entropy": 0.0,
            }

        updates = max(
            1,
            int(
                self.config.sac_steps_per_iter
                * max(self.config.sac_update_to_data_ratio, 0.0)
                / max(self.config.sac_batch_size, 1)
            ),
        )

        critic_loss_total = 0.0
        actor_loss_total = 0.0
        alpha_loss_total = 0.0
        entropy_total = 0.0

        self.actor.train()
        self.critic.train()

        for _ in range(updates):
            batch = self.replay.sample(
                batch_size=self.config.sac_batch_size,
                current_complexity=self.current_complexity,
                current_fraction=self.config.sac_current_complexity_fraction,
                success_fraction=self.config.sac_success_fraction,
                recent_fraction=self.config.sac_recent_fraction,
            )
            if not batch:
                continue

            obs_batch = [self._obs_to_device(t.obs) for t in batch]
            next_obs_batch = [self._obs_to_device(t.next_obs) for t in batch]
            actions = torch.tensor(
                [t.action for t in batch], dtype=torch.long, device=self.device
            )
            rewards = torch.tensor(
                [t.reward for t in batch], dtype=torch.float32, device=self.device
            )
            dones = torch.tensor(
                [t.done for t in batch], dtype=torch.float32, device=self.device
            )
            discounts = torch.tensor(
                [t.discount for t in batch], dtype=torch.float32, device=self.device
            )

            masks = torch.stack([obs["mask"] for obs in obs_batch]).bool()
            next_masks = torch.stack([obs["mask"] for obs in next_obs_batch]).bool()

            with torch.no_grad():
                next_logits = self.actor.forward_batch(next_obs_batch)
                next_log_probs, next_probs, _ = masked_categorical_stats(
                    next_logits, next_masks
                )
                target_q1, target_q2 = self.target_critic.forward_batch(next_obs_batch)
                target_q = torch.min(target_q1, target_q2)
                alpha_detached = self.alpha.detach()
                next_v = (
                    next_probs * (target_q - alpha_detached * next_log_probs)
                ).sum(dim=-1)
                q_target = rewards + (1.0 - dones) * discounts * next_v

            q1, q2 = self.critic.forward_batch(obs_batch)
            q1_a = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_a = q2.gather(1, actions.unsqueeze(1)).squeeze(1)
            critic_loss = F.mse_loss(q1_a, q_target) + F.mse_loss(q2_a, q_target)

            if self.config.sac_use_cql and self.config.sac_cql_alpha > 0.0:
                cql1 = (masked_logsumexp(q1, masks) - q1_a).mean()
                cql2 = (masked_logsumexp(q2, masks) - q2_a).mean()
                critic_loss = critic_loss + self.config.sac_cql_alpha * (cql1 + cql2)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.critic_optimizer.step()

            logits = self.actor.forward_batch(obs_batch)
            log_probs, probs, entropy = masked_categorical_stats(logits, masks)
            with torch.no_grad():
                q1_pi, q2_pi = self.critic.forward_batch(obs_batch)
                q_min = torch.min(q1_pi, q2_pi)

            actor_loss = (probs * (self.alpha.detach() * log_probs - q_min)).sum(
                dim=-1
            ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()

            valid_counts = masks.float().sum(dim=-1).clamp_min(2.0)
            target_entropy = -self.config.sac_target_entropy_scale * torch.log(valid_counts)
            alpha_loss = (
                self.alpha * (entropy.detach() - target_entropy)
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            min_log_alpha = math.log(self.config.sac_alpha_min)
            max_log_alpha = math.log(self.config.sac_alpha_max)
            with torch.no_grad():
                self.log_alpha.clamp_(min=min_log_alpha, max=max_log_alpha)

            self._soft_update_target()

            critic_loss_total += critic_loss.item()
            actor_loss_total += actor_loss.item()
            alpha_loss_total += alpha_loss.item()
            entropy_total += entropy.mean().item()

        denom = max(updates, 1)
        return {
            "critic_loss": critic_loss_total / denom,
            "actor_loss": actor_loss_total / denom,
            "alpha_loss": alpha_loss_total / denom,
            "alpha": self.alpha.item(),
            "entropy": entropy_total / denom,
        }

    def _maybe_advance_curriculum(self):
        if not self.config.curriculum_enabled:
            return
        window = self.config.sac_curriculum_window
        if len(self.success_history) < window:
            return

        recent = self.success_history[-window:]
        rate = sum(recent) / len(recent)

        if (
            rate >= self.config.advance_threshold
            and self.current_complexity < self.config.max_complexity
        ):
            self.current_complexity += 1
            self.success_history.clear()
            print(f"[Curriculum] Advanced to complexity {self.current_complexity}")
        elif (
            rate <= self.config.backoff_threshold
            and self.current_complexity > self.config.starting_complexity
        ):
            self.current_complexity -= 1
            self.success_history.clear()
            print(f"[Curriculum] Backed off to complexity {self.current_complexity}")

    def _get_fixed_phase_complexity(self, iteration: int) -> Optional[int]:
        if self.config.sac_fixed_complexity_iters <= 0:
            return None

        fixed = [
            c
            for c in self.config.sac_fixed_complexities
            if self.config.starting_complexity <= c <= self.config.max_complexity
        ]
        if not fixed:
            return None

        total_fixed_iters = len(fixed) * self.config.sac_fixed_complexity_iters
        if iteration > total_fixed_iters:
            return None

        phase = (iteration - 1) // self.config.sac_fixed_complexity_iters
        return fixed[phase]

    def train(self, num_iterations: int):
        if self.config.sac_bc_warmstart_enabled:
            print("[SAC] Running BC warm start...")
            self._behavior_clone_warmstart()

        for iteration in range(1, num_iterations + 1):
            fixed_complexity = self._get_fixed_phase_complexity(iteration)
            in_fixed_phase = fixed_complexity is not None
            if in_fixed_phase:
                self.current_complexity = fixed_complexity

            rollout_info = self._collect_experience()
            loss_info = self._update()

            if not in_fixed_phase:
                self._maybe_advance_curriculum()

            if iteration % self.config.log_interval == 0:
                phase = "fixed" if in_fixed_phase else "curriculum"
                lib_str = (
                    f"lib={rollout_info['library_size']} "
                    f"fhits={rollout_info['factor_hits']} "
                    f"lhits={rollout_info['library_hits']} "
                    if self.config.factor_library_enabled else ""
                )
                print(
                    f"[SAC iter {iteration}] "
                    f"phase={phase} "
                    f"complexity={rollout_info['complexity']} "
                    f"success={rollout_info['success_rate']:.2%} "
                    f"reward={rollout_info['avg_reward']:.3f} "
                    f"buffer={rollout_info['replay_size']} "
                    f"{lib_str}"
                    f"critic={loss_info['critic_loss']:.4f} "
                    f"actor={loss_info['actor_loss']:.4f} "
                    f"alpha={loss_info['alpha']:.4f} "
                    f"entropy={loss_info['entropy']:.4f}"
                )

    def evaluate(
        self,
        complexities: Optional[List[int]] = None,
        num_trials: int = 100,
        verbose: bool = True,
    ) -> Dict:
        return evaluate_model(
            self.actor,
            self.config,
            algorithm="sac",
            complexities=complexities,
            num_trials=num_trials,
            device=self.device,
            verbose=verbose,
        )

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "current_complexity": self.current_complexity,
                "total_env_steps": self.total_env_steps,
                "config": self.config,
                "algorithm": "sac",
            },
            path,
        )

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])

        if "log_alpha" in state:
            self.log_alpha = (
                state["log_alpha"].to(self.device).detach().requires_grad_(True)
            )
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.config.sac_alpha_lr
            )

        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "alpha_optimizer" in state:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])

        self.current_complexity = state.get("current_complexity", self.current_complexity)
        self.total_env_steps = state.get("total_env_steps", self.total_env_steps)

