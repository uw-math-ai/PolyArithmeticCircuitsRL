"""Discrete-action SAC trainer for top-down polynomial decomposition.

This is the *breakdown* counterpart to ``src/algorithms/sac.py``. It trains
a discrete masked-action SAC agent on the :class:`BreakdownGame`
environment defined in ``breakdown_env``, using the same FactorLibrary-based
guided splits as the PPO+MCTS breakdown trainer.

Architecture
------------
* :class:`BreakdownStateEncoder` — shared MLP encoder fusing the focus
  polynomial, original target, and context features into a single
  embedding.
* :class:`BreakdownSACActor` — masked categorical policy over the
  ``max_options`` decomposition slots.
* :class:`BreakdownSACCritic` — twin-Q critic; each Q-head outputs
  ``max_options`` action-values from the same encoder.
* :class:`SACBreakdownTrainer` — orchestrates the
  collect → critic-update → actor-update → soft-target loop with adaptive
  entropy temperature and a stratified replay buffer.

The trainer reuses many of the existing :class:`Config` knobs
(``sac_*``, ``gamma``, ``max_grad_norm``, curriculum thresholds, etc.) so
it can be plugged into existing experiment scripts with minimal changes,
but it does **not** mutate any existing module — all new code lives here.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ...config import Config
from ...environment.factor_library import FactorLibrary
from ...environment.fast_polynomial import FastPoly
from ...game_board.generator import (
    build_game_board,
    generate_random_circuit,
    sample_target,
)

from .breakdown_env import (
    CANDIDATE_FEATURE_DIM,
    CONTEXT_FEATURE_DIM,
    BreakdownGame,
    BreakdownObservation,
    observation_to_tensors,
    stack_observations,
)


# ----------------------------------------------------------------------
# Network components
# ----------------------------------------------------------------------


class BreakdownStateEncoder(nn.Module):
    """Shared encoder mapping a breakdown observation to a state embedding.

    Three streams (focus polynomial, original target, scalar context) are
    each passed through a small MLP and fused with one more MLP. The
    embedding is then broadcast across the ``max_options`` action slots
    by downstream heads.
    """

    def __init__(
        self,
        target_size: int,
        context_dim: int,
        hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.focus_encoder = nn.Sequential(
            nn.Linear(target_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(target_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        focus = obs["focus"]
        target = obs["target"]
        context = obs["context"]
        if focus.dim() == 1:
            focus = focus.unsqueeze(0)
            target = target.unsqueeze(0)
            context = context.unsqueeze(0)
        f = self.focus_encoder(focus)
        t = self.target_encoder(target)
        c = self.context_encoder(context)
        return self.fusion(torch.cat([f, t, c], dim=-1))


class BreakdownSACActor(nn.Module):
    """Masked categorical policy for discrete-action SAC on the breakdown env.

    Produces ``K`` logits — one per candidate slot — by running a small
    per-candidate scoring MLP over (state_embedding, candidate_features)
    pairs. Invalid (mask=False) slots are set to ``-inf`` so that
    ``log_softmax`` and sampling work directly.
    """

    def __init__(
        self,
        target_size: int,
        context_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = BreakdownStateEncoder(
            target_size=target_size,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim + candidate_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        emb = self.encoder(obs)
        cand_feats = obs["cand_feats"]
        mask = obs["mask"]
        if cand_feats.dim() == 2:
            cand_feats = cand_feats.unsqueeze(0)
            mask = mask.unsqueeze(0)
        b, k, _ = cand_feats.shape
        emb_rep = emb.unsqueeze(1).expand(b, k, emb.size(-1))
        merged = torch.cat([emb_rep, cand_feats], dim=-1)
        logits = self.scorer(merged).squeeze(-1)
        return logits.masked_fill(~mask.bool(), float("-inf"))

    def forward_batch(self, obs_batch: List[dict]) -> torch.Tensor:
        """Forward each obs-dict separately and stack the logits.

        Used when the replay batch is heterogeneous and we need per-sample
        forward passes (the breakdown observation tensors are already
        stackable, so callers can prefer ``forward(stack_observations(...))``
        when possible).
        """
        rows = []
        for obs in obs_batch:
            rows.append(self.forward(obs).squeeze(0))
        return torch.stack(rows, dim=0)


class BreakdownSACCritic(nn.Module):
    """Twin-Q critic with per-candidate action values."""

    def __init__(
        self,
        target_size: int,
        context_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = BreakdownStateEncoder(
            target_size=target_size,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
        self.q1_head = nn.Sequential(
            nn.Linear(embedding_dim + candidate_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2_head = nn.Sequential(
            nn.Linear(embedding_dim + candidate_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _score_with(self, head: nn.Module, emb: torch.Tensor, cand_feats: torch.Tensor) -> torch.Tensor:
        b, k, _ = cand_feats.shape
        emb_rep = emb.unsqueeze(1).expand(b, k, emb.size(-1))
        merged = torch.cat([emb_rep, cand_feats], dim=-1)
        return head(merged).squeeze(-1)

    def forward(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(obs)
        cand_feats = obs["cand_feats"]
        if cand_feats.dim() == 2:
            cand_feats = cand_feats.unsqueeze(0)
        return (
            self._score_with(self.q1_head, emb, cand_feats),
            self._score_with(self.q2_head, emb, cand_feats),
        )

    def forward_batch(self, obs_batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        q1s = []
        q2s = []
        for obs in obs_batch:
            q1, q2 = self.forward(obs)
            q1s.append(q1.squeeze(0))
            q2s.append(q2.squeeze(0))
        return torch.stack(q1s, dim=0), torch.stack(q2s, dim=0)


# ----------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------


@dataclass
class _BTransition:
    """Single replay transition for the breakdown SAC agent.

    Observations are stored as :class:`BreakdownObservation` (numpy) so the
    buffer remains light. Conversion to torch tensors happens at sample
    time on the training device.
    """

    uid: int
    obs: BreakdownObservation
    action: int
    reward: float
    next_obs: BreakdownObservation
    done: float
    discount: float
    complexity: int
    episode_success: bool


@dataclass
class _RawStep:
    """Single env step held in a deque while we form n-step transitions."""

    obs: BreakdownObservation
    action: int
    reward: float
    next_obs: BreakdownObservation
    done: bool


def _build_n_step(raw: Deque[_RawStep], n_step: int, gamma: float):
    """Mirror of :func:`sac.build_n_step_transition` for breakdown obs."""
    first = raw[0]
    reward_sum = 0.0
    used = 0
    done = False
    next_obs = first.next_obs
    for st in raw:
        reward_sum += (gamma ** used) * st.reward
        used += 1
        next_obs = st.next_obs
        if st.done or used >= n_step:
            done = st.done
            break
    discount = gamma ** used
    return first.obs, first.action, reward_sum, next_obs, done, discount


class _StratifiedReplay:
    """Same idea as ``StratifiedReplayBuffer`` in ``sac.py`` but typed for breakdown.

    Implements complexity / success / recent stratified sampling in the
    exact same proportions as the forward SAC trainer. Kept as a private
    helper here so the breakdown module is fully self-contained.
    """

    def __init__(self, capacity: int, recent_window: int) -> None:
        self.capacity = capacity
        self.storage: List[Optional[_BTransition]] = [None] * capacity
        self.size = 0
        self.pos = 0
        self.next_uid = 0
        self.success_indices: set = set()
        self.failure_indices: set = set()
        self.indices_by_complexity: Dict[int, set] = defaultdict(set)
        self.recent_indices: Deque[Tuple[int, int]] = deque(maxlen=recent_window)

    def __len__(self) -> int:
        return self.size

    def _remove_meta(self, idx: int) -> None:
        tr = self.storage[idx]
        if tr is None:
            return
        self.success_indices.discard(idx)
        self.failure_indices.discard(idx)
        bucket = self.indices_by_complexity[tr.complexity]
        bucket.discard(idx)
        if not bucket:
            del self.indices_by_complexity[tr.complexity]

    def add(
        self,
        obs: BreakdownObservation,
        action: int,
        reward: float,
        next_obs: BreakdownObservation,
        done: bool,
        discount: float,
        complexity: int,
        episode_success: bool,
    ) -> None:
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            idx = self.pos
            self._remove_meta(idx)
            self.pos = (self.pos + 1) % self.capacity

        tr = _BTransition(
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
        self.storage[idx] = tr
        if episode_success:
            self.success_indices.add(idx)
        else:
            self.failure_indices.add(idx)
        self.indices_by_complexity[complexity].add(idx)
        self.recent_indices.append((idx, tr.uid))

    def _take(self, candidates: List[int], n: int, selected: set) -> List[int]:
        if n <= 0:
            return []
        avail = [i for i in candidates if i not in selected]
        if not avail:
            return []
        if len(avail) <= n:
            return avail
        return random.sample(avail, n)

    def _valid_recent(self) -> List[int]:
        seen: set = set()
        out: List[int] = []
        for idx, uid in self.recent_indices:
            if idx in seen:
                continue
            tr = self.storage[idx]
            if tr is None or tr.uid != uid:
                continue
            seen.add(idx)
            out.append(idx)
        return out

    def sample(
        self,
        batch_size: int,
        current_complexity: int,
        current_fraction: float,
        success_fraction: float,
        recent_fraction: float,
    ) -> List[_BTransition]:
        if self.size == 0:
            return []
        batch_size = min(batch_size, self.size)
        n_cur = int(batch_size * current_fraction)
        n_suc = int(batch_size * success_fraction)
        n_rec = int(batch_size * recent_fraction)

        chosen: List[int] = []
        chosen_set: set = set()

        cur_cands = list(self.indices_by_complexity.get(current_complexity, set()))
        take = self._take(cur_cands, n_cur, chosen_set)
        chosen.extend(take)
        chosen_set.update(take)

        take = self._take(list(self.success_indices), n_suc, chosen_set)
        chosen.extend(take)
        chosen_set.update(take)

        take = self._take(self._valid_recent(), n_rec, chosen_set)
        chosen.extend(take)
        chosen_set.update(take)

        all_idx = list(range(self.size))
        take = self._take(all_idx, batch_size - len(chosen), chosen_set)
        chosen.extend(take)
        chosen_set.update(take)

        while len(chosen) < batch_size:
            chosen.append(random.choice(all_idx))

        return [self.storage[i] for i in chosen if self.storage[i] is not None]


# ----------------------------------------------------------------------
# Masked categorical helpers (mirrors ``sac.masked_categorical_stats``)
# ----------------------------------------------------------------------


def _masked_categorical_stats(
    logits: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Numerically-stable masked categorical statistics.

    Identical in spirit to ``sac.masked_categorical_stats`` but kept local
    so that this module does not depend on the existing SAC trainer.

    Returns:
        ``(log_probs, probs, entropy)`` — all with the action dimension
        last.
    """
    mask = mask.bool()
    masked_logits = logits.masked_fill(~mask, -1e9)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    probs = torch.exp(log_probs) * mask.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_probs = torch.log(probs.clamp_min(1e-8))
    entropy = -(probs * log_probs).sum(dim=-1)
    return log_probs, probs, entropy


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------


class SACBreakdownTrainer:
    """Discrete-action SAC trainer for the polynomial breakdown task.

    Mirrors the structure of ``src/algorithms/sac.SACTrainer`` (collection
    via stochastic actor → n-step transitions → stratified replay → twin-Q
    critic update → masked-actor update → adaptive entropy temperature →
    soft target update) but operates on :class:`BreakdownGame`.

    The trainer instantiates its own :class:`FactorLibrary`. Existing
    PPO/SAC trainers are left untouched.

    Attributes:
        config (Config): Shared hyperparameters.
        device (str): PyTorch device.
        actor (BreakdownSACActor): Policy network.
        critic (BreakdownSACCritic): Twin-Q critic.
        target_critic (BreakdownSACCritic): EMA target critic.
        env (BreakdownGame): Decomposition environment.
        replay (_StratifiedReplay): Off-policy replay buffer.
        factor_library (FactorLibrary): Library used for guided splits and
            cross-episode subgoal rewards.
    """

    MAX_BOARD_COMPLEXITY = 4

    def __init__(
        self,
        config: Config,
        device: str = "cpu",
        max_options: int = 32,
        max_breakdown_steps: int = 16,
        size_penalty_per_node: float = 0.05,
        log_path: Optional[str] = None,
    ) -> None:
        """Instantiate the trainer with fresh networks, buffer, and env.

        Args:
            config: Shared :class:`Config`. Read-only; any breakdown-only
                knobs are passed explicitly.
            device: PyTorch device.
            max_options: Width of the candidate action space at each step.
            max_breakdown_steps: Hard step limit per decomposition episode.
            size_penalty_per_node: Soft penalty applied at episode end for
                trees larger than ``config.max_complexity``.
            log_path: Optional file for textual logs.
        """
        self.config = config
        self.device = device
        self.max_options = max_options
        self.max_breakdown_steps = max_breakdown_steps
        self.log_path = log_path

        self.actor = BreakdownSACActor(
            target_size=config.target_size,
            context_dim=CONTEXT_FEATURE_DIM,
            candidate_feature_dim=CANDIDATE_FEATURE_DIM,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
        ).to(device)
        self.critic = BreakdownSACCritic(
            target_size=config.target_size,
            context_dim=CONTEXT_FEATURE_DIM,
            candidate_feature_dim=CANDIDATE_FEATURE_DIM,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
        ).to(device)
        self.target_critic = BreakdownSACCritic(
            target_size=config.target_size,
            context_dim=CONTEXT_FEATURE_DIM,
            candidate_feature_dim=CANDIDATE_FEATURE_DIM,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.sac_actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.sac_critic_lr
        )
        # log_alpha → alpha = exp(log_alpha) is the entropy temperature.
        self.log_alpha = torch.tensor(
            math.log(config.sac_alpha_init),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.sac_alpha_lr)

        self.factor_library = FactorLibrary(
            mod=config.mod,
            n_vars=config.n_variables,
            max_degree=config.effective_max_degree,
        )
        self.env = BreakdownGame(
            config=config,
            factor_library=self.factor_library,
            max_options=max_options,
            max_steps=max_breakdown_steps,
            size_penalty_per_node=size_penalty_per_node,
        )

        self.replay = _StratifiedReplay(
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
        self._boards: Dict[int, dict] = {}

    @property
    def alpha(self) -> torch.Tensor:
        """Clamped current entropy temperature (matches forward SAC)."""
        return self.log_alpha.exp().clamp(
            min=self.config.sac_alpha_min, max=self.config.sac_alpha_max
        )

    # ------------------------------------------------------------------
    # Logging / target sampling helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        print(msg)
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(msg + "\n")

    def _get_board(self, complexity: int) -> dict:
        if complexity not in self._boards:
            self._boards[complexity] = build_game_board(self.config, complexity)
        return self._boards[complexity]

    def _sample_target(self, complexity: int) -> FastPoly:
        if complexity <= self.MAX_BOARD_COMPLEXITY:
            board = self._get_board(complexity)
            poly, _ = sample_target(self.config, complexity, board)
            return poly
        poly, _ = generate_random_circuit(self.config, complexity)
        return poly

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _sample_random_valid_action(self, mask: np.ndarray) -> int:
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return 0
        return int(np.random.choice(valid))

    def _select_action(
        self, obs: BreakdownObservation, deterministic: bool = False
    ) -> int:
        if (
            self.total_env_steps < self.config.sac_initial_random_steps
            and not deterministic
        ):
            return self._sample_random_valid_action(obs.mask)

        obs_t = observation_to_tensors(obs, device=self.device)
        with torch.no_grad():
            logits = self.actor(obs_t)
            mask = obs_t["mask"]
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if deterministic:
                return int(logits.argmax(dim=-1).item())
            _, probs, _ = _masked_categorical_stats(logits, mask)
            dist = torch.distributions.Categorical(probs=probs)
            return int(dist.sample().item())

    # ------------------------------------------------------------------
    # Experience collection
    # ------------------------------------------------------------------

    def _collect_experience(self) -> dict:
        """Populate the replay with on-policy transitions for one iteration."""
        self.actor.eval()
        steps_collected = 0
        episodes = 0
        successes = 0
        total_reward = 0.0
        factor_hits = 0
        library_hits = 0
        skipped_trivial = 0

        while steps_collected < self.config.sac_steps_per_iter:
            target = self._sample_target(self.current_complexity)
            obs = self.env.reset(target)

            # Trivial / immediately-successful targets — count and skip.
            if self.env.done:
                episodes += 1
                successes += 1
                self.success_history.append(True)
                skipped_trivial += 1
                continue

            episode_reward = 0.0
            raw_steps: Deque[_RawStep] = deque()
            episode_n: List[Tuple] = []
            info: dict = {"is_success": False}

            while (
                not self.env.done
                and steps_collected < self.config.sac_steps_per_iter
            ):
                action = self._select_action(obs, deterministic=False)
                next_obs, reward, done, info = self.env.step(action)
                self.total_env_steps += 1
                steps_collected += 1
                episode_reward += reward

                if info.get("factor_hit", False):
                    factor_hits += 1
                if info.get("library_hit", False):
                    library_hits += 1

                raw_steps.append(
                    _RawStep(
                        obs=obs,
                        action=action,
                        reward=reward,
                        next_obs=next_obs,
                        done=done,
                    )
                )
                if len(raw_steps) >= self.config.sac_n_step:
                    episode_n.append(
                        _build_n_step(
                            raw_steps,
                            self.config.sac_n_step,
                            self.config.gamma,
                        )
                    )
                    raw_steps.popleft()

                obs = next_obs

            while raw_steps:
                episode_n.append(
                    _build_n_step(
                        raw_steps, self.config.sac_n_step, self.config.gamma
                    )
                )
                raw_steps.popleft()

            episode_success = bool(info.get("is_success", False))
            for tr in episode_n:
                self.replay.add(
                    obs=tr[0],
                    action=tr[1],
                    reward=tr[2],
                    next_obs=tr[3],
                    done=tr[4],
                    discount=tr[5],
                    complexity=self.current_complexity,
                    episode_success=episode_success,
                )

            episodes += 1
            total_reward += episode_reward
            successes += int(episode_success)
            self.success_history.append(episode_success)

            if episode_success:
                self.env.register_decomposition_in_library()

        return {
            "episodes": episodes,
            "success_rate": successes / max(episodes, 1),
            "avg_reward": total_reward / max(episodes, 1),
            "complexity": self.current_complexity,
            "replay_size": len(self.replay),
            "factor_hits": factor_hits,
            "library_hits": library_hits,
            "library_size": len(self.factor_library),
            "trivial_skipped": skipped_trivial,
        }

    # ------------------------------------------------------------------
    # Optimisation step
    # ------------------------------------------------------------------

    def _soft_update_target(self) -> None:
        tau = self.config.sac_tau
        for tp, sp in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)

    def _update(self) -> dict:
        if len(self.replay) < self.config.sac_min_replay_size:
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "alpha_loss": 0.0,
                "alpha": float(self.alpha.item()),
                "entropy": 0.0,
            }

        n_updates = max(
            1,
            int(
                self.config.sac_steps_per_iter
                * max(self.config.sac_update_to_data_ratio, 0.0)
                / max(self.config.sac_batch_size, 1)
            ),
        )

        crit_total = 0.0
        actor_total = 0.0
        alpha_total = 0.0
        ent_total = 0.0

        self.actor.train()
        self.critic.train()

        for _ in range(n_updates):
            batch = self.replay.sample(
                batch_size=self.config.sac_batch_size,
                current_complexity=self.current_complexity,
                current_fraction=self.config.sac_current_complexity_fraction,
                success_fraction=self.config.sac_success_fraction,
                recent_fraction=self.config.sac_recent_fraction,
            )
            if not batch:
                continue

            obs_t = stack_observations(
                [t.obs for t in batch], device=self.device
            )
            next_obs_t = stack_observations(
                [t.next_obs for t in batch], device=self.device
            )
            actions = torch.tensor(
                [t.action for t in batch],
                dtype=torch.long,
                device=self.device,
            )
            rewards = torch.tensor(
                [t.reward for t in batch],
                dtype=torch.float32,
                device=self.device,
            )
            dones = torch.tensor(
                [t.done for t in batch],
                dtype=torch.float32,
                device=self.device,
            )
            discounts = torch.tensor(
                [t.discount for t in batch],
                dtype=torch.float32,
                device=self.device,
            )

            mask = obs_t["mask"].bool()
            next_mask = next_obs_t["mask"].bool()

            # --- Critic target ---
            with torch.no_grad():
                next_logits = self.actor(next_obs_t)
                next_lp, next_probs, _ = _masked_categorical_stats(
                    next_logits, next_mask
                )
                tq1, tq2 = self.target_critic(next_obs_t)
                tq = torch.min(tq1, tq2)
                alpha_d = self.alpha.detach()
                next_v = (
                    next_probs * (tq - alpha_d * next_lp)
                ).sum(dim=-1)
                # Episodes that ended via timeout/failure carry valid masks
                # in next_obs only when ``done=False``; for done=True we
                # already zero out the bootstrap.
                valid_next = next_mask.any(dim=-1).float()
                next_v = next_v * valid_next
                q_target = rewards + (1.0 - dones) * discounts * next_v

            # --- Critic update ---
            q1, q2 = self.critic(obs_t)
            q1_a = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_a = q2.gather(1, actions.unsqueeze(1)).squeeze(1)
            critic_loss = F.mse_loss(q1_a, q_target) + F.mse_loss(q2_a, q_target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.config.max_grad_norm
            )
            self.critic_optimizer.step()

            # --- Actor update ---
            logits = self.actor(obs_t)
            log_probs, probs, entropy = _masked_categorical_stats(logits, mask)
            with torch.no_grad():
                q1_pi, q2_pi = self.critic(obs_t)
                q_min = torch.min(q1_pi, q2_pi)
            actor_loss = (
                probs * (self.alpha.detach() * log_probs - q_min)
            ).sum(dim=-1).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.max_grad_norm
            )
            self.actor_optimizer.step()

            # --- Adaptive entropy temperature ---
            valid_counts = mask.float().sum(dim=-1).clamp_min(2.0)
            target_entropy = -self.config.sac_target_entropy_scale * torch.log(
                valid_counts
            )
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

            crit_total += critic_loss.item()
            actor_total += actor_loss.item()
            alpha_total += alpha_loss.item()
            ent_total += entropy.mean().item()

        denom = max(n_updates, 1)
        return {
            "critic_loss": crit_total / denom,
            "actor_loss": actor_total / denom,
            "alpha_loss": alpha_total / denom,
            "alpha": float(self.alpha.item()),
            "entropy": ent_total / denom,
        }

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _maybe_advance_curriculum(self) -> None:
        if not self.config.curriculum_enabled:
            return
        window = self.config.sac_curriculum_window
        if len(self.success_history) < window:
            return
        rate = sum(self.success_history[-window:]) / window
        if (
            rate >= self.config.advance_threshold
            and self.current_complexity < self.config.max_complexity
        ):
            self.current_complexity += 1
            self.success_history.clear()
            self._log(
                f"[Curriculum] Advanced to complexity {self.current_complexity}"
            )
        elif (
            rate <= self.config.backoff_threshold
            and self.current_complexity > self.config.starting_complexity
        ):
            self.current_complexity -= 1
            self.success_history.clear()
            self._log(
                f"[Curriculum] Backed off to complexity {self.current_complexity}"
            )

    # ------------------------------------------------------------------
    # Public training / evaluation entry-points
    # ------------------------------------------------------------------

    def train(self, num_iterations: int) -> dict:
        """Run ``num_iterations`` SAC training cycles.

        Returns:
            History dict with the same keys the forward SAC trainer emits
            (``pg_loss`` is mapped to actor loss, ``vf_loss`` to critic
            loss) so plotting code remains compatible.
        """
        history = {
            "pg_loss": [],
            "vf_loss": [],
            "entropy": [],
            "success_rate": [],
            "avg_reward": [],
            "complexity": [],
        }

        for it in range(1, num_iterations + 1):
            rollout = self._collect_experience()
            losses = self._update()
            self._maybe_advance_curriculum()

            history["pg_loss"].append(losses["actor_loss"])
            history["vf_loss"].append(losses["critic_loss"])
            history["entropy"].append(losses["entropy"])
            history["success_rate"].append(rollout["success_rate"])
            history["avg_reward"].append(rollout["avg_reward"])
            history["complexity"].append(rollout["complexity"])

            if it % self.config.log_interval == 0:
                self._log(
                    f"[SAC Breakdown iter {it}] "
                    f"complexity={rollout['complexity']} "
                    f"success={rollout['success_rate']:.2%} "
                    f"reward={rollout['avg_reward']:.3f} "
                    f"buffer={rollout['replay_size']} "
                    f"lib={rollout['library_size']} "
                    f"fhits={rollout['factor_hits']} "
                    f"lhits={rollout['library_hits']} "
                    f"trivial={rollout['trivial_skipped']} "
                    f"critic={losses['critic_loss']:.4f} "
                    f"actor={losses['actor_loss']:.4f} "
                    f"alpha={losses['alpha']:.4f} "
                    f"entropy={losses['entropy']:.4f}"
                )

        return history

    @torch.no_grad()
    def evaluate(
        self,
        complexities: Optional[List[int]] = None,
        num_trials: int = 50,
    ) -> Dict[int, float]:
        """Greedy roll-out evaluation, mirroring the PPO+MCTS variant."""
        if complexities is None:
            complexities = list(
                range(
                    self.config.starting_complexity,
                    self.config.max_complexity + 1,
                )
            )
        results: Dict[int, float] = {}
        self.actor.eval()
        for c in complexities:
            successes = 0
            for _ in range(num_trials):
                target = self._sample_target(c)
                obs = self.env.reset(target)
                if self.env.done:
                    successes += 1
                    continue
                while not self.env.done:
                    action = self._select_action(obs, deterministic=True)
                    obs, _, _, info = self.env.step(action)
                if info.get("is_success", False):
                    successes += 1
            results[c] = successes / max(num_trials, 1)
        return results

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save actor / critic / target / optimisers / temperature."""
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
                "algorithm": "sac_breakdown",
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Restore actor / critic / target / optimisers / temperature."""
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
        self.current_complexity = state.get(
            "current_complexity", self.current_complexity
        )
        self.total_env_steps = state.get("total_env_steps", self.total_env_steps)
