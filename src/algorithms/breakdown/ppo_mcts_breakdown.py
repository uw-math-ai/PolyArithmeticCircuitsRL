"""PPO + MCTS trainer for top-down polynomial decomposition.

This is the *breakdown* counterpart to ``src/algorithms/ppo_mcts.py``. The
forward trainer learns to *build* a circuit; this trainer learns to
*decompose* a target polynomial into successively smaller pieces using the
:class:`BreakdownGame` environment defined in ``breakdown_env``.

Concretely:

* The policy / value network (:class:`BreakdownPolicyValueNet`) takes a
  flat encoding of the current focus polynomial, the original target, a
  short context vector, and per-candidate features for each of the
  ``max_options`` decomposition slots. It outputs masked logits over those
  slots and a scalar value estimate.
* :class:`BreakdownMCTS` runs PUCT-guided search on the deterministic
  decomposition tree, using the network for priors and leaf evaluations,
  identically in spirit to ``src/algorithms/mcts.py``.
* :class:`PPOMCTSBreakdownTrainer` combines the two with the standard PPO
  clipped-surrogate update and Generalised Advantage Estimation, mirroring
  the design of ``ppo_mcts.PPOMCTSTrainer``.

The whole file is **self-contained**: it imports only from the existing
``Config``, ``FactorLibrary``, ``FastPoly``, and the local
``breakdown_env`` module, plus standard numpy / torch. No existing code is
touched.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
# Policy / value network
# ----------------------------------------------------------------------


class BreakdownPolicyValueNet(nn.Module):
    """Compact MLP-style policy-value network for :class:`BreakdownGame`.

    The network has three input streams:

    * ``focus`` — flat coefficient vector of the current focus polynomial.
    * ``target`` — flat coefficient vector of the original target.
    * ``context`` — short scalar episode-level features.

    Their fused embedding is concatenated with per-candidate features
    (``cand_feats``, shape ``[K, CANDIDATE_FEATURE_DIM]``) and passed
    through a per-candidate scoring head to produce ``K`` logits. A
    separate value head reads the fused state embedding only.

    The masked logits are returned as ``-inf`` for invalid candidate slots
    so that ``log_softmax`` and sampling work directly.
    """

    def __init__(
        self,
        target_size: int,
        context_dim: int,
        candidate_feature_dim: int,
        max_options: int,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
    ) -> None:
        """Construct the network.

        Args:
            target_size: Length of the flat polynomial coefficient vector.
            context_dim: Length of the per-state context vector.
            candidate_feature_dim: Length of each per-candidate feature row.
            max_options: Size of the discrete action space (= ``K``).
            hidden_dim: Hidden width of the MLPs.
            embedding_dim: Width of the fused state embedding.
        """
        super().__init__()
        self.max_options = max_options

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

        # Per-candidate scoring head: state_emb (broadcast over K)
        # concatenated with each candidate feature row, then projected to a
        # scalar logit. Implemented as a small MLP.
        self.candidate_scorer = nn.Sequential(
            nn.Linear(embedding_dim + candidate_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _encode_state(self, obs: dict) -> torch.Tensor:
        """Compute the fused state embedding from a tensor obs dict."""
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

    def forward(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(masked_logits, value)`` for one or a batch of obs dicts.

        Args:
            obs: dict with keys ``focus``, ``target``, ``context``,
                ``cand_feats``, ``mask`` as produced by
                :func:`observation_to_tensors` /
                :func:`stack_observations`.

        Returns:
            ``logits`` of shape ``[B, K]`` (with masked positions at -inf)
            and ``value`` of shape ``[B]``.
        """
        state_emb = self._encode_state(obs)
        cand_feats = obs["cand_feats"]
        mask = obs["mask"]
        if cand_feats.dim() == 2:
            cand_feats = cand_feats.unsqueeze(0)
            mask = mask.unsqueeze(0)

        b, k, _ = cand_feats.shape
        # Broadcast state embedding across the K candidate slots.
        state_repeated = state_emb.unsqueeze(1).expand(b, k, state_emb.size(-1))
        merged = torch.cat([state_repeated, cand_feats], dim=-1)
        logits = self.candidate_scorer(merged).squeeze(-1)  # [B, K]
        logits = logits.masked_fill(~mask.bool(), float("-inf"))

        value = self.value_head(state_emb).squeeze(-1)  # [B]
        return logits, value

    @torch.no_grad()
    def get_policy_and_value(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(probs, value)`` for one observation, used by MCTS."""
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        return probs.squeeze(0), value.squeeze(0)


# ----------------------------------------------------------------------
# MCTS over the breakdown tree
# ----------------------------------------------------------------------


class _BreakdownMCTSNode:
    """Tree node for PUCT-guided MCTS on the breakdown game."""

    __slots__ = ("prior", "visit_count", "total_value", "children", "is_expanded")

    def __init__(self, prior: float = 0.0) -> None:
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[int, "_BreakdownMCTSNode"] = {}
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class BreakdownMCTS:
    """PUCT MCTS adapted for the deterministic breakdown environment.

    Mirrors the design of ``src/algorithms/mcts.MCTS`` but operates on
    :class:`BreakdownGame` and the breakdown observation format. Each
    simulation deep-copies the env via :meth:`BreakdownGame.clone`, walks
    down the tree by PUCT, expands a leaf with the network's prior, and
    backs up the network's value (or the actual terminal return).
    """

    def __init__(
        self,
        model: BreakdownPolicyValueNet,
        config: Config,
        device: str = "cpu",
        num_simulations: int = 64,
        c_puct: float = 1.4,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    @torch.no_grad()
    def search(self, game: BreakdownGame) -> Dict[int, int]:
        """Run ``num_simulations`` from the current ``game`` state.

        Returns:
            Mapping ``action_idx -> visit_count`` for all children of the
            root node.
        """
        root = _BreakdownMCTSNode()
        self._expand(root, game)

        for _ in range(self.num_simulations):
            node = root
            sim = game.clone()
            path = [node]
            last_info: dict = {}
            last_reward = 0.0

            while node.is_expanded and not sim.done:
                action, child = self._select_child(node)
                if action == -1 or child is None:
                    break
                _, last_reward, _, last_info = sim.step(action)
                node = child
                path.append(node)

            if sim.done:
                value = (
                    self.config.success_reward
                    if last_info.get("is_success", False)
                    else 0.0
                )
            else:
                value = self._expand(node, sim)

            for n in reversed(path):
                n.visit_count += 1
                n.total_value += value

        return {a: c.visit_count for a, c in root.children.items()}

    def get_action_probs(
        self, game: BreakdownGame, temperature: float = 1.0
    ) -> Tuple[int, np.ndarray]:
        """Run search and return ``(selected_action, prob_vector)``.

        Args:
            game: Live ``BreakdownGame`` (will be cloned per simulation).
            temperature: Standard MCTS sampling temperature; ``0`` selects
                the most-visited action greedily.

        Returns:
            ``selected_action`` (int) and a probability vector of length
            ``game.max_options`` summing to 1 (or zero everywhere if no
            simulation produced any visits — should not happen in practice).
        """
        visit_counts = self.search(game)
        K = game.max_options
        counts = np.zeros(K, dtype=np.float64)
        for a, c in visit_counts.items():
            counts[a] = c

        if counts.sum() == 0:
            # Degenerate fallback: pick any valid candidate uniformly.
            mask = np.zeros(K, dtype=np.float64)
            for i in range(game.num_valid_options):
                mask[i] = 1.0
            mask /= max(mask.sum(), 1.0)
            action = int(np.random.choice(K, p=mask)) if mask.sum() > 0 else 0
            return action, mask.astype(np.float32)

        if temperature == 0:
            action = int(counts.argmax())
            probs = np.zeros(K, dtype=np.float64)
            probs[action] = 1.0
        else:
            scaled = counts ** (1.0 / max(temperature, 1e-6))
            probs = scaled / scaled.sum()
            action = int(np.random.choice(K, p=probs))
        return action, probs.astype(np.float32)

    @torch.no_grad()
    def _expand(self, node: _BreakdownMCTSNode, game: BreakdownGame) -> float:
        """Expand a leaf node using the network's prior + value."""
        obs = game._build_observation()
        obs_t = observation_to_tensors(obs, device=self.device)
        probs, value = self.model.get_policy_and_value(obs_t)
        probs_np = probs.cpu().numpy()
        for a in range(game.max_options):
            if obs.mask[a]:
                node.children[a] = _BreakdownMCTSNode(prior=float(probs_np[a]))
        node.is_expanded = True
        return float(value.item())

    def _select_child(
        self, node: _BreakdownMCTSNode
    ) -> Tuple[int, Optional[_BreakdownMCTSNode]]:
        """Select the child maximising the PUCT score."""
        if not node.children:
            return -1, None
        sqrt_total = math.sqrt(max(node.visit_count, 1))
        best_score = -float("inf")
        best_action = -1
        best_child: Optional[_BreakdownMCTSNode] = None
        for a, child in node.children.items():
            puct = (
                child.q_value
                + self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            )
            if puct > best_score:
                best_score = puct
                best_action = a
                best_child = child
        return best_action, best_child


# ----------------------------------------------------------------------
# Rollout buffer & trainer
# ----------------------------------------------------------------------


@dataclass
class _BreakdownRolloutStep:
    obs: BreakdownObservation
    action: int
    reward: float
    network_log_prob: float
    search_policy: np.ndarray
    value: float
    done: bool


class PPOMCTSBreakdownTrainer:
    """Expert-iteration trainer combining MCTS rollouts with PPO updates.

    The control flow per iteration is:

    1. **Collect** ``steps_per_update`` decomposition transitions by running
       :class:`BreakdownMCTS` from the current network at every state.
    2. **Compute GAE** over the collected rewards using the network's value
       estimates as the baseline.
    3. **Run PPO** for ``ppo_epochs`` epochs of mini-batch updates with the
       clipped-surrogate objective. The "old" log probs come from the
       network's own policy at collection time (matching the design of the
       forward ``PPOMCTSTrainer``).
    4. **Adjust curriculum** based on the recent success-rate window, just
       like the forward trainer.

    A separate :class:`FactorLibrary` is created here. It is **not** shared
    with any other trainer or with the forward :class:`CircuitGame` so that
    enabling this trainer cannot influence existing experiments.

    Attributes:
        config (Config): Shared hyperparameters (see :mod:`src.config`).
        model (BreakdownPolicyValueNet): The policy / value net being trained.
        device (str): PyTorch device string.
        env (BreakdownGame): Decomposition environment.
        mcts (BreakdownMCTS): MCTS search engine.
    """

    MAX_BOARD_COMPLEXITY = 4

    def __init__(
        self,
        config: Config,
        device: str = "cpu",
        max_options: int = 32,
        max_breakdown_steps: int = 16,
        mcts_simulations: int = 32,
        mcts_c_puct: float = 1.4,
        log_path: Optional[str] = None,
        size_penalty_per_node: float = 0.05,
    ) -> None:
        """Initialise the trainer with a fresh model and environment.

        Args:
            config: Existing :class:`Config` instance — read-only here.
            device: PyTorch device (``"cpu"``, ``"cuda"``, ...).
            max_options: ``K`` — width of the candidate action space at
                each decomposition step.
            max_breakdown_steps: Hard limit on decomposition steps per
                episode.
            mcts_simulations: Number of MCTS simulations per move.
            mcts_c_puct: PUCT exploration constant for MCTS.
            log_path: Optional file path for textual logs (in addition to
                stdout).
            size_penalty_per_node: Per-node soft penalty applied at the
                end of an episode for trees larger than
                ``config.max_complexity``.
        """
        self.config = config
        self.device = device
        self.log_path = log_path
        self.max_options = max_options
        self.max_breakdown_steps = max_breakdown_steps

        self.model = BreakdownPolicyValueNet(
            target_size=config.target_size,
            context_dim=CONTEXT_FEATURE_DIM,
            candidate_feature_dim=CANDIDATE_FEATURE_DIM,
            max_options=max_options,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.ppo_lr)

        # Dedicated factor library — purposely separate from the forward
        # CircuitGame's library so the two trainers cannot interfere.
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
        self.mcts = BreakdownMCTS(
            self.model,
            config=config,
            device=device,
            num_simulations=mcts_simulations,
            c_puct=mcts_c_puct,
        )

        # Curriculum state.
        self.current_complexity = (
            config.starting_complexity
            if config.curriculum_enabled
            else config.max_complexity
        )
        self.success_history: List[bool] = []

        # Lazily-built BFS boards keyed by complexity.
        self._boards: Dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Logging / curriculum helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        print(msg)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a") as fh:
                fh.write(msg + "\n")

    def _get_board(self, complexity: int) -> dict:
        if complexity not in self._boards:
            self._boards[complexity] = build_game_board(self.config, complexity)
        return self._boards[complexity]

    def _sample_target(self, complexity: int) -> FastPoly:
        """Sample a target polynomial via BFS for low complexity, RNG otherwise."""
        if complexity <= self.MAX_BOARD_COMPLEXITY:
            board = self._get_board(complexity)
            poly, _ = sample_target(self.config, complexity, board)
            return poly
        poly, _ = generate_random_circuit(self.config, complexity)
        return poly

    def _get_temperature(self, step: int) -> float:
        """Same MCTS temperature schedule as the forward PPO+MCTS trainer."""
        decay_frac = min(step / max(self.config.temperature_decay_steps, 1), 1.0)
        return self.config.temperature_init + (
            self.config.temperature_final - self.config.temperature_init
        ) * decay_frac

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(self) -> Tuple[List[_BreakdownRolloutStep], dict]:
        """Run MCTS-guided rollouts until the buffer holds enough steps.

        Each per-step record stores the observation, MCTS action, reward,
        the network's log-prob of the chosen action (used as the PPO "old"
        log-prob), the full search policy, and the value estimate.

        Returns:
            ``(buffer, info)`` where ``buffer`` is the ordered list of
            transitions and ``info`` is a small dict of diagnostic stats.
        """
        buffer: List[_BreakdownRolloutStep] = []
        episodes = 0
        successes = 0
        total_reward = 0.0
        factor_hits = 0
        library_hits = 0
        skipped_trivial = 0

        self.model.eval()

        while len(buffer) < self.config.steps_per_update:
            target_poly = self._sample_target(self.current_complexity)
            obs = self.env.reset(target_poly)

            # Trivial targets (already a base node) have done=True from
            # reset — count them as immediate successes and move on.
            if self.env.done:
                episodes += 1
                successes += 1
                self.success_history.append(True)
                skipped_trivial += 1
                continue

            episode_reward = 0.0
            step = 0

            while not self.env.done:
                obs_t = observation_to_tensors(obs, device=self.device)
                with torch.no_grad():
                    logits, value = self.model(obs_t)

                temp = self._get_temperature(step)
                action, search_probs = self.mcts.get_action_probs(
                    self.env, temperature=temp
                )

                # Network log-prob of the chosen action — used as PPO old_lp.
                with torch.no_grad():
                    dist = torch.distributions.Categorical(logits=logits)
                    a_t = torch.tensor([action], device=self.device)
                    network_lp = float(dist.log_prob(a_t).item())

                next_obs, reward, done, info = self.env.step(action)

                buffer.append(
                    _BreakdownRolloutStep(
                        obs=obs,
                        action=int(action),
                        reward=float(reward),
                        network_log_prob=network_lp,
                        search_policy=np.asarray(search_probs, dtype=np.float32),
                        value=float(value.item()),
                        done=bool(done),
                    )
                )

                if info.get("factor_hit", False):
                    factor_hits += 1
                if info.get("library_hit", False):
                    library_hits += 1

                episode_reward += reward
                obs = next_obs
                step += 1

            episodes += 1
            total_reward += episode_reward
            episode_success = bool(info.get("is_success", False))
            successes += int(episode_success)
            self.success_history.append(episode_success)

            if episode_success:
                self.env.register_decomposition_in_library()

        self.model.train()

        info_dict = {
            "episodes": episodes,
            "success_rate": successes / max(episodes, 1),
            "avg_reward": total_reward / max(episodes, 1),
            "complexity": self.current_complexity,
            "factor_hits": factor_hits,
            "library_hits": library_hits,
            "library_size": len(self.factor_library),
            "trivial_skipped": skipped_trivial,
        }
        return buffer, info_dict

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(
        self, buffer: List[_BreakdownRolloutStep]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standard GAE-λ computation on the collected buffer."""
        n = len(buffer)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        last_value = 0.0
        for t in reversed(range(n)):
            if buffer[t].done:
                next_value = 0.0
                last_gae = 0.0
            elif t + 1 < n:
                next_value = buffer[t + 1].value
            else:
                next_value = last_value
            delta = (
                buffer[t].reward + self.config.gamma * next_value - buffer[t].value
            )
            last_gae = (
                delta + self.config.gamma * self.config.gae_lambda * last_gae
            )
            advantages[t] = last_gae
            returns[t] = advantages[t] + buffer[t].value
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(
        self,
        buffer: List[_BreakdownRolloutStep],
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict:
        """Run the PPO clipped-surrogate update over MCTS-collected data.

        The importance ratio is computed against the network's own log-prob
        at collection time — same convention as the forward
        :class:`PPOMCTSTrainer`. A small auxiliary cross-entropy term
        distils the MCTS visit distribution into the policy when
        ``config.gumbel_distill_coef`` is positive.
        """
        n = len(buffer)
        if n == 0:
            return {
                "pg_loss": 0.0,
                "vf_loss": 0.0,
                "entropy": 0.0,
                "distill_loss": 0.0,
            }

        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_lp = torch.tensor(
            [s.network_log_prob for s in buffer],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            [s.action for s in buffer], dtype=torch.long, device=self.device
        )
        search_probs = torch.tensor(
            np.stack([s.search_policy for s in buffer], axis=0),
            dtype=torch.float32,
            device=self.device,
        )

        total_pg = 0.0
        total_vf = 0.0
        total_ent = 0.0
        total_distill = 0.0
        n_updates = 0

        for _ in range(self.config.ppo_epochs):
            idx_perm = np.random.permutation(n)
            for start in range(0, n, self.config.batch_size):
                end = min(start + self.config.batch_size, n)
                bidx = idx_perm[start:end]

                batch_obs = stack_observations(
                    [buffer[i].obs for i in bidx], device=self.device
                )
                logits, values = self.model(batch_obs)

                batch_actions = actions[bidx]
                batch_adv = adv_t[bidx]
                batch_ret = ret_t[bidx]
                batch_old = old_lp[bidx]
                batch_search = search_probs[bidx]

                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - batch_old)
                surr1 = ratio * batch_adv
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.ppo_clip,
                        1 + self.config.ppo_clip,
                    )
                    * batch_adv
                )
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = F.mse_loss(values, batch_ret)

                # Distillation against the MCTS visit distribution. Cheap
                # KL approximation; matches the forward trainer's optional
                # gumbel-distill term.
                log_probs = torch.log_softmax(logits, dim=-1)
                distill_terms = torch.where(
                    batch_search > 0,
                    batch_search * log_probs,
                    torch.zeros_like(log_probs),
                )
                distill_loss = -distill_terms.sum(dim=-1).mean()

                loss = (
                    pg_loss
                    + self.config.vf_coef * vf_loss
                    - self.config.ent_coef * entropy
                    + self.config.gumbel_distill_coef * distill_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_pg += pg_loss.item()
                total_vf += vf_loss.item()
                total_ent += entropy.item()
                total_distill += distill_loss.item()
                n_updates += 1

        return {
            "pg_loss": total_pg / max(n_updates, 1),
            "vf_loss": total_vf / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
            "distill_loss": total_distill / max(n_updates, 1),
        }

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _maybe_advance_curriculum(self) -> None:
        """Sliding-window curriculum identical to the forward trainer's."""
        if not self.config.curriculum_enabled:
            return
        window = 50
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
    # Public training entry-point
    # ------------------------------------------------------------------

    def train(self, num_iterations: int) -> dict:
        """Run ``num_iterations`` collect → GAE → PPO update cycles.

        Returns:
            A history dict matching the forward trainer's keys
            (``pg_loss``, ``vf_loss``, ``entropy``, ``success_rate``,
            ``avg_reward``, ``complexity``) so downstream plotting code can
            be reused unchanged.
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
            buffer, info = self.collect_rollouts()
            adv, ret = self.compute_gae(buffer)
            loss_info = self.update(buffer, adv, ret)
            self._maybe_advance_curriculum()

            history["pg_loss"].append(loss_info["pg_loss"])
            history["vf_loss"].append(loss_info["vf_loss"])
            history["entropy"].append(loss_info["entropy"])
            history["success_rate"].append(info["success_rate"])
            history["avg_reward"].append(info["avg_reward"])
            history["complexity"].append(info["complexity"])

            if it % self.config.log_interval == 0:
                self._log(
                    f"[PPO+MCTS Breakdown iter {it}] "
                    f"complexity={info['complexity']} "
                    f"episodes={info['episodes']} "
                    f"success={info['success_rate']:.2%} "
                    f"reward={info['avg_reward']:.3f} "
                    f"lib={info['library_size']} "
                    f"fhits={info['factor_hits']} "
                    f"lhits={info['library_hits']} "
                    f"trivial={info['trivial_skipped']} "
                    f"pg_loss={loss_info['pg_loss']:.4f} "
                    f"vf_loss={loss_info['vf_loss']:.4f} "
                    f"entropy={loss_info['entropy']:.4f} "
                    f"distill={loss_info['distill_loss']:.4f}"
                )

        return history

    # ------------------------------------------------------------------
    # Lightweight evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        complexities: Optional[List[int]] = None,
        num_trials: int = 50,
    ) -> Dict[int, float]:
        """Greedy roll-out evaluation over a list of complexities.

        For each complexity ``c``, samples ``num_trials`` targets and runs
        the policy without MCTS using the most-likely candidate at every
        step. Returns a dict ``{complexity: success_rate}``.
        """
        if complexities is None:
            complexities = list(
                range(
                    self.config.starting_complexity,
                    self.config.max_complexity + 1,
                )
            )
        results: Dict[int, float] = {}
        self.model.eval()
        for c in complexities:
            successes = 0
            for _ in range(num_trials):
                target = self._sample_target(c)
                obs = self.env.reset(target)
                if self.env.done:
                    successes += 1
                    continue
                while not self.env.done:
                    obs_t = observation_to_tensors(obs, device=self.device)
                    logits, _ = self.model(obs_t)
                    action = int(logits.argmax(dim=-1).item())
                    obs, _, done, info = self.env.step(action)
                if info.get("is_success", False):
                    successes += 1
            results[c] = successes / max(num_trials, 1)
        return results
