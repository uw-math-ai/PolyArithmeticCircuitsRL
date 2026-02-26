"""Core circuit construction game environment with Gymnasium-style API.

Uses FastPoly (numpy-based) for all polynomial arithmetic instead of SymPy,
giving 10-100x speedup per environment step.

Optionally integrates with FactorLibrary to provide factor-subgoal rewards:
when the target polynomial is factorizable, non-trivial factors become
per-episode sub-goals. The agent receives a bonus for constructing them and
an extra bonus if those factors were previously discovered in past episodes.
"""

from typing import Dict, List, Optional, Set, Tuple

import torch

from ..config import Config
from .fast_polynomial import FastPoly
from .action_space import decode_action, get_valid_actions_mask
from .factor_library import FactorLibrary

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None


class CircuitGame:
    """Circuit construction game environment.

    The agent builds an arithmetic circuit by selecting operations (add/multiply)
    on pairs of existing nodes. The goal is to construct a target polynomial with
    as few operations as possible.

    Nodes start as [x0, x1, ..., x_{n-1}, 1] and grow as operations are applied.
    All polynomial arithmetic uses FastPoly (dense numpy coefficient arrays mod p).

    If a FactorLibrary is provided, the environment additionally:
      - Computes non-trivial factors of the target at each reset() call.
      - Awards factor_subgoal_reward when the agent builds a factor node.
      - Awards an extra factor_library_bonus when that factor was previously seen
        in a past successful episode.
      - Dynamically discovers new subgoals at each step (when v_new is library-
        known): factorizes T - v_new and T / v_new (exact) to extend the active
        subgoal set for the remainder of the episode.
      - Awards completion_bonus when the circuit contains both pieces for a single
        final operation: additive (T - v_new already in circuit) or multiplicative
        (exact quotient T / v_new already in circuit or is a scalar). Each fires at
        most once per direction per episode.
      - Registers all constructed nodes into the library after a successful episode.

    Attributes:
        config (Config): Shared configuration object.
        factor_library (FactorLibrary | None): Optional session-level factor cache.
    """

    def __init__(
        self,
        config: Config,
        factor_library: Optional[FactorLibrary] = None,
    ) -> None:
        """Initialise the CircuitGame.

        Args:
            config: Configuration dataclass (n_variables, mod, max_nodes, etc.).
            factor_library: Optional session-level FactorLibrary. When provided
                and config.factor_library_enabled is True, factor subgoal rewards
                are active and the library is updated on successful episodes.
        """
        self.config = config
        self.factor_library = factor_library
        self.n_vars = config.n_variables
        self.mod = config.mod
        self.max_deg = config.effective_max_degree
        self.max_nodes = config.max_nodes
        self.max_actions = config.max_actions

        # Precompute initial node polynomials once; reused on every reset().
        # Shape: [x0, x1, ..., x_{n-1}, 1]
        self._init_polys: List[FastPoly] = []
        for i in range(self.n_vars):
            self._init_polys.append(
                FastPoly.variable(i, self.n_vars, self.max_deg, self.mod)
            )
        self._init_polys.append(
            FastPoly.constant(1, self.n_vars, self.max_deg, self.mod)
        )

        # Mutable episode state — initialised properly in reset().
        self.nodes: List[FastPoly] = []
        self.node_types: List[Tuple[int, int, int, float]] = []
        self.edges: List[Tuple[int, int]] = []
        self.target_poly: Optional[FastPoly] = None
        self.steps_taken: int = 0
        self.done: bool = True

        # Per-episode factor subgoal state — refreshed in reset().
        # _subgoal_keys: canonical keys of known subgoal polynomials this episode.
        #   Starts with non-trivial factors of the target (from reset), and grows
        #   dynamically during stepping as new residuals / quotients are discovered.
        # _library_known_keys: subset of _subgoal_keys that are in the library.
        # _subgoals_hit: keys already rewarded this episode (no double-rewarding).
        self._subgoal_keys: Set[bytes] = set()
        self._library_known_keys: Set[bytes] = set()
        self._subgoals_hit: Set[bytes] = set()

        # Completion bonus guards: each direction fires at most once per episode.
        # _additive_complete_hit: True after the first additive completion bonus.
        # _mult_complete_hit: True after the first multiplicative completion bonus.
        self._additive_complete_hit: bool = False
        self._mult_complete_hit: bool = False

    def reset(self, target_poly: FastPoly) -> Dict[str, torch.Tensor]:
        """Reset the game with a new target polynomial and return the initial observation.

        Reinitialises all circuit nodes to the base set [x0, ..., x_{n-1}, 1],
        clears all edges and step counters, and — if a FactorLibrary is attached —
        factorizes the new target to set up subgoal tracking for this episode.

        Args:
            target_poly: The FastPoly the agent must construct. Must be compatible
                         with this environment's n_vars, max_degree, and mod.

        Returns:
            Initial observation dict with keys 'graph', 'target', 'mask'.
        """
        self.target_poly = target_poly
        self.steps_taken = 0
        self.done = False

        # Initialise circuit nodes to the base inputs (copies to keep them mutable).
        self.nodes = [p.copy() for p in self._init_polys]
        self.node_types = []
        self.edges = []

        # Assign feature vectors: input vars get (1,0,0,0), constant gets (0,1,0,0).
        for _ in range(self.n_vars):
            self.node_types.append((1, 0, 0, 0.0))
        self.node_types.append((0, 1, 0, 0.0))  # constant node '1'

        # --- Factor subgoal setup ---
        # Reset per-episode tracking regardless of whether the library is active,
        # so clone() and subsequent steps always have consistent state.
        self._subgoal_keys = set()
        self._library_known_keys = set()
        self._subgoals_hit = set()
        self._additive_complete_hit = False
        self._mult_complete_hit = False

        if (
            self.factor_library is not None
            and self.config.factor_library_enabled
        ):
            # Factorize the target over Z (done once per episode; SymPy call here).
            # Returns non-trivial polynomial factors reduced mod p.
            factors = self.factor_library.factorize_target(target_poly)

            # Collect canonical keys for O(1) lookup during step().
            for f in factors:
                self._subgoal_keys.add(f.canonical_key())

            # Identify which subgoals the agent has previously built (library bonus).
            self._library_known_keys = self.factor_library.filter_known(factors)

        return self.get_observation()

    def step(
        self, action_idx: int
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, dict]:
        """Apply one action and return (observation, reward, done, info).

        Decodes the action index into (operation, node_i, node_j), computes the
        resulting polynomial, appends the new node to the circuit, then checks for
        success and computes the reward.

        Reward components:
          - step_penalty (-0.1 default): applied every step.
          - success_reward (+10.0 default): applied when the target is matched.
          - potential shaping (gamma * phi_after - phi_before): when enabled.
          - factor_subgoal_reward (+1.0 default): when a factor subgoal is built
            (at most once per distinct subgoal per episode).
          - factor_library_bonus (+0.5 default): stacks on subgoal reward when
            the factor was previously seen in the library.
          - completion_bonus (+3.0 default): when the circuit now contains both
            pieces for one final operation to reach T:
              * Additive: T - v_new already in the circuit (fires at most once).
              * Multiplicative: T / v_new is exact and quotient is in circuit or
                is a scalar (fires at most once, separate from additive).

        After a successful episode, all constructed nodes are registered in the
        FactorLibrary so future episodes can benefit from them.

        Args:
            action_idx: Integer index encoding (op, i, j) — see action_space.py.

        Returns:
            Tuple of (observation, reward, done, info) where info includes
            'is_success', 'steps_taken', 'num_nodes', 'new_poly', 'op',
            'operands', 'factor_hit', 'library_hit', 'additive_complete',
            'mult_complete'.

        Raises:
            AssertionError: If the game is already done or if node indices are
                            out of range (invalid action for the current state).
        """
        assert not self.done, "Game is already done. Call reset() first."

        op, i, j = decode_action(action_idx, self.max_nodes)
        num_nodes = len(self.nodes)

        assert i < num_nodes and j < num_nodes, (
            f"Invalid action: nodes {i},{j} but only {num_nodes} nodes exist"
        )

        # Compute the new polynomial via numpy arithmetic (mod p, truncated to max_deg).
        if op == 0:
            new_poly = self.nodes[i] + self.nodes[j]
        else:
            new_poly = self.nodes[i] * self.nodes[j]

        # Snapshot best similarity *before* adding the new node (for shaping delta).
        if self.config.use_reward_shaping:
            phi_before = self._best_similarity()

        # Append new node to the circuit.
        new_idx = len(self.nodes)
        self.nodes.append(new_poly)
        # Feature vector for operation nodes: (0, 0, 1, op_value)
        # op_value = 0.5 for add, 1.0 for multiply.
        op_value = 0.5 if op == 0 else 1.0
        self.node_types.append((0, 0, 1, op_value))

        # Add bidirectional edges (both operand→result directions).
        self.edges.append((i, new_idx))
        self.edges.append((j, new_idx))

        self.steps_taken += 1

        # Success check: exact match against target coefficient array (numpy equality).
        is_success = (new_poly == self.target_poly)

        # Termination conditions.
        at_max_steps = self.steps_taken >= self.config.max_steps
        at_max_nodes = len(self.nodes) >= self.max_nodes
        self.done = is_success or at_max_steps or at_max_nodes

        # --- Reward computation ---
        reward = self.config.step_penalty

        if is_success:
            reward += self.config.success_reward
        elif self.config.use_reward_shaping:
            # Potential-based shaping: reward the improvement in term similarity.
            # By the Ng et al. (1999) theorem this preserves the optimal policy.
            phi_after = self._best_similarity()
            reward += self.config.gamma * phi_after - phi_before

        # --- Factor subgoal reward ---
        # Check whether the newly created node matches any subgoal.
        # This is an O(1) set lookup — no SymPy involved here.
        factor_hit = False
        library_hit = False

        if self.factor_library is not None and self.config.factor_library_enabled:
            new_key = new_poly.canonical_key()
            if new_key in self._subgoal_keys and new_key not in self._subgoals_hit:
                # First time this subgoal has been built this episode.
                self._subgoals_hit.add(new_key)
                factor_hit = True
                reward += self.config.factor_subgoal_reward

                if new_key in self._library_known_keys:
                    # Also seen in a prior successful episode → extra bonus.
                    library_hit = True
                    reward += self.config.factor_library_bonus

        # --- Completion bonus + dynamic subgoal discovery ---
        # Runs when the factor library is active and success has not yet occurred
        # (no point awarding a completion bonus on the success step itself).
        additive_complete = False
        mult_complete = False

        if (
            self.factor_library is not None
            and self.config.factor_library_enabled
            and not is_success
        ):
            # Compute T - v_new (fast: numpy subtraction, no SymPy).
            residual = self.target_poly - new_poly

            # Snapshot canonical keys of all nodes present BEFORE this step.
            # (self.nodes[-1] is new_poly, already appended above.)
            existing_keys: Set[bytes] = {
                n.canonical_key() for n in self.nodes[:-1]
            }

            # 1. Additive completion bonus (fires at most once per episode).
            #    If T - v_new is already in the circuit, one more ADD gives T.
            if (
                not self._additive_complete_hit
                and not residual.is_zero()
                and residual.canonical_key() in existing_keys
            ):
                reward += self.config.completion_bonus
                self._additive_complete_hit = True
                additive_complete = True

            # 2. Library-gated: dynamic subgoal discovery + multiplicative checks.
            #    Only pay SymPy costs when v_new is already in the library,
            #    indicating it's a polynomially meaningful intermediate.
            if self.factor_library.contains(new_poly):

                # 2a. Add T - v_new as a direct additive subgoal.
                #     If the agent later builds the residual, one ADD gives T.
                if not residual.is_zero() and not self.factor_library.is_base(residual):
                    r_key = residual.canonical_key()
                    if r_key not in self._subgoal_keys:
                        self._subgoal_keys.add(r_key)
                        if self.factor_library.contains(residual):
                            self._library_known_keys.add(r_key)

                # 2b. Factorize T - v_new over Z → discover more additive subgoals.
                if not residual.is_zero():
                    for f in self.factor_library.factorize_poly(
                        residual, self._subgoal_keys
                    ):
                        k = f.canonical_key()
                        self._subgoal_keys.add(k)
                        if self.factor_library.contains(f):
                            self._library_known_keys.add(k)

                # 2c. Exact division T / v_new (uses SymPy — gated here).
                quotient = self.factor_library.exact_quotient(
                    self.target_poly, new_poly
                )
                if quotient is not None:
                    # Multiplicative completion bonus (fires at most once).
                    #   T / v_new is a scalar  →  just multiply v_new by that constant.
                    #   T / v_new in circuit   →  one MUL away from T.
                    if not self._mult_complete_hit:
                        q_key = quotient.canonical_key()
                        if quotient.is_scalar() or q_key in existing_keys:
                            reward += self.config.completion_bonus
                            self._mult_complete_hit = True
                            mult_complete = True

                    # Add quotient as a direct multiplicative subgoal.
                    if (
                        not quotient.is_scalar()
                        and not self.factor_library.is_base(quotient)
                    ):
                        q_key = quotient.canonical_key()
                        if q_key not in self._subgoal_keys:
                            self._subgoal_keys.add(q_key)
                            if self.factor_library.contains(quotient):
                                self._library_known_keys.add(q_key)

                        # Factorize quotient over Z → discover multiplicative subgoals.
                        for f in self.factor_library.factorize_poly(
                            quotient, self._subgoal_keys
                        ):
                            k = f.canonical_key()
                            self._subgoal_keys.add(k)
                            if self.factor_library.contains(f):
                                self._library_known_keys.add(k)

        # --- Library update on success ---
        # Register all agent-built nodes so future episodes can discover them.
        if is_success and self.factor_library is not None:
            n_initial = self.n_vars + 1  # x0,...,x_{n-1} plus the constant 1
            self.factor_library.register_episode_nodes(self.nodes, n_initial)

        info = {
            "is_success": is_success,
            "steps_taken": self.steps_taken,
            "num_nodes": len(self.nodes),
            "new_poly": new_poly,
            "op": "add" if op == 0 else "mul",
            "operands": (i, j),
            "factor_hit": factor_hit,           # Built a known subgoal this step.
            "library_hit": library_hit,         # Subgoal was also in the library.
            "additive_complete": additive_complete,   # Additive completion bonus fired.
            "mult_complete": mult_complete,           # Multiplicative completion bonus fired.
        }

        return self.get_observation(), reward, self.done, info

    def get_observation(self) -> Dict[str, torch.Tensor]:
        """Construct and return the current observation dict.

        The observation contains everything the policy network needs to select
        the next action: the circuit graph, the target polynomial encoding, and
        a validity mask over the action space.

        Returns:
            Dict with keys:
              'graph': PyG Data (or dict fallback) with node features and edges,
                       padded to max_nodes to allow batching.
              'target': FloatTensor of shape (target_size,) with coefficients
                        normalised to [0, 1] by dividing by mod.
              'mask': BoolTensor of shape (max_actions,) — True for valid actions.
        """
        graph = self._build_graph()
        target = self._encode_target(self.target_poly)
        mask = get_valid_actions_mask(len(self.nodes), self.max_nodes)

        return {
            "graph": graph,
            "target": target,
            "mask": mask,
        }

    def _build_graph(self):
        """Build a PyG Data object (or dict fallback) from the current circuit state.

        Nodes are padded to max_nodes with zero feature vectors so that batched
        GNN processing works without dynamic graph re-allocation. Only the first
        num_nodes_actual nodes carry meaningful information.

        Returns:
            torch_geometric.data.Data (if PyG is installed) or a plain dict with
            the same keys: x (node features), edge_index, num_nodes, num_nodes_actual.
        """
        # Node feature matrix: shape (max_nodes, node_feature_dim).
        # Rows beyond the current node count are left as zeros (padding).
        x = torch.zeros(self.max_nodes, self.config.node_feature_dim)
        for idx, features in enumerate(self.node_types):
            x[idx] = torch.tensor(features, dtype=torch.float32)

        # Edge index in COO format; edges stored bidirectionally for message passing.
        if self.edges:
            src = [e[0] for e in self.edges]
            dst = [e[1] for e in self.edges]
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        num_nodes_actual = len(self.nodes)

        if Data is not None:
            return Data(
                x=x,
                edge_index=edge_index,
                num_nodes=self.max_nodes,
                num_nodes_actual=num_nodes_actual,
            )
        else:
            return {
                "x": x,
                "edge_index": edge_index,
                "num_nodes": self.max_nodes,
                "num_nodes_actual": num_nodes_actual,
            }

    def _encode_target(self, poly: FastPoly) -> torch.Tensor:
        """Encode the target polynomial as a normalised flat coefficient vector.

        The dense coefficient array is flattened and divided by mod so that all
        values lie in [0, 1]. This is the representation fed to the target encoder
        MLP in the policy network.

        Args:
            poly: The target FastPoly to encode.

        Returns:
            FloatTensor of shape (target_size,) with values in [0, 1].
        """
        vec = poly.to_vector()  # numpy float64 flat array of shape (target_size,)
        return torch.tensor(vec, dtype=torch.float32) / self.mod

    def _best_similarity(self) -> float:
        """Compute the shaping potential phi(s) for the current state.

        The potential is the maximum term_similarity between any current circuit
        node and the target polynomial. A value of 1.0 means some node already
        matches the target exactly.

        Returns:
            Float in [0, 1] representing the best term-level match achieved so far.
        """
        best = 0.0
        for node_poly in self.nodes:
            sim = node_poly.term_similarity(self.target_poly)
            if sim > best:
                best = sim
                if best == 1.0:
                    break  # Perfect match found; no need to check further.
        return best

    def clone(self) -> "CircuitGame":
        """Create a deep copy of this game state for use in MCTS tree search.

        The FactorLibrary is NOT deep-copied — the clone shares the same
        session-level library instance (which is intentional: library updates
        from one branch of MCTS should not interfere with another, but both
        should read the same shared knowledge).

        Per-episode subgoal tracking sets are shallow-copied so that each MCTS
        clone tracks its own subgoal progress independently.

        Returns:
            A new CircuitGame with identical state that can be stepped independently.
        """
        new_game = CircuitGame.__new__(CircuitGame)
        new_game.config = self.config
        new_game.n_vars = self.n_vars
        new_game.mod = self.mod
        new_game.max_deg = self.max_deg
        new_game.max_nodes = self.max_nodes
        new_game.max_actions = self.max_actions
        new_game._init_polys = self._init_polys  # shared; never mutated after __init__

        # Shared (immutable) state: target and library reference.
        new_game.target_poly = self.target_poly
        new_game.factor_library = self.factor_library  # shared session-level object

        # Scalar state.
        new_game.steps_taken = self.steps_taken
        new_game.done = self.done

        # Deep-copy mutable circuit state (numpy arrays inside FastPoly are mutable).
        new_game.nodes = [p.copy() for p in self.nodes]
        new_game.node_types = list(self.node_types)
        new_game.edges = list(self.edges)

        # Per-episode factor tracking. _subgoal_keys and _library_known_keys can
        # grow during the episode (dynamic discovery), so copy them too.
        # _subgoals_hit tracks which subgoals each MCTS branch has already claimed.
        new_game._subgoal_keys = set(self._subgoal_keys)          # mutable copy
        new_game._library_known_keys = set(self._library_known_keys)  # mutable copy
        new_game._subgoals_hit = set(self._subgoals_hit)           # mutable copy

        # Completion bonus guards: each branch tracks its own fired state.
        new_game._additive_complete_hit = self._additive_complete_hit
        new_game._mult_complete_hit = self._mult_complete_hit

        return new_game
