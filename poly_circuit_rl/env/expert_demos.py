"""Expert demonstration generator using BFS-optimal trajectories.

Uses graph_enumeration to find shortest paths from base nodes to targets,
then replays those paths through PolyCircuitEnv to collect transition tuples
that can be pre-loaded into the replay buffer.

This directly addresses the exploration catastrophe: the agent sees successful
trajectories from the start of training.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..config import Config
from ..core.action_codec import ACTION_ADD, ACTION_MUL, ACTION_SET_OUTPUT, encode_action
from ..core.poly import Poly, poly_hashkey
from . import graph_cache
from .graph_enumeration import build_game_graph, analyze_graph
from .samplers import _sympy_expr_to_poly

log = logging.getLogger(__name__)


class ExpertDemoGenerator:
    """Generates expert demonstration trajectories from BFS-optimal paths."""

    def __init__(self, config: Config):
        self.config = config
        self.n_vars = config.n_vars
        self.var_names: List[str] = (
            ["x"] if config.n_vars == 1 else [f"x{i}" for i in range(config.n_vars)]
        )
        self._G = None
        self._dist: Optional[Dict[str, float]] = None
        self._roots: Optional[Set[str]] = None
        # Mapping: poly_hashkey -> DAG node ID (canonical SymPy string)
        self._poly_to_dag_id: Dict[tuple, str] = {}
        self._built_up_to: int = 0

    def build_graph(self, max_steps: int) -> None:
        """Build the game DAG up to max_steps."""
        if max_steps <= self._built_up_to:
            return

        key = graph_cache.compute_cache_key(
            n_vars=self.n_vars,
            steps=max_steps,
            gen_max_graph_nodes=self.config.gen_max_graph_nodes,
            gen_max_successors=self.config.gen_max_successors,
            gen_max_seconds=self.config.gen_max_seconds,
        )

        t_load = time.perf_counter()
        cached = graph_cache.load(key)
        if cached is not None:
            self._G = cached["G"]
            self._dist = cached["dist"]
            self._roots = cached["roots"]
            self._poly_to_dag_id = cached["poly_to_dag_id"]
            self._built_up_to = cached["built_up_to"]
            log.info(
                "ExpertDemoGenerator: loaded cached DAG (key=%s, %d nodes, %d mapped to Poly) in %.2fs",
                key,
                self._G.number_of_nodes(),
                len(self._poly_to_dag_id),
                time.perf_counter() - t_load,
            )
            return

        t0 = time.perf_counter()
        self._G = build_game_graph(
            steps=max_steps,
            num_vars=self.n_vars,
            max_nodes=self.config.gen_max_graph_nodes,
            max_successors_per_node=self.config.gen_max_successors,
            max_seconds=self.config.gen_max_seconds,
        )
        log.info(
            "ExpertDemoGenerator: build_game_graph finished (%d nodes) in %.2fs",
            self._G.number_of_nodes(),
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        _records, self._dist, self._roots = analyze_graph(
            self._G,
            only_multipath=False,
            only_shortcut=False,
            min_shortcut_gap=0,
            max_step=max_steps,
            num_vars=self.n_vars,
        )
        log.info(
            "ExpertDemoGenerator: analyze_graph finished in %.2fs",
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        self._build_poly_map()
        log.info(
            "ExpertDemoGenerator: _build_poly_map finished (%d mapped) in %.2fs",
            len(self._poly_to_dag_id),
            time.perf_counter() - t0,
        )

        self._built_up_to = max_steps

        try:
            t0 = time.perf_counter()
            graph_cache.save(key, {
                "G": self._G,
                "dist": self._dist,
                "roots": self._roots,
                "poly_to_dag_id": self._poly_to_dag_id,
                "built_up_to": self._built_up_to,
            })
            log.info(
                "ExpertDemoGenerator: saved DAG cache (key=%s) in %.2fs",
                key,
                time.perf_counter() - t0,
            )
        except Exception as e:
            log.warning("ExpertDemoGenerator: failed to save DAG cache (key=%s): %s", key, e)

    def _build_poly_map(self) -> None:
        """Map every DAG node's SymPy expression to its internal Poly hashkey."""
        self._poly_to_dag_id.clear()
        for node_id, data in self._G.nodes(data=True):
            expr_str = data.get("expr_str")
            if expr_str is None:
                expr = data.get("expr")
                if expr is None:
                    continue
                expr_str = str(expr)
            try:
                poly = _sympy_expr_to_poly(expr_str, self.var_names)
                key = poly_hashkey(poly)
                self._poly_to_dag_id[key] = node_id
            except Exception:
                continue

    def _extract_action_sequence(
        self, target_dag_id: str,
    ) -> Optional[List[Tuple[str, str, str]]]:
        """BFS backward from target to find one shortest action sequence.

        Returns list of (op, source_dag_id, operand_dag_id) in forward order,
        or None if no path exists.
        """
        if self._dist is None or self._roots is None:
            return None

        target_dist = self._dist.get(target_dag_id)
        if target_dist is None or target_dist == float("inf"):
            return None
        if target_dist == 0:
            return []  # Target is a root node

        # BFS backward to find all nodes on shortest paths
        on_path: Set[str] = set()
        queue: deque[str] = deque([target_dag_id])
        while queue:
            v = queue.popleft()
            if v in on_path:
                continue
            on_path.add(v)
            for u in self._G.predecessors(v):
                if u in self._dist and self._dist[u] + 1 == self._dist[v]:
                    queue.append(u)

        # Topological sort of on_path nodes by distance from roots
        sorted_nodes = sorted(
            on_path,
            key=lambda nid: (self._dist.get(nid, float("inf")), nid),
        )

        # Build action sequence: for each non-root node (in order), find its
        # producing edge (source + operand + op)
        actions: List[Tuple[str, str, str]] = []
        for node_id in sorted_nodes:
            if node_id in self._roots:
                continue

            # Find an incoming edge on a shortest path
            found = False
            for pred in self._G.predecessors(node_id):
                if pred not in on_path and pred not in self._roots:
                    continue
                if self._dist.get(pred, float("inf")) + 1 != self._dist[node_id]:
                    continue
                edge_data = self._G.edges[pred, node_id]
                op = edge_data.get("op", "add")
                operand = edge_data.get("operand", pred)
                actions.append((op, pred, operand))
                found = True
                break

            if not found:
                return None  # Shouldn't happen, but fail gracefully

        return actions

    def generate_demo(
        self,
        env,
        target_poly: Poly,
        max_ops: int,
    ) -> Optional[List]:
        """Generate one expert demonstration for a target polynomial.

        Returns list of Transition objects, or None if demo generation fails.
        """
        from ..rl.replay_buffer import Transition

        target_key = poly_hashkey(target_poly)
        target_dag_id = self._poly_to_dag_id.get(target_key)
        if target_dag_id is None:
            return None

        action_seq = self._extract_action_sequence(target_dag_id)
        if action_seq is None:
            return None

        # Reset env with this specific target
        obs_dict, _ = env.reset(options={
            "max_ops": max_ops,
            "target_poly": target_poly,
        })

        # Build a mapping from DAG node ID to circuit builder node index
        dag_to_builder: Dict[str, int] = {}

        # Map root DAG nodes to initial builder nodes
        for i, node in enumerate(env.builder.nodes):
            node_key = poly_hashkey(node.poly)
            dag_id = self._poly_to_dag_id.get(node_key)
            if dag_id is not None:
                dag_to_builder[dag_id] = i

        transitions: List[Transition] = []

        for op, source_dag_id, operand_dag_id in action_seq:
            source_idx = dag_to_builder.get(source_dag_id)
            operand_idx = dag_to_builder.get(operand_dag_id)

            if source_idx is None or operand_idx is None:
                return None  # Can't map DAG path to builder indices

            # Canonical pair order: i <= j
            i, j = min(source_idx, operand_idx), max(source_idx, operand_idx)

            if op == "add":
                action = encode_action(ACTION_ADD, i, j, self.config.L)
            else:
                action = encode_action(ACTION_MUL, i, j, self.config.L)

            obs = obs_dict["obs"]
            mask = obs_dict["action_mask"]

            # Verify action is valid
            if mask[action] == 0:
                return None

            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get decomposed rewards from trajectory
            traj = env.get_trajectory()
            last_traj = traj[-1] if traj else {}

            transitions.append(Transition(
                obs=obs.copy(),
                action=action,
                reward=reward,
                next_obs=next_obs_dict["obs"].copy(),
                done=done,
                action_mask=mask.copy(),
                next_action_mask=next_obs_dict["action_mask"].copy(),
                is_demo=True,
                base_reward=last_traj.get("base_reward", reward),
                shaping_reward=last_traj.get("shaping_reward", 0.0),
                solve_bonus=last_traj.get("solve_bonus", 0.0),
            ))

            # Map newly created node back to DAG
            result_node = env.builder.nodes[-1]
            result_key = poly_hashkey(result_node.poly)
            result_dag_id = self._poly_to_dag_id.get(result_key)
            if result_dag_id is not None:
                dag_to_builder[result_dag_id] = result_node.node_id

            obs_dict = next_obs_dict

            if done:
                break

        # If not yet solved, add SET_OUTPUT for the last created node + check
        if transitions and not transitions[-1].done:
            last_node_idx = env.builder.nodes[-1].node_id
            visible = min(len(env.builder.nodes), self.config.L)
            if last_node_idx < visible:
                set_action = encode_action(
                    ACTION_SET_OUTPUT, last_node_idx, None, self.config.L,
                )
                obs = obs_dict["obs"]
                mask = obs_dict["action_mask"]
                if mask[set_action] == 1:
                    next_obs_dict, reward, terminated, truncated, info = env.step(set_action)
                    done = terminated or truncated

                    traj = env.get_trajectory()
                    last_traj = traj[-1] if traj else {}

                    transitions.append(Transition(
                        obs=obs.copy(),
                        action=set_action,
                        reward=reward,
                        next_obs=next_obs_dict["obs"].copy(),
                        done=done,
                        action_mask=mask.copy(),
                        next_action_mask=next_obs_dict["action_mask"].copy(),
                        is_demo=True,
                        base_reward=last_traj.get("base_reward", 0.0),
                        shaping_reward=last_traj.get("shaping_reward", 0.0),
                        solve_bonus=last_traj.get("solve_bonus", 0.0),
                    ))

        return transitions if transitions else None

    def generate_demos(
        self,
        env,
        num_demos: int,
        curriculum_levels: Tuple[int, ...] = (1, 2, 3, 4),
        rng: Optional[random.Random] = None,
    ) -> List:
        """Generate up to num_demos expert demonstrations across curriculum levels.

        Distributes demos across levels proportionally.
        Returns list of Transition objects ready for buffer insertion.
        """
        if rng is None:
            rng = random.Random(self.config.seed + 500)

        max_level = max(curriculum_levels) if curriculum_levels else 4
        self.build_graph(max_level)

        # Collect all reachable targets grouped by shortest_length
        targets_by_ops: Dict[int, List[Poly]] = {}
        for key, dag_id in self._poly_to_dag_id.items():
            dist = self._dist.get(dag_id, float("inf"))
            if dist == float("inf") or dist == 0:
                continue
            ops = int(dist)
            if ops > max_level:
                continue
            node_data = self._G.nodes[dag_id]
            expr_str = node_data.get("expr_str")
            if expr_str is None:
                expr = node_data.get("expr")
                if expr is None:
                    continue
                expr_str = str(expr)
            try:
                poly = _sympy_expr_to_poly(expr_str, self.var_names)
                targets_by_ops.setdefault(ops, []).append(poly)
            except Exception:
                continue

        if not targets_by_ops:
            log.warning("ExpertDemoGenerator: no reachable targets found")
            return []

        # Distribute demos across levels
        all_transitions = []
        demos_per_level = max(1, num_demos // len(curriculum_levels))

        for level_ops in curriculum_levels:
            candidates = []
            for ops, polys in targets_by_ops.items():
                if ops <= level_ops:
                    candidates.extend(polys)

            if not candidates:
                continue

            rng.shuffle(candidates)
            generated = 0

            for target_poly in candidates:
                if generated >= demos_per_level:
                    break

                demo = self.generate_demo(env, target_poly, max_ops=level_ops)
                if demo is not None:
                    all_transitions.extend(demo)
                    generated += 1

        log.info(
            "ExpertDemoGenerator: generated %d transitions from %d levels",
            len(all_transitions),
            len(curriculum_levels),
        )
        return all_transitions
