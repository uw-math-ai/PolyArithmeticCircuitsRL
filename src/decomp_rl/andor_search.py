"""Memoized AND/OR search with cost-minimizing PUCT selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .baseline_cost import BaselineCostModel
from .config import SearchConfig
from .cost_model import rebuild_cost, unresolved_children
from .factor_fp import FiniteFieldFactorizer
from .model import HeuristicPolicyValueModel, PolicyValueModel
from .polynomial import SparsePolynomial
from .split_proposals import SplitAction, propose_splits


@dataclass
class DecompositionTrace:
    poly: SparsePolynomial
    total_cost: float
    chosen_action: SplitAction | None = None
    children: tuple["DecompositionTrace", ...] = ()


@dataclass
class ActionStats:
    action: SplitAction
    prior: float
    visits: int = 0
    total_cost: float = 0.0
    best_cost: float = math.inf
    heuristic_cost: float = math.inf
    best_trace: DecompositionTrace | None = None

    @property
    def mean_cost(self) -> float:
        if self.visits == 0:
            return self.heuristic_cost
        return self.total_cost / self.visits


@dataclass
class NodeStats:
    poly: SparsePolynomial
    baseline_cost: float
    value_estimate: float
    actions: list[ActionStats] = field(default_factory=list)
    visits: int = 0
    best_cost: float = math.inf
    best_trace: DecompositionTrace | None = None


@dataclass
class SearchResult:
    root: SparsePolynomial
    best_cost: float
    best_trace: DecompositionTrace
    root_candidates: tuple[SplitAction, ...]
    root_policy: tuple[float, ...]
    root_value: float
    stats: "SearchStats"


@dataclass(frozen=True)
class SearchStats:
    simulations: int
    node_expansions: int
    transposition_hits: int
    total_candidates: int
    factor_cache_requests: int
    factor_cache_hits: int

    @property
    def average_branch_factor(self) -> float:
        if self.node_expansions == 0:
            return 0.0
        return self.total_candidates / self.node_expansions

    @property
    def transposition_hit_rate(self) -> float:
        denominator = self.node_expansions + self.transposition_hits
        if denominator == 0:
            return 0.0
        return self.transposition_hits / denominator

    @property
    def factor_cache_hit_rate(self) -> float:
        if self.factor_cache_requests == 0:
            return 0.0
        return self.factor_cache_hits / self.factor_cache_requests


class AndOrSearch:
    def __init__(
        self,
        factorizer: FiniteFieldFactorizer | None = None,
        baseline_model: BaselineCostModel | None = None,
        model: PolicyValueModel | None = None,
        search_config: SearchConfig | None = None,
    ) -> None:
        self.factorizer = factorizer or FiniteFieldFactorizer()
        self.baseline_model = baseline_model or BaselineCostModel()
        self.model = model or HeuristicPolicyValueModel(self.baseline_model)
        self.search_config = search_config or SearchConfig()
        self._table: dict[str, NodeStats] = {}
        self._node_expansions = 0
        self._transposition_hits = 0
        self._total_candidates = 0

    def close(self) -> None:
        self.factorizer.close()

    def search(self, root: SparsePolynomial) -> SearchResult:
        self._table = {}
        self._node_expansions = 0
        self._transposition_hits = 0
        self._total_candidates = 0
        self.factorizer.reset_stats()
        for _ in range(self.search_config.simulations):
            self._simulate(root, depth=0)

        root_node = self._table.get(root.to_key())
        stats = SearchStats(
            simulations=self.search_config.simulations,
            node_expansions=self._node_expansions,
            transposition_hits=self._transposition_hits,
            total_candidates=self._total_candidates,
            factor_cache_requests=self.factorizer.cache_requests,
            factor_cache_hits=self.factorizer.cache_hits,
        )
        if root_node is None:
            baseline = self.baseline_model.direct_construction_cost(root)
            trace = DecompositionTrace(poly=root, total_cost=baseline)
            return SearchResult(root, baseline, trace, (), (), 0.0, stats)

        root_visits = sum(action.visits for action in root_node.actions)
        if root_visits == 0:
            root_policy = tuple(0.0 for _ in root_node.actions)
        else:
            root_policy = tuple(action.visits / root_visits for action in root_node.actions)
        trace = root_node.best_trace or DecompositionTrace(root, root_node.baseline_cost)
        value = (root_node.baseline_cost - root_node.best_cost) / max(1.0, root_node.baseline_cost)
        return SearchResult(
            root=root,
            best_cost=root_node.best_cost,
            best_trace=trace,
            root_candidates=tuple(action.action for action in root_node.actions),
            root_policy=root_policy,
            root_value=value,
            stats=stats,
        )

    def _expand(self, poly: SparsePolynomial) -> NodeStats:
        baseline = float(self.baseline_model.direct_construction_cost(poly))
        candidates = propose_splits(poly, self.search_config.expand_top_k, baseline_model=self.baseline_model)
        priors, value_estimate = self.model.score_candidates(poly, candidates)
        actions = []
        for prior, candidate in zip(priors, candidates):
            heuristic_cost = 1 + self.baseline_model.direct_construction_cost(candidate.g) + self.baseline_model.direct_construction_cost(candidate.h)
            actions.append(
                ActionStats(
                    action=candidate,
                    prior=prior,
                    heuristic_cost=float(heuristic_cost),
                )
            )
        node = NodeStats(
            poly=poly,
            baseline_cost=baseline,
            value_estimate=value_estimate,
            actions=actions,
            best_cost=baseline,
            best_trace=DecompositionTrace(poly=poly, total_cost=baseline),
        )
        self._table[poly.to_key()] = node
        self._node_expansions += 1
        self._total_candidates += len(candidates)
        return node

    def _select_action(self, node: NodeStats) -> ActionStats:
        exploration_scale = self.search_config.puct_exploration * math.sqrt(max(1, node.visits))
        best_score = -math.inf
        best_action = node.actions[0]
        for action in node.actions:
            exploitation = -action.mean_cost
            exploration = exploration_scale * action.prior / (1 + action.visits)
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _simulate(self, poly: SparsePolynomial, depth: int) -> tuple[float, DecompositionTrace]:
        if self.baseline_model.is_base_case(poly):
            cost = float(self.baseline_model.exact_base_cost(poly))
            return cost, DecompositionTrace(poly=poly, total_cost=cost)

        if depth >= self.search_config.max_depth:
            baseline = float(self.baseline_model.direct_construction_cost(poly))
            return baseline, DecompositionTrace(poly=poly, total_cost=baseline)

        node = self._table.get(poly.to_key())
        if node is None:
            node = self._expand(poly)
            if not node.actions:
                return node.baseline_cost, node.best_trace or DecompositionTrace(poly, node.baseline_cost)
        else:
            self._transposition_hits += 1

        if not node.actions:
            return node.baseline_cost, node.best_trace or DecompositionTrace(poly, node.baseline_cost)

        chosen = self._select_action(node)
        action_cost, action_trace = self._evaluate_action(poly, chosen.action, depth)

        chosen.visits += 1
        chosen.total_cost += action_cost
        if action_cost < chosen.best_cost:
            chosen.best_cost = action_cost
            chosen.best_trace = action_trace

        node.visits += 1
        if action_cost < node.best_cost:
            node.best_cost = action_cost
            node.best_trace = action_trace
        return node.best_cost, node.best_trace or DecompositionTrace(poly, node.baseline_cost)

    def _evaluate_action(self, poly: SparsePolynomial, action: SplitAction, depth: int) -> tuple[float, DecompositionTrace]:
        g_factors = self.factorizer.factor(action.g)
        h_factors = self.factorizer.factor(action.h)
        total = 1 + rebuild_cost(g_factors) + rebuild_cost(h_factors)

        children_map: dict[str, SparsePolynomial] = {}
        for child in unresolved_children(g_factors) + unresolved_children(h_factors):
            if self.baseline_model.is_base_case(child):
                total += self.baseline_model.exact_base_cost(child)
                continue
            children_map.setdefault(child.to_key(), child)

        child_traces = []
        for child in children_map.values():
            child_cost, child_trace = self._simulate(child, depth + 1)
            total += child_cost
            child_traces.append(child_trace)

        trace = DecompositionTrace(
            poly=poly,
            total_cost=total,
            chosen_action=action,
            children=tuple(child_traces),
        )
        return total, trace
