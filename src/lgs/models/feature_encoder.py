"""Fixed-schema feature encoder for candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass, field

from lgs.env.candidate import Candidate
from lgs.env.circuit_state import CircuitState
from lgs.env.problem_instance import ProblemInstance


@dataclass(frozen=True)
class CandidateFeatureEncoder:
    feature_names: tuple[str, ...] = field(default_factory=lambda: FEATURE_NAMES)

    def encode(
        self,
        instance: ProblemInstance,
        state: CircuitState,
        candidate: Candidate,
    ) -> list[float]:
        if not isinstance(instance, ProblemInstance):
            raise TypeError("instance must be a ProblemInstance")
        if not isinstance(state, CircuitState):
            raise TypeError("state must be a CircuitState")
        if not isinstance(candidate, Candidate):
            raise TypeError("candidate must be a Candidate")

        target_support = instance.target.support()
        best_node_overlap = 0.0
        for node in state.nodes:
            node_support = node.support()
            overlap = len(node_support & target_support) / max(1, len(target_support))
            best_node_overlap = max(best_node_overlap, overlap)

        n_nodes = max(1, state.num_nodes())
        values: dict[str, float] = {
            "problem_num_variables": float(len(instance.variables)),
            "target_degree": float(instance.target.degree()),
            "target_support_size": float(len(target_support)),
            "operation_budget": float(instance.op_budget),
            "state_num_nodes": float(state.num_nodes()),
            "state_num_ops": float(state.num_ops()),
            "state_remaining_budget": float(state.remaining_budget()),
            "state_best_target_support_overlap": best_node_overlap,
            "state_contains_target": float(state.contains(instance.target)),
            "action_is_add": float(candidate.action.op == "add"),
            "action_is_mul": float(candidate.action.op == "mul"),
            "action_i_norm": float(candidate.action.i / n_nodes),
            "action_j_norm": float(candidate.action.j / n_nodes),
            "action_same_operand": float(candidate.action.i == candidate.action.j),
            "tag_exact_target": _tag(candidate, "exact_target"),
            "tag_divides_target": _tag(candidate, "divides_target"),
            "tag_quotient_exists": _tag(candidate, "quotient_exists"),
            "tag_one_step_completion_add": _tag(candidate, "one_step_completion_add"),
            "tag_one_step_completion_mul": _tag(candidate, "one_step_completion_mul"),
            "tag_support_overlap": _tag(candidate, "support_overlap"),
            "tag_residual_exists": _tag(candidate, "residual_exists"),
        }
        for feature_name in CANDIDATE_FEATURE_NAMES:
            values[f"feature_{feature_name}"] = float(
                candidate.features.get(feature_name, 0.0)
            )

        return [values[name] for name in self.feature_names]


def _tag(candidate: Candidate, tag: str) -> float:
    return float(tag in candidate.source_tags)


CANDIDATE_FEATURE_NAMES = (
    "equals_target",
    "degree_result",
    "degree_target",
    "degree_gap",
    "support_size_result",
    "support_size_target",
    "support_overlap_count",
    "support_overlap_frac",
    "target_coverage_frac",
    "outside_support_count",
    "residual_exists",
    "residual_support_size",
    "residual_target_overlap",
    "divides_target",
    "quotient_exists",
    "quotient_degree",
    "quotient_support_size",
    "one_step_completion_add",
    "one_step_completion_mul",
)


FEATURE_NAMES = (
    "problem_num_variables",
    "target_degree",
    "target_support_size",
    "operation_budget",
    "state_num_nodes",
    "state_num_ops",
    "state_remaining_budget",
    "state_best_target_support_overlap",
    "state_contains_target",
    "action_is_add",
    "action_is_mul",
    "action_i_norm",
    "action_j_norm",
    "action_same_operand",
    *(f"feature_{name}" for name in CANDIDATE_FEATURE_NAMES),
    "tag_exact_target",
    "tag_divides_target",
    "tag_quotient_exists",
    "tag_one_step_completion_add",
    "tag_one_step_completion_mul",
    "tag_support_overlap",
    "tag_residual_exists",
)
