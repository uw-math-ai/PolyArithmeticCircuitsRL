"""Deterministic symbolic heuristic scores for candidate transitions."""

from __future__ import annotations


def score_tier1(features: dict[str, float]) -> float:
    score = 0.0
    score += 1000.0 * features["equals_target"]
    score += 80.0 * features["residual_exists"]
    score += 5.0 * features["support_overlap_count"]
    score += 5.0 * features["target_coverage_frac"]
    score -= 1.0 * features["outside_support_count"]
    score -= 0.25 * max(0.0, features["degree_gap"])
    score -= 0.05 * features["support_size_result"]
    return score


def score_tier2(features: dict[str, float]) -> float:
    score = 0.0
    score += 250.0 * features.get("one_step_completion_add", 0.0)
    score += 250.0 * features.get("one_step_completion_mul", 0.0)
    score += 80.0 * features.get("quotient_exists", 0.0)
    score += 40.0 * features.get("divides_target", 0.0)
    return score
