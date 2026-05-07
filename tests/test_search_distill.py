from random import Random

from decomp_rl.andor_search import AndOrSearch
from decomp_rl.baseline_cost import BaselineCostModel
from decomp_rl.family_generators import planted_factorable_example
from decomp_rl.model import HeuristicPolicyValueModel
from decomp_rl.train_search_distill import distill_targets


def test_distill_targets_supports_fresh_search_per_target_and_progress():
    rng = Random(0)
    baseline_model = BaselineCostModel()
    search = AndOrSearch(
        baseline_model=baseline_model,
        model=HeuristicPolicyValueModel(baseline_model),
    )
    targets = [planted_factorable_example(rng, 3, ("x", "y")).target for _ in range(3)]
    progress = []

    distilled = distill_targets(
        targets,
        search,
        fresh_search_per_target=True,
        retry_failures=1,
        progress_callback=progress.append,
    )

    assert len(distilled) == 3
    assert [payload["status"] for payload in progress if payload["status"] == "done"] == ["done", "done", "done"]
