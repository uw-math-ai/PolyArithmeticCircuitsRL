"""Tests for adaptive PPO curriculum bookkeeping."""

from types import SimpleNamespace

from src.algorithms.ppo_mcts import PPOMCTSTrainer
from src.config import Config


def test_max_build_complexity_controls_episode_capacity_separately():
    config = Config(n_variables=2, max_complexity=3, max_build_complexity=5)

    assert config.max_complexity == 3
    assert config.effective_max_build_complexity == 5
    assert config.max_nodes == 8  # x0, x1, 1 plus five operation nodes.
    assert config.max_actions == 72


class _CurriculumHarness(SimpleNamespace):
    def _log(self, _msg: str) -> None:
        return None


def _harness(
    *,
    current_complexity: int = 2,
    window: int = 2,
    min_dwell: int = 1,
    backoff_threshold: float = 0.4,
    backoff_patience: int = 0,
    successes=None,
) -> _CurriculumHarness:
    config = Config(
        curriculum_enabled=True,
        starting_complexity=2,
        max_complexity=4,
        advance_threshold=0.7,
        backoff_threshold=backoff_threshold,
        curriculum_window=window,
        curriculum_min_dwell_iterations=min_dwell,
        curriculum_backoff_patience_iterations=backoff_patience,
    )
    return _CurriculumHarness(
        config=config,
        current_complexity=current_complexity,
        success_history=list(successes or []),
        dwell_iterations_at_level=0,
        window_success_rate=0.0,
        backoff_patience_counter=0,
    )


def test_curriculum_dwell_counts_outer_train_iterations_before_advancing():
    trainer = _harness(successes=[True, True])

    PPOMCTSTrainer._maybe_advance_curriculum(trainer)
    assert trainer.current_complexity == 2
    assert trainer.dwell_iterations_at_level == 0

    # The train loop increments this once per completed outer PPO iteration.
    trainer.dwell_iterations_at_level += 1
    PPOMCTSTrainer._maybe_advance_curriculum(trainer)
    assert trainer.current_complexity == 3
    assert trainer.dwell_iterations_at_level == 0
    assert trainer.success_history == []


def test_curriculum_backoff_respects_min_dwell_symmetrically():
    trainer = _harness(current_complexity=3, successes=[False, False])

    PPOMCTSTrainer._maybe_advance_curriculum(trainer)
    assert trainer.current_complexity == 3

    trainer.dwell_iterations_at_level += 1
    PPOMCTSTrainer._maybe_advance_curriculum(trainer)
    assert trainer.current_complexity == 2
    assert trainer.dwell_iterations_at_level == 0
    assert trainer.success_history == []


def test_curriculum_window_controls_decision_and_logged_rate():
    trainer = _harness(window=3, successes=[True, True])
    trainer.dwell_iterations_at_level = 10

    PPOMCTSTrainer._maybe_advance_curriculum(trainer)
    assert trainer.current_complexity == 2
    assert trainer.window_success_rate == 1.0

    trainer.success_history.append(True)
    PPOMCTSTrainer._maybe_advance_curriculum(trainer)
    assert trainer.current_complexity == 3
    assert trainer.window_success_rate == 0.0


def test_negative_backoff_threshold_disables_backoff():
    trainer = _harness(
        current_complexity=3,
        backoff_threshold=-1.0,
        successes=[False, False],
    )
    trainer.dwell_iterations_at_level = 10

    PPOMCTSTrainer._maybe_advance_curriculum(trainer)

    assert trainer.current_complexity == 3
    assert trainer.window_success_rate == 0.0


def test_backoff_patience_requires_consecutive_bad_windows():
    trainer = _harness(
        current_complexity=3,
        backoff_threshold=0.4,
        backoff_patience=2,
        successes=[False, False],
    )
    trainer.dwell_iterations_at_level = 10

    PPOMCTSTrainer._maybe_advance_curriculum(trainer)
    assert trainer.current_complexity == 3
    assert trainer.backoff_patience_counter == 1

    PPOMCTSTrainer._maybe_advance_curriculum(trainer)
    assert trainer.current_complexity == 2
    assert trainer.backoff_patience_counter == 0
