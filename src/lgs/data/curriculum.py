"""Minimal fixed curriculum for bootstrap training."""

from __future__ import annotations

from dataclasses import dataclass

from lgs.env.problem_instance import ProblemInstance


@dataclass
class FixedCurriculum:
    train_instances: list[ProblemInstance]
    validation_instances: list[ProblemInstance]

    def sample_training_instances(self, round_idx: int) -> list[ProblemInstance]:
        if type(round_idx) is not int or round_idx < 0:
            raise ValueError("round_idx must be a non-negative int")
        return list(self.train_instances)

    def validation_set(self) -> list[ProblemInstance]:
        return list(self.validation_instances)


# TODO: add an adaptive curriculum keyed by solve rate once the bootstrap loop
# has enough non-tiny families to make progression meaningful.
