"""Circuit construction actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActionOp = Literal["add", "mul"]


@dataclass(frozen=True)
class Action:
    """A canonical commutative arithmetic action."""

    op: ActionOp
    i: int
    j: int

    @classmethod
    def make(cls, op: ActionOp, i: int, j: int) -> "Action":
        return cls(op, i, j)

    def __post_init__(self) -> None:
        if self.op not in ("add", "mul"):
            raise ValueError(f"unsupported action op {self.op!r}")
        if type(self.i) is not int or type(self.j) is not int:
            raise ValueError("action indices must be ints")
        if self.i < 0 or self.j < 0:
            raise ValueError("action indices must be non-negative")
        if self.i > self.j:
            i, j = self.j, self.i
            object.__setattr__(self, "i", i)
            object.__setattr__(self, "j", j)
