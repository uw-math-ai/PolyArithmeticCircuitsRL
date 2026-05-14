"""Exact circuit search state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from lgs.env.action import Action
from lgs.env.problem_instance import ProblemInstance
from lgs.poly.fast_poly import Polynomial, PolynomialDomainError
from lgs.poly.poly_utils import assert_degree_cap, require_same_domain

Parent = tuple[int, int, str]


class InvalidCircuitStateError(ValueError):
    """Raised when a CircuitState is internally inconsistent."""


class InvalidActionError(ValueError):
    """Raised when an action cannot be applied to the current state."""


class BudgetExceededError(InvalidActionError):
    """Raised when applying an action would exceed the operation budget."""


@dataclass(frozen=True)
class CircuitState:
    nodes: tuple[Polynomial, ...]
    actions: tuple[Action, ...] = ()
    parents: tuple[Parent | None, ...] = ()
    node_keys: frozenset[Hashable] | None = None
    op_budget: int = 0
    degree_cap: int | None = None

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        if not nodes:
            raise InvalidCircuitStateError("CircuitState requires at least one node")
        for node in nodes:
            if not isinstance(node, Polynomial):
                raise PolynomialDomainError("all circuit nodes must be Polynomials")
        require_same_domain(*nodes)
        object.__setattr__(self, "nodes", nodes)

        actions = tuple(self.actions)
        for action in actions:
            if not isinstance(action, Action):
                raise InvalidCircuitStateError("actions must contain only Action objects")
        object.__setattr__(self, "actions", actions)

        parents = tuple(self.parents)
        if not parents:
            if actions:
                raise InvalidCircuitStateError("non-initial states require parent metadata")
            parents = (None,) * len(nodes)
        if len(parents) != len(nodes):
            raise InvalidCircuitStateError("parents length must match nodes length")

        normalized_parents: list[Parent | None] = []
        action_index = 0
        for node_index, parent in enumerate(parents):
            if parent is None:
                normalized_parents.append(None)
                continue
            if len(parent) != 3:
                raise InvalidCircuitStateError("parent metadata must be (i, j, op)")
            i, j, op = parent
            action = Action(op, i, j)
            if action.i >= node_index or action.j >= node_index:
                raise InvalidCircuitStateError(
                    "parent metadata must reference earlier nodes only"
                )
            if action_index >= len(actions) or actions[action_index] != action:
                raise InvalidCircuitStateError("actions do not match parent metadata")
            normalized_parents.append((action.i, action.j, action.op))
            action_index += 1
        if action_index != len(actions):
            raise InvalidCircuitStateError("not every action has a parent entry")
        object.__setattr__(self, "parents", tuple(normalized_parents))

        if type(self.op_budget) is not int or self.op_budget < 0:
            raise InvalidCircuitStateError("op_budget must be a non-negative int")
        if len(actions) > self.op_budget:
            raise InvalidCircuitStateError("state already exceeds op_budget")

        node_degree_cap = nodes[0].degree_cap
        if self.degree_cap is None:
            object.__setattr__(self, "degree_cap", node_degree_cap)
        elif type(self.degree_cap) is not int or self.degree_cap < 0:
            raise InvalidCircuitStateError("degree_cap must be None or a non-negative int")
        elif node_degree_cap != self.degree_cap:
            raise InvalidCircuitStateError(
                f"state degree_cap {self.degree_cap} does not match node domain "
                f"{node_degree_cap}"
            )
        if self.degree_cap is not None:
            for node in nodes:
                assert_degree_cap(node, self.degree_cap)

        computed_keys = frozenset(node.key() for node in nodes)
        if self.node_keys is None:
            object.__setattr__(self, "node_keys", computed_keys)
        else:
            node_keys = frozenset(self.node_keys)
            if node_keys != computed_keys:
                raise InvalidCircuitStateError("node_keys do not match circuit nodes")
            object.__setattr__(self, "node_keys", node_keys)

    @classmethod
    def initial(cls, instance: ProblemInstance) -> CircuitState:
        if not isinstance(instance, ProblemInstance):
            raise InvalidCircuitStateError("initial state requires a ProblemInstance")
        nodes = instance.base_polynomials()
        return cls(
            nodes=nodes,
            actions=(),
            parents=(None,) * len(nodes),
            op_budget=instance.op_budget,
            degree_cap=instance.degree_cap,
        )

    def apply(self, action: Action) -> CircuitState:
        if not isinstance(action, Action):
            raise InvalidActionError("apply requires an Action")
        if self.remaining_budget() <= 0:
            raise BudgetExceededError("operation budget exhausted")
        self._validate_node_index(action.i)
        self._validate_node_index(action.j)

        left = self.nodes[action.i]
        right = self.nodes[action.j]
        if action.op == "add":
            result = left + right
        elif action.op == "mul":
            result = left * right
        else:
            raise InvalidActionError(f"unsupported action op {action.op!r}")

        if self.degree_cap is not None:
            assert_degree_cap(result, self.degree_cap)

        return CircuitState(
            nodes=self.nodes + (result,),
            actions=self.actions + (action,),
            parents=self.parents + ((action.i, action.j, action.op),),
            node_keys=frozenset((*self.node_keys, result.key())),
            op_budget=self.op_budget,
            degree_cap=self.degree_cap,
        )

    def contains(self, target: Polynomial) -> bool:
        if not isinstance(target, Polynomial):
            raise PolynomialDomainError("target must be a Polynomial")
        require_same_domain(self.nodes[0], target)
        return target.key() in self.node_keys

    def get_node(self, index: int) -> Polynomial:
        self._validate_node_index(index)
        return self.nodes[index]

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_ops(self) -> int:
        return len(self.actions)

    def remaining_budget(self) -> int:
        return self.op_budget - len(self.actions)

    def node_key(self, index: int) -> Hashable:
        return self.get_node(index).key()

    def _validate_node_index(self, index: int) -> None:
        if type(index) is not int:
            raise InvalidActionError("node index must be an int")
        if index < 0 or index >= len(self.nodes):
            raise InvalidActionError(
                f"node index {index} out of range for {len(self.nodes)} nodes"
            )
