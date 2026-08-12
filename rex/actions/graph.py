"""Minimal validated action dependency graph for safe execution planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from rex.tools.execution import ToolOperation


class InvalidActionGraph(ValueError):
    """Raised when an action graph is structurally unsafe or ambiguous."""


@dataclass(frozen=True)
class ActionNode:
    """One planned action plus the metadata needed for safe scheduling."""

    action_id: str
    tool_name: str
    args: Mapping[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    operation: ToolOperation = ToolOperation.READ
    authorized: bool = True
    confirmation_required: bool = False
    verification_required: bool = False
    postcondition: str | None = None
    conflict_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        action_id = self.action_id.strip()
        tool_name = self.tool_name.strip()
        if not action_id:
            raise InvalidActionGraph("action_id is required")
        if not tool_name:
            raise InvalidActionGraph("tool_name is required")
        if action_id in self.dependencies:
            raise InvalidActionGraph(f"action {action_id!r} cannot depend on itself")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "operation", ToolOperation(self.operation))
        object.__setattr__(self, "args", MappingProxyType(dict(self.args)))
        object.__setattr__(self, "dependencies", tuple(dict.fromkeys(self.dependencies)))
        object.__setattr__(self, "conflict_keys", tuple(dict.fromkeys(self.conflict_keys)))
        if self.postcondition and not self.verification_required:
            raise InvalidActionGraph("postcondition requires verification_required")


@dataclass(frozen=True)
class ActionGraph:
    """A deterministic DAG of planned actions."""

    plan_id: str
    nodes: tuple[ActionNode, ...]

    def __post_init__(self) -> None:
        plan_id = self.plan_id.strip()
        if not plan_id:
            raise InvalidActionGraph("plan_id is required")
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "nodes", tuple(self.nodes))
        by_id = {node.action_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise InvalidActionGraph("action IDs must be unique")
        for node in self.nodes:
            for dependency in node.dependencies:
                if dependency not in by_id:
                    raise InvalidActionGraph(
                        f"unknown dependency {dependency!r} for action {node.action_id!r}"
                    )
        self._assert_acyclic(by_id)

    def node(self, action_id: str) -> ActionNode:
        for node in self.nodes:
            if node.action_id == action_id:
                return node
        raise KeyError(action_id)

    def dependencies_of(self, action_id: str) -> tuple[str, ...]:
        return self.node(action_id).dependencies

    @staticmethod
    def _assert_acyclic(by_id: dict[str, ActionNode]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(action_id: str) -> None:
            if action_id in visiting:
                raise InvalidActionGraph("action dependency cycle detected")
            if action_id in visited:
                return
            visiting.add(action_id)
            for dependency in by_id[action_id].dependencies:
                visit(dependency)
            visiting.remove(action_id)
            visited.add(action_id)

        for action_id in by_id:
            visit(action_id)


__all__ = ["ActionGraph", "ActionNode", "InvalidActionGraph"]
