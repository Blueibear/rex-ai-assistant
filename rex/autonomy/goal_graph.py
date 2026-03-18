"""GoalGraph data structure for multi-goal planning in the Rex autonomy engine.

Provides:
- :class:`GoalStatus` — lifecycle states for a :class:`Goal`.
- :class:`Goal` — a Pydantic model representing a single planning goal with
  optional dependencies on other goals.
- :class:`GoalGraph` — an ordered collection of :class:`Goal` objects that
  supports topological ordering and readiness queries.
- :exc:`CyclicDependencyError` — raised when a dependency cycle is detected.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class GoalStatus(str, Enum):
    """Lifecycle status of a :class:`Goal`."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Goal model
# ---------------------------------------------------------------------------


class Goal(BaseModel):
    """A single planning goal, optionally dependent on other goals.

    Args:
        id: Unique identifier for this goal.
        description: Natural-language description of what the goal achieves.
        depends_on: IDs of goals that must be completed before this goal
            can run.  Defaults to an empty list (no dependencies).
        status: Current lifecycle status.  Defaults to ``PENDING``.
        ambiguous: Whether the goal was flagged as ambiguous by the parser.
            When ``True`` the runner should seek clarification before planning.
            Defaults to ``False``.
    """

    id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    ambiguous: bool = False


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class CyclicDependencyError(Exception):
    """Raised when :class:`GoalGraph` contains a dependency cycle."""


# ---------------------------------------------------------------------------
# GoalGraph
# ---------------------------------------------------------------------------


class GoalGraph:
    """An ordered collection of :class:`Goal` objects with dependency support.

    Args:
        goals: The goals that form the graph.  Goal IDs must be unique.
    """

    def __init__(self, goals: list[Goal]) -> None:
        self._goals: list[Goal] = list(goals)
        self._by_id: dict[str, Goal] = {g.id: g for g in self._goals}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def goals(self) -> list[Goal]:
        """All goals in this graph (original insertion order)."""
        return list(self._goals)

    def topological_sort(self) -> list[Goal]:
        """Return goals in a valid execution order (dependencies first).

        Uses Kahn's algorithm (BFS-based topological sort).

        Returns:
            A list of :class:`Goal` objects ordered so that every goal
            appears after all of its dependencies.

        Raises:
            CyclicDependencyError: If the dependency graph contains a cycle.
        """
        in_degree: dict[str, int] = {g.id: 0 for g in self._goals}
        dependents: dict[str, list[str]] = {g.id: [] for g in self._goals}

        for goal in self._goals:
            for dep_id in goal.depends_on:
                if dep_id in in_degree:
                    in_degree[goal.id] += 1
                    dependents[dep_id].append(goal.id)
                # Dependencies that reference unknown IDs are ignored.

        queue: list[str] = [gid for gid, deg in in_degree.items() if deg == 0]
        result: list[Goal] = []

        while queue:
            current_id = queue.pop(0)
            result.append(self._by_id[current_id])
            for dependent_id in dependents[current_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        if len(result) != len(self._goals):
            raise CyclicDependencyError(
                "GoalGraph contains a dependency cycle — topological sort is not possible."
            )

        return result

    def ready_goals(self) -> list[Goal]:
        """Return goals that are ready to execute.

        A goal is *ready* when its status is ``PENDING`` and all of its
        dependency goals have status ``COMPLETED``.

        Returns:
            A list of :class:`Goal` objects that can be started immediately.
        """
        ready: list[Goal] = []
        for goal in self._goals:
            if goal.status != GoalStatus.PENDING:
                continue
            deps_done = all(
                self._by_id.get(dep_id, goal).status == GoalStatus.COMPLETED
                for dep_id in goal.depends_on
                if dep_id in self._by_id
            )
            if deps_done:
                ready.append(goal)
        return ready


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CyclicDependencyError",
    "Goal",
    "GoalGraph",
    "GoalStatus",
]
