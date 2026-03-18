"""Unit tests for rex.autonomy.goal_graph — Goal, GoalGraph, topological_sort, ready_goals."""

from __future__ import annotations

import pytest

from rex.autonomy.goal_graph import (
    CyclicDependencyError,
    Goal,
    GoalGraph,
    GoalStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _goal(gid: str, depends_on: list[str] | None = None) -> Goal:
    return Goal(id=gid, description=f"Goal {gid}", depends_on=depends_on or [])


# ---------------------------------------------------------------------------
# Goal model
# ---------------------------------------------------------------------------


class TestGoalModel:
    def test_default_status_is_pending(self) -> None:
        goal = _goal("g1")
        assert goal.status == GoalStatus.PENDING

    def test_default_depends_on_is_empty(self) -> None:
        goal = _goal("g1")
        assert goal.depends_on == []

    def test_description_stored(self) -> None:
        goal = Goal(id="g1", description="Do the thing")
        assert goal.description == "Do the thing"

    def test_depends_on_stored(self) -> None:
        goal = _goal("g2", depends_on=["g1"])
        assert goal.depends_on == ["g1"]


# ---------------------------------------------------------------------------
# GoalGraph.topological_sort — linear chain
# ---------------------------------------------------------------------------


class TestTopologicalSortLinear:
    def test_single_goal_returns_itself(self) -> None:
        graph = GoalGraph([_goal("g1")])
        result = graph.topological_sort()
        assert [g.id for g in result] == ["g1"]

    def test_two_goals_no_deps_preserves_order(self) -> None:
        graph = GoalGraph([_goal("g1"), _goal("g2")])
        result = graph.topological_sort()
        assert {g.id for g in result} == {"g1", "g2"}
        assert len(result) == 2

    def test_linear_chain_sorted_correctly(self) -> None:
        """g1 → g2 → g3: result must be [g1, g2, g3]."""
        g1 = _goal("g1")
        g2 = _goal("g2", depends_on=["g1"])
        g3 = _goal("g3", depends_on=["g2"])
        graph = GoalGraph([g3, g2, g1])  # intentionally scrambled

        result = graph.topological_sort()
        ids = [g.id for g in result]

        assert ids.index("g1") < ids.index("g2")
        assert ids.index("g2") < ids.index("g3")

    def test_linear_chain_length(self) -> None:
        goals = [_goal(f"g{i}", depends_on=[f"g{i-1}"] if i > 0 else []) for i in range(5)]
        graph = GoalGraph(goals)
        assert len(graph.topological_sort()) == 5


# ---------------------------------------------------------------------------
# GoalGraph.topological_sort — parallel goals with shared dependency
# ---------------------------------------------------------------------------


class TestTopologicalSortParallel:
    def test_parallel_goals_after_shared_dep(self) -> None:
        """g1 ← g2, g1 ← g3: g1 must appear before both g2 and g3."""
        g1 = _goal("g1")
        g2 = _goal("g2", depends_on=["g1"])
        g3 = _goal("g3", depends_on=["g1"])
        graph = GoalGraph([g2, g3, g1])

        result = graph.topological_sort()
        ids = [g.id for g in result]

        assert ids.index("g1") < ids.index("g2")
        assert ids.index("g1") < ids.index("g3")
        assert len(result) == 3

    def test_diamond_dependency(self) -> None:
        """g1 → g2, g1 → g3, g2+g3 → g4."""
        g1 = _goal("g1")
        g2 = _goal("g2", depends_on=["g1"])
        g3 = _goal("g3", depends_on=["g1"])
        g4 = _goal("g4", depends_on=["g2", "g3"])
        graph = GoalGraph([g4, g2, g3, g1])

        result = graph.topological_sort()
        ids = [g.id for g in result]

        assert ids.index("g1") < ids.index("g2")
        assert ids.index("g1") < ids.index("g3")
        assert ids.index("g2") < ids.index("g4")
        assert ids.index("g3") < ids.index("g4")


# ---------------------------------------------------------------------------
# GoalGraph.topological_sort — cycle detection
# ---------------------------------------------------------------------------


class TestTopologicalSortCycles:
    def test_direct_cycle_raises(self) -> None:
        """g1 → g2 → g1."""
        g1 = _goal("g1", depends_on=["g2"])
        g2 = _goal("g2", depends_on=["g1"])
        graph = GoalGraph([g1, g2])

        with pytest.raises(CyclicDependencyError):
            graph.topological_sort()

    def test_self_loop_raises(self) -> None:
        """g1 depends on itself."""
        g1 = _goal("g1", depends_on=["g1"])
        graph = GoalGraph([g1])

        with pytest.raises(CyclicDependencyError):
            graph.topological_sort()

    def test_longer_cycle_raises(self) -> None:
        """g1 → g2 → g3 → g1."""
        g1 = _goal("g1", depends_on=["g3"])
        g2 = _goal("g2", depends_on=["g1"])
        g3 = _goal("g3", depends_on=["g2"])
        graph = GoalGraph([g1, g2, g3])

        with pytest.raises(CyclicDependencyError):
            graph.topological_sort()

    def test_error_message_mentions_cycle(self) -> None:
        g1 = _goal("g1", depends_on=["g2"])
        g2 = _goal("g2", depends_on=["g1"])
        graph = GoalGraph([g1, g2])

        with pytest.raises(CyclicDependencyError, match="cycle"):
            graph.topological_sort()


# ---------------------------------------------------------------------------
# GoalGraph.ready_goals
# ---------------------------------------------------------------------------


class TestReadyGoals:
    def test_all_pending_no_deps_all_ready(self) -> None:
        graph = GoalGraph([_goal("g1"), _goal("g2")])
        ready = graph.ready_goals()
        assert {g.id for g in ready} == {"g1", "g2"}

    def test_goal_with_incomplete_dep_not_ready(self) -> None:
        g1 = _goal("g1")
        g2 = _goal("g2", depends_on=["g1"])
        graph = GoalGraph([g1, g2])

        # g1 is pending — g2 depends on g1, so g2 is not ready
        ready = graph.ready_goals()
        assert "g2" not in {g.id for g in ready}

    def test_goal_ready_when_dep_completed(self) -> None:
        g1 = Goal(id="g1", description="d", status=GoalStatus.COMPLETED)
        g2 = _goal("g2", depends_on=["g1"])
        graph = GoalGraph([g1, g2])

        ready = graph.ready_goals()
        assert any(g.id == "g2" for g in ready)

    def test_completed_goal_not_in_ready(self) -> None:
        g1 = Goal(id="g1", description="d", status=GoalStatus.COMPLETED)
        graph = GoalGraph([g1])

        ready = graph.ready_goals()
        assert ready == []

    def test_running_goal_not_in_ready(self) -> None:
        g1 = Goal(id="g1", description="d", status=GoalStatus.RUNNING)
        graph = GoalGraph([g1])

        ready = graph.ready_goals()
        assert ready == []

    def test_failed_goal_not_in_ready(self) -> None:
        g1 = Goal(id="g1", description="d", status=GoalStatus.FAILED)
        graph = GoalGraph([g1])

        ready = graph.ready_goals()
        assert ready == []

    def test_empty_graph_returns_empty_ready(self) -> None:
        graph = GoalGraph([])
        assert graph.ready_goals() == []

    def test_parallel_goals_both_ready_when_dep_done(self) -> None:
        g1 = Goal(id="g1", description="d", status=GoalStatus.COMPLETED)
        g2 = _goal("g2", depends_on=["g1"])
        g3 = _goal("g3", depends_on=["g1"])
        graph = GoalGraph([g1, g2, g3])

        ready = graph.ready_goals()
        assert {g.id for g in ready} == {"g2", "g3"}
