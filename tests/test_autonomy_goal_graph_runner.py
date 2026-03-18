"""Integration tests for US-231: GoalGraph execution in autonomy runner."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from rex.autonomy.goal_graph import Goal, GoalGraph, GoalStatus
from rex.autonomy.goal_parser import GoalParser
from rex.autonomy.llm_planner import LLMPlanner
from rex.autonomy.runner import execute_goal_graph, run_goals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_planner(tool: str = "noop") -> MagicMock:
    """Planner that always returns a single-step plan for the given tool."""
    response = json.dumps([{"tool": tool, "args": {}, "description": f"Call {tool}"}])
    backend = MagicMock()
    backend.generate.return_value = response
    return LLMPlanner(tools=[], backend=backend)


def _goal(gid: str, desc: str = "", depends_on: list[str] | None = None) -> Goal:
    return Goal(id=gid, description=desc or f"Goal {gid}", depends_on=depends_on or [])


def _noop(**_: object) -> str:
    return "done"


def _fail(**_: object) -> str:
    raise RuntimeError("tool failed")


# ---------------------------------------------------------------------------
# execute_goal_graph — basic execution
# ---------------------------------------------------------------------------


class TestExecuteGoalGraph:
    def test_single_goal_completed_on_success(self) -> None:
        graph = GoalGraph([_goal("g1")])
        planner = _mock_planner()

        execute_goal_graph(graph, {"noop": _noop}, planner=planner)

        assert graph.goals[0].status == GoalStatus.COMPLETED

    def test_single_goal_failed_on_tool_failure(self) -> None:
        graph = GoalGraph([_goal("g1")])
        planner = _mock_planner()

        execute_goal_graph(graph, {"noop": _fail}, planner=planner)

        assert graph.goals[0].status == GoalStatus.FAILED

    def test_two_sequential_goals_both_completed(self) -> None:
        g1 = _goal("g1")
        g2 = _goal("g2", depends_on=["g1"])
        graph = GoalGraph([g1, g2])
        planner = _mock_planner()

        execute_goal_graph(graph, {"noop": _noop}, planner=planner)

        assert graph.goals[0].status == GoalStatus.COMPLETED
        assert graph.goals[1].status == GoalStatus.COMPLETED

    def test_dependent_goal_skipped_when_dep_fails(self) -> None:
        g1 = _goal("g1")
        g2 = _goal("g2", depends_on=["g1"])
        graph = GoalGraph([g1, g2])
        planner = _mock_planner()

        execute_goal_graph(graph, {"noop": _fail}, planner=planner)

        assert graph.goals[0].status == GoalStatus.FAILED
        assert graph.goals[1].status == GoalStatus.SKIPPED

    def test_returns_graph(self) -> None:
        graph = GoalGraph([_goal("g1")])
        planner = _mock_planner()

        result = execute_goal_graph(graph, {"noop": _noop}, planner=planner)

        assert result is graph

    def test_independent_goals_both_run(self) -> None:
        g1 = _goal("g1")
        g2 = _goal("g2")  # no deps
        graph = GoalGraph([g1, g2])
        planner = _mock_planner()

        execute_goal_graph(graph, {"noop": _noop}, planner=planner)

        assert g1.status == GoalStatus.COMPLETED
        assert g2.status == GoalStatus.COMPLETED

    def test_skipped_dep_causes_downstream_skip(self) -> None:
        """g1 fails → g2 skipped → g3 skipped."""
        g1 = _goal("g1")
        g2 = _goal("g2", depends_on=["g1"])
        g3 = _goal("g3", depends_on=["g2"])
        graph = GoalGraph([g1, g2, g3])
        planner = _mock_planner()

        execute_goal_graph(graph, {"noop": _fail}, planner=planner)

        assert g1.status == GoalStatus.FAILED
        assert g2.status == GoalStatus.SKIPPED
        assert g3.status == GoalStatus.SKIPPED


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------


class TestGoalGraphLogging:
    def test_progress_logged_per_goal(self, caplog: pytest.LogCaptureFixture) -> None:
        g1 = _goal("g1", "First task")
        g2 = _goal("g2", "Second task", depends_on=["g1"])
        graph = GoalGraph([g1, g2])
        planner = _mock_planner()

        with caplog.at_level(logging.INFO, logger="rex.autonomy.runner"):
            execute_goal_graph(graph, {"noop": _noop}, planner=planner)

        messages = [r.message for r in caplog.records]
        assert any("Executing goal g1" in m and "1/2" in m for m in messages)
        assert any("Executing goal g2" in m and "2/2" in m for m in messages)

    def test_skipped_goal_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        g1 = _goal("g1")
        g2 = _goal("g2", depends_on=["g1"])
        graph = GoalGraph([g1, g2])
        planner = _mock_planner()

        with caplog.at_level(logging.INFO, logger="rex.autonomy.runner"):
            execute_goal_graph(graph, {"noop": _fail}, planner=planner)

        assert any("skipped" in r.message and "g2" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# run_goals — parser integration smoke test
# ---------------------------------------------------------------------------


class TestRunGoals:
    def test_run_goals_two_sequential_goals(self) -> None:
        """Smoke test: two goals parsed → executed → both completed."""
        parser_response = json.dumps(
            [
                {"id": "g1", "description": "Step one", "depends_on": []},
                {"id": "g2", "description": "Step two", "depends_on": ["g1"]},
            ]
        )
        parser_backend = MagicMock()
        parser_backend.generate.return_value = parser_response
        goal_parser = GoalParser(backend=parser_backend)

        plan_response = json.dumps([{"tool": "noop", "args": {}, "description": "Call noop"}])
        plan_backend = MagicMock()
        plan_backend.generate.return_value = plan_response

        with MagicMock() as _:
            from unittest.mock import patch

            with patch("rex.autonomy.runner.create_planner") as mock_create:
                mock_create.return_value = LLMPlanner(tools=[], backend=plan_backend)
                result = run_goals(
                    "Do step one then step two",
                    {"noop": _noop},
                    goal_parser=goal_parser,
                )

        assert isinstance(result, GoalGraph)
        assert all(g.status == GoalStatus.COMPLETED for g in result.goals)
