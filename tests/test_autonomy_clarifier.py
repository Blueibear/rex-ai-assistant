"""Unit tests for rex.autonomy.clarifier — Clarifier."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from rex.autonomy.clarifier import Clarifier
from rex.autonomy.goal_graph import Goal, GoalGraph, GoalStatus
from rex.autonomy.runner import execute_goal_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backend(response: str = "Did you mean X or Y?") -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = response
    return backend


def _goal(gid: str, ambiguous: bool = False) -> Goal:
    return Goal(id=gid, description=f"Do {gid}", ambiguous=ambiguous)


# ---------------------------------------------------------------------------
# needs_clarification
# ---------------------------------------------------------------------------


class TestNeedsClarification:
    def test_returns_true_when_goal_is_ambiguous(self) -> None:
        clarifier = Clarifier(backend=_mock_backend())
        goal = _goal("g1", ambiguous=True)
        assert clarifier.needs_clarification(goal) is True

    def test_returns_false_when_goal_is_not_ambiguous(self) -> None:
        clarifier = Clarifier(backend=_mock_backend())
        goal = _goal("g1", ambiguous=False)
        assert clarifier.needs_clarification(goal) is False

    def test_default_goal_is_not_ambiguous(self) -> None:
        clarifier = Clarifier(backend=_mock_backend())
        goal = Goal(id="g1", description="Clear goal")
        assert clarifier.needs_clarification(goal) is False


# ---------------------------------------------------------------------------
# generate_question
# ---------------------------------------------------------------------------


class TestGenerateQuestion:
    def test_returns_llm_response(self) -> None:
        backend = _mock_backend("Would you like X or Y?")
        clarifier = Clarifier(backend=backend)
        goal = _goal("g1", ambiguous=True)

        result = clarifier.generate_question(goal)

        assert result == "Would you like X or Y?"

    def test_strips_whitespace_from_response(self) -> None:
        backend = _mock_backend("  Do you want A?  \n")
        clarifier = Clarifier(backend=backend)
        goal = _goal("g1", ambiguous=True)

        result = clarifier.generate_question(goal)

        assert result == "Do you want A?"

    def test_calls_backend_generate_once(self) -> None:
        backend = _mock_backend()
        clarifier = Clarifier(backend=backend)
        goal = _goal("g1", ambiguous=True)

        clarifier.generate_question(goal)

        backend.generate.assert_called_once()

    def test_prompt_contains_goal_description(self) -> None:
        backend = _mock_backend()
        clarifier = Clarifier(backend=backend)
        goal = Goal(id="g1", description="Send an email to the team", ambiguous=True)

        clarifier.generate_question(goal)

        prompt = backend.generate.call_args.kwargs["messages"][0]["content"]
        assert "Send an email to the team" in prompt

    def test_result_is_non_empty_string(self) -> None:
        backend = _mock_backend("What time zone?")
        clarifier = Clarifier(backend=backend)
        goal = _goal("g1", ambiguous=True)

        result = clarifier.generate_question(goal)

        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Runner integration — clarifier called for ambiguous goals
# ---------------------------------------------------------------------------


def _noop_planner(tool: str = "noop") -> MagicMock:
    import json

    from rex.autonomy.llm_planner import LLMPlanner

    response = json.dumps([{"tool": tool, "args": {}, "description": "call"}])
    backend = MagicMock()
    backend.generate.return_value = response
    return LLMPlanner(tools=[], backend=backend)


def _noop(**_: object) -> str:
    return "done"


class TestRunnerClarifierIntegration:
    def test_on_question_called_for_ambiguous_goal(self) -> None:
        clarifier_backend = _mock_backend("Which format do you prefer?")
        clarifier = Clarifier(backend=clarifier_backend)

        questions: list[tuple[str, str]] = []

        def on_question(goal_id: str, question: str) -> None:
            questions.append((goal_id, question))

        graph = GoalGraph([_goal("g1", ambiguous=True)])
        planner = _noop_planner()

        execute_goal_graph(
            graph,
            {"noop": _noop},
            planner=planner,
            clarifier=clarifier,
            on_question=on_question,
        )

        assert len(questions) == 1
        assert questions[0][0] == "g1"
        assert questions[0][1] == "Which format do you prefer?"

    def test_on_question_not_called_for_unambiguous_goal(self) -> None:
        clarifier = Clarifier(backend=_mock_backend())
        questions: list[str] = []

        def on_question(goal_id: str, question: str) -> None:
            questions.append(question)

        graph = GoalGraph([_goal("g1", ambiguous=False)])
        planner = _noop_planner()

        execute_goal_graph(
            graph,
            {"noop": _noop},
            planner=planner,
            clarifier=clarifier,
            on_question=on_question,
        )

        assert questions == []

    def test_goal_still_executes_after_clarification(self) -> None:
        clarifier = Clarifier(backend=_mock_backend("Which option?"))
        graph = GoalGraph([_goal("g1", ambiguous=True)])
        planner = _noop_planner()

        execute_goal_graph(
            graph,
            {"noop": _noop},
            planner=planner,
            clarifier=clarifier,
        )

        assert graph.goals[0].status == GoalStatus.COMPLETED

    def test_clarifier_failure_does_not_prevent_execution(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bad_backend = MagicMock()
        bad_backend.generate.side_effect = RuntimeError("LLM down")
        clarifier = Clarifier(backend=bad_backend)

        graph = GoalGraph([_goal("g1", ambiguous=True)])
        planner = _noop_planner()

        with caplog.at_level(logging.WARNING, logger="rex.autonomy.runner"):
            execute_goal_graph(
                graph,
                {"noop": _noop},
                planner=planner,
                clarifier=clarifier,
            )

        # Goal should still be attempted (not skipped due to clarifier failure)
        assert graph.goals[0].status == GoalStatus.COMPLETED
        assert any("clarifier failed" in r.message for r in caplog.records)

    def test_default_logging_when_no_on_question_callback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        clarifier = Clarifier(backend=_mock_backend("What do you mean?"))
        graph = GoalGraph([_goal("g1", ambiguous=True)])
        planner = _noop_planner()

        with caplog.at_level(logging.INFO, logger="rex.autonomy.runner"):
            execute_goal_graph(
                graph,
                {"noop": _noop},
                planner=planner,
                clarifier=clarifier,
            )

        assert any(
            "clarification needed" in r.message and "g1" in r.message for r in caplog.records
        )
