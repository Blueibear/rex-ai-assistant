"""Unit tests for rex.autonomy.goal_parser — GoalParser."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from rex.autonomy.goal_graph import GoalGraph, GoalStatus
from rex.autonomy.goal_parser import GoalParser, GoalParsingError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backend(response: str) -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = response
    return backend


def _goals_response(*goals: dict) -> str:  # type: ignore[type-arg]
    """Build a valid JSON array response."""
    return json.dumps(list(goals))


def _single_goal(
    gid: str = "goal_1",
    description: str = "Do the thing",
    depends_on: list[str] | None = None,
) -> dict:  # type: ignore[type-arg]
    return {"id": gid, "description": description, "depends_on": depends_on or []}


# ---------------------------------------------------------------------------
# Single goal
# ---------------------------------------------------------------------------


class TestGoalParserSingleGoal:
    def test_single_goal_returns_one_node_graph(self) -> None:
        resp = _goals_response(_single_goal())
        parser = GoalParser(backend=_mock_backend(resp))

        result = parser.parse("Do the thing")

        assert len(result.goals) == 1

    def test_single_goal_is_goalgraph(self) -> None:
        resp = _goals_response(_single_goal())
        parser = GoalParser(backend=_mock_backend(resp))

        result = parser.parse("Do the thing")

        assert isinstance(result, GoalGraph)

    def test_single_goal_id_preserved(self) -> None:
        resp = _goals_response(_single_goal(gid="my_goal"))
        parser = GoalParser(backend=_mock_backend(resp))

        result = parser.parse("Do the thing")

        assert result.goals[0].id == "my_goal"

    def test_single_goal_description_preserved(self) -> None:
        resp = _goals_response(_single_goal(description="Book a flight to Paris"))
        parser = GoalParser(backend=_mock_backend(resp))

        result = parser.parse("Book a flight to Paris")

        assert result.goals[0].description == "Book a flight to Paris"

    def test_single_goal_status_is_pending(self) -> None:
        resp = _goals_response(_single_goal())
        parser = GoalParser(backend=_mock_backend(resp))

        result = parser.parse("Do the thing")

        assert result.goals[0].status == GoalStatus.PENDING

    def test_single_goal_no_deps(self) -> None:
        resp = _goals_response(_single_goal())
        parser = GoalParser(backend=_mock_backend(resp))

        result = parser.parse("Do the thing")

        assert result.goals[0].depends_on == []


# ---------------------------------------------------------------------------
# Multiple goals
# ---------------------------------------------------------------------------


class TestGoalParserMultipleGoals:
    def test_two_goals_returns_two_node_graph(self) -> None:
        resp = _goals_response(
            _single_goal("g1", "First goal"),
            _single_goal("g2", "Second goal"),
        )
        parser = GoalParser(backend=_mock_backend(resp))

        result = parser.parse("Do two things")

        assert len(result.goals) == 2

    def test_dependencies_preserved(self) -> None:
        resp = _goals_response(
            _single_goal("g1", "First"),
            _single_goal("g2", "Second", depends_on=["g1"]),
        )
        parser = GoalParser(backend=_mock_backend(resp))

        result = parser.parse("Do two things in order")

        g2 = next(g for g in result.goals if g.id == "g2")
        assert "g1" in g2.depends_on

    def test_three_goals_all_ids_present(self) -> None:
        resp = _goals_response(
            _single_goal("a", "A"), _single_goal("b", "B"), _single_goal("c", "C")
        )
        parser = GoalParser(backend=_mock_backend(resp))

        result = parser.parse("Three goals")

        assert {g.id for g in result.goals} == {"a", "b", "c"}

    def test_topological_sort_works_on_result(self) -> None:
        resp = _goals_response(
            _single_goal("g1", "First"),
            _single_goal("g2", "Second", depends_on=["g1"]),
        )
        parser = GoalParser(backend=_mock_backend(resp))

        result = parser.parse("Ordered goals")
        sorted_goals = result.topological_sort()

        ids = [g.id for g in sorted_goals]
        assert ids.index("g1") < ids.index("g2")


# ---------------------------------------------------------------------------
# Code-fence stripping
# ---------------------------------------------------------------------------


class TestGoalParserCodeFences:
    def test_strips_json_code_fences(self) -> None:
        raw = "```json\n" + _goals_response(_single_goal()) + "\n```"
        parser = GoalParser(backend=_mock_backend(raw))

        result = parser.parse("input")

        assert len(result.goals) == 1

    def test_strips_plain_code_fences(self) -> None:
        raw = "```\n" + _goals_response(_single_goal()) + "\n```"
        parser = GoalParser(backend=_mock_backend(raw))

        result = parser.parse("input")

        assert len(result.goals) == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestGoalParserErrors:
    def test_invalid_json_raises(self) -> None:
        parser = GoalParser(backend=_mock_backend("not json"))

        with pytest.raises(GoalParsingError, match="not valid JSON"):
            parser.parse("input")

    def test_non_list_response_raises(self) -> None:
        parser = GoalParser(backend=_mock_backend('{"id": "g1"}'))

        with pytest.raises(GoalParsingError, match="expected a JSON array"):
            parser.parse("input")

    def test_empty_list_raises(self) -> None:
        parser = GoalParser(backend=_mock_backend("[]"))

        with pytest.raises(GoalParsingError, match="empty goal list"):
            parser.parse("input")

    def test_missing_id_raises(self) -> None:
        resp = json.dumps([{"description": "No id here", "depends_on": []}])
        parser = GoalParser(backend=_mock_backend(resp))

        with pytest.raises(GoalParsingError, match="missing 'id'"):
            parser.parse("input")

    def test_backend_called_once(self) -> None:
        backend = _mock_backend(_goals_response(_single_goal()))
        parser = GoalParser(backend=backend)

        parser.parse("Do something")

        backend.generate.assert_called_once()

    def test_prompt_contains_user_input(self) -> None:
        backend = _mock_backend(_goals_response(_single_goal()))
        parser = GoalParser(backend=backend)

        parser.parse("Book flight and hotel for my trip")

        prompt = backend.generate.call_args.kwargs["messages"][0]["content"]
        assert "Book flight and hotel for my trip" in prompt
