"""Unit tests for rex.autonomy.llm_planner.LLMPlanner.

All tests mock the AI backend so no network access is required.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rex.autonomy.llm_planner import LLMPlanner, PlanningError, ToolDefinition
from rex.autonomy.models import Plan, PlanStep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(response: str) -> MagicMock:
    """Return a mock LLMBackend whose generate() returns *response*."""
    backend = MagicMock()
    backend.generate.return_value = response
    return backend


def _tool(name: str = "web_search", desc: str = "Search the web") -> ToolDefinition:
    return ToolDefinition(name=name, description=desc)


def _valid_json_response(steps: list[dict[str, Any]]) -> str:
    return json.dumps(steps)


# ---------------------------------------------------------------------------
# Valid JSON response → correct Plan with N steps
# ---------------------------------------------------------------------------


class TestValidResponse:
    def test_single_step_plan(self) -> None:
        payload = [
            {
                "tool": "web_search",
                "args": {"query": "weather"},
                "description": "Search for weather",
            }
        ]
        backend = _make_backend(_valid_json_response(payload))
        planner = LLMPlanner(tools=[_tool("web_search")], backend=backend)

        plan = planner.plan("Get the weather", {})

        assert isinstance(plan, Plan)
        assert plan.goal == "Get the weather"
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert isinstance(step, PlanStep)
        assert step.tool == "web_search"
        assert step.args == {"query": "weather"}
        assert step.description == "Search for weather"

    def test_multi_step_plan(self) -> None:
        payload = [
            {"tool": "get_calendar", "args": {}, "description": "Fetch calendar events"},
            {
                "tool": "send_email",
                "args": {"to": "boss@example.com"},
                "description": "Email summary",
            },
        ]
        backend = _make_backend(_valid_json_response(payload))
        planner = LLMPlanner(
            tools=[_tool("get_calendar"), _tool("send_email")],
            backend=backend,
        )

        plan = planner.plan("Email my boss today's schedule", {})

        assert len(plan.steps) == 2
        assert plan.steps[0].tool == "get_calendar"
        assert plan.steps[1].tool == "send_email"
        assert plan.steps[1].args == {"to": "boss@example.com"}

    def test_plan_ids_are_strings(self) -> None:
        payload = [{"tool": "web_search", "args": {}, "description": "Search"}]
        backend = _make_backend(_valid_json_response(payload))
        planner = LLMPlanner(tools=[_tool()], backend=backend)

        plan = planner.plan("Do something", {})

        assert isinstance(plan.id, str)
        assert plan.id  # non-empty
        assert plan.steps[0].id == "step-1"

    def test_step_ids_are_sequential(self) -> None:
        payload = [
            {"tool": "a", "args": {}, "description": "Step A"},
            {"tool": "b", "args": {}, "description": "Step B"},
            {"tool": "c", "args": {}, "description": "Step C"},
        ]
        backend = _make_backend(_valid_json_response(payload))
        planner = LLMPlanner(
            tools=[_tool("a"), _tool("b"), _tool("c")],
            backend=backend,
        )

        plan = planner.plan("Three steps", {})

        assert [s.id for s in plan.steps] == ["step-1", "step-2", "step-3"]

    def test_markdown_fenced_json_accepted(self) -> None:
        payload = [{"tool": "web_search", "args": {}, "description": "Search"}]
        fenced = "```json\n" + json.dumps(payload) + "\n```"
        backend = _make_backend(fenced)
        planner = LLMPlanner(tools=[_tool()], backend=backend)

        plan = planner.plan("Goal", {})

        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "web_search"


# ---------------------------------------------------------------------------
# Malformed JSON → PlanningError raised
# ---------------------------------------------------------------------------


class TestMalformedJSON:
    def test_completely_invalid_json(self) -> None:
        backend = _make_backend("This is not JSON at all!")
        planner = LLMPlanner(tools=[_tool()], backend=backend)

        with pytest.raises(PlanningError, match="not valid JSON"):
            planner.plan("Goal", {})

    def test_partial_json(self) -> None:
        backend = _make_backend('[{"tool": "web_search"')
        planner = LLMPlanner(tools=[_tool()], backend=backend)

        with pytest.raises(PlanningError, match="not valid JSON"):
            planner.plan("Goal", {})

    def test_json_object_instead_of_array(self) -> None:
        backend = _make_backend('{"tool": "web_search", "args": {}}')
        planner = LLMPlanner(tools=[_tool()], backend=backend)

        with pytest.raises(PlanningError, match="Expected a JSON array"):
            planner.plan("Goal", {})

    def test_json_string_instead_of_array(self) -> None:
        backend = _make_backend('"just a string"')
        planner = LLMPlanner(tools=[_tool()], backend=backend)

        with pytest.raises(PlanningError, match="Expected a JSON array"):
            planner.plan("Goal", {})

    def test_json_null(self) -> None:
        backend = _make_backend("null")
        planner = LLMPlanner(tools=[_tool()], backend=backend)

        with pytest.raises(PlanningError, match="Expected a JSON array"):
            planner.plan("Goal", {})


# ---------------------------------------------------------------------------
# Empty steps response → PlanningError raised
# ---------------------------------------------------------------------------


class TestEmptySteps:
    def test_empty_array(self) -> None:
        backend = _make_backend("[]")
        planner = LLMPlanner(tools=[_tool()], backend=backend)

        with pytest.raises(PlanningError, match="empty step list"):
            planner.plan("Goal", {})

    def test_empty_array_with_whitespace(self) -> None:
        backend = _make_backend("  [  ]  ")
        planner = LLMPlanner(tools=[_tool()], backend=backend)

        with pytest.raises(PlanningError, match="empty step list"):
            planner.plan("Goal", {})


# ---------------------------------------------------------------------------
# Unknown tool name → step flagged with warning but plan still created
# ---------------------------------------------------------------------------


class TestUnknownTool:
    def test_unknown_tool_plan_still_created(self, caplog: pytest.LogCaptureFixture) -> None:
        payload = [{"tool": "nonexistent_tool", "args": {}, "description": "Do something unknown"}]
        backend = _make_backend(_valid_json_response(payload))
        planner = LLMPlanner(tools=[_tool("web_search")], backend=backend)

        with caplog.at_level(logging.WARNING, logger="rex.autonomy.llm_planner"):
            plan = planner.plan("Use unknown tool", {})

        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "nonexistent_tool"
        assert any("unknown tool" in record.message for record in caplog.records)

    def test_unknown_tool_warning_contains_tool_name(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = [{"tool": "mystery_tool", "args": {}, "description": "Mystery"}]
        backend = _make_backend(_valid_json_response(payload))
        planner = LLMPlanner(tools=[_tool("web_search")], backend=backend)

        with caplog.at_level(logging.WARNING, logger="rex.autonomy.llm_planner"):
            planner.plan("Goal", {})

        warning_messages = " ".join(r.message for r in caplog.records)
        assert "mystery_tool" in warning_messages

    def test_mixed_known_and_unknown_tools(self) -> None:
        payload = [
            {"tool": "web_search", "args": {}, "description": "Known step"},
            {"tool": "mystery_tool", "args": {}, "description": "Unknown step"},
        ]
        backend = _make_backend(_valid_json_response(payload))
        planner = LLMPlanner(tools=[_tool("web_search")], backend=backend)

        plan = planner.plan("Goal with mixed tools", {})

        assert len(plan.steps) == 2
        assert plan.steps[0].tool == "web_search"
        assert plan.steps[1].tool == "mystery_tool"

    def test_no_tools_registered_no_warning_for_any_tool(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When the tool list is empty, unknown-tool logic is skipped."""
        payload = [{"tool": "anything", "args": {}, "description": "Does something"}]
        backend = _make_backend(_valid_json_response(payload))
        planner = LLMPlanner(tools=[], backend=backend)

        with caplog.at_level(logging.WARNING, logger="rex.autonomy.llm_planner"):
            plan = planner.plan("Goal", {})

        assert len(plan.steps) == 1
        # No warnings emitted because the tool name check is skipped when no tools registered
        assert not any("unknown tool" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Backend is mocked (no network access)
# ---------------------------------------------------------------------------


class TestNoNetworkAccess:
    def test_backend_generate_called_with_messages(self) -> None:
        payload = [{"tool": "web_search", "args": {}, "description": "Search"}]
        backend = _make_backend(_valid_json_response(payload))
        planner = LLMPlanner(tools=[_tool()], backend=backend)

        planner.plan("Do a search", {"user": "alice"})

        backend.generate.assert_called_once()
        call_kwargs = backend.generate.call_args
        # Called with keyword argument 'messages'
        assert "messages" in call_kwargs.kwargs

    def test_lazy_backend_not_instantiated_when_provided(self) -> None:
        """Providing an explicit backend suppresses lazy LanguageModel init."""
        payload = [{"tool": "web_search", "args": {}, "description": "Search"}]
        backend = _make_backend(_valid_json_response(payload))
        planner = LLMPlanner(tools=[_tool()], backend=backend)

        with patch(
            "rex.autonomy.llm_planner.LLMPlanner._get_backend", wraps=planner._get_backend
        ) as spy:
            planner.plan("Goal", {})
            # _get_backend is still called (it's the normal path), but it returns our mock
            spy.assert_called_once()

        # The pre-supplied backend's generate should have been called
        backend.generate.assert_called_once()
