"""Planner-to-router end-to-end integration tests (COR-001 regression guard).

For every tool in EXECUTABLE_TOOLS:
1. Build a registry that includes the tool.
2. Use the Planner to generate a minimal plan containing a step for that tool.
3. Execute each step's tool_call through execute_tool().
4. Assert the result is a non-empty string (not an exception).

All external services are mocked — no real API calls are made.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rex.openclaw.tool_registry import ToolMeta, ToolRegistry, set_tool_registry
from rex.planner import Planner
from rex.tool_catalog import EXECUTABLE_TOOLS
from rex.tool_router import execute_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry_with_all_catalog_tools() -> ToolRegistry:
    """Return a ToolRegistry that has every EXECUTABLE_TOOLS entry registered.

    The default global registry is missing calendar_create_event and
    home_assistant_call_service; we register them explicitly here so the
    Planner's _can_use_tool() guard passes.
    """
    reg = ToolRegistry()
    for name in EXECUTABLE_TOOLS:
        reg.register_tool(
            ToolMeta(
                name=name,
                description=f"{name} (test stub)",
                required_credentials=[],
            )
        )
    return reg


def _goal_for_tool(tool_name: str) -> str:
    """Return a natural-language goal that triggers a plan for *tool_name*."""
    return {
        "time_now": "check the time",
        "weather_now": "check the weather in London",
        "web_search": "search for Python news",
        "send_email": "send email to test@example.com",
        "calendar_create_event": "schedule a meeting tomorrow",
        "home_assistant_call_service": "turn on the lights",
    }[tool_name]


def _find_step_for_tool(workflow: Any, tool_name: str) -> Any | None:
    """Return the first WorkflowStep whose tool_call.tool matches tool_name."""
    for step in workflow.steps:
        if step.tool_call is not None and step.tool_call.tool == tool_name:
            return step
    return None


# ---------------------------------------------------------------------------
# Shared mock context: patches all external I/O
# ---------------------------------------------------------------------------


def _all_service_patches() -> list:
    """Return a list of patch() context managers covering every external service."""
    # Weather mock
    async def _fake_weather(location: str, api_key: str) -> dict:
        return {
            "city": location,
            "description": "sunny",
            "temp_f": 72.0,
            "temp_c": 22.2,
            "humidity": 55,
            "wind_mph": 8.0,
        }

    mock_cm = MagicMock()
    mock_cm.get_token.return_value = "fake-api-key"

    mock_email_svc = MagicMock()
    mock_email_svc.send.return_value = {"ok": True, "message_id": "test-id", "error": None}

    mock_cal_event = MagicMock()
    mock_cal_event.title = "Test Event"
    mock_cal_event.start_time = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    mock_cal_event.end_time = datetime(2026, 4, 1, 11, 0, tzinfo=timezone.utc)
    mock_cal_svc = MagicMock()
    mock_cal_svc.create_event.return_value = mock_cal_event

    return [
        patch("rex.tool_router.get_credential_manager", return_value=mock_cm),
        patch("rex.tool_router.get_weather", side_effect=_fake_weather),
        patch("rex.tool_router.EmailService", return_value=mock_email_svc),
        patch("rex.tool_router.CalendarService", return_value=mock_cal_svc),
        patch("plugins.web_search.search_web", return_value="Result - https://example.com\nSnippet"),
    ]


# ---------------------------------------------------------------------------
# Main parametrized test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Ensure the global registry is reset after each test."""
    yield
    set_tool_registry(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("tool_name", sorted(EXECUTABLE_TOOLS))
def test_planner_emits_tool_and_router_executes_it(tool_name: str) -> None:
    """Generate a plan containing *tool_name* and execute it through execute_tool()."""
    reg = _make_registry_with_all_catalog_tools()
    planner = Planner(tool_registry=reg)

    goal = _goal_for_tool(tool_name)
    workflow = planner.plan(goal)

    # The plan must contain at least one step for our target tool
    step = _find_step_for_tool(workflow, tool_name)
    assert step is not None, (
        f"Planner did not generate a step for '{tool_name}' from goal '{goal}'. "
        f"Steps in plan: {[s.tool_call.tool for s in workflow.steps if s.tool_call]}"
    )

    # Execute through the router (all external I/O is mocked)
    patches = _all_service_patches()
    for p in patches:
        p.start()
    try:
        result = execute_tool(tool_name, step.tool_call.args or {})
    finally:
        for p in patches:
            p.stop()

    assert isinstance(result, str), f"execute_tool({tool_name!r}) did not return str"
    assert len(result) > 0, f"execute_tool({tool_name!r}) returned empty string"


# ---------------------------------------------------------------------------
# Individual tool execution tests (direct, no planner)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool_name", sorted(EXECUTABLE_TOOLS))
def test_direct_execute_tool_returns_nonempty_string(tool_name: str) -> None:
    """Calling execute_tool() directly with minimal args returns a non-empty string."""
    minimal_args: dict[str, Any] = {
        "time_now": {"location": "UTC"},
        "weather_now": {"location": "Paris"},
        "web_search": {"query": "test"},
        "send_email": {"to": "a@b.com", "subject": "S", "body": "B"},
        "calendar_create_event": {"title": "Test", "start": "2026-04-01T10:00:00"},
        "home_assistant_call_service": {"domain": "light", "service": "turn_on", "entity_id": "light.main"},
    }[tool_name]

    patches = _all_service_patches()
    for p in patches:
        p.start()
    try:
        result = execute_tool(tool_name, minimal_args)
    finally:
        for p in patches:
            p.stop()

    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Non-catalog tools are rejected
# ---------------------------------------------------------------------------


def test_planner_does_not_emit_uncatalogued_tools() -> None:
    """A planner with an uncatalogued tool registered never puts it in a plan."""
    reg = _make_registry_with_all_catalog_tools()
    # Register a tool NOT in the catalog
    reg.register_tool(ToolMeta(name="uncatalogued_tool", description="Not in catalog"))
    planner = Planner(tool_registry=reg)

    # Try goals that could (naively) use the tool
    from rex.planner import UnableToPlanError

    try:
        workflow = planner.plan("search for something")
        for step in workflow.steps:
            if step.tool_call:
                assert step.tool_call.tool in EXECUTABLE_TOOLS, (
                    f"Planner emitted non-catalog tool: {step.tool_call.tool}"
                )
    except UnableToPlanError:
        pass  # no plan is fine


def test_execute_tool_rejects_uncatalogued_tool() -> None:
    """execute_tool() raises UnknownToolError for any tool outside the catalog."""
    from rex.tool_router import UnknownToolError

    with pytest.raises(UnknownToolError):
        execute_tool("uncatalogued_tool", {})
