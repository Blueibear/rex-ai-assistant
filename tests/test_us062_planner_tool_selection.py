"""Tests for US-062: Planner tool selection.

Verifies that the planner selects the most appropriate tool from multiple
options and validates the selection.
"""

from __future__ import annotations

import pytest

from rex.openclaw.tool_registry import ToolMeta, ToolRegistry, reset_tool_registry
from rex.planner import Planner
from rex.policy_engine import get_policy_engine, reset_policy_engine


@pytest.fixture(autouse=True)
def isolated_registry():
    reset_tool_registry()
    reset_policy_engine()
    yield
    reset_tool_registry()
    reset_policy_engine()


def _make_planner(*tool_names: str) -> Planner:
    """Create a Planner whose registry contains the named tools (all enabled)."""
    registry = ToolRegistry()
    for name in tool_names:
        registry.register_tool(ToolMeta(name=name, description=f"Tool: {name}"))
    return Planner(tool_registry=registry, policy_engine=get_policy_engine())


class TestPlannerSelectsAppropriateTools:
    """select_tool() returns the first suitable registered tool."""

    def test_returns_registered_tool(self):
        planner = _make_planner("time_now")
        result = planner.select_tool(["time_now"])
        assert result == "time_now"

    def test_returns_none_for_unregistered_tool(self):
        planner = _make_planner()
        result = planner.select_tool(["nonexistent"])
        assert result is None

    def test_returns_none_for_empty_candidates(self):
        planner = _make_planner("time_now")
        result = planner.select_tool([])
        assert result is None

    def test_selects_first_preferred_tool(self):
        """When the preferred tool is available, it wins."""
        planner = _make_planner("time_now", "weather_now")
        result = planner.select_tool(["time_now", "weather_now"])
        assert result == "time_now"


class TestMultipleToolOptionsSupported:
    """select_tool() falls back to alternatives when the preferred is unavailable."""

    def test_falls_back_to_second_option(self):
        """First candidate not registered → second is returned."""
        planner = _make_planner("weather_now")
        result = planner.select_tool(["time_now", "weather_now"])
        assert result == "weather_now"

    def test_falls_back_to_third_option(self):
        planner = _make_planner("web_search")
        result = planner.select_tool(["time_now", "weather_now", "web_search"])
        assert result == "web_search"

    def test_all_unavailable_returns_none(self):
        planner = _make_planner()
        result = planner.select_tool(["time_now", "weather_now", "web_search"])
        assert result is None

    def test_disabled_tool_skipped(self):
        """Disabled tool is skipped; next candidate is tried."""
        registry = ToolRegistry()
        registry.register_tool(ToolMeta(name="time_now", description="Get time", enabled=False))
        registry.register_tool(ToolMeta(name="weather_now", description="Get weather"))
        planner = Planner(tool_registry=registry, policy_engine=get_policy_engine())

        result = planner.select_tool(["time_now", "weather_now"])
        assert result == "weather_now"

    def test_priority_order_respected(self):
        """Order of candidates determines priority."""
        planner = _make_planner("time_now", "weather_now", "web_search")
        # All three are available; first should win
        result = planner.select_tool(["web_search", "time_now", "weather_now"])
        assert result == "web_search"

    def test_single_available_among_many_candidates(self):
        planner = _make_planner("web_search")
        result = planner.select_tool(
            ["time_now", "weather_now", "send_email", "web_search", "calendar_create_event"]
        )
        assert result == "web_search"


class TestToolSelectionValidated:
    """select_tool() only returns tools that pass validation checks."""

    def test_unregistered_tool_not_returned(self):
        planner = _make_planner("time_now")
        result = planner.select_tool(["definitely_not_a_tool"])
        assert result is None

    def test_disabled_tool_not_returned(self):
        registry = ToolRegistry()
        registry.register_tool(ToolMeta(name="time_now", description="Get time", enabled=False))
        planner = Planner(tool_registry=registry, policy_engine=get_policy_engine())

        result = planner.select_tool(["time_now"])
        assert result is None

    def test_selected_tool_is_in_registry(self):
        planner = _make_planner("time_now", "weather_now")
        result = planner.select_tool(["time_now", "weather_now"])
        assert result is not None
        assert planner.tool_registry.has_tool(result)

    def test_selected_tool_is_enabled(self):
        planner = _make_planner("time_now")
        result = planner.select_tool(["time_now"])
        assert result is not None
        meta = planner.tool_registry.get_tool(result)
        assert meta is not None
        assert meta.enabled
