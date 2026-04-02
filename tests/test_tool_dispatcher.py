"""Integration tests for ToolDispatcher — US-TD-002.

Covers:
- Email intent selects email tool
- Weather intent selects weather tool
- Compound question selects both tools
- No tool invoked when no intent match
- Multiple-domain message selects all relevant tools
- select_tools with config filtering
- execute_tools aggregates results
- format_tool_context returns empty string for empty results
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rex.tools.dispatcher import ToolDispatcher
from rex.tools.registry import Tool, ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, tags: list[str], requires: list[str] | None = None) -> Tool:
    return Tool(
        name=name,
        description=f"Tool: {name}",
        capability_tags=tags,
        requires_config=requires or [],
        handler=lambda **kw: {"tool": name, "kw": kw},
    )


def _make_registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def email_tool() -> Tool:
    return _make_tool("send_email", ["email", "messaging", "send"])


@pytest.fixture()
def weather_tool() -> Tool:
    return _make_tool("weather_now", ["weather", "forecast"])


@pytest.fixture()
def search_tool() -> Tool:
    return _make_tool("web_search", ["search", "web"])


@pytest.fixture()
def calendar_tool() -> Tool:
    return _make_tool("calendar_create", ["calendar", "schedule", "event"])


@pytest.fixture()
def ha_tool() -> Tool:
    return _make_tool("home_assistant_call_service", ["smart_home", "home_assistant", "iot"])


@pytest.fixture()
def full_registry(
    email_tool: Tool,
    weather_tool: Tool,
    search_tool: Tool,
    calendar_tool: Tool,
    ha_tool: Tool,
) -> ToolRegistry:
    return _make_registry(email_tool, weather_tool, search_tool, calendar_tool, ha_tool)


@pytest.fixture()
def dispatcher(full_registry: ToolRegistry) -> ToolDispatcher:
    return ToolDispatcher(full_registry)


# ---------------------------------------------------------------------------
# select_tools tests
# ---------------------------------------------------------------------------


class TestSelectTools:
    def test_email_intent_selects_email_tool(self, dispatcher: ToolDispatcher) -> None:
        tools = dispatcher.select_tools("Do I have any new email?")
        names = [t.name for t in tools]
        assert "send_email" in names

    def test_weather_intent_selects_weather_tool(self, dispatcher: ToolDispatcher) -> None:
        tools = dispatcher.select_tools("What's the weather like today?")
        names = [t.name for t in tools]
        assert "weather_now" in names

    def test_search_intent_selects_search_tool(self, dispatcher: ToolDispatcher) -> None:
        tools = dispatcher.select_tools("Search for the latest news on AI")
        names = [t.name for t in tools]
        assert "web_search" in names

    def test_calendar_intent_selects_calendar_tool(self, dispatcher: ToolDispatcher) -> None:
        tools = dispatcher.select_tools("Add a meeting to my calendar tomorrow")
        names = [t.name for t in tools]
        assert "calendar_create" in names

    def test_smart_home_intent_selects_ha_tool(self, dispatcher: ToolDispatcher) -> None:
        tools = dispatcher.select_tools("Turn off the lights in the living room")
        names = [t.name for t in tools]
        assert "home_assistant_call_service" in names

    def test_compound_question_selects_multiple_tools(self, dispatcher: ToolDispatcher) -> None:
        tools = dispatcher.select_tools("What's the weather and check my email?")
        names = [t.name for t in tools]
        assert "weather_now" in names
        assert "send_email" in names

    def test_no_intent_returns_empty_list(self, dispatcher: ToolDispatcher) -> None:
        tools = dispatcher.select_tools("Tell me a joke")
        assert tools == []

    def test_no_duplicate_tools(self, dispatcher: ToolDispatcher) -> None:
        # Repeated keywords should not produce duplicate entries
        tools = dispatcher.select_tools("email email email")
        names = [t.name for t in tools]
        assert len(names) == len(set(names))

    def test_config_filters_unavailable_tools(self, email_tool: Tool, weather_tool: Tool) -> None:
        # email_tool has no requires_config so always available
        # Add a tool that requires a config field
        gated_tool = _make_tool("gated_search", ["search", "web"], requires=["brave_api_key"])
        reg = _make_registry(email_tool, weather_tool, gated_tool)

        # Config without brave_api_key
        config = MagicMock(spec=[])
        disp = ToolDispatcher(reg, config=config)
        tools = disp.select_tools("Search the web for something")
        names = [t.name for t in tools]
        assert "gated_search" not in names


# ---------------------------------------------------------------------------
# execute_tools tests
# ---------------------------------------------------------------------------


class TestExecuteTools:
    def test_results_aggregated(self, dispatcher: ToolDispatcher, weather_tool: Tool) -> None:
        results = dispatcher.execute_tools([weather_tool], "weather in London?")
        assert "weather_now" in results

    def test_multiple_tools_all_present(
        self,
        dispatcher: ToolDispatcher,
        email_tool: Tool,
        weather_tool: Tool,
    ) -> None:
        results = dispatcher.execute_tools([email_tool, weather_tool], "compound")
        assert "send_email" in results
        assert "weather_now" in results

    def test_failing_tool_stored_as_error_string(self, full_registry: ToolRegistry) -> None:
        bad_tool = Tool(
            name="bad_tool",
            description="Always fails",
            capability_tags=["search"],
            requires_config=[],
            handler=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        disp = ToolDispatcher(_make_registry(bad_tool))
        results = disp.execute_tools([bad_tool], "anything")
        assert "bad_tool" in results
        assert "boom" in str(results["bad_tool"])


# ---------------------------------------------------------------------------
# format_tool_context tests
# ---------------------------------------------------------------------------


class TestFormatToolContext:
    def test_empty_results_returns_empty_string(self) -> None:
        assert ToolDispatcher.format_tool_context({}) == ""

    def test_single_result_formatted(self) -> None:
        ctx = ToolDispatcher.format_tool_context({"weather_now": {"temp": 22}})
        assert "weather_now" in ctx
        assert "22" in ctx

    def test_multiple_results_all_present(self) -> None:
        ctx = ToolDispatcher.format_tool_context({"weather_now": "sunny", "send_email": "3 unread"})
        assert "weather_now" in ctx
        assert "send_email" in ctx
