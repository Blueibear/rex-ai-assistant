"""Tests for US-032: Tool auto-selection system.

Covers:
- rex/tool_catalog.py exposes CatalogEntry with intent patterns
- Weather query routes to weather tool
- "Turn on light" routes to HA tool
- Calendar query routes to calendar tool
- No intent match returns empty list (LLM fallback)
- Highest-confidence tool appears first when multiple match
"""

from __future__ import annotations

from rex.tool_catalog import TOOL_CATALOG, CatalogEntry
from rex.tools.dispatcher import ToolDispatcher
from rex.tools.registry import Tool, ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, tags: list[str]) -> Tool:
    return Tool(
        name=name,
        description=f"Tool: {name}",
        capability_tags=tags,
        requires_config=[],
        handler=lambda **kw: {"tool": name},
    )


def _dispatcher(*tools: Tool) -> ToolDispatcher:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return ToolDispatcher(reg)


# ---------------------------------------------------------------------------
# Catalog structure tests
# ---------------------------------------------------------------------------


class TestToolCatalog:
    def test_catalog_is_non_empty(self) -> None:
        assert len(TOOL_CATALOG) > 0

    def test_entries_are_catalog_entry_instances(self) -> None:
        for entry in TOOL_CATALOG:
            assert isinstance(entry, CatalogEntry)

    def test_every_entry_has_name_and_patterns(self) -> None:
        for entry in TOOL_CATALOG:
            assert entry.name, f"CatalogEntry missing name: {entry}"
            assert entry.intent_patterns, f"CatalogEntry {entry.name!r} has no intent_patterns"

    def test_known_tools_in_catalog(self) -> None:
        names = {e.name for e in TOOL_CATALOG}
        assert "weather_now" in names
        assert "home_assistant_call_service" in names
        assert "calendar_create_event" in names
        assert "web_search" in names
        assert "send_email" in names


# ---------------------------------------------------------------------------
# Routing tests (AC5: weather, HA, calendar)
# ---------------------------------------------------------------------------


class TestToolRouting:
    def test_weather_query_routes_to_weather_tool(self) -> None:
        """'What's the weather?' selects the weather tool."""
        weather = _make_tool("weather_now", ["weather", "forecast"])
        disp = _dispatcher(weather)
        tools = disp.select_tools("What's the weather like today?")
        assert any(t.name == "weather_now" for t in tools)

    def test_turn_on_light_routes_to_ha_tool(self) -> None:
        """'Turn on the light' selects the Home Assistant tool."""
        ha = _make_tool("home_assistant_call_service", ["smart_home", "home_assistant", "iot"])
        disp = _dispatcher(ha)
        tools = disp.select_tools("Turn on the living room lights")
        assert any(t.name == "home_assistant_call_service" for t in tools)

    def test_calendar_query_routes_to_calendar_tool(self) -> None:
        """'What's on my calendar' selects the calendar tool."""
        cal = _make_tool("calendar_create", ["calendar", "schedule", "event"])
        disp = _dispatcher(cal)
        tools = disp.select_tools("What's on my calendar today?")
        assert any(t.name == "calendar_create" for t in tools)

    def test_no_match_returns_empty_list_for_llm_fallback(self) -> None:
        """Unrelated query returns empty list so caller falls back to LLM."""
        weather = _make_tool("weather_now", ["weather", "forecast"])
        disp = _dispatcher(weather)
        tools = disp.select_tools("Tell me a funny joke")
        assert tools == []

    def test_unmatched_with_multiple_tools_still_empty(self) -> None:
        """No intent match even with a large registry returns empty list."""
        weather = _make_tool("weather_now", ["weather", "forecast"])
        ha = _make_tool("home_assistant_call_service", ["smart_home", "home_assistant", "iot"])
        cal = _make_tool("calendar_create", ["calendar", "schedule", "event"])
        disp = _dispatcher(weather, ha, cal)
        tools = disp.select_tools("How are you doing?")
        assert tools == []


# ---------------------------------------------------------------------------
# Confidence scoring tests (AC3: highest-confidence chosen first)
# ---------------------------------------------------------------------------


class TestConfidenceScoring:
    def test_higher_confidence_tool_comes_first(self) -> None:
        """Tool matching more fired tags appears before lower-confidence tool."""
        # multi_tag matches both "weather" and "search" tags when both rules fire
        multi_tag = _make_tool("multi_tag_tool", ["weather", "search"])
        single_tag = _make_tool("single_tag_tool", ["weather"])
        disp = _dispatcher(multi_tag, single_tag)
        # "Search for the weather" triggers both weather and search intent rules
        tools = disp.select_tools("Search for the weather forecast today")
        names = [t.name for t in tools]
        assert "multi_tag_tool" in names
        assert (
            names[0] == "multi_tag_tool"
        ), f"Expected multi_tag_tool first (higher confidence), got {names}"

    def test_equal_confidence_tools_both_included(self) -> None:
        """Tools with equal confidence scores are both returned."""
        # Both tools share the same "weather" tag — they tie on score
        tool_a = _make_tool("tool_a", ["weather"])
        tool_b = _make_tool("tool_b", ["weather"])
        disp = _dispatcher(tool_a, tool_b)
        tools = disp.select_tools("What's the weather today?")
        names = [t.name for t in tools]
        assert "tool_a" in names
        assert "tool_b" in names

    def test_no_duplicate_tools_in_output(self) -> None:
        """The same tool is never returned more than once."""
        weather = _make_tool("weather_now", ["weather", "forecast"])
        disp = _dispatcher(weather)
        tools = disp.select_tools("weather weather weather forecast forecast")
        names = [t.name for t in tools]
        assert len(names) == len(set(names))
