"""Tests for US-061: Planner prompt generation.

Verifies that the planner builds prompts that include the task description
and available tools so the LLM has full context for reasoning.
"""

from __future__ import annotations

import pytest

from rex.planner import Planner
from rex.policy_engine import get_policy_engine, reset_policy_engine
from rex.tool_registry import ToolMeta, ToolRegistry, get_tool_registry, reset_tool_registry


@pytest.fixture(autouse=True)
def isolated_registry():
    reset_tool_registry()
    reset_policy_engine()
    yield
    reset_tool_registry()
    reset_policy_engine()


def _make_planner(*tool_names_and_descs: tuple[str, str]) -> Planner:
    """Create a Planner with a fresh registry containing specified tools."""
    registry = ToolRegistry()
    for name, desc in tool_names_and_descs:
        registry.register_tool(ToolMeta(name=name, description=desc))
    return Planner(tool_registry=registry, policy_engine=get_policy_engine())


class TestPlannerBuildsPromptsCorrectly:
    """build_prompt() returns a non-empty, well-structured string."""

    def test_returns_string(self):
        planner = _make_planner(("time_now", "Get current time"))
        result = planner.build_prompt("check the time")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_prompt_has_three_sections(self):
        planner = _make_planner(("time_now", "Get current time"))
        result = planner.build_prompt("check the time")
        # Sections separated by blank lines
        parts = [p.strip() for p in result.split("\n\n") if p.strip()]
        assert len(parts) >= 2

    def test_empty_goal_still_builds_prompt(self):
        planner = _make_planner(("time_now", "Get current time"))
        result = planner.build_prompt("")
        # Should still return a prompt, even if goal is empty
        assert isinstance(result, str)

    def test_no_tools_builds_prompt(self):
        planner = Planner(tool_registry=ToolRegistry(), policy_engine=get_policy_engine())
        result = planner.build_prompt("do something")
        assert isinstance(result, str)
        assert len(result) > 0


class TestTaskDescriptionIncluded:
    """The goal text appears in the generated prompt."""

    def test_goal_in_prompt(self):
        planner = _make_planner(("time_now", "Get current time"))
        goal = "check the time in Dallas"
        result = planner.build_prompt(goal)
        assert goal in result

    def test_goal_labeled_as_task(self):
        planner = _make_planner(("time_now", "Get current time"))
        goal = "send a newsletter"
        result = planner.build_prompt(goal)
        # Goal should appear after a "Task:" label
        assert "Task:" in result
        assert goal in result

    def test_multiword_goal_preserved(self):
        planner = _make_planner(("web_search", "Search the web"))
        goal = "search for Python tutorials and summarize results"
        result = planner.build_prompt(goal)
        assert goal in result

    def test_special_characters_in_goal(self):
        planner = _make_planner(("time_now", "Get current time"))
        goal = "check time in São Paulo, Brazil"
        result = planner.build_prompt(goal)
        assert goal in result


class TestAvailableToolsListed:
    """Registered tools appear by name in the generated prompt."""

    def test_single_tool_listed(self):
        planner = _make_planner(("time_now", "Get current time"))
        result = planner.build_prompt("check time")
        assert "time_now" in result

    def test_multiple_tools_all_listed(self):
        planner = _make_planner(
            ("time_now", "Get current time"),
            ("web_search", "Search the web"),
            ("send_email", "Send an email"),
        )
        result = planner.build_prompt("do a task")
        assert "time_now" in result
        assert "web_search" in result
        assert "send_email" in result

    def test_tool_descriptions_included(self):
        planner = _make_planner(("weather_now", "Get current weather conditions"))
        result = planner.build_prompt("check weather")
        assert "Get current weather conditions" in result

    def test_no_tools_returns_none_marker(self):
        planner = Planner(tool_registry=ToolRegistry(), policy_engine=get_policy_engine())
        result = planner.build_prompt("do something")
        assert "none" in result.lower() or "no tools" in result.lower() or "Available tools" in result

    def test_tools_section_labeled(self):
        planner = _make_planner(("time_now", "Get current time"))
        result = planner.build_prompt("check time")
        assert "Available tools" in result

    def test_prompt_consistent_for_same_inputs(self):
        """build_prompt is deterministic for the same inputs."""
        planner = _make_planner(
            ("time_now", "Get current time"),
            ("web_search", "Search the web"),
        )
        result1 = planner.build_prompt("check time")
        result2 = planner.build_prompt("check time")
        assert result1 == result2
