"""Integration smoke tests for rex.autonomy.runner.

These tests verify that the autonomy runner correctly dispatches to the
configured planner and produces a valid, non-empty Plan.  All LLM calls
are mocked so no network access is required.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rex.autonomy.llm_planner import ToolDefinition
from rex.autonomy.models import Plan, PlanStatus
from rex.autonomy.runner import create_planner, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backend(response: str) -> MagicMock:
    """Return a mock LLMBackend that returns *response* from generate()."""
    backend = MagicMock()
    backend.generate.return_value = response
    return backend


def _weather_response() -> str:
    return json.dumps(
        [
            {
                "tool": "web_search",
                "args": {"query": "current weather"},
                "description": "Search for the current weather",
            }
        ]
    )


# ---------------------------------------------------------------------------
# Integration smoke test: run() with LLMPlanner produces a non-empty Plan
# ---------------------------------------------------------------------------


class TestRunWithLLMPlanner:
    def test_simple_goal_produces_non_empty_plan(self) -> None:
        """Calling run() with goal 'Get the weather' returns a Plan with steps."""
        backend = _mock_backend(_weather_response())
        tools = [ToolDefinition(name="web_search", description="Search the web")]

        with patch("rex.autonomy.llm_planner.LLMPlanner._get_backend", return_value=backend):
            plan = run(
                "Get the weather",
                tools=tools,
                planner_key="llm",
            )

        assert isinstance(plan, Plan)
        assert len(plan.steps) > 0
        assert plan.goal == "Get the weather"
        assert plan.status == PlanStatus.PENDING

    def test_plan_step_has_expected_tool(self) -> None:
        backend = _mock_backend(_weather_response())
        tools = [ToolDefinition(name="web_search", description="Search the web")]

        with patch("rex.autonomy.llm_planner.LLMPlanner._get_backend", return_value=backend):
            plan = run("Get the weather", tools=tools, planner_key="llm")

        assert plan.steps[0].tool == "web_search"

    def test_context_forwarded_to_planner(self) -> None:
        """Context dict is forwarded; plan is still produced."""
        backend = _mock_backend(_weather_response())

        with patch("rex.autonomy.llm_planner.LLMPlanner._get_backend", return_value=backend):
            plan = run(
                "Get the weather",
                context={"user": "alice", "location": "London"},
                planner_key="llm",
            )

        assert isinstance(plan, Plan)
        assert len(plan.steps) > 0


# ---------------------------------------------------------------------------
# create_planner() respects planner_key argument
# ---------------------------------------------------------------------------


class TestCreatePlanner:
    def test_llm_key_returns_llm_planner(self) -> None:
        from rex.autonomy.llm_planner import LLMPlanner

        planner = create_planner(planner_key="llm")
        assert isinstance(planner, LLMPlanner)

    def test_rule_key_returns_rule_planner(self) -> None:
        from rex.autonomy.rule_planner import RulePlanner

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            planner = create_planner(planner_key="rule")

        assert isinstance(planner, RulePlanner)

    def test_rule_planner_emits_deprecation_warning(self) -> None:
        with pytest.warns(DeprecationWarning, match="deprecated"):
            create_planner(planner_key="rule")


# ---------------------------------------------------------------------------
# Config-file planner selection
# ---------------------------------------------------------------------------


class TestPlannerKeyFromConfig:
    def test_llm_key_in_config_file(self, tmp_path: Path) -> None:
        from rex.autonomy.llm_planner import LLMPlanner

        config = tmp_path / "autonomy.json"
        config.write_text(json.dumps({"planner": "llm"}), encoding="utf-8")

        planner = create_planner(config_path=config)
        assert isinstance(planner, LLMPlanner)

    def test_rule_key_in_config_file(self, tmp_path: Path) -> None:
        from rex.autonomy.rule_planner import RulePlanner

        config = tmp_path / "autonomy.json"
        config.write_text(json.dumps({"planner": "rule"}), encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            planner = create_planner(config_path=config)

        assert isinstance(planner, RulePlanner)

    def test_missing_planner_key_defaults_to_llm(self, tmp_path: Path) -> None:
        from rex.autonomy.llm_planner import LLMPlanner

        config = tmp_path / "autonomy.json"
        config.write_text(json.dumps({"default_mode": "suggest"}), encoding="utf-8")

        planner = create_planner(config_path=config)
        assert isinstance(planner, LLMPlanner)

    def test_invalid_planner_key_defaults_to_llm(self, tmp_path: Path) -> None:
        from rex.autonomy.llm_planner import LLMPlanner

        config = tmp_path / "autonomy.json"
        config.write_text(json.dumps({"planner": "unknown_value"}), encoding="utf-8")

        planner = create_planner(config_path=config)
        assert isinstance(planner, LLMPlanner)

    def test_missing_config_file_defaults_to_llm(self, tmp_path: Path) -> None:
        from rex.autonomy.llm_planner import LLMPlanner

        config = tmp_path / "nonexistent.json"
        planner = create_planner(config_path=config)
        assert isinstance(planner, LLMPlanner)


# ---------------------------------------------------------------------------
# Rule planner smoke test (deprecated path)
# ---------------------------------------------------------------------------


class TestRulePlannerSmoke:
    def test_rule_planner_produces_non_empty_plan(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            plan = run("Get the weather", planner_key="rule")

        assert isinstance(plan, Plan)
        assert len(plan.steps) > 0
        assert plan.goal == "Get the weather"
