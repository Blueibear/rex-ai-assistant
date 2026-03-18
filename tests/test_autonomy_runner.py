"""Integration smoke tests for rex.autonomy.runner.

These tests verify that the autonomy runner correctly dispatches to the
configured planner and produces a valid, non-empty Plan.  All LLM calls
are mocked so no network access is required.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.autonomy.llm_planner import ToolDefinition
from rex.autonomy.models import Plan, PlanStatus, PlanStep, StepStatus
from rex.autonomy.runner import create_planner, execute_plan, run

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


# ---------------------------------------------------------------------------
# execute_plan() — step execution result tracking (US-221)
# ---------------------------------------------------------------------------


def _make_plan(*step_ids: str) -> Plan:
    """Build a Plan with one PlanStep per id, all using tool 'noop'."""
    steps = [PlanStep(id=sid, tool="noop", description=f"Step {sid}") for sid in step_ids]
    return Plan(id="test-plan", goal="Test goal", steps=steps)


class TestExecutePlan:
    def test_all_steps_succeed_sets_plan_completed(self) -> None:
        plan = _make_plan("s1", "s2")
        tools = {"noop": lambda **_: "ok"}
        result = execute_plan(plan, tools)
        assert result.status == PlanStatus.COMPLETED

    def test_all_steps_succeed_sets_step_status_success(self) -> None:
        plan = _make_plan("s1", "s2")
        tools = {"noop": lambda **_: "ok"}
        execute_plan(plan, tools)
        for step in plan.steps:
            assert step.status == StepStatus.SUCCESS

    def test_step_result_populated_on_success(self) -> None:
        plan = _make_plan("s1")
        tools = {"noop": lambda **_: "tool-output"}
        execute_plan(plan, tools)
        assert plan.steps[0].result == "tool-output"

    def test_step_error_none_on_success(self) -> None:
        plan = _make_plan("s1")
        tools = {"noop": lambda **_: "ok"}
        execute_plan(plan, tools)
        assert plan.steps[0].error is None

    def test_failing_step_sets_plan_failed(self) -> None:
        plan = _make_plan("s1")

        def fail_tool(**_: object) -> str:
            raise RuntimeError("boom")

        execute_plan(plan, {"noop": fail_tool})
        assert plan.status == PlanStatus.FAILED

    def test_failing_step_sets_step_status_failed(self) -> None:
        plan = _make_plan("s1")

        def fail_tool(**_: object) -> str:
            raise RuntimeError("boom")

        execute_plan(plan, {"noop": fail_tool})
        assert plan.steps[0].status == StepStatus.FAILED

    def test_failing_step_populates_error_field(self) -> None:
        plan = _make_plan("s1")

        def fail_tool(**_: object) -> str:
            raise RuntimeError("boom")

        execute_plan(plan, {"noop": fail_tool})
        assert plan.steps[0].error == "boom"

    def test_unknown_tool_sets_step_failed(self) -> None:
        plan = _make_plan("s1")
        result = execute_plan(plan, tools={})
        assert result.steps[0].status == StepStatus.FAILED
        assert "noop" in (result.steps[0].error or "")

    def test_unknown_tool_sets_plan_failed(self) -> None:
        plan = _make_plan("s1")
        result = execute_plan(plan, tools={})
        assert result.status == PlanStatus.FAILED

    def test_skipped_step_is_not_executed(self) -> None:
        plan = _make_plan("s1")
        plan.steps[0].status = StepStatus.SKIPPED
        call_count = 0

        def counting_tool(**_: object) -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        execute_plan(plan, {"noop": counting_tool})
        assert call_count == 0
        assert plan.steps[0].status == StepStatus.SKIPPED

    def test_one_failed_step_leaves_others_completed(self) -> None:
        """Steps after a failure still execute (no short-circuit)."""
        plan = _make_plan("s1", "s2")
        call_count = 0

        def first_fails(**_: object) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first step fails")
            return "ok"

        execute_plan(plan, {"noop": first_fails})
        assert plan.steps[0].status == StepStatus.FAILED
        assert plan.steps[1].status == StepStatus.SUCCESS
        assert plan.status == PlanStatus.FAILED

    def test_returns_same_plan_object(self) -> None:
        plan = _make_plan("s1")
        tools = {"noop": lambda **_: "ok"}
        result = execute_plan(plan, tools)
        assert result is plan
