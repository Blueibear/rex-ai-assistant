"""Tests for US-063: Planner fallback behavior.

Verifies that the planner detects tool failure, attempts alternate strategies,
and logs errors when multi-step tasks need to recover.
"""

from __future__ import annotations

import logging

import pytest

from rex.contracts import ToolCall
from rex.planner import Planner, UnableToPlanError
from rex.policy_engine import get_policy_engine, reset_policy_engine
from rex.tool_registry import ToolMeta, ToolRegistry, reset_tool_registry
from rex.workflow import WorkflowStep, generate_step_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _make_step(tool_name: str) -> WorkflowStep:
    return WorkflowStep(
        step_id=generate_step_id(),
        description=f"Execute {tool_name}",
        tool_call=ToolCall(tool=tool_name, args={}),
    )


# ---------------------------------------------------------------------------
# plan_with_fallback — planner detects tool failure and tries alternate goal
# ---------------------------------------------------------------------------


class TestPlannerDetectsToolFailure:
    """plan_with_fallback() succeeds when the primary goal can be planned."""

    def test_primary_goal_succeeds(self):
        planner = _make_planner("web_search")
        wf = planner.plan_with_fallback("search for weather forecast")
        assert wf is not None
        assert len(wf.steps) > 0

    def test_primary_goal_returns_workflow_for_matching_pattern(self):
        planner = _make_planner("web_search")
        wf = planner.plan_with_fallback("search for news")
        assert wf.title == "search for news"


class TestAlternateStrategyAttempted:
    """plan_with_fallback() tries each fallback when the primary fails."""

    def test_fallback_used_when_primary_fails(self):
        # Primary goal "xyzzy" won't match any pattern → fallback to web search
        planner = _make_planner("web_search")
        wf = planner.plan_with_fallback("xyzzy", fallback_goals=["search for something"])
        assert len(wf.steps) > 0

    def test_second_fallback_used_when_first_also_fails(self):
        planner = _make_planner("web_search")
        wf = planner.plan_with_fallback(
            "primary_unknown_goal",
            fallback_goals=["another_unknown_goal", "search for backup"],
        )
        assert len(wf.steps) > 0

    def test_all_fallbacks_exhausted_raises(self):
        planner = _make_planner()
        with pytest.raises(UnableToPlanError):
            planner.plan_with_fallback(
                "unknown_goal",
                fallback_goals=["also_unknown", "still_unknown"],
            )

    def test_no_fallbacks_and_primary_fails_raises(self):
        planner = _make_planner()
        with pytest.raises(UnableToPlanError):
            planner.plan_with_fallback("totally_unrecognised_goal_xyz")

    def test_empty_fallbacks_list_behaves_like_no_fallback(self):
        planner = _make_planner("web_search")
        wf = planner.plan_with_fallback("search for data", fallback_goals=[])
        assert len(wf.steps) > 0

    def test_fallback_workflow_has_steps(self):
        planner = _make_planner("web_search")
        wf = planner.plan_with_fallback(
            "unknowable_goal_abc",
            fallback_goals=["search for results"],
        )
        assert len(wf.steps) >= 1


class TestErrorsLogged:
    """plan_with_fallback() logs failures at WARNING level."""

    def test_primary_failure_is_logged(self, caplog):
        planner = _make_planner("web_search")
        with caplog.at_level(logging.WARNING, logger="rex.planner"):
            planner.plan_with_fallback(
                "unknown_goal_xyz",
                fallback_goals=["search for something"],
            )
        # At least one warning for the failed primary
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) >= 1

    def test_all_failures_logged_before_raising(self, caplog):
        planner = _make_planner()
        with caplog.at_level(logging.WARNING, logger="rex.planner"):
            with pytest.raises(UnableToPlanError):
                planner.plan_with_fallback(
                    "goal_a",
                    fallback_goals=["goal_b", "goal_c"],
                )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        # 3 attempts → 3 warning entries
        assert len(warnings) >= 3


# ---------------------------------------------------------------------------
# execute_step_with_fallback — detects execution failure, tries alternates
# ---------------------------------------------------------------------------


class TestExecuteStepWithFallback:
    """execute_step_with_fallback() handles tool execution failures."""

    def test_primary_step_succeeds(self):
        planner = _make_planner("web_search")
        step = _make_step("web_search")
        result = planner.execute_step_with_fallback(step, [], executor=lambda s: "ok")
        assert result == "ok"

    def test_fallback_step_used_on_primary_failure(self):
        planner = _make_planner("web_search")
        primary = _make_step("failing_tool")
        fallback = _make_step("web_search")

        def executor(step: WorkflowStep) -> str:
            if step.tool_call and step.tool_call.tool == "failing_tool":
                raise RuntimeError("tool failed")
            return "fallback_ok"

        result = planner.execute_step_with_fallback(primary, [fallback], executor=executor)
        assert result == "fallback_ok"

    def test_all_steps_fail_reraises_last_exception(self):
        planner = _make_planner()
        primary = _make_step("tool_a")
        fallback = _make_step("tool_b")

        def executor(step: WorkflowStep) -> str:
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            planner.execute_step_with_fallback(primary, [fallback], executor=executor)

    def test_execution_failure_is_logged(self, caplog):
        planner = _make_planner()
        step = _make_step("bad_tool")

        def executor(step: WorkflowStep) -> str:
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger="rex.planner"):
            with pytest.raises(RuntimeError):
                planner.execute_step_with_fallback(step, [], executor=executor)

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) >= 1

    def test_second_fallback_tried_after_first_fails(self):
        planner = _make_planner()
        primary = _make_step("tool_a")
        fb1 = _make_step("tool_b")
        fb2 = _make_step("tool_c")
        call_order: list[str] = []

        def executor(step: WorkflowStep) -> str:
            tool = step.tool_call.tool if step.tool_call else ""
            call_order.append(tool)
            if tool in ("tool_a", "tool_b"):
                raise RuntimeError("fail")
            return "success"

        result = planner.execute_step_with_fallback(primary, [fb1, fb2], executor=executor)
        assert result == "success"
        assert call_order == ["tool_a", "tool_b", "tool_c"]

    def test_no_fallbacks_and_primary_fails_reraises(self):
        planner = _make_planner()
        step = _make_step("bad_tool")

        with pytest.raises(KeyError):
            planner.execute_step_with_fallback(
                step, [], executor=lambda s: (_ for _ in ()).throw(KeyError("missing"))
            )
