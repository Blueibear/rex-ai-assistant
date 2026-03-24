"""Unit tests for US-224: LLMPlanner.plan_with_alternatives and Replanner integration."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from rex.autonomy.llm_planner import LLMPlanner, PlanningError
from rex.autonomy.models import Plan, PlanStatus, PlanStep
from rex.autonomy.replanner import Replanner
from rex.autonomy.runner import execute_plan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backend(response: str) -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = response
    return backend


def _three_plans_response(*plan_tools: tuple[str, ...]) -> str:
    """Build a valid JSON array-of-arrays response for plan_with_alternatives."""
    outer = []
    for tools in plan_tools:
        inner = [{"tool": t, "args": {}, "description": f"Execute {t}"} for t in tools]
        outer.append(inner)
    return json.dumps(outer)


def _make_failing_plan(tool: str = "noop") -> Plan:
    step = PlanStep(id="s1", tool=tool, description="Step 1")
    return Plan(id="plan-1", goal="Test goal", steps=[step])


# ---------------------------------------------------------------------------
# LLMPlanner.plan_with_alternatives — response parsing
# ---------------------------------------------------------------------------


class TestPlanWithAlternatives:
    def test_returns_three_plans(self) -> None:
        resp = _three_plans_response(("tool_a",), ("tool_b",), ("tool_c",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)

        result = planner.plan_with_alternatives("Do something", {})

        assert len(result) == 3

    def test_all_elements_are_plan_instances(self) -> None:
        resp = _three_plans_response(("tool_a",), ("tool_b",), ("tool_c",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)

        result = planner.plan_with_alternatives("Do something", {})

        assert all(isinstance(p, Plan) for p in result)

    def test_each_plan_has_correct_goal(self) -> None:
        resp = _three_plans_response(("tool_a",), ("tool_b",), ("tool_c",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)

        result = planner.plan_with_alternatives("Achieve the goal", {})

        for plan in result:
            assert plan.goal == "Achieve the goal"

    def test_primary_plan_has_correct_tool(self) -> None:
        resp = _three_plans_response(("primary_tool",), ("alt1",), ("alt2",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)

        result = planner.plan_with_alternatives("Goal", {})

        assert result[0].steps[0].tool == "primary_tool"

    def test_alternative_plans_have_correct_tools(self) -> None:
        resp = _three_plans_response(("primary",), ("alt1_tool",), ("alt2_tool",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)

        result = planner.plan_with_alternatives("Goal", {})

        assert result[1].steps[0].tool == "alt1_tool"
        assert result[2].steps[0].tool == "alt2_tool"

    def test_multi_step_plans(self) -> None:
        resp = _three_plans_response(
            ("step_a", "step_b"),
            ("alt_step1", "alt_step2"),
            ("other_step",),
        )
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)

        result = planner.plan_with_alternatives("Goal", {})

        assert len(result[0].steps) == 2
        assert len(result[1].steps) == 2
        assert len(result[2].steps) == 1

    def test_step_ids_include_alt_prefix(self) -> None:
        resp = _three_plans_response(("t1",), ("t2",), ("t3",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)

        result = planner.plan_with_alternatives("Goal", {})

        # Primary plan uses alt0-step prefix
        assert result[0].steps[0].id.startswith("alt0-step")
        assert result[1].steps[0].id.startswith("alt1-step")
        assert result[2].steps[0].id.startswith("alt2-step")

    def test_invalid_json_raises_planning_error(self) -> None:
        backend = _mock_backend("not json")
        planner = LLMPlanner(tools=[], backend=backend)

        with pytest.raises(PlanningError, match="not valid JSON"):
            planner.plan_with_alternatives("Goal", {})

    def test_wrong_number_of_plans_raises_planning_error(self) -> None:
        # Only 2 plans instead of 3
        resp = json.dumps(
            [
                [{"tool": "t1", "args": {}, "description": "d"}],
                [{"tool": "t2", "args": {}, "description": "d"}],
            ]
        )
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)

        with pytest.raises(PlanningError, match="exactly 3"):
            planner.plan_with_alternatives("Goal", {})

    def test_strips_code_fences(self) -> None:
        raw = "```json\n" + _three_plans_response(("t1",), ("t2",), ("t3",)) + "\n```"
        backend = _mock_backend(raw)
        planner = LLMPlanner(tools=[], backend=backend)

        result = planner.plan_with_alternatives("Goal", {})

        assert len(result) == 3

    def test_calls_backend_generate(self) -> None:
        resp = _three_plans_response(("t1",), ("t2",), ("t3",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)

        planner.plan_with_alternatives("Goal", {})

        backend.generate.assert_called_once()


# ---------------------------------------------------------------------------
# Replanner with planner — alternatives cycling
# ---------------------------------------------------------------------------


class TestReplannerWithAlternatives:
    def test_first_replan_returns_first_alternative(self) -> None:
        resp = _three_plans_response(("primary",), ("alt1",), ("alt2",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)
        replanner = Replanner(planner=planner)

        plan = _make_failing_plan()
        failed_step = plan.steps[0]
        result = replanner.replan(plan, failed_step, "error")

        assert result.steps[0].tool == "primary"

    def test_second_replan_returns_second_alternative(self) -> None:
        resp = _three_plans_response(("primary",), ("alt1",), ("alt2",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)
        replanner = Replanner(planner=planner)

        plan = _make_failing_plan()
        failed_step = plan.steps[0]
        replanner.replan(plan, failed_step, "error1")
        result = replanner.replan(plan, failed_step, "error2")

        assert result.steps[0].tool == "alt1"

    def test_third_replan_returns_third_alternative(self) -> None:
        resp = _three_plans_response(("primary",), ("alt1",), ("alt2",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)
        replanner = Replanner(planner=planner)

        plan = _make_failing_plan()
        failed_step = plan.steps[0]
        replanner.replan(plan, failed_step, "e1")
        replanner.replan(plan, failed_step, "e2")
        result = replanner.replan(plan, failed_step, "e3")

        assert result.steps[0].tool == "alt2"

    def test_plan_with_alternatives_called_only_once(self) -> None:
        """LLM is called once to generate all alternatives, not on every replan."""
        resp = _three_plans_response(("p",), ("a1",), ("a2",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)
        replanner = Replanner(planner=planner)

        plan = _make_failing_plan()
        failed_step = plan.steps[0]
        replanner.replan(plan, failed_step, "e1")
        replanner.replan(plan, failed_step, "e2")
        replanner.replan(plan, failed_step, "e3")

        backend.generate.assert_called_once()

    def test_alternatives_logged_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        resp = _three_plans_response(("p",), ("a1",), ("a2",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)
        replanner = Replanner(planner=planner)

        plan = _make_failing_plan()
        plan.goal = "Achieve target"
        failed_step = plan.steps[0]

        with caplog.at_level(logging.INFO, logger="rex.autonomy.replanner"):
            replanner.replan(plan, failed_step, "error")

        assert any(
            "Trying alternative plan" in r.message and "Achieve target" in r.message
            for r in caplog.records
        )

    def test_alternative_log_contains_n_of_2(self, caplog: pytest.LogCaptureFixture) -> None:
        resp = _three_plans_response(("p",), ("a1",), ("a2",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)
        replanner = Replanner(planner=planner)

        plan = _make_failing_plan()
        failed_step = plan.steps[0]

        with caplog.at_level(logging.INFO, logger="rex.autonomy.replanner"):
            replanner.replan(plan, failed_step, "e1")
            replanner.replan(plan, failed_step, "e2")

        messages = [r.message for r in caplog.records if "Trying alternative" in r.message]
        assert "1/2" in messages[0]
        assert "2/2" in messages[1]


# ---------------------------------------------------------------------------
# Runner integration — three plans attempted on consecutive failures
# ---------------------------------------------------------------------------


class TestRunnerWithAlternatives:
    def test_runner_attempts_each_alternative_on_consecutive_failures(self) -> None:
        """Runner tries all 3 plans when steps always fail."""
        resp = _three_plans_response(("noop",), ("noop",), ("noop",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)
        replanner = Replanner(planner=planner)

        plan = _make_failing_plan("noop")
        result = execute_plan(
            plan,
            {"noop": lambda **_: (_ for _ in ()).throw(RuntimeError("fail"))},
            replanner=replanner,
            max_replan_attempts=2,
        )

        assert result.status == PlanStatus.FAILED
        # LLM called once (to generate all alternatives)
        backend.generate.assert_called_once()

    def test_runner_succeeds_when_alternative_plan_succeeds(self) -> None:
        """Runner marks plan completed when an alternative plan's step succeeds."""
        call_count = [0]

        def sometimes_ok(**_: object) -> str:
            call_count[0] += 1
            if call_count[0] <= 1:
                raise RuntimeError("first attempt fails")
            return "ok"

        resp = _three_plans_response(("noop",), ("noop",), ("noop",))
        backend = _mock_backend(resp)
        planner = LLMPlanner(tools=[], backend=backend)
        replanner = Replanner(planner=planner)

        plan = _make_failing_plan("noop")
        result = execute_plan(
            plan,
            {"noop": sometimes_ok},
            replanner=replanner,
            max_replan_attempts=2,
        )

        assert result.status == PlanStatus.COMPLETED
