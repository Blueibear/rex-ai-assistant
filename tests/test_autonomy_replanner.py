"""Unit tests for rex.autonomy.replanner — Replanner class and runner integration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from rex.autonomy.models import Plan, PlanStatus, PlanStep, StepStatus
from rex.autonomy.replanner import Replanner
from rex.autonomy.runner import execute_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backend(response: str) -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = response
    return backend


def _make_plan(*step_ids: str, tool: str = "noop") -> Plan:
    steps = [PlanStep(id=sid, tool=tool, description=f"Step {sid}") for sid in step_ids]
    return Plan(id="plan-1", goal="Test goal", steps=steps)


def _replan_response(*tools: str) -> str:
    return json.dumps(
        [{"tool": t, "args": {}, "description": f"Replanned: {t}"} for t in tools]
    )


# ---------------------------------------------------------------------------
# Replanner.replan() — prompt content
# ---------------------------------------------------------------------------


class TestReplannerPrompt:
    def test_replan_calls_backend_generate(self) -> None:
        backend = _mock_backend(_replan_response("fallback_tool"))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")
        plan.steps[0].status = StepStatus.FAILED
        plan.steps[0].error = "timeout"

        replanner.replan(plan, plan.steps[0], "timeout")

        backend.generate.assert_called_once()

    def test_prompt_contains_goal(self) -> None:
        backend = _mock_backend(_replan_response("fallback_tool"))
        replanner = Replanner(backend=backend)
        plan = Plan(id="p1", goal="Send a report to finance")
        step = PlanStep(id="s1", tool="send_email", description="Send email", status=StepStatus.FAILED)
        plan.steps = [step]

        replanner.replan(plan, step, "SMTP auth failed")

        call_kwargs: Any = backend.generate.call_args
        messages = call_kwargs[1]["messages"]
        prompt_text = messages[0]["content"]
        assert "Send a report to finance" in prompt_text

    def test_prompt_contains_completed_steps(self) -> None:
        backend = _mock_backend(_replan_response("noop"))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1", "s2")
        plan.steps[0].status = StepStatus.SUCCESS
        plan.steps[0].result = "done"
        plan.steps[1].status = StepStatus.FAILED
        plan.steps[1].error = "boom"

        replanner.replan(plan, plan.steps[1], "boom")

        call_kwargs: Any = backend.generate.call_args
        prompt_text = call_kwargs[1]["messages"][0]["content"]
        assert "s1" in prompt_text

    def test_prompt_contains_failed_step_info(self) -> None:
        backend = _mock_backend(_replan_response("noop"))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")
        plan.steps[0].status = StepStatus.FAILED

        replanner.replan(plan, plan.steps[0], "Connection refused")

        call_kwargs: Any = backend.generate.call_args
        prompt_text = call_kwargs[1]["messages"][0]["content"]
        assert "Connection refused" in prompt_text
        assert "s1" in prompt_text

    def test_prompt_contains_error_context(self) -> None:
        backend = _mock_backend(_replan_response("noop"))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")
        step = plan.steps[0]

        replanner.replan(plan, step, "HTTP 503 Service Unavailable")

        call_kwargs: Any = backend.generate.call_args
        prompt_text = call_kwargs[1]["messages"][0]["content"]
        assert "HTTP 503 Service Unavailable" in prompt_text


# ---------------------------------------------------------------------------
# Replanner.replan() — response parsing
# ---------------------------------------------------------------------------


class TestReplannerParsing:
    def test_returns_plan_with_correct_goal(self) -> None:
        backend = _mock_backend(_replan_response("tool_a"))
        replanner = Replanner(backend=backend)
        plan = Plan(id="p1", goal="Archive old files")
        step = PlanStep(id="s1", tool="archive", description="Archive", status=StepStatus.FAILED)

        result = replanner.replan(plan, step, "disk full")

        assert result.goal == "Archive old files"

    def test_returns_plan_with_steps(self) -> None:
        backend = _mock_backend(_replan_response("tool_a", "tool_b"))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        result = replanner.replan(plan, plan.steps[0], "error")

        assert len(result.steps) == 2
        assert result.steps[0].tool == "tool_a"
        assert result.steps[1].tool == "tool_b"

    def test_replanned_steps_are_pending(self) -> None:
        backend = _mock_backend(_replan_response("tool_a"))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        result = replanner.replan(plan, plan.steps[0], "error")

        assert result.steps[0].status == StepStatus.PENDING

    def test_strips_code_fences(self) -> None:
        raw = "```json\n" + _replan_response("tool_x") + "\n```"
        backend = _mock_backend(raw)
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        result = replanner.replan(plan, plan.steps[0], "error")

        assert result.steps[0].tool == "tool_x"

    def test_invalid_json_raises_value_error(self) -> None:
        backend = _mock_backend("not json at all")
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        with pytest.raises(ValueError, match="not valid JSON"):
            replanner.replan(plan, plan.steps[0], "error")

    def test_non_list_response_raises_value_error(self) -> None:
        backend = _mock_backend('{"tool": "x"}')
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        with pytest.raises(ValueError, match="expected a JSON array"):
            replanner.replan(plan, plan.steps[0], "error")

    def test_empty_list_raises_value_error(self) -> None:
        backend = _mock_backend("[]")
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        with pytest.raises(ValueError, match="empty step list"):
            replanner.replan(plan, plan.steps[0], "error")


# ---------------------------------------------------------------------------
# execute_plan() with replanning — runner integration
# ---------------------------------------------------------------------------


class TestExecutePlanWithReplanner:
    def test_no_replanner_failed_step_sets_plan_failed(self) -> None:
        """Baseline: without replanner, behaviour is unchanged."""
        plan = _make_plan("s1")
        execute_plan(plan, {"noop": lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))})
        assert plan.status == PlanStatus.FAILED

    def test_replanner_called_on_step_failure(self) -> None:
        backend = _mock_backend(_replan_response("noop"))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        def fail_then_pass(**_: object) -> str:
            raise RuntimeError("first fail")

        execute_plan(plan, {"noop": fail_then_pass}, replanner=replanner, max_replan_attempts=1)

        backend.generate.assert_called_once()

    def test_replanner_success_sets_plan_completed(self) -> None:
        call_count = [0]

        def sometimes_fails(**_: object) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("transient error")
            return "ok"

        backend = _mock_backend(_replan_response("noop"))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        result = execute_plan(plan, {"noop": sometimes_fails}, replanner=replanner)

        assert result.status == PlanStatus.COMPLETED

    def test_max_replan_attempts_respected(self) -> None:
        backend = _mock_backend(_replan_response("noop"))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        execute_plan(
            plan,
            {"noop": lambda **_: (_ for _ in ()).throw(RuntimeError("always fails"))},
            replanner=replanner,
            max_replan_attempts=2,
        )

        assert backend.generate.call_count == 2

    def test_exceeding_max_attempts_sets_plan_failed(self) -> None:
        backend = _mock_backend(_replan_response("noop"))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        result = execute_plan(
            plan,
            {"noop": lambda **_: (_ for _ in ()).throw(RuntimeError("always fails"))},
            replanner=replanner,
            max_replan_attempts=2,
        )

        assert result.status == PlanStatus.FAILED

    def test_default_max_replan_attempts_is_two(self) -> None:
        backend = _mock_backend(_replan_response("noop"))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        execute_plan(
            plan,
            {"noop": lambda **_: (_ for _ in ()).throw(RuntimeError("always fails"))},
            replanner=replanner,
        )

        assert backend.generate.call_count == 2

    def test_replanned_steps_replace_original_steps(self) -> None:
        """After replanning, plan.steps contains the new steps."""
        call_count = [0]

        def tool(**_: object) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first fail")
            return "ok"

        backend = _mock_backend(json.dumps([
            {"tool": "noop", "args": {}, "description": "Replanned step"}
        ]))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        execute_plan(plan, {"noop": tool}, replanner=replanner)

        # After replanning the step IDs come from the replanner response
        assert plan.steps[0].id.startswith("replan-step-")

    def test_returns_same_plan_object_with_replanner(self) -> None:
        backend = _mock_backend(_replan_response("noop"))
        replanner = Replanner(backend=backend)
        plan = _make_plan("s1")

        call_count = [0]

        def tool(**_: object) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("fail once")
            return "ok"

        result = execute_plan(plan, {"noop": tool}, replanner=replanner)

        assert result is plan

    def test_no_replanner_no_replan_attempt(self) -> None:
        """Without replanner, no LLM calls are made."""
        plan = _make_plan("s1")
        # Just verify execute_plan completes without error and no replanning
        result = execute_plan(plan, {})  # unknown tool -> fails immediately
        assert result.status == PlanStatus.FAILED
