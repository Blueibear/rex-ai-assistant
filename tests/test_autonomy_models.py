"""Unit tests for rex.autonomy.models — PlanStep, Plan, and PlannerProtocol."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from rex.autonomy.models import (
    Plan,
    PlannerProtocol,
    PlanStatus,
    PlanStep,
    StepStatus,
)

# ---------------------------------------------------------------------------
# PlanStep tests
# ---------------------------------------------------------------------------


class TestPlanStep:
    def test_minimal_instantiation(self) -> None:
        step = PlanStep(id="s1", tool="web_search", description="Search the web")
        assert step.id == "s1"
        assert step.tool == "web_search"
        assert step.description == "Search the web"
        assert step.args == {}
        assert step.status == StepStatus.PENDING
        assert step.result is None
        assert step.error is None

    def test_full_instantiation(self) -> None:
        step = PlanStep(
            id="s2",
            tool="send_email",
            args={"to": "user@example.com", "subject": "Hello"},
            description="Send a greeting email",
            status=StepStatus.SUCCESS,
            result="message_id: abc123",
            error=None,
        )
        assert step.args == {"to": "user@example.com", "subject": "Hello"}
        assert step.status == StepStatus.SUCCESS
        assert step.result == "message_id: abc123"

    def test_failed_step(self) -> None:
        step = PlanStep(
            id="s3",
            tool="calendar_create_event",
            description="Create event",
            status=StepStatus.FAILED,
            error="Network timeout",
        )
        assert step.status == StepStatus.FAILED
        assert step.error == "Network timeout"

    def test_serialization_round_trip(self) -> None:
        step = PlanStep(
            id="s4",
            tool="time_now",
            description="Get current time",
            status=StepStatus.RUNNING,
        )
        data = step.model_dump()
        restored = PlanStep.model_validate(data)
        assert restored == step

    def test_json_serialization(self) -> None:
        step = PlanStep(id="s5", tool="weather", description="Get weather")
        json_str = step.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "s5"
        assert parsed["tool"] == "weather"
        assert parsed["status"] == "pending"

    def test_all_step_statuses(self) -> None:
        for status in StepStatus:
            step = PlanStep(id="sx", tool="t", description="d", status=status)
            assert step.status == status


# ---------------------------------------------------------------------------
# Plan tests
# ---------------------------------------------------------------------------


class TestPlan:
    def test_minimal_instantiation(self) -> None:
        plan = Plan(id="p1", goal="Find the latest news")
        assert plan.id == "p1"
        assert plan.goal == "Find the latest news"
        assert plan.steps == []
        assert plan.status == PlanStatus.PENDING
        assert isinstance(plan.created_at, datetime)
        assert plan.completed_at is None

    def test_with_steps(self) -> None:
        step = PlanStep(id="s1", tool="web_search", description="Search")
        plan = Plan(id="p2", goal="Research topic", steps=[step])
        assert len(plan.steps) == 1
        assert plan.steps[0].id == "s1"

    def test_completed_plan(self) -> None:
        now = datetime.now(timezone.utc)
        plan = Plan(
            id="p3",
            goal="Send a report",
            status=PlanStatus.COMPLETED,
            completed_at=now,
        )
        assert plan.status == PlanStatus.COMPLETED
        assert plan.completed_at == now

    def test_serialization_round_trip(self) -> None:
        step = PlanStep(id="s1", tool="time_now", description="Get time")
        plan = Plan(id="p4", goal="Check time", steps=[step])
        data = plan.model_dump()
        restored = Plan.model_validate(data)
        assert restored.id == plan.id
        assert restored.goal == plan.goal
        assert len(restored.steps) == 1
        assert restored.steps[0].tool == "time_now"

    def test_json_serialization(self) -> None:
        plan = Plan(id="p5", goal="Do something", status=PlanStatus.RUNNING)
        json_str = plan.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "p5"
        assert parsed["status"] == "running"
        assert parsed["completed_at"] is None

    def test_all_plan_statuses(self) -> None:
        for status in PlanStatus:
            plan = Plan(id="px", goal="g", status=status)
            assert plan.status == status

    def test_created_at_defaults_to_utc(self) -> None:
        plan = Plan(id="p6", goal="time test")
        assert plan.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# PlannerProtocol tests
# ---------------------------------------------------------------------------


class TestPlannerProtocol:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            PlannerProtocol()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_plan(self) -> None:
        class BrokenPlanner(PlannerProtocol):
            pass

        with pytest.raises(TypeError):
            BrokenPlanner()  # type: ignore[abstract]

    def test_concrete_implementation_works(self) -> None:
        class EchoPlanner(PlannerProtocol):
            def plan(self, goal: str, context: dict[str, Any]) -> Plan:
                step = PlanStep(id="s1", tool="echo", description=goal)
                return Plan(id="plan-1", goal=goal, steps=[step])

        planner = EchoPlanner()
        result = planner.plan("Say hello", {"user": "james"})

        assert isinstance(result, Plan)
        assert result.goal == "Say hello"
        assert len(result.steps) == 1
        assert result.steps[0].tool == "echo"

    def test_plan_method_signature(self) -> None:
        """Verify plan() accepts goal str and context dict, returns Plan."""

        class MinimalPlanner(PlannerProtocol):
            def plan(self, goal: str, context: dict[str, Any]) -> Plan:
                return Plan(id="p", goal=goal)

        planner = MinimalPlanner()
        plan = planner.plan("test goal", {})
        assert plan.goal == "test goal"
