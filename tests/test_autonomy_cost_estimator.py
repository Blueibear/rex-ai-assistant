"""Unit tests for US-234: CostEstimator and runner budget integration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from rex.autonomy.cost_estimator import CostEstimate, CostEstimator, check_budget
from rex.autonomy.llm_planner import LLMPlanner
from rex.autonomy.models import Plan, PlanStatus, PlanStep
from rex.autonomy.runner import execute_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(sid: str, description: str = "Do something") -> PlanStep:
    return PlanStep(id=sid, tool="noop", description=description)


def _plan(steps: list[PlanStep] | None = None) -> Plan:
    return Plan(id="p1", goal="Test goal", steps=steps if steps is not None else [_step("s1")])


def _noop(**_: Any) -> str:
    return "done"


def _mock_planner_for(plan: Plan) -> LLMPlanner:
    backend = MagicMock()
    backend.generate.return_value = json.dumps(
        [{"tool": "noop", "args": {}, "description": s.description} for s in plan.steps]
    )
    return LLMPlanner(tools=[], backend=backend)


# ---------------------------------------------------------------------------
# CostEstimate model
# ---------------------------------------------------------------------------


class TestCostEstimateModel:
    def test_has_low_usd(self) -> None:
        est = CostEstimate(low_usd=0.01, high_usd=0.05, step_count=2)
        assert est.low_usd == pytest.approx(0.01)

    def test_has_high_usd(self) -> None:
        est = CostEstimate(low_usd=0.01, high_usd=0.05, step_count=2)
        assert est.high_usd == pytest.approx(0.05)

    def test_has_step_count(self) -> None:
        est = CostEstimate(low_usd=0.0, high_usd=0.0, step_count=3)
        assert est.step_count == 3

    def test_low_leq_high(self) -> None:
        estimator = CostEstimator()
        result = estimator.estimate(_plan([_step("s1"), _step("s2")]))
        assert result.low_usd <= result.high_usd


# ---------------------------------------------------------------------------
# CostEstimator.estimate
# ---------------------------------------------------------------------------


class TestCostEstimatorEstimate:
    def test_returns_cost_estimate_instance(self) -> None:
        est = CostEstimator()
        result = est.estimate(_plan())
        assert isinstance(result, CostEstimate)

    def test_step_count_matches_plan(self) -> None:
        est = CostEstimator()
        plan = _plan([_step("s1"), _step("s2"), _step("s3")])
        result = est.estimate(plan)
        assert result.step_count == 3

    def test_empty_plan_returns_zero_cost(self) -> None:
        est = CostEstimator()
        plan = _plan([])
        result = est.estimate(plan)
        assert result.low_usd == 0.0
        assert result.high_usd == 0.0

    def test_longer_description_increases_estimate(self) -> None:
        est = CostEstimator()
        short = est.estimate(_plan([_step("s1", "Do it")]))
        long_ = est.estimate(_plan([_step("s1", " ".join(["word"] * 50))]))
        assert long_.high_usd > short.high_usd

    def test_more_steps_increases_estimate(self) -> None:
        est = CostEstimator()
        one = est.estimate(_plan([_step("s1")]))
        three = est.estimate(_plan([_step("s1"), _step("s2"), _step("s3")]))
        assert three.high_usd > one.high_usd

    def test_custom_prices_applied(self) -> None:
        est = CostEstimator(low_price_per_token=0.0, high_price_per_token=0.0)
        result = est.estimate(_plan([_step("s1", "hello world")]))
        assert result.low_usd == 0.0
        assert result.high_usd == 0.0


# ---------------------------------------------------------------------------
# check_budget helper
# ---------------------------------------------------------------------------


class TestCheckBudget:
    def test_zero_budget_always_proceeds(self) -> None:
        est = CostEstimator()
        plan = _plan([_step("s1")])
        assert check_budget(est, plan, budget_usd=0.0) is True

    def test_negative_budget_always_proceeds(self) -> None:
        est = CostEstimator()
        plan = _plan([_step("s1")])
        assert check_budget(est, plan, budget_usd=-1.0) is True

    def test_within_budget_proceeds_without_callback(self) -> None:
        est = CostEstimator(high_price_per_token=0.0)  # zero cost
        plan = _plan([_step("s1")])
        assert check_budget(est, plan, budget_usd=1.0) is True

    def test_over_budget_calls_callback(self) -> None:
        est = CostEstimator(high_price_per_token=1.0)  # very expensive
        plan = _plan([_step("s1", "hello")])
        callback = MagicMock(return_value=True)
        result = check_budget(est, plan, budget_usd=0.000001, on_budget_exceeded=callback)
        callback.assert_called_once()
        assert result is True

    def test_callback_returning_false_aborts(self) -> None:
        est = CostEstimator(high_price_per_token=1.0)
        plan = _plan([_step("s1", "hello")])
        callback = MagicMock(return_value=False)
        result = check_budget(est, plan, budget_usd=0.000001, on_budget_exceeded=callback)
        assert result is False

    def test_callback_receives_cost_estimate(self) -> None:
        est = CostEstimator()
        plan = _plan([_step("s1")])
        received: list[CostEstimate] = []

        def _cb(estimate: CostEstimate) -> bool:
            received.append(estimate)
            return True

        # Force over-budget by using a very tight budget.
        check_budget(est, plan, budget_usd=0.0000001, on_budget_exceeded=_cb)
        assert len(received) == 1
        assert isinstance(received[0], CostEstimate)


# ---------------------------------------------------------------------------
# execute_plan — budget integration
# ---------------------------------------------------------------------------


class TestExecutePlanBudget:
    def test_no_cost_estimator_runs_normally(self) -> None:
        plan = _plan()
        result = execute_plan(plan, {"noop": _noop})
        assert result.status == PlanStatus.COMPLETED

    def test_zero_budget_runs_normally(self) -> None:
        est = CostEstimator(high_price_per_token=1.0)  # would fail if checked
        plan = _plan()
        result = execute_plan(plan, {"noop": _noop}, cost_estimator=est, budget_usd=0.0)
        assert result.status == PlanStatus.COMPLETED

    def test_within_budget_runs_normally(self) -> None:
        est = CostEstimator(high_price_per_token=0.0)  # zero cost always within budget
        plan = _plan()
        result = execute_plan(plan, {"noop": _noop}, cost_estimator=est, budget_usd=1.0)
        assert result.status == PlanStatus.COMPLETED

    def test_over_budget_approved_by_callback_runs(self) -> None:
        est = CostEstimator(high_price_per_token=1.0)  # very expensive
        plan = _plan([_step("s1", "hello world")])
        result = execute_plan(
            plan,
            {"noop": _noop},
            cost_estimator=est,
            budget_usd=0.000001,
            on_budget_exceeded=lambda _e: True,
        )
        assert result.status == PlanStatus.COMPLETED

    def test_over_budget_denied_by_callback_cancels(self) -> None:
        est = CostEstimator(high_price_per_token=1.0)
        plan = _plan([_step("s1", "hello world")])
        result = execute_plan(
            plan,
            {"noop": _noop},
            cost_estimator=est,
            budget_usd=0.000001,
            on_budget_exceeded=lambda _e: False,
        )
        assert result.status == PlanStatus.CANCELLED

    def test_cancelled_plan_does_not_execute_steps(self) -> None:
        call_log: list[str] = []

        def _tracked(**_: Any) -> str:
            call_log.append("called")
            return "done"

        est = CostEstimator(high_price_per_token=1.0)
        plan = _plan([_step("s1", "hello world")])
        execute_plan(
            plan,
            {"noop": _tracked},
            cost_estimator=est,
            budget_usd=0.000001,
            on_budget_exceeded=lambda _e: False,
        )
        assert call_log == []
