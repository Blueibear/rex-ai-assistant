"""Unit tests for US-236: per-step and global budget configuration."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from rex.autonomy.cost_estimator import CostEstimator
from rex.autonomy.models import Plan, PlanStatus, PlanStep, StepStatus
from rex.autonomy.runner import execute_plan
from rex.config import AppConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(sid: str, description: str = "Do something") -> PlanStep:
    return PlanStep(id=sid, tool="noop", description=description)


def _plan(steps: list[PlanStep] | None = None) -> Plan:
    return Plan(id="p1", goal="Test", steps=steps if steps is not None else [_step("s1")])


def _noop(**_: Any) -> str:
    return "done"


# ---------------------------------------------------------------------------
# AppConfig budget fields
# ---------------------------------------------------------------------------


class TestAppConfigBudgetFields:
    def test_budget_per_plan_defaults_to_zero(self) -> None:
        cfg = AppConfig()
        assert cfg.autonomy_budget_per_plan_usd == 0.0

    def test_budget_per_step_defaults_to_zero(self) -> None:
        cfg = AppConfig()
        assert cfg.autonomy_budget_per_step_usd == 0.0

    def test_budget_per_plan_can_be_set(self) -> None:
        cfg = AppConfig(autonomy_budget_per_plan_usd=1.50)
        assert cfg.autonomy_budget_per_plan_usd == pytest.approx(1.50)

    def test_budget_per_step_can_be_set(self) -> None:
        cfg = AppConfig(autonomy_budget_per_step_usd=0.005)
        assert cfg.autonomy_budget_per_step_usd == pytest.approx(0.005)

    def test_config_to_dict_includes_budget_fields(self) -> None:
        cfg = AppConfig(autonomy_budget_per_plan_usd=2.0, autonomy_budget_per_step_usd=0.1)
        d = cfg.to_dict()
        assert d["autonomy_budget_per_plan_usd"] == pytest.approx(2.0)
        assert d["autonomy_budget_per_step_usd"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# execute_plan — per_step_budget_usd
# ---------------------------------------------------------------------------


class TestPerStepBudget:
    def test_zero_per_step_budget_runs_all_steps(self) -> None:
        est = CostEstimator(high_price_per_token=1.0)  # very expensive
        plan = _plan([_step("s1", "hello world")])
        result = execute_plan(plan, {"noop": _noop}, cost_estimator=est, per_step_budget_usd=0.0)
        assert plan.steps[0].status == StepStatus.SUCCESS
        assert result.status == PlanStatus.COMPLETED

    def test_no_estimator_ignores_per_step_budget(self) -> None:
        plan = _plan([_step("s1", "hello world")])
        result = execute_plan(plan, {"noop": _noop}, per_step_budget_usd=0.000001)
        assert plan.steps[0].status == StepStatus.SUCCESS
        assert result.status == PlanStatus.COMPLETED

    def test_step_within_budget_runs(self) -> None:
        est = CostEstimator(high_price_per_token=0.0)  # zero cost always within budget
        plan = _plan([_step("s1")])
        execute_plan(plan, {"noop": _noop}, cost_estimator=est, per_step_budget_usd=1.0)
        assert plan.steps[0].status == StepStatus.SUCCESS

    def test_step_over_budget_is_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        est = CostEstimator(high_price_per_token=1.0)  # very expensive
        plan = _plan([_step("s1", "hello world")])
        with caplog.at_level(logging.WARNING, logger="rex.autonomy.runner"):
            execute_plan(
                plan,
                {"noop": _noop},
                cost_estimator=est,
                per_step_budget_usd=0.000001,
            )
        assert plan.steps[0].status == StepStatus.SKIPPED
        assert any("per-step budget" in r.message for r in caplog.records)

    def test_over_budget_step_warning_includes_step_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        est = CostEstimator(high_price_per_token=1.0)
        plan = _plan([_step("my_step", "expensive")])
        with caplog.at_level(logging.WARNING, logger="rex.autonomy.runner"):
            execute_plan(
                plan,
                {"noop": _noop},
                cost_estimator=est,
                per_step_budget_usd=0.000001,
            )
        assert any("my_step" in r.message for r in caplog.records)

    def test_second_step_runs_when_first_is_within_budget(self) -> None:
        """A zero-cost step (within budget) runs; only expensive steps are skipped."""
        est = CostEstimator(high_price_per_token=0.0)  # zero cost → always within budget
        plan = _plan([_step("s1", "cheap"), _step("s2", "also cheap")])
        execute_plan(plan, {"noop": _noop}, cost_estimator=est, per_step_budget_usd=1.0)
        assert plan.steps[0].status == StepStatus.SUCCESS
        assert plan.steps[1].status == StepStatus.SUCCESS

    def test_tool_not_called_for_skipped_step(self) -> None:
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
            per_step_budget_usd=0.000001,
        )
        assert call_log == []
