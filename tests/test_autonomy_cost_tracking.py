"""Unit tests for US-233: per-step token/cost tracking and Plan.total_cost_usd."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rex.autonomy.history import ExecutionRecord, HistoryStore, OutcomeType
from rex.autonomy.models import Plan, PlanStep, StepStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(
    sid: str,
    *,
    tokens_used: int | None = None,
    cost_usd: float | None = None,
    status: StepStatus = StepStatus.PENDING,
) -> PlanStep:
    return PlanStep(
        id=sid,
        tool="noop",
        description=f"Step {sid}",
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        status=status,
    )


def _plan(steps: list[PlanStep]) -> Plan:
    return Plan(id="p1", goal="Test goal", steps=steps)


def _record(plan: Plan, outcome: OutcomeType = "success") -> ExecutionRecord:
    return ExecutionRecord(
        goal=plan.goal,
        plan=plan,
        outcome=outcome,
        duration_s=1.0,
        total_cost_usd=plan.total_cost_usd,
    )


# ---------------------------------------------------------------------------
# PlanStep — new fields default to None
# ---------------------------------------------------------------------------


class TestPlanStepDefaults:
    def test_tokens_used_defaults_to_none(self) -> None:
        step = _step("s1")
        assert step.tokens_used is None

    def test_cost_usd_defaults_to_none(self) -> None:
        step = _step("s1")
        assert step.cost_usd is None

    def test_tokens_used_can_be_set(self) -> None:
        step = _step("s1", tokens_used=512)
        assert step.tokens_used == 512

    def test_cost_usd_can_be_set(self) -> None:
        step = _step("s1", cost_usd=0.0015)
        assert pytest.approx(step.cost_usd) == 0.0015


# ---------------------------------------------------------------------------
# Plan.total_cost_usd — computed property
# ---------------------------------------------------------------------------


class TestPlanTotalCostUsd:
    def test_no_steps_returns_zero(self) -> None:
        plan = _plan([])
        assert plan.total_cost_usd == 0.0

    def test_all_none_costs_returns_zero(self) -> None:
        plan = _plan([_step("s1"), _step("s2")])
        assert plan.total_cost_usd == 0.0

    def test_single_step_with_cost(self) -> None:
        plan = _plan([_step("s1", cost_usd=0.01)])
        assert pytest.approx(plan.total_cost_usd) == 0.01

    def test_multiple_steps_sums_costs(self) -> None:
        steps = [
            _step("s1", cost_usd=0.01),
            _step("s2", cost_usd=0.02),
            _step("s3", cost_usd=0.03),
        ]
        plan = _plan(steps)
        assert pytest.approx(plan.total_cost_usd) == 0.06

    def test_mixed_none_and_known_costs(self) -> None:
        steps = [
            _step("s1", cost_usd=0.01),
            _step("s2"),  # cost_usd=None — should be ignored
            _step("s3", cost_usd=0.02),
        ]
        plan = _plan(steps)
        assert pytest.approx(plan.total_cost_usd) == 0.03

    def test_zero_cost_step_included(self) -> None:
        steps = [_step("s1", cost_usd=0.0), _step("s2", cost_usd=0.05)]
        plan = _plan(steps)
        assert pytest.approx(plan.total_cost_usd) == 0.05


# ---------------------------------------------------------------------------
# ExecutionRecord — total_cost_usd field
# ---------------------------------------------------------------------------


class TestExecutionRecordCostField:
    def test_default_is_zero(self) -> None:
        plan = _plan([])
        rec = ExecutionRecord(goal="g", plan=plan, outcome="success", duration_s=1.0)
        assert rec.total_cost_usd == 0.0

    def test_field_can_be_set(self) -> None:
        plan = _plan([_step("s1", cost_usd=0.05)])
        rec = _record(plan)
        assert pytest.approx(rec.total_cost_usd) == 0.05

    def test_field_persists_through_model_copy(self) -> None:
        plan = _plan([_step("s1", cost_usd=0.07)])
        rec = _record(plan)
        copy = rec.model_copy()
        assert pytest.approx(copy.total_cost_usd) == 0.07


# ---------------------------------------------------------------------------
# HistoryStore — persists and retrieves total_cost_usd
# ---------------------------------------------------------------------------


class TestHistoryStoreCostPersistence:
    def test_cost_persisted_and_retrieved(self, tmp_path: Path) -> None:
        db = tmp_path / "history.db"
        store = HistoryStore(db_path=db)

        steps = [_step("s1", cost_usd=0.03), _step("s2", cost_usd=0.04)]
        plan = _plan(steps)
        rec = _record(plan)

        asyncio.run(store.append(rec))
        rows = asyncio.run(store.recent(1))

        assert len(rows) == 1
        assert pytest.approx(rows[0].total_cost_usd) == 0.07

    def test_zero_cost_stored_correctly(self, tmp_path: Path) -> None:
        db = tmp_path / "history.db"
        store = HistoryStore(db_path=db)

        plan = _plan([_step("s1")])  # no cost
        rec = _record(plan)

        asyncio.run(store.append(rec))
        rows = asyncio.run(store.recent(1))

        assert rows[0].total_cost_usd == 0.0

    def test_migration_adds_column_to_existing_db(self, tmp_path: Path) -> None:
        """Simulate a pre-US-233 DB (without total_cost_usd column)."""
        import aiosqlite

        db = tmp_path / "history.db"

        # Create old-schema DB manually.
        async def _create_old() -> None:
            conn = await aiosqlite.connect(str(db))
            await conn.execute("""
                CREATE TABLE execution_records (
                    id TEXT PRIMARY KEY, goal TEXT NOT NULL,
                    plan_json TEXT NOT NULL, outcome TEXT NOT NULL,
                    duration_s REAL NOT NULL, replan_count INTEGER NOT NULL,
                    error_summary TEXT, timestamp TEXT NOT NULL
                )
                """)
            await conn.commit()
            await conn.close()

        asyncio.run(_create_old())

        # Open via HistoryStore — migration should add the missing column.
        store = HistoryStore(db_path=db)
        plan = _plan([_step("s1", cost_usd=0.01)])
        rec = _record(plan)
        asyncio.run(store.append(rec))

        rows = asyncio.run(store.recent(1))
        assert pytest.approx(rows[0].total_cost_usd) == 0.01
