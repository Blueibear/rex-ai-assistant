"""Integration tests for US-226: runner persists ExecutionRecord to HistoryStore."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from rex.autonomy.history import HistoryStore
from rex.autonomy.models import Plan, PlanStatus, PlanStep
from rex.autonomy.runner import execute_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(goal: str = "Test goal") -> Plan:
    step = PlanStep(id="s1", tool="noop", description="Do nothing")
    return Plan(id=str(uuid.uuid4()), goal=goal, steps=[step])


def _noop(**_: object) -> str:
    return "done"


def _fail(**_: object) -> str:
    raise RuntimeError("step failed")


def _query_recent(store: HistoryStore, n: int = 10) -> list:  # type: ignore[type-arg]
    """Synchronous helper to query recent records."""
    return asyncio.run(store.recent(n=n))


# ---------------------------------------------------------------------------
# Runner persists history — success case
# ---------------------------------------------------------------------------


class TestRunnerHistorySuccess:
    def test_successful_run_record_in_db(self, tmp_path: Path) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        plan = _make_plan(goal="Complete this task")
        execute_plan(plan, {"noop": _noop}, history_store=store)

        records = _query_recent(store)
        assert len(records) == 1

    def test_successful_run_outcome_is_success(self, tmp_path: Path) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        plan = _make_plan()
        execute_plan(plan, {"noop": _noop}, history_store=store)

        records = _query_recent(store, n=1)
        assert records[0].outcome == "success"

    def test_successful_run_goal_preserved(self, tmp_path: Path) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        plan = _make_plan(goal="My specific goal")
        execute_plan(plan, {"noop": _noop}, history_store=store)

        records = _query_recent(store, n=1)
        assert records[0].goal == "My specific goal"

    def test_successful_run_plan_steps_preserved(self, tmp_path: Path) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        plan = _make_plan()
        execute_plan(plan, {"noop": _noop}, history_store=store)

        records = _query_recent(store, n=1)
        assert len(records[0].plan.steps) == 1
        assert records[0].plan.steps[0].tool == "noop"

    def test_successful_run_duration_positive(self, tmp_path: Path) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        plan = _make_plan()
        execute_plan(plan, {"noop": _noop}, history_store=store)

        records = _query_recent(store, n=1)
        assert records[0].duration_s >= 0.0

    def test_successful_run_replan_count_zero(self, tmp_path: Path) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        plan = _make_plan()
        execute_plan(plan, {"noop": _noop}, history_store=store)

        records = _query_recent(store, n=1)
        assert records[0].replan_count == 0

    def test_successful_run_error_summary_none(self, tmp_path: Path) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        plan = _make_plan()
        execute_plan(plan, {"noop": _noop}, history_store=store)

        records = _query_recent(store, n=1)
        assert records[0].error_summary is None


# ---------------------------------------------------------------------------
# Runner persists history — failure case
# ---------------------------------------------------------------------------


class TestRunnerHistoryFailure:
    def test_failed_run_record_in_db(self, tmp_path: Path) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        plan = _make_plan()
        execute_plan(plan, {"noop": _fail}, history_store=store)

        records = _query_recent(store)
        assert len(records) == 1

    def test_failed_run_outcome_is_failed(self, tmp_path: Path) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        plan = _make_plan()
        execute_plan(plan, {"noop": _fail}, history_store=store)

        records = _query_recent(store, n=1)
        assert records[0].outcome == "failed"

    def test_failed_run_error_summary_populated(self, tmp_path: Path) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        plan = _make_plan()
        execute_plan(plan, {"noop": _fail}, history_store=store)

        records = _query_recent(store, n=1)
        assert records[0].error_summary is not None
        assert "step failed" in records[0].error_summary

    def test_multiple_runs_all_recorded(self, tmp_path: Path) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        for _ in range(3):
            plan = _make_plan()
            execute_plan(plan, {"noop": _noop}, history_store=store)

        records = _query_recent(store, n=10)
        assert len(records) == 3


# ---------------------------------------------------------------------------
# History write failure does not propagate
# ---------------------------------------------------------------------------


class TestRunnerHistoryWriteFailure:
    def test_history_write_failure_does_not_raise(self, tmp_path: Path) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        plan = _make_plan()

        with patch.object(store, "append", side_effect=RuntimeError("DB down")):
            result = execute_plan(plan, {"noop": _noop}, history_store=store)

        assert result.status == PlanStatus.COMPLETED

    def test_history_write_failure_logged_as_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = HistoryStore(db_path=tmp_path / "test.db")
        plan = _make_plan()

        with patch.object(store, "append", side_effect=RuntimeError("DB down")):
            with caplog.at_level(logging.WARNING, logger="rex.autonomy.runner"):
                execute_plan(plan, {"noop": _noop}, history_store=store)

        assert any("history write failed" in r.message for r in caplog.records)

    def test_no_history_store_no_error(self) -> None:
        """Default behaviour (no history_store) is unchanged."""
        plan = _make_plan()
        result = execute_plan(plan, {"noop": _noop})
        assert result.status == PlanStatus.COMPLETED
