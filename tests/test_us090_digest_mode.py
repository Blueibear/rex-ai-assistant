"""Tests for US-090: Digest mode.

Acceptance criteria:
- digest job runs on a configurable interval (default: 60 minutes)
- digest collects all queued medium and low notifications since the last run
- digest payload delivered to dashboard as a single grouped message
- digest job logs output when no real delivery backend is configured (stub)
- digest queue is cleared after each successful run
- Typecheck passes
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.digest_job import DigestJob, DigestJobConfig, DigestResult
from rex.notification_priority import NotificationPriority
from rex.priority_notification_router import (
    PriorityNotificationRouter,
    RoutableNotification,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _notif(
    nid: str,
    priority: NotificationPriority,
    title: str = "T",
    body: str = "",
) -> RoutableNotification:
    return RoutableNotification(
        id=nid,
        priority=priority,
        title=title,
        body=body,
    )


def _queue_medium_low(router: PriorityNotificationRouter, count: int = 3) -> None:
    """Add *count* medium and *count* low notifications to the digest queue."""
    for i in range(count):
        router.route(_notif(f"m{i}", NotificationPriority.MEDIUM, title=f"Medium {i}"))
        router.route(_notif(f"l{i}", NotificationPriority.LOW, title=f"Low {i}"))


# ---------------------------------------------------------------------------
# DigestJobConfig tests
# ---------------------------------------------------------------------------


class TestDigestJobConfig:
    def test_default_interval(self) -> None:
        cfg = DigestJobConfig()
        assert cfg.interval_minutes == 60

    def test_custom_interval(self) -> None:
        cfg = DigestJobConfig(interval_minutes=30)
        assert cfg.interval_minutes == 30

    def test_interval_one_minute(self) -> None:
        cfg = DigestJobConfig(interval_minutes=1)
        assert cfg.interval_minutes == 1

    def test_interval_zero(self) -> None:
        # Edge: 0 is technically valid (run as fast as possible)
        cfg = DigestJobConfig(interval_minutes=0)
        assert cfg.interval_minutes == 0


# ---------------------------------------------------------------------------
# DigestJob construction and config
# ---------------------------------------------------------------------------


class TestDigestJobInit:
    def test_default_config(self) -> None:
        router = PriorityNotificationRouter()
        job = DigestJob(router=router)
        assert job.interval_minutes == 60

    def test_custom_config(self) -> None:
        router = PriorityNotificationRouter()
        cfg = DigestJobConfig(interval_minutes=15)
        job = DigestJob(router=router, config=cfg)
        assert job.interval_minutes == 15

    def test_last_run_initially_none(self) -> None:
        router = PriorityNotificationRouter()
        job = DigestJob(router=router)
        assert job.last_run is None

    def test_configure_replaces_config(self) -> None:
        router = PriorityNotificationRouter()
        job = DigestJob(router=router)
        job.configure(DigestJobConfig(interval_minutes=45))
        assert job.interval_minutes == 45

    def test_configure_from_dict(self) -> None:
        router = PriorityNotificationRouter()
        job = DigestJob(router=router)
        job.configure_from_dict({"interval_minutes": 20})
        assert job.interval_minutes == 20

    def test_configure_from_dict_default_when_missing(self) -> None:
        router = PriorityNotificationRouter()
        job = DigestJob(router=router)
        job.configure_from_dict({})
        assert job.interval_minutes == 60

    def test_config_property_returns_current(self) -> None:
        router = PriorityNotificationRouter()
        cfg = DigestJobConfig(interval_minutes=90)
        job = DigestJob(router=router, config=cfg)
        assert job.config is cfg


# ---------------------------------------------------------------------------
# Empty queue behaviour
# ---------------------------------------------------------------------------


class TestDigestJobEmptyQueue:
    def test_empty_queue_returns_zero_processed(self) -> None:
        router = PriorityNotificationRouter()
        job = DigestJob(router=router)
        result = job.run()
        assert result.entries_processed == 0

    def test_empty_queue_delivered_false(self) -> None:
        router = PriorityNotificationRouter()
        job = DigestJob(router=router)
        result = job.run()
        assert result.delivered is False

    def test_empty_queue_sets_last_run(self) -> None:
        router = PriorityNotificationRouter()
        job = DigestJob(router=router)
        before = datetime.now(timezone.utc)
        result = job.run()
        after = datetime.now(timezone.utc)
        assert job.last_run is not None
        assert before <= job.last_run <= after
        assert before <= result.ran_at <= after

    def test_empty_queue_no_notification_id(self) -> None:
        router = PriorityNotificationRouter()
        job = DigestJob(router=router)
        result = job.run()
        assert result.notification_id is None


# ---------------------------------------------------------------------------
# Stub delivery (no store configured)
# ---------------------------------------------------------------------------


class TestDigestJobStubDelivery:
    def test_stub_logs_digest(self, caplog: pytest.LogCaptureFixture) -> None:
        router = PriorityNotificationRouter()
        _queue_medium_low(router, count=2)
        job = DigestJob(router=router)

        with caplog.at_level(logging.INFO, logger="rex.digest_job"):
            result = job.run()

        assert result.delivered is True
        assert result.entries_processed == 4  # 2 medium + 2 low
        assert any("STUB delivery" in r.message for r in caplog.records)

    def test_stub_returns_notification_id(self) -> None:
        router = PriorityNotificationRouter()
        _queue_medium_low(router, count=1)
        job = DigestJob(router=router)
        result = job.run()
        assert result.notification_id is not None
        assert result.notification_id.startswith("digest_")

    def test_stub_sets_last_run(self) -> None:
        router = PriorityNotificationRouter()
        _queue_medium_low(router, count=1)
        job = DigestJob(router=router)
        before = datetime.now(timezone.utc)
        job.run()
        assert job.last_run is not None
        assert job.last_run >= before

    def test_stub_no_error_field(self) -> None:
        router = PriorityNotificationRouter()
        _queue_medium_low(router, count=1)
        job = DigestJob(router=router)
        result = job.run()
        assert result.error is None


# ---------------------------------------------------------------------------
# Queue cleared after run (AC: digest queue is cleared after each successful run)
# ---------------------------------------------------------------------------


class TestDigestQueueCleared:
    def test_queue_empty_after_run(self) -> None:
        router = PriorityNotificationRouter()
        _queue_medium_low(router, count=3)
        assert len(router.digest_queue) == 6  # 3 medium + 3 low

        job = DigestJob(router=router)
        job.run()

        assert len(router.digest_queue) == 0

    def test_second_run_processes_only_new_entries(self) -> None:
        router = PriorityNotificationRouter()
        _queue_medium_low(router, count=2)

        job = DigestJob(router=router)
        r1 = job.run()
        assert r1.entries_processed == 4

        # Queue is now empty — second run gets nothing.
        r2 = job.run()
        assert r2.entries_processed == 0
        assert r2.delivered is False

    def test_new_entries_after_run_picked_up_next_time(self) -> None:
        router = PriorityNotificationRouter()
        _queue_medium_low(router, count=1)

        job = DigestJob(router=router)
        job.run()  # clears initial 2 entries

        # Add fresh entries after the run.
        router.route(_notif("new1", NotificationPriority.MEDIUM, title="New"))
        r2 = job.run()
        assert r2.entries_processed == 1
        assert r2.delivered is True


# ---------------------------------------------------------------------------
# Collect medium and low (AC: collects queued medium and low since last run)
# ---------------------------------------------------------------------------


class TestDigestCollectsNotifications:
    def test_collects_medium_notifications(self) -> None:
        router = PriorityNotificationRouter()
        for i in range(3):
            router.route(_notif(f"m{i}", NotificationPriority.MEDIUM, title=f"M{i}"))

        job = DigestJob(router=router)
        result = job.run()
        assert result.entries_processed == 3

    def test_collects_low_notifications(self) -> None:
        router = PriorityNotificationRouter()
        for i in range(5):
            router.route(_notif(f"l{i}", NotificationPriority.LOW, title=f"L{i}"))

        job = DigestJob(router=router)
        result = job.run()
        assert result.entries_processed == 5

    def test_collects_mixed_medium_and_low(self) -> None:
        router = PriorityNotificationRouter()
        _queue_medium_low(router, count=4)

        job = DigestJob(router=router)
        result = job.run()
        assert result.entries_processed == 8  # 4 medium + 4 low

    def test_critical_not_in_digest_queue(self) -> None:
        router = PriorityNotificationRouter()
        # Critical goes to immediate dispatch, NOT the queue.
        router.route(_notif("c1", NotificationPriority.CRITICAL))
        router.route(_notif("m1", NotificationPriority.MEDIUM))

        assert len(router.digest_queue) == 1  # only medium queued

        job = DigestJob(router=router)
        result = job.run()
        assert result.entries_processed == 1

    def test_high_not_in_digest_queue(self) -> None:
        router = PriorityNotificationRouter()
        router.route(_notif("h1", NotificationPriority.HIGH))
        router.route(_notif("l1", NotificationPriority.LOW))

        assert len(router.digest_queue) == 1  # only low queued

        job = DigestJob(router=router)
        result = job.run()
        assert result.entries_processed == 1


# ---------------------------------------------------------------------------
# Grouped message delivery to dashboard store
# ---------------------------------------------------------------------------


class TestDigestDashboardDelivery:
    def test_single_write_to_store(self, tmp_path: Path) -> None:
        from rex.dashboard_store import DashboardStore

        store = DashboardStore(db_path=tmp_path / "notifs.db")

        router = PriorityNotificationRouter()
        _queue_medium_low(router, count=2)

        job = DigestJob(router=router, store=store)
        result = job.run()

        assert result.delivered is True
        assert result.entries_processed == 4

        # Exactly one notification in the store (grouped message).
        rows = store.query_recent(limit=10)
        assert len(rows) == 1
        assert rows[0].id == result.notification_id

    def test_grouped_message_title_contains_count(self, tmp_path: Path) -> None:
        from rex.dashboard_store import DashboardStore

        store = DashboardStore(db_path=tmp_path / "notifs.db")
        router = PriorityNotificationRouter()
        _queue_medium_low(router, count=3)

        job = DigestJob(router=router, store=store)
        result = job.run()

        rows = store.query_recent(limit=1)
        assert "6" in rows[0].title  # "Notification digest (6 items)"
        assert result.notification_id is not None

    def test_grouped_body_lists_individual_titles(self, tmp_path: Path) -> None:
        from rex.dashboard_store import DashboardStore

        store = DashboardStore(db_path=tmp_path / "notifs.db")
        router = PriorityNotificationRouter()
        router.route(
            _notif("m1", NotificationPriority.MEDIUM, title="Alpha update", body="")
        )
        router.route(_notif("l1", NotificationPriority.LOW, title="Beta news", body=""))

        job = DigestJob(router=router, store=store)
        job.run()

        rows = store.query_recent(limit=1)
        body = rows[0].body
        assert "Alpha update" in body
        assert "Beta news" in body

    def test_delivered_true_with_store(self, tmp_path: Path) -> None:
        from rex.dashboard_store import DashboardStore

        store = DashboardStore(db_path=tmp_path / "notifs.db")
        router = PriorityNotificationRouter()
        router.route(_notif("l1", NotificationPriority.LOW, title="Low alert"))

        job = DigestJob(router=router, store=store)
        result = job.run()

        assert result.delivered is True
        assert result.error is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestDigestJobErrorHandling:
    def test_store_failure_captured_in_result(self) -> None:
        bad_store = MagicMock()
        bad_store.write.side_effect = RuntimeError("DB unavailable")

        router = PriorityNotificationRouter()
        _queue_medium_low(router, count=1)

        job = DigestJob(router=router, store=bad_store)
        result = job.run()

        assert result.delivered is False
        assert result.error is not None
        assert "DB unavailable" in result.error

    def test_store_failure_does_not_raise(self) -> None:
        bad_store = MagicMock()
        bad_store.write.side_effect = RuntimeError("oops")

        router = PriorityNotificationRouter()
        router.route(_notif("m1", NotificationPriority.MEDIUM))

        job = DigestJob(router=router, store=bad_store)
        # Should not raise.
        result = job.run()
        assert isinstance(result, DigestResult)

    def test_store_failure_sets_last_run(self) -> None:
        bad_store = MagicMock()
        bad_store.write.side_effect = RuntimeError("oops")

        router = PriorityNotificationRouter()
        router.route(_notif("m1", NotificationPriority.MEDIUM))

        job = DigestJob(router=router, store=bad_store)
        job.run()

        assert job.last_run is not None


# ---------------------------------------------------------------------------
# Logging (stub behaviour)
# ---------------------------------------------------------------------------


class TestDigestJobLogging:
    def test_empty_queue_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        router = PriorityNotificationRouter()
        job = DigestJob(router=router)

        with caplog.at_level(logging.INFO, logger="rex.digest_job"):
            job.run()

        assert any("no queued notifications" in r.message for r in caplog.records)

    def test_delivery_success_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        router = PriorityNotificationRouter()
        router.route(_notif("m1", NotificationPriority.MEDIUM, title="Msg"))

        job = DigestJob(router=router)

        with caplog.at_level(logging.INFO, logger="rex.digest_job"):
            job.run()

        assert any("Digest delivered" in r.message for r in caplog.records)

    def test_stub_log_contains_title_and_body(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        router = PriorityNotificationRouter()
        router.route(_notif("m1", NotificationPriority.MEDIUM, title="Hello world"))

        job = DigestJob(router=router)

        with caplog.at_level(logging.INFO, logger="rex.digest_job"):
            job.run()

        combined = " ".join(r.message for r in caplog.records)
        assert "Hello world" in combined


# ---------------------------------------------------------------------------
# DigestResult dataclass
# ---------------------------------------------------------------------------


class TestDigestResult:
    def test_default_values(self) -> None:
        result = DigestResult()
        assert result.entries_processed == 0
        assert result.delivered is False
        assert result.notification_id is None
        assert result.error is None
        assert isinstance(result.ran_at, datetime)

    def test_custom_values(self) -> None:
        ts = datetime(2026, 3, 11, 10, 0, tzinfo=timezone.utc)
        result = DigestResult(
            ran_at=ts,
            entries_processed=5,
            delivered=True,
            notification_id="digest_abc",
        )
        assert result.entries_processed == 5
        assert result.delivered is True
        assert result.notification_id == "digest_abc"
        assert result.ran_at == ts
