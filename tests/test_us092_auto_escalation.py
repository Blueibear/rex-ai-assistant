"""Tests for US-092: Auto-escalation of unacknowledged high-priority notifications."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rex.escalation_job import (
    EscalationAttempt,
    EscalationConfig,
    EscalationJob,
    EscalationResult,
    TrackedNotification,
)
from rex.notification_priority import NotificationPriority
from rex.priority_notification_router import RoutableNotification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _notif(
    nid: str = "n1",
    priority: NotificationPriority = NotificationPriority.HIGH,
    title: str = "Test notification",
    body: str = "",
    created_at: datetime | None = None,
) -> RoutableNotification:
    return RoutableNotification(
        id=nid,
        priority=priority,
        title=title,
        body=body,
        created_at=created_at or datetime.now(timezone.utc),
    )


def _old_notif(
    nid: str = "n1",
    priority: NotificationPriority = NotificationPriority.HIGH,
    minutes_ago: int = 20,
    title: str = "Old notification",
) -> RoutableNotification:
    """Notification created `minutes_ago` minutes in the past."""
    created_at = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return _notif(nid=nid, priority=priority, title=title, created_at=created_at)


# ---------------------------------------------------------------------------
# EscalationConfig tests
# ---------------------------------------------------------------------------


class TestEscalationConfig:
    def test_defaults(self) -> None:
        cfg = EscalationConfig()
        assert cfg.timeout_minutes_by_priority == {NotificationPriority.HIGH: 15}
        assert cfg.max_attempts == 3

    def test_from_dict_defaults(self) -> None:
        cfg = EscalationConfig.from_dict({})
        assert cfg.timeout_minutes_by_priority[NotificationPriority.HIGH] == 15
        assert cfg.max_attempts == 3

    def test_from_dict_custom_timeout(self) -> None:
        cfg = EscalationConfig.from_dict(
            {"timeout_minutes_by_priority": {"high": 30}, "max_attempts": 5}
        )
        assert cfg.timeout_minutes_by_priority[NotificationPriority.HIGH] == 30
        assert cfg.max_attempts == 5

    def test_from_dict_unknown_priority_maps_to_medium(self) -> None:
        cfg = EscalationConfig.from_dict(
            {"timeout_minutes_by_priority": {"unknown_level": 10}}
        )
        assert NotificationPriority.MEDIUM in cfg.timeout_minutes_by_priority

    def test_from_dict_multiple_priorities(self) -> None:
        cfg = EscalationConfig.from_dict(
            {"timeout_minutes_by_priority": {"high": 15, "critical": 5}}
        )
        assert cfg.timeout_minutes_by_priority[NotificationPriority.HIGH] == 15
        assert cfg.timeout_minutes_by_priority[NotificationPriority.CRITICAL] == 5


# ---------------------------------------------------------------------------
# EscalationJob.track tests
# ---------------------------------------------------------------------------


class TestEscalationJobTrack:
    def test_track_stores_notification(self) -> None:
        job = EscalationJob()
        n = _notif()
        job.track(n)
        assert len(job.tracked) == 1
        assert job.tracked[0].notification_id == "n1"

    def test_track_duplicate_ignored(self) -> None:
        job = EscalationJob()
        n = _notif()
        job.track(n)
        job.track(n)
        assert len(job.tracked) == 1

    def test_track_multiple_notifications(self) -> None:
        job = EscalationJob()
        job.track(_notif("a"))
        job.track(_notif("b"))
        job.track(_notif("c"))
        assert len(job.tracked) == 3

    def test_tracked_initial_state(self) -> None:
        job = EscalationJob()
        n = _notif()
        job.track(n)
        record = job.tracked[0]
        assert record.attempt_count == 0
        assert record.acknowledged is False
        assert record.last_escalated_at is None

    def test_tracked_returns_copy(self) -> None:
        job = EscalationJob()
        job.track(_notif())
        snapshot = job.tracked
        job.track(_notif("n2"))
        assert len(snapshot) == 1


# ---------------------------------------------------------------------------
# EscalationJob.acknowledge tests
# ---------------------------------------------------------------------------


class TestEscalationJobAcknowledge:
    def test_acknowledge_sets_flag(self) -> None:
        job = EscalationJob()
        job.track(_notif())
        job.acknowledge("n1")
        assert job.tracked[0].acknowledged is True

    def test_acknowledge_unknown_id_no_error(self) -> None:
        job = EscalationJob()
        job.acknowledge("nonexistent")  # must not raise

    def test_acknowledge_stops_future_escalation(self) -> None:
        job = EscalationJob()
        n = _old_notif(minutes_ago=20)
        job.track(n)
        job.acknowledge(n.id)
        result = job.run()
        assert result.escalated == []


# ---------------------------------------------------------------------------
# EscalationJob.run — escalation timing
# ---------------------------------------------------------------------------


class TestEscalationJobRun:
    def test_run_returns_result_type(self) -> None:
        job = EscalationJob()
        result = job.run()
        assert isinstance(result, EscalationResult)

    def test_run_no_notifications_empty_result(self) -> None:
        job = EscalationJob()
        result = job.run()
        assert result.escalated == []
        assert result.max_reached_ids == []

    def test_run_notification_not_yet_due(self) -> None:
        job = EscalationJob()
        # Created 5 minutes ago, timeout is 15 minutes.
        n = _old_notif(minutes_ago=5)
        job.track(n)
        result = job.run()
        assert result.escalated == []

    def test_run_notification_past_timeout_escalates(self) -> None:
        job = EscalationJob()
        n = _old_notif(minutes_ago=20)
        job.track(n)
        result = job.run()
        assert len(result.escalated) == 1
        attempt = result.escalated[0]
        assert attempt.notification_id == n.id
        assert attempt.attempt_number == 1

    def test_run_escalation_attempt_has_timestamp(self) -> None:
        job = EscalationJob()
        n = _old_notif(minutes_ago=20)
        job.track(n)
        before = datetime.now(timezone.utc)
        result = job.run()
        after = datetime.now(timezone.utc)
        attempt = result.escalated[0]
        assert before <= attempt.escalated_at <= after

    def test_run_escalation_attempt_has_priority(self) -> None:
        job = EscalationJob()
        n = _old_notif(priority=NotificationPriority.HIGH, minutes_ago=20)
        job.track(n)
        result = job.run()
        assert result.escalated[0].priority == NotificationPriority.HIGH

    def test_run_escalation_attempt_has_title(self) -> None:
        job = EscalationJob()
        n = _old_notif(title="Disk almost full", minutes_ago=20)
        job.track(n)
        result = job.run()
        assert result.escalated[0].title == "Disk almost full"

    def test_run_increments_attempt_count(self) -> None:
        job = EscalationJob()
        n = _old_notif(minutes_ago=20)
        job.track(n)
        job.run()
        assert job.tracked[0].attempt_count == 1

    def test_run_updates_last_escalated_at(self) -> None:
        job = EscalationJob()
        n = _old_notif(minutes_ago=20)
        job.track(n)
        job.run()
        assert job.tracked[0].last_escalated_at is not None

    def test_run_second_call_before_timeout_no_escalation(self) -> None:
        job = EscalationJob()
        n = _old_notif(minutes_ago=20)
        job.track(n)
        job.run()
        # Second run immediately — not enough time since last escalation.
        result2 = job.run()
        assert result2.escalated == []

    def test_run_low_priority_not_escalated(self) -> None:
        job = EscalationJob()
        n = _old_notif(priority=NotificationPriority.LOW, minutes_ago=20)
        job.track(n)
        result = job.run()
        assert result.escalated == []

    def test_run_medium_priority_not_escalated_by_default(self) -> None:
        job = EscalationJob()
        n = _old_notif(priority=NotificationPriority.MEDIUM, minutes_ago=20)
        job.track(n)
        result = job.run()
        assert result.escalated == []

    def test_run_critical_priority_not_escalated_by_default(self) -> None:
        """Critical notifications bypass quiet hours but are NOT in the default
        escalation set (only HIGH is escalated by default)."""
        job = EscalationJob()
        n = _old_notif(priority=NotificationPriority.CRITICAL, minutes_ago=20)
        job.track(n)
        result = job.run()
        assert result.escalated == []


# ---------------------------------------------------------------------------
# Max attempts
# ---------------------------------------------------------------------------


class TestMaxAttempts:
    def _job_with_fast_timeout(self, max_attempts: int = 3) -> EscalationJob:
        cfg = EscalationConfig(
            timeout_minutes_by_priority={NotificationPriority.HIGH: 0},
            max_attempts=max_attempts,
        )
        return EscalationJob(config=cfg)

    def test_escalation_stops_at_max_attempts(self) -> None:
        job = self._job_with_fast_timeout(max_attempts=3)
        n = _notif()
        job.track(n)
        for _ in range(5):
            job.run()
        assert job.tracked[0].attempt_count == 3

    def test_max_reached_ids_populated(self) -> None:
        job = self._job_with_fast_timeout(max_attempts=1)
        n = _notif()
        job.track(n)
        result = job.run()
        assert n.id in result.max_reached_ids

    def test_max_reached_ids_empty_when_not_at_cap(self) -> None:
        job = self._job_with_fast_timeout(max_attempts=3)
        n = _notif()
        job.track(n)
        result = job.run()
        assert result.max_reached_ids == []

    def test_subsequent_run_after_max_no_further_escalation(self) -> None:
        job = self._job_with_fast_timeout(max_attempts=2)
        n = _notif()
        job.track(n)
        job.run()
        job.run()
        result3 = job.run()
        assert result3.escalated == []

    def test_attempt_number_increments_correctly(self) -> None:
        job = self._job_with_fast_timeout(max_attempts=3)
        n = _notif()
        job.track(n)
        r1 = job.run()
        r2 = job.run()
        r3 = job.run()
        assert r1.escalated[0].attempt_number == 1
        assert r2.escalated[0].attempt_number == 2
        assert r3.escalated[0].attempt_number == 3


# ---------------------------------------------------------------------------
# Cumulative attempts log
# ---------------------------------------------------------------------------


class TestCumulativeAttempts:
    def test_attempts_property_empty_initially(self) -> None:
        job = EscalationJob()
        assert job.attempts == []

    def test_attempts_accumulates_across_runs(self) -> None:
        cfg = EscalationConfig(
            timeout_minutes_by_priority={NotificationPriority.HIGH: 0},
            max_attempts=3,
        )
        job = EscalationJob(config=cfg)
        n = _notif()
        job.track(n)
        job.run()
        job.run()
        assert len(job.attempts) == 2

    def test_attempts_returns_copy(self) -> None:
        cfg = EscalationConfig(
            timeout_minutes_by_priority={NotificationPriority.HIGH: 0},
            max_attempts=3,
        )
        job = EscalationJob(config=cfg)
        job.track(_notif())
        snap = job.attempts
        job.run()
        assert len(snap) == 0  # snapshot taken before run


# ---------------------------------------------------------------------------
# configure / configure_from_dict
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_configure_replaces_config(self) -> None:
        job = EscalationJob()
        new_cfg = EscalationConfig(
            timeout_minutes_by_priority={NotificationPriority.HIGH: 30},
            max_attempts=5,
        )
        job.configure(new_cfg)
        assert job.config.max_attempts == 5
        assert job.config.timeout_minutes_by_priority[NotificationPriority.HIGH] == 30

    def test_configure_from_dict(self) -> None:
        job = EscalationJob()
        job.configure_from_dict(
            {"timeout_minutes_by_priority": {"high": 10}, "max_attempts": 2}
        )
        assert job.config.max_attempts == 2

    def test_configure_takes_effect_next_run(self) -> None:
        # Start with 60-minute timeout (notification won't escalate yet).
        job = EscalationJob(
            config=EscalationConfig(
                timeout_minutes_by_priority={NotificationPriority.HIGH: 60},
                max_attempts=3,
            )
        )
        n = _old_notif(minutes_ago=20)
        job.track(n)
        r1 = job.run()
        assert r1.escalated == []

        # Switch to 0-minute timeout.
        job.configure(
            EscalationConfig(
                timeout_minutes_by_priority={NotificationPriority.HIGH: 0},
                max_attempts=3,
            )
        )
        r2 = job.run()
        assert len(r2.escalated) == 1


# ---------------------------------------------------------------------------
# untrack
# ---------------------------------------------------------------------------


class TestUntrack:
    def test_untrack_removes_notification(self) -> None:
        job = EscalationJob()
        job.track(_notif())
        job.untrack("n1")
        assert job.tracked == []

    def test_untrack_unknown_no_error(self) -> None:
        job = EscalationJob()
        job.untrack("does_not_exist")  # must not raise

    def test_untracked_notification_not_escalated(self) -> None:
        cfg = EscalationConfig(
            timeout_minutes_by_priority={NotificationPriority.HIGH: 0},
            max_attempts=3,
        )
        job = EscalationJob(config=cfg)
        job.track(_notif())
        job.untrack("n1")
        result = job.run()
        assert result.escalated == []


# ---------------------------------------------------------------------------
# Multiple notifications in one run
# ---------------------------------------------------------------------------


class TestMultipleNotifications:
    def test_only_eligible_notifications_escalated(self) -> None:
        cfg = EscalationConfig(
            timeout_minutes_by_priority={NotificationPriority.HIGH: 0},
            max_attempts=3,
        )
        job = EscalationJob(config=cfg)
        job.track(_notif("due", priority=NotificationPriority.HIGH))
        job.track(
            _notif(
                "not_due",
                priority=NotificationPriority.LOW,
            )
        )
        result = job.run()
        escalated_ids = [a.notification_id for a in result.escalated]
        assert "due" in escalated_ids
        assert "not_due" not in escalated_ids

    def test_multiple_high_priority_notifications_escalated(self) -> None:
        cfg = EscalationConfig(
            timeout_minutes_by_priority={NotificationPriority.HIGH: 0},
            max_attempts=3,
        )
        job = EscalationJob(config=cfg)
        job.track(_notif("a"))
        job.track(_notif("b"))
        result = job.run()
        assert len(result.escalated) == 2

    def test_acknowledged_notification_skipped_alongside_others(self) -> None:
        cfg = EscalationConfig(
            timeout_minutes_by_priority={NotificationPriority.HIGH: 0},
            max_attempts=3,
        )
        job = EscalationJob(config=cfg)
        job.track(_notif("ack"))
        job.track(_notif("active"))
        job.acknowledge("ack")
        result = job.run()
        escalated_ids = [a.notification_id for a in result.escalated]
        assert "ack" not in escalated_ids
        assert "active" in escalated_ids
