"""Tests for US-088: Notification priority levels.

Acceptance criteria:
- `NotificationPriority` enum defined with values: critical, high, medium, low
- all notification creation paths accept a `priority` parameter
- priority stored alongside notification record in the database
- existing notifications without a stored priority default to `medium` on read
- Typecheck passes
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rex.notification_priority import NotificationPriority
from rex.dashboard_store import DashboardStore, DashboardNotification


# ---------------------------------------------------------------------------
# NotificationPriority enum
# ---------------------------------------------------------------------------


class TestNotificationPriorityEnum:
    def test_critical_value(self) -> None:
        assert NotificationPriority.CRITICAL.value == "critical"

    def test_high_value(self) -> None:
        assert NotificationPriority.HIGH.value == "high"

    def test_medium_value(self) -> None:
        assert NotificationPriority.MEDIUM.value == "medium"

    def test_low_value(self) -> None:
        assert NotificationPriority.LOW.value == "low"

    def test_four_members_total(self) -> None:
        assert len(NotificationPriority) == 4

    def test_all_values_are_strings(self) -> None:
        for member in NotificationPriority:
            assert isinstance(member.value, str)

    def test_label_is_title_case(self) -> None:
        assert NotificationPriority.CRITICAL.label == "Critical"
        assert NotificationPriority.HIGH.label == "High"
        assert NotificationPriority.MEDIUM.label == "Medium"
        assert NotificationPriority.LOW.label == "Low"

    def test_inherits_str(self) -> None:
        """NotificationPriority is also a str (useful for SQLite storage)."""
        assert isinstance(NotificationPriority.HIGH, str)

    def test_equality_with_string(self) -> None:
        assert NotificationPriority.MEDIUM == "medium"


# ---------------------------------------------------------------------------
# NotificationPriority.from_str
# ---------------------------------------------------------------------------


class TestFromStr:
    def test_parse_critical(self) -> None:
        assert NotificationPriority.from_str("critical") == NotificationPriority.CRITICAL

    def test_parse_high(self) -> None:
        assert NotificationPriority.from_str("high") == NotificationPriority.HIGH

    def test_parse_medium(self) -> None:
        assert NotificationPriority.from_str("medium") == NotificationPriority.MEDIUM

    def test_parse_low(self) -> None:
        assert NotificationPriority.from_str("low") == NotificationPriority.LOW

    def test_legacy_normal_maps_to_medium(self) -> None:
        assert NotificationPriority.from_str("normal") == NotificationPriority.MEDIUM

    def test_unknown_string_maps_to_medium(self) -> None:
        assert NotificationPriority.from_str("banana") == NotificationPriority.MEDIUM

    def test_none_maps_to_medium(self) -> None:
        assert NotificationPriority.from_str(None) == NotificationPriority.MEDIUM

    def test_case_insensitive(self) -> None:
        assert NotificationPriority.from_str("CRITICAL") == NotificationPriority.CRITICAL
        assert NotificationPriority.from_str("High") == NotificationPriority.HIGH

    def test_whitespace_stripped(self) -> None:
        assert NotificationPriority.from_str("  low  ") == NotificationPriority.LOW

    def test_empty_string_maps_to_medium(self) -> None:
        assert NotificationPriority.from_str("") == NotificationPriority.MEDIUM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> DashboardStore:
    return DashboardStore(db_path=tmp_path / "us088_test.db")


# ---------------------------------------------------------------------------
# DashboardStore.write() accepts priority parameter
# ---------------------------------------------------------------------------


class TestWriteAcceptsPriority:
    def test_write_with_critical_enum(self, store: DashboardStore) -> None:
        nid = store.write(
            title="Critical alert",
            body="Fix it now",
            priority=NotificationPriority.CRITICAL,
        )
        results = store.query_recent(limit=10)
        notif = next(n for n in results if n.id == nid)
        assert notif.priority == "critical"

    def test_write_with_high_enum(self, store: DashboardStore) -> None:
        nid = store.write(title="High", body="b", priority=NotificationPriority.HIGH)
        results = store.query_recent(limit=10)
        notif = next(n for n in results if n.id == nid)
        assert notif.priority == "high"

    def test_write_with_medium_enum(self, store: DashboardStore) -> None:
        nid = store.write(title="Med", body="b", priority=NotificationPriority.MEDIUM)
        results = store.query_recent(limit=10)
        notif = next(n for n in results if n.id == nid)
        assert notif.priority == "medium"

    def test_write_with_low_enum(self, store: DashboardStore) -> None:
        nid = store.write(title="Low", body="b", priority=NotificationPriority.LOW)
        results = store.query_recent(limit=10)
        notif = next(n for n in results if n.id == nid)
        assert notif.priority == "low"

    def test_write_with_string_critical(self, store: DashboardStore) -> None:
        nid = store.write(title="Crit str", body="b", priority="critical")
        results = store.query_recent(limit=10)
        notif = next(n for n in results if n.id == nid)
        assert notif.priority == "critical"

    def test_write_with_string_high(self, store: DashboardStore) -> None:
        nid = store.write(title="High str", body="b", priority="high")
        results = store.query_recent(limit=10)
        notif = next(n for n in results if n.id == nid)
        assert notif.priority == "high"

    def test_default_priority_is_medium(self, store: DashboardStore) -> None:
        nid = store.write(title="Default", body="b")
        results = store.query_recent(limit=10)
        notif = next(n for n in results if n.id == nid)
        assert notif.priority == "medium"

    def test_unknown_string_priority_stored_as_medium(self, store: DashboardStore) -> None:
        nid = store.write(title="Unknown prio", body="b", priority="banana")
        results = store.query_recent(limit=10)
        notif = next(n for n in results if n.id == nid)
        assert notif.priority == "medium"


# ---------------------------------------------------------------------------
# Priority stored alongside notification record in the database
# ---------------------------------------------------------------------------


class TestPriorityStoredInDatabase:
    def test_priority_persisted_to_sqlite(self, store: DashboardStore) -> None:
        """Verify priority is actually stored in the DB column."""
        nid = store.write(title="DB test", body="b", priority=NotificationPriority.HIGH)
        # Read directly from SQLite to confirm column value
        with sqlite3.connect(str(store._db_path)) as conn:
            row = conn.execute(
                "SELECT priority FROM notifications WHERE id = ?", (nid,)
            ).fetchone()
        assert row is not None
        assert row[0] == "high"

    def test_priority_column_present_for_all_records(self, store: DashboardStore) -> None:
        for p in NotificationPriority:
            store.write(title=f"Title {p.value}", body="b", priority=p)
        with sqlite3.connect(str(store._db_path)) as conn:
            rows = conn.execute(
                "SELECT priority FROM notifications ORDER BY timestamp"
            ).fetchall()
        stored_priorities = {row[0] for row in rows}
        assert stored_priorities == {"critical", "high", "medium", "low"}

    def test_query_recent_filter_by_priority(self, store: DashboardStore) -> None:
        store.write(title="C", body="b", priority=NotificationPriority.CRITICAL)
        store.write(title="H", body="b", priority=NotificationPriority.HIGH)
        store.write(title="M", body="b", priority=NotificationPriority.MEDIUM)
        store.write(title="L", body="b", priority=NotificationPriority.LOW)
        highs = store.query_recent(priority="high")
        assert all(n.priority == "high" for n in highs)
        assert len(highs) == 1


# ---------------------------------------------------------------------------
# Legacy / missing priority defaults to `medium` on read
# ---------------------------------------------------------------------------


class TestLegacyPriorityDefaultsToMedium:
    def test_legacy_normal_value_reads_as_medium(self, store: DashboardStore) -> None:
        """Directly insert a row with priority='normal' and verify read normalisation."""
        nid = "legacy_001"
        import json
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(str(store._db_path)) as conn:
            conn.execute(
                "INSERT INTO notifications "
                "(id, priority, title, body, channel, timestamp, read, user_id, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?)",
                (nid, "normal", "Legacy", "Old body", "dashboard", ts, json.dumps({})),
            )
            conn.commit()

        results = store.query_recent(limit=50)
        legacy = next((n for n in results if n.id == nid), None)
        assert legacy is not None
        assert legacy.priority == "medium"

    def test_null_priority_reads_as_medium(self, store: DashboardStore) -> None:
        """A NULL priority value in the DB must read back as medium."""
        nid = "null_prio_001"
        import json
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).isoformat()
        # Insert with NULL priority (workaround: use a direct SQL INSERT
        # bypassing the NOT NULL constraint by setting default after schema patch)
        # Since the schema has NOT NULL, insert empty string instead
        with sqlite3.connect(str(store._db_path)) as conn:
            conn.execute(
                "INSERT INTO notifications "
                "(id, priority, title, body, channel, timestamp, read, user_id, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?)",
                (nid, "", "EmptyPrio", "body", "dashboard", ts, json.dumps({})),
            )
            conn.commit()

        results = store.query_recent(limit=50)
        notif = next((n for n in results if n.id == nid), None)
        assert notif is not None
        assert notif.priority == "medium"

    def test_unknown_priority_reads_as_medium(self, store: DashboardStore) -> None:
        """Any unrecognised priority string stored in DB reads back as medium."""
        nid = "unknown_prio_001"
        import json
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(str(store._db_path)) as conn:
            conn.execute(
                "INSERT INTO notifications "
                "(id, priority, title, body, channel, timestamp, read, user_id, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?)",
                (nid, "urgent_legacy", "OldUrgent", "body", "dashboard", ts, json.dumps({})),
            )
            conn.commit()

        results = store.query_recent(limit=50)
        notif = next((n for n in results if n.id == nid), None)
        assert notif is not None
        assert notif.priority == "medium"


# ---------------------------------------------------------------------------
# Enum importable from dashboard_store (re-export)
# ---------------------------------------------------------------------------


class TestReExport:
    def test_notification_priority_importable_from_dashboard_store(self) -> None:
        from rex.dashboard_store import NotificationPriority as NP  # noqa: F401

        assert NP.CRITICAL.value == "critical"

    def test_notification_priority_importable_from_notification_priority(self) -> None:
        from rex.notification_priority import NotificationPriority as NP  # noqa: F401

        assert NP.MEDIUM.value == "medium"
