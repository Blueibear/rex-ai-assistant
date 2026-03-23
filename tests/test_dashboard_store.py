"""Tests for dashboard notification store (SQLite backend)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rex.dashboard_store import (
    DashboardNotification,
    DashboardStore,
    DashboardStoreConfig,
    get_dashboard_store,
    load_dashboard_store_config,
    set_dashboard_store,
)


@pytest.fixture
def store(tmp_path: Path) -> DashboardStore:
    """A DashboardStore with a temp database."""
    return DashboardStore(db_path=tmp_path / "test_notifications.db")


# --- DashboardStoreConfig tests ---


def test_store_config_defaults():
    """Default config uses sqlite with 30-day retention."""
    config = DashboardStoreConfig()
    assert config.type == "sqlite"
    assert config.path is None
    assert config.retention_days == 30


def test_load_store_config_missing():
    """Returns defaults when config section is absent."""
    config = load_dashboard_store_config({})
    assert config.type == "sqlite"
    assert config.retention_days == 30


def test_load_store_config_valid():
    """Parses a full config section."""
    raw = {
        "notifications": {
            "dashboard": {
                "store": {
                    "type": "sqlite",
                    "path": "/custom/path/notifications.db",
                    "retention_days": 7,
                }
            }
        }
    }
    config = load_dashboard_store_config(raw)
    assert config.path == "/custom/path/notifications.db"
    assert config.retention_days == 7


# --- Write tests ---


def test_write_notification(store: DashboardStore):
    """Write a notification and verify it's stored."""
    nid = store.write(
        title="Test Alert",
        body="Something happened",
        priority="high",
        user_id="alice",
    )
    assert nid.startswith("dash_")

    notifications = store.query_recent()
    assert len(notifications) == 1
    assert notifications[0].title == "Test Alert"
    assert notifications[0].priority == "high"
    assert notifications[0].user_id == "alice"
    assert notifications[0].read is False


def test_write_notification_custom_id(store: DashboardStore):
    """Write with a custom notification ID."""
    nid = store.write(
        notification_id="notif_custom_123",
        title="Custom",
        body="Body",
    )
    assert nid == "notif_custom_123"


def test_write_notification_with_metadata(store: DashboardStore):
    """Write with metadata dict."""
    store.write(
        title="Meta Test",
        body="Body",
        metadata={"source": "email", "importance": 0.9},
    )
    notifications = store.query_recent()
    assert len(notifications) == 1
    d = notifications[0].to_dict()
    assert d["metadata"]["source"] == "email"
    assert d["metadata"]["importance"] == 0.9


# --- Query tests ---


def test_query_recent_ordering(store: DashboardStore):
    """Notifications are returned newest first."""
    for i in range(5):
        store.write(title=f"Notif {i}", body=f"Body {i}")
        time.sleep(0.01)  # Ensure distinct timestamps

    results = store.query_recent(limit=5)
    assert len(results) == 5
    # Newest first
    assert results[0].title == "Notif 4"
    assert results[4].title == "Notif 0"


def test_query_recent_limit(store: DashboardStore):
    """Limit parameter is respected."""
    for i in range(10):
        store.write(title=f"N{i}", body="B")

    results = store.query_recent(limit=3)
    assert len(results) == 3


def test_query_recent_unread_only(store: DashboardStore):
    """Filter to unread only."""
    nid1 = store.write(title="Read Me", body="Body")
    store.write(title="Unread", body="Body")
    store.mark_as_read(nid1)

    results = store.query_recent(unread_only=True)
    assert len(results) == 1
    assert results[0].title == "Unread"


def test_query_recent_by_priority(store: DashboardStore):
    """Filter by priority."""
    store.write(title="High", body="B", priority="high")
    store.write(title="Medium", body="B", priority="medium")
    store.write(title="Low", body="B", priority="low")

    high = store.query_recent(priority="high")
    assert len(high) == 1
    assert high[0].title == "High"


def test_query_recent_by_user(store: DashboardStore):
    """Filter by user_id."""
    store.write(title="Alice Notif", body="B", user_id="alice")
    store.write(title="Bob Notif", body="B", user_id="bob")
    store.write(title="No User", body="B")

    alice_results = store.query_recent(user_id="alice")
    assert len(alice_results) == 1
    assert alice_results[0].title == "Alice Notif"


# --- Count tests ---


def test_count_unread(store: DashboardStore):
    """Count unread notifications."""
    store.write(title="N1", body="B")
    nid2 = store.write(title="N2", body="B")
    store.write(title="N3", body="B")
    store.mark_as_read(nid2)

    assert store.count_unread() == 2


def test_count_unread_by_user(store: DashboardStore):
    """Count unread scoped by user."""
    store.write(title="A1", body="B", user_id="alice")
    store.write(title="A2", body="B", user_id="alice")
    store.write(title="B1", body="B", user_id="bob")

    assert store.count_unread(user_id="alice") == 2
    assert store.count_unread(user_id="bob") == 1


# --- Update tests ---


def test_mark_as_read(store: DashboardStore):
    """Mark a notification as read."""
    nid = store.write(title="Test", body="B")
    assert store.mark_as_read(nid) is True
    # Second call returns False (already read)
    assert store.mark_as_read(nid) is False

    results = store.query_recent()
    assert results[0].read is True


def test_mark_as_read_nonexistent(store: DashboardStore):
    """Mark nonexistent notification returns False."""
    assert store.mark_as_read("nonexistent_id") is False


def test_mark_all_read(store: DashboardStore):
    """Mark all notifications as read."""
    for i in range(3):
        store.write(title=f"N{i}", body="B")

    count = store.mark_all_read()
    assert count == 3
    assert store.count_unread() == 0


def test_mark_all_read_by_user(store: DashboardStore):
    """Mark all read scoped by user."""
    store.write(title="A1", body="B", user_id="alice")
    store.write(title="A2", body="B", user_id="alice")
    store.write(title="B1", body="B", user_id="bob")

    count = store.mark_all_read(user_id="alice")
    assert count == 2
    assert store.count_unread(user_id="alice") == 0
    assert store.count_unread(user_id="bob") == 1


# --- Maintenance tests ---


def test_cleanup_old(tmp_path: Path):
    """Cleanup removes notifications older than retention."""
    store = DashboardStore(db_path=tmp_path / "test.db", retention_days=1)

    # Write a notification with old timestamp
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "test.db"))
    old_ts = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    conn.execute(
        """
        INSERT INTO notifications (id, priority, title, body, channel, timestamp, read, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, 0, '{}')
        """,
        ("old_notif", "normal", "Old", "Body", "dashboard", old_ts),
    )
    conn.commit()
    conn.close()

    # Write a recent notification
    store.write(title="Recent", body="Body")

    removed = store.cleanup_old()
    assert removed == 1
    assert len(store.query_recent()) == 1


# --- DashboardNotification tests ---


def test_dashboard_notification_to_dict():
    """DashboardNotification.to_dict() returns expected structure."""
    notif = DashboardNotification(
        id="test_1",
        priority="urgent",
        title="Alert",
        body="Important",
        timestamp="2025-01-01T00:00:00+00:00",
        user_id="alice",
        metadata_json='{"key": "value"}',
    )
    d = notif.to_dict()
    assert d["id"] == "test_1"
    assert d["priority"] == "urgent"
    assert d["metadata"] == {"key": "value"}
    assert d["read"] is False


# --- Global accessors ---


def test_global_dashboard_store():
    """Global getter/setter works."""
    original = None
    try:
        store1 = get_dashboard_store()
        store2 = get_dashboard_store()
        assert store1 is store2

        custom = DashboardStore(db_path=Path("/tmp/test_custom_dash.db"))
        set_dashboard_store(custom)
        assert get_dashboard_store() is custom
    finally:
        set_dashboard_store(original)
