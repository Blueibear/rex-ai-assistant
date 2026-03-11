"""Tests for US-068: Notification persistence.

Acceptance criteria:
- notifications stored in database
- notifications retrieved
- timestamps recorded
- Typecheck passes
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rex.dashboard_store import DashboardStore


@pytest.fixture
def store(tmp_path: Path) -> DashboardStore:
    return DashboardStore(db_path=tmp_path / "us068_test.db")


def test_notification_stored_in_database(store: DashboardStore) -> None:
    """Notifications are persisted to the SQLite database."""
    nid = store.write(title="Test Alert", body="Something happened", priority="normal")
    assert nid is not None
    results = store.query_recent(limit=10)
    assert any(n.id == nid for n in results)


def test_notifications_retrieved(store: DashboardStore) -> None:
    """Stored notifications can be retrieved from the database."""
    store.write(title="First", body="Body 1")
    store.write(title="Second", body="Body 2")
    results = store.query_recent(limit=10)
    titles = [n.title for n in results]
    assert "First" in titles
    assert "Second" in titles


def test_timestamps_recorded(store: DashboardStore) -> None:
    """Each stored notification has a non-empty ISO timestamp."""
    nid = store.write(title="Timed Alert", body="Check the time")
    results = store.query_recent(limit=10)
    record = next((n for n in results if n.id == nid), None)
    assert record is not None
    assert record.timestamp
    # Ensure the timestamp is a valid ISO 8601 string
    from datetime import datetime

    dt = datetime.fromisoformat(record.timestamp)
    assert dt.year >= 2024


def test_multiple_notifications_persist(store: DashboardStore) -> None:
    """Multiple notifications all survive a new store instance pointing at the same db."""
    db_path = store._db_path
    store.write(title="A", body="alpha")
    store.write(title="B", body="beta")

    store2 = DashboardStore(db_path=db_path)
    results = store2.query_recent(limit=10)
    titles = {n.title for n in results}
    assert "A" in titles
    assert "B" in titles


def test_notification_stored_with_priority(store: DashboardStore) -> None:
    """Priority field is persisted and returned correctly."""
    store.write(title="Urgent Note", body="Act now", priority="urgent")
    results = store.query_recent(priority="urgent", limit=10)
    assert any(n.title == "Urgent Note" for n in results)
