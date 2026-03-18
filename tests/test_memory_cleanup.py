"""Tests for US-071: Memory cleanup — expired removal, scheduled cleanup, compaction."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from rex.memory import (
    LongTermMemory,
    MemoryEntry,
    schedule_memory_cleanup,
    set_long_term_memory,
)

# =============================================================================
# Expired Memory Removal
# =============================================================================


class TestExpiredMemoryRemoval:
    """Expired entries are removed when retention policy runs."""

    def test_expired_entries_removed_on_retention(self, tmp_path: Path) -> None:
        """run_retention_policy removes expired entries."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        past = datetime.now(timezone.utc) - timedelta(days=1)
        for i in range(3):
            entry = MemoryEntry(
                entry_id=f"expired_{i}",
                category="temp",
                content={"i": i},
                expires_at=past,
            )
            ltm._entries[entry.entry_id] = entry

        active = ltm.add_entry(category="active", content={"keep": True})

        removed = ltm.run_retention_policy()

        assert removed == 3
        assert len(ltm) == 1
        assert ltm.get_entry(active.entry_id) is not None
        for i in range(3):
            assert ltm.get_entry(f"expired_{i}") is None

    def test_non_expired_entries_retained(self, tmp_path: Path) -> None:
        """run_retention_policy does not remove non-expired entries."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        for i in range(5):
            ltm.add_entry(
                category="active",
                content={"i": i},
                expires_in=timedelta(days=7),
            )

        removed = ltm.run_retention_policy()
        assert removed == 0
        assert len(ltm) == 5

    def test_entries_without_expiry_never_removed(self, tmp_path: Path) -> None:
        """Entries with no expiry are never removed by retention policy."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        ltm.add_entry(category="permanent", content={"permanent": True})
        ltm.add_entry(category="permanent", content={"permanent": True})

        removed = ltm.run_retention_policy()
        assert removed == 0
        assert len(ltm) == 2


# =============================================================================
# Scheduled Cleanup
# =============================================================================


class TestScheduledCleanup:
    """Cleanup can be registered with the scheduler."""

    def test_schedule_memory_cleanup_registers_callback(self, tmp_path: Path) -> None:
        """schedule_memory_cleanup registers a callback on the scheduler."""
        mock_scheduler = MagicMock()

        schedule_memory_cleanup(mock_scheduler, interval_seconds=600)

        mock_scheduler.register_callback.assert_called_once_with(
            "memory_cleanup", mock_scheduler.register_callback.call_args[0][1]
        )

    def test_schedule_memory_cleanup_adds_job(self, tmp_path: Path) -> None:
        """schedule_memory_cleanup adds a job to the scheduler."""
        mock_scheduler = MagicMock()

        schedule_memory_cleanup(mock_scheduler, interval_seconds=600, job_id="test_cleanup")

        mock_scheduler.add_job.assert_called_once()
        _, kwargs = mock_scheduler.add_job.call_args
        assert kwargs["job_id"] == "test_cleanup"
        assert kwargs["schedule"] == "interval:600"
        assert kwargs["callback_name"] == "memory_cleanup"

    def test_schedule_memory_cleanup_default_interval(self, tmp_path: Path) -> None:
        """schedule_memory_cleanup defaults to 1-hour interval."""
        mock_scheduler = MagicMock()

        schedule_memory_cleanup(mock_scheduler)

        _, kwargs = mock_scheduler.add_job.call_args
        assert kwargs["schedule"] == "interval:3600"

    def test_cleanup_callback_calls_compact(self, tmp_path: Path) -> None:
        """The registered callback invokes compact() on the global LTM."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        # Add an expired entry
        past = datetime.now(timezone.utc) - timedelta(days=1)
        entry = MemoryEntry(
            category="old",
            content={"data": "stale"},
            expires_at=past,
        )
        ltm._entries[entry.entry_id] = entry
        set_long_term_memory(ltm)

        # Capture the callback that gets registered
        captured_callback = None

        def capture_callback(name: str, cb: object) -> None:
            nonlocal captured_callback
            captured_callback = cb

        mock_scheduler = MagicMock()
        mock_scheduler.register_callback.side_effect = capture_callback

        schedule_memory_cleanup(mock_scheduler, interval_seconds=60)

        assert captured_callback is not None
        # Invoke the callback (pass a mock job)
        captured_callback(MagicMock())

        # The expired entry should now be gone
        assert len(ltm) == 0


# =============================================================================
# Memory Store Compaction
# =============================================================================


class TestMemoryStoreCompaction:
    """compact() removes expired entries and rewrites storage."""

    def test_compact_removes_expired_entries(self, tmp_path: Path) -> None:
        """compact() removes expired entries from in-memory store."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        past = datetime.now(timezone.utc) - timedelta(hours=1)
        for i in range(4):
            e = MemoryEntry(
                entry_id=f"stale_{i}",
                category="stale",
                content={"i": i},
                expires_at=past,
            )
            ltm._entries[e.entry_id] = e

        active = ltm.add_entry(category="keep", content={"value": "active"})

        removed = ltm.compact()

        assert removed == 4
        assert len(ltm) == 1
        assert ltm.get_entry(active.entry_id) is not None

    def test_compact_rewrites_storage_file(self, tmp_path: Path) -> None:
        """compact() always rewrites the storage file."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        past = datetime.now(timezone.utc) - timedelta(hours=1)
        e = MemoryEntry(
            entry_id="stale_one",
            category="stale",
            content={"x": 1},
            expires_at=past,
        )
        ltm._entries[e.entry_id] = e
        ltm._save()  # persist with expired entry

        # compact should remove stale_one and rewrite
        ltm.compact()

        # Reload from disk and confirm stale entry gone
        ltm2 = LongTermMemory(storage_path=storage)
        assert ltm2.get_entry("stale_one") is None

    def test_compact_on_empty_store_does_not_error(self, tmp_path: Path) -> None:
        """compact() on an empty store returns 0 without error."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        removed = ltm.compact()
        assert removed == 0

    def test_compact_returns_count_of_removed(self, tmp_path: Path) -> None:
        """compact() returns the number of entries removed."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        for i in range(7):
            e = MemoryEntry(
                entry_id=f"exp_{i}",
                category="exp",
                content={},
                expires_at=past,
            )
            ltm._entries[e.entry_id] = e

        removed = ltm.compact()
        assert removed == 7
