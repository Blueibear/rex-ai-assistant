"""Tests for the memory module (WorkingMemory and LongTermMemory)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rex.memory import (
    LongTermMemory,
    MemoryEntry,
    WorkingMemory,
    add_user_preference,
    get_long_term_memory,
    get_recent_context,
    get_user_preferences,
    get_working_memory,
    remember_context,
    set_long_term_memory,
    set_working_memory,
)

# =============================================================================
# WorkingMemory Tests
# =============================================================================


class TestWorkingMemory:
    """Tests for the WorkingMemory class."""

    def test_add_entry_and_retrieve(self, tmp_path):
        """Test adding entries and retrieving them."""
        storage = tmp_path / "working_memory.json"
        wm = WorkingMemory(storage_path=storage)

        wm.add_entry("First entry")
        wm.add_entry("Second entry")
        wm.add_entry("Third entry")

        recent = wm.get_recent(2)
        assert len(recent) == 2
        assert recent[0] == "Second entry"
        assert recent[1] == "Third entry"

    def test_get_recent_returns_all_when_fewer(self, tmp_path):
        """Test get_recent when fewer entries exist than requested."""
        storage = tmp_path / "working_memory.json"
        wm = WorkingMemory(storage_path=storage)

        wm.add_entry("Only entry")

        recent = wm.get_recent(10)
        assert len(recent) == 1
        assert recent[0] == "Only entry"

    def test_get_recent_with_timestamps(self, tmp_path):
        """Test retrieving entries with timestamps."""
        storage = tmp_path / "working_memory.json"
        wm = WorkingMemory(storage_path=storage)

        wm.add_entry("Test entry")

        entries = wm.get_recent_with_timestamps(1)
        assert len(entries) == 1
        assert "content" in entries[0]
        assert "timestamp" in entries[0]
        assert entries[0]["content"] == "Test entry"

    def test_clear_removes_all_entries(self, tmp_path):
        """Test clearing all entries."""
        storage = tmp_path / "working_memory.json"
        wm = WorkingMemory(storage_path=storage)

        wm.add_entry("Entry 1")
        wm.add_entry("Entry 2")
        assert len(wm) == 2

        wm.clear()
        assert len(wm) == 0
        assert wm.get_recent(10) == []

    def test_max_entries_limit(self, tmp_path):
        """Test that entries beyond max_entries are removed."""
        storage = tmp_path / "working_memory.json"
        wm = WorkingMemory(storage_path=storage, max_entries=3)

        for i in range(5):
            wm.add_entry(f"Entry {i}")

        assert len(wm) == 3
        recent = wm.get_recent(5)
        assert recent == ["Entry 2", "Entry 3", "Entry 4"]

    def test_persistence(self, tmp_path):
        """Test that entries persist across instances."""
        storage = tmp_path / "working_memory.json"

        # Create and populate
        wm1 = WorkingMemory(storage_path=storage)
        wm1.add_entry("Persistent entry")

        # Create new instance
        wm2 = WorkingMemory(storage_path=storage)
        recent = wm2.get_recent(1)
        assert recent[0] == "Persistent entry"

    def test_len(self, tmp_path):
        """Test the __len__ method."""
        storage = tmp_path / "working_memory.json"
        wm = WorkingMemory(storage_path=storage)

        assert len(wm) == 0
        wm.add_entry("Entry")
        assert len(wm) == 1


# =============================================================================
# LongTermMemory Tests
# =============================================================================


class TestMemoryEntry:
    """Tests for the MemoryEntry model."""

    def test_is_expired_when_not_set(self):
        """Test that entry without expires_at is not expired."""
        entry = MemoryEntry(category="test", content={"key": "value"})
        assert not entry.is_expired()

    def test_is_expired_when_future(self):
        """Test that entry with future expiry is not expired."""
        future = datetime.now(timezone.utc) + timedelta(days=7)
        entry = MemoryEntry(
            category="test",
            content={"key": "value"},
            expires_at=future,
        )
        assert not entry.is_expired()

    def test_is_expired_when_past(self):
        """Test that entry with past expiry is expired."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        entry = MemoryEntry(
            category="test",
            content={"key": "value"},
            expires_at=past,
        )
        assert entry.is_expired()

    def test_to_safe_dict_hides_sensitive(self):
        """Test that sensitive content is hidden in safe dict."""
        entry = MemoryEntry(
            category="secrets",  # pragma: allowlist secret
            content={"api_key": "secret123"},  # pragma: allowlist secret
            sensitive=True,
        )
        safe = entry.to_safe_dict()
        assert safe["content"] == {"[SENSITIVE]": "Content hidden"}

    def test_to_safe_dict_shows_non_sensitive(self):
        """Test that non-sensitive content is shown in safe dict."""
        entry = MemoryEntry(
            category="preferences",
            content={"theme": "dark"},
            sensitive=False,
        )
        safe = entry.to_safe_dict()
        assert safe["content"] == {"theme": "dark"}


class TestLongTermMemory:
    """Tests for the LongTermMemory class."""

    def test_add_and_get_entry(self, tmp_path):
        """Test adding and retrieving an entry."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        entry = ltm.add_entry(
            category="facts",
            content={"name": "Rex", "version": "1.0"},
        )

        retrieved = ltm.get_entry(entry.entry_id)
        assert retrieved is not None
        assert retrieved.category == "facts"
        assert retrieved.content["name"] == "Rex"

    def test_add_entry_with_expiration(self, tmp_path):
        """Test adding entry with TTL."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        entry = ltm.add_entry(
            category="temp",
            content={"temp": True},
            expires_in=timedelta(hours=24),
        )

        assert entry.expires_at is not None
        assert not entry.is_expired()

    def test_add_entry_with_sensitive_flag(self, tmp_path):
        """Test adding sensitive entry."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        entry = ltm.add_entry(
            category="secrets",  # pragma: allowlist secret
            content={"password": "hunter2"},  # pragma: allowlist secret
            sensitive=True,
        )

        assert entry.sensitive is True

    def test_search_by_category(self, tmp_path):
        """Test searching by category."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        ltm.add_entry(category="preferences", content={"theme": "dark"})
        ltm.add_entry(category="preferences", content={"language": "en"})
        ltm.add_entry(category="facts", content={"fact": "test"})

        results = ltm.search(category="preferences")
        assert len(results) == 2
        for r in results:
            assert r.category == "preferences"

    def test_search_by_keyword(self, tmp_path):
        """Test searching by keyword."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        ltm.add_entry(category="facts", content={"topic": "weather", "info": "sunny"})
        ltm.add_entry(category="facts", content={"topic": "sports", "info": "football"})

        results = ltm.search(keyword="weather")
        assert len(results) == 1
        assert results[0].content["topic"] == "weather"

    def test_search_keyword_in_nested_content(self, tmp_path):
        """Test searching for keyword in nested content."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        ltm.add_entry(
            category="complex",
            content={"outer": {"inner": {"deep": "findme"}}},
        )

        results = ltm.search(keyword="findme")
        assert len(results) == 1

    def test_search_excludes_expired(self, tmp_path):
        """Test that search excludes expired entries by default."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        # Add an already-expired entry
        past = datetime.now(timezone.utc) - timedelta(days=1)
        entry = MemoryEntry(
            category="expired",
            content={"expired": True},
            expires_at=past,
        )
        ltm._entries[entry.entry_id] = entry

        results = ltm.search(category="expired")
        assert len(results) == 0

    def test_search_includes_expired_when_requested(self, tmp_path):
        """Test that search can include expired entries."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        past = datetime.now(timezone.utc) - timedelta(days=1)
        entry = MemoryEntry(
            category="expired",
            content={"expired": True},
            expires_at=past,
        )
        ltm._entries[entry.entry_id] = entry

        results = ltm.search(category="expired", include_expired=True)
        assert len(results) == 1

    def test_search_excludes_sensitive_when_requested(self, tmp_path):
        """Test that search can exclude sensitive entries."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        ltm.add_entry(
            category="secrets",
            content={"secret": True},
            sensitive=True,
        )
        ltm.add_entry(
            category="secrets",
            content={"public": True},
            sensitive=False,
        )

        results = ltm.search(category="secrets", include_sensitive=False)
        assert len(results) == 1
        assert results[0].content.get("public") is True

    def test_forget_removes_entry(self, tmp_path):
        """Test forgetting an entry."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        entry = ltm.add_entry(category="temp", content={"temp": True})
        assert ltm.get_entry(entry.entry_id) is not None

        result = ltm.forget(entry.entry_id)
        assert result is True
        assert ltm.get_entry(entry.entry_id) is None

    def test_forget_returns_false_for_missing(self, tmp_path):
        """Test forgetting a non-existent entry."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        result = ltm.forget("nonexistent")
        assert result is False

    def test_run_retention_policy(self, tmp_path):
        """Test retention policy deletes expired entries."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        # Add expired entries directly
        past = datetime.now(timezone.utc) - timedelta(days=1)
        for i in range(3):
            entry = MemoryEntry(
                entry_id=f"expired_{i}",
                category="expired",
                content={"index": i},
                expires_at=past,
            )
            ltm._entries[entry.entry_id] = entry

        # Add non-expired entry
        ltm.add_entry(category="active", content={"active": True})

        deleted = ltm.run_retention_policy()
        assert deleted == 3
        assert len(ltm) == 1

    def test_list_categories(self, tmp_path):
        """Test listing all categories."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        ltm.add_entry(category="preferences", content={})
        ltm.add_entry(category="facts", content={})
        ltm.add_entry(category="preferences", content={})

        categories = ltm.list_categories()
        assert sorted(categories) == ["facts", "preferences"]

    def test_count_by_category(self, tmp_path):
        """Test counting entries by category."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        ltm.add_entry(category="facts", content={})
        ltm.add_entry(category="preferences", content={})
        ltm.add_entry(category="preferences", content={})

        counts = ltm.count_by_category()
        assert counts["facts"] == 1
        assert counts["preferences"] == 2

    def test_persistence(self, tmp_path):
        """Test that entries persist across instances."""
        storage = tmp_path / "ltm.json"

        ltm1 = LongTermMemory(storage_path=storage)
        entry = ltm1.add_entry(category="persist", content={"persisted": True})

        ltm2 = LongTermMemory(storage_path=storage)
        retrieved = ltm2.get_entry(entry.entry_id)
        assert retrieved is not None
        assert retrieved.content["persisted"] is True

    def test_len(self, tmp_path):
        """Test the __len__ method excludes expired entries."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        ltm.add_entry(category="active", content={})

        past = datetime.now(timezone.utc) - timedelta(days=1)
        entry = MemoryEntry(
            category="expired",
            content={},
            expires_at=past,
        )
        ltm._entries[entry.entry_id] = entry

        # Only active entry should count
        assert len(ltm) == 1


# =============================================================================
# Singleton and Convenience Function Tests
# =============================================================================


class TestSingletonPattern:
    """Tests for the global singleton instances."""

    def test_get_and_set_working_memory(self, tmp_path):
        """Test get/set working memory singleton."""
        storage = tmp_path / "wm.json"
        wm = WorkingMemory(storage_path=storage)
        set_working_memory(wm)

        retrieved = get_working_memory()
        assert retrieved is wm

    def test_get_and_set_long_term_memory(self, tmp_path):
        """Test get/set long-term memory singleton."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)
        set_long_term_memory(ltm)

        retrieved = get_long_term_memory()
        assert retrieved is ltm


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_add_user_preference(self, tmp_path):
        """Test adding a user preference."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)
        set_long_term_memory(ltm)

        entry = add_user_preference("theme", "dark")
        assert entry.category == "user_preferences"
        assert entry.content == {"theme": "dark"}

    def test_get_user_preferences(self, tmp_path):
        """Test getting user preferences."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)
        set_long_term_memory(ltm)

        add_user_preference("theme", "dark")
        add_user_preference("language", "en")

        prefs = get_user_preferences()
        assert len(prefs) == 2

        prefs = get_user_preferences("theme")
        assert len(prefs) == 1

    def test_remember_context(self, tmp_path):
        """Test remember_context adds to working memory."""
        storage = tmp_path / "wm.json"
        wm = WorkingMemory(storage_path=storage)
        set_working_memory(wm)

        remember_context("User asked about weather")
        recent = get_recent_context(1)
        assert recent[0] == "User asked about weather"

    def test_get_recent_context(self, tmp_path):
        """Test get_recent_context retrieves from working memory."""
        storage = tmp_path / "wm.json"
        wm = WorkingMemory(storage_path=storage)
        set_working_memory(wm)

        wm.add_entry("Context 1")
        wm.add_entry("Context 2")
        wm.add_entry("Context 3")

        recent = get_recent_context(2)
        assert len(recent) == 2
        assert recent == ["Context 2", "Context 3"]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_working_memory_handles_corrupt_file(self, tmp_path):
        """Test that corrupt storage file is handled gracefully."""
        storage = tmp_path / "corrupt.json"
        storage.write_text("not valid json {{{")

        wm = WorkingMemory(storage_path=storage)
        assert len(wm) == 0  # Should start fresh

    def test_long_term_memory_handles_corrupt_file(self, tmp_path):
        """Test that corrupt LTM storage is handled gracefully."""
        storage = tmp_path / "corrupt.json"
        storage.write_text("not valid json {{{")

        ltm = LongTermMemory(storage_path=storage)
        assert len(ltm) == 0  # Should start fresh

    def test_search_with_special_characters(self, tmp_path):
        """Test searching with special characters in keyword."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        ltm.add_entry(
            category="test",
            content={"query": "hello@world.com"},
        )

        results = ltm.search(keyword="@world")
        assert len(results) == 1

    def test_empty_content(self, tmp_path):
        """Test adding entry with empty content."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        entry = ltm.add_entry(category="empty", content={})
        assert entry.content == {}

    def test_unicode_content(self, tmp_path):
        """Test handling unicode content."""
        storage = tmp_path / "ltm.json"
        ltm = LongTermMemory(storage_path=storage)

        entry = ltm.add_entry(
            category="unicode",
            content={"text": "Hello 世界 🌍"},
        )

        # Reload and verify
        ltm2 = LongTermMemory(storage_path=storage)
        retrieved = ltm2.get_entry(entry.entry_id)
        assert retrieved.content["text"] == "Hello 世界 🌍"
