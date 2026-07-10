"""US-303: per-user isolation for working memory and long-term memory.

Covers:
- Cross-user read/search/get/forget/clear denial for both stores
- Per-user searches, categories, counts, statistics, and retention
- Sensitive entries remain isolated per owner
- Registry returns distinct per-user instances; testing setters are scoped
- Ownership survives registry reset (restart simulation)
- Missing / blank / malformed / traversal identity fails closed
- Legacy unscoped files migrate only for the explicit ``default`` profile,
  idempotently and crash-safely, preserving the original as a backup
- CLI ``rex memory`` operations resolve identity and stay owner-scoped
- Scheduled cleanup compacts each user's store independently and ignores
  invalid directory names
"""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

BAD_USER_IDS = ["", "  ", "..", ".", "../evil", "a/b", "a\\b", "..\\..\\evil", "a" * 65]


@pytest.fixture()
def mem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """rex.memory with an isolated data dir and empty per-user registries."""
    import rex.memory as memory_module

    monkeypatch.setattr(memory_module, "_DATA_DIR", tmp_path / "memdata")
    monkeypatch.setattr(memory_module, "_working_memories", {})
    monkeypatch.setattr(memory_module, "_long_term_memories", {})
    return memory_module


def _write_legacy_files(mem) -> None:
    """Create pre-isolation shared store files at the data-dir root."""
    mem._DATA_DIR.mkdir(parents=True, exist_ok=True)
    (mem._DATA_DIR / "working_memory.json").write_text(
        json.dumps(
            {"entries": [{"content": "legacy shared note", "timestamp": "2024-01-01T00:00:00Z"}]}
        ),
        encoding="utf-8",
    )
    (mem._DATA_DIR / "long_term_memory.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "entry_id": "mem_legacy000001",
                        "category": "facts",
                        "content": {"note": "legacy shared fact"},
                        "created_at": "2024-01-01T00:00:00Z",
                        "expires_at": None,
                        "sensitive": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


# =============================================================================
# Store-level cross-user isolation
# =============================================================================


class TestWorkingMemoryIsolation:
    def test_entries_invisible_to_other_user(self, mem) -> None:
        mem.get_working_memory(user_id="alice").add_entry("alice private context")

        assert mem.get_working_memory(user_id="bob").get_recent(10) == []
        assert mem.get_working_memory(user_id="alice").get_recent(10) == ["alice private context"]

    def test_clear_does_not_touch_other_user(self, mem) -> None:
        mem.get_working_memory(user_id="alice").add_entry("alice keeps this")
        mem.get_working_memory(user_id="bob").add_entry("bob entry")

        mem.get_working_memory(user_id="bob").clear()

        assert mem.get_working_memory(user_id="alice").get_recent(10) == ["alice keeps this"]
        assert mem.get_working_memory(user_id="bob").get_recent(10) == []

    def test_stats_are_per_user(self, mem) -> None:
        mem.get_working_memory(user_id="alice").add_entry("one")
        mem.get_working_memory(user_id="alice").add_entry("two")
        mem.get_working_memory(user_id="bob").add_entry("only")

        assert mem.get_working_memory(user_id="alice").stats()["entries"] == 2
        assert mem.get_working_memory(user_id="bob").stats()["entries"] == 1

    def test_storage_files_are_partitioned_per_user(self, mem) -> None:
        mem.get_working_memory(user_id="alice").add_entry("alice data")
        mem.get_working_memory(user_id="bob").add_entry("bob data")

        alice_file = mem._DATA_DIR / "alice" / "working_memory.json"
        bob_file = mem._DATA_DIR / "bob" / "working_memory.json"
        assert alice_file.exists()
        assert bob_file.exists()
        assert "bob data" not in alice_file.read_text(encoding="utf-8")
        assert "alice data" not in bob_file.read_text(encoding="utf-8")


class TestLongTermMemoryIsolation:
    def test_entries_invisible_to_other_user(self, mem) -> None:
        mem.get_long_term_memory(user_id="alice").add_entry(
            category="facts", content={"note": "alice fact"}
        )

        assert mem.get_long_term_memory(user_id="bob").search() == []
        assert len(mem.get_long_term_memory(user_id="alice").search()) == 1

    def test_get_by_id_denies_non_owner(self, mem) -> None:
        entry = mem.get_long_term_memory(user_id="alice").add_entry(
            category="facts", content={"note": "alice fact"}
        )

        assert mem.get_long_term_memory(user_id="bob").get_entry(entry.entry_id) is None
        assert mem.get_long_term_memory(user_id="alice").get_entry(entry.entry_id) is not None

    def test_forget_denies_non_owner(self, mem) -> None:
        entry = mem.get_long_term_memory(user_id="alice").add_entry(
            category="facts", content={"note": "alice fact"}
        )

        assert mem.get_long_term_memory(user_id="bob").forget(entry.entry_id) is False
        assert mem.get_long_term_memory(user_id="alice").get_entry(entry.entry_id) is not None

    def test_search_categories_counts_are_per_user(self, mem) -> None:
        alice = mem.get_long_term_memory(user_id="alice")
        bob = mem.get_long_term_memory(user_id="bob")
        alice.add_entry(category="medical", content={"note": "alice topic"})
        alice.add_entry(category="prefs", content={"theme": "dark"})
        bob.add_entry(category="shopping", content={"item": "milk"})

        assert bob.search(keyword="alice topic") == []
        assert bob.list_categories() == ["shopping"]
        assert alice.list_categories() == ["medical", "prefs"]
        assert bob.count_by_category() == {"shopping": 1}
        assert alice.stats()["entries"] == 2
        assert bob.stats()["entries"] == 1

    def test_sensitive_entries_remain_isolated(self, mem) -> None:
        mem.get_long_term_memory(user_id="alice").add_entry(
            category="medical",
            content={"diagnosis": "confidential"},
            sensitive=True,
        )

        bob = mem.get_long_term_memory(user_id="bob")
        assert bob.search(include_sensitive=True) == []
        assert bob.search(keyword="confidential", include_sensitive=True) == []

    def test_retention_is_per_user(self, mem) -> None:
        alice = mem.get_long_term_memory(user_id="alice")
        bob = mem.get_long_term_memory(user_id="bob")
        past = datetime.now(UTC) - timedelta(days=1)
        expired = mem.MemoryEntry(category="temp", content={"stale": True}, expires_at=past)
        alice._entries[expired.entry_id] = expired
        bob.add_entry(category="keep", content={"fresh": True})

        assert bob.run_retention_policy() == 0
        assert alice.run_retention_policy() == 1
        assert len(bob) == 1

    def test_same_entry_id_does_not_collide_across_users(self, mem) -> None:
        alice = mem.get_long_term_memory(user_id="alice")
        bob = mem.get_long_term_memory(user_id="bob")
        alice.add_entry(category="facts", content={"owner": "alice"}, entry_id="mem_shared0001")
        bob.add_entry(category="facts", content={"owner": "bob"}, entry_id="mem_shared0001")

        assert alice.get_entry("mem_shared0001").content == {"owner": "alice"}
        assert bob.get_entry("mem_shared0001").content == {"owner": "bob"}


# =============================================================================
# Registry / singleton behavior and persistence
# =============================================================================


class TestRegistry:
    def test_distinct_instances_per_user(self, mem) -> None:
        assert mem.get_working_memory(user_id="alice") is not mem.get_working_memory(user_id="bob")
        assert mem.get_long_term_memory(user_id="alice") is not mem.get_long_term_memory(
            user_id="bob"
        )

    def test_same_instance_for_same_user(self, mem) -> None:
        assert mem.get_working_memory(user_id="alice") is mem.get_working_memory(user_id="alice")
        assert mem.get_long_term_memory(user_id="alice") is mem.get_long_term_memory(
            user_id="alice"
        )

    def test_instance_never_backed_by_other_users_path(self, mem) -> None:
        alice_wm = mem.get_working_memory(user_id="alice")
        bob_wm = mem.get_working_memory(user_id="bob")
        assert "alice" in str(alice_wm.storage_path)
        assert "alice" not in str(bob_wm.storage_path)
        assert alice_wm.storage_path != bob_wm.storage_path

    def test_testing_setters_are_user_scoped(self, mem, tmp_path: Path) -> None:
        custom = mem.WorkingMemory(storage_path=tmp_path / "custom_wm.json")
        custom.add_entry("injected for alice")
        mem.set_working_memory(custom, user_id="alice")

        assert mem.get_working_memory(user_id="alice") is custom
        assert mem.get_working_memory(user_id="bob") is not custom
        assert mem.get_working_memory(user_id="bob").get_recent(10) == []

        custom_ltm = mem.LongTermMemory(storage_path=tmp_path / "custom_ltm.json")
        mem.set_long_term_memory(custom_ltm, user_id="alice")
        assert mem.get_long_term_memory(user_id="alice") is custom_ltm
        assert mem.get_long_term_memory(user_id="bob") is not custom_ltm

    def test_testing_setters_validate_user(self, mem, tmp_path: Path) -> None:
        wm = mem.WorkingMemory(storage_path=tmp_path / "wm.json")
        with pytest.raises(ValueError):
            mem.set_working_memory(wm, user_id="../evil")
        ltm = mem.LongTermMemory(storage_path=tmp_path / "ltm.json")
        with pytest.raises(ValueError):
            mem.set_long_term_memory(ltm, user_id="")

    def test_restart_preserves_ownership(self, mem, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = mem.get_long_term_memory(user_id="alice").add_entry(
            category="facts", content={"note": "persists"}
        )
        mem.get_working_memory(user_id="alice").add_entry("alice context persists")

        # Simulate a process restart: fresh registries, reload from disk.
        monkeypatch.setattr(mem, "_working_memories", {})
        monkeypatch.setattr(mem, "_long_term_memories", {})

        assert mem.get_long_term_memory(user_id="alice").get_entry(entry.entry_id) is not None
        assert mem.get_long_term_memory(user_id="bob").get_entry(entry.entry_id) is None
        assert mem.get_working_memory(user_id="alice").get_recent(10) == ["alice context persists"]
        assert mem.get_working_memory(user_id="bob").get_recent(10) == []


# =============================================================================
# Fail-closed identity validation
# =============================================================================


class TestFailClosedIdentity:
    def test_missing_identity_fails_closed(self, mem) -> None:
        with pytest.raises(TypeError):
            mem.get_working_memory()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            mem.get_long_term_memory()  # type: ignore[call-arg]
        with pytest.raises(ValueError):
            mem.get_working_memory(user_id=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            mem.get_long_term_memory(user_id=None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_user", BAD_USER_IDS)
    def test_invalid_identity_rejected_everywhere(self, mem, bad_user: str) -> None:
        with pytest.raises(ValueError):
            mem.get_working_memory(user_id=bad_user)
        with pytest.raises(ValueError):
            mem.get_long_term_memory(user_id=bad_user)
        with pytest.raises(ValueError):
            mem.add_user_preference("theme", "dark", user_id=bad_user)
        with pytest.raises(ValueError):
            mem.get_user_preferences(user_id=bad_user)
        with pytest.raises(ValueError):
            mem.add_fact("topic", {"x": 1}, user_id=bad_user)
        with pytest.raises(ValueError):
            mem.remember_context("summary", user_id=bad_user)
        with pytest.raises(ValueError):
            mem.get_recent_context(user_id=bad_user)

    def test_convenience_functions_require_user_id(self, mem) -> None:
        with pytest.raises(TypeError):
            mem.remember_context("no identity")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            mem.get_recent_context()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            mem.add_user_preference("theme", "dark")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            mem.get_user_preferences()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            mem.add_fact("topic", {"x": 1})  # type: ignore[call-arg]

    @pytest.mark.parametrize("bad_user", ["..", "../evil", "a/b", "a\\b", "..\\..\\evil"])
    def test_traversal_ids_cannot_influence_storage_paths(self, mem, bad_user: str) -> None:
        data_dir = mem._DATA_DIR
        with pytest.raises(ValueError):
            mem.get_working_memory(user_id=bad_user)
        with pytest.raises(ValueError):
            mem.get_long_term_memory(user_id=bad_user)
        # Nothing may have been created anywhere (inside or outside the data dir).
        assert not data_dir.exists() or list(data_dir.iterdir()) == []
        assert not (data_dir.parent / "evil").exists()

    def test_direct_construction_without_identity_or_path_fails(self, mem) -> None:
        with pytest.raises(ValueError):
            mem.WorkingMemory()
        with pytest.raises(ValueError):
            mem.LongTermMemory()

    def test_case_variant_ids_cannot_alias_another_users_store(self, mem) -> None:
        """Windows/macOS filesystems are case-insensitive: 'James' must never
        open (or overwrite) 'james''s store, on any platform."""
        mem.get_working_memory(user_id="james").add_entry("james private data")
        mem.get_long_term_memory(user_id="james").add_entry(
            category="facts", content={"note": "james fact"}
        )

        for variant in ("James", "JAMES", "jAmEs"):
            with pytest.raises(ValueError):
                mem.get_working_memory(user_id=variant)
            with pytest.raises(ValueError):
                mem.get_long_term_memory(user_id=variant)
            with pytest.raises(ValueError):
                mem.remember_context("hijack attempt", user_id=variant)

        # The original owner is unaffected.
        assert mem.get_working_memory(user_id="james").get_recent(10) == ["james private data"]

    def test_case_variant_rejected_before_any_file_exists(self, mem) -> None:
        """Registry-only collision (no directory saved yet) is also rejected."""
        mem.get_working_memory(user_id="james")  # cached, nothing written yet

        with pytest.raises(ValueError):
            mem.get_working_memory(user_id="James")

    def test_case_variant_rejected_across_restart(self, mem, monkeypatch) -> None:
        mem.get_working_memory(user_id="james").add_entry("persisted")

        monkeypatch.setattr(mem, "_working_memories", {})
        monkeypatch.setattr(mem, "_long_term_memories", {})

        with pytest.raises(ValueError):
            mem.get_working_memory(user_id="James")
        assert mem.get_working_memory(user_id="james").get_recent(10) == ["persisted"]

    def test_case_variant_rejected_in_testing_setters(self, mem, tmp_path: Path) -> None:
        mem.get_working_memory(user_id="james")
        injected = mem.WorkingMemory(storage_path=tmp_path / "wm.json")
        with pytest.raises(ValueError):
            mem.set_working_memory(injected, user_id="James")
        injected_ltm = mem.LongTermMemory(storage_path=tmp_path / "ltm.json")
        mem.get_long_term_memory(user_id="james")
        with pytest.raises(ValueError):
            mem.set_long_term_memory(injected_ltm, user_id="James")


# =============================================================================
# Convenience functions are owner-scoped
# =============================================================================


class TestConvenienceFunctions:
    def test_preferences_are_per_user(self, mem) -> None:
        mem.add_user_preference("theme", "dark", user_id="alice")

        assert mem.get_user_preferences(user_id="bob") == []
        prefs = mem.get_user_preferences(user_id="alice")
        assert len(prefs) == 1
        assert prefs[0].content == {"theme": "dark"}

    def test_facts_are_per_user(self, mem) -> None:
        mem.add_fact("weather", {"info": "sunny"}, user_id="alice")

        assert mem.get_long_term_memory(user_id="bob").search(category="facts") == []
        assert len(mem.get_long_term_memory(user_id="alice").search(category="facts")) == 1

    def test_context_is_per_user(self, mem) -> None:
        mem.remember_context("alice asked about results", user_id="alice")

        assert mem.get_recent_context(5, user_id="bob") == []
        assert mem.get_recent_context(5, user_id="alice") == ["alice asked about results"]


# =============================================================================
# Legacy unscoped-file migration (default profile only)
# =============================================================================


class TestLegacyMigration:
    def test_legacy_files_migrate_only_for_explicit_default(self, mem) -> None:
        _write_legacy_files(mem)

        wm = mem.get_working_memory(user_id="default")
        ltm = mem.get_long_term_memory(user_id="default")

        assert wm.get_recent(10) == ["legacy shared note"]
        assert ltm.get_entry("mem_legacy000001") is not None
        assert (mem._DATA_DIR / "default" / "working_memory.json").exists()
        assert (mem._DATA_DIR / "default" / "long_term_memory.json").exists()

    def test_named_user_never_sees_or_consumes_legacy_data(self, mem) -> None:
        _write_legacy_files(mem)
        legacy_wm = (mem._DATA_DIR / "working_memory.json").read_text(encoding="utf-8")
        legacy_ltm = (mem._DATA_DIR / "long_term_memory.json").read_text(encoding="utf-8")

        assert mem.get_working_memory(user_id="james").get_recent(10) == []
        assert mem.get_long_term_memory(user_id="james").get_entry("mem_legacy000001") is None
        assert mem.get_long_term_memory(user_id="james").search() == []

        # Originals untouched; no default-profile store was created.
        assert (mem._DATA_DIR / "working_memory.json").read_text(encoding="utf-8") == legacy_wm
        assert (mem._DATA_DIR / "long_term_memory.json").read_text(encoding="utf-8") == legacy_ltm
        assert not (mem._DATA_DIR / "default").exists()

    def test_migration_is_idempotent(self, mem, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_legacy_files(mem)

        assert mem.get_working_memory(user_id="default").get_recent(10) == ["legacy shared note"]

        # Repeated access and a simulated restart must not duplicate or lose data.
        monkeypatch.setattr(mem, "_working_memories", {})
        monkeypatch.setattr(mem, "_long_term_memories", {})
        wm = mem.get_working_memory(user_id="default")
        ltm = mem.get_long_term_memory(user_id="default")
        assert wm.get_recent(10) == ["legacy shared note"]
        assert len(ltm.search(include_expired=True)) == 1

    def test_legacy_source_preserved_as_backup_after_migration(self, mem) -> None:
        _write_legacy_files(mem)
        original_wm = (mem._DATA_DIR / "working_memory.json").read_text(encoding="utf-8")
        original_ltm = (mem._DATA_DIR / "long_term_memory.json").read_text(encoding="utf-8")

        mem.get_working_memory(user_id="default")
        mem.get_long_term_memory(user_id="default")

        wm_backup = mem._DATA_DIR / ("working_memory.json" + mem.LEGACY_BACKUP_SUFFIX)
        ltm_backup = mem._DATA_DIR / ("long_term_memory.json" + mem.LEGACY_BACKUP_SUFFIX)
        assert wm_backup.exists()
        assert ltm_backup.exists()
        assert wm_backup.read_text(encoding="utf-8") == original_wm
        assert ltm_backup.read_text(encoding="utf-8") == original_ltm
        # The unscoped originals no longer sit at the shared path.
        assert not (mem._DATA_DIR / "working_memory.json").exists()
        assert not (mem._DATA_DIR / "long_term_memory.json").exists()

    def test_failed_migration_does_not_destroy_original(self, mem) -> None:
        _write_legacy_files(mem)
        original = (mem._DATA_DIR / "long_term_memory.json").read_text(encoding="utf-8")

        def _boom(src: object, dst: object) -> None:
            raise OSError("simulated copy failure")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mem.shutil, "copyfile", _boom)
            with pytest.raises(OSError):
                mem.get_long_term_memory(user_id="default")

        # Original intact, nothing partial exposed as the default store.
        assert (mem._DATA_DIR / "long_term_memory.json").read_text(encoding="utf-8") == original
        assert not (mem._DATA_DIR / "default" / "long_term_memory.json").exists()

        # Recovery: once copying works again, migration completes.
        ltm = mem.get_long_term_memory(user_id="default")
        assert ltm.get_entry("mem_legacy000001") is not None


# =============================================================================
# CLI path
# =============================================================================


class TestCliOwnership:
    def _cmd(self):
        from rex.cli import cmd_memory

        return cmd_memory

    @staticmethod
    def _args(**kwargs) -> argparse.Namespace:
        return argparse.Namespace(**kwargs)

    def test_add_and_search_are_scoped_to_requesting_user(
        self, mem, capsys: pytest.CaptureFixture[str]
    ) -> None:
        add_args = self._args(
            memory_command="add",
            category="medical",
            content='{"note": "alice cli secret"}',
            ttl=None,
            sensitive=False,
            user="alice",
        )
        assert self._cmd()(add_args) == 0
        capsys.readouterr()

        bob_search = self._args(
            memory_command="search",
            keyword="alice cli secret",
            category=None,
            show_sensitive=True,
            user="bob",
        )
        assert self._cmd()(bob_search) == 0
        out = capsys.readouterr().out
        assert "alice cli secret" not in out
        assert "No matching memory entries" in out

        alice_search = self._args(
            memory_command="search",
            keyword="alice cli secret",
            category=None,
            show_sensitive=True,
            user="alice",
        )
        assert self._cmd()(alice_search) == 0
        assert "alice cli secret" in capsys.readouterr().out

    def test_recent_is_scoped_to_requesting_user(
        self, mem, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mem.get_working_memory(user_id="alice").add_entry("alice recent context")

        args = self._args(memory_command="recent", count=10, user="bob")
        assert self._cmd()(args) == 0
        out = capsys.readouterr().out
        assert "alice recent context" not in out
        assert "No working memory entries" in out

    def test_forget_denies_non_owner(self, mem, capsys: pytest.CaptureFixture[str]) -> None:
        entry = mem.get_long_term_memory(user_id="alice").add_entry(
            category="facts", content={"note": "keep"}
        )

        args = self._args(memory_command="forget", entry_id=entry.entry_id, user="bob")
        assert self._cmd()(args) == 1
        assert mem.get_long_term_memory(user_id="alice").get_entry(entry.entry_id) is not None

    def test_clear_only_clears_requesting_user(
        self, mem, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mem.get_working_memory(user_id="alice").add_entry("alice keeps this")
        mem.get_working_memory(user_id="bob").add_entry("bob entry")

        args = self._args(memory_command="clear", user="bob")
        assert self._cmd()(args) == 0
        assert mem.get_working_memory(user_id="alice").get_recent(10) == ["alice keeps this"]
        assert mem.get_working_memory(user_id="bob").get_recent(10) == []

    def test_retention_only_touches_requesting_user(
        self, mem, capsys: pytest.CaptureFixture[str]
    ) -> None:
        alice = mem.get_long_term_memory(user_id="alice")
        past = datetime.now(UTC) - timedelta(days=1)
        expired = mem.MemoryEntry(category="temp", content={"stale": True}, expires_at=past)
        alice._entries[expired.entry_id] = expired

        args = self._args(memory_command="retention", user="bob")
        assert self._cmd()(args) == 0
        assert "deleted 0" in capsys.readouterr().out
        assert expired.entry_id in alice._entries

        args = self._args(memory_command="retention", user="alice")
        assert self._cmd()(args) == 0
        assert "deleted 1" in capsys.readouterr().out

    def test_stats_are_scoped_to_requesting_user(
        self, mem, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mem.get_working_memory(user_id="alice").add_entry("one")
        mem.get_long_term_memory(user_id="alice").add_entry(category="facts", content={"a": 1})

        args = self._args(memory_command="stats", user="bob")
        assert self._cmd()(args) == 0
        out = capsys.readouterr().out
        assert "Working Memory: 0" in out
        assert "Long-Term Memory: 0" in out

    def test_missing_identity_fails_closed(
        self, mem, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("rex.identity.resolve_active_user", lambda *a, **k: None)
        args = self._args(memory_command="stats", user=None)
        assert self._cmd()(args) == 1
        out = capsys.readouterr().out
        assert "No active user" in out
        assert "rex identify" in out

    def test_malformed_explicit_identity_fails_closed(
        self, mem, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = self._args(memory_command="recent", count=5, user="../evil")
        assert self._cmd()(args) == 1


# =============================================================================
# Bridge (Electron GUI) path
# =============================================================================


def _run_bridge(stdin_data: str) -> dict:
    """Invoke the memories bridge main() with the given stdin JSON."""
    mod = importlib.import_module("rex_memories_bridge")

    captured = StringIO()
    with patch("sys.stdin", StringIO(stdin_data)):
        with patch("sys.stdout", captured):
            try:
                mod.main()
            except SystemExit:
                pass
    return json.loads(captured.getvalue().strip())


class TestBridgeOwnership:
    def test_add_records_the_requesting_user(self, mem) -> None:
        result = _run_bridge(
            json.dumps({"command": "add", "user": "alice", "data": {"text": "alice gui memory"}})
        )
        assert result["ok"] is True
        assert len(mem.get_long_term_memory(user_id="alice").search()) == 1
        assert mem.get_long_term_memory(user_id="bob").search() == []

    def test_list_is_scoped_to_requesting_user(self, mem) -> None:
        mem.get_long_term_memory(user_id="alice").add_entry(
            category="general", content={"text": "alice item"}
        )

        result = _run_bridge(json.dumps({"command": "list", "user": "bob"}))
        assert result["ok"] is True
        assert result["memories"] == []

    def test_update_denies_non_owner(self, mem) -> None:
        entry = mem.get_long_term_memory(user_id="alice").add_entry(
            category="general", content={"text": "alice original"}
        )

        result = _run_bridge(
            json.dumps(
                {
                    "command": "update",
                    "user": "bob",
                    "id": entry.entry_id,
                    "data": {"text": "hijacked"},
                }
            )
        )
        assert result["ok"] is False
        intact = mem.get_long_term_memory(user_id="alice").get_entry(entry.entry_id)
        assert intact.content["text"] == "alice original"

    def test_delete_denies_non_owner(self, mem) -> None:
        entry = mem.get_long_term_memory(user_id="alice").add_entry(
            category="general", content={"text": "alice keeps"}
        )

        result = _run_bridge(json.dumps({"command": "delete", "user": "bob", "id": entry.entry_id}))
        assert result["ok"] is False
        assert mem.get_long_term_memory(user_id="alice").get_entry(entry.entry_id) is not None

    def test_missing_identity_fails_closed(self, mem, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rex.identity.resolve_active_user", lambda *a, **k: None)
        result = _run_bridge(json.dumps({"command": "list"}))
        assert result["ok"] is False
        assert "No active user" in result["error"]

    def test_malformed_explicit_identity_fails_closed(self, mem) -> None:
        result = _run_bridge(json.dumps({"command": "list", "user": "../evil"}))
        assert result["ok"] is False
        assert "No active user" in result["error"]


# =============================================================================
# Scheduled cleanup
# =============================================================================


class TestScheduledCleanup:
    @staticmethod
    def _seed_expired(mem, user_id: str, count: int) -> None:
        ltm = mem.get_long_term_memory(user_id=user_id)
        past = datetime.now(UTC) - timedelta(days=1)
        for i in range(count):
            entry = mem.MemoryEntry(
                entry_id=f"exp_{user_id}_{i}",
                category="temp",
                content={"i": i},
                expires_at=past,
            )
            ltm._entries[entry.entry_id] = entry
        ltm._save()

    def test_cleanup_compacts_each_user_store_independently(self, mem) -> None:
        self._seed_expired(mem, "alice", 2)
        self._seed_expired(mem, "bob", 3)
        keeper = mem.get_long_term_memory(user_id="alice").add_entry(
            category="keep", content={"fresh": True}
        )

        results = mem.run_memory_cleanup()

        assert results["alice"] == 2
        assert results["bob"] == 3
        assert mem.get_long_term_memory(user_id="alice").get_entry(keeper.entry_id) is not None

    def test_failure_in_one_store_does_not_stop_others(
        self, mem, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._seed_expired(mem, "alice", 1)
        self._seed_expired(mem, "bob", 2)

        broken = mem.get_long_term_memory(user_id="alice")
        monkeypatch.setattr(broken, "compact", MagicMock(side_effect=OSError("disk error")))

        results = mem.run_memory_cleanup()

        assert "alice" not in results
        assert results["bob"] == 2
        # Bob's store was compacted on disk despite Alice's failure.
        bob_file = mem._DATA_DIR / "bob" / "long_term_memory.json"
        assert "exp_bob_0" not in bob_file.read_text(encoding="utf-8")

    def test_invalid_directory_names_are_never_treated_as_users(self, mem) -> None:
        self._seed_expired(mem, "alice", 1)
        (mem._DATA_DIR / "not a valid user!").mkdir(parents=True)

        assert mem.list_memory_user_ids() == ["alice"]
        results = mem.run_memory_cleanup()
        assert set(results) == {"alice"}

    def test_cleanup_ignores_legacy_root_files(self, mem) -> None:
        _write_legacy_files(mem)
        legacy = (mem._DATA_DIR / "long_term_memory.json").read_text(encoding="utf-8")

        results = mem.run_memory_cleanup()

        assert results == {}
        # Legacy file untouched: cleanup is not an explicit default request.
        assert (mem._DATA_DIR / "long_term_memory.json").read_text(encoding="utf-8") == legacy

    def test_schedule_memory_cleanup_callback_runs_per_user(self, mem) -> None:
        self._seed_expired(mem, "alice", 1)
        self._seed_expired(mem, "bob", 1)

        captured_callback = None

        def capture_callback(name: str, cb: object) -> None:
            nonlocal captured_callback
            captured_callback = cb

        scheduler = MagicMock()
        scheduler.register_callback.side_effect = capture_callback

        mem.schedule_memory_cleanup(scheduler, interval_seconds=60)
        assert captured_callback is not None
        captured_callback(MagicMock())

        assert len(mem.get_long_term_memory(user_id="alice").search(include_expired=True)) == 0
        assert len(mem.get_long_term_memory(user_id="bob").search(include_expired=True)) == 0


# =============================================================================
# Supervisor metrics
# =============================================================================


class TestStoreMetrics:
    def test_metrics_aggregate_counts_without_content(self, mem) -> None:
        mem.get_working_memory(user_id="alice").add_entry("secret alice context")
        mem.get_long_term_memory(user_id="bob").add_entry(
            category="facts", content={"note": "secret bob fact"}
        )

        metrics = mem.memory_store_metrics()

        assert metrics["user_count"] == 2
        assert metrics["working_memory_entries"] == 1
        assert metrics["long_term_memory_entries"] == 1
        assert "secret" not in json.dumps(metrics)
