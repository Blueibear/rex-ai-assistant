"""US-032: Memory storage acceptance tests.

Verifies:
- memory records saved
- storage persistent
- retrieval possible
- Typecheck passes (mypy rex/)
"""

from __future__ import annotations

from pathlib import Path

from rex.memory import (
    LongTermMemory,
    WorkingMemory,
)


def test_memory_records_saved(tmp_path: Path) -> None:
    """Memory records are saved to disk after add_entry."""
    storage = tmp_path / "ltm.json"
    ltm = LongTermMemory(storage_path=storage)
    ltm.add_entry(category="facts", content={"key": "us032"})
    assert storage.exists()
    assert len(ltm) == 1


def test_storage_persistent(tmp_path: Path) -> None:
    """Entries survive a new LongTermMemory instance loading the same file."""
    storage = tmp_path / "ltm.json"
    ltm1 = LongTermMemory(storage_path=storage)
    entry = ltm1.add_entry(category="persist", content={"value": "hello"})

    ltm2 = LongTermMemory(storage_path=storage)
    assert len(ltm2) == 1
    retrieved = ltm2.get_entry(entry.entry_id)
    assert retrieved is not None
    assert retrieved.content["value"] == "hello"


def test_retrieval_possible(tmp_path: Path) -> None:
    """Stored entry can be retrieved by ID and via search."""
    storage = tmp_path / "ltm.json"
    ltm = LongTermMemory(storage_path=storage)
    entry = ltm.add_entry(category="retrieval", content={"topic": "us032-test"})

    # retrieval by ID
    by_id = ltm.get_entry(entry.entry_id)
    assert by_id is not None
    assert by_id.entry_id == entry.entry_id

    # retrieval via search
    results = ltm.search(category="retrieval")
    assert len(results) == 1
    assert results[0].entry_id == entry.entry_id


def test_working_memory_records_saved(tmp_path: Path) -> None:
    """WorkingMemory entries are saved and retrievable."""
    storage = tmp_path / "wm.json"
    wm = WorkingMemory(storage_path=storage)
    wm.add_entry("us032 working memory entry")
    assert storage.exists()
    recent = wm.get_recent(1)
    assert recent == ["us032 working memory entry"]


def test_working_memory_persistent(tmp_path: Path) -> None:
    """WorkingMemory entries survive a new instance loading the same file."""
    storage = tmp_path / "wm.json"
    wm1 = WorkingMemory(storage_path=storage)
    wm1.add_entry("persistent working memory")

    wm2 = WorkingMemory(storage_path=storage)
    recent = wm2.get_recent(1)
    assert recent == ["persistent working memory"]
