"""Tests for US-070: Memory search."""

from __future__ import annotations

from rex.memory import LongTermMemory


def _make_ltm(tmp_path):
    storage = tmp_path / "ltm.json"
    return LongTermMemory(storage_path=storage)


class TestMemorySearchable:
    """AC: memory entries searchable."""

    def test_search_returns_list(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_entry("facts", {"topic": "python"})
        results = ltm.search(keyword="python")
        assert isinstance(results, list)

    def test_search_all_entries_no_filter(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_entry("facts", {"a": "1"})
        ltm.add_entry("prefs", {"b": "2"})
        results = ltm.search()
        assert len(results) == 2

    def test_search_by_category(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_entry("facts", {"x": "y"})
        ltm.add_entry("prefs", {"x": "y"})
        results = ltm.search(category="facts")
        assert len(results) == 1
        assert results[0].category == "facts"


class TestKeywordReturnsEntry:
    """AC: search for a stored memory keyword returns that memory entry in results."""

    def test_keyword_in_value_found(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_entry("notes", {"content": "favourite_food"})
        ltm.add_entry("notes", {"content": "other_entry"})
        results = ltm.search(keyword="favourite_food")
        assert len(results) == 1
        assert results[0].content["content"] == "favourite_food"

    def test_keyword_in_key_found(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_entry("prefs", {"darkmode": True})
        results = ltm.search(keyword="darkmode")
        assert len(results) == 1

    def test_keyword_case_insensitive(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_entry("facts", {"info": "Python Language"})
        results = ltm.search(keyword="python")
        assert len(results) == 1

    def test_keyword_in_category_found(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_entry("shopping_list", {"item": "milk"})
        results = ltm.search(keyword="shopping")
        assert len(results) == 1

    def test_keyword_not_present_returns_empty(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_entry("facts", {"topic": "weather"})
        results = ltm.search(keyword="zzznomatch")
        assert results == []


class TestQueryFailuresHandled:
    """AC: query failures handled."""

    def test_empty_store_returns_empty_list(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        results = ltm.search(keyword="anything")
        assert results == []

    def test_corrupt_storage_starts_fresh(self, tmp_path):
        storage = tmp_path / "ltm.json"
        storage.write_text("not valid json")
        ltm = LongTermMemory(storage_path=storage)
        results = ltm.search()
        assert results == []

    def test_search_after_forget_entry(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        entry = ltm.add_entry("temp", {"key": "gone"})
        ltm.forget(entry.entry_id)
        results = ltm.search(keyword="gone")
        assert results == []
