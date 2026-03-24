"""Tests for rex.openclaw.memory_adapter — US-P3-003.

Covers write, read, and trim of conversation history via MemoryAdapter.
All filesystem I/O is isolated using pytest's tmp_path fixture.
"""

from __future__ import annotations

from rex.openclaw.memory_adapter import MemoryAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adapter(tmp_path) -> MemoryAdapter:
    """Return a MemoryAdapter scoped to a temporary directory."""
    return MemoryAdapter(memory_root=str(tmp_path))


# ---------------------------------------------------------------------------
# Import and instantiation
# ---------------------------------------------------------------------------


class TestMemoryAdapterImport:
    def test_importable(self):
        import rex.openclaw.memory_adapter  # noqa: F401

    def test_instantiates_without_args(self):
        adapter = MemoryAdapter()
        assert adapter is not None

    def test_instantiates_with_memory_root(self, tmp_path):
        adapter = _adapter(tmp_path)
        assert adapter is not None



# ---------------------------------------------------------------------------
# trim_history
# ---------------------------------------------------------------------------


class TestTrimHistory:
    def test_trim_returns_list(self, tmp_path):
        adapter = _adapter(tmp_path)
        history = [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}]
        result = adapter.trim_history(history)
        assert isinstance(result, list)

    def test_trim_respects_limit(self, tmp_path):
        adapter = _adapter(tmp_path)
        history = [{"role": "user", "text": f"msg {i}"} for i in range(10)]
        result = adapter.trim_history(history, limit=3)
        assert len(result) == 3

    def test_trim_keeps_most_recent(self, tmp_path):
        adapter = _adapter(tmp_path)
        history = [{"role": "user", "text": f"msg {i}"} for i in range(5)]
        result = adapter.trim_history(history, limit=2)
        assert result[0]["text"] == "msg 3"
        assert result[1]["text"] == "msg 4"

    def test_trim_empty_history_returns_empty(self, tmp_path):
        adapter = _adapter(tmp_path)
        result = adapter.trim_history([])
        assert result == []

    def test_trim_limit_larger_than_history(self, tmp_path):
        adapter = _adapter(tmp_path)
        history = [{"role": "user", "text": "only one"}]
        result = adapter.trim_history(history, limit=100)
        assert len(result) == 1
        assert result[0]["text"] == "only one"

    def test_trim_accepts_generator(self, tmp_path):
        adapter = _adapter(tmp_path)
        history = ({"role": "user", "text": f"item {i}"} for i in range(4))
        result = adapter.trim_history(history, limit=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# append_entry and load_recent
# ---------------------------------------------------------------------------


class TestAppendAndLoad:
    def test_load_recent_empty_returns_empty_list(self, tmp_path):
        adapter = _adapter(tmp_path)
        result = adapter.load_recent("alice")
        assert result == []

    def test_append_then_load_returns_entry(self, tmp_path):
        adapter = _adapter(tmp_path)
        entry = {"role": "user", "text": "Hello!"}
        adapter.append_entry("alice", entry, max_turns=50)
        result = adapter.load_recent("alice")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["text"] == "Hello!"

    def test_append_multiple_entries_preserves_order(self, tmp_path):
        adapter = _adapter(tmp_path)
        for i in range(3):
            adapter.append_entry("bob", {"role": "user", "text": f"turn {i}"}, max_turns=50)
        result = adapter.load_recent("bob")
        assert [r["text"] for r in result] == ["turn 0", "turn 1", "turn 2"]

    def test_load_recent_limit_applied(self, tmp_path):
        adapter = _adapter(tmp_path)
        for i in range(5):
            adapter.append_entry("carol", {"role": "user", "text": f"msg {i}"}, max_turns=50)
        result = adapter.load_recent("carol", limit=2)
        assert len(result) == 2
        assert result[-1]["text"] == "msg 4"

    def test_append_auto_adds_timestamp(self, tmp_path):
        adapter = _adapter(tmp_path)
        adapter.append_entry("dave", {"role": "user", "text": "hi"}, max_turns=50)
        result = adapter.load_recent("dave")
        assert "timestamp" in result[0]

    def test_append_respects_max_turns(self, tmp_path):
        adapter = _adapter(tmp_path)
        for i in range(10):
            adapter.append_entry("eve", {"role": "user", "text": f"msg {i}"}, max_turns=3)
        result = adapter.load_recent("eve")
        assert len(result) == 3
        assert result[-1]["text"] == "msg 9"

    def test_different_users_isolated(self, tmp_path):
        adapter = _adapter(tmp_path)
        adapter.append_entry("user-a", {"role": "user", "text": "from a"}, max_turns=50)
        adapter.append_entry("user-b", {"role": "user", "text": "from b"}, max_turns=50)
        result_a = adapter.load_recent("user-a")
        result_b = adapter.load_recent("user-b")
        assert result_a[0]["text"] == "from a"
        assert result_b[0]["text"] == "from b"

    def test_load_recent_returns_list_of_dicts(self, tmp_path):
        adapter = _adapter(tmp_path)
        adapter.append_entry("frank", {"role": "assistant", "text": "Hi!"}, max_turns=50)
        result = adapter.load_recent("frank")
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
