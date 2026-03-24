"""Tests for rex.openclaw.memory_adapter — US-P3-003 / US-010.

Covers write, read, and trim of conversation history via MemoryAdapter.
All filesystem I/O is isolated using pytest's tmp_path fixture.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# US-010: dual-write behaviour and local-only reads
# ---------------------------------------------------------------------------


class TestDualWriteBehaviour:
    """Verify that append_entry always writes locally and load_recent always
    reads locally, regardless of whether the OpenClaw gateway is configured.

    The OpenClaw "dual-write" is: write locally NOW + OpenClaw session is
    updated automatically on the next RexAgent.respond() call via the `user`
    field in /v1/chat/completions.  MemoryAdapter itself makes no HTTP calls.
    """

    def test_append_entry_always_calls_rex_append(self, tmp_path):
        """append_entry() must delegate to rex.memory.append_history_entry."""
        with patch("rex.openclaw.memory_adapter._rex_append") as mock_append:
            adapter = MemoryAdapter(memory_root=str(tmp_path))
            entry = {"role": "user", "text": "hello"}
            adapter.append_entry("alice", entry, max_turns=50)
            mock_append.assert_called_once()
            call_args = mock_append.call_args
            assert call_args.args[0] == "alice"
            assert call_args.args[1] == entry

    def test_append_entry_calls_rex_append_when_gateway_configured(self, tmp_path):
        """append_entry() writes locally even when an OpenClaw gateway client exists."""
        mock_client = MagicMock()
        with (
            patch("rex.openclaw.memory_adapter._rex_append") as mock_append,
            patch(
                "rex.openclaw.http_client.get_openclaw_client",
                return_value=mock_client,
            ),
        ):
            adapter = MemoryAdapter(memory_root=str(tmp_path))
            adapter.append_entry("bob", {"role": "user", "text": "test"}, max_turns=50)
            # Local write always happens
            mock_append.assert_called_once()
            # append_entry itself makes no HTTP calls (OpenClaw sync is via respond())
            mock_client.post.assert_not_called()

    def test_load_recent_always_calls_rex_load_recent(self, tmp_path):
        """load_recent() must always delegate to rex.memory.load_recent_history."""
        mock_client = MagicMock()
        with (
            patch(
                "rex.openclaw.memory_adapter._rex_load_recent",
                return_value=[{"role": "user", "text": "hi", "timestamp": "t"}],
            ) as mock_load,
            patch(
                "rex.openclaw.http_client.get_openclaw_client",
                return_value=mock_client,
            ),
        ):
            adapter = MemoryAdapter(memory_root=str(tmp_path))
            result = adapter.load_recent("carol", limit=10)
            # Always reads from local Rex memory
            mock_load.assert_called_once()
            # No HTTP calls made for reads
            mock_client.get.assert_not_called()
            assert isinstance(result, list)

    def test_load_recent_reads_local_without_gateway(self, tmp_path):
        """load_recent() reads locally even with no gateway configured."""
        with patch(
            "rex.openclaw.memory_adapter._rex_load_recent",
            return_value=[],
        ) as mock_load:
            adapter = MemoryAdapter(memory_root=str(tmp_path))
            adapter.load_recent("dave")
            mock_load.assert_called_once()

    def test_trim_history_local_only(self, tmp_path):
        """trim_history() always operates locally — no HTTP calls."""
        mock_client = MagicMock()
        with patch(
            "rex.openclaw.http_client.get_openclaw_client",
            return_value=mock_client,
        ):
            adapter = MemoryAdapter(memory_root=str(tmp_path))
            history = [{"role": "user", "text": f"msg {i}"} for i in range(5)]
            result = adapter.trim_history(history, limit=2)
            assert len(result) == 2
            mock_client.get.assert_not_called()
            mock_client.post.assert_not_called()
