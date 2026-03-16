"""Unit tests for rex.notifications.digest — DigestBuilder."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.notifications.digest import DigestBackend, DigestBuilder
from rex.notifications.models import Notification, NotificationStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> NotificationStore:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return NotificationStore(db_path=Path(tmp.name))


def _make_notification(
    title: str = "Test",
    digest_eligible: bool = True,
    source: str = "test",
) -> Notification:
    return Notification(
        title=title,
        body="Body",
        source=source,
        priority="low",
        digest_eligible=digest_eligible,
    )


def _mock_backend(response: str = "Mocked digest summary.") -> DigestBackend:
    mock: MagicMock = MagicMock(spec=DigestBackend)
    mock.generate.return_value = response
    return mock  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# build_digest — returns None when nothing to digest
# ---------------------------------------------------------------------------


class TestBuildDigestEmpty:
    def test_returns_none_when_no_eligible(self) -> None:
        store = _make_store()
        builder = DigestBuilder(store)
        assert builder.build_digest() is None

    def test_returns_none_when_all_not_eligible(self) -> None:
        store = _make_store()
        n = _make_notification(digest_eligible=False)
        store.add(n)
        builder = DigestBuilder(store)
        assert builder.build_digest() is None

    def test_returns_none_when_all_already_read(self) -> None:
        store = _make_store()
        n = _make_notification(digest_eligible=True)
        store.add(n)
        store.mark_read(n.id)
        builder = DigestBuilder(store)
        assert builder.build_digest() is None


# ---------------------------------------------------------------------------
# build_digest — LLM path
# ---------------------------------------------------------------------------


class TestBuildDigestLLM:
    def test_calls_backend_generate(self) -> None:
        store = _make_store()
        n = _make_notification()
        store.add(n)
        backend = _mock_backend("You have 1 update(s): ...")
        builder = DigestBuilder(store, backend=backend)
        result = builder.build_digest()
        backend.generate.assert_called_once()  # type: ignore[union-attr]
        assert result == "You have 1 update(s): ..."

    def test_returns_stripped_llm_response(self) -> None:
        store = _make_store()
        store.add(_make_notification())
        backend = _mock_backend("  Summary text.  \n")
        builder = DigestBuilder(store, backend=backend)
        assert builder.build_digest() == "Summary text."

    def test_multiple_notifications_included_in_prompt(self) -> None:
        store = _make_store()
        store.add(_make_notification(title="Alert A"))
        store.add(_make_notification(title="Alert B"))
        backend = _mock_backend("Two updates.")
        builder = DigestBuilder(store, backend=backend)
        builder.build_digest()
        call_args = backend.generate.call_args  # type: ignore[union-attr]
        messages = call_args[0][0]
        user_content = messages[1]["content"]
        assert "Alert A" in user_content
        assert "Alert B" in user_content

    def test_falls_back_on_llm_exception(self) -> None:
        store = _make_store()
        store.add(_make_notification(title="Item"))
        backend = _mock_backend()
        backend.generate.side_effect = RuntimeError("LLM offline")  # type: ignore[union-attr]
        builder = DigestBuilder(store, backend=backend)
        result = builder.build_digest()
        assert result is not None
        assert "Item" in result


# ---------------------------------------------------------------------------
# build_digest — fallback (no backend)
# ---------------------------------------------------------------------------


class TestBuildDigestFallback:
    def test_no_backend_produces_plain_summary(self) -> None:
        store = _make_store()
        store.add(_make_notification(title="Reminder"))
        builder = DigestBuilder(store)
        result = builder.build_digest()
        assert result is not None
        assert "1 update" in result
        assert "Reminder" in result

    def test_multiple_items_in_fallback(self) -> None:
        store = _make_store()
        store.add(_make_notification(title="First"))
        store.add(_make_notification(title="Second"))
        builder = DigestBuilder(store)
        result = builder.build_digest()
        assert result is not None
        assert "First" in result
        assert "Second" in result


# ---------------------------------------------------------------------------
# run_digest
# ---------------------------------------------------------------------------


class TestRunDigest:
    def test_no_op_when_nothing_to_digest(self) -> None:
        store = _make_store()
        builder = DigestBuilder(store)
        with patch("rex.notifications.digest._send_desktop") as mock_desktop:
            builder.run_digest()
        mock_desktop.assert_not_called()

    def test_sends_desktop_notification(self) -> None:
        store = _make_store()
        store.add(_make_notification(title="Item"))
        builder = DigestBuilder(store, backend=_mock_backend("Summary"))
        with patch("rex.notifications.digest._send_desktop") as mock_desktop:
            builder.run_digest()
        mock_desktop.assert_called_once_with("Rex Digest", "Summary")

    def test_marks_notifications_as_read(self) -> None:
        store = _make_store()
        n = _make_notification()
        store.add(n)
        builder = DigestBuilder(store, backend=_mock_backend("S"))
        with patch("rex.notifications.digest._send_desktop"):
            builder.run_digest()
        assert store.get_unread() == []

    def test_sets_delivered_at(self) -> None:
        store = _make_store()
        n = _make_notification()
        store.add(n)
        builder = DigestBuilder(store, backend=_mock_backend("S"))
        with patch("rex.notifications.digest._send_desktop"):
            builder.run_digest()
        # After mark_read the notification is gone from get_unread;
        # verify delivered_at was set by reading directly from DB.
        import sqlite3

        con = sqlite3.connect(store._db_path)  # type: ignore[attr-defined]
        row = con.execute(
            "SELECT delivered_at FROM notifications WHERE id = ?", (n.id,)
        ).fetchone()
        con.close()
        assert row is not None
        assert row[0] is not None

    def test_dispatches_only_one_desktop_notification(self) -> None:
        store = _make_store()
        store.add(_make_notification(title="A"))
        store.add(_make_notification(title="B"))
        builder = DigestBuilder(store, backend=_mock_backend("S"))
        with patch("rex.notifications.digest._send_desktop") as mock_desktop:
            builder.run_digest()
        assert mock_desktop.call_count == 1
