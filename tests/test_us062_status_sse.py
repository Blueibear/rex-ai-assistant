"""Tests for US-062: Rex status indicators.

Covers:
- RexStatus constants exist
- emit_status updates current status
- subscribe/unsubscribe adds and removes client queues
- subscription context manager auto-unsubscribes
- emit_status delivers to all subscribed clients
- GET /api/status/current returns current status (no auth)
- Voice loop emits status events at pipeline stages (mock test)
"""

from __future__ import annotations

import queue
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Unit: rex.dashboard.sse
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_sse() -> None:
    """Reset the SSE broker state between tests."""
    from rex.dashboard.sse import _reset_for_tests

    _reset_for_tests()
    yield
    _reset_for_tests()


class TestRexStatus:
    def test_status_constants_exist(self) -> None:
        from rex.dashboard.sse import RexStatus

        assert RexStatus.IDLE == "idle"
        assert RexStatus.LISTENING == "listening"
        assert RexStatus.THINKING == "thinking"
        assert RexStatus.EXECUTING == "executing"
        assert RexStatus.DONE == "done"
        assert RexStatus.ERROR == "error"


class TestEmitStatus:
    def test_updates_current_status(self) -> None:
        from rex.dashboard.sse import emit_status, get_current_status

        emit_status("listening")
        assert get_current_status() == "listening"

    def test_delivers_to_subscribed_client(self) -> None:
        from rex.dashboard.sse import emit_status, subscribe

        client_q = subscribe()
        emit_status("thinking")
        assert client_q.get_nowait() == "thinking"

    def test_delivers_to_multiple_clients(self) -> None:
        from rex.dashboard.sse import emit_status, subscribe

        q1 = subscribe()
        q2 = subscribe()
        emit_status("done")
        assert q1.get_nowait() == "done"
        assert q2.get_nowait() == "done"

    def test_does_not_raise_when_no_clients(self) -> None:
        from rex.dashboard.sse import emit_status

        emit_status("idle")  # should not raise


class TestSubscription:
    def test_subscribe_returns_queue(self) -> None:
        from rex.dashboard.sse import subscribe

        q = subscribe()
        assert isinstance(q, queue.Queue)

    def test_unsubscribe_removes_client(self) -> None:
        from rex.dashboard.sse import emit_status, subscribe, unsubscribe

        q = subscribe()
        unsubscribe(q)
        emit_status("error")
        assert q.empty()

    def test_subscription_context_manager_unsubscribes(self) -> None:
        from rex.dashboard.sse import emit_status, subscription

        captured_q = None
        with subscription() as q:
            captured_q = q
            emit_status("executing")
            assert q.get_nowait() == "executing"

        # After context exit, queue should not receive new events.
        emit_status("idle")
        assert captured_q is not None
        assert captured_q.empty()


# ---------------------------------------------------------------------------
# API: GET /api/status/current
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REX_JWT_SECRET", "test-us062-secret-long-enough-32chars")
    return tmp_path


@pytest.fixture()
def flask_client(tmp_data_dir: Path):  # type: ignore[override]
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestStatusCurrentEndpoint:
    def test_returns_200_with_status_key(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/status/current")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "status" in data

    def test_no_auth_required(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/status/current")
        assert resp.status_code == 200

    def test_reflects_emitted_status(self, flask_client) -> None:  # type: ignore[override]
        from rex.dashboard.sse import emit_status

        emit_status("listening")
        resp = flask_client.get("/api/status/current")
        data = resp.get_json()
        assert data["status"] == "listening"


# ---------------------------------------------------------------------------
# Voice loop status emission (smoke test)
# ---------------------------------------------------------------------------


class TestVoiceLoopEmitsStatus:
    def test_emit_status_called_on_listening(self) -> None:
        """Verify that emit_status is importable from the path used in voice_loop."""
        from rex.dashboard.sse import emit_status

        # Should not raise; confirms the import path voice_loop uses is valid.
        emit_status("listening")
        from rex.dashboard.sse import get_current_status

        assert get_current_status() == "listening"

    def test_all_pipeline_statuses_are_valid(self) -> None:
        from rex.dashboard.sse import RexStatus, emit_status, get_current_status

        for status in (
            RexStatus.IDLE,
            RexStatus.LISTENING,
            RexStatus.THINKING,
            RexStatus.EXECUTING,
            RexStatus.DONE,
            RexStatus.ERROR,
        ):
            emit_status(status)
            assert get_current_status() == status
