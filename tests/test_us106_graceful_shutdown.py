"""Tests for US-106: Graceful shutdown.

Acceptance criteria:
- SIGTERM signal registered and handled in the main process
- on SIGTERM, no new requests accepted and in-flight requests given up to
  a configurable drain timeout (default: 10s) to complete
- open database connections and background jobs closed cleanly on shutdown
- process exits with code 0 after clean shutdown
- Typecheck passes
"""

from __future__ import annotations

import signal
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rex.graceful_shutdown import (
    GracefulShutdown,
    get_shutdown_handler,
    reset_shutdown_handler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton() -> Any:
    """Reset the module-level singleton before and after each test."""
    reset_shutdown_handler()
    yield
    reset_shutdown_handler()


# ---------------------------------------------------------------------------
# GracefulShutdown construction and config
# ---------------------------------------------------------------------------


class TestGracefulShutdownInit:
    def test_default_drain_timeout(self) -> None:
        handler = GracefulShutdown()
        assert handler.drain_timeout == 10

    def test_explicit_drain_timeout(self) -> None:
        handler = GracefulShutdown(drain_timeout=30)
        assert handler.drain_timeout == 30

    def test_drain_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REX_SHUTDOWN_TIMEOUT", "25")
        handler = GracefulShutdown()
        assert handler.drain_timeout == 25

    def test_invalid_env_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REX_SHUTDOWN_TIMEOUT", "not-a-number")
        handler = GracefulShutdown()
        assert handler.drain_timeout == 10

    def test_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REX_SHUTDOWN_TIMEOUT", "99")
        handler = GracefulShutdown(drain_timeout=5)
        assert handler.drain_timeout == 5

    def test_not_shutting_down_initially(self) -> None:
        handler = GracefulShutdown()
        assert not handler.is_shutting_down


# ---------------------------------------------------------------------------
# install() — SIGTERM registration
# ---------------------------------------------------------------------------


class TestInstall:
    def test_install_registers_sigterm(self) -> None:
        handler = GracefulShutdown()
        handler.install()
        registered = signal.getsignal(signal.SIGTERM)
        assert registered == handler._handle_sigterm

    def test_install_idempotent(self) -> None:
        handler = GracefulShutdown()
        handler.install()
        handler.install()
        registered = signal.getsignal(signal.SIGTERM)
        assert registered == handler._handle_sigterm

    def test_install_logs_info(self) -> None:
        handler = GracefulShutdown(drain_timeout=7)
        with patch("rex.graceful_shutdown.logger") as mock_logger:
            handler.install()
        mock_logger.info.assert_called_once()
        logged_msg = str(mock_logger.info.call_args)
        assert "7" in logged_msg


# ---------------------------------------------------------------------------
# trigger_shutdown() — sets flag without spawning drain thread
# ---------------------------------------------------------------------------


class TestTriggerShutdown:
    def test_trigger_sets_flag(self) -> None:
        handler = GracefulShutdown()
        assert not handler.is_shutting_down
        handler.trigger_shutdown()
        assert handler.is_shutting_down

    def test_trigger_idempotent(self) -> None:
        handler = GracefulShutdown()
        handler.trigger_shutdown()
        handler.trigger_shutdown()
        assert handler.is_shutting_down


# ---------------------------------------------------------------------------
# register_cleanup() — cleanup callbacks
# ---------------------------------------------------------------------------


class TestRegisterCleanup:
    def test_cleanup_registered(self) -> None:
        handler = GracefulShutdown()
        fn = MagicMock()
        handler.register_cleanup(fn)
        assert fn in handler._cleanups

    def test_multiple_cleanups_registered(self) -> None:
        handler = GracefulShutdown()
        fns = [MagicMock() for _ in range(3)]
        for fn in fns:
            handler.register_cleanup(fn)
        assert handler._cleanups == fns

    def test_cleanups_called_in_order_during_drain(self) -> None:
        call_order: list[str] = []
        handler = GracefulShutdown(drain_timeout=0)
        handler.register_cleanup(lambda: call_order.append("first"))
        handler.register_cleanup(lambda: call_order.append("second"))
        handler.register_cleanup(lambda: call_order.append("third"))

        with patch("sys.exit"):
            handler._drain_and_exit()

        assert call_order == ["first", "second", "third"]

    def test_cleanup_exception_does_not_stop_subsequent_cleanups(self) -> None:
        call_order: list[str] = []
        handler = GracefulShutdown(drain_timeout=0)
        handler.register_cleanup(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        handler.register_cleanup(lambda: call_order.append("after"))

        with patch("sys.exit"):
            handler._drain_and_exit()

        assert call_order == ["after"]

    def test_cleanup_for_background_job(self) -> None:
        scheduler = MagicMock()
        handler = GracefulShutdown(drain_timeout=0)
        handler.register_cleanup(scheduler.stop)

        with patch("sys.exit"):
            handler._drain_and_exit()

        scheduler.stop.assert_called_once()

    def test_cleanup_for_db_connection(self) -> None:
        db = MagicMock()
        handler = GracefulShutdown(drain_timeout=0)
        handler.register_cleanup(db.close)

        with patch("sys.exit"):
            handler._drain_and_exit()

        db.close.assert_called_once()


# ---------------------------------------------------------------------------
# _drain_and_exit() — drain then sys.exit(0)
# ---------------------------------------------------------------------------


class TestDrainAndExit:
    def test_exits_with_code_0(self) -> None:
        handler = GracefulShutdown(drain_timeout=0)
        with patch("sys.exit") as mock_exit:
            handler._drain_and_exit()
        mock_exit.assert_called_once_with(0)

    def test_logs_during_drain(self) -> None:
        handler = GracefulShutdown(drain_timeout=0)
        with patch("rex.graceful_shutdown.logger") as mock_logger, patch("sys.exit"):
            handler._drain_and_exit()
        # Should log at least "Drain" and "shutdown complete"
        assert mock_logger.info.call_count >= 2

    def test_drain_respects_timeout_zero(self) -> None:
        """drain_timeout=0 should return almost immediately."""
        handler = GracefulShutdown(drain_timeout=0)
        exited = threading.Event()

        def run() -> None:
            with patch("sys.exit", side_effect=SystemExit(0)):
                try:
                    handler._drain_and_exit()
                except SystemExit:
                    pass
            exited.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        assert exited.wait(timeout=2), "drain did not complete within 2s"


# ---------------------------------------------------------------------------
# _handle_sigterm() — spawns drain thread
# ---------------------------------------------------------------------------


class TestHandleSigterm:
    def test_sigterm_sets_shutting_down_flag(self) -> None:
        handler = GracefulShutdown(drain_timeout=0)

        with patch.object(handler, "_drain_and_exit"):
            handler._handle_sigterm(signal.SIGTERM, None)

        assert handler.is_shutting_down

    def test_sigterm_spawns_drain_thread(self) -> None:
        handler = GracefulShutdown(drain_timeout=0)

        with patch("rex.graceful_shutdown.threading") as mock_threading:
            mock_event = MagicMock()
            mock_threading.Event.return_value = mock_event
            mock_thread = MagicMock()
            mock_threading.Thread.return_value = mock_thread
            handler._handle_sigterm(signal.SIGTERM, None)

        mock_thread.start.assert_called_once()


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_shutdown_handler_returns_instance(self) -> None:
        h = get_shutdown_handler()
        assert isinstance(h, GracefulShutdown)

    def test_get_shutdown_handler_returns_same_instance(self) -> None:
        h1 = get_shutdown_handler()
        h2 = get_shutdown_handler()
        assert h1 is h2

    def test_get_shutdown_handler_with_drain_timeout(self) -> None:
        h = get_shutdown_handler(drain_timeout=20)
        assert h.drain_timeout == 20

    def test_reset_allows_new_singleton(self) -> None:
        h1 = get_shutdown_handler()
        reset_shutdown_handler()
        h2 = get_shutdown_handler()
        assert h1 is not h2

    def test_reset_creates_clean_handler(self) -> None:
        h1 = get_shutdown_handler()
        h1.trigger_shutdown()
        assert h1.is_shutting_down
        reset_shutdown_handler()
        h2 = get_shutdown_handler()
        assert not h2.is_shutting_down


# ---------------------------------------------------------------------------
# Flask integration — rex_speak_api.py
# ---------------------------------------------------------------------------


class TestFlaskIntegrationSpeakApi:
    @pytest.fixture()
    def speak_client(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Return a test client for rex_speak_api."""
        monkeypatch.setenv("REX_SPEAK_API_KEY", "test-key-abc")
        import rex_speak_api as api

        api.app.config["TESTING"] = True
        return api.app.test_client()

    def test_speak_endpoint_returns_503_when_shutting_down(
        self, speak_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REX_SPEAK_API_KEY", "test-key-abc")
        # Put the singleton into shutdown mode
        handler = get_shutdown_handler()
        handler.trigger_shutdown()

        resp = speak_client.post(
            "/speak",
            json={"text": "hello"},
            headers={"X-API-Key": "test-key-abc"},
        )
        assert resp.status_code == 503

    def test_speak_endpoint_not_503_before_shutdown(
        self, speak_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REX_SPEAK_API_KEY", "test-key-abc")
        handler = get_shutdown_handler()
        assert not handler.is_shutting_down
        # Mock TTS so it doesn't try to load models; use a TextToSpeechError
        # to return a known non-503 status.
        from rex.assistant_errors import TextToSpeechError as TtsErr

        with patch("rex_speak_api._get_tts_engine", side_effect=TtsErr("no tts")):
            resp = speak_client.post(
                "/speak",
                json={"text": "hello"},
                headers={"X-API-Key": "test-key-abc"},
            )
        assert resp.status_code != 503


# ---------------------------------------------------------------------------
# Flask integration — agent_server.py
# ---------------------------------------------------------------------------


class TestFlaskIntegrationAgentServer:
    @pytest.fixture()
    def agent_app(self) -> Any:
        from rex.computers.agent_server import create_app

        flask_app = create_app(token="test-token", allowlist=frozenset({"echo"}))
        flask_app.config["TESTING"] = True
        return flask_app.test_client()

    def test_health_returns_503_when_shutting_down(self, agent_app: Any) -> None:
        handler = get_shutdown_handler()
        handler.trigger_shutdown()

        resp = agent_app.get("/health", headers={"X-Auth-Token": "test-token"})
        assert resp.status_code == 503

    def test_health_returns_200_when_not_shutting_down(self, agent_app: Any) -> None:
        handler = get_shutdown_handler()
        assert not handler.is_shutting_down

        resp = agent_app.get("/health", headers={"X-Auth-Token": "test-token"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# main() installs the handler (rex_speak_api, agent_server)
# ---------------------------------------------------------------------------


class TestMainInstallsHandler:
    def test_speak_api_main_installs_sigterm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REX_SPEAK_API_KEY", "test-key-abc")
        import rex_speak_api as api

        with (
            patch("rex_speak_api.get_shutdown_handler") as mock_get,
            patch.object(api.app, "run"),
        ):
            mock_handler = MagicMock()
            mock_get.return_value = mock_handler
            api.main.__wrapped__()  # call without @wrap_entrypoint side-effects

        mock_handler.install.assert_called_once()

    def test_agent_server_main_installs_sigterm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REX_AGENT_TOKEN", "test-token")
        from rex.computers import agent_server

        with (
            patch("rex.computers.agent_server.get_shutdown_handler") as mock_get,
            patch("rex.computers.agent_server._build_default_app") as mock_app_builder,
        ):
            mock_handler = MagicMock()
            mock_get.return_value = mock_handler
            mock_flask_app = MagicMock()
            mock_app_builder.return_value = mock_flask_app

            agent_server.main.__wrapped__()

        mock_handler.install.assert_called_once()
