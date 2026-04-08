"""Tests for US-039: Telegram bot client."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from rex.assistant_errors import IntegrationNotConfiguredError
from rex.integrations.telegram.client import TelegramClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(payload: dict) -> MagicMock:
    """Return a mock HTTP response that yields *payload* as JSON."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


_OK_RESPONSE = {
    "ok": True,
    "result": {
        "message_id": 42,
        "chat": {"id": 12345, "type": "private"},
        "text": "hello",
    },
}


# ---------------------------------------------------------------------------
# Configuration checks
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_raises_when_no_token(self):
        client = TelegramClient(bot_token=None, chat_id="123")
        with pytest.raises(IntegrationNotConfiguredError, match="TELEGRAM_BOT_TOKEN"):
            client.send_message("hi")

    def test_raises_when_empty_token(self):
        client = TelegramClient(bot_token="", chat_id="123")
        with pytest.raises(IntegrationNotConfiguredError, match="TELEGRAM_BOT_TOKEN"):
            client.send_message("hi")

    def test_raises_when_no_chat_id(self):
        client = TelegramClient(bot_token="tok:ABC", chat_id=None)
        with pytest.raises(IntegrationNotConfiguredError, match="telegram_chat_id"):
            client.send_message("hi")

    def test_raises_when_empty_chat_id(self):
        client = TelegramClient(bot_token="tok:ABC", chat_id="")
        with pytest.raises(IntegrationNotConfiguredError, match="telegram_chat_id"):
            client.send_message("hi")


# ---------------------------------------------------------------------------
# send_message happy path
# ---------------------------------------------------------------------------

class TestSendMessage:
    def _client(self) -> TelegramClient:
        return TelegramClient(bot_token="tok:TEST123", chat_id="99999")

    def test_returns_api_response(self):
        client = self._client()
        with patch("urllib.request.urlopen", return_value=_make_response(_OK_RESPONSE)):
            result = client.send_message("hello")
        assert result["ok"] is True
        assert result["result"]["message_id"] == 42

    def test_posts_to_correct_url(self):
        client = self._client()
        captured: list[str] = []

        def fake_urlopen(req, timeout=None):
            captured.append(req.full_url)
            return _make_response(_OK_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.send_message("test")

        assert "tok:TEST123/sendMessage" in captured[0]

    def test_payload_contains_chat_id_and_text(self):
        client = self._client()
        captured_payloads: list[dict] = []

        def fake_urlopen(req, timeout=None):
            captured_payloads.append(json.loads(req.data.decode()))
            return _make_response(_OK_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.send_message("greetings")

        payload = captured_payloads[0]
        assert payload["chat_id"] == "99999"
        assert payload["text"] == "greetings"

    def test_raises_value_error_on_empty_text(self):
        client = self._client()
        with pytest.raises(ValueError, match="non-empty"):
            client.send_message("")

    def test_raises_value_error_on_whitespace_only_text(self):
        client = self._client()
        with pytest.raises(ValueError, match="non-empty"):
            client.send_message("   ")

    def test_send_message_with_multiline_text(self):
        client = self._client()
        with patch("urllib.request.urlopen", return_value=_make_response(_OK_RESPONSE)):
            result = client.send_message("line1\nline2\nline3")
        assert result["ok"] is True

    def test_uses_configured_timeout(self):
        client = TelegramClient(bot_token="tok:X", chat_id="1", timeout=5.0)
        timeouts: list[float] = []

        def fake_urlopen(req, timeout=None):
            timeouts.append(timeout)
            return _make_response(_OK_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.send_message("ping")

        assert timeouts[0] == 5.0


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def _client(self) -> TelegramClient:
        return TelegramClient(bot_token="tok:TEST", chat_id="111")

    def test_http_error_propagates(self):
        client = self._client()
        error = urllib.error.HTTPError(
            url="https://example.com",
            code=400,
            msg="Bad Request",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(b'{"description":"Bad Request"}'),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(urllib.error.HTTPError):
                client.send_message("hello")

    def test_url_error_propagates(self):
        client = self._client()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("network unreachable"),
        ):
            with pytest.raises(urllib.error.URLError):
                client.send_message("hello")


# ---------------------------------------------------------------------------
# Config field integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    def test_appconfig_has_telegram_fields(self):
        from rex.config import AppConfig

        cfg = AppConfig()
        assert hasattr(cfg, "telegram_bot_token")
        assert hasattr(cfg, "telegram_chat_id")
        assert cfg.telegram_bot_token is None
        assert cfg.telegram_chat_id is None

    def test_appconfig_accepts_telegram_values(self):
        from rex.config import AppConfig

        cfg = AppConfig(telegram_bot_token="tok:XYZ", telegram_chat_id="54321")
        assert cfg.telegram_bot_token == "tok:XYZ"
        assert cfg.telegram_chat_id == "54321"

    def test_build_app_config_reads_token_from_env(self, monkeypatch):
        from rex.config import build_app_config

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok:ENV_TEST")
        cfg = build_app_config(json_config={})
        assert cfg.telegram_bot_token == "tok:ENV_TEST"

    def test_build_app_config_reads_chat_id_from_json(self):
        from rex.config import build_app_config

        cfg = build_app_config(json_config={"telegram": {"chat_id": "777"}})
        assert cfg.telegram_chat_id == "777"
