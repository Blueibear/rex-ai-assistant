"""Tests for US-040: Telegram bot integration (receive commands)."""

from __future__ import annotations

import asyncio
import json
import threading
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from rex.assistant_errors import IntegrationNotConfiguredError
from rex.integrations.telegram.receiver import TelegramReceiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_assistant(reply: str = "hello back") -> MagicMock:
    """Return a mock assistant whose generate_reply coroutine returns *reply*."""
    assistant = MagicMock()
    assistant.generate_reply = AsyncMock(return_value=reply)
    return assistant


def _make_update(
    update_id: int,
    text: str,
    chat_id: str = "42",
) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id * 10,
            "chat": {"id": int(chat_id), "type": "private"},
            "text": text,
        },
    }


def _make_getUpdates_response(updates: list[dict]) -> MagicMock:
    payload = {"ok": True, "result": updates}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_send_response() -> MagicMock:
    payload = {"ok": True, "result": {"message_id": 99}}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Configuration / instantiation tests
# ---------------------------------------------------------------------------


def test_missing_token_raises():
    receiver = TelegramReceiver(bot_token=None, chat_id="42", assistant=_make_assistant())
    with pytest.raises(IntegrationNotConfiguredError, match="TELEGRAM_BOT_TOKEN"):
        receiver.start()


def test_missing_chat_id_raises():
    receiver = TelegramReceiver(bot_token="tok", chat_id=None, assistant=_make_assistant())
    with pytest.raises(IntegrationNotConfiguredError, match="telegram_chat_id"):
        receiver.start()


def test_missing_assistant_raises():
    receiver = TelegramReceiver(bot_token="tok", chat_id="42", assistant=None)
    with pytest.raises(IntegrationNotConfiguredError, match="assistant"):
        receiver.start()


def test_double_start_raises():
    assistant = _make_assistant()
    receiver = TelegramReceiver(bot_token="tok", chat_id="42", assistant=assistant)

    stop_called = threading.Event()

    def fake_poll_loop():
        stop_called.wait(timeout=5)

    with patch.object(receiver, "_poll_loop", side_effect=fake_poll_loop):
        receiver.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                receiver.start()
        finally:
            receiver._stop_event.set()
            stop_called.set()
            receiver.stop()


# ---------------------------------------------------------------------------
# _handle_update tests
# ---------------------------------------------------------------------------


def test_handle_update_routes_to_assistant_and_replies():
    assistant = _make_assistant("pong")
    receiver = TelegramReceiver(bot_token="tok", chat_id="42", assistant=assistant)

    with patch.object(receiver._client, "send_message") as mock_send:
        receiver._handle_update(_make_update(1, "ping", chat_id="42"))

    assistant.generate_reply.assert_awaited_once_with("ping")
    mock_send.assert_called_once_with("pong")


def test_handle_update_ignores_wrong_chat():
    assistant = _make_assistant("should not be called")
    receiver = TelegramReceiver(bot_token="tok", chat_id="42", assistant=assistant)

    with patch.object(receiver._client, "send_message") as mock_send:
        receiver._handle_update(_make_update(2, "hi", chat_id="99"))

    assistant.generate_reply.assert_not_awaited()
    mock_send.assert_not_called()


def test_handle_update_skips_empty_text():
    assistant = _make_assistant()
    receiver = TelegramReceiver(bot_token="tok", chat_id="42", assistant=assistant)
    update = {
        "update_id": 3,
        "message": {"chat": {"id": 42}, "text": ""},
    }
    with patch.object(receiver._client, "send_message") as mock_send:
        receiver._handle_update(update)

    assistant.generate_reply.assert_not_awaited()
    mock_send.assert_not_called()


def test_handle_update_advances_offset():
    assistant = _make_assistant()
    receiver = TelegramReceiver(bot_token="tok", chat_id="42", assistant=assistant)
    receiver._offset = 0

    with patch.object(receiver._client, "send_message"):
        receiver._handle_update(_make_update(7, "hello", chat_id="42"))

    assert receiver._offset == 8  # update_id + 1


def test_handle_update_sends_error_message_on_assistant_failure():
    assistant = MagicMock()
    assistant.generate_reply = AsyncMock(side_effect=RuntimeError("boom"))
    receiver = TelegramReceiver(bot_token="tok", chat_id="42", assistant=assistant)

    with patch.object(receiver._client, "send_message") as mock_send:
        receiver._handle_update(_make_update(5, "crash me", chat_id="42"))

    mock_send.assert_called_once()
    assert "error" in mock_send.call_args[0][0].lower()


def test_handle_update_tolerates_send_failure(caplog):
    assistant = _make_assistant("reply")
    receiver = TelegramReceiver(bot_token="tok", chat_id="42", assistant=assistant)

    with patch.object(
        receiver._client, "send_message", side_effect=RuntimeError("network down")
    ):
        # Should not raise
        receiver._handle_update(_make_update(6, "hello", chat_id="42"))


# ---------------------------------------------------------------------------
# Inbound message -> assistant -> outbound response flow (integration-style)
# ---------------------------------------------------------------------------


def test_full_inbound_to_outbound_flow():
    """Simulates one poll cycle returning a message and verifies the reply is sent."""
    assistant = _make_assistant("I can help with that!")
    receiver = TelegramReceiver(bot_token="tok", chat_id="42", assistant=assistant)

    updates = [_make_update(10, "What's the weather?", chat_id="42")]

    with (
        patch("urllib.request.urlopen", return_value=_make_getUpdates_response(updates)),
        patch.object(receiver._client, "send_message") as mock_send,
    ):
        result = receiver._get_updates()
        for upd in result:
            receiver._handle_update(upd)

    assistant.generate_reply.assert_awaited_once_with("What's the weather?")
    mock_send.assert_called_once_with("I can help with that!")


def test_unrecognized_command_gets_llm_response():
    """Unrecognized commands (no special handling) fall through to generate_reply."""
    assistant = _make_assistant("I'm not sure what you mean, but I'll try to help.")
    receiver = TelegramReceiver(bot_token="tok", chat_id="42", assistant=assistant)

    with patch.object(receiver._client, "send_message") as mock_send:
        receiver._handle_update(_make_update(11, "/unknowncommand", chat_id="42"))

    # generate_reply is called for ALL text, including unrecognized slash commands
    assistant.generate_reply.assert_awaited_once_with("/unknowncommand")
    mock_send.assert_called_once_with("I'm not sure what you mean, but I'll try to help.")


# ---------------------------------------------------------------------------
# is_running / stop tests
# ---------------------------------------------------------------------------


def test_is_running_false_before_start():
    receiver = TelegramReceiver(bot_token="tok", chat_id="42", assistant=_make_assistant())
    assert not receiver.is_running


def test_stop_is_safe_when_not_running():
    receiver = TelegramReceiver(bot_token="tok", chat_id="42", assistant=_make_assistant())
    receiver.stop()  # should not raise
