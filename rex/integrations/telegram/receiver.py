"""Telegram bot long-polling receiver for US-040.

Polls ``getUpdates`` in a background thread, routes each inbound text message
through ``Assistant.generate_reply()``, and sends the reply back via
``TelegramClient.send_message()``.

Usage::

    from rex.integrations.telegram.receiver import TelegramReceiver

    receiver = TelegramReceiver(bot_token="...", chat_id="...", assistant=my_assistant)
    receiver.start()   # non-blocking; runs in a daemon thread
    ...
    receiver.stop()    # signals the polling loop to exit
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, cast

from rex.assistant_errors import IntegrationNotConfiguredError
from rex.integrations.telegram.client import TelegramClient

logger = logging.getLogger(__name__)

_TELEGRAM_API_BASE = "https://api.telegram.org/bot"

# Default long-poll timeout (seconds).  Telegram allows up to 50 s.
_POLL_TIMEOUT = 30


class TelegramReceiver:
    """Long-polling receiver that forwards Telegram messages to the assistant.

    Parameters
    ----------
    bot_token:
        The Telegram Bot API token.  Required.
    chat_id:
        The authorised chat / user ID.  Only messages from this chat are
        processed (others are silently ignored).  Required.
    assistant:
        An object with an ``async generate_reply(text: str) -> str`` coroutine.
        Typically ``rex.assistant.Assistant``.
    poll_timeout:
        Long-poll timeout sent to ``getUpdates`` (seconds).
    http_timeout:
        urllib socket timeout for each ``getUpdates`` call (seconds).
        Must be greater than *poll_timeout* to avoid premature timeouts.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        assistant: Any = None,
        poll_timeout: int = _POLL_TIMEOUT,
        http_timeout: float = _POLL_TIMEOUT + 5,
    ) -> None:
        self._bot_token = (bot_token or "").strip()
        self._chat_id = str(chat_id).strip() if chat_id else ""
        self._assistant = assistant
        self._poll_timeout = poll_timeout
        self._http_timeout = http_timeout

        self._client = TelegramClient(
            bot_token=self._bot_token,
            chat_id=self._chat_id,
        )

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset: int = 0

    # ------------------------------------------------------------------
    # Configuration guard
    # ------------------------------------------------------------------

    def _check_configured(self) -> None:
        if not self._bot_token:
            raise IntegrationNotConfiguredError(
                "TelegramReceiver: not configured (set TELEGRAM_BOT_TOKEN in .env)"
            )
        if not self._chat_id:
            raise IntegrationNotConfiguredError(
                "TelegramReceiver: not configured (set telegram_chat_id in rex_config.json)"
            )
        if self._assistant is None:
            raise IntegrationNotConfiguredError(
                "TelegramReceiver: no assistant provided"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _api_url(self, method: str) -> str:
        return f"{_TELEGRAM_API_BASE}{self._bot_token}/{method}"

    def _get_updates(self) -> list[dict[str, Any]]:
        """Call ``getUpdates`` with long-polling; return list of update objects."""
        params = {
            "timeout": self._poll_timeout,
            "offset": self._offset,
            "allowed_updates": ["message"],
        }
        url = self._api_url("getUpdates") + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=self._http_timeout) as resp:  # noqa: S310
                data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
                return cast(list[dict[str, Any]], data.get("result", []))
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            logger.warning("TelegramReceiver getUpdates error: %s", exc)
            return []

    def _handle_update(self, update: dict[str, Any]) -> None:
        """Process one Telegram update dict."""
        update_id: int = update.get("update_id", 0)
        # Advance offset so this update is not re-delivered.
        self._offset = update_id + 1

        message = update.get("message") or {}
        chat = message.get("chat") or {}
        incoming_chat_id = str(chat.get("id", ""))
        text: str = (message.get("text") or "").strip()

        if not text:
            logger.debug("TelegramReceiver: skipping non-text update %d", update_id)
            return

        # Only process messages from the authorised chat.
        if incoming_chat_id != self._chat_id:
            logger.debug(
                "TelegramReceiver: ignoring message from unauthorised chat %s",
                incoming_chat_id,
            )
            return

        logger.info("TelegramReceiver: received message: %r", text)

        try:
            reply = asyncio.run(self._assistant.generate_reply(text))
        except Exception as exc:  # noqa: BLE001
            logger.error("TelegramReceiver: assistant error: %s", exc)
            reply = "Sorry, I encountered an error processing your request."

        try:
            self._client.send_message(reply)
        except Exception as exc:  # noqa: BLE001
            logger.error("TelegramReceiver: failed to send reply: %s", exc)

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        logger.info("TelegramReceiver: polling loop started")
        while not self._stop_event.is_set():
            updates = self._get_updates()
            for upd in updates:
                if self._stop_event.is_set():
                    break
                self._handle_update(upd)
        logger.info("TelegramReceiver: polling loop stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread.

        Raises
        ------
        IntegrationNotConfiguredError
            If ``bot_token``, ``chat_id``, or ``assistant`` is missing.
        RuntimeError
            If the receiver is already running.
        """
        self._check_configured()
        if self._thread and self._thread.is_alive():
            raise RuntimeError("TelegramReceiver is already running")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="telegram-receiver",
            daemon=True,
        )
        self._thread.start()
        logger.info("TelegramReceiver started")

    def stop(self) -> None:
        """Signal the polling loop to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._http_timeout + 2)
        logger.info("TelegramReceiver stopped")

    @property
    def is_running(self) -> bool:
        """True while the polling thread is alive."""
        return bool(self._thread and self._thread.is_alive())
