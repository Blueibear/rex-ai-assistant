"""Telegram bot client for sending messages via the Bot API.

If ``bot_token`` or ``chat_id`` is falsy the client starts in
"not configured" mode and all methods raise
:exc:`rex.assistant_errors.IntegrationNotConfiguredError`.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, cast

from rex.assistant_errors import IntegrationNotConfiguredError

logger = logging.getLogger(__name__)

_TELEGRAM_API_BASE = "https://api.telegram.org/bot"


class TelegramClient:
    """Minimal Telegram Bot API client.

    Parameters
    ----------
    bot_token:
        The Telegram Bot API token (from ``TELEGRAM_BOT_TOKEN`` in ``.env``).
    chat_id:
        The target chat / user ID (from ``telegram_chat_id`` in
        ``rex_config.json``).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._bot_token = (bot_token or "").strip()
        self._chat_id = str(chat_id).strip() if chat_id else ""
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_configured(self) -> None:
        """Raise :exc:`IntegrationNotConfiguredError` if not fully configured."""
        if not self._bot_token:
            raise IntegrationNotConfiguredError(
                "Telegram: not configured (set TELEGRAM_BOT_TOKEN in .env)"
            )
        if not self._chat_id:
            raise IntegrationNotConfiguredError(
                "Telegram: not configured (set telegram_chat_id in rex_config.json)"
            )

    def _api_url(self, method: str) -> str:
        return f"{_TELEGRAM_API_BASE}{self._bot_token}/{method}"

    def _post(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._api_url(method)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                return cast(dict[str, Any], json.loads(resp.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.error("Telegram API HTTP error %s: %s", exc.code, body)
            raise
        except urllib.error.URLError as exc:
            logger.error("Telegram API network error: %s", exc.reason)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_message(self, text: str) -> dict[str, Any]:
        """Send *text* to the configured chat.

        Parameters
        ----------
        text:
            Message body.  Must be non-empty.

        Returns
        -------
        dict
            The raw Telegram API response (``{"ok": True, "result": {...}}``).

        Raises
        ------
        IntegrationNotConfiguredError
            If ``bot_token`` or ``chat_id`` is not set.
        ValueError
            If *text* is empty.
        """
        self._check_configured()
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
        }
        logger.debug("Sending Telegram message to chat %s", self._chat_id)
        response = self._post("sendMessage", payload)
        logger.info("Telegram message sent (ok=%s)", response.get("ok"))
        return response
