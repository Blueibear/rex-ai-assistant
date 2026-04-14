"""Push notification support for Rex (US-042).

Supported providers:
- ntfy.sh  (push_provider = "ntfy")
- Pushover (push_provider = "pushover")

Usage::

    from rex.notifications.push import send_push
    send_push("Alert", "Your timer has finished", priority="high")
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from rex.assistant_errors import IntegrationNotConfiguredError

if TYPE_CHECKING:
    from rex.config import AppConfig

logger = logging.getLogger(__name__)

# ntfy.sh priority mapping: low / normal / high / urgent
_NTFY_PRIORITY: dict[str, str] = {
    "low": "2",
    "normal": "3",
    "high": "4",
    "urgent": "5",
}

# Pushover priority mapping
_PUSHOVER_PRIORITY: dict[str, int] = {
    "low": -1,
    "normal": 0,
    "high": 1,
}

_NTFY_DEFAULT_URL = "https://ntfy.sh"
_PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def send_push(
    title: str,
    message: str,
    priority: str = "normal",
    *,
    config: AppConfig | None = None,
) -> None:
    """Send a push notification via the configured provider.

    Args:
        title: Notification title.
        message: Notification body.
        priority: One of "low", "normal", "high", "urgent".
        config: AppConfig instance.  When *None* the global config is loaded.

    Raises:
        IntegrationNotConfiguredError: If push notifications are not configured.
        RuntimeError: If the HTTP request to the push provider fails.
    """
    if config is None:
        from rex.config import load_config

        config = load_config()

    provider = (config.push_provider or "").lower().strip()
    if not provider:
        raise IntegrationNotConfiguredError(
            "Push notifications are not configured. "
            "Set push_provider (ntfy or pushover) in rex_config.json."
        )

    if provider == "ntfy":
        _send_ntfy(title, message, priority, config)
    elif provider == "pushover":
        _send_pushover(title, message, priority, config)
    else:
        raise IntegrationNotConfiguredError(
            f"Unknown push provider '{provider}'. Supported: ntfy, pushover."
        )


def _send_ntfy(title: str, message: str, priority: str, config: AppConfig) -> None:
    topic = (config.push_topic or "").strip()
    if not topic:
        raise IntegrationNotConfiguredError(
            "ntfy push requires push_topic to be set in rex_config.json."
        )

    base_url = (_NTFY_DEFAULT_URL).rstrip("/")
    url = f"{base_url}/{topic}"
    ntfy_priority = _NTFY_PRIORITY.get(priority, _NTFY_PRIORITY["normal"])

    headers: dict[str, str] = {
        "Title": title,
        "Priority": ntfy_priority,
        "Content-Type": "text/plain; charset=utf-8",
    }
    token = (config.push_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = message.encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            status = resp.status
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ntfy push notification failed: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ntfy push notification failed: {exc.reason}") from exc

    logger.debug("ntfy push sent to topic '%s' (HTTP %s)", topic, status)


def _send_pushover(title: str, message: str, priority: str, config: AppConfig) -> None:
    token = (config.push_token or "").strip()
    topic = (config.push_topic or "").strip()  # topic = user key for Pushover
    if not token or not topic:
        raise IntegrationNotConfiguredError(
            "Pushover push requires push_token (app token) and "
            "push_topic (user key) in rex_config.json."
        )

    po_priority = _PUSHOVER_PRIORITY.get(priority, _PUSHOVER_PRIORITY["normal"])
    payload: dict[str, object] = {
        "token": token,
        "user": topic,
        "title": title,
        "message": message,
        "priority": po_priority,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _PUSHOVER_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            status = resp.status
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Pushover push notification failed: HTTP {exc.code} {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Pushover push notification failed: {exc.reason}") from exc

    logger.debug("Pushover push sent (HTTP %s)", status)
