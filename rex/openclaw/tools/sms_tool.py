"""OpenClaw tool adapter — send_sms (stub).

The legacy ``rex.messaging_service`` backend has been retired.
SMS delivery routes through OpenClaw's messaging channel once that
integration is complete.  Until then, :func:`send_sms` logs a warning
and returns an error dict so callers degrade gracefully.

When the ``openclaw`` package is not installed, :func:`register` logs a
warning and returns ``None``.

Typical usage::

    from rex.openclaw.tools.sms_tool import send_sms, register

    result = send_sms("+15551234567", "Hello!")
    # {'ok': False, 'message_id': None, 'error': 'SMS backend not available ...'}

    register()   # no-op if openclaw not installed
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Tool name used when registering with OpenClaw.
TOOL_NAME = "send_sms"

#: Human-readable description forwarded to OpenClaw's tool registry.
TOOL_DESCRIPTION = (
    "Send an SMS text message to a phone number. "
    'Args: {"to": "+15551234567", "body": "Message text"}'
)


def send_sms(
    to: str,
    body: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stub: SMS delivery via OpenClaw (legacy SMS backend retired).

    The ``rex.messaging_service`` backend has been retired.  SMS delivery
    will route through OpenClaw's messaging channel once that integration
    is complete.

    Args:
        to:      Recipient phone number (E.164 format recommended).
        body:    Plain-text message body.
        context: Optional ambient context dict (reserved for future use).

    Returns:
        A dict with keys ``ok`` (bool), ``message_id`` (str | None), and
        ``error`` (str | None).
    """
    logger.warning(
        "send_sms: SMS backend not available (migrating to OpenClaw messaging backend). "
        "to=%s body_length=%d",
        to,
        len(body),
    )
    return {
        "ok": False,
        "message_id": None,
        "error": "SMS backend not available (migrating to OpenClaw messaging backend)",
    }
