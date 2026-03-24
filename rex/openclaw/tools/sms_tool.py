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
from importlib.util import find_spec
from typing import Any

logger = logging.getLogger(__name__)

OPENCLAW_AVAILABLE: bool = find_spec("openclaw") is not None

if OPENCLAW_AVAILABLE:  # pragma: no cover
    import openclaw as _openclaw
else:
    _openclaw = None
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


def register(agent: Any = None) -> Any:
    """Register the ``send_sms`` tool with OpenClaw.

    When the ``openclaw`` package is available this function calls
    OpenClaw's tool registration API, passing :func:`send_sms` as the
    handler.  When OpenClaw is not installed it logs a warning and returns
    ``None``.

    Args:
        agent: Optional OpenClaw agent handle.

    Returns:
        The registration handle returned by OpenClaw, or ``None``.
    """
    if not OPENCLAW_AVAILABLE:
        logger.warning(
            "openclaw package not installed — %s tool not registered with OpenClaw",
            TOOL_NAME,
        )
        return None

    # TODO: replace with real OpenClaw tool registration once API is confirmed.
    # Expected shape (to be verified):
    #   handle = _openclaw.register_tool(
    #       name=TOOL_NAME,
    #       description=TOOL_DESCRIPTION,
    #       handler=send_sms,
    #       agent=agent,
    #   )
    #   return handle
    logger.warning(
        "OpenClaw tool registration stub for %s — update once API is confirmed (PRD §8.3)",
        TOOL_NAME,
    )
    return None
