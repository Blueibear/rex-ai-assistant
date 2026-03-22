"""OpenClaw tool adapter — send_sms.

Wraps Rex's existing ``SMSService.send()`` from :mod:`rex.messaging_service`
and exposes it for registration with OpenClaw's tool system.

This is a *policy-gated* tool: in normal operation the policy engine
requires approval before send is attempted (MEDIUM risk).  The callable
itself does not enforce policy — that is the caller's responsibility.

When the ``openclaw`` package is not installed, :func:`register` logs a
warning and returns ``None``.  The :func:`send_sms` callable works
independently of OpenClaw.

Typical usage::

    from rex.openclaw.tools.sms_tool import send_sms, register

    result = send_sms("+15551234567", "Hello!")
    # {'ok': True, 'message_id': '...', 'error': None}

    register()   # no-op if openclaw not installed
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

from rex.messaging_service import Message, get_sms_service as _get_sms_service

logger = logging.getLogger(__name__)

OPENCLAW_AVAILABLE: bool = find_spec("openclaw") is not None

if OPENCLAW_AVAILABLE:  # pragma: no cover
    import openclaw as _openclaw  # type: ignore[import-not-found]
else:
    _openclaw = None  # type: ignore[assignment]

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
    """Send an SMS message via Rex's SMSService.

    Delegates to :func:`rex.messaging_service.get_sms_service().send`.  In
    stub mode (no Twilio backend configured) the message is saved to a
    local mock file.

    .. note::
        This tool is policy-gated (MEDIUM risk).  Callers are responsible
        for obtaining policy approval before invoking this function.

    Args:
        to:      Recipient phone number (E.164 format recommended).
        body:    Plain-text message body.
        context: Optional ambient context dict (unused; reserved for future
            locale injection).

    Returns:
        A dict with keys ``ok`` (bool), ``message_id`` (str), and
        ``channel`` (str).
    """
    service = _get_sms_service()
    # channel and from_ are required by the Message model; the SMSService will
    # override from_ with its configured from_number when from_ is empty.
    message = Message(channel="sms", to=to, body=body, **{"from": ""})
    sent = service.send(message)
    return {
        "ok": True,
        "message_id": sent.id,
        "channel": sent.channel or "sms",
    }


def register(agent: Any = None) -> Any:
    """Register the ``send_sms`` tool with OpenClaw.

    When the ``openclaw`` package is available this function calls
    OpenClaw's tool registration API, passing :func:`send_sms` as the
    handler.  When OpenClaw is not installed it logs a warning and returns
    ``None``.

    .. note::
        The exact OpenClaw tool registration call is a stub (see PRD §8.3 —
        *"Confirm OpenClaw's tool registration mechanism"*).  Replace the
        ``# TODO`` line once the API is confirmed.

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
