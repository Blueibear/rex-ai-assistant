"""OpenClaw tool adapter — send_email.

Wraps Rex's existing ``EmailService.send()`` from :mod:`rex.email_service`
and exposes it for registration with OpenClaw's tool system.

This is a *policy-gated* tool: in normal operation the policy engine
requires approval before send is attempted (MEDIUM risk).  The callable
itself does not enforce policy — that is the caller's responsibility.

When the ``openclaw`` package is not installed, :func:`register` logs a
warning and returns ``None``.  The :func:`send_email` callable works
independently of OpenClaw.

Typical usage::

    from rex.openclaw.tools.email_tool import send_email, register

    result = send_email("bob@example.com", "Hello", "Hi Bob!")
    # {'ok': True, 'message_id': '...', 'error': None}

    register()   # no-op if openclaw not installed
"""

from __future__ import annotations

import logging
from typing import Any

from rex.email_service import get_email_service as _get_email_service

logger = logging.getLogger(__name__)

#: Tool name used when registering with OpenClaw.
TOOL_NAME = "send_email"

#: Human-readable description forwarded to OpenClaw's tool registry.
TOOL_DESCRIPTION = (
    "Send an email to one or more recipients. "
    'Args: {"to": "recipient@example.com", "subject": "Subject line", "body": "Plain-text body"}'
)


def send_email(
    to: str | list[str],
    subject: str,
    body: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send an email via Rex's EmailService.

    Delegates to :func:`rex.email_service.get_email_service().send`.  In
    stub mode (no backend configured) the email is logged and a success
    response is returned without actually sending.

    .. note::
        This tool is policy-gated (MEDIUM risk).  Callers are responsible
        for obtaining policy approval before invoking this function.

    Args:
        to:      Recipient address or list of addresses.
        subject: Email subject line.
        body:    Plain-text message body.
        context: Optional ambient context dict (unused; reserved for future
            timezone / locale injection).

    Returns:
        A dict with keys ``ok`` (bool), ``message_id`` (str|None), and
        ``error`` (str|None).
    """
    service = _get_email_service()
    return service.send(to=to, subject=subject, body=body)


def register(agent: Any = None) -> Any:
    """Register the ``send_email`` tool with OpenClaw.

    When the ``openclaw`` package is available this function calls
    OpenClaw's tool registration API, passing :func:`send_email` as the
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
    from rex.config import load_config as _load_config
    from rex.openclaw.http_client import get_openclaw_client

    if get_openclaw_client(_load_config()) is None:
        logger.warning(
            "OpenClaw gateway not configured — %s tool not registered with OpenClaw",
            TOOL_NAME,
        )
        return None

    # TODO: replace with real OpenClaw tool registration once API is confirmed.
    # Expected shape (to be verified):
    #   handle = _openclaw.register_tool(
    #       name=TOOL_NAME,
    #       description=TOOL_DESCRIPTION,
    #       handler=send_email,
    #       agent=agent,
    #   )
    #   return handle
    logger.warning(
        "OpenClaw tool registration stub for %s — update once API is confirmed (PRD §8.3)",
        TOOL_NAME,
    )
    return None
