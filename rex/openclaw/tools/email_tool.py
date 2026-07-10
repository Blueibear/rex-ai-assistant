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
    to: str | list[str] = "",
    subject: str = "",
    body: str = "",
    context: dict[str, Any] | None = None,
    account_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Send an email via Rex's EmailService as the dispatching user.

    Delegates to :func:`rex.email_service.get_email_service().send`.  The
    dispatcher-injected ``_user_id`` is required and validated **before** any
    backend or credential resolution; missing or invalid identity fails
    closed.  An optional ``account_id`` routes through a specific account
    only after ownership validation — a named user can never fall back to
    the global default or first configured account.

    .. note::
        This tool is policy-gated (MEDIUM risk).  Callers are responsible
        for obtaining policy approval before invoking this function.

    Args:
        to:         Recipient address or list of addresses.
        subject:    Email subject line.
        body:       Plain-text message body.
        context:    Optional ambient context dict (unused; reserved for
            future timezone / locale injection).
        account_id: Optional explicit account to send from (must belong to
            the requesting user).
        **kwargs:   Absorbs dispatcher-injected keys such as ``transcript``
            and ``_user_id`` without raising TypeError.

    Returns:
        A dict with keys ``ok`` (bool), ``message_id`` (str|None), and
        ``error`` (str|None).
    """
    from rex.email_accounts import require_user_id

    try:
        user_id = require_user_id(kwargs.get("_user_id"))
    except PermissionError as exc:
        logger.warning("send_email tool refused: %s", exc)
        return {
            "ok": False,
            "message_id": None,
            "error": "send_email requires a valid user identity",
        }

    service = _get_email_service()
    # Audit metadata: requesting user and selected account only — never
    # credentials or message bodies.
    logger.info("send_email tool: user=%s account=%s", user_id, account_id or "(default)")
    try:
        return service.send(
            to=to,
            subject=subject,
            body=body,
            account_id=account_id,
            user_id=user_id,
        )
    except PermissionError as exc:
        logger.warning("send_email tool refused for user=%s: %s", user_id, exc)
        return {"ok": False, "message_id": None, "error": str(exc)}
