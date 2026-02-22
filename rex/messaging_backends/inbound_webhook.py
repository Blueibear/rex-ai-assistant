"""Flask blueprint for the Twilio inbound SMS webhook.

Provides a ``POST /webhooks/twilio/sms`` endpoint that:
1. Validates the Twilio request signature (HMAC-SHA1, stdlib only).
2. Routes the inbound message to the correct messaging account by matching
   the ``To`` phone number against ``messaging.accounts[].from_number``.
3. Persists the message in the inbound SMS store.
4. Returns an empty TwiML ``<Response/>`` so Twilio does not retry.

Security:
- Signature verification is mandatory unless explicitly disabled (dev only).
- Secrets (auth token) are never logged.
- All external input is treated as untrusted; phone numbers and body are
  stored as-is but never interpolated into logs at INFO level.
- Rate limiting is applied when a Flask-Limiter instance is provided.

The blueprint is registered with the main Flask app at startup only when
``messaging.inbound.enabled`` is ``true`` in the runtime config.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, Response, request
from werkzeug.datastructures import MultiDict

from rex.messaging_backends.inbound_store import (
    InboundSmsRecord,
    InboundSmsStore,
)
from rex.messaging_backends.twilio_signature import validate_twilio_signature

logger = logging.getLogger(__name__)

# Content-Type for TwiML responses
_TWIML_CONTENT_TYPE = "text/xml"

# Empty TwiML response — tells Twilio not to take further action
_EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response/>'


def _build_account_phone_map(raw_config: dict[str, Any]) -> dict[str, str]:
    """Build a mapping of normalized phone number -> account ID.

    Args:
        raw_config: Full runtime config dict.

    Returns:
        Dict mapping ``from_number`` to ``account.id`` for each configured
        messaging account.
    """
    mapping: dict[str, str] = {}
    messaging = raw_config.get("messaging", {})
    if not isinstance(messaging, dict):
        return mapping
    for acct in messaging.get("accounts", []):
        if isinstance(acct, dict):
            number = acct.get("from_number", "")
            acct_id = acct.get("id", "")
            if number and acct_id:
                mapping[_normalize_phone(number)] = acct_id
    return mapping


def _normalize_phone(number: str) -> str:
    """Normalize a phone number by stripping whitespace.

    We do minimal normalization: strip leading/trailing whitespace.
    Twilio already sends E.164 format.
    """
    return number.strip()


def create_inbound_sms_blueprint(
    *,
    auth_token: str,
    inbound_store: InboundSmsStore,
    raw_config: dict[str, Any] | None = None,
    signature_verification: bool = True,
    limiter: Any | None = None,
    rate_limit_string: str = "120 per minute",
) -> Blueprint:
    """Create and configure the inbound SMS webhook blueprint.

    Each call creates a fresh ``Blueprint`` instance so that multiple Flask
    apps can be created (e.g. in tests) without hitting Flask's
    "already registered" guard.

    Args:
        auth_token: Twilio Auth Token for signature verification.
        inbound_store: The inbound SMS store instance.
        raw_config: Full runtime config dict for account routing.
        signature_verification: Whether to enforce Twilio signature
            verification.  Should only be ``False`` in local dev.
        limiter: Optional Flask-Limiter instance.  When provided the
            webhook route is decorated with the given *rate_limit_string*.
        rate_limit_string: Rate limit specification in Flask-Limiter format
            (default: ``"120 per minute"``).

    Returns:
        A configured Flask Blueprint ready to be registered.
    """
    bp = Blueprint(
        "inbound_sms",
        __name__,
        url_prefix="/webhooks/twilio",
    )
    phone_map = _build_account_phone_map(raw_config or {})

    # Build rate-limit decorator (no-op when limiter is not available)
    if limiter is not None and rate_limit_string:
        try:
            _rate_decorator = limiter.limit(rate_limit_string)
        except Exception:
            logger.warning(
                "Failed to create rate limit decorator; proceeding without rate limiting"
            )
            _rate_decorator = None
    else:
        _rate_decorator = None

    def receive_sms() -> Response:
        """Handle an inbound SMS from Twilio."""
        # --- Signature verification ---
        if signature_verification:
            twilio_sig = request.headers.get("X-Twilio-Signature", "")
            # Reconstruct the full URL that Twilio signed.
            # In production behind a reverse proxy, configure Werkzeug
            # ProxyFix/trusted proxy headers so request.url matches the
            # externally visible Twilio webhook URL exactly.
            url = request.url
            params: list[tuple[str, str]] = []
            form: MultiDict[str, str] = request.form
            for key in form.keys():
                for value in form.getlist(key):
                    params.append((key, value))

            if not validate_twilio_signature(auth_token, url, params, twilio_sig):
                logger.warning("Twilio signature verification failed")
                return Response("Forbidden", status=403, content_type="text/plain")

        # --- Extract message fields ---
        form = request.form
        message_sid = form.get("MessageSid", "")
        from_number = form.get("From", "")
        to_number = form.get("To", "")
        body = form.get("Body", "")

        # --- Route to account by To number ---
        normalized_to = _normalize_phone(to_number)
        account_id = phone_map.get(normalized_to)
        routed = account_id is not None

        if not routed:
            logger.warning(
                "Inbound SMS to unrecognized number; storing as unrouted "
                "(configure messaging.accounts with matching from_number)"
            )

        # --- Persist ---
        record = InboundSmsRecord(
            sid=message_sid,
            from_number=from_number,
            to_number=to_number,
            body=body,
            account_id=account_id,
            routed=routed,
        )

        try:
            inbound_store.write(record)
            logger.debug("Inbound SMS stored (id=%s, routed=%s)", record.id, routed)
        except Exception:
            logger.exception("Failed to persist inbound SMS")
            # Still return 200 so Twilio does not retry endlessly
            return Response(
                _EMPTY_TWIML,
                status=200,
                content_type=_TWIML_CONTENT_TYPE,
            )

        return Response(
            _EMPTY_TWIML,
            status=200,
            content_type=_TWIML_CONTENT_TYPE,
        )

    # Apply rate-limit decorator before registering the route so that
    # Flask-Limiter wraps the view function correctly.
    if _rate_decorator is not None:
        receive_sms = _rate_decorator(receive_sms)

    bp.add_url_rule("/sms", "receive_sms", receive_sms, methods=["POST"])

    return bp


__all__ = [
    "create_inbound_sms_blueprint",
]
