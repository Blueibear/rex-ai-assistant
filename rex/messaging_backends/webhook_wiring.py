"""Startup wiring for the Twilio inbound SMS webhook.

Provides ``register_inbound_sms_webhook`` which reads the runtime config,
resolves credentials, initializes the inbound store, and conditionally
registers the webhook blueprint on a Flask app.

Usage (in a Flask app entry point such as ``flask_proxy.py``)::

    from rex.messaging_backends.webhook_wiring import register_inbound_sms_webhook

    register_inbound_sms_webhook(app)

The function is safe to call even when inbound is disabled — it reads
``messaging.inbound.enabled`` from the config and returns early if
``false`` (the default).

Rate limiting is applied via Flask-Limiter when the library is installed.
The limit string is configurable via ``messaging.inbound.rate_limit``
(default: ``"120 per minute"``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

from flask import Flask

from rex.config_manager import load_config
from rex.credentials import CredentialManager, get_credential_manager
from rex.messaging_backends.inbound_store import (
    InboundStoreConfig,
    init_inbound_store,
    load_inbound_store_config,
)
from rex.messaging_backends.inbound_webhook import create_inbound_sms_blueprint

logger = logging.getLogger(__name__)


def _create_limiter(app: Flask) -> Any | None:
    """Create a Flask-Limiter instance for the app.

    Returns ``None`` if flask_limiter is not installed or setup fails.
    Uses the same storage URI resolution order as ``rex_speak_api.py``.
    """
    try:
        from flask_limiter import Limiter
    except ImportError:
        logger.debug("flask_limiter not installed; webhook rate limiting disabled")
        return None

    storage_uri = (
        os.getenv("REX_SPEAK_STORAGE_URI") or os.getenv("FLASK_LIMITER_STORAGE_URI") or "memory://"
    )

    try:
        limiter = Limiter(
            app=app,
            key_func=_webhook_rate_key,
            storage_uri=storage_uri,
            default_limits=[],
        )
    except Exception:
        logger.warning(
            "Failed to initialize Flask-Limiter for webhook; rate limiting disabled",
            exc_info=True,
        )
        return None

    return limiter


def _webhook_rate_key() -> str:
    """Rate-limit key function for the inbound webhook.

    Uses the remote address (or X-Forwarded-For behind a trusted proxy).
    """
    from flask import request

    trusted_proxies = {
        ip.strip()
        for ip in os.getenv("REX_TRUSTED_PROXIES", "127.0.0.1,::1").split(",")
        if ip.strip()
    }
    remote_addr = request.remote_addr or "unknown"
    if remote_addr in trusted_proxies:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return cast(str, forwarded.split(",")[-1].strip())
    return remote_addr


def register_inbound_sms_webhook(
    app: Flask,
    raw_config: dict[str, Any] | None = None,
    credential_manager: CredentialManager | None = None,
) -> bool:
    """Register the Twilio inbound SMS webhook blueprint on *app*.

    This function is the single entry point for wiring inbound SMS into
    a running Flask server.  It is safe to call unconditionally at startup;
    when ``messaging.inbound.enabled`` is ``false`` (the default) the
    function returns immediately without registering any routes.

    Args:
        app: The Flask application instance.
        raw_config: Full runtime config dict.  If ``None``, the config is
            loaded via ``rex.config_manager.load_config()``.
        credential_manager: Credential manager instance.  If ``None``,
            the global singleton from ``rex.credentials`` is used.

    Returns:
        ``True`` if the webhook blueprint was registered, ``False`` otherwise.
    """
    if raw_config is None:
        try:
            raw_config = load_config()
        except Exception:
            logger.warning("Failed to load config; inbound webhook not registered")
            return False

    # Parse inbound config
    inbound_cfg: InboundStoreConfig = load_inbound_store_config(raw_config)

    if not inbound_cfg.enabled:
        logger.debug("Inbound SMS webhook is disabled in config")
        return False

    # Resolve Twilio auth token
    if credential_manager is None:
        credential_manager = get_credential_manager()

    auth_token = credential_manager.get_token(inbound_cfg.auth_token_ref)
    if not auth_token:
        logger.warning(
            "Inbound SMS webhook enabled but auth token not found "
            "(credential ref: %s); webhook not registered",
            inbound_cfg.auth_token_ref,
        )
        return False

    # Initialize the inbound store
    store = init_inbound_store(raw_config)
    if store is None:
        logger.warning("Failed to initialize inbound SMS store; webhook not registered")
        return False

    # Create rate limiter (optional — graceful fallback if unavailable)
    limiter = _create_limiter(app)
    rate_limit_string = inbound_cfg.rate_limit

    # Create and register the blueprint
    bp = create_inbound_sms_blueprint(
        auth_token=auth_token,
        inbound_store=store,
        raw_config=raw_config,
        signature_verification=True,
        limiter=limiter,
        rate_limit_string=rate_limit_string,
    )
    app.register_blueprint(bp)
    logger.info(
        "Inbound SMS webhook registered at /webhooks/twilio/sms " "(rate_limit=%s)",
        rate_limit_string,
    )
    return True


__all__ = [
    "register_inbound_sms_webhook",
]
