"""Factory for creating SMS backends from configuration.

Reads the ``messaging`` section of the runtime config and returns the
appropriate backend instance.  If the backend is ``"twilio"`` but
credentials are missing, a warning is logged and the stub backend is
returned as a safe fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rex.credentials import get_credential_manager
from rex.messaging_backends.account_config import MessagingConfig, load_messaging_config
from rex.messaging_backends.base import SmsBackend
from rex.messaging_backends.stub import StubSmsBackend

logger = logging.getLogger(__name__)


def create_sms_backend(
    raw_config: dict[str, Any] | None = None,
    *,
    account_id: str | None = None,
    fixture_path: Path | None = None,
) -> SmsBackend:
    """Create an SMS backend from configuration.

    Args:
        raw_config: Full runtime config dict (or None for stub defaults).
        account_id: Explicit account to use (overrides default).
        fixture_path: Override fixture path for stub backend (tests).

    Returns:
        A configured ``SmsBackend`` instance.
    """
    if raw_config is None:
        logger.info("No config provided; using stub SMS backend")
        return StubSmsBackend(fixture_path=fixture_path)

    config = load_messaging_config(raw_config)
    return _create_from_config(config, account_id=account_id, fixture_path=fixture_path)


def _create_from_config(
    config: MessagingConfig,
    *,
    account_id: str | None = None,
    fixture_path: Path | None = None,
) -> SmsBackend:
    """Internal: create backend from a validated config model."""
    if config.backend == "stub":
        from_number = "+15555551234"
        acct = config.get_account(account_id)
        if acct:
            from_number = acct.from_number
        return StubSmsBackend(fixture_path=fixture_path, default_from=from_number)

    if config.backend == "twilio":
        return _create_twilio(config, account_id=account_id, fixture_path=fixture_path)

    logger.warning("Unknown messaging backend '%s'; falling back to stub", config.backend)
    return StubSmsBackend(fixture_path=fixture_path)


def _create_twilio(
    config: MessagingConfig,
    *,
    account_id: str | None = None,
    fixture_path: Path | None = None,
) -> SmsBackend:
    """Attempt to create a Twilio backend; fall back to stub on failure."""
    acct = config.get_account(account_id)
    if acct is None:
        logger.warning("Twilio backend configured but no accounts defined; falling back to stub")
        return StubSmsBackend(fixture_path=fixture_path)

    try:
        from rex.messaging_backends.twilio_backend import create_twilio_backend_from_credentials

        cred_manager = get_credential_manager()
        return create_twilio_backend_from_credentials(
            credential_manager=cred_manager,
            credential_ref=acct.credential_ref,
            from_number=acct.from_number,
        )
    except ValueError as exc:
        logger.warning(
            "Failed to create Twilio backend for account '%s': %s. " "Falling back to stub.",
            acct.id,
            exc,
        )
        return StubSmsBackend(fixture_path=fixture_path, default_from=acct.from_number)
    except Exception as exc:
        logger.error(
            "Unexpected error creating Twilio backend: %s. Falling back to stub.",
            exc,
        )
        return StubSmsBackend(fixture_path=fixture_path)


__all__ = ["create_sms_backend"]
