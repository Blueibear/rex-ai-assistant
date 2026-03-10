"""Account router: selects the right email backend for a given account.

Routing precedence:
1. Explicit ``account_id`` argument.
2. ``default_account_id`` from ``EmailConfig``.
3. First account in the config list (deterministic fallback).
4. If no accounts are configured -> return the stub backend.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rex.email_backends.account_config import EmailAccountConfig, EmailConfig
from rex.email_backends.base import EmailBackend
from rex.email_backends.stub import StubEmailBackend

logger = logging.getLogger(__name__)


def resolve_backend(
    email_config: EmailConfig,
    account_id: str | None = None,
    credential_getter: object | None = None,
    stub_fixture: Path | None = None,
) -> tuple[EmailBackend, EmailAccountConfig | None]:
    """Return an ``EmailBackend`` for the requested account.

    Args:
        email_config:       Parsed multi-account email config.
        account_id:         Explicit account to use (highest priority).
        credential_getter:  A callable ``(credential_ref: str) -> str|None``
                            that returns the raw credential token.
        stub_fixture:       Path to JSON fixture for the stub backend.

    Returns:
        A 2-tuple of ``(backend_instance, account_config_or_None)``.
    """
    account = email_config.get_account(account_id)

    if account is None:
        logger.info("No email accounts configured; using stub backend")
        return StubEmailBackend(fixture_path=stub_fixture), None

    cred_token = _get_credential(account.credential_ref, credential_getter)
    if cred_token is None:
        logger.warning(
            "No credential found for ref '%s' (account '%s'); " "falling back to stub backend",
            account.credential_ref,
            account.id,
        )
        return StubEmailBackend(fixture_path=stub_fixture), None

    username, password = _parse_credential_token(cred_token, account)
    if not username or not password:
        logger.warning(
            "Credential for '%s' could not be parsed into username:password; "
            "falling back to stub backend",
            account.credential_ref,
        )
        return StubEmailBackend(fixture_path=stub_fixture), None

    from rex.email_backends.imap_smtp import ImapSmtpEmailBackend

    backend = ImapSmtpEmailBackend(
        imap_host=account.imap.host,
        imap_port=account.imap.port,
        smtp_host=account.smtp.host,
        smtp_port=account.smtp.port,
        username=username,
        password=password,
        use_starttls=account.smtp.starttls,
    )
    return backend, account


def _get_credential(
    credential_ref: str,
    credential_getter: object | None,
) -> str | None:
    if credential_getter is None:
        return None
    try:
        return credential_getter(credential_ref)  # type: ignore[no-any-return, operator]
    except Exception as exc:
        logger.debug("credential_getter(%r) raised: %s", credential_ref, exc)
        return None


def _parse_credential_token(
    token: str,
    account: EmailAccountConfig,
) -> tuple[str | None, str | None]:
    """Split a credential token into (username, password).

    Accepted formats:
    - ``username:password``  (first colon is the separator)
    - Plain password only -> use ``account.address`` as username.
    """
    if ":" in token:
        username, _, password = token.partition(":")
        return username.strip(), password.strip()
    return account.address, token.strip()


__all__ = ["resolve_backend"]
