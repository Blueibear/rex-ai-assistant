"""Multi-account EmailService for the transport-layer email interface (US-208).

Routes ``fetch_unread()`` and ``send()`` calls to the correct
:class:`~rex.integrations.email.backends.base.EmailBackend` instance based on
the requested ``account_id``, falling back to ``default_account_id`` when
``account_id`` is omitted.

Backends are constructed lazily on first use via a configurable factory so that
the service can be tested without live network connections.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from rex.config import EmailAccountConfig
from rex.integrations.email.backends.base import EmailBackend
from rex.integrations.email.backends.imap_smtp import IMAPSMTPBackend

logger = logging.getLogger(__name__)


def _default_backend_factory(account: EmailAccountConfig) -> EmailBackend:
    """Build a real :class:`IMAPSMTPBackend` from an :class:`EmailAccountConfig`."""
    return IMAPSMTPBackend(
        host=account.imap_host,
        port=account.imap_port,
        username=account.address,
        password="",  # resolved at send-time via credential_ref
        smtp_host=account.smtp_host,
        smtp_port=account.smtp_port,
        use_starttls=account.use_starttls,
        credential_ref=account.credential_ref or None,
    )


class EmailService:
    """Multi-account email service routing read/send operations to the correct backend.

    Args:
        accounts:           List of :class:`~rex.config.EmailAccountConfig` objects.
        default_account_id: Account ID used when ``account_id`` is not supplied to
                            :meth:`fetch_unread` or :meth:`send`.  If empty, defaults
                            to the first account in *accounts* (if any).
        backend_factory:    Callable ``(EmailAccountConfig) -> EmailBackend`` used to
                            construct backends on first use.  Inject for unit tests to
                            avoid real network calls.
    """

    def __init__(
        self,
        accounts: list[EmailAccountConfig],
        default_account_id: str = "",
        backend_factory: Callable[[EmailAccountConfig], EmailBackend] | None = None,
    ) -> None:
        self._accounts: dict[str, EmailAccountConfig] = {a.id: a for a in accounts}
        self._default_id: str = default_account_id or (accounts[0].id if accounts else "")
        self._factory: Callable[[EmailAccountConfig], EmailBackend] = (
            backend_factory or _default_backend_factory
        )
        self._backends: dict[str, EmailBackend] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_account_id(self, account_id: str | None) -> str:
        """Return the effective account id, falling back to the default."""
        return account_id if account_id is not None else self._default_id

    def _get_backend(self, account_id: str) -> EmailBackend:
        """Return the (lazily constructed) backend for *account_id*.

        Raises:
            ValueError: If *account_id* is not in the accounts list.
        """
        if account_id not in self._accounts:
            known = ", ".join(sorted(self._accounts)) or "(none configured)"
            raise ValueError(
                f"Unknown email account_id {account_id!r}. " f"Known accounts: {known}"
            )
        if account_id not in self._backends:
            self._backends[account_id] = self._factory(self._accounts[account_id])
        return self._backends[account_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_unread(self, limit: int = 10, *, account_id: str | None = None) -> list[dict]:
        """Return up to *limit* unread messages for the specified account.

        Args:
            limit:      Maximum number of messages to return.
            account_id: Account to query.  Falls back to ``default_account_id``.

        Returns:
            List of message dicts (``id``, ``from``, ``subject``, ``snippet``,
            ``received_at``).  Empty list when no messages or not configured.

        Raises:
            ValueError: If *account_id* is provided but does not match any account.
        """
        effective_id = self._resolve_account_id(account_id)
        if not effective_id:
            logger.warning("EmailService.fetch_unread called but no accounts are configured")
            return []
        backend = self._get_backend(effective_id)
        return backend.fetch_unread(limit=limit)

    def send(
        self,
        to: str,
        subject: str,
        body: str,
        *,
        account_id: str | None = None,
    ) -> None:
        """Send an email via the specified account's SMTP backend.

        Args:
            to:         Recipient address.
            subject:    Subject line.
            body:       Plain-text message body.
            account_id: Account to send from.  Falls back to ``default_account_id``.

        Raises:
            ValueError: If *account_id* is provided but does not match any account.
        """
        effective_id = self._resolve_account_id(account_id)
        if not effective_id:
            raise ValueError(
                "EmailService.send called but no accounts are configured and no "
                "account_id was provided."
            )
        backend = self._get_backend(effective_id)
        backend.send(to=to, subject=subject, body=body)

    @property
    def account_ids(self) -> list[str]:
        """Return a sorted list of configured account IDs."""
        return sorted(self._accounts)


__all__ = ["EmailService"]
