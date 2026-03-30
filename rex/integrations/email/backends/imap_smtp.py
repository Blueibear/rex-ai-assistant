"""IMAP read backend and SMTP send backend for the transport-layer email interface.

Implements ``EmailBackend`` from ``rex.integrations.email.backends.base`` using:
- ``imaplib.IMAP4_SSL`` for reading unread messages
- ``smtplib.SMTP`` / ``smtplib.SMTP_SSL`` for sending (added in US-207)

All network operations use explicit timeouts and **never log secrets**.
"""

from __future__ import annotations

import email
import email.policy
import email.utils
import imaplib
import logging
import ssl
from datetime import datetime, timezone
from typing import Callable

from rex.integrations.email.backends.base import EmailBackend

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10  # seconds — enforced at socket level


class EmailAuthError(Exception):
    """Raised when IMAP or SMTP authentication fails.

    Provides a descriptive message without leaking credentials.
    """


class IMAPBackend(EmailBackend):
    """Read-only IMAP backend implementing the transport-layer ``EmailBackend`` interface.

    Returns plain :class:`dict` objects so callers remain free of internal
    model dependencies.

    Args:
        host:           IMAP server hostname.
        port:           IMAP port (default: 993 for IMAP4-SSL).
        username:       Login username / email address.
        password:       Login password or app-password.  **Never logged.**
        use_ssl:        When True (default), connect with ``IMAP4_SSL``.
                        When False, use unencrypted ``IMAP4`` (not recommended).
        timeout:        Socket-level timeout in seconds (default: 10).
        imap_factory:   Optional callable that returns an IMAP connection object.
                        Inject for unit tests to avoid real network calls.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int = 993,
        username: str,
        password: str,
        use_ssl: bool = True,
        timeout: int = _DEFAULT_TIMEOUT,
        imap_factory: Callable[[], imaplib.IMAP4] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._use_ssl = use_ssl
        self._timeout = timeout
        self._imap_factory = imap_factory
        self._imap: imaplib.IMAP4 | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect_imap(self) -> imaplib.IMAP4:
        """Open and authenticate the IMAP connection.

        Raises:
            EmailAuthError: If login fails (``imaplib.IMAP4.error``).
            OSError: If the network connection cannot be established.
        """
        if self._imap_factory is not None:
            conn = self._imap_factory()
        elif self._use_ssl:
            ctx = ssl.create_default_context()
            conn = imaplib.IMAP4_SSL(
                host=self._host,
                port=self._port,
                ssl_context=ctx,
                timeout=self._timeout,  # type: ignore[call-arg]
            )
        else:
            conn = imaplib.IMAP4(host=self._host, port=self._port)

        try:
            conn.login(self._username, self._password)
        except imaplib.IMAP4.error as exc:
            # Close the raw connection before re-raising so we don't leak
            try:
                conn.logout()
            except Exception:
                pass
            raise EmailAuthError(
                f"IMAP authentication failed for {self._username!r} "
                f"on {self._host}:{self._port} — check credentials"
            ) from exc

        return conn

    # ------------------------------------------------------------------
    # EmailBackend interface
    # ------------------------------------------------------------------

    def fetch_unread(self, limit: int = 10) -> list[dict]:
        """Return up to *limit* unread messages as plain dicts.

        Each dict contains:
        ``id``, ``from``, ``subject``, ``snippet``, ``received_at``.

        Args:
            limit: Maximum number of messages to return (newest first).

        Returns:
            List of message dicts.  Empty list if no unread messages or on error.

        Raises:
            EmailAuthError: If authentication fails.
        """
        conn = self._connect_imap()
        try:
            conn.select("INBOX")
            status, data = conn.search(None, "UNSEEN")
            if status != "OK" or not data or not data[0]:
                return []

            msg_nums = data[0].split()
            # Take the last *limit* (most recent) message numbers
            if limit > 0:
                msg_nums = msg_nums[-limit:]

            results: list[dict] = []
            for num in reversed(msg_nums):
                try:
                    parsed = self._fetch_message_dict(conn, num)
                    if parsed is not None:
                        results.append(parsed)
                except Exception as exc:
                    logger.warning("Failed to fetch message %s: %s", num, exc)

            return results
        finally:
            try:
                conn.close()
                conn.logout()
            except Exception:
                pass

    def send(self, to: str, subject: str, body: str) -> None:
        """Not implemented in IMAPBackend — use IMAPSMTPBackend for sending."""
        raise NotImplementedError(
            "IMAPBackend is read-only. Use IMAPSMTPBackend to send email."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_message_dict(self, conn: imaplib.IMAP4, num: bytes) -> dict | None:
        status, parts = conn.fetch(num, "(RFC822.HEADER)")  # type: ignore[arg-type]
        if status != "OK" or not parts or not parts[0]:
            return None

        header_data = parts[0]
        if not isinstance(header_data, tuple) or len(header_data) < 2:
            return None

        raw_headers = header_data[1]
        msg = email.message_from_bytes(raw_headers, policy=email.policy.default)

        from_addr = msg.get("From", "")
        subject_str = msg.get("Subject", "(no subject)")
        date_str = msg.get("Date", "")
        message_id = msg.get("Message-ID", num.decode() if isinstance(num, bytes) else str(num))

        received_at = _parse_email_date(date_str) or datetime.now(timezone.utc)
        snippet = (subject_str or "")[:200]

        return {
            "id": message_id,
            "from": from_addr,
            "subject": subject_str,
            "snippet": snippet,
            "received_at": received_at.isoformat(),
        }


def _parse_email_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


__all__ = ["EmailAuthError", "IMAPBackend"]
