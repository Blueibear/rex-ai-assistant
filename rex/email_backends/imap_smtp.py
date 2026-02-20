"""Real email backend using stdlib imaplib (read) and smtplib (send).

Supports:
- IMAP4-SSL for reading (``imaplib.IMAP4_SSL``)
- SMTP with STARTTLS (``smtplib.SMTP`` + ``starttls()``)
- SMTPS (``smtplib.SMTP_SSL``)

All network operations use explicit timeouts and never log secrets.
"""

from __future__ import annotations

import email
import email.policy
import email.utils
import imaplib
import logging
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage

from rex.email_backends.base import EmailBackend, EmailEnvelope, SendResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30


class ImapSmtpEmailBackend(EmailBackend):
    """Production email backend using IMAP4-SSL (read) and SMTP (send).

    Args:
        imap_host:  IMAP server hostname.
        imap_port:  IMAP server port (default 993 for SSL).
        smtp_host:  SMTP server hostname.
        smtp_port:  SMTP server port (default 587 for STARTTLS).
        username:   Login username (typically the email address).
        password:   Login password / app-password.  **Never logged.**
        use_starttls: If True, use SMTP STARTTLS on *smtp_port*.
                      If False, use SMTPS (``SMTP_SSL``).
        timeout:    Socket timeout in seconds.
        ssl_context: Optional custom ``ssl.SSLContext``.
        imap_client_factory: Callable for DI / testing.
        smtp_client_factory: Callable for DI / testing.
    """

    def __init__(
        self,
        *,
        imap_host: str,
        imap_port: int = 993,
        smtp_host: str,
        smtp_port: int = 587,
        username: str,
        password: str,
        use_starttls: bool = True,
        timeout: int = _DEFAULT_TIMEOUT,
        ssl_context: ssl.SSLContext | None = None,
        imap_client_factory: object | None = None,
        smtp_client_factory: object | None = None,
    ) -> None:
        self._imap_host = imap_host
        self._imap_port = imap_port
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._username = username
        self._password = password
        self._use_starttls = use_starttls
        self._timeout = timeout
        self._ssl_ctx = ssl_context or ssl.create_default_context()

        self._imap_factory = imap_client_factory
        self._smtp_factory = smtp_client_factory

        self._imap: imaplib.IMAP4_SSL | None = None
        self._connected = False

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            if self._imap_factory is not None:
                self._imap = self._imap_factory()  # type: ignore[operator]
            else:
                self._imap = imaplib.IMAP4_SSL(
                    host=self._imap_host,
                    port=self._imap_port,
                    ssl_context=self._ssl_ctx,
                    timeout=self._timeout,
                )
            self._imap.login(self._username, self._password)
            self._imap.select("INBOX")
            self._connected = True
            logger.info(
                "IMAP connected to %s:%d as %s",
                self._imap_host,
                self._imap_port,
                self._username,
            )
            return True
        except imaplib.IMAP4.error as exc:
            logger.error("IMAP authentication failed for %s: %s", self._username, exc)
            self._connected = False
            return False
        except Exception as exc:
            logger.error(
                "IMAP connection failed (%s:%d): %s",
                self._imap_host,
                self._imap_port,
                exc,
            )
            self._connected = False
            return False

    def fetch_unread(self, limit: int = 10) -> list[EmailEnvelope]:
        if not self._connected or self._imap is None:
            logger.warning("IMAP not connected; call connect() first")
            return []

        try:
            status, data = self._imap.search(None, "UNSEEN")
            if status != "OK" or not data or not data[0]:
                return []

            msg_nums = data[0].split()
            msg_nums = msg_nums[-limit:] if limit else msg_nums

            envelopes: list[EmailEnvelope] = []
            for num in reversed(msg_nums):
                try:
                    env = self._fetch_envelope(num)
                    if env is not None:
                        envelopes.append(env)
                except Exception as exc:
                    logger.warning("Failed to fetch message %s: %s", num, exc)

            return envelopes
        except Exception as exc:
            logger.error("IMAP fetch_unread failed: %s", exc)
            return []

    def _fetch_envelope(self, msg_num: bytes) -> EmailEnvelope | None:
        assert self._imap is not None
        status, parts = self._imap.fetch(msg_num, "(RFC822.HEADER)")
        if status != "OK" or not parts or not parts[0]:
            return None

        header_data = parts[0]
        if isinstance(header_data, tuple) and len(header_data) >= 2:
            raw_headers = header_data[1]
        else:
            return None

        msg = email.message_from_bytes(raw_headers, policy=email.policy.default)
        from_addr = msg.get("From", "")
        subject = msg.get("Subject", "(no subject)")
        date_str = msg.get("Date", "")
        to_addrs_raw = msg.get("To", "")
        message_id = msg.get("Message-ID", str(msg_num))

        received_at = _parse_email_date(date_str) or datetime.now(timezone.utc)
        snippet = subject[:200] if subject else ""
        to_addrs = [addr.strip() for addr in to_addrs_raw.split(",") if addr.strip()]

        return EmailEnvelope(
            message_id=message_id,
            from_addr=from_addr,
            subject=subject,
            snippet=snippet,
            received_at=received_at,
            to_addrs=to_addrs,
            labels=["unread"],
        )

    def list_mailboxes(self) -> list[str]:
        if not self._connected or self._imap is None:
            return []
        try:
            status, data = self._imap.list()
            if status != "OK" or not data:
                return []
            boxes: list[str] = []
            for item in data:
                if isinstance(item, bytes):
                    decoded = item.decode("utf-8", errors="replace")
                    parts = decoded.rsplit('"', 2)
                    if len(parts) >= 2:
                        boxes.append(parts[-2])
                    else:
                        boxes.append(decoded)
            return boxes
        except Exception as exc:
            logger.error("IMAP list_mailboxes failed: %s", exc)
            return []

    def mark_as_read(self, message_id: str) -> bool:
        if not self._connected or self._imap is None:
            return False
        try:
            status, data = self._imap.search(None, f'(HEADER Message-ID "{message_id}")')
            if status != "OK" or not data or not data[0]:
                return False
            for num in data[0].split():
                self._imap.store(num, "+FLAGS", "\\Seen")
            return True
        except Exception as exc:
            logger.error("IMAP mark_as_read failed for %s: %s", message_id, exc)
            return False

    # ------------------------------------------------------------------
    # Send operations
    # ------------------------------------------------------------------

    def send(
        self,
        *,
        from_addr: str,
        to_addrs: list[str],
        subject: str,
        body: str,
        reply_to: str | None = None,
    ) -> SendResult:
        try:
            msg = EmailMessage()
            msg["From"] = from_addr
            msg["To"] = ", ".join(to_addrs)
            msg["Subject"] = subject
            if reply_to:
                msg["Reply-To"] = reply_to
            msg.set_content(body)

            smtp_conn = self._create_smtp_connection()
            try:
                smtp_conn.login(self._username, self._password)
                smtp_conn.send_message(msg)
                logger.info(
                    "Email sent via SMTP to %s (subject=%r)",
                    to_addrs,
                    subject,
                )
                return SendResult(ok=True, message_id=msg.get("Message-ID"))
            finally:
                try:
                    smtp_conn.quit()
                except Exception:
                    pass
        except smtplib.SMTPAuthenticationError as exc:
            logger.error("SMTP authentication failed for %s: %s", self._username, exc)
            return SendResult(ok=False, error="SMTP authentication failed")
        except smtplib.SMTPException as exc:
            logger.error("SMTP send failed: %s", exc)
            return SendResult(ok=False, error=f"SMTP error: {exc}")
        except Exception as exc:
            logger.error("Email send failed: %s", exc)
            return SendResult(ok=False, error=str(exc))

    def _create_smtp_connection(self) -> smtplib.SMTP:
        if self._smtp_factory is not None:
            return self._smtp_factory()  # type: ignore[operator]

        if self._use_starttls:
            conn = smtplib.SMTP(
                host=self._smtp_host,
                port=self._smtp_port,
                timeout=self._timeout,
            )
            conn.ehlo()
            conn.starttls(context=self._ssl_ctx)
            conn.ehlo()
        else:
            conn = smtplib.SMTP_SSL(
                host=self._smtp_host,
                port=self._smtp_port,
                context=self._ssl_ctx,
                timeout=self._timeout,
            )
        return conn

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        if self._imap is not None:
            try:
                self._imap.close()
                self._imap.logout()
            except Exception:
                pass
            self._imap = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


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


__all__ = ["ImapSmtpEmailBackend"]
