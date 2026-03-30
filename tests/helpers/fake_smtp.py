"""Fake SMTP transport for offline integration tests (US-212).

Provides :class:`FakeSMTP` — a configurable, in-memory replacement for both
``smtplib.SMTP`` and ``smtplib.SMTP_SSL`` that records every call and returns
pre-programmed responses.  No network connection is made.

Usage::

    from tests.helpers.fake_smtp import FakeSMTP

    fake = FakeSMTP()
    backend = IMAPSMTPBackend(
        host="imap.example.com",
        username="user@example.com",
        password="pw",
        smtp_host="smtp.example.com",
        smtp_factory=lambda: fake,
    )
    backend.send(to="bob@example.com", subject="Hi", body="Hello")
"""

from __future__ import annotations

import smtplib
from email.message import Message
from typing import Any


class FakeSMTP:
    """In-memory fake for ``smtplib.SMTP`` / ``smtplib.SMTP_SSL``.

    Args:
        ehlo_result:    Return value of ``ehlo()``.
        starttls_result: Return value of ``starttls()``.
        login_raises:   If not ``None``, ``login()`` raises this exception.
        send_raises:    If not ``None``, ``send_message()`` raises this.
        quit_result:    Return value of ``quit()``.
    """

    def __init__(
        self,
        *,
        ehlo_result: tuple[int, bytes] = (250, b"OK"),
        starttls_result: tuple[int, bytes] = (220, b"Ready"),
        login_raises: Exception | None = None,
        send_raises: Exception | None = None,
        quit_result: tuple[int, bytes] = (221, b"Bye"),
    ) -> None:
        self._ehlo_result = ehlo_result
        self._starttls_result = starttls_result
        self._login_raises = login_raises
        self._send_raises = send_raises
        self._quit_result = quit_result

        # Call-recording attributes (used for assertions in tests)
        self.ehlo_calls: int = 0
        self.starttls_calls: int = 0
        self.login_calls: list[tuple[str, str]] = []
        self.send_message_calls: list[Message] = []
        self.quit_calls: int = 0

    # ------------------------------------------------------------------
    # smtplib.SMTP interface
    # ------------------------------------------------------------------

    def ehlo(self, name: str = "") -> tuple[int, bytes]:  # noqa: ARG002
        self.ehlo_calls += 1
        return self._ehlo_result

    def starttls(self, **kwargs: Any) -> tuple[int, bytes]:  # noqa: ARG002
        self.starttls_calls += 1
        return self._starttls_result

    def login(self, user: str, password: str) -> tuple[int, bytes]:
        self.login_calls.append((user, password))
        if self._login_raises is not None:
            raise self._login_raises
        return (235, b"Authentication successful")

    def send_message(
        self,
        msg: Message,
        from_addr: str | None = None,  # noqa: ARG002
        to_addrs: list[str] | None = None,  # noqa: ARG002
    ) -> dict:
        self.send_message_calls.append(msg)
        if self._send_raises is not None:
            raise self._send_raises
        return {}

    def quit(self) -> tuple[int, bytes]:
        self.quit_calls += 1
        return self._quit_result

    # Make smtplib exceptions available for isinstance/raise compatibility
    SMTPException = smtplib.SMTPException
    SMTPAuthenticationError = smtplib.SMTPAuthenticationError


__all__ = ["FakeSMTP"]
