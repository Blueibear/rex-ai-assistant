"""Fake IMAP4_SSL transport for offline integration tests (US-212).

Provides :class:`FakeIMAP4SSL` — a configurable, in-memory replacement for
``imaplib.IMAP4_SSL`` that records every call and returns pre-programmed
responses.  No network connection is made.

Usage::

    from tests.helpers.fake_imap import FakeIMAP4SSL

    fake = FakeIMAP4SSL(search_result=(\"OK\", [b\"1 2\"]))
    backend = IMAPBackend(
        host=\"imap.example.com\",
        username=\"user@example.com\",
        password=\"pw\",
        imap_factory=lambda: fake,
    )
    messages = backend.fetch_unread()
"""

from __future__ import annotations

import builtins
import imaplib
from typing import Any, TypeAlias

IMAPResponse: TypeAlias = tuple[str, builtins.list[object]]
IMAPFetchPart: TypeAlias = tuple[bytes, bytes]
IMAPFetchResponse: TypeAlias = tuple[str, builtins.list[IMAPFetchPart]]

# Minimal raw RFC-822 headers used as the default fetch payload.
DEFAULT_RAW_HEADERS: bytes = (
    b"From: Alice <alice@example.com>\r\n"
    b"To: Bob <bob@example.com>\r\n"
    b"Subject: Hello from Alice\r\n"
    b"Date: Mon, 30 Mar 2026 10:00:00 +0000\r\n"
    b"Message-ID: <test-msg-001@example.com>\r\n"
    b"\r\n"
)


class FakeIMAP4SSL:
    """In-memory fake for ``imaplib.IMAP4_SSL``.

    Args:
        search_result:  Return value of ``search()``.  Defaults to a single
                        message number ``b"1"``.
        fetch_result:   Return value of ``fetch()``.  Defaults to a tuple
                        containing :data:`DEFAULT_RAW_HEADERS`.
        login_raises:   If not ``None``, ``login()`` raises this exception.
        select_result:  Return value of ``select()``.
        close_result:   Return value of ``close()``.
        logout_result:  Return value of ``logout()``.
    """

    def __init__(
        self,
        *,
        search_result: IMAPResponse = ("OK", [b"1"]),
        fetch_result: IMAPFetchResponse | None = None,
        login_raises: Exception | None = None,
        select_result: IMAPResponse = ("OK", [b"1"]),
        list_result: IMAPResponse = ("OK", [b'(\\HasNoChildren) "/" "INBOX"']),
        close_result: IMAPResponse = ("OK", [b"Closed"]),
        logout_result: IMAPResponse = ("BYE", [b"Logging out"]),
    ) -> None:
        self._search_result = search_result
        self._fetch_result = fetch_result or (
            "OK",
            [(b"1 (RFC822.HEADER {350})", DEFAULT_RAW_HEADERS)],
        )
        self._login_raises = login_raises
        self._select_result = select_result
        self._list_result = list_result
        self._close_result = close_result
        self._logout_result = logout_result

        # Call-recording attributes (used for assertions in tests)
        self.login_calls: list[tuple[str, str]] = []
        self.select_calls: list[str] = []
        self.list_calls: list[tuple[str, str]] = []
        self.search_calls: list[tuple] = []
        self.fetch_calls: list[tuple] = []
        self.store_calls: list[tuple] = []
        self.close_calls: int = 0
        self.logout_calls: int = 0

    # ------------------------------------------------------------------
    # imaplib.IMAP4 interface
    # ------------------------------------------------------------------

    def login(self, user: str, password: str) -> IMAPResponse:
        self.login_calls.append((user, password))
        if self._login_raises is not None:
            raise self._login_raises
        return ("OK", [b"Logged in"])

    def select(self, mailbox: str = "INBOX", readonly: bool = False) -> IMAPResponse:
        self.select_calls.append(mailbox)
        return self._select_result

    def list(self, directory: str = "", pattern: str = "*") -> IMAPResponse:
        self.list_calls.append((directory, pattern))
        return self._list_result

    def search(self, charset: Any, *criteria: str) -> IMAPResponse:
        self.search_calls.append((charset, *criteria))
        return self._search_result

    def fetch(self, message_set: Any, message_parts: str) -> IMAPFetchResponse:
        self.fetch_calls.append((message_set, message_parts))
        return self._fetch_result

    def store(self, message_set: Any, command: str, flags: str) -> IMAPResponse:
        self.store_calls.append((message_set, command, flags))
        return ("OK", [b"Stored"])

    def close(self) -> IMAPResponse:
        self.close_calls += 1
        return self._close_result

    def logout(self) -> IMAPResponse:
        self.logout_calls += 1
        return self._logout_result

    # Make IMAP4.error available for isinstance/raise compatibility
    error = imaplib.IMAP4.error


__all__ = ["DEFAULT_RAW_HEADERS", "FakeIMAP4SSL"]
