"""Fake Twilio Client for offline integration tests (US-212).

Provides :class:`FakeTwilioClient` — a configurable, in-memory replacement for
``twilio.rest.Client`` that records every call and returns pre-programmed
responses.  No network connection is made.

Usage::

    from tests.helpers.fake_twilio import FakeTwilioClient

    fake_client = FakeTwilioClient()
    backend = TwilioSMSBackend(
        account_sid="ACtest",
        auth_token="tok",
        from_number="+15550001234",
        twilio_client_factory=lambda sid, tok: fake_client,
    )
    backend.send(to="+15559998888", body="Hello")
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class FakeMessage:
    """Minimal representation of a sent Twilio message."""

    sid: str = "SMfake000000000000000000000000000"
    to: str = ""
    from_: str = ""
    body: str = ""
    status: str = "queued"


class FakeMessagesResource:
    """Fake ``client.messages`` sub-resource."""

    def __init__(self, create_raises: Exception | None = None) -> None:
        self._create_raises = create_raises
        self.create_calls: list[dict] = []

    def create(self, *, to: str, from_: str, body: str) -> FakeMessage:
        self.create_calls.append({"to": to, "from_": from_, "body": body})
        if self._create_raises is not None:
            raise self._create_raises
        return FakeMessage(to=to, from_=from_, body=body)


class FakeTwilioClient:
    """In-memory fake for ``twilio.rest.Client``.

    Args:
        create_raises:  If not ``None``, ``messages.create()`` raises this
                        exception.
    """

    def __init__(self, *, create_raises: Exception | None = None) -> None:
        self.messages = FakeMessagesResource(create_raises=create_raises)

    # Convenience accessor for assertions
    @property
    def create_calls(self) -> list[dict]:
        return self.messages.create_calls


@pytest.fixture
def fake_twilio_client() -> FakeTwilioClient:
    """Reusable fake Twilio client fixture for offline tests."""
    return FakeTwilioClient()


__all__ = ["FakeMessage", "FakeTwilioClient", "fake_twilio_client"]
