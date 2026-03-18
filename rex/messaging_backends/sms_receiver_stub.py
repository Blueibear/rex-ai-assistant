"""In-memory SMS receiver stub for offline development and testing.

``SmsReceiverStub`` exposes an ``inject`` method that accepts a test inbound
message and routes it through the same handler that real inbound SMS messages
would use, without requiring a live Twilio webhook.

The stub makes no network calls.  All state is in-memory and accessible via
the ``received_messages`` property for test assertions.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class InboundSmsHandlerResult:
    """Result produced by the inbound message handler."""

    status: str  # "received" | "error"
    sid: str
    from_number: str
    to_number: str
    body: str
    error: str | None = None


@dataclass
class ReceivedSmsRecord:
    """Structured record of a single injected inbound SMS."""

    sid: str
    from_number: str
    to_number: str
    body: str
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Type alias for a handler callable
InboundHandler = Callable[[ReceivedSmsRecord], InboundSmsHandlerResult]


def _default_handler(record: ReceivedSmsRecord) -> InboundSmsHandlerResult:
    """Default inbound handler: acknowledge the message and return it unchanged.

    Real production code wires a richer handler here (e.g. one that persists
    to ``InboundSmsStore`` and dispatches to the planner).  The default is
    intentionally minimal so the stub works without any production dependencies.
    """
    logger.info(
        "[SmsReceiverStub] Received inbound SMS from %s: %s",
        record.from_number,
        record.body[:80],
    )
    return InboundSmsHandlerResult(
        status="received",
        sid=record.sid,
        from_number=record.from_number,
        to_number=record.to_number,
        body=record.body,
    )


class SmsReceiverStub:
    """In-memory SMS receive stub.

    Injected messages are processed by a configurable handler and stored in
    ``self._inbox``.  Both the inbox and handler responses are accessible for
    test assertions.  No real network calls are ever made.

    Args:
        handler: Optional callable to process each inbound message.
            When omitted, ``_default_handler`` is used.
        default_to_number: Default destination number for injected messages
            when ``to_number`` is not supplied.
    """

    def __init__(
        self,
        handler: InboundHandler | None = None,
        default_to_number: str = "+15555559999",
    ) -> None:
        self._handler: InboundHandler = handler or _default_handler
        self._default_to_number = default_to_number
        self._inbox: list[ReceivedSmsRecord] = []
        self._responses: list[InboundSmsHandlerResult] = []

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def inject(
        self,
        from_number: str,
        body: str,
        *,
        to_number: str | None = None,
        sid: str | None = None,
        received_at: datetime | None = None,
    ) -> InboundSmsHandlerResult:
        """Inject a test inbound SMS and route it through the handler.

        This is the method that replaces a live Twilio webhook POST in tests.
        The injected message is passed to the configured handler exactly as a
        real inbound message would be, so handler logic can be tested without
        any network infrastructure.

        Args:
            from_number: Sender phone number (E.164 format preferred).
            body: Message body text.
            to_number: Destination number.  Defaults to ``default_to_number``.
            sid: Message SID.  Auto-generated when omitted.
            received_at: Timestamp.  Defaults to ``datetime.now(UTC)``.

        Returns:
            The ``InboundSmsHandlerResult`` produced by the handler.
        """
        record = ReceivedSmsRecord(
            sid=sid or f"stub_rx_{uuid.uuid4().hex[:16]}",
            from_number=from_number,
            to_number=to_number or self._default_to_number,
            body=body,
            received_at=received_at or datetime.now(timezone.utc),
        )
        self._inbox.append(record)
        result = self._handler(record)
        self._responses.append(result)
        logger.debug(
            "[SmsReceiverStub] Handler returned status=%s for sid=%s",
            result.status,
            record.sid,
        )
        return result

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    @property
    def received_messages(self) -> list[ReceivedSmsRecord]:
        """Return a copy of all injected inbound messages."""
        return list(self._inbox)

    @property
    def handler_responses(self) -> list[InboundSmsHandlerResult]:
        """Return a copy of all handler results, in injection order."""
        return list(self._responses)

    def clear(self) -> None:
        """Clear inbox and responses (useful between test cases)."""
        self._inbox.clear()
        self._responses.clear()


__all__ = [
    "InboundHandler",
    "InboundSmsHandlerResult",
    "ReceivedSmsRecord",
    "SmsReceiverStub",
]
