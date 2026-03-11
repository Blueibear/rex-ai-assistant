"""TwilioAdapter Protocol — common interface for outbound SMS delivery.

All concrete adapters (stub or real Twilio) must implement this Protocol so
that calling code can swap implementations without any change outside the
adapter registration point.

Usage example::

    from rex.messaging_backends.twilio_adapter import TwilioAdapter
    from rex.messaging_backends.sms_sender_stub import SmsSenderStub

    adapter: TwilioAdapter = SmsSenderStub()
    adapter.send_sms(to="+15555550100", body="Hello from Rex")
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TwilioAdapter(Protocol):
    """Minimal interface that every SMS adapter must satisfy.

    A class satisfies this Protocol when it exposes a ``send_sms`` method
    with the specified signature.  The return value is intentionally typed as
    ``dict[str, Any]`` so that both the stub (which returns rich metadata) and
    a real Twilio client (which may return the Twilio API response) can coexist
    under the same type without wrapping.

    Methods:
        send_sms: Send an outbound SMS message.
    """

    def send_sms(self, to: str, body: str) -> dict[str, Any]:
        """Send an SMS message.

        Args:
            to: Destination phone number (E.164 format preferred, e.g. ``+15551234567``).
            body: Text body of the message.

        Returns:
            A mapping containing at minimum an ``"ok"`` key (bool) indicating
            success, plus any adapter-specific metadata (e.g. ``"sid"``).
        """
        ...  # pragma: no cover


__all__ = ["TwilioAdapter"]
