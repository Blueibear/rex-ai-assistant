"""SMSService: credential-ready stub/live SMS access.

Without Twilio credentials (``TWILIO_ACCOUNT_SID`` / ``TWILIO_AUTH_TOKEN``
unset or empty), all methods return realistic mock data so the GUI and
autonomy engine work out of the box.

When credentials are present the service connects to the Twilio REST API
using the ``twilio`` Python library.  The library is an optional dependency
— importing it at the module level is avoided deliberately so the service
degrades gracefully in environments where Twilio is not installed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from rex.integrations.models import SMSMessage, SMSThread

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub data helpers
# ---------------------------------------------------------------------------

_STUB_FROM = "+15550001234"
_STUB_USER = "+15559876543"


def _build_stub_threads() -> list[SMSThread]:
    """Return two realistic stub SMS threads."""
    now = datetime.now(timezone.utc)

    def _mins_ago(m: int) -> datetime:
        return now - timedelta(minutes=m)

    alice_msgs = [
        SMSMessage(
            id="sms-alice-001",
            thread_id="thread-alice",
            direction="inbound",
            body="Hey, are you free for a call tomorrow?",
            from_number="+14155550101",
            to_number=_STUB_USER,
            sent_at=_mins_ago(30),
            status="delivered",
        ),
        SMSMessage(
            id="sms-alice-002",
            thread_id="thread-alice",
            direction="outbound",
            body="Sure, how about 3pm?",
            from_number=_STUB_USER,
            to_number="+14155550101",
            sent_at=_mins_ago(25),
            status="delivered",
        ),
        SMSMessage(
            id="sms-alice-003",
            thread_id="thread-alice",
            direction="inbound",
            body="Perfect, talk then!",
            from_number="+14155550101",
            to_number=_STUB_USER,
            sent_at=_mins_ago(20),
            status="delivered",
        ),
    ]

    bob_msgs = [
        SMSMessage(
            id="sms-bob-001",
            thread_id="thread-bob",
            direction="inbound",
            body="Don't forget the team lunch at noon.",
            from_number="+14155550202",
            to_number=_STUB_USER,
            sent_at=_mins_ago(120),
            status="delivered",
        ),
        SMSMessage(
            id="sms-bob-002",
            thread_id="thread-bob",
            direction="outbound",
            body="Thanks for the reminder, see you there!",
            from_number=_STUB_USER,
            to_number="+14155550202",
            sent_at=_mins_ago(115),
            status="delivered",
        ),
    ]

    return [
        SMSThread(
            id="thread-alice",
            contact_name="Alice",
            contact_number="+14155550101",
            messages=alice_msgs,
            last_message_at=alice_msgs[-1].sent_at,
            unread_count=1,
        ),
        SMSThread(
            id="thread-bob",
            contact_name="Bob",
            contact_number="+14155550202",
            messages=bob_msgs,
            last_message_at=bob_msgs[-1].sent_at,
            unread_count=0,
        ),
    ]


# ---------------------------------------------------------------------------
# SMSService
# ---------------------------------------------------------------------------


class SMSService:
    """Unified SMS access layer with stub and Twilio live backends.

    Args:
        sms_provider: One of ``"none"`` (stub) or ``"twilio"``.
            Defaults to ``"none"``.
        account_sid: Twilio Account SID (read from ``TWILIO_ACCOUNT_SID``
            env var if not supplied explicitly).
        auth_token: Twilio Auth Token (read from ``TWILIO_AUTH_TOKEN``
            env var if not supplied explicitly).
        from_number: Twilio phone number to send from in E.164 format
            (read from ``TWILIO_FROM_NUMBER`` env var if not supplied).
    """

    def __init__(
        self,
        sms_provider: str = "none",
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_number: str | None = None,
    ) -> None:
        import os

        self._provider = sms_provider.lower()
        self._sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID", "")
        self._token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN", "")
        self._from_number = from_number or os.environ.get("TWILIO_FROM_NUMBER", _STUB_FROM)
        logger.debug("SMSService initialised with provider=%s", self._provider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_threads(self) -> list[SMSThread]:
        """Return a list of SMS conversation threads.

        In stub mode returns 2 mock threads.  In live mode queries the
        Twilio Messages API and groups by remote number.

        Returns:
            List of :class:`~rex.integrations.models.SMSThread` objects.
        """
        if self._provider == "twilio" and self._sid and self._token:
            return self._twilio_list_threads()
        return _build_stub_threads()

    def get_thread(self, thread_id: str) -> SMSThread:
        """Return a single thread by its identifier.

        In stub mode searches the built-in stub threads.

        Args:
            thread_id: The thread identifier.

        Returns:
            Matching :class:`~rex.integrations.models.SMSThread`.

        Raises:
            KeyError: If no thread with *thread_id* exists.
        """
        threads = self.list_threads()
        for t in threads:
            if t.id == thread_id:
                return t
        raise KeyError(f"Thread not found: {thread_id!r}")

    def send(self, to: str, body: str) -> SMSMessage:
        """Send an SMS message.

        In stub mode, logs the message and returns an :class:`SMSMessage`
        with ``status='stub'``.  In live mode, calls Twilio.

        Args:
            to: Recipient phone number in E.164 format.
            body: Message body text.

        Returns:
            :class:`~rex.integrations.models.SMSMessage` representing the
            sent (or stub) message.
        """
        if self._provider == "twilio" and self._sid and self._token:
            return self._twilio_send(to, body)
        logger.info("[SMS STUB] Would send to %s: %s", to, body)
        return SMSMessage(
            id=f"stub-{uuid.uuid4().hex[:8]}",
            thread_id=f"thread-{to.replace('+', '')}",
            direction="outbound",
            body=body,
            from_number=self._from_number,
            to_number=to,
            sent_at=datetime.now(timezone.utc),
            status="stub",
        )

    # ------------------------------------------------------------------
    # Twilio live backend
    # ------------------------------------------------------------------

    def _twilio_list_threads(self) -> list[SMSThread]:
        try:
            from twilio.rest import Client

            client = Client(self._sid, self._token)
            raw_messages = client.messages.list(limit=200)
            threads: dict[str, SMSThread] = {}
            for m in raw_messages:
                direction: Literal["inbound", "outbound"] = (
                    "inbound" if m.direction == "inbound" else "outbound"
                )
                remote = m.from_ if direction == "inbound" else m.to
                thread_id = f"thread-{remote.replace('+', '')}"
                msg = SMSMessage(
                    id=m.sid,
                    thread_id=thread_id,
                    direction=direction,
                    body=m.body,
                    from_number=m.from_,
                    to_number=m.to,
                    sent_at=(
                        m.date_sent.replace(tzinfo=timezone.utc)
                        if m.date_sent
                        else datetime.now(timezone.utc)
                    ),
                    status="delivered",
                )
                if thread_id not in threads:
                    threads[thread_id] = SMSThread(
                        id=thread_id,
                        contact_name=remote,
                        contact_number=remote,
                        messages=[],
                        last_message_at=msg.sent_at,
                        unread_count=0,
                    )
                threads[thread_id].messages.append(msg)
                if msg.sent_at > threads[thread_id].last_message_at:
                    threads[thread_id] = threads[thread_id].model_copy(
                        update={"last_message_at": msg.sent_at}
                    )
            return sorted(threads.values(), key=lambda t: t.last_message_at, reverse=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("Twilio list_threads failed: %s — falling back to stub", exc)
            return _build_stub_threads()

    def _twilio_send(self, to: str, body: str) -> SMSMessage:
        try:
            from twilio.rest import Client  # noqa: PLC0415

            client = Client(self._sid, self._token)
            m = client.messages.create(body=body, from_=self._from_number, to=to)
            return SMSMessage(
                id=m.sid,
                thread_id=f"thread-{to.replace('+', '')}",
                direction="outbound",
                body=body,
                from_number=self._from_number,
                to_number=to,
                sent_at=datetime.now(timezone.utc),
                status="sent",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Twilio send failed to %s: %s — falling back to stub", to, exc)
            logger.info("[SMS STUB] Would send to %s: %s", to, body)
            return SMSMessage(
                id=f"stub-{uuid.uuid4().hex[:8]}",
                thread_id=f"thread-{to.replace('+', '')}",
                direction="outbound",
                body=body,
                from_number=self._from_number,
                to_number=to,
                sent_at=datetime.now(timezone.utc),
                status="stub",
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["SMSService"]
