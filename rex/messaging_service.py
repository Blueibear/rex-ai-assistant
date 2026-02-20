"""Messaging service framework for Rex.

This module provides an extensible messaging interface that supports
multiple channels (SMS, Telegram, Discord, etc.). It includes:

- Message model for representing messages across channels
- Abstract MessagingService base class
- Concrete SMSService implementation backed by pluggable backends
  (stub for offline dev, Twilio for real delivery)

The framework is designed to be channel-agnostic, allowing easy addition
of new messaging channels in the future.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global SMS service instance
_sms_service: SMSService | None = None


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


# --- Models ---


class Message(BaseModel):
    """A message sent or received via a messaging channel.

    This model represents messages across all channels (SMS, Telegram, etc.)
    and includes fields for threading and channel identification.
    """

    id: str = Field(
        default_factory=lambda: f"msg_{uuid.uuid4().hex[:16]}",
        description="Unique message identifier",
    )
    channel: str = Field(
        ...,
        description="Channel type (sms, telegram, discord, etc.)",
    )
    to: str = Field(
        ...,
        description="Recipient identifier (phone number, user ID, etc.)",
    )
    from_: str = Field(
        ...,
        description="Sender identifier",
        alias="from",
    )
    body: str = Field(
        ...,
        description="Message content",
    )
    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="Message timestamp (UTC)",
    )
    thread_id: str | None = Field(
        default=None,
        description="Thread identifier for grouping related messages",
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": "msg_abc123",
                    "channel": "sms",
                    "to": "+15551234567",
                    "from": "+15559876543",
                    "body": "Hello from Rex!",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "thread_id": "thread_xyz",
                }
            ]
        },
    }


# --- Base Messaging Service ---


class MessagingService(ABC):
    """Abstract base class for messaging services.

    Subclasses implement specific messaging channels (SMS, Telegram, etc.)
    and provide methods for sending, receiving, and replying to messages.
    """

    @abstractmethod
    def send(self, message: Message, account_id: str | None = None) -> Message:
        """Send a message via this channel.

        Args:
            message: The message to send
            account_id: Optional account ID for multi-account routing

        Returns:
            The sent message with updated timestamp and ID

        Raises:
            PermissionError: If credentials are missing or invalid
            ValueError: If message format is invalid for this channel
        """
        pass

    @abstractmethod
    def receive(self, limit: int = 10) -> list[Message]:
        """Retrieve recent inbound messages.

        Args:
            limit: Maximum number of messages to retrieve

        Returns:
            List of received messages, newest first
        """
        pass

    @abstractmethod
    def reply(self, thread_id: str, body: str) -> Message:
        """Send a reply to an existing message thread.

        Args:
            thread_id: The thread to reply to
            body: The reply message body

        Returns:
            The sent reply message

        Raises:
            ValueError: If thread_id is not found
        """
        pass


# --- SMS Service Implementation ---


class SMSService(MessagingService):
    """SMS messaging service backed by a pluggable SmsBackend.

    When configured with a backend (stub or Twilio), send/receive operations
    are delegated to the backend.  The service layer adds thread awareness,
    validation, and the high-level ``Message`` model on top.

    For backward compatibility, the service falls back to its original
    JSON mock-file behavior when no backend is explicitly provided.
    """

    def __init__(
        self,
        mock_file: Path | None = None,
        from_number: str | None = None,
        *,
        backend: Any | None = None,
        raw_config: dict[str, Any] | None = None,
    ):
        """Initialize the SMS service.

        Args:
            mock_file: Path to mock SMS data file (for testing/legacy).
            from_number: Default sender phone number.
            backend: An explicit SmsBackend instance (takes priority).
            raw_config: Runtime config dict to auto-create a backend from.
        """
        self._backend = backend
        self._raw_config = raw_config
        self.mock_file = mock_file or Path("data/mock_sms.json")
        self.from_number = from_number or "+15555551234"

        if self._backend is None and self._raw_config is not None:
            self._backend = self._create_backend_from_config()

        if self._backend is None:
            self._ensure_mock_file()

    def _create_backend_from_config(self) -> Any:
        """Create a backend from runtime config."""
        try:
            from rex.messaging_backends.factory import create_sms_backend

            backend = create_sms_backend(self._raw_config, fixture_path=self.mock_file)
            # Sync from_number from backend config
            from rex.messaging_backends.account_config import load_messaging_config

            config = load_messaging_config(self._raw_config or {})
            acct = config.get_account()
            if acct:
                self.from_number = acct.from_number
            return backend
        except Exception as exc:
            logger.warning("Failed to create messaging backend from config: %s", exc)
            return None

    @property
    def active_backend(self) -> Any:
        """The active SmsBackend, or None if using legacy mock mode."""
        return self._backend

    def _ensure_mock_file(self) -> None:
        """Ensure the mock data file exists."""
        self.mock_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.mock_file.exists():
            self.mock_file.write_text(json.dumps({"messages": []}, indent=2))

    def _load_messages(self) -> list[Message]:
        """Load messages from the mock file."""
        try:
            with open(self.mock_file) as f:
                data = json.load(f)
            return [Message.model_validate(msg) for msg in data.get("messages", [])]
        except Exception as e:
            logger.warning("Failed to load mock messages: %s", e)
            return []

    def _save_messages(self, messages: list[Message]) -> None:
        """Save messages to the mock file."""
        try:
            data = {"messages": [msg.model_dump(mode="json", by_alias=True) for msg in messages]}
            with open(self.mock_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to save mock messages: %s", e)

    def _get_thread_id(self, to: str, from_: str) -> str:
        """Generate a thread ID from to/from numbers."""
        # Sort to ensure consistent thread ID regardless of direction
        participants = sorted([to, from_])
        return f"sms_thread_{hash(tuple(participants)) & 0xFFFFFFFF:08x}"

    def send(self, message: Message, account_id: str | None = None) -> Message:
        """Send an SMS message.

        When a backend is configured, delegates to ``SmsBackend.send_sms()``.
        Otherwise falls back to mock-file storage.

        Args:
            message: The message to send.
            account_id: Optional account ID for multi-account routing.

        Returns:
            The sent message with updated timestamp.

        Raises:
            ValueError: If message format is invalid.
        """
        if not message.to:
            raise ValueError("Message 'to' field is required")
        if not message.body:
            raise ValueError("Message 'body' field is required")

        # Set channel and from number if not set
        if not message.channel:
            message.channel = "sms"
        if not message.from_:
            message.from_ = self.from_number

        # Generate thread ID if not set
        if not message.thread_id:
            message.thread_id = self._get_thread_id(message.to, message.from_)

        # Update timestamp
        message.timestamp = _utc_now()

        if self._backend is not None:
            return self._send_via_backend(message, account_id=account_id)

        return self._send_via_mock(message)

    def _send_via_backend(self, message: Message, *, account_id: str | None = None) -> Message:
        """Delegate send to the configured backend."""
        from rex.messaging_backends.base import SmsBackend

        backend: SmsBackend = self._backend
        from_number = message.from_

        if account_id and self._raw_config is not None:
            try:
                from rex.messaging_backends.account_config import load_messaging_config
                from rex.messaging_backends.factory import create_sms_backend

                config = load_messaging_config(self._raw_config)
                account = config.get_account(account_id)
                if account is None:
                    raise ValueError(f"Unknown messaging account: {account_id}")

                backend = create_sms_backend(
                    self._raw_config,
                    account_id=account_id,
                    fixture_path=self.mock_file,
                )
                from_number = account.from_number
            except Exception as exc:
                logger.warning(
                    "Failed account-specific SMS backend routing for '%s': %s",
                    account_id,
                    exc,
                )

        result = backend.send_sms(
            to=message.to,
            body=message.body,
            from_number=from_number,
        )
        if not result.ok:
            logger.warning("SMS backend send failed: %s", result.error)
            raise RuntimeError(f"SMS send failed: {result.error}")

        logger.info("Sent SMS to %s via backend", message.to)
        return message

    def _send_via_mock(self, message: Message) -> Message:
        """Legacy mock-file send path."""
        messages = self._load_messages()
        messages.append(message)
        self._save_messages(messages)
        logger.info("Sent SMS to %s (mock): %s", message.to, message.body[:50])
        return message

    def receive(self, limit: int = 10) -> list[Message]:
        """Retrieve recent inbound SMS messages.

        When a backend is configured, delegates to
        ``SmsBackend.fetch_recent_inbound()``.  Otherwise reads from the
        mock file.

        Args:
            limit: Maximum number of messages to retrieve.

        Returns:
            List of received messages, newest first.
        """
        if self._backend is not None:
            return self._receive_via_backend(limit)
        return self._receive_via_mock(limit)

    def _receive_via_backend(self, limit: int) -> list[Message]:
        """Delegate receive to the configured backend."""
        from rex.messaging_backends.base import SmsBackend

        backend: SmsBackend = self._backend
        inbound = backend.fetch_recent_inbound(limit=limit)
        return [
            Message(
                channel="sms",
                to=msg.to_number,
                from_=msg.from_number,
                body=msg.body,
                timestamp=msg.received_at,
            )
            for msg in inbound
        ]

    def _receive_via_mock(self, limit: int) -> list[Message]:
        """Legacy mock-file receive path."""
        messages = self._load_messages()
        inbound = [msg for msg in messages if msg.to == self.from_number]
        inbound.sort(key=lambda m: m.timestamp, reverse=True)
        return inbound[:limit]

    def reply(self, thread_id: str, body: str) -> Message:
        """Send a reply to an existing SMS thread.

        Args:
            thread_id: The thread to reply to.
            body: The reply message body.

        Returns:
            The sent reply message.

        Raises:
            ValueError: If thread_id is not found.
        """
        # Find the thread (use mock file for thread lookup)
        messages = self._load_messages()
        thread_messages = [msg for msg in messages if msg.thread_id == thread_id]

        if not thread_messages:
            raise ValueError(f"Thread {thread_id} not found")

        # Get the most recent message to determine the recipient
        latest = thread_messages[-1]

        # Reply to the sender of the latest message
        to = latest.from_ if latest.to == self.from_number else latest.to

        reply_msg = Message(
            channel="sms",
            to=to,
            from_=self.from_number,
            body=body,
            thread_id=thread_id,
        )

        return self.send(reply_msg)


# --- Global Service Accessors ---


def get_sms_service() -> SMSService:
    """Get the global SMS service instance.

    Returns:
        The global SMS service, creating it if needed
    """
    global _sms_service
    if _sms_service is None:
        _sms_service = SMSService()
    return _sms_service


def set_sms_service(service: SMSService | None) -> None:
    """Set the global SMS service instance.

    Args:
        service: The SMS service to use (or None to reset)
    """
    global _sms_service
    _sms_service = service
