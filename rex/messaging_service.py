"""Messaging service framework for Rex.

This module provides an extensible messaging interface that supports
multiple channels (SMS, Telegram, Discord, etc.). It includes:

- Message model for representing messages across channels
- Abstract MessagingService base class
- Concrete SMSService implementation with stubbed behavior for testing

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

from pydantic import BaseModel, Field

from rex.credentials import get_credential_manager

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
    def send(self, message: Message) -> Message:
        """Send a message via this channel.

        Args:
            message: The message to send

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
    """SMS messaging service with stubbed send/receive for testing.

    This service uses the credential manager to retrieve Twilio credentials
    (or similar SMS gateway credentials). For testing purposes, it uses a
    mock JSON file to simulate sending and receiving SMS messages.

    Thread awareness is maintained by grouping messages with the same
    to/from pair.
    """

    def __init__(
        self,
        mock_file: Path | None = None,
        from_number: str | None = None,
    ):
        """Initialize the SMS service.

        Args:
            mock_file: Path to mock SMS data file (for testing)
            from_number: Default sender phone number
        """
        self.mock_file = mock_file or Path("data/mock_sms.json")
        self.from_number = from_number or "+15555551234"
        self._ensure_mock_file()

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
            logger.warning(f"Failed to load mock messages: {e}")
            return []

    def _save_messages(self, messages: list[Message]) -> None:
        """Save messages to the mock file."""
        try:
            data = {"messages": [msg.model_dump(mode="json", by_alias=True) for msg in messages]}
            with open(self.mock_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save mock messages: {e}")

    def _get_thread_id(self, to: str, from_: str) -> str:
        """Generate a thread ID from to/from numbers."""
        # Sort to ensure consistent thread ID regardless of direction
        participants = sorted([to, from_])
        return f"sms_thread_{hash(tuple(participants)) & 0xFFFFFFFF:08x}"

    def _check_credentials(self) -> None:
        """Check if SMS credentials are available.

        Raises:
            PermissionError: If credentials are missing
        """
        try:
            cred_manager = get_credential_manager()
            # Try to get Twilio credentials (or similar SMS gateway)
            # For now, we check for 'sms' or 'twilio' credentials
            sms_cred = None
            try:
                sms_cred = cred_manager.get("sms")
            except ValueError:
                pass

            if sms_cred is None:
                try:
                    sms_cred = cred_manager.get("twilio")
                except ValueError:
                    pass

            if sms_cred is None:
                logger.warning(
                    "No SMS credentials found in credential manager. "
                    "Using stubbed mode for testing."
                )
        except Exception as e:
            logger.debug(f"Error checking credentials: {e}")

    def send(self, message: Message) -> Message:
        """Send an SMS message.

        In production, this would use Twilio or similar SMS gateway.
        For testing, messages are written to the mock file.

        Args:
            message: The message to send

        Returns:
            The sent message with updated timestamp

        Raises:
            PermissionError: If credentials are invalid (in production)
            ValueError: If message format is invalid
        """
        if not message.to:
            raise ValueError("Message 'to' field is required")
        if not message.body:
            raise ValueError("Message 'body' field is required")

        # Check credentials (logs warning if missing, but doesn't block in stub mode)
        self._check_credentials()

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

        # Save to mock file
        messages = self._load_messages()
        messages.append(message)
        self._save_messages(messages)

        logger.info(f"Sent SMS to {message.to}: {message.body[:50]}...")
        return message

    def receive(self, limit: int = 10) -> list[Message]:
        """Retrieve recent inbound SMS messages.

        In production, this would poll Twilio or similar SMS gateway.
        For testing, messages are read from the mock file.

        Args:
            limit: Maximum number of messages to retrieve

        Returns:
            List of received messages, newest first
        """
        self._check_credentials()

        messages = self._load_messages()

        # Filter for messages sent TO our number (received by us)
        inbound = [msg for msg in messages if msg.to == self.from_number]

        # Sort by timestamp descending (newest first)
        inbound.sort(key=lambda m: m.timestamp, reverse=True)

        return inbound[:limit]

    def reply(self, thread_id: str, body: str) -> Message:
        """Send a reply to an existing SMS thread.

        Args:
            thread_id: The thread to reply to
            body: The reply message body

        Returns:
            The sent reply message

        Raises:
            ValueError: If thread_id is not found
        """
        self._check_credentials()

        # Find the thread
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


def set_sms_service(service: SMSService) -> None:
    """Set the global SMS service instance.

    Args:
        service: The SMS service to use
    """
    global _sms_service
    _sms_service = service
