"""Tests for messaging service framework and SMS implementation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from rex.messaging_service import (
    Message,
    MessagingService,
    SMSService,
    get_sms_service,
    set_sms_service,
)


@pytest.fixture
def temp_sms_file(tmp_path):
    """Create a temporary SMS mock file."""
    mock_file = tmp_path / "mock_sms.json"
    mock_file.write_text(json.dumps({"messages": []}, indent=2))
    return mock_file


@pytest.fixture
def sms_service(temp_sms_file):
    """Create an SMS service instance with temporary file."""
    return SMSService(mock_file=temp_sms_file, from_number="+15555551234")


def test_message_model():
    """Test Message model creation and serialization."""
    message = Message(
        channel="sms",
        to="+15551234567",
        from_="+15559876543",
        body="Test message",
    )

    assert message.channel == "sms"
    assert message.to == "+15551234567"
    assert message.from_ == "+15559876543"
    assert message.body == "Test message"
    assert message.id.startswith("msg_")
    assert isinstance(message.timestamp, datetime)
    assert message.thread_id is None


def test_message_serialization():
    """Test Message JSON serialization."""
    message = Message(
        id="msg_test123",
        channel="sms",
        to="+15551234567",
        from_="+15559876543",
        body="Test message",
        thread_id="thread_xyz",
    )

    data = message.model_dump(mode="json", by_alias=True)
    assert data["id"] == "msg_test123"
    assert data["channel"] == "sms"
    assert data["from"] == "+15559876543"
    assert data["to"] == "+15551234567"
    assert data["body"] == "Test message"
    assert data["thread_id"] == "thread_xyz"


def test_sms_service_initialization(temp_sms_file):
    """Test SMS service initialization."""
    service = SMSService(mock_file=temp_sms_file)
    assert service.mock_file == temp_sms_file
    assert service.from_number == "+15555551234"
    assert temp_sms_file.exists()


def test_sms_service_send(sms_service):
    """Test sending SMS messages."""
    message = Message(
        channel="sms",
        to="+15551234567",
        from_=sms_service.from_number,
        body="Hello, World!",
    )

    sent = sms_service.send(message)

    assert sent.id == message.id
    assert sent.to == "+15551234567"
    assert sent.body == "Hello, World!"
    assert sent.thread_id is not None
    assert isinstance(sent.timestamp, datetime)


def test_sms_service_send_validation(sms_service):
    """Test SMS send validation."""
    # Missing 'to' field
    with pytest.raises(ValueError, match="'to' field is required"):
        sms_service.send(Message(channel="sms", to="", from_="+1555", body="Test"))

    # Missing 'body' field
    with pytest.raises(ValueError, match="'body' field is required"):
        sms_service.send(Message(channel="sms", to="+1555", from_="+1555", body=""))


def test_sms_service_receive_empty(sms_service):
    """Test receiving messages when none exist."""
    messages = sms_service.receive(limit=10)
    assert messages == []


def test_sms_service_receive_messages(sms_service):
    """Test receiving SMS messages."""
    # Send some messages TO the service (inbound)
    for i in range(5):
        msg = Message(
            channel="sms",
            to=sms_service.from_number,  # TO the service = inbound
            from_=f"+155512340{i}",
            body=f"Inbound message {i}",
        )
        sms_service.send(msg)

    # Send some messages FROM the service (outbound, should not appear in receive)
    for i in range(3):
        msg = Message(
            channel="sms",
            to=f"+155598760{i}",  # FROM the service = outbound
            from_=sms_service.from_number,
            body=f"Outbound message {i}",
        )
        sms_service.send(msg)

    # Receive should only return inbound messages
    received = sms_service.receive(limit=10)
    assert len(received) == 5
    assert all(msg.to == sms_service.from_number for msg in received)

    # Check they're sorted by timestamp descending (newest first)
    timestamps = [msg.timestamp for msg in received]
    assert timestamps == sorted(timestamps, reverse=True)


def test_sms_service_receive_limit(sms_service):
    """Test receive message limit."""
    # Send 10 inbound messages
    for i in range(10):
        msg = Message(
            channel="sms",
            to=sms_service.from_number,
            from_=f"+155512340{i}",
            body=f"Message {i}",
        )
        sms_service.send(msg)

    # Receive with limit
    received = sms_service.receive(limit=3)
    assert len(received) == 3


def test_sms_service_thread_awareness(sms_service):
    """Test thread ID generation for conversations."""
    # Send messages between two numbers
    msg1 = Message(
        channel="sms",
        to="+15551111111",
        from_="+15552222222",
        body="Message 1",
    )
    sent1 = sms_service.send(msg1)

    msg2 = Message(
        channel="sms",
        to="+15552222222",  # Reversed direction
        from_="+15551111111",
        body="Message 2",
    )
    sent2 = sms_service.send(msg2)

    # Both should have the same thread ID
    assert sent1.thread_id == sent2.thread_id


def test_sms_service_reply(sms_service):
    """Test replying to a message thread."""
    # Send an inbound message
    inbound = Message(
        channel="sms",
        to=sms_service.from_number,
        from_="+15551234567",
        body="Hello, Rex!",
    )
    sent_inbound = sms_service.send(inbound)

    # Reply to the thread
    reply = sms_service.reply(sent_inbound.thread_id, "Hello back!")

    assert reply.to == "+15551234567"
    assert reply.from_ == sms_service.from_number
    assert reply.body == "Hello back!"
    assert reply.thread_id == sent_inbound.thread_id


def test_sms_service_reply_invalid_thread(sms_service):
    """Test replying to non-existent thread."""
    with pytest.raises(ValueError, match="Thread .* not found"):
        sms_service.reply("invalid_thread_id", "Test reply")


def test_sms_service_persistence(temp_sms_file):
    """Test that messages are persisted to disk."""
    service = SMSService(mock_file=temp_sms_file)

    # Send a message
    msg = Message(
        channel="sms",
        to="+15551234567",
        from_=service.from_number,
        body="Persistent message",
    )
    service.send(msg)

    # Create a new service instance and verify message exists
    new_service = SMSService(mock_file=temp_sms_file)
    messages = new_service._load_messages()

    assert len(messages) == 1
    assert messages[0].body == "Persistent message"


def test_global_sms_service():
    """Test global SMS service getter/setter."""
    service1 = get_sms_service()
    service2 = get_sms_service()
    assert service1 is service2  # Should return same instance

    # Set a custom service
    custom_service = SMSService(mock_file=Path("/tmp/custom_sms.json"))
    set_sms_service(custom_service)

    service3 = get_sms_service()
    assert service3 is custom_service

    # Reset for other tests
    set_sms_service(None)


def test_messaging_service_abstract_base():
    """Test that MessagingService is abstract."""
    with pytest.raises(TypeError):
        MessagingService()  # type: ignore


def test_sms_service_send_uses_explicit_account_backend(tmp_path):
    """Explicit account_id creates account-specific backend routing."""
    mock_file = tmp_path / "mock_sms.json"

    raw_config = {
        "messaging": {
            "backend": "stub",
            "default_account_id": "primary",
            "accounts": [
                {
                    "id": "primary",
                    "label": "Primary",
                    "from_number": "+15550000001",
                    "credential_ref": "twilio:primary",
                },
                {
                    "id": "secondary",
                    "label": "Secondary",
                    "from_number": "+15550000002",
                    "credential_ref": "twilio:secondary",
                },
            ],
        }
    }

    service = SMSService(mock_file=mock_file, raw_config=raw_config)

    with patch("rex.messaging_backends.factory.create_sms_backend") as mock_factory:
        per_account_backend = mock_factory.return_value
        per_account_backend.send_sms.return_value.ok = True
        per_account_backend.send_sms.return_value.error = None

        msg = Message(
            channel="sms",
            to="+15551234567",
            from_=service.from_number,
            body="hello",
        )
        service.send(msg, account_id="secondary")

        mock_factory.assert_called_with(
            raw_config,
            account_id="secondary",
            fixture_path=mock_file,
        )
        per_account_backend.send_sms.assert_called_once()
        assert per_account_backend.send_sms.call_args.kwargs["from_number"] == "+15550000002"


def test_sms_service_send_invalid_account_falls_back_to_default_backend(tmp_path):
    """Unknown account_id should not crash and should use the existing backend."""
    mock_file = tmp_path / "mock_sms.json"

    class _Backend:
        def __init__(self):
            self.calls = []

        def send_sms(self, *, to, body, from_number=None):
            self.calls.append((to, body, from_number))

            class _Result:
                ok = True
                error = None

            return _Result()

    backend = _Backend()
    raw_config = {
        "messaging": {
            "backend": "stub",
            "accounts": [
                {
                    "id": "primary",
                    "label": "Primary",
                    "from_number": "+15550000001",
                    "credential_ref": "twilio:primary",
                }
            ],
        }
    }
    service = SMSService(mock_file=mock_file, raw_config=raw_config, backend=backend)

    msg = Message(channel="sms", to="+15551230000", from_=service.from_number, body="x")
    service.send(msg, account_id="missing")

    assert backend.calls
    assert backend.calls[0][2] == service.from_number
