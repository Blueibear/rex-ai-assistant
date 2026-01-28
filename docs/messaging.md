# Messaging Service

Rex's messaging service provides a unified framework for sending and receiving messages across multiple channels. Currently, SMS is supported with a design that allows easy extension to other platforms like Telegram, Discord, WhatsApp, etc.

## Overview

The messaging service consists of:

- **Message Model**: A universal message format that works across all channels
- **MessagingService Base Class**: An abstract interface for implementing channel-specific services
- **SMSService**: A concrete implementation for SMS messaging (stubbed for testing)

## Architecture

### Message Model

All messages share a common structure:

```python
from rex.messaging_service import Message

message = Message(
    channel="sms",              # Channel type (sms, telegram, discord, etc.)
    to="+15551234567",          # Recipient identifier
    from_="+15559876543",       # Sender identifier
    body="Hello from Rex!",     # Message content
    thread_id="thread_xyz"      # Optional: Thread/conversation ID
)
```

**Fields:**
- `id`: Unique message identifier (auto-generated)
- `channel`: Channel type (e.g., "sms", "telegram")
- `to`: Recipient identifier (phone number, user ID, etc.)
- `from_`: Sender identifier
- `body`: Message content
- `timestamp`: Message timestamp (UTC, auto-generated)
- `thread_id`: Optional thread identifier for grouping related messages

### Thread Awareness

Messages are automatically grouped into threads based on participants:
- For SMS: Thread ID is derived from the to/from phone numbers
- For other channels: Thread ID can be conversation ID, chat ID, etc.

This enables:
- Following conversation context
- Replying to specific threads
- Tracking message history per conversation

## Using the SMS Service

### Configuration

The SMS service requires credentials from your SMS gateway (e.g., Twilio):

1. **Via Environment Variables:**
   ```bash
   export SMS_TOKEN="your_twilio_auth_token"
   export TWILIO_ACCOUNT_SID="your_account_sid"
   ```

2. **Via Credential Manager:**
   ```python
   from rex.credentials import get_credential_manager

   cred_manager = get_credential_manager()
   cred_manager.set("sms", "your_auth_token")
   cred_manager.set("twilio", "your_account_sid")
   ```

### Sending Messages

**Python API:**
```python
from rex.messaging_service import Message, get_sms_service

sms = get_sms_service()

# Send a message
message = Message(
    channel="sms",
    to="+15551234567",
    from_=sms.from_number,
    body="Hello from Rex!"
)

sent = sms.send(message)
print(f"Sent message ID: {sent.id}")
print(f"Thread ID: {sent.thread_id}")
```

**CLI:**
```bash
# Send an SMS
rex msg send --channel sms --to "+15551234567" --body "Hello from Rex!"
```

### Receiving Messages

**Python API:**
```python
# Get recent inbound messages
messages = sms.receive(limit=10)

for msg in messages:
    print(f"From: {msg.from_}")
    print(f"Body: {msg.body}")
    print(f"Thread: {msg.thread_id}")
```

**CLI:**
```bash
# Receive recent SMS messages
rex msg receive --channel sms --limit 10
```

### Replying to Messages

**Python API:**
```python
# Reply to a specific thread
reply = sms.reply(thread_id="thread_xyz", body="Thanks for your message!")
```

## Extending to New Channels

To add support for a new messaging channel (e.g., Telegram):

1. **Create a subclass of MessagingService:**

```python
from rex.messaging_service import MessagingService, Message

class TelegramService(MessagingService):
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        # Initialize Telegram bot client

    def send(self, message: Message) -> Message:
        # Implement Telegram API call
        # Update message.timestamp
        # Return updated message
        pass

    def receive(self, limit: int = 10) -> list[Message]:
        # Poll Telegram for updates
        # Convert to Message objects
        # Return list
        pass

    def reply(self, thread_id: str, body: str) -> Message:
        # Send reply in Telegram chat
        # Return sent message
        pass
```

2. **Register global accessor functions:**

```python
_telegram_service = None

def get_telegram_service() -> TelegramService:
    global _telegram_service
    if _telegram_service is None:
        _telegram_service = TelegramService(token=...)
    return _telegram_service
```

3. **Update CLI to support the new channel:**

Add cases for "telegram" in `cmd_msg()` in `rex/cli.py`.

## Testing

The SMS service includes stubbed behavior for testing:

- Messages are stored in `data/mock_sms.json`
- No real SMS gateway calls are made
- All functionality is simulated locally

To run tests:

```bash
# Test messaging service
pytest tests/test_messaging_service.py -v

# Test CLI commands
pytest tests/test_cli_messaging_notification.py -k msg -v
```

## Security Considerations

1. **No Secrets in Code**: Never hardcode phone numbers, API tokens, or other sensitive data
2. **Use Credential Manager**: Always retrieve tokens via the credential manager
3. **Validate Input**: The service validates recipient and body fields before sending
4. **Rate Limiting**: Consider implementing rate limits for production use
5. **Logging**: Sensitive data is automatically redacted in logs

## Mock Mode

When credentials are not available, the SMS service operates in mock mode:

- Sends are written to `data/mock_sms.json`
- Receives read from the same file
- No external API calls are made
- Perfect for development and testing

## Future Enhancements

Planned features:
- **MMS Support**: Send images and media via SMS
- **Telegram Integration**: Native Telegram bot support
- **Discord Integration**: Discord channel messaging
- **WhatsApp Business API**: WhatsApp messaging
- **Message Templates**: Pre-defined message formats
- **Scheduling**: Schedule messages for future delivery
- **Delivery Receipts**: Track message delivery status
- **Two-Way Webhooks**: Real-time inbound message handling

## API Reference

### Message

**Fields:**
- `id: str` - Unique identifier
- `channel: str` - Channel type
- `to: str` - Recipient
- `from_: str` - Sender (use `from` in JSON)
- `body: str` - Message content
- `timestamp: datetime` - When sent (UTC)
- `thread_id: Optional[str]` - Thread identifier

### MessagingService

**Abstract Methods:**
- `send(message: Message) -> Message` - Send a message
- `receive(limit: int = 10) -> list[Message]` - Retrieve inbound messages
- `reply(thread_id: str, body: str) -> Message` - Reply to thread

### SMSService

**Constructor:**
- `SMSService(mock_file: Path = None, from_number: str = None)`

**Methods:**
- Inherits all MessagingService methods
- `_check_credentials()` - Verify SMS credentials available

**Raises:**
- `PermissionError` - If credentials are invalid (production)
- `ValueError` - If message format is invalid

## Troubleshooting

### "No SMS credentials found"

This is a warning, not an error. The service will operate in mock mode. To fix:
1. Set `SMS_TOKEN` environment variable
2. Or configure via credential manager

### "Message 'to' field is required"

Ensure you're providing a valid recipient identifier:
```python
message = Message(channel="sms", to="+15551234567", from_=..., body=...)
```

### Messages not appearing in receive()

- Check that messages were sent TO your from_number (inbound)
- Messages sent FROM your number are outbound and won't appear
- Verify mock_sms.json contains the expected messages

## Examples

### Simple Send/Receive

```python
from rex.messaging_service import Message, get_sms_service

sms = get_sms_service()

# Send
msg = Message(channel="sms", to="+15551234567", from_=sms.from_number, body="Hi!")
sms.send(msg)

# Receive
messages = sms.receive(limit=5)
for m in messages:
    print(f"{m.from_}: {m.body}")
```

### Conversation Flow

```python
# User sends message to Rex
inbound = Message(
    channel="sms",
    to="+15555551234",  # Rex's number
    from_="+15551234567",
    body="What's the weather?"
)
sms.send(inbound)

# Rex receives and processes
messages = sms.receive(limit=1)
user_message = messages[0]

# Rex replies
reply = sms.reply(user_message.thread_id, "It's sunny and 75°F!")
```

## Support

For questions or issues:
- Check existing tests in `tests/test_messaging_service.py`
- Review code in `rex/messaging_service.py`
- Open an issue on GitHub
