# Messaging Service

## Current Implementation

**Status: Beta (real backend available)**

The messaging framework supports real SMS delivery via Twilio when credentials are configured. Without credentials the SMS service operates in stub/mock mode — messages are written to a local JSON file and no external API calls are made.

| Channel | Status | Details |
|---------|--------|---------|
| **SMS** | Beta (real when configured) | Uses Twilio REST API for real delivery; defaults to stub mode with `data/mock_sms.json`. Multi-account support included. |
| **Telegram** | Not implemented | Extension example provided in docs; no built-in adapter. |
| **Discord / WhatsApp** | Not implemented | Planned for future releases. |

---

Rex's messaging service provides a unified framework for sending and receiving messages across multiple channels.

## Overview

The messaging service consists of:

- **Message Model**: A universal message format that works across all channels
- **MessagingService Base Class**: An abstract interface for implementing channel-specific services
- **SMSService**: A concrete implementation backed by pluggable SMS backends
- **SMS Backends**: Stub (offline/mock) and Twilio (real delivery)

## Architecture

### Backend System

SMS delivery is handled by pluggable backends in `rex/messaging_backends/`:

| Backend | Module | Description |
|---------|--------|-------------|
| **Stub** | `rex.messaging_backends.stub` | Writes to local JSON file; no real delivery. Default for offline dev. |
| **Twilio** | `rex.messaging_backends.twilio_backend` | Calls Twilio REST API via `requests`. Requires credentials. |

The backend is selected via config (`messaging.backend`). If Twilio credentials are missing, the system falls back to stub with a warning.

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

## Configuration

### Backend Selection

Set the backend in `config/rex_config.json`:

```json
{
  "messaging": {
    "backend": "twilio",
    "default_account_id": "primary",
    "accounts": [
      {
        "id": "primary",
        "label": "Main Twilio",
        "from_number": "+15551234567",
        "credential_ref": "twilio:primary"
      }
    ]
  }
}
```

**Config keys:**
- `messaging.backend`: `"stub"` (default) or `"twilio"`
- `messaging.default_account_id`: Account to use when none is specified
- `messaging.accounts[]`: List of SMS accounts with `id`, `label`, `from_number`, and `credential_ref`

### Credentials

Credentials are stored separately from config (never in code or config files):

1. **Via Environment Variables:**
   ```bash
   # Format: account_sid:auth_token
   export TWILIO_PRIMARY="ACxxxxxxxxx:your_auth_token"
   ```

2. **Via Credential Manager (config/credentials.json):**
   ```json
   {
     "credentials": {
       "twilio:primary": "ACxxxxxxxxx:your_auth_token"
     }
   }
   ```

The `credential_ref` in the account config maps to the key in the credential store.

### No-Credential Behavior

When Twilio credentials are not available:
- A warning is logged
- The system falls back to stub mode automatically
- No errors are thrown; the service degrades gracefully
- Messages are written to `data/mock_sms.json`

## Using the SMS Service

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

# Send with specific account
rex msg send --channel sms --to "+15551234567" --body "Hello" --account-id secondary

# Send as specific user
rex msg send --channel sms --to "+15551234567" --body "Hello" --user alice
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

## Multi-Account Support

Multiple SMS accounts can be configured for different purposes (personal, business, etc.):

```json
{
  "messaging": {
    "backend": "twilio",
    "default_account_id": "personal",
    "accounts": [
      {
        "id": "personal",
        "from_number": "+15551234567",
        "credential_ref": "twilio:personal"
      },
      {
        "id": "business",
        "label": "Business Line",
        "from_number": "+15559876543",
        "credential_ref": "twilio:business"
      }
    ]
  }
}
```

Account selection priority:
1. Explicit `--account-id` flag
2. `default_account_id` from config
3. First account in list

## Extending to New Channels

To add support for a new messaging channel (e.g., Telegram):

1. **Create a subclass of MessagingService:**

```python
from rex.messaging_service import MessagingService, Message

class TelegramService(MessagingService):
    def __init__(self, bot_token: str):
        self.bot_token = bot_token

    def send(self, message: Message, account_id=None) -> Message:
        # Implement Telegram API call
        pass

    def receive(self, limit: int = 10) -> list[Message]:
        # Poll Telegram for updates
        pass

    def reply(self, thread_id: str, body: str) -> Message:
        # Send reply in Telegram chat
        pass
```

2. **Register global accessor functions** and update CLI.

## Testing

The SMS service includes stubbed behavior for testing:

- Messages are stored in a temporary file (or `data/mock_sms.json` in mock mode)
- No real SMS gateway calls are made
- All functionality is simulated locally

To run tests:

```bash
# Test messaging backends
pytest tests/test_messaging_backends.py -v

# Test messaging service
pytest tests/test_messaging_service.py -v

# Test CLI commands
pytest tests/test_cli_messaging_notification.py -k msg -v
```

## Security Considerations

1. **No Secrets in Code**: Never hardcode phone numbers, API tokens, or other sensitive data
2. **Use Credential Manager**: Always retrieve tokens via the credential manager
3. **Validate Input**: The service validates recipient and body fields before sending
4. **Credential Split**: Non-secret config (from_number) in `rex_config.json`; secrets in `.env` or `credentials.json`
5. **Safe Fallback**: Missing credentials result in stub mode, not errors
6. **Logging**: Phone numbers in logs are not treated as secrets; Twilio auth tokens are masked

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
- `send(message: Message, account_id=None) -> Message` - Send a message
- `receive(limit: int = 10) -> list[Message]` - Retrieve inbound messages
- `reply(thread_id: str, body: str) -> Message` - Reply to thread

### SMSService

**Constructor:**
- `SMSService(mock_file: Path = None, from_number: str = None, backend=None, raw_config=None)`

**Properties:**
- `active_backend` - The active SmsBackend, or None if using legacy mock mode

### SmsBackend (ABC)

**Methods:**
- `send_sms(to, body, from_number=None) -> SmsSendResult` - Send an SMS
- `fetch_recent_inbound(limit=20) -> list[InboundSms]` - Fetch inbound messages

### StubSmsBackend

- Writes to local JSON file
- `inject_inbound()` for test setup
- `sent_messages` property for test assertions

### TwilioSmsBackend

- Uses Twilio REST API via `requests`
- Configurable timeout
- Handles error responses, timeouts, and connection errors

## What Remains Stubbed

- Telegram, Discord, WhatsApp channels (not implemented)
- MMS / media messages
- Inbound webhook receiver for real-time Twilio messages
- Per-user phone number routing (foundation exists)

## Troubleshooting

### "No SMS credentials found"

This is a warning, not an error. The service will operate in stub mode. To fix:
1. Configure `messaging.backend: "twilio"` in `rex_config.json`
2. Add account with `credential_ref`
3. Set credential via environment variable or `credentials.json`

### "Message 'to' field is required"

Ensure you're providing a valid recipient identifier:
```python
message = Message(channel="sms", to="+15551234567", from_=..., body=...)
```

### Messages not appearing in receive()

- In stub mode: check that messages were sent TO your from_number (inbound)
- In Twilio mode: messages fetch from the Twilio API filtered by your number

## Support

For questions or issues:
- Check existing tests in `tests/test_messaging_backends.py` and `tests/test_messaging_service.py`
- Review code in `rex/messaging_service.py` and `rex/messaging_backends/`
- Open an issue on GitHub
