# Messaging Service

## Current Implementation

**Status: Beta (real backend available, inbound webhook supported when configured)**

The messaging framework supports real SMS delivery via Twilio when credentials are configured. Without credentials the SMS service operates in stub/mock mode — messages are written to a local JSON file and no external API calls are made.

Inbound SMS support is available via a Twilio webhook receiver. When `messaging.inbound.enabled` is `true`, incoming messages are persisted to a local SQLite store and are retrievable via `rex msg receive`.

| Channel | Status | Details |
|---------|--------|---------|
| **SMS (outbound)** | Beta (real when configured) | Uses Twilio REST API for real delivery; defaults to stub mode with `data/mock_sms.json`. Multi-account support included. |
| **SMS (inbound)** | Beta (webhook when configured) | Twilio webhook receiver with HMAC-SHA1 signature verification. Messages persisted to SQLite and routed by phone number to the matching account. |
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
- **Inbound Webhook**: A Flask blueprint that receives Twilio inbound SMS via a POST endpoint

## Architecture

### Backend System

SMS delivery is handled by pluggable backends in `rex/messaging_backends/`:

| Backend | Module | Description |
|---------|--------|-------------|
| **Stub** | `rex.messaging_backends.stub` | Writes to local JSON file; no real delivery. Default for offline dev. |
| **Twilio** | `rex.messaging_backends.twilio_backend` | Calls Twilio REST API via `requests`. Requires credentials. |
| **Inbound Store** | `rex.messaging_backends.inbound_store` | SQLite-backed persistence for webhook-received inbound messages. |
| **Inbound Webhook** | `rex.messaging_backends.inbound_webhook` | Flask blueprint for receiving Twilio inbound SMS with signature verification. |
| **Signature Verification** | `rex.messaging_backends.twilio_signature` | HMAC-SHA1 signature validation using stdlib only (no Twilio SDK). |

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

### Inbound SMS Configuration

To receive inbound SMS via webhook, add the `inbound` section:

```json
{
  "messaging": {
    "backend": "twilio",
    "default_account_id": "primary",
    "accounts": [
      {
        "id": "primary",
        "from_number": "+15551234567",
        "credential_ref": "twilio:primary"
      }
    ],
    "inbound": {
      "enabled": true,
      "auth_token_ref": "twilio:inbound",
      "store_path": "data/inbound_sms.db",
      "retention_days": 90,
      "rate_limit": "120 per minute"
    }
  }
}
```

**Inbound config keys:**
- `messaging.inbound.enabled`: `false` (default) — set to `true` to enable the webhook endpoint
- `messaging.inbound.auth_token_ref`: Credential ref for the Twilio auth token used for webhook signature verification (default: `"twilio:inbound"`)
- `messaging.inbound.store_path`: SQLite database path for inbound messages (default: `data/inbound_sms.db`)
- `messaging.inbound.retention_days`: Days to retain inbound messages before automatic cleanup (default: `90`)
- `messaging.inbound.rate_limit`: Flask-Limiter rate limit string for the webhook endpoint (default: `"120 per minute"`)

**Inbound webhook endpoint:**
- `POST /webhooks/twilio/sms` — receives inbound SMS from Twilio

**Routing behavior:**
- The `To` phone number in the Twilio request is matched against `messaging.accounts[].from_number`
- If a match is found, the message is associated with that account ID
- If no match is found, the message is stored as "unrouted" with a warning logged

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
       "twilio:primary": "ACxxxxxxxxx:your_auth_token",
       "twilio:inbound": "your_twilio_auth_token_for_signature_verification"
     }
   }
   ```

The `credential_ref` in the account config maps to the key in the credential store.

For inbound webhook signature verification, the `auth_token_ref` specifies which credential to use for HMAC-SHA1 validation.

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
# Get recent inbound messages (includes webhook-received messages)
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

When the inbound store is enabled, `rex msg receive` merges messages from both the backend (Twilio API or stub) and the inbound webhook store. Messages are sorted newest-first and deduplicated.

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

For inbound messages received via webhook, routing is automatic: the `To` number is matched against configured account `from_number` values.

## User Routing

Inbound SMS messages can be automatically associated with a Rex user profile
at ingest time. This enables per-user filtering with `rex msg receive --user <id>`
and `SMSService.receive(user_id=...)`.

### How it works

Each messaging account can declare an `owner_user_id` — the user profile that
owns the phone number. When the inbound webhook receives a message, it matches
the `To` number to an account and tags the stored record with that account's
`owner_user_id`.

```json
{
  "messaging": {
    "accounts": [
      {
        "id": "personal",
        "from_number": "+15551111111",
        "credential_ref": "twilio:personal",
        "owner_user_id": "alice"
      },
      {
        "id": "business",
        "from_number": "+15552222222",
        "credential_ref": "twilio:business"
      }
    ]
  }
}
```

In this example:
- Messages to `+15551111111` are stored with `user_id="alice"`.
- Messages to `+15552222222` have no `user_id` (the field is null).

### Filtering by user

**CLI:**
```bash
# Show only messages for alice
rex msg receive --channel sms --user alice

# Combine with account filter
rex msg receive --channel sms --user alice --account-id personal
```

**Python API:**
```python
sms = get_sms_service()
messages = sms.receive(limit=10, user_id="alice")
```

### Behavior when not configured

When `owner_user_id` is not set on an account (or is `null`), inbound messages
are still stored and retrievable — they simply have no `user_id` attached.
Filtering by `--user` will exclude these messages (only messages explicitly
tagged with the requested user are returned).

### Schema migration

Existing inbound SMS databases created before this feature are migrated
automatically. The store detects the missing `user_id` column via
`PRAGMA table_info` and runs an idempotent `ALTER TABLE` at startup. No
data is lost.

## Inbound SMS Webhook

### Hosting

The inbound SMS webhook is hosted by `flask_proxy.py` (the main dashboard/proxy Flask app). At startup, the app calls `register_inbound_sms_webhook()` which:

1. Reads `messaging.inbound` from `config/rex_config.json`
2. If `enabled` is `true`, resolves the Twilio auth token via `CredentialManager`
3. Initializes the SQLite inbound store
4. Creates and registers the webhook Flask blueprint with rate limiting
5. The endpoint becomes available at `POST /webhooks/twilio/sms`

If `enabled` is `false` (the default), no route is registered and no network exposure is added.

### How It Works

1. Twilio sends a POST request to `/webhooks/twilio/sms` when an SMS is received
2. The webhook verifies the request signature using HMAC-SHA1 (stdlib `hmac` + `hashlib`)
3. The inbound message is routed to the correct account by matching the `To` number
4. The message is persisted to a local SQLite database
5. An empty TwiML `<Response/>` is returned to Twilio

### Security

- **Signature verification**: All requests are verified using Twilio's HMAC-SHA1 algorithm. Requests with invalid or missing signatures are rejected with HTTP 403.
- **No secrets logged**: Auth tokens are never logged. Phone numbers and message bodies are logged at DEBUG level only.
- **No bypass for localhost**: Signature verification applies equally to all source addresses.
- **Rate limiting**: The webhook endpoint is rate-limited via Flask-Limiter (default: `120 per minute`). Configurable via `messaging.inbound.rate_limit`. Requests exceeding the limit receive HTTP 429.

### Rate Limiting

The webhook rate limit is configurable via `messaging.inbound.rate_limit` in `config/rex_config.json`:

```json
{
  "messaging": {
    "inbound": {
      "enabled": true,
      "rate_limit": "120 per minute"
    }
  }
}
```

The format follows Flask-Limiter syntax (e.g., `"60 per minute"`, `"10 per second"`). If Flask-Limiter is not installed, rate limiting is skipped gracefully.

### Reverse Proxy Requirements

When running behind a reverse proxy (nginx, Caddy, Cloudflare Tunnel, etc.), Twilio signature verification requires that `request.url` matches the externally visible webhook URL exactly. To ensure this:

1. Configure Werkzeug `ProxyFix` middleware or set trusted proxy headers so Flask reconstructs the correct scheme, host, and path.
2. Set `REX_TRUSTED_PROXIES` to include your proxy's IP addresses (comma-separated, default: `127.0.0.1,::1`).
3. The URL configured in the Twilio console (e.g., `https://yourdomain.example.com/webhooks/twilio/sms`) must match the URL Flask sees after proxy header processing.

If signature verification fails behind a proxy, the most common cause is a scheme mismatch (`http` vs `https`) or a missing `X-Forwarded-Proto` header.

### Doctor Validation

Run `python scripts/doctor.py` to validate inbound webhook readiness:

- If inbound is disabled: reports PASS (expected default).
- If inbound is enabled and auth token is resolved: reports PASS.
- If inbound is enabled but auth token is missing: reports WARN with credential ref hint.

### Local Dev Simulation

To test inbound SMS locally without a real Twilio account:

1. Enable inbound in config:
   ```json
   {
     "messaging": {
       "inbound": {
         "enabled": true,
         "auth_token_ref": "twilio:inbound"
       }
     }
   }
   ```

2. Set a test auth token via `config/credentials.json`:
   ```json
   {
     "credentials": {
       "twilio:inbound": "test_token_for_dev"
     }
   }
   ```

   Or via environment variable:
   ```bash
   export REX_TWILIO_INBOUND="test_token_for_dev"
   ```

3. Start the Flask proxy app:
   ```bash
   python flask_proxy.py
   ```
   You should see a log line: `Inbound SMS webhook registered at /webhooks/twilio/sms`.

4. Send a test POST:
   ```bash
   # Compute a valid signature for the test token and send
   curl -X POST http://localhost:5000/webhooks/twilio/sms \
     -d "MessageSid=SM0001&From=+15559999999&To=+15551111111&Body=Test"
   ```

   Note: In production, signature verification must remain enabled. The test above will return 403 unless a valid signature is provided. For local dev testing with the test client, use the test helpers in `tests/test_inbound_webhook_wiring.py`.

5. Check received messages:
   ```bash
   rex msg receive --channel sms
   ```

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

# Test inbound webhook
pytest tests/test_inbound_webhook.py -v

# Test inbound store
pytest tests/test_inbound_store.py -v

# Test Twilio signature verification
pytest tests/test_twilio_signature.py -v

# Test SMS + inbound store integration
pytest tests/test_sms_inbound_integration.py -v

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
7. **Webhook Signature Verification**: Inbound webhook validates Twilio HMAC-SHA1 signatures using constant-time comparison

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
- `SMSService(mock_file: Path = None, from_number: str = None, backend=None, raw_config=None, inbound_store=None)`

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

### InboundSmsStore

- SQLite-backed persistence for webhook-received inbound messages
- `write(record)` - Store an inbound message
- `query_recent(limit, user_id, account_id)` - Query recent messages with optional filters
- `count(user_id)` - Count stored messages
- `cleanup_old()` - Remove records older than retention period

### Inbound Webhook Blueprint

- `POST /webhooks/twilio/sms` - Receive inbound SMS from Twilio
- Validates Twilio signature (HMAC-SHA1)
- Routes by `To` number to matching account
- Returns empty TwiML `<Response/>`

## What Remains Stubbed

- Telegram, Discord, WhatsApp channels (not implemented)
- MMS / media messages
- From-number-based user mapping (inbound caller -> user; currently only To-number -> account owner is implemented)

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
- If using inbound webhook: check that `messaging.inbound.enabled` is `true` and that the webhook endpoint is receiving POST requests

### Inbound webhook returning 403

- Verify that the `auth_token_ref` credential is set correctly
- Ensure the Twilio auth token matches the one configured in your Twilio account
- Check that the webhook URL configured in Twilio matches the URL your server is listening on

## Support

For questions or issues:
- Check existing tests in `tests/test_messaging_backends.py` and `tests/test_messaging_service.py`
- Review code in `rex/messaging_service.py` and `rex/messaging_backends/`
- Open an issue on GitHub
