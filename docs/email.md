# Email Integration

The AskRex Assistant includes email triage functionality that allows Rex to read, categorize, and process emails automatically. This enables Rex to keep you informed of important communications and trigger workflows based on email content.

## Implementation Status

**Status: Beta**

The email integration supports two modes:

| Mode | Read | Send | When |
|------|------|------|------|
| **Stub** (default) | JSON fixture | Logged no-op | No accounts configured |
| **Real** | IMAP4-SSL | SMTP (STARTTLS/SMTPS) | Accounts configured with credentials |

Multi-account support is included. Default mode is stub; real backends activate when accounts are configured in `config/rex_config.json` and credentials are available.

## Overview

The email service provides:
- Read and send email (IMAP4-SSL read + SMTP send when configured)
- Automatic categorization (important, promo, social, newsletter, general)
- Email summarization
- Multi-account support with explicit, default, and fallback routing
- Integration with the scheduler for automated checking
- Event publishing for reactive workflows
- Notification email channel uses real SMTP when configured

## Architecture

The email service consists of two main components:

1. **EmailSummary**: A Pydantic model representing an email summary
2. **EmailService**: The service class that handles email operations

## EmailSummary Model

An `EmailSummary` contains the following fields:

```python
class EmailSummary(BaseModel):
    id: str                        # Unique email identifier
    from_addr: str                 # Sender email address
    subject: str                   # Email subject line
    snippet: str                   # Brief preview of body
    received_at: datetime          # When received
    labels: list[str] = []         # Labels/tags (e.g., 'unread', 'important')
    importance_score: float = 0.5  # Score from 0.0 (low) to 1.0 (high)
    category: Optional[str] = None # Categorization result
```

## Using the Email Service

### Per-User Ownership (required)

Every operation that reads or mutates email data requires an explicit,
validated `user_id`. Missing, blank, malformed, or traversal-style
identities fail closed (`PermissionError`) **before** any account or
credential lookup. There is no fallback to `default`, to the active user,
to the global default account, or to another user's backend or credential.

Single-user setups select the legacy profile explicitly with
`user_id="default"` (or `rex identify --user default` / `--user default`
on the CLI).

### Getting the Email Service Instance

```python
from rex.email_service import get_email_service

email_service = get_email_service()
```

The returned service enforces per-user account ownership internally;
backends are resolved lazily per validated user and authorized account,
and a backend resolved for one user is never reused for another.

### Connecting to Email

```python
# Connect the requesting user's authorized backend
if email_service.connect("default"):
    print("Connected to email service")
else:
    print("Failed to connect")
```

### Fetching Unread Emails

```python
# Fetch up to 10 unread emails for one user
unread = email_service.fetch_unread(limit=10, user_id="default")

for email in unread:
    print(f"{email.id}: {email.subject}")
    print(f"From: {email.from_addr}")
    print(f"Snippet: {email.snippet}")
```

### Categorizing Emails

The email service can automatically categorize emails based on keywords and patterns:

```python
category = email_service.categorize(email)
print(f"Category: {category}")
```

**Categories:**

- `important`: Urgent or high-priority emails
- `promo`: Promotional and marketing emails
- `social`: Social media notifications
- `newsletter`: Newsletter subscriptions
- `general`: Everything else

**Categorization Rules:**

1. **Promo**: Contains keywords like "sale", "discount", "offer", "deal"
2. **Social**: From social media domains or contains "liked", "commented"
3. **Newsletter**: Contains "unsubscribe" or "newsletter"
4. **Important**: Contains "urgent", "important", "action required" or high importance score (≥ 0.8)
5. **General**: Default category

### Marking Emails as Read

```python
email_service.mark_as_read('email-123', user_id="default")
```

### Summarizing Emails

```python
summary = email_service.summarize('email-123', user_id="default")
print(summary)
```

## CLI Commands

Every `rex email` subcommand runs as one validated user. The user comes
from `--user <id>`, the `rex identify` session, or `runtime.active_user`
in config; without one the command fails closed with instructions.

### Fetch Unread Emails

Display unread emails with categorization:

```bash
rex email unread --user james
rex email unread --user james --limit 5
rex email unread --user james -v  # Verbose output with importance scores
```

## Configuration

### Mock Data

The email service currently uses mock data from `data/mock_emails.json`. This file contains sample emails for testing.

**Example mock_emails.json:**

```json
[
  {
    "id": "email-001",
    "from_addr": "boss@company.com",
    "subject": "URGENT: Project deadline tomorrow",
    "snippet": "Hi team, just a reminder that...",
    "received_at": "2026-01-28T09:30:00",
    "labels": ["unread", "important"],
    "importance_score": 0.95
  },
  {
    "id": "email-002",
    "from_addr": "newsletter@techblog.com",
    "subject": "Weekly Tech Digest",
    "snippet": "This week's top stories... Click to unsubscribe.",
    "received_at": "2026-01-28T08:00:00",
    "labels": ["unread"],
    "importance_score": 0.3
  }
]
```

### Credentials

Email credentials should be configured in the credential manager:

```python
from rex.credentials import get_credential_manager

cred_manager = get_credential_manager()

# Future: Real IMAP credentials
# cred_manager.set_credential('email', {
#     'username': 'your-email@example.com',
#     'password': 'your-password',  # pragma: allowlist secret
#     'imap_server': 'imap.gmail.com',
#     'imap_port': 993
# })
```

**For Gmail:**
- Username: Your Gmail address
- Password: App-specific password (not your regular password)
- IMAP Server: `imap.gmail.com`
- IMAP Port: `993`

**Note**: App-specific passwords are required when 2FA is enabled.

## Scheduled Email Checking

The email service integrates with the scheduler to check for new emails automatically:

```python
from rex.integrations import initialize_scheduler_system

# Initialize scheduler (includes email check job)
initialize_scheduler_system(start_scheduler=True)
```

This creates a job that:
- Runs every 10 minutes (configurable)
- Iterates the explicitly configured account owners, each in an isolated
  owner context (one owner's failure never falls through to another)
- Fetches and categorizes each owner's unread email as that owner
- Publishes a per-owner `email.unread.user.<user_id>` event with the
  message payload, plus a safe envelope (count/user only) on the shared
  `email.unread` topic

If no owners are configured, the job does not touch real email (fail
closed).

## Event Integration

Private email fields (subjects, senders, snippets) are published only on
user-scoped topics — `email.unread.user.<user_id>` — so a consumer acting
for one user cannot observe another user's payload. The shared
`email.unread` topic carries only `{count, user_id, account_id}`.

Subscribe to a user's email events to trigger actions:

```python
from rex.event_bus import get_event_bus

event_bus = get_event_bus()

def handle_new_emails(event):
    emails = event.payload['emails']
    for email in emails:
        if email['category'] == 'important':
            print(f"⚠️  Important: {email['subject']}")

event_bus.subscribe('email.unread.user.james', handle_new_emails)
```

## Example: Email Notification Workflow

```python
from rex.event_bus import get_event_bus
from rex.integrations import initialize_scheduler_system

# Initialize system
initialize_scheduler_system(start_scheduler=True)
event_bus = get_event_bus()

# Define handler
def email_notifier(event):
    """Notify user of important emails."""
    emails = event.payload['emails']

    # Filter important emails
    important = [e for e in emails if e.get('category') == 'important']

    if important:
        print(f"\n🔔 You have {len(important)} important email(s):\n")
        for email in important:
            print(f"  • {email['subject']}")
            print(f"    From: {email['from_addr']}")
            print()

# Subscribe to this user's email events (private payloads are user-scoped)
event_bus.subscribe('email.unread.user.james', email_notifier)

print("Email notification system active")
```

## Example: Auto-Reply Detection

```python
from rex.email_service import get_email_service

email_service = get_email_service()
email_service.connect("james")

unread = email_service.fetch_unread(user_id="james")

for email in unread:
    # Check for vacation auto-replies
    if 'out of office' in email.subject.lower() or 'vacation' in email.snippet.lower():
        print(f"Auto-reply detected from {email.from_addr}")
        # Mark as read automatically
        email_service.mark_as_read(email.id, user_id="james")
```

## Multi-Account Configuration

Rex supports multiple email accounts per user. Account **definitions**
(non-secret connection metadata) live in `config/rex_config.json` under the
`email` key; account **ownership** is assigned per user under the `users`
key.

### Ownership Model

- `email.accounts` is the canonical list of account definitions
  (host/port/address/`credential_ref`). It grants no access by itself.
- `users.{user_id}.email_accounts` is the authoritative authorization map:
  a user may use only the accounts assigned to them there, and each
  `account_id` must reference a real `email.accounts` definition.
- `users.{user_id}.default_email_account_id` selects that user's default
  account (it must be one of their own accounts).
- Credential lookup happens only after ownership validation, and only with
  the authorized account definition's own `credential_ref`.
- Unauthorized and nonexistent accounts are indistinguishable to callers.

### Legacy Accounts (single-user configs)

`email.accounts` entries not assigned to any user belong **only** to the
distinct `default` profile. They are never shared with, or silently
reassigned to, named users — a named user gets email access only through an
explicit `users.{user_id}.email_accounts` assignment. The legacy
`email.default_account_id` applies only to the `default` profile, and the
legacy global `GMAIL_ACCESS_TOKEN` environment behaviour (GUI inbox) is
likewise `default`-profile-only.

To assign an existing account to a named user, add (no secrets involved —
credentials stay in the credential manager):

```json
{
  "users": {
    "james": {
      "email_accounts": [
        {"account_id": "personal", "backend": "imap", "credentials_key": "email:personal"}
      ],
      "default_email_account_id": "personal"
    }
  }
}
```

### Configuration Format

```json
{
  "email": {
    "default_account_id": "personal",
    "accounts": [
      {
        "id": "personal",
        "label": "Personal Gmail",
        "address": "you@gmail.com",
        "imap": {
          "host": "imap.gmail.com",
          "port": 993,
          "ssl": true
        },
        "smtp": {
          "host": "smtp.gmail.com",
          "port": 587,
          "starttls": true
        },
        "credential_ref": "email:personal"
      },
      {
        "id": "work",
        "label": "Work Outlook",
        "address": "you@company.com",
        "imap": {
          "host": "outlook.office365.com",
          "port": 993,
          "ssl": true
        },
        "smtp": {
          "host": "smtp.office365.com",
          "port": 587,
          "starttls": true
        },
        "credential_ref": "email:work"
      }
    ]
  }
}
```

### Config Keys

| Key | Description |
|-----|-------------|
| `email.default_account_id` | Account ID used when none is explicitly specified |
| `email.accounts[].id` | Unique identifier for the account |
| `email.accounts[].label` | Human-friendly display name |
| `email.accounts[].address` | Email address |
| `email.accounts[].imap` | IMAP server settings (`host`, `port`, `ssl`) |
| `email.accounts[].smtp` | SMTP server settings (`host`, `port`, `starttls`) |
| `email.accounts[].credential_ref` | Key for credential lookup (see below) |

### Credentials

Credentials are stored **outside** the config file. Each account references a `credential_ref` that maps to an environment variable or `config/credentials.json` entry via the `CredentialManager`.

**Setting credentials via environment variable:**

```bash
# Format: username:password (or just password — address is used as username)
export EMAIL_PERSONAL="you@gmail.com:your_app_password"
export EMAIL_WORK="you@company.com:your_app_password"
```

Then configure the credential mapping to point `email:personal` to `EMAIL_PERSONAL`, etc.

**Important:** Use app-specific passwords for services with 2FA (e.g., Gmail).

### Account Selection (Routing)

Accounts are always selected **within the requesting user's authorized
set**, using this precedence:

1. **Explicit** `account_id` argument (after ownership validation — a
   foreign or nonexistent account is rejected with a generic error)
2. That user's `users.{user_id}.default_email_account_id`
3. The legacy `email.default_account_id` — only for the explicit `default`
   profile
4. The first account assigned to that user (deterministic, config order)
5. No authorized account: the operation fails closed / reports
   not-configured — never another user's account

If no accounts are configured anywhere, the stub backend is used
automatically for offline development.

### CLI Commands

```bash
# List your own configured accounts (only yours are shown)
rex email accounts list --user james

# Set your default account (must be one of your own accounts;
# never changes another user's routing)
rex email accounts set-active --account-id work --user james

# Test connectivity for one of your accounts
rex email test-connection --account-id personal --user james

# Send an email via a specific account you own
rex email send --account-id work --to recipient@example.com \
  --subject "Hello" --body "Message body" --user james
```

### Notification Account Selection

Notifications that use the email channel must name the owning user via the
`email_user_id` metadata key (delivery is skipped otherwise — it never
falls back to another user's account), and may pick one of that user's
accounts with `email_account_id`:

```python
notification = NotificationRequest(
    priority="urgent",
    title="Alert",
    body="Something happened",
    channel_preferences=["email"],
    metadata={
        "to_email": "admin@example.com",
        "email_user_id": "james",     # Owner whose account sends the email
        "email_account_id": "work",   # One of that user's accounts (optional)
    },
)
```

If `email_account_id` is not set, the owner's per-user default routing
rules apply.

## Security Considerations

1. **Credentials**: Never hardcode email credentials. Use the credential manager.
2. **App Passwords**: Use app-specific passwords for Gmail and services with 2FA
3. **Permissions**: Request minimal IMAP permissions (read-only if possible)
4. **Logging**: Be careful not to log sensitive email content
5. **Data Retention**: Don't store email content longer than necessary

## Best Practices

1. **Use mock data for testing**: Keep real email credentials separate
2. **Handle connection failures gracefully**: IMAP connections can be unreliable
3. **Rate limit email checks**: Don't check too frequently (10 minutes is reasonable)
4. **Categorize before acting**: Use categorization to filter emails
5. **Mark as read selectively**: Only mark emails as read after processing
6. **Subscribe to events**: Use the event bus for reactive workflows
7. **Monitor for important emails**: Set up alerts for high-priority messages

## Troubleshooting

### Connection Issues

```python
# Check if credentials are configured
from rex.credentials import get_credential_manager
cred_manager = get_credential_manager()
email_creds = cred_manager.get_credential('email')
if not email_creds:
    print("Email credentials not configured")
```

### No Unread Emails

If `fetch_unread()` returns empty:
- Check that mock data file exists: `data/mock_emails.json`
- Verify emails have `"unread"` in their labels
- Check email service connection status

### Categorization Issues

If emails are mis-categorized:
- Adjust importance scores in mock data
- Add custom keywords to categorization rules
- Override the `categorize()` method for custom logic

## Future Enhancements

Planned improvements for email integration:

- OAuth2 authentication (currently uses app passwords)
- Email threading and conversation tracking
- Attachment handling
- Advanced filtering and rules
- Email templates
- Batch operations
- Search functionality
