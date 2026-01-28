# Email Integration

The Rex AI Assistant includes email triage functionality that allows Rex to read, categorize, and process emails automatically. This enables Rex to keep you informed of important communications and trigger workflows based on email content.

## Overview

The email service provides:
- Read-only access to unread emails
- Automatic categorization (important, promo, social, newsletter, general)
- Email summarization
- Integration with the scheduler for automated checking
- Event publishing for reactive workflows

## Current Implementation

**Note**: The current implementation uses mock/stub functionality for testing. Real IMAP/SMTP integration will be added in future releases.

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

### Getting the Email Service Instance

```python
from rex.email_service import get_email_service

email_service = get_email_service()
```

### Connecting to Email

```python
# Connect (loads credentials and initializes)
if email_service.connect():
    print("Connected to email service")
else:
    print("Failed to connect")
```

### Fetching Unread Emails

```python
# Fetch up to 10 unread emails
unread = email_service.fetch_unread(limit=10)

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
email_service.mark_as_read(email_id='email-123')
```

### Summarizing Emails

```python
summary = email_service.summarize(email_id='email-123')
print(summary)
```

## CLI Commands

### Fetch Unread Emails

Display unread emails with categorization:

```bash
rex email unread
rex email unread --limit 5
rex email unread -v  # Verbose output with importance scores
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
#     'password': 'your-password',
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
- Fetches unread emails
- Categorizes them
- Publishes an `email.unread` event

## Event Integration

Subscribe to email events to trigger actions:

```python
from rex.event_bus import get_event_bus

event_bus = get_event_bus()

def handle_new_emails(event):
    emails = event.payload['emails']
    for email in emails:
        if email['category'] == 'important':
            print(f"⚠️  Important: {email['subject']}")

event_bus.subscribe('email.unread', handle_new_emails)
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

# Subscribe to email events
event_bus.subscribe('email.unread', email_notifier)

print("Email notification system active")
```

## Example: Auto-Reply Detection

```python
from rex.email_service import get_email_service

email_service = get_email_service()
email_service.connect()

unread = email_service.fetch_unread()

for email in unread:
    # Check for vacation auto-replies
    if 'out of office' in email.subject.lower() or 'vacation' in email.snippet.lower():
        print(f"Auto-reply detected from {email.from_addr}")
        # Mark as read automatically
        email_service.mark_as_read(email.id)
```

## Future: Real IMAP Integration

Future releases will include real IMAP support:

```python
class EmailService:
    def connect(self):
        # Connect to IMAP server
        import imaplib
        self.imap = imaplib.IMAP4_SSL(
            self.imap_server,
            self.imap_port
        )
        self.imap.login(username, password)
        self.imap.select('INBOX')

    def fetch_unread(self, limit=10):
        # Fetch unread emails via IMAP
        status, messages = self.imap.search(None, 'UNSEEN')
        # Parse and return EmailSummary objects
```

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

- Real IMAP/SMTP support (Gmail, Outlook, etc.)
- OAuth2 authentication
- Email sending capabilities
- Email threading and conversation tracking
- Attachment handling
- Advanced filtering and rules
- Email templates
- Batch operations
- Search functionality
- Multiple account support
