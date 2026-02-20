# Notification System

## Current Implementation

**Status: Beta**

The notification system provides priority routing, digest logic, quiet hours, and escalation management. The email channel now uses real SMTP delivery when configured; other channels remain placeholders.

| Channel | Status | Details |
|---------|--------|---------|
| **Dashboard** | Placeholder | Logs to console; no web UI yet. |
| **Email** | Beta (real when configured) | Uses `EmailService.send()` with real SMTP when a backend is configured. Falls back to logged stub in offline/dev mode. Supports multi-account selection via `email_account_id` metadata. |
| **SMS** | Placeholder | Delegates to SMS service, which requires real Twilio credentials to deliver. |
| **Home Assistant TTS** | Stub | Logs intent; no Home Assistant API call. |

All channel dispatch is simulated in tests — no external calls are made.

---

Rex's multi-channel notification system provides intelligent routing, priority-based delivery, quiet hours management, and automatic escalation for urgent alerts. It integrates with the event bus to automatically create notifications for important events.

## Overview

The notification system consists of:

- **NotificationRequest**: A model for representing notifications with priority and channel preferences
- **Notifier**: Dispatches notifications to multiple channels based on priority
- **EscalationManager**: Manages quiet hours, do-not-disturb mode, and escalation rules

## Architecture

### Notification Priorities

Rex supports three priority levels:

1. **Urgent**: Sent immediately to ALL preferred channels, bypasses quiet hours
2. **Normal**: Sent to the first available channel, respects quiet hours
3. **Digest**: Queued and sent as a batch summary at configured intervals

### Supported Channels

- **Dashboard**: In-app notifications (placeholder)
- **Email**: Send notification via email service
- **SMS**: Send notification via SMS service
- **Home Assistant TTS**: Text-to-speech announcements (stub)
- **Other**: Extensible for future channels

### Quiet Hours

Non-urgent notifications are suppressed during configured quiet hours (default: 22:00-07:00). Urgent notifications always go through.

### Escalation

Urgent notifications are tracked for acknowledgement. If not acknowledged within a configured delay (default: 5 minutes), they're automatically resent via an alternative channel.

## Creating Notifications

### Python API

```python
from rex.notification import NotificationRequest, get_notifier

notifier = get_notifier()

# Create and send an urgent notification
notification = NotificationRequest(
    priority="urgent",
    title="Critical System Alert",
    body="Database connection lost. Immediate attention required.",
    channel_preferences=["sms", "email", "dashboard"]
)

notifier.send(notification)
```

### CLI

```bash
# Send an urgent notification
rex notify send --priority urgent \
    --title "Critical Alert" \
    --body "System failure detected" \
    --channels sms,email

# Send a normal notification
rex notify send --priority normal \
    --title "Build Complete" \
    --body "Your build finished successfully"

# Send a digest notification (queued)
rex notify send --priority digest \
    --title "Daily Update" \
    --body "3 new items in your inbox"
```

## Priority Routing

### Urgent Notifications

Urgent notifications are sent to **all** preferred channels immediately:

```python
notification = NotificationRequest(
    priority="urgent",
    title="Server Down",
    body="Production server is not responding",
    channel_preferences=["sms", "email", "dashboard"]
)

notifier.send(notification)
# Sends to: SMS, Email, AND Dashboard
```

**Behavior:**
- Bypasses quiet hours
- Bypasses do-not-disturb mode
- Sent to ALL specified channels
- Tracked for escalation

### Normal Notifications

Normal notifications are sent to the **first available** channel:

```python
notification = NotificationRequest(
    priority="normal",
    title="Deployment Success",
    body="Application deployed to staging",
    channel_preferences=["email", "dashboard"]
)

notifier.send(notification)
# Sends to: Email only (first channel)
```

**Behavior:**
- Respects quiet hours (suppressed if active)
- Respects do-not-disturb mode
- Tries first channel, falls back to next if unavailable
- Not tracked for escalation

### Digest Notifications

Digest notifications are queued and sent as a summary:

```python
# Queue multiple digest notifications
for item in daily_items:
    notification = NotificationRequest(
        priority="digest",
        title=item.title,
        body=item.summary,
        channel_preferences=["email"]
    )
    notifier.send(notification)

# Flush manually or wait for scheduled flush
notifier.flush_digests()
```

**Behavior:**
- Added to per-channel queues
- Not sent immediately
- Flushed on schedule (default: hourly)
- Combined into a single summary message

## Digest Management

### Viewing Queued Digests

**Python:**
```python
digests = notifier.list_digests()

for channel, notifications in digests.items():
    print(f"Channel {channel}: {len(notifications)} queued")
    for notif in notifications:
        print(f"  - {notif['title']}")
```

**CLI:**
```bash
# List all queued digests
rex notify list-digests
```

### Flushing Digests

**Python:**
```python
# Flush all channels
count = notifier.flush_digests()

# Flush specific channel
count = notifier.flush_digests(channel="email")
```

**CLI:**
```bash
# Flush all digest queues
rex notify flush-digests

# Flush specific channel
rex notify flush-digests --channel email
```

### Automatic Flushing

Digest queues are automatically flushed by the scheduler:

- **Job**: "Flush notification digests"
- **Interval**: Every 3600 seconds (1 hour)
- **Handler**: `notifier.flush_digests()`

## Escalation Management

### Tracking Notifications

Urgent notifications are automatically tracked for escalation:

```python
from rex.notification import get_escalation_manager

escalation_manager = get_escalation_manager()

# Send urgent notification
notification = NotificationRequest(priority="urgent", ...)
notifier.send(notification)

# Track for escalation
escalation_manager.track_notification(
    notification,
    next_channel="email"  # Escalate to email if not acknowledged
)
```

### Acknowledging Notifications

**Python:**
```python
# User acknowledges notification
acknowledged = escalation_manager.acknowledge("notif_abc123")

if acknowledged:
    print("Notification acknowledged, escalation cancelled")
```

**CLI:**
```bash
# Acknowledge a notification
rex notify ack notif_abc123
```

### Automatic Escalation

The escalation manager checks pending notifications every 5 minutes:

- **Job**: "Check notification escalations"
- **Interval**: Every 300 seconds (5 minutes)
- **Behavior**: Resends unacknowledged urgent notifications via alternative channel

**Configuration:**
```python
escalation_manager = EscalationManager(
    escalation_delay_minutes=10  # Wait 10 minutes before escalating
)
```

## Quiet Hours

### Configuration

```python
from datetime import time

escalation_manager = EscalationManager(
    quiet_hours_start=time(22, 0),  # 10:00 PM
    quiet_hours_end=time(7, 0)      # 7:00 AM
)
```

### Behavior

During quiet hours:
- **Urgent** notifications: Still delivered
- **Normal** notifications: Suppressed
- **Digest** notifications: Suppressed (queued until flushed)

### Checking Quiet Hours

```python
from datetime import datetime, timezone

dt = datetime.now(timezone.utc)
is_quiet = escalation_manager.is_quiet_hours(dt)

if is_quiet:
    print("Currently in quiet hours")
```

## Do Not Disturb

### Enabling DND

```python
# Enable do-not-disturb
escalation_manager.set_dnd(True)

# Disable do-not-disturb
escalation_manager.set_dnd(False)
```

### Behavior

When DND is enabled:
- **Urgent** notifications: Still delivered
- **Normal** notifications: Suppressed
- **Digest** notifications: Suppressed

DND overrides quiet hours (both suppress non-urgent notifications).

## Event Bus Integration

The notifier automatically subscribes to event bus events and creates notifications:

### Email Events

Listens to `email.unread` events:

```python
# High importance emails trigger urgent notifications
if email.importance_score >= 0.8:
    notification = NotificationRequest(
        priority="urgent",
        title="High Importance Email",
        body=f"From: {email.from_addr}\nSubject: {email.subject}",
        channel_preferences=["sms", "email", "dashboard"]
    )
    notifier.send(notification)
```

### Calendar Events

Listens to `calendar.update` events:

```python
# Events starting within 15 minutes trigger notifications
if 0 < time_until_event <= 900:  # 15 minutes
    notification = NotificationRequest(
        priority="normal",
        title="Upcoming Calendar Event",
        body=f"{event.title} starts at {event.start_time}",
        channel_preferences=["dashboard", "ha_tts"]
    )
    notifier.send(notification)
```

### Manual Subscription Setup

```python
notifier.setup_event_subscriptions()
# Subscribes to email.unread and calendar.update
```

## Channel Implementation

### Dashboard

Currently a placeholder that logs to console. Future: Web dashboard UI.

```python
def _send_to_dashboard(self, notification):
    logger.info(f"[DASHBOARD] {notification.title}: {notification.body}")
```

### Email

Integrates with email service:

```python
def _send_to_email(self, notification):
    email_service = get_email_service()
    # email_service.send(to=user_email, subject=..., body=...)
```

### SMS

Integrates with SMS service:

```python
def _send_to_sms(self, notification):
    sms_service = get_sms_service()
    to_number = notification.metadata.get("to_number", "+15555551234")
    message = Message(
        channel="sms",
        to=to_number,
        from_=sms_service.from_number,
        body=f"{notification.title}: {notification.body}"
    )
    sms_service.send(message)
```

### Home Assistant TTS

Stub for text-to-speech announcements:

```python
def _send_to_ha_tts(self, notification):
    logger.info(f"[HA_TTS] Would announce: {notification.title}")
    # Future: Call Home Assistant TTS API
```

## Persistence

### Digest Queues

Digest queues are persisted to disk:

**Location**: `data/notifications/digests.json`

**Format**:
```json
{
  "email": {
    "notifications": [],
    "last_flush_at": "2024-01-15T10:30:00Z"
  },
  "dashboard": {
    "notifications": [],
    "last_flush_at": "2024-01-15T11:00:00Z"
  }
}
```

Queues are automatically:
- Loaded on initialization
- Saved when notifications are queued
- Updated when flushed

## Testing

### Running Tests

```bash
# Test notification system
pytest tests/test_notification.py -v

# Test escalation
pytest tests/test_escalation.py -v

# Test CLI commands
pytest tests/test_cli_messaging_notification.py -k notify -v
```

### Mock Behavior

In tests, channel dispatch is mocked:
- No actual SMS sent
- No actual emails sent
- All operations are simulated

## Examples

### Simple Urgent Alert

```python
from rex.notification import NotificationRequest, get_notifier

notifier = get_notifier()

notification = NotificationRequest(
    priority="urgent",
    title="System Alert",
    body="High CPU usage detected",
    channel_preferences=["sms", "email"]
)

notifier.send(notification)
```

### Daily Digest Workflow

```python
# Throughout the day, queue updates
for update in daily_updates:
    notif = NotificationRequest(
        priority="digest",
        title=update.title,
        body=update.description,
        channel_preferences=["email"]
    )
    notifier.send(notif)

# Later (or via scheduler), flush
notifier.flush_digests()
# Sends single email with all updates combined
```

### Escalation Flow

```python
from rex.notification import (
    NotificationRequest,
    get_notifier,
    get_escalation_manager
)

notifier = get_notifier()
escalation_mgr = get_escalation_manager()

# Send urgent notification
notif = NotificationRequest(
    priority="urgent",
    title="Critical Alert",
    body="Action required",
    channel_preferences=["dashboard", "email"]
)

notifier.send(notif)

# Track for escalation
escalation_mgr.track_notification(notif, next_channel="sms")

# If not acknowledged within 5 minutes:
# Scheduler runs check_escalations()
# Notification is resent via SMS
```

## Integration with Services

### Initializing Services

```python
from rex.services import initialize_services

services = initialize_services(
    notifications_path="data/notifications",
    # ... other params
)

# Services now include:
# - services.notifier
# - services.escalation_manager
```

### Scheduler Integration

Default jobs are registered automatically:

1. **Flush notification digests** - Every hour
2. **Check notification escalations** - Every 5 minutes

View with:
```bash
rex scheduler list
```

## API Reference

### NotificationRequest

**Fields:**
- `id: str` - Unique identifier
- `priority: Literal["urgent", "normal", "digest"]` - Priority level
- `title: str` - Notification title
- `body: str` - Notification body
- `timestamp: datetime` - When created (UTC)
- `channel_preferences: list[str]` - Ordered list of channels
- `metadata: dict` - Additional data
- `acknowledged_at: Optional[datetime]` - Acknowledgement timestamp

### Notifier

**Constructor:**
- `Notifier(digest_interval_seconds=3600, storage_path=None)`

**Methods:**
- `send(notification: NotificationRequest)` - Send notification
- `flush_digests(channel: str = None) -> int` - Flush digest queues
- `list_digests() -> dict` - List queued digests
- `setup_event_subscriptions()` - Subscribe to event bus

### EscalationManager

**Constructor:**
- `EscalationManager(quiet_hours_start=time(22,0), quiet_hours_end=time(7,0), escalation_delay_minutes=5)`

**Methods:**
- `is_quiet_hours(dt: datetime = None) -> bool` - Check if in quiet hours
- `should_suppress(notification) -> bool` - Check if should suppress
- `track_notification(notification, next_channel)` - Track for escalation
- `acknowledge(notification_id) -> bool` - Acknowledge notification
- `check_escalations() -> list` - Check for due escalations
- `set_dnd(enabled: bool)` - Set do-not-disturb mode

## Troubleshooting

### Notifications not being sent

1. Check priority level and quiet hours
2. Verify DND is not enabled
3. Check channel is supported
4. Review logs for dispatch errors

### Digest not flushing

1. Check scheduler is running: `rex scheduler list`
2. Manually flush: `rex notify flush-digests`
3. Verify digest queue has items: `rex notify list-digests`

### Escalation not triggering

1. Ensure notification priority is "urgent"
2. Check escalation delay has passed
3. Verify notification was tracked
4. Check scheduler job is running

## Future Enhancements

- **Notification Templates**: Pre-defined notification formats
- **User Preferences**: Per-user channel preferences
- **Rich Notifications**: Images, buttons, actions
- **Delivery Tracking**: Track read/delivered status
- **Notification History**: Store and search past notifications
- **Custom Escalation Rules**: Configurable escalation chains
- **Snooze**: Temporarily suppress notifications
- **Channel Fallback**: Automatic retry on different channels

## Support

For questions or issues:
- Check tests in `tests/test_notification.py`
- Review code in `rex/notification.py`
- Open an issue on GitHub
