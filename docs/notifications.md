# Notification System

## Current Implementation

**Status: Beta (dashboard + email channels real)**

The notification system provides priority routing, digest logic, quiet hours, and escalation management. The dashboard channel persists notifications to a local SQLite store with API endpoints for retrieval. The email channel uses real SMTP delivery when configured. SMS channel delegates to the messaging service (real Twilio delivery when configured).

| Channel | Status | Details |
|---------|--------|---------|
| **Dashboard** | Real (local SQLite store) | Persists to `data/dashboard_notifications.db`; read-only API endpoints at `/api/notifications`. Authenticated, localhost-only by default. |
| **Email** | Beta (real when configured) | Uses `EmailService.send()` with real SMTP when a backend is configured. Falls back to logged stub in offline/dev mode. Supports multi-account selection via `email_account_id` metadata. |
| **SMS** | Beta (real when configured) | Delegates to SMS service backed by Twilio when configured; stub mode otherwise. Requires `to_number` in notification metadata. |
| **Home Assistant TTS** | Stub | Logs intent; no Home Assistant API call. |

All channel dispatch is simulated in tests — no external calls are made.

---

Rex's multi-channel notification system provides intelligent routing, priority-based delivery, quiet hours management, and automatic escalation for urgent alerts. It integrates with the event bus to automatically create notifications for important events.

## Overview

The notification system consists of:

- **NotificationRequest**: A model for representing notifications with priority and channel preferences
- **Notifier**: Dispatches notifications to multiple channels based on priority
- **EscalationManager**: Manages quiet hours, do-not-disturb mode, and escalation rules
- **DashboardStore**: SQLite-backed local notification store for the dashboard channel

## Architecture

### Notification Priorities

Rex supports three priority levels:

1. **Urgent**: Sent immediately to ALL preferred channels, bypasses quiet hours
2. **Normal**: Sent to the first available channel, respects quiet hours
3. **Digest**: Queued and sent as a batch summary at configured intervals

### Supported Channels

- **Dashboard**: Persisted to local SQLite store, queryable via API
- **Email**: Send notification via email service
- **SMS**: Send notification via SMS service (Twilio when configured)
- **Home Assistant TTS**: Text-to-speech announcements (stub)
- **Other**: Extensible for future channels

### Quiet Hours

Non-urgent notifications are suppressed during configured quiet hours (default: 22:00-07:00). Urgent notifications always go through.

### Escalation

Urgent notifications are tracked for acknowledgement. If not acknowledged within a configured delay (default: 5 minutes), they're automatically resent via an alternative channel.

## Configuration

### Dashboard Store

Configure the dashboard notification store in `config/rex_config.json`:

```json
{
  "notifications": {
    "dashboard": {
      "store": {
        "type": "sqlite",
        "path": "data/dashboard_notifications.db",
        "retention_days": 30
      }
    }
  }
}
```

**Config keys:**
- `notifications.dashboard.store.type`: `"sqlite"` (only supported type currently)
- `notifications.dashboard.store.path`: Database file path (default: `data/dashboard_notifications.db`)
- `notifications.dashboard.store.retention_days`: Days to retain notifications (default: 30)
- `notifications.dashboard.store.cleanup_schedule`: Scheduler interval for retention cleanup (default: `"interval:86400"` — daily). Set to `null` to disable automatic cleanup.

### Retention Cleanup Scheduling

Dashboard notification cleanup runs as a scheduled background job.  Old
notifications (older than `retention_days`) are pruned automatically once per
day by default.

**How it works:**
- At startup, `wire_retention_cleanup()` registers a `dashboard_retention_cleanup`
  job in the global scheduler.
- The scheduler background thread runs the cleanup at the configured interval.
- Each run calls `DashboardStore.cleanup_old()`, which is idempotent — multiple
  runs on the same data are safe.
- If `cleanup_schedule` is `null`, no job is registered and no automatic cleanup
  occurs.

**To disable automatic cleanup**, set `cleanup_schedule` to `null` in config:

```json
{
  "notifications": {
    "dashboard": {
      "store": {
        "cleanup_schedule": null
      }
    }
  }
}
```

**To run cleanup manually** (one-shot via CLI):
```bash
rex scheduler run dashboard_retention_cleanup
```

### Security Notes

- The dashboard API requires authentication consistent with the existing dashboard auth approach
- Default binding is localhost only — not exposed to the network
- Notification data is stored locally, never transmitted externally by default
- The store does not log secrets; metadata is stored as JSON and may contain routing info

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
    channel_preferences=["sms", "email", "dashboard"],
    metadata={"to_number": "+15551234567", "user_id": "alice"},
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

# Send a normal notification (defaults to dashboard)
rex notify send --priority normal \
    --title "Build Complete" \
    --body "Your build finished successfully"

# Send as specific user
rex notify send --priority normal \
    --title "Update" \
    --body "Task completed" \
    --user alice

# Send a digest notification (queued)
rex notify send --priority digest \
    --title "Daily Update" \
    --body "3 new items in your inbox"
```

## Dashboard Notifications

### API Endpoints

The dashboard provides read-only API endpoints for notifications:

**List notifications:**
```
GET /api/notifications?limit=50&unread=true&priority=urgent&user_id=alice
```

Response:
```json
{
  "notifications": [
    {
      "id": "notif_abc123",
      "priority": "urgent",
      "title": "Alert",
      "body": "Something happened",
      "channel": "dashboard",
      "timestamp": "2025-01-15T10:30:00+00:00",
      "read": false,
      "user_id": "alice",
      "metadata": {}
    }
  ],
  "total": 1,
  "unread_count": 1
}
```

**Mark as read:**
```
POST /api/notifications/<id>/read
```

**Mark all as read:**
```
POST /api/notifications/read-all?user_id=alice
```

**Real-time stream (SSE):**
```
GET /api/notifications/stream
```

The stream sends:
- `event: init` with current `unread_count`
- `event: notification` for new notifications scoped to the authenticated user
- periodic keep-alive comments (`: keep-alive`)

All endpoints require authentication (see dashboard auth docs).

### SSE Authentication and token query parameter safety

`/api/notifications/stream` uses normal dashboard auth (Authorization header, `X-Dashboard-Token`, or session cookie).

Because browser `EventSource` cannot set custom headers, a `token` query parameter is accepted only for safer contexts:
- HTTPS requests (or localhost loopback in development)
- same-origin usage patterns

Security notes:
- Prefer cookie/session auth when possible.
- Query tokens can appear in browser history and intermediary logs; avoid sharing stream URLs.
- Do not use query-token auth over plain HTTP on non-localhost hosts.

### Python API

```python
from rex.dashboard_store import get_dashboard_store

store = get_dashboard_store()

# Query recent
notifications = store.query_recent(limit=10, unread_only=True)

# Count unread
count = store.count_unread(user_id="alice")

# Mark as read
store.mark_as_read("notif_abc123")

# Cleanup old notifications
store.cleanup_old()
```

## Priority Routing

### Urgent Notifications

Urgent notifications are sent to **all** preferred channels immediately:

```python
notification = NotificationRequest(
    priority="urgent",
    title="Server Down",
    body="Production server is not responding",
    channel_preferences=["sms", "email", "dashboard"],
    metadata={"to_number": "+15551234567"},
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
    channel_preferences=["email", "dashboard"],
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
        channel_preferences=["email"],
    )
    notifier.send(notification)

# Flush manually or wait for scheduled flush
notifier.flush_digests()
```

## Digest Management

### CLI

```bash
# List all queued digests
rex notify list-digests

# Flush all digest queues
rex notify flush-digests

# Flush specific channel
rex notify flush-digests --channel email
```

## Escalation Management

### Acknowledging Notifications

```bash
# Acknowledge a notification
rex notify ack notif_abc123
```

### Automatic Escalation

The escalation manager checks pending notifications every 5 minutes:
- Resends unacknowledged urgent notifications via alternative channel

## Identity Integration

Notifications and messaging respect the resolved user identity:

- `--user` flag takes highest priority
- Falls back to session active user (set via `rex identify`)
- Falls back to `runtime.active_user` config

The `user_id` is stored in notification metadata and used to scope dashboard queries.

## Testing

### Running Tests

```bash
# Test notification system
pytest tests/test_notification.py -v

# Test dashboard store
pytest tests/test_dashboard_store.py -v

# Test CLI commands
pytest tests/test_cli_messaging_notification.py -k notify -v
```

### Mock Behavior

In tests, channel dispatch is mocked:
- No actual SMS sent
- No actual emails sent
- Dashboard store uses temp SQLite database
- All operations are simulated

## Notification Inbox UI

The dashboard includes a notification inbox page backed by `DashboardStore` and the existing
notification API endpoints.

### Accessing the Inbox

Open the dashboard and click **Notifications** in the sidebar (or mobile bottom navigation),
or navigate directly to:

```
http://localhost:5001/dashboard/notifications
```

The page uses the same single-page application as the main dashboard.  Authentication is
enforced at the API level — the HTML page itself is always served; the JavaScript layer
authenticates before fetching notification data.

### Filters

| Filter | Type | Description |
|--------|------|-------------|
| Unread only | Checkbox | When checked, only unread notifications are fetched from the server via `?unread=true`. |
| Priority | Dropdown | Filter by `urgent`, `normal`, or `digest` (server-side via `?priority=`). |
| Channel | Dropdown | Filter by delivery channel (`dashboard`, `email`, `sms`, `home_assistant_tts`). Applied client-side from the fetched result set. |

### Actions

- **Mark read** — button on each unread notification calls `POST /api/notifications/<id>/read`.
- **Mark all read** — header button calls `POST /api/notifications/read-all`; optionally scoped to a user via `?user_id=`.

Both actions reload the list and refresh the unread count badge on the nav link.

### Real-Time Push (Server-Sent Events)

When the Notifications section is active the dashboard UI opens a Server-Sent Events
(SSE) connection to `GET /api/notifications/stream`.  New notifications are pushed to
the browser as they are persisted, so the list and unread badge update within seconds.

**Endpoint:**

```
GET /api/notifications/stream?token=<session_token>&user_id=<optional>
```

**Authentication:** The stream endpoint requires a valid dashboard session.  Because
`EventSource` cannot set custom headers, the token can be passed via the `token` query
parameter (or via the `rex_dashboard_token` cookie which is set on login).

**Initial event:** The first SSE event has `type: "init"` and includes the current
`unread_count`.

**Notification events:** Each newly persisted notification triggers an event with
`type: "notification"` and fields `id`, `priority`, `title`, `channel`, `timestamp`,
and `user_id`.

**Keep-alive:** A comment line (`: keepalive <timestamp>`) is sent every 30 seconds
to prevent proxies from closing idle connections.

**Fallback:** If the SSE connection fails (network error, unsupported browser, 401/403),
the UI automatically falls back to the existing 30-second polling.  On logout or
section switch the stream is closed cleanly.

**Known limitations:**

- The in-process broadcaster only works within a single Python process.  If you run
  multiple Flask workers (e.g. gunicorn with `--workers N`), each worker has its own
  broadcaster instance and clients connected to one worker will not see events published
  in another.  For single-process deployments (the default) this is not an issue.
- The `max_events` and `timeout_seconds` query parameters are honoured only when the
  Flask app is in testing mode (`app.config["TESTING"] = True`).

### Polling Fallback

The inbox falls back to polling `GET /api/notifications` every **30 seconds** when the
SSE stream is unavailable or unsupported.  Polling stops automatically when you switch
to another section or log out.

### Unread Badge

The **Notifications** nav link shows a red badge with the current unread count.  The badge
is refreshed every time the notifications list is loaded.

## What Remains Stubbed

- Home Assistant TTS channel (logs only)
- Multi-process SSE broadcast (current broadcaster is in-process only; does not work across gunicorn workers)
- Notification templates
- Per-user channel preferences (foundation exists via metadata)
- Snooze/postpone functionality

## Support

For questions or issues:
- Check tests in `tests/test_notification.py` and `tests/test_dashboard_store.py`
- Review code in `rex/notification.py` and `rex/dashboard_store.py`
- Open an issue on GitHub
