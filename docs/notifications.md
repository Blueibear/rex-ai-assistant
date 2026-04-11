# Notifications

AskRex has two notification-related paths:

- `rex.notification` for Python-side notification requests, digest queues,
  quiet hours, and escalation tracking.
- The Electron notifications page under `gui/`, which uses the Electron IPC
  bridge (`window.rex.*`) to list, mark, and dismiss GUI notifications.

Older docs described Flask dashboard routes such as `/api/notifications`. Those
routes are not present in the current `rex-gui` Flask app.

## Python Notification Service

Core module:

```text
rex/notification.py
```

Main types:

- `NotificationRequest`
- `Notifier`
- `DigestQueue`
- `EscalationManager`
- `get_notifier()`
- `get_escalation_manager()`

Priority values:

- `urgent`
- `normal`
- `digest`

Channel preferences supported by the dispatcher:

- `dashboard`
- `email`
- `sms`
- `ha_tts`

The dashboard channel currently logs the notification. The legacy persistent
dashboard notification store is pending retirement in favor of the newer UI and
OpenClaw-oriented path.

## CLI Usage

Send a dashboard/logged notification:

```bash
rex notify send --priority normal \
  --title "Build Complete" \
  --body "Your build finished successfully"
```

Send an urgent notification through multiple channels:

```bash
rex notify send --priority urgent \
  --title "Critical Alert" \
  --body "System failure detected" \
  --channels sms,email,dashboard
```

Digest commands:

```bash
rex notify list-digests
rex notify flush-digests
rex notify flush-digests --channel email
```

Acknowledge an urgent notification to stop escalation:

```bash
rex notify ack notif_abc123
```

## Python API

```python
from rex.notification import NotificationRequest, get_notifier

notifier = get_notifier()

notification = NotificationRequest(
    priority="urgent",
    title="High Priority Email",
    body="You have a high importance email.",
    channel_preferences=["sms", "email", "dashboard"],
    metadata={
        "to_number": "+15551234567",
        "to_email": "user@example.com",
        "email_account_id": "primary",
    },
)

notifier.send(notification)
```

Email delivery uses `EmailService.send()` when `metadata.to_email` is present
and the email backend is configured. SMS delivery uses the messaging service
when a phone number is present and the SMS backend is configured. Home Assistant
TTS uses `rex.ha_tts.client` when `notifications.ha_tts` is enabled in
`config/rex_config.json`.

## Configuration

Runtime notification settings belong in `config/rex_config.json`, especially:

```json
{
  "notifications": {
    "ha_tts": {
      "enabled": false,
      "base_url": null,
      "token_ref": "ha:tts_token",
      "default_entity_id": null,
      "default_tts_domain": "tts",
      "default_tts_service": "speak",
      "timeout_seconds": 10.0,
      "allow_http": false
    }
  }
}
```

Secrets such as Home Assistant tokens, email credentials, and Twilio credentials
belong in `.env` or the repo's credential lookup path.

## Electron Notification UI

The Electron route is:

```text
#/notifications
```

The page lives at:

```text
gui/src/pages/NotificationsPage.tsx
```

It supports:

- priority grouping
- unread/read state
- detail panel
- mark read
- mark all read
- dismiss
- optional action URL navigation

The page calls Electron bridge methods such as:

- `window.rex.getNotifications()`
- `window.rex.markNotificationRead(id)`
- `window.rex.dismissNotification(id)`
- `window.rex.onNewNotification(callback)`

To validate the Electron UI:

```powershell
cd gui
npm.cmd run typecheck
npm.cmd run build
```

For Electron-only verification harnesses, build first and require
`gui/dist-electron/main/index.js` from `gui/tmp_verify_*.cjs`.

## Tests

Useful test targets:

```bash
pytest tests/test_notification.py -v
pytest tests/test_cli_messaging_notification.py -k notify -v
```

Run the broader baseline when notification changes touch shared behavior:

```bash
pytest -q
python -m rex doctor
```
