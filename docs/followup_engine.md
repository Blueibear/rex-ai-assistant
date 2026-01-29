# Conversational Follow-Up Engine

Rex includes a conversational follow-up engine that makes interactions feel more personal and natural. The engine can follow up on calendar events, reminders, and other activities.

## Overview

The follow-up engine allows Rex to:

1. **Generate cues from calendar events** - After a meeting or appointment ends, Rex can ask "How did [event] go?"
2. **Create follow-up reminders** - When you set a reminder with `--follow-up`, Rex will ask if you completed the task
3. **Natural conversation injection** - Follow-ups feel like small talk, not a robotic checklist

## What is a Cue?

A **Cue** is a lightweight record that Rex can bring up in conversation. Cues are:

- **Per user/profile** - Each user has their own cues
- **Persisted** - Survive restarts (stored in `data/cues/cues.json`)
- **Consumable** - Asked once, then marked as "asked"
- **Rate-limited** - Default max 1 cue per conversation session
- **Time-windowed** - Only consider cues within a configurable time window

## Configuration

Add to your `config/rex_config.json`:

```json
{
  "conversation": {
    "followups": {
      "enabled": true,
      "max_per_session": 1,
      "lookback_hours": 72,
      "expire_hours": 168
    }
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Enable/disable follow-up cues |
| `max_per_session` | `1` | Maximum cues to ask per conversation |
| `lookback_hours` | `72` | Only show cues from this time window |
| `expire_hours` | `168` | Cues expire after this many hours (7 days) |

## Reminders CLI

Manage reminders with the `rex reminders` command:

### Add a reminder

```bash
rex reminders add "Call mom" --at "2024-01-15 14:00"
```

Add with follow-up (Rex will ask if you completed it):

```bash
rex reminders add "Submit report" --at "2024-01-15 17:00" --follow-up
```

### List reminders

```bash
rex reminders list
rex reminders list --status pending
```

### Mark as done

```bash
rex reminders done <reminder_id>
```

### Cancel a reminder

```bash
rex reminders cancel <reminder_id>
```

## Cues CLI

Manage follow-up cues with the `rex cues` command:

### List cues

```bash
rex cues list
rex cues list --status pending
```

### Dismiss a cue

If you don't want Rex to ask about something:

```bash
rex cues dismiss <cue_id>
```

### Prune expired cues

Remove old, expired cues:

```bash
rex cues prune
```

## Calendar Integration

The follow-up engine automatically generates cues from calendar events that have ended. This happens when a conversation starts.

### Skipped Events

Some events are automatically skipped:

1. **All-day holidays** - Events like "Company Holiday", "PTO", "Vacation"
2. **No-followup events** - Events with `[no-followup]` or `nofollowup` in the title or description

To prevent a follow-up for a specific event, add `[no-followup]` to the event title or description in your calendar.

## How It Works

1. **Session start**: When you start a chat or voice session, the engine:
   - Generates cues from recent calendar events
   - Checks for pending cues within the lookback window
   - Selects one cue to ask (respecting rate limits)

2. **Prompt injection**: The selected cue is injected into the conversation prompt as a hint for Rex to naturally ask the question.

3. **Marking asked**: Once Rex asks the follow-up, the cue is marked as "asked" and won't be asked again.

## Data Storage

- **Cues**: `data/cues/cues.json`
- **Reminders**: `data/reminders/reminders.json`

## API Reference

### CueStore

```python
from rex.cue_store import get_cue_store, Cue

store = get_cue_store()

# Add a cue
cue = store.add_cue(
    user_id="default",
    source_type="calendar",  # or "reminder", "manual"
    source_id="event-123",
    title="Doctor appointment",
    prompt="How did your doctor appointment go?",
    expires_in=timedelta(hours=168),
)

# List pending cues
pending = store.list_pending_cues("default", window_hours=72)

# Mark as asked
store.mark_asked(cue.cue_id)

# Dismiss
store.dismiss(cue.cue_id)
```

### ReminderService

```python
from rex.reminder_service import get_reminder_service

service = get_reminder_service()

# Create reminder
reminder = service.create_reminder(
    user_id="default",
    title="Call mom",
    remind_at=datetime.now() + timedelta(hours=2),
    followup_enabled=True,
)

# List reminders
reminders = service.list_reminders("default", status="pending")

# Mark done
service.mark_done(reminder.reminder_id)
```

### FollowupEngine

```python
from rex.followup_engine import get_followup_engine

engine = get_followup_engine()

# Start a session
engine.start_session("default")

# Get a follow-up prompt
prompt = engine.get_followup_prompt("default")
if prompt:
    print(f"Rex will ask: {prompt}")

# Mark the cue as asked
engine.mark_current_cue_asked("default")
```

## Disabling Follow-ups

To completely disable follow-ups:

1. Set `conversation.followups.enabled` to `false` in config
2. Or set `max_per_session` to `0`

## Known Limitations

- Email-based cues are not yet implemented (planned for v2)
- Cue detection from conversation content is basic (marks as asked after first user response)
- No recurring reminders (one-off only for v1)
