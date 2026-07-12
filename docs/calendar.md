# Calendar Integration

**Implementation Status: Beta (ICS read-only backend available + stub fallback)**

The AskRex Assistant includes calendar integration that allows Rex to read calendar events, detect conflicts, and keep you informed of upcoming appointments. This enables Rex to proactively manage your schedule and trigger workflows based on calendar events.

## Overview

The calendar service provides:
- Read access to calendar events from ICS files or HTTPS ICS feeds
- Stub/mock data for offline development and testing
- Upcoming event queries
- Conflict detection
- Event creation and management (stub backend only)
- Integration with the scheduler for automated syncing
- Event publishing for reactive workflows

## Current Implementation

The calendar service supports two backends:

| Backend | Status | Capabilities |
|---------|--------|-------------|
| **stub** (default) | Available for local dev/test | Read/write mock events from JSON |
| **ics** | Beta | Read-only from local `.ics` files or HTTPS ICS feeds |

CalDAV and Google Calendar OAuth backends are planned for future releases.

## Architecture

The calendar service consists of two main components:

1. **CalendarEvent**: A Pydantic model representing a calendar event
2. **CalendarService**: The service class that handles calendar operations

## CalendarEvent Model

A `CalendarEvent` contains the following fields:

```python
class CalendarEvent(BaseModel):
    id: str                       # Unique event identifier
    title: str                    # Event title/summary
    start_time: datetime          # Event start time
    end_time: datetime            # Event end time
    attendees: list[str] = []     # Attendee email addresses
    location: Optional[str] = None # Event location
    description: Optional[str] = None # Event description
    all_day: bool = False         # Whether this is an all-day event
```

## Per-User Account Ownership (issue #303)

Every calendar operation runs under **one explicit, validated user
identity**.  Account authorization and routing are handled by
`rex.calendar_accounts.CalendarAccountResolver`:

- `calendar.accounts` in `config/rex_config.json` is the canonical
  non-secret account list (provider, ICS source, `credential_ref` — an
  environment-variable *name*, never a secret value).
- `users.{user_id}.calendar_accounts` assigns account IDs to users.  A named
  user can only ever use accounts explicitly assigned to them.
- `users.{user_id}.default_calendar_account_id` selects that user's default
  account (must reference an account the user owns).
- Missing, blank, malformed, or traversal-style user IDs fail closed before
  any account or credential lookup.
- Unauthorized and nonexistent account IDs produce the same generic error,
  so users cannot enumerate other users' accounts.
- Backends and event stores are cached per `(user_id, account_id)`; a
  backend created for one user is never reused for another.

### Legacy policy

The pre-#303 global configuration keeps working **for the explicit
`default` profile only**:

- `calendar.backend = "ics"` + `calendar.ics.source` is synthesized into a
  reserved legacy account owned by the `default` profile.
- `calendar.provider = "google"` + the `GOOGLE_CALENDAR_ACCESS_TOKEN`
  environment variable serve only the `default` profile.
- Named users (e.g. `james`, `cole`) are **never** automatically assigned to
  legacy accounts or legacy environment tokens — not even via an explicit
  `users.{user}.calendar_accounts` entry pointing at a legacy account ID.

### Assigning calendar accounts to named users

Administrators assign accounts explicitly in `config/rex_config.json`:

```json
{
  "calendar": {
    "accounts": [
      {
        "id": "james-work",
        "label": "James work calendar",
        "provider": "ics",
        "ics": {"source": "https://example.com/james.ics", "url_timeout": 15}
      },
      {
        "id": "cole-google",
        "label": "Cole Google calendar",
        "provider": "google",
        "credential_ref": "GOOGLE_CALENDAR_TOKEN_COLE"
      }
    ]
  },
  "users": {
    "james": {
      "calendar_accounts": [{"account_id": "james-work"}],
      "default_calendar_account_id": "james-work"
    },
    "cole": {
      "calendar_accounts": [{"account_id": "cole-google"}]
    }
  }
}
```

For provider accounts (`google`), put the token itself in `.env` under the
variable named by `credential_ref` (e.g. `GOOGLE_CALENDAR_TOKEN_COLE=...`).
Credential *references* are config; credential *values* are secrets and
belong in `.env` only.  CLI output, API responses, events, and error
messages never echo credential references or token values.

Routing order for a request by user U:

1. Explicit account ID (after ownership validation)
2. `users.U.default_calendar_account_id`
3. Legacy global default — `default` profile only
4. First account assigned to U (config order)
5. Fail closed: not-configured result (never another user's account)

## Using the Calendar Service

### Getting the Calendar Service Instance

```python
from rex.calendar_service import get_calendar_service

calendar_service = get_calendar_service()
```

### Connecting to Calendar

```python
# Connect for one validated user (loads that user's events)
if calendar_service.connect(user_id="james"):
    print("Connected to calendar service")
else:
    print("Failed to connect")
```

### Fetching Events

#### Get Events in a Time Range

```python
from datetime import datetime, timedelta

# Get events for the next 7 days (requesting user's own events only)
start = datetime.now()
end = start + timedelta(days=7)
events = calendar_service.get_events(start, end, user_id="james")

for event in events:
    print(f"{event.title}: {event.start_time}")
```

#### Get Upcoming Events

```python
# Get events in the next 7 days
events = calendar_service.get_upcoming_events(days=7, user_id="james")

# Get events in the next 14 days
events = calendar_service.get_upcoming_events(days=14, user_id="james")
```

### Creating Events

```python
from datetime import datetime, timedelta
from rex.calendar_service import CalendarEvent

# Create event
start = datetime.now() + timedelta(days=1)
end = start + timedelta(hours=1)

event = CalendarEvent(
    id="",  # Will be auto-generated
    title="Team Meeting",
    start_time=start,
    end_time=end,
    attendees=["alice@example.com", "bob@example.com"],
    location="Conference Room A",
    description="Weekly team sync"
)

created_event = calendar_service.create_event(event)
print(f"Created event: {created_event.id}")
```

### Updating Events

```python
# Update event properties
updated = calendar_service.update_event(
    event_id="event-123",
    updates={
        "title": "Updated Meeting Title",
        "location": "Conference Room B"
    }
)
```

### Deleting Events

```python
# Delete an event
success = calendar_service.delete_event(event_id="event-123")
if success:
    print("Event deleted")
```

### Detecting Conflicts

```python
# Check for conflicts in upcoming events
events = calendar_service.get_upcoming_events(days=7)
conflicts = calendar_service.find_conflicts(events)

for event1, event2 in conflicts:
    print(f"Conflict: '{event1.title}' overlaps with '{event2.title}'")
```

### Checking Event Overlap

```python
from rex.calendar_service import CalendarEvent

# Check if two events overlap
if event1.overlaps_with(event2):
    print("Events overlap!")
```

## CLI Commands

Every calendar subcommand resolves one validated user (from `--user`, the
`rex identify` session, or `runtime.active_user`) and fails closed with an
actionable message when no user can be resolved.  Single-user setups select
the legacy profile explicitly: `rex identify --user default`.

### Show Upcoming Events

Display upcoming calendar events for the resolved user only:

```bash
rex calendar upcoming
rex calendar upcoming --days 14          # Next 14 days
rex calendar upcoming --conflicts        # Include conflict detection
rex calendar upcoming -v                 # Verbose with descriptions
rex calendar upcoming --user cole        # Override active user context
```

### Manage Calendar Accounts

List the resolved user's own accounts (never another user's) and set the
per-user default:

```bash
rex calendar accounts list --user james
rex calendar accounts set-active --account-id james-work --user james
```

`set-active` writes only `users.james.default_calendar_account_id`; it
never modifies another user's routing or the legacy global default.

### Test Account Connection

Verify the resolved user's own calendar account (foreign accounts cannot be
tested):

```bash
rex calendar test-connection
rex calendar test-connection --account-id james-work --user james
```

## Configuration

### Backend Selection

Configure the calendar backend in `config/rex_config.json`:

```json
{
  "calendar": {
    "backend": "ics",
    "ics": {
      "source": "/path/to/calendar.ics",
      "url_timeout": 15
    }
  }
}
```

#### Config Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `calendar.backend` | string | `"stub"` | Backend to use: `"stub"` or `"ics"` |
| `calendar.ics.source` | string | `null` | Path to local `.ics` file or HTTPS URL |
| `calendar.ics.url_timeout` | int | `15` | Timeout in seconds for HTTPS ICS feed fetch |

#### ICS Source Options

`calendar.ics.source` accepts:
- **Local file path**: absolute or project-relative path to a `.ics` file
- **HTTPS URL**: an `https://` URL to an ICS feed (e.g. Google Calendar public feed)

HTTP (non-HTTPS) URLs are rejected for security.

For HTTPS feeds, Rex also rejects `localhost` and hosts that resolve to local/private/reserved
IP ranges to reduce SSRF risk.

### ICS Parsing Notes and Limits

- `DTSTART`/`DTEND` values support `DATE` and `DATE-TIME` forms.
- All-day events (`VALUE=DATE`) are represented with `all_day=True` and an end time of +1 day when
  `DTEND` is omitted.
- Folded lines (RFC 5545 continuations) are unfolded before parsing.
- Multiple `VEVENT` blocks are supported.
- `RRULE` is currently not expanded; each `VEVENT` is treated as one event.
- `TZID` parameters are accepted but timezone conversion is not performed; floating/non-UTC
  datetimes are currently treated as UTC.

### Stub / Mock Data

When `calendar.backend` is `"stub"` (the default), mock data comes from `data/mock_calendar.json`.

**Example mock_calendar.json:**

```json
[
  {
    "id": "event-001",
    "title": "Team Standup",
    "start_time": "2026-01-29T09:00:00",
    "end_time": "2026-01-29T09:30:00",
    "attendees": ["alice@company.com", "bob@company.com"],
    "location": "Conference Room A",
    "description": "Daily team standup meeting",
    "all_day": false
  }
]
```

### Test Connection

Verify your calendar backend configuration without side effects:

```bash
rex calendar test-connection
```

### Credentials

For future CalDAV/Google OAuth backends, credentials will be stored via the credential manager.
Secrets belong in `.env` only (never in `rex_config.json`).

## Scheduled Calendar Syncing

The calendar service integrates with the scheduler to sync events automatically:

```python
from rex.integrations import initialize_scheduler_system

# Initialize scheduler (includes calendar sync job)
initialize_scheduler_system(start_scheduler=True)
```

This creates a job that:
- Runs every hour (configurable)
- Iterates configured calendar owners; each owner runs in an isolated
  context (one owner's failure never falls through to another's account)
- Fetches upcoming events (next 7 days) per owner
- Publishes a **safe envelope** (`{count, user_id}`) on the shared
  `calendar.update` topic and the full event payload only on the
  owner-scoped `calendar.update.user.<user_id>` topic

Private event fields (titles, attendees, locations, descriptions,
conferencing links) are never published on shared topics.

## Event Integration

Subscribe to calendar events to trigger actions:

```python
from rex.event_bus import get_event_bus

event_bus = get_event_bus()

def handle_calendar_update(event):
    # Full payloads are published only on the owner-scoped topic
    # calendar.update.user.<user_id>; the shared calendar.update topic
    # carries a safe envelope ({count, user_id}) without event details.
    events = event.payload.get('events', [])
    print(f"Calendar updated: {len(events)} upcoming events")

    # Check for events today
    today = datetime.now().date()
    today_events = [
        e for e in events
        if datetime.fromisoformat(e['start_time']).date() == today
    ]

    if today_events:
        print(f"You have {len(today_events)} event(s) today")

event_bus.subscribe('calendar.update', handle_calendar_update)
```

## Example: Daily Agenda Notification

```python
from datetime import datetime, timedelta
from rex.event_bus import get_event_bus
from rex.integrations import initialize_scheduler_system

# Initialize system
initialize_scheduler_system(start_scheduler=True)
event_bus = get_event_bus()

# Define handler
def daily_agenda(event):
    """Display daily agenda."""
    events = event.payload['events']

    # Get today's events
    today = datetime.now().date()
    today_events = [
        e for e in events
        if datetime.fromisoformat(e['start_time']).date() == today
    ]

    if today_events:
        print(f"\n📅 Today's Schedule ({len(today_events)} events):\n")
        for event_data in today_events:
            start = datetime.fromisoformat(event_data['start_time'])
            print(f"  {start.strftime('%H:%M')} - {event_data['title']}")
            if event_data.get('location'):
                print(f"    📍 {event_data['location']}")
        print()

# Subscribe to calendar events
event_bus.subscribe('calendar.update', daily_agenda)

print("Daily agenda system active")
```

## Example: Conflict Warning

```python
from rex.calendar_service import get_calendar_service

calendar_service = get_calendar_service()
calendar_service.connect(user_id="james")

# Get upcoming events
events = calendar_service.get_upcoming_events(days=7, user_id="james")

# Check for conflicts
conflicts = calendar_service.find_conflicts(events, user_id="james")

if conflicts:
    print(f"⚠️  Warning: {len(conflicts)} scheduling conflict(s) detected!")
    for event1, event2 in conflicts:
        print(f"  • '{event1.title}' overlaps with '{event2.title}'")
        print(f"    {event1.start_time} - {event1.end_time}")
        print(f"    {event2.start_time} - {event2.end_time}")
else:
    print("✓ No scheduling conflicts")
```

## Example: Meeting Preparation Reminder

```python
from datetime import datetime, timedelta
from rex.calendar_service import get_calendar_service

calendar_service = get_calendar_service()
calendar_service.connect(user_id="james")

# Get events in the next hour
now = datetime.now()
soon = now + timedelta(hours=1)
upcoming = calendar_service.get_events(now, soon, user_id="james")

for event in upcoming:
    # Send reminder for meetings with multiple attendees
    if len(event.attendees) > 1:
        time_until = event.start_time - now
        minutes = int(time_until.total_seconds() / 60)

        print(f"🔔 Reminder: '{event.title}' in {minutes} minutes")
        print(f"   Attendees: {', '.join(event.attendees)}")
        if event.location:
            print(f"   Location: {event.location}")
```

## Future: Additional Calendar Backends

Planned backend rollout order (per CLAUDE.md):
1. **ICS read-only** (implemented) - local files and HTTPS feeds
2. **CalDAV** - standard protocol for read/write calendar access
3. **Google Calendar OAuth** - direct Google Calendar API integration

## Event Types

### All-Day Events

All-day events have `all_day=True` and typically span full days:

```python
event = CalendarEvent(
    id="vacation",
    title="Vacation",
    start_time=datetime(2026, 2, 1),
    end_time=datetime(2026, 2, 8),
    all_day=True
)
```

### Recurring Events

**Note**: Recurring events are not yet fully supported. Each occurrence should be created as a separate event.

Future support will include:
- RRULE syntax
- Exception dates
- Modified occurrences

## Security Considerations

1. **Credentials**: Never hardcode calendar credentials. Use the credential manager.
2. **OAuth Tokens**: Store OAuth tokens securely and refresh them automatically
3. **Permissions**: Request minimal calendar permissions (read-only if write not needed)
4. **Data Privacy**: Be careful with event details in logs and notifications
5. **Rate Limits**: Respect API rate limits for calendar providers

## Best Practices

1. **Use mock data for testing**: Keep real calendar credentials separate
2. **Handle API failures gracefully**: Calendar APIs can be rate-limited
3. **Sync periodically**: Don't sync too frequently (hourly is reasonable)
4. **Check for conflicts**: Always validate new events for conflicts
5. **Batch operations**: Group multiple calendar operations when possible
6. **Subscribe to events**: Use the event bus for reactive workflows
7. **Cache event data**: Reduce API calls by caching recent events

## Troubleshooting

### Connection Issues

```python
# Check if credentials are configured
from rex.credentials import get_credential_manager
cred_manager = get_credential_manager()
calendar_creds = cred_manager.get_credential('calendar')
if not calendar_creds:
    print("Calendar credentials not configured")
```

### No Events Returned

If `get_events()` returns empty:
- Check that mock data file exists: `data/mock_calendar.json`
- Verify time range includes events
- Check calendar service connection status

### Conflict Detection Issues

If conflicts aren't detected:
- Verify event times overlap correctly
- Check that `all_day` events are handled properly
- Use `overlaps_with()` method to test manually

## Future Enhancements

Planned improvements for calendar integration:

- CalDAV backend for standard read/write calendar access
- Google Calendar OAuth2 backend
- Recurring event support (RRULE expansion)
- Event reminders and notifications
- Attendee management (invites, responses)
- Calendar sharing
- Multiple calendar support
- Free/busy time queries
- Time zone handling improvements
- Event search functionality
- Calendar sync across multiple accounts
