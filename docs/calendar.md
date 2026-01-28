# Calendar Integration

The Rex AI Assistant includes calendar integration that allows Rex to read and write calendar events, detect conflicts, and keep you informed of upcoming appointments. This enables Rex to proactively manage your schedule and trigger workflows based on calendar events.

## Overview

The calendar service provides:
- Read and write access to calendar events
- Upcoming event queries
- Conflict detection
- Event creation and management
- Integration with the scheduler for automated syncing
- Event publishing for reactive workflows

## Current Implementation

**Note**: The current implementation uses mock/stub functionality for testing. Real calendar API integration (Google Calendar, Outlook Calendar, etc.) will be added in future releases.

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

## Using the Calendar Service

### Getting the Calendar Service Instance

```python
from rex.calendar_service import get_calendar_service

calendar_service = get_calendar_service()
```

### Connecting to Calendar

```python
# Connect (loads credentials and initializes)
if calendar_service.connect():
    print("Connected to calendar service")
else:
    print("Failed to connect")
```

### Fetching Events

#### Get Events in a Time Range

```python
from datetime import datetime, timedelta

# Get events for the next 7 days
start = datetime.now()
end = start + timedelta(days=7)
events = calendar_service.get_events(start, end)

for event in events:
    print(f"{event.title}: {event.start_time}")
```

#### Get Upcoming Events

```python
# Get events in the next 7 days
events = calendar_service.get_upcoming_events(days=7)

# Get events in the next 14 days
events = calendar_service.get_upcoming_events(days=14)
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

### Show Upcoming Events

Display upcoming calendar events:

```bash
rex calendar upcoming
rex calendar upcoming --days 14          # Next 14 days
rex calendar upcoming --conflicts        # Include conflict detection
rex calendar upcoming -v                 # Verbose with descriptions
```

## Configuration

### Mock Data

The calendar service currently uses mock data from `data/mock_calendar.json`. This file contains sample events for testing.

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
  },
  {
    "id": "event-002",
    "title": "Conference",
    "start_time": "2026-02-10T00:00:00",
    "end_time": "2026-02-11T23:59:59",
    "attendees": [],
    "location": "Convention Center",
    "description": "Annual AI conference",
    "all_day": true
  }
]
```

### Credentials

Calendar credentials should be configured in the credential manager:

```python
from rex.credentials import get_credential_manager

cred_manager = get_credential_manager()

# Future: Real calendar API credentials
# cred_manager.set_credential('calendar', {
#     'access_token': 'your-token',
#     'refresh_token': 'your-refresh-token',
#     'client_id': 'your-client-id',
#     'client_secret': 'your-client-secret'
# })
```

**For Google Calendar:**
- Use OAuth2 authentication
- Request calendar.readonly or calendar scope
- Store access and refresh tokens securely

## Scheduled Calendar Syncing

The calendar service integrates with the scheduler to sync events automatically:

```python
from rex.integrations import initialize_scheduler_system

# Initialize scheduler (includes calendar sync job)
initialize_scheduler_system(start_scheduler=True)
```

This creates a job that:
- Runs every hour (configurable)
- Fetches upcoming events (next 7 days)
- Publishes a `calendar.update` event

## Event Integration

Subscribe to calendar events to trigger actions:

```python
from rex.event_bus import get_event_bus

event_bus = get_event_bus()

def handle_calendar_update(event):
    events = event.payload['events']
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
calendar_service.connect()

# Get upcoming events
events = calendar_service.get_upcoming_events(days=7)

# Check for conflicts
conflicts = calendar_service.find_conflicts(events)

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
calendar_service.connect()

# Get events in the next hour
now = datetime.now()
soon = now + timedelta(hours=1)
upcoming = calendar_service.get_events(now, soon)

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

## Future: Real Calendar API Integration

Future releases will include real calendar API support:

```python
# Google Calendar example
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

class CalendarService:
    def connect(self):
        # Authenticate with Google Calendar
        creds = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret
        )
        self.service = build('calendar', 'v3', credentials=creds)

    def get_events(self, start, end):
        # Fetch events from Google Calendar
        events_result = self.service.events().list(
            calendarId='primary',
            timeMin=start.isoformat() + 'Z',
            timeMax=end.isoformat() + 'Z',
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        # Parse and return CalendarEvent objects
```

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

- Real calendar API support (Google Calendar, Outlook Calendar, etc.)
- OAuth2 authentication flow
- Recurring event support (RRULE)
- Event reminders and notifications
- Attendee management (invites, responses)
- Calendar sharing
- Multiple calendar support
- Free/busy time queries
- Meeting room booking
- Time zone handling
- Event search functionality
- Calendar sync across multiple accounts
