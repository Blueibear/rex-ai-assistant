# Event Bus

The AskRex Assistant includes an event bus that provides publish-subscribe messaging for internal components. The event bus enables loose coupling between modules and facilitates reactive programming patterns.

## Overview

The event bus allows components to:
- Publish events when interesting things happen
- Subscribe handlers to specific event types
- Receive notifications without tight coupling
- Build reactive workflows

## Architecture

The event bus consists of two main components:

1. **Event**: A dataclass representing an event with type, payload, and timestamp
2. **EventBus**: The main bus class that manages subscriptions and publishing

## Event Dataclass

An `Event` contains the following fields:

```python
@dataclass
class Event:
    event_type: str              # Event type identifier (e.g., 'email.unread')
    payload: dict[str, Any]      # Event data
    timestamp: datetime          # When the event occurred (auto-generated)
```

## Using the Event Bus

### Getting the Event Bus Instance

```python
from rex.event_bus import get_event_bus, Event

event_bus = get_event_bus()
```

### Publishing Events

```python
# Create an event
event = Event(
    event_type='user.login',
    payload={
        'user_id': '12345',
        'ip_address': '192.168.1.1',
        'timestamp': '2026-01-28T10:00:00'
    }
)

# Publish the event
event_bus.publish(event)
```

### Subscribing to Events

#### Basic Subscription

```python
def handle_login(event):
    user_id = event.payload['user_id']
    print(f"User {user_id} logged in at {event.timestamp}")

event_bus.subscribe('user.login', handle_login)
```

#### Multiple Handlers

Multiple handlers can subscribe to the same event type:

```python
def log_login(event):
    print(f"Logging login: {event.payload}")

def notify_admin(event):
    print(f"Notifying admin of login: {event.payload['user_id']}")

event_bus.subscribe('user.login', log_login)
event_bus.subscribe('user.login', notify_admin)
```

#### Wildcard Subscription

Subscribe to all events using the wildcard `*`:

```python
def log_all_events(event):
    print(f"Event: {event.event_type} at {event.timestamp}")

event_bus.subscribe('*', log_all_events)
```

### Unsubscribing

```python
event_bus.unsubscribe('user.login', handle_login)
```

### Managing Subscriptions

```python
# Get subscription count for an event type
count = event_bus.get_subscription_count('user.login')

# Get statistics
stats = event_bus.get_stats()
print(f"Total events published: {stats['total_events']}")
print(f"Error count: {stats['error_count']}")
print(f"Subscriptions: {stats['subscriptions']}")

# Clear subscriptions
event_bus.clear_subscriptions('user.login')  # Clear specific type
event_bus.clear_subscriptions()  # Clear all
```

## Event Types

Rex uses a convention for event type naming:

- Use dot notation: `category.action`
- Use lowercase
- Be specific and descriptive

### Built-in Event Types

The following event types are used by Rex's default jobs:

#### email.unread

Published when new unread emails are detected.

**Payload:**
```python
{
    'count': int,           # Number of unread emails
    'emails': [             # List of email summaries
        {
            'id': str,
            'from_addr': str,
            'subject': str,
            'snippet': str,
            'received_at': str,  # ISO format datetime
            'category': str,
            'importance_score': float
        }
    ]
}
```

#### calendar.update

Published when calendar events are synced.

**Payload:**
```python
{
    'count': int,           # Number of events
    'events': [             # List of calendar events
        {
            'id': str,
            'title': str,
            'start_time': str,  # ISO format datetime
            'end_time': str,    # ISO format datetime
            'attendees': list[str],
            'location': str | None,
            'description': str | None
        }
    ]
}
```

### Custom Event Types

You can define your own event types for custom workflows:

```python
# System events
event_bus.publish(Event('system.health', {'status': 'healthy'}))
event_bus.publish(Event('system.shutdown', {'reason': 'maintenance'}))

# Workflow events
event_bus.publish(Event('workflow.started', {'workflow_id': 'abc-123'}))
event_bus.publish(Event('workflow.completed', {'workflow_id': 'abc-123', 'status': 'success'}))

# Custom domain events
event_bus.publish(Event('task.created', {'task_id': '456', 'priority': 'high'}))
event_bus.publish(Event('file.uploaded', {'filename': 'doc.pdf', 'size': 1024}))
```

## Error Handling

If a handler raises an exception:
- The error is logged
- The error count is incremented
- Other handlers for the same event still execute
- Publishing continues normally

```python
def faulty_handler(event):
    raise ValueError("Something went wrong")

def good_handler(event):
    print("This still runs")

event_bus.subscribe('test.event', faulty_handler)
event_bus.subscribe('test.event', good_handler)

# Both handlers are called, but faulty_handler's error is caught
event_bus.publish(Event('test.event', {}))
```

## Integration with Scheduler

The scheduler uses the event bus to notify listeners when scheduled tasks complete:

```python
from rex.event_bus import get_event_bus
from rex.integrations import initialize_scheduler_system

# Initialize scheduler (sets up default jobs and event handlers)
initialize_scheduler_system(start_scheduler=True)

# Subscribe to scheduled events
event_bus = get_event_bus()

def handle_email_updates(event):
    count = event.payload['count']
    if count > 0:
        print(f"You have {count} unread emails!")

event_bus.subscribe('email.unread', handle_email_updates)
```

## Example: Building a Notification System

```python
from rex.event_bus import get_event_bus, Event

event_bus = get_event_bus()

# Define notification handler
def send_notification(event):
    """Send notifications based on event type."""
    if event.event_type == 'email.unread':
        emails = event.payload['emails']
        important = [e for e in emails if e.get('category') == 'important']
        if important:
            print(f"⚠️  {len(important)} important emails!")

    elif event.event_type == 'calendar.update':
        events = event.payload['events']
        today = [e for e in events if is_today(e['start_time'])]
        if today:
            print(f"📅 {len(today)} events today")

# Subscribe to all events
event_bus.subscribe('*', send_notification)
```

## Example: Event-Driven Workflow

```python
from rex.event_bus import get_event_bus, Event

event_bus = get_event_bus()

# Step 1: Trigger workflow on email
def start_email_workflow(event):
    emails = event.payload['emails']
    for email in emails:
        if 'urgent' in email['subject'].lower():
            # Publish new event to trigger urgent handling
            event_bus.publish(Event(
                'workflow.urgent_email',
                {'email_id': email['id']}
            ))

event_bus.subscribe('email.unread', start_email_workflow)

# Step 2: Handle urgent emails
def handle_urgent_email(event):
    email_id = event.payload['email_id']
    print(f"Processing urgent email: {email_id}")
    # Process email...
    # Publish completion event
    event_bus.publish(Event(
        'workflow.urgent_email.completed',
        {'email_id': email_id, 'status': 'processed'}
    ))

event_bus.subscribe('workflow.urgent_email', handle_urgent_email)

# Step 3: Log completion
def log_completion(event):
    print(f"Urgent email {event.payload['email_id']} processed")

event_bus.subscribe('workflow.urgent_email.completed', log_completion)
```

## Thread Safety

The event bus is thread-safe:
- Subscriptions are protected by locks
- Multiple threads can safely publish and subscribe
- Handlers are called sequentially (not in parallel)

## Best Practices

1. **Use descriptive event types**: Follow `category.action` naming
2. **Keep handlers lightweight**: Long-running tasks should be async or queued
3. **Handle errors in handlers**: Don't let exceptions crash the system
4. **Use wildcard subscriptions sparingly**: They receive all events
5. **Document event payloads**: Define clear contracts for event data
6. **Avoid circular dependencies**: Don't publish events from handlers that could trigger themselves
7. **Clean up subscriptions**: Unsubscribe when handlers are no longer needed

## Testing with the Event Bus

```python
from rex.event_bus import EventBus, Event

def test_event_handling():
    # Create isolated event bus for testing
    event_bus = EventBus()

    received_events = []

    def handler(event):
        received_events.append(event)

    event_bus.subscribe('test.event', handler)
    event_bus.publish(Event('test.event', {'data': 'test'}))

    assert len(received_events) == 1
    assert received_events[0].payload['data'] == 'test'
```

## Future Enhancements

Planned improvements for the event bus:

- Event filtering by payload fields
- Event priority levels
- Async handler support
- Event persistence and replay
- Event middleware/interceptors
- Dead letter queue for failed events
- Event metrics and monitoring
- Event schema validation
