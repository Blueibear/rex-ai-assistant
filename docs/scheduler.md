# Scheduler

The Rex AI Assistant includes a job scheduler that enables automated task execution on recurring schedules. The scheduler provides persistent storage, background execution, and integration with the workflow system and event bus.

## Overview

The scheduler allows you to:
- Schedule jobs to run at regular intervals
- Execute callbacks or trigger workflows
- Persist job definitions across restarts
- Run jobs in a background thread
- Track execution history and limits

## Architecture

The scheduler consists of two main components:

1. **ScheduledJob**: A Pydantic model representing a scheduled job
2. **Scheduler**: The main scheduler class that manages jobs and execution

## ScheduledJob Model

A `ScheduledJob` contains the following fields:

```python
class ScheduledJob(BaseModel):
    job_id: str                    # Unique identifier
    name: str                      # Human-readable name
    schedule: str                  # Schedule specification
    enabled: bool = True           # Whether the job is active
    next_run: datetime             # Next scheduled execution time
    max_runs: Optional[int] = None # Maximum executions (None = unlimited)
    run_count: int = 0             # Number of times executed
    callback_name: Optional[str]   # Name of callback function
    workflow_id: Optional[str]     # Workflow to trigger
    metadata: dict = {}            # Additional metadata
```

### Schedule Format

Currently, the scheduler supports interval-based schedules:

- `interval:60` - Run every 60 seconds
- `interval:600` - Run every 10 minutes (600 seconds)
- `interval:3600` - Run every hour (3600 seconds)

Future versions may support cron-like syntax.

## Using the Scheduler

### Getting the Scheduler Instance

```python
from rex.scheduler import get_scheduler

scheduler = get_scheduler()
```

### Adding Jobs

#### With a Callback Function

```python
def my_callback(job):
    print(f"Job {job.job_id} is running!")
    # Do work here

scheduler.register_callback('my_callback', my_callback)

job = scheduler.add_job(
    job_id='my-job',
    name='My Job',
    schedule='interval:600',  # Every 10 minutes
    callback_name='my_callback',
    enabled=True
)
```

#### With a Workflow

```python
job = scheduler.add_job(
    job_id='workflow-job',
    name='Workflow Job',
    schedule='interval:3600',  # Every hour
    workflow_id='my-workflow-id',
    enabled=True
)
```

#### With Metadata and Limits

```python
job = scheduler.add_job(
    job_id='limited-job',
    name='Limited Job',
    schedule='interval:60',
    callback_name='my_callback',
    max_runs=10,  # Run only 10 times
    metadata={'priority': 'high', 'tags': ['important']},
    enabled=True
)
```

### Managing Jobs

```python
# List all jobs
jobs = scheduler.list_jobs()

# Get a specific job
job = scheduler.get_job('my-job')

# Update a job
scheduler.update_job('my-job', name='Updated Name', enabled=False)

# Remove a job
scheduler.remove_job('my-job')

# Run a job immediately (ignoring schedule)
scheduler.run_job('my-job', force=True)
```

### Starting and Stopping

```python
# Start the scheduler background thread
scheduler.start()

# Stop the scheduler
scheduler.stop()
```

## Persistence

Jobs are automatically persisted to `data/scheduler/jobs.json` and reloaded on startup. This ensures that jobs survive application restarts.

## CLI Commands

The scheduler provides several CLI commands:

### Initialize Scheduler

Set up the scheduler with default email and calendar jobs:

```bash
rex scheduler init
```

### List Jobs

Display all scheduled jobs:

```bash
rex scheduler list
rex scheduler list -v  # Verbose output with callbacks/workflows
```

### Run Job Immediately

Execute a job without waiting for its scheduled time:

```bash
rex scheduler run <job_id>
```

## Default Jobs

When initialized via `rex.integrations.initialize_scheduler_system()`, the scheduler automatically creates two default jobs:

### Email Check Job

- **ID**: `email_check`
- **Schedule**: Every 10 minutes (600 seconds)
- **Action**: Fetches unread emails, categorizes them, and publishes an `email.unread` event

### Calendar Sync Job

- **ID**: `calendar_sync`
- **Schedule**: Every hour (3600 seconds)
- **Action**: Fetches upcoming events and publishes a `calendar.update` event

## Integration with Event Bus

The scheduler integrates with the event bus to trigger actions when jobs complete. Default jobs publish events that can be subscribed to:

```python
from rex.event_bus import get_event_bus

event_bus = get_event_bus()

def handle_email_event(event):
    emails = event.payload['emails']
    print(f"Received {len(emails)} new emails")

event_bus.subscribe('email.unread', handle_email_event)
```

## Integration with Workflows

Jobs can trigger workflows by specifying a `workflow_id`. When the job runs, it will execute the specified workflow using the workflow engine.

## Example: Custom Job

```python
from datetime import datetime
from rex.scheduler import get_scheduler
from rex.event_bus import get_event_bus, Event

# Get instances
scheduler = get_scheduler()
event_bus = get_event_bus()

# Define callback
def check_system_health(job):
    # Perform health check
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'status': 'healthy',
        'checks': ['database', 'api', 'cache']
    }

    # Publish event
    event = Event(
        event_type='system.health',
        payload=health_status
    )
    event_bus.publish(event)

# Register callback
scheduler.register_callback('check_health', check_system_health)

# Add job (every 5 minutes)
scheduler.add_job(
    job_id='health-check',
    name='System Health Check',
    schedule='interval:300',
    callback_name='check_health',
    enabled=True
)

# Start scheduler
scheduler.start()
```

## Thread Safety

The scheduler is thread-safe and uses locks to protect shared state. Multiple threads can safely add, remove, and update jobs.

## Error Handling

If a job's callback raises an exception:
- The error is logged
- The job's `next_run` is still updated
- Other jobs continue to execute normally
- The scheduler remains running

## Best Practices

1. **Use descriptive job IDs**: Make them unique and meaningful
2. **Register callbacks early**: Register callbacks before adding jobs that use them
3. **Handle errors in callbacks**: Jobs should handle their own errors gracefully
4. **Use appropriate intervals**: Don't schedule jobs too frequently (minimum 1 second)
5. **Clean up jobs**: Remove jobs that are no longer needed
6. **Use max_runs for one-time tasks**: Set `max_runs=1` for jobs that should run once
7. **Monitor job execution**: Use the CLI or logging to track job execution

## Future Enhancements

Planned improvements for the scheduler:

- Cron-like scheduling syntax
- Job dependencies (run job B after job A)
- Job priorities
- Retry logic with exponential backoff
- Job execution timeouts
- Web UI for job management
- Job execution statistics and reporting
