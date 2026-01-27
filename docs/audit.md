# Audit Log System

Rex includes a comprehensive audit logging system that records all tool executions for accountability, traceability, and debugging purposes.

## Overview

The audit log captures:

- **Every tool invocation** - Whether successful or failed
- **Policy decisions** - Allowed, denied, or requiring approval
- **Execution timing** - Duration of each tool call
- **Arguments and results** - With automatic redaction of sensitive data

## Components

### LogEntry Model

Each audit log entry contains:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | When the action occurred (UTC) |
| `action_id` | string | Unique identifier for this action |
| `task_id` | string? | Parent task ID, if any |
| `tool` | string | Name of the tool invoked |
| `tool_call_args` | dict | Arguments passed to the tool (redacted) |
| `policy_decision` | string | "allowed", "denied", or "requires_approval" |
| `tool_result` | dict? | Result from tool execution (redacted) |
| `error` | string? | Error message if execution failed |
| `redacted` | bool | Whether sensitive data was redacted |
| `requested_by` | string? | Who/what initiated the action |
| `duration_ms` | int? | Execution time in milliseconds |

### AuditLogger Class

The `AuditLogger` class provides:

- **Persistent storage** in JSON Lines format
- **Thread-safe** write operations
- **Time-range queries** for log retrieval
- **Export functionality** for archival

## Usage

### Basic Logging

Logging happens automatically when tools are executed via `execute_tool()`. No manual intervention is required.

```python
from rex.tool_router import execute_tool

# Audit logging happens automatically
result = execute_tool(
    {"tool": "time_now", "args": {"location": "Dallas, TX"}},
    {},
)
```

### Manual Logging

For custom logging scenarios:

```python
from rex.audit import AuditLogger, LogEntry

logger = AuditLogger()

entry = LogEntry(
    action_id="act_001",
    tool="custom_tool",
    tool_call_args={"key": "value"},
    policy_decision="allowed",
    tool_result={"status": "success"},
)

logger.log(entry)
```

### Reading Logs

```python
from rex.audit import get_audit_logger
from datetime import datetime, timedelta, timezone

logger = get_audit_logger()

# Read all entries
all_entries = logger.read()

# Read entries from the last hour
one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
recent_entries = logger.read(start_time=one_hour_ago)

# Read entries within a specific range
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
end = datetime(2024, 1, 31, tzinfo=timezone.utc)
january_entries = logger.read(start_time=start, end_time=end)

# Read a specific entry by action ID
entry = logger.read_by_action_id("act_123abc456def")
```

### Exporting Logs

```python
from rex.audit import get_audit_logger

logger = get_audit_logger()

# Export all logs to a file
count = logger.export("/path/to/export/audit_backup.log")
print(f"Exported {count} entries")
```

## Storage Location

By default, audit logs are stored in:

```
data/logs/audit.log
```

This can be customized when creating an `AuditLogger` instance:

```python
from pathlib import Path
from rex.audit import AuditLogger

logger = AuditLogger(
    log_dir=Path("/custom/log/directory"),
    log_file="custom_audit.log",
)
```

## Redaction and Privacy

### Automatic Redaction

Sensitive data is automatically redacted before being written to the audit log. The following patterns are redacted:

- `token`, `access_token`, `refresh_token`
- `password`
- `secret`, `client_secret`
- `api_key`
- `authorization`

Pattern matching is **case-insensitive**, so `API_KEY`, `Api_Key`, and `api_key` are all redacted.

### Example

```python
# Original entry
entry = LogEntry(
    action_id="act_001",
    tool="api_call",
    tool_call_args={
        "url": "https://api.example.com",
        "headers": {
            "Authorization": "Bearer secret123",
            "Content-Type": "application/json",
        },
    },
    policy_decision="allowed",
)

# After logging and reading back:
# {
#     "url": "https://api.example.com",
#     "headers": {
#         "Authorization": "[REDACTED]",
#         "Content-Type": "application/json"
#     }
# }
```

### Nested Redaction

Redaction applies recursively to nested dictionaries and lists:

```python
{
    "config": {
        "database": {
            "host": "localhost",      # Preserved
            "password": "[REDACTED]"  # Redacted
        }
    }
}
```

## Replay Mechanism

### Current State (Stub)

The replay mechanism is currently a **stub implementation** that:

1. Reconstructs a `ToolCall` from a `LogEntry`
2. Returns a placeholder result
3. Does **not** actually execute tools

This is intentional - full replay with actual tool execution will be implemented in a future phase.

### Usage

```python
from rex.audit import get_audit_logger
from rex.replay import replay, batch_replay

logger = get_audit_logger()
entries = logger.read()

# Replay a single entry
if entries:
    result = replay(entries[0])
    print(f"Original tool: {result.original_entry.tool}")
    print(f"Notes: {result.notes}")

# Batch replay
results = batch_replay(entries[:10])
for r in results:
    print(f"Replayed {r.original_entry.action_id}")
```

### ReplayResult Structure

```python
@dataclass
class ReplayResult:
    original_entry: LogEntry      # The original audit log entry
    replayed_tool_call: ToolCall  # Reconstructed ToolCall
    new_result: dict | None       # Result from replay (stub)
    comparison: dict              # Comparison data
    dry_run: bool                 # Whether side effects were committed
    replayed_at: datetime         # When replay was performed
    notes: str                    # Additional information
```

### Future Extensions

Planned replay enhancements:

- **Actual tool execution** in dry-run mode (no side effects)
- **Result comparison** between original and replayed executions
- **Determinism testing** to verify consistent behavior
- **Debugging mode** with detailed execution tracing

## Skipping Audit Logging

In special cases (e.g., testing), audit logging can be skipped:

```python
result = execute_tool(
    {"tool": "time_now", "args": {"location": "Dallas"}},
    {},
    skip_audit_log=True,  # Don't log this execution
)
```

**Use with caution** - skipping audit logs reduces accountability.

## Dependencies

The audit system requires `pydantic>=2.0.0` for data validation and serialization. Ensure this dependency is installed:

```bash
pip install -e .
```

Or explicitly:

```bash
pip install pydantic>=2.0.0
```

## Testing

Run the audit tests:

```bash
# Run audit tests only
python -m pytest tests/test_audit.py -v

# Run replay tests only
python -m pytest tests/test_replay.py -v

# Run both
python -m pytest tests/test_audit.py tests/test_replay.py -v
```

## JSON Lines Format

The audit log uses [JSON Lines](https://jsonlines.org/) format - one JSON object per line. This format:

- Supports streaming reads
- Allows easy appending
- Works well with standard Unix tools

Example log content:

```json
{"timestamp":"2024-01-15T10:30:00Z","action_id":"act_001","tool":"time_now","tool_call_args":{"location":"Dallas, TX"},"policy_decision":"allowed","tool_result":{"local_time":"2024-01-15 10:30"},"redacted":true,"duration_ms":15}
{"timestamp":"2024-01-15T10:31:00Z","action_id":"act_002","tool":"weather_now","tool_call_args":{"location":"Dallas, TX"},"policy_decision":"allowed","error":"Tool weather_now is not implemented","redacted":true,"duration_ms":2}
```

## Best Practices

1. **Don't disable audit logging** in production
2. **Regularly export** logs for backup
3. **Monitor log size** and implement rotation if needed
4. **Review denied actions** to identify policy issues
5. **Use task IDs** to correlate related actions

## See Also

- [Policy Engine Documentation](policy.md) - How tool execution is governed
- [Contracts Documentation](contracts.md) - Data model definitions
