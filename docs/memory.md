# Memory System

Rex includes a comprehensive memory system with two types of memory:

1. **Working Memory** - Short-term buffer for recent interactions and context
2. **Long-Term Memory** - Structured storage with categories, expiration, and search

## Overview

The memory system enables Rex to:
- Maintain context across conversations
- Store user preferences and facts
- Remember important information with automatic cleanup
- Protect sensitive data from being exposed

## Working Memory

Working memory stores recent interactions and task summaries, providing immediate context for conversations.

### Features

- Ordered list of recent entries (most recent last)
- Automatic persistence to disk
- Configurable maximum entries (default: 100)
- Loads automatically on startup

### Usage

```python
from rex.memory import get_working_memory, remember_context, get_recent_context

# Get the working memory instance
wm = get_working_memory()

# Add an entry
wm.add_entry("User asked about weather in Dallas")

# Get recent entries (content only)
recent = wm.get_recent(5)  # Returns list of strings

# Get entries with timestamps
entries = wm.get_recent_with_timestamps(5)
# Returns: [{"content": "...", "timestamp": "..."}, ...]

# Clear all entries
wm.clear()

# Convenience functions
remember_context("User prefers dark mode")
context = get_recent_context(3)
```

### CLI Commands

```bash
# Show recent working memory entries
rex memory recent 10

# Clear working memory
rex memory clear

# Show memory statistics
rex memory stats
```

## Long-Term Memory

Long-term memory stores structured entries organized by category with support for expiration, search, and sensitive data protection.

### Features

- Categorized entries (preferences, facts, etc.)
- Optional expiration (TTL)
- Sensitive data flagging
- Keyword search across content
- Automatic cleanup of expired entries

### Memory Entry Structure

Each entry contains:
- `entry_id`: Unique identifier
- `category`: Category name (e.g., "preferences", "facts")
- `content`: Dictionary with the stored data
- `created_at`: Creation timestamp
- `expires_at`: Optional expiration timestamp
- `sensitive`: Boolean flag for sensitive data

### Usage

```python
from datetime import timedelta
from rex.memory import (
    get_long_term_memory,
    add_user_preference,
    get_user_preferences,
)

# Get the long-term memory instance
ltm = get_long_term_memory()

# Add an entry
entry = ltm.add_entry(
    category="preferences",
    content={"theme": "dark", "language": "en"},
)

# Add with expiration
entry = ltm.add_entry(
    category="temp_data",
    content={"session_id": "abc123"},
    expires_in=timedelta(hours=24),
)

# Add sensitive data
entry = ltm.add_entry(
    category="secrets",
    content={"api_key": "secret123"},
    sensitive=True,
)

# Search entries
results = ltm.search(category="preferences")
results = ltm.search(keyword="theme")
results = ltm.search(category="preferences", keyword="dark")

# Get a specific entry
entry = ltm.get_entry("mem_abc123")

# Delete an entry
ltm.forget("mem_abc123")

# Run retention policy (delete expired)
deleted_count = ltm.run_retention_policy()

# List categories
categories = ltm.list_categories()

# Count by category
counts = ltm.count_by_category()

# Convenience functions
add_user_preference("notification_sound", "chime")
prefs = get_user_preferences("notification")
```

### CLI Commands

```bash
# Add a long-term memory entry
rex memory add "preferences" '{"theme": "dark"}'

# Add with expiration (7 days)
rex memory add "temp" '{"key": "value"}' --ttl=7d

# Add sensitive entry
rex memory add "secrets" '{"api_key": "xxx"}' --sensitive

# Search entries
rex memory search "theme"
rex memory search --category preferences

# Search and show sensitive content
rex memory search "api_key" --show-sensitive

# Delete an entry
rex memory forget mem_abc123

# Run retention policy
rex memory retention

# Show statistics
rex memory stats
```

### TTL Formats

When specifying expiration times:
- `7d` - 7 days
- `24h` - 24 hours
- `30m` - 30 minutes
- `2w` - 2 weeks
- `10s` - 10 seconds

## Sensitive Data Handling

Entries marked as sensitive are protected:

1. Content is hidden by default in CLI output
2. Use `--show-sensitive` flag to view content
3. The `to_safe_dict()` method returns redacted content
4. Sensitive entries are included in searches but content is hidden

```python
# Check if entry is sensitive
if entry.sensitive:
    # Use safe dict for logging
    safe_data = entry.to_safe_dict()
    # safe_data["content"] = {"[SENSITIVE]": "Content hidden"}
```

## Retention Policies

Expired entries are automatically managed:

1. **On Startup**: Expired entries are deleted when LongTermMemory loads
2. **Manual**: Call `run_retention_policy()` to clean up
3. **CLI**: Run `rex memory retention`

Entries without `expires_at` never expire.

## Storage

Memory is persisted to JSON files:

- Working Memory: `data/memory/working_memory.json`
- Long-Term Memory: `data/memory/long_term_memory.json`

### Storage Format

**Working Memory:**
```json
{
  "entries": [
    {
      "content": "User asked about weather",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ]
}
```

**Long-Term Memory:**
```json
{
  "entries": [
    {
      "entry_id": "mem_abc123",
      "category": "preferences",
      "content": {"theme": "dark"},
      "created_at": "2024-01-15T10:30:00Z",
      "expires_at": null,
      "sensitive": false
    }
  ]
}
```

## Categories

Common categories for organizing entries:

| Category | Description |
|----------|-------------|
| `user_preferences` | User settings and preferences |
| `facts` | Learned facts about the user or environment |
| `context` | Conversational context |
| `secrets` | Sensitive credentials (mark as sensitive!) |
| `temp_data` | Temporary data with expiration |

## Best Practices

1. **Use categories consistently** - Group related entries
2. **Set expiration for temporary data** - Prevent unbounded growth
3. **Mark sensitive data** - Protect credentials and personal info
4. **Clean up regularly** - Run retention policy periodically
5. **Use convenience functions** - `add_user_preference()` for common tasks

## Integration

The memory system integrates with other Rex components:

```python
from rex.memory import remember_context, get_recent_context

# In assistant.py - track conversation context
async def generate_reply(self, user_input: str) -> str:
    # Add to working memory
    remember_context(f"User: {user_input}")

    # Get recent context for LLM
    context = get_recent_context(5)

    # ... generate reply ...

    remember_context(f"Rex: {reply}")
    return reply
```

## API Reference

### WorkingMemory

| Method | Description |
|--------|-------------|
| `add_entry(content)` | Add a new entry |
| `get_recent(n)` | Get last n entries (content only) |
| `get_recent_with_timestamps(n)` | Get last n entries with timestamps |
| `clear()` | Remove all entries |

### LongTermMemory

| Method | Description |
|--------|-------------|
| `add_entry(category, content, expires_in, sensitive)` | Add entry |
| `get_entry(entry_id)` | Get entry by ID |
| `search(category, keyword, include_sensitive, include_expired)` | Search entries |
| `forget(entry_id)` | Delete entry |
| `run_retention_policy()` | Delete expired entries |
| `list_categories()` | List all categories |
| `count_by_category()` | Count entries per category |

### Convenience Functions

| Function | Description |
|----------|-------------|
| `add_user_preference(key, value, ...)` | Add user preference |
| `get_user_preferences(key)` | Get user preferences |
| `add_fact(topic, content, ...)` | Add a fact |
| `remember_context(summary)` | Add to working memory |
| `get_recent_context(n)` | Get recent working memory |

---

## User Profiles & Personalization

Each user has a dedicated profile in `Memory/<username>/`:

```
Memory/james/
├── core.json         # User preferences and voice settings
├── history.log       # Conversation history
└── notes.md          # Freeform notes about the user
```

**Example `core.json`:**
```json
{
  "name": "James",
  "email": "james@example.com",
  "preferences": {
    "preferred_name": "Jim",
    "timezone": "America/New_York"
  },
  "voice": {
    "sample_path": "Memory/james/voice_sample.wav",
    "gender": "male",
    "style": "friendly and warm"
  }
}
```

Rex uses voice cloning with XTTS when a valid `voice.sample_path` is provided.
