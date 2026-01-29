# Conversational Follow-Up Engine v1

## Overview
The Conversational Follow-Up Engine provides lightweight, deterministic follow-up prompts based on past calendar events and optional reminder follow-ups. It **does not use email-based cues** in this version. Follow-ups are generated from calendar events that ended within a recent lookback window and from reminders that request a follow-up question.

Key behaviors:
- Uses persistent storage in `data/followups/`.
- Works on Windows 11 and other supported platforms (paths are `Path`-based).
- Does not add new dependencies.
- Respects safe defaults when configuration keys are missing.

## Configuration
Configuration lives under `conversation.followups` in `config/rex_config.json`:

```json
{
  "conversation": {
    "followups": {
      "enabled": false,
      "max_per_session": 2,
      "lookback_hours": 72,
      "expire_hours": 168
    }
  }
}
```

Settings:
- `enabled` (bool, default `false`): Turns follow-up injection on or off.
- `max_per_session` (int, default `2`): Maximum follow-up prompts injected per assistant session.
- `lookback_hours` (int, default `72`): How far back to look for completed calendar events.
- `expire_hours` (int, default `168`): Expiration window for removing old cues.

Values are clamped to safe ranges in the follow-up engine to avoid errors from invalid configuration values.

## CLI Commands
### Reminders
- Add a reminder:
  ```bash
  rex reminders add "Call mom" --at "2024-01-02 09:00" --follow-up
  ```

- List reminders:
  ```bash
  rex reminders list
  rex reminders list --status pending
  ```

- Mark a reminder done:
  ```bash
  rex reminders done <id>
  ```

- Cancel a reminder:
  ```bash
  rex reminders cancel <id>
  ```

### Cues
- List cues:
  ```bash
  rex cues list
  rex cues list --status pending
  ```

- Dismiss a cue:
  ```bash
  rex cues dismiss <cue_id>
  ```

- Prune expired cues:
  ```bash
  rex cues prune
  ```

## How it Works
1. **Calendar cues**: When enabled, the engine inspects calendar events that ended within `lookback_hours`. Each qualifying event produces a cue like: “How did ‘Event Title’ go?”. Cues are deduplicated by event ID and won’t be re-created on subsequent runs.
2. **Reminders**: `rex reminders add ... --follow-up` creates a reminder plus a follow-up cue tied to that reminder. The cue becomes due at the reminder time.
3. **Injection**: During a chat session, the assistant injects at most `max_per_session` cues into the prompt. Asked cues are marked as `asked` and will not repeat in the same session.
4. **Expiration**: Old cues are pruned based on `expire_hours`.

## Data Storage
- Cues: `data/followups/cues.json`
- Reminders: `data/followups/reminders.json`

These files use stable IDs and persist across runs.
