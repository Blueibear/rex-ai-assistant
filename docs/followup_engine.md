# Conversational Follow-Up Engine v1

## Overview
The Conversational Follow-Up Engine provides lightweight, deterministic follow-up prompts to make Rex feel more natural and personal.

This v1 engine generates follow-ups from:
- **Recent calendar events** that ended within a configurable lookback window
- **Reminders** that request a follow-up question (created via `--follow-up`)

This version **does not use email-based cues**.

Key behaviors:
- Uses persistent storage in `data/followups/`
- Works on Windows 11 and other supported platforms (all paths are `Path`-based)
- Adds no new dependencies
- Uses safe defaults when configuration keys are missing
- Clamps configuration values to safe ranges to avoid crashes from invalid configs

## What is a Cue?
A **Cue** is a small record that Rex can bring up in conversation.

Cues are:
- **Per user/profile**: each user has their own cues
- **Persisted**: survive restarts
- **Consumable**: asked once, then marked as asked
- **Rate-limited**: capped per session via configuration
- **Time-windowed**: only considered within a configurable lookback window
- **Deduplicated**: calendar cues are deduped by the calendar event ID

## Configuration
Configuration lives under `conversation.followups` in `config/rex_config.json`:

```json
{
  "conversation": {
    "followups": {
      "enabled": false,
      "max_per_session": 1,
      "lookback_hours": 72,
      "expire_hours": 168
    }
  }
}

