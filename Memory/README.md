# Memory Profiles

This directory holds per-user memory profiles for AskRex Assistant.

## Structure

Each user has a subdirectory created automatically at first login:

```
Memory/<user_id>/
    core.json          # User identity, preferences, voice config (generated at runtime)
    notes.md           # Freeform notes about the user (generated at runtime)
    history.jsonl      # Conversation history (generated at runtime)
    history_meta.json  # History metadata (generated at runtime)
```

## Policy — Do NOT Commit Memory Profile Subdirectories

Memory profile files are **generated at runtime** by AskRex and contain personal
preferences and conversation data. They must never be committed to git.

`.gitignore` excludes the entire `Memory/` tree except this README file. If a
profile directory somehow appears as untracked, add it to your local
`.git/info/exclude` or ensure the `.gitignore` rule is in effect.

## Creating a Profile Template for Development

To create a placeholder profile for testing purposes, create a directory with
clearly fictional data and add the directory path to `.git/info/exclude` so it
is never accidentally committed:

```json
{
  "name": "Alex Example",
  "user": "alex_example",
  "role": "Test user",
  "personality_traits": ["curious"],
  "preferences": {
    "tone": "casual",
    "conversation_style": "brief",
    "topics": ["testing"]
  },
  "voice": {
    "sample_path": "",
    "speaker_name": "",
    "gender": "unspecified",
    "style": "neutral"
  },
  "testing_mode": true,
  "created_at": "2025-01-01T00:00:00Z",
  "last_updated": "2025-01-01T00:00:00Z",
  "notes_path": "Memory/alex_example/notes.md"
}
```

Real user data must never be committed to this repository.
