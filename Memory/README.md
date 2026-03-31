# Memory Profiles

This directory contains per-user memory profiles for Rex AI Assistant.

## Structure

Each user has a subdirectory with the following files:

```
Memory/<user_id>/
    core.json          # User identity, preferences, voice config
    notes.md           # Freeform notes about the user
    history.jsonl      # Conversation history (generated at runtime)
    history_meta.json  # History metadata (generated at runtime)
```

## Example Profiles

The subdirectories in this repository (`alice/`, `cole/`, `james/`, `voice-user/`) are
**example profiles with synthetic data only**. They exist to demonstrate the directory
structure and schema — they do not contain real personal information.

Do not commit real user profiles to this repository. Runtime memory files
(`history.jsonl`, `history_meta.json`, `history.log`) are automatically excluded
by `.gitignore`.

## Adding a Real User

Copy `Memory/james/` as a template and replace the example values in `core.json`:

```json
{
  "name": "Your Display Name",
  "user": "your_user_id",
  "role": "Owner and primary user",
  ...
}
```

Real user data should never be committed to git. Add `Memory/<your_user_id>/` to
your local `.git/info/exclude` if needed.
