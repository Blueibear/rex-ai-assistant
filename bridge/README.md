# bridge/

These are Electron IPC bridge processes. Each is spawned by the Electron main
process and communicates over stdin/stdout JSON.

The Electron main process resolves paths via `resolveBridgePath()` in
`gui/src/main/bridgeResolver.ts`. All spawn calls route through that function —
bridge filenames are never hardcoded inline in handler files.

## Files

| Script | Purpose |
|---|---|
| `rex_calendar_bridge.py` | Calendar read/write via configured backend |
| `rex_chat_bridge.py` | Single-turn LLM chat |
| `rex_chat_stream_bridge.py` | Streaming LLM chat |
| `rex_context_policy_bridge.py` | Owner-bound context/privacy settings and safe metadata |
| `rex_email_bridge.py` | Email read/send via configured backend |
| `rex_file_extract_bridge.py` | File content extraction |
| `rex_history_bridge.py` | Per-user command history read |
| `rex_ha_mutation_bridge.py` | Policy-controlled Home Assistant mutations and verification |
| `rex_identity_bridge.py` | Immutable Electron session identity resolution |
| `rex_memories_bridge.py` | User memory read/write |
| `rex_reminders_bridge.py` | Reminder CRUD |
| `rex_shopping_list_bridge.py` | Shopping list CRUD |
| `rex_sms_bridge.py` | SMS send via configured backend |
| `rex_speaker_bridge.py` | Speaker/audio device enumeration |
| `rex_stt_bridge.py` | Speech-to-text transcription |
| `rex_tasks_bridge.py` | Task CRUD |
| `rex_tts_bridge.py` | Text-to-speech synthesis *(reserved)* |
| `rex_voice_bridge.py` | Voice loop control |
| `rex_voice_enrollment_bridge.py` | Voice identity enrollment |
| `rex_voice_sample_bridge.py` | Voice sample recording |
| `rex_voice_upload_bridge.py` | Voice sample upload |
| `rex_voices_bridge.py` | Voice profile listing |
| `rex_wakeword_list_bridge.py` | Wake-word model listing |
| `rex_wakeword_sample_bridge.py` | Wake-word sample recording |
| `rex_wakeword_train_bridge.py` | Wake-word model training |
