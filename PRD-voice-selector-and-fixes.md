# PRD: GUI Voice Selector, Transcription Fix, Voice Listener Fix, and Real-Time Context Tools

## Introduction

This PRD covers four workstreams for the Rex AI Assistant Electron/React GUI:

1. **Voice model selector with sample playback** -- a dropdown in Settings > Voice that lists all voices available from the active TTS provider (XTTS, edge-tts, pyttsx3) and lets the user preview each one.
2. **Slow Chat-tab transcription** -- the mic/STT path in the Chat tab takes unreasonably long for short sentences because `rex_chat_bridge.py` spawns a fresh Python process (loading Whisper, services, etc.) per message and the Whisper model size may be too large for the hardware.
3. **Voice tab "Start Listening" does nothing** -- clicking "Start Listening" updates the UI state but the `rex_voice_bridge.py` falls through to the stub loop because `VoiceLoop` requires constructor arguments (`assistant`, `wake_listener`, `detection_source`, `record_phrase`, `transcribe`, `speak`) that the bridge never provides; it only passes `on_state_change` and `on_transcript`, which are not part of the `VoiceLoop.__init__` signature.
4. **Time, date, and weather accuracy** -- `_resolve_timezone` only knows four Dallas aliases and falls back to UTC for everything else; `weather_now` is a stub that always returns "not implemented"; and there is no default-location config or geolocation fallback.

## Goals

- Users can browse and audition all TTS voices from the active provider without editing config files.
- Chat-tab speech-to-text completes within 3 seconds for a short sentence on CPU.
- The Voice tab's "Start Listening" button activates real microphone capture and wake word detection.
- Rex accurately reports the current time, date, and weather for any major city worldwide, and uses the user's configured home location when no city is specified.

## User Stories

### US-001: Add `default_location` and `weather_api_key` to AppConfig
**Description:** As a developer, I need config fields for the user's home location and OpenWeatherMap API key so that time/weather tools can resolve a default location and call the weather API.

**Acceptance Criteria:**
- [x] `AppConfig` in `rex/config.py` gains `default_location: Optional[str] = None` (e.g. "Dallas, TX") and `default_timezone: Optional[str] = None` (e.g. "America/Chicago")
- [x] `AppConfig` gains `openweathermap_api_key: Optional[str] = None`
- [x] `rex_config.json` example in docs shows new fields under a `location` section and `openweathermap_api_key` under secrets
- [x] `config_manager.py` maps the new JSON keys to `AppConfig` fields
- [x] Existing tests in `tests/` still pass
- [x] Typecheck passes

### US-002: Implement IP geolocation fallback for default location
**Description:** As a user, I want Rex to auto-detect my approximate location on startup so that time and weather queries work without manual config.

**Acceptance Criteria:**
- [x] New module `rex/geolocation.py` with `async def detect_location() -> dict` that calls a free IP geolocation API (e.g. `http://ip-api.com/json/`) and returns `{"city": str, "timezone": str, "lat": float, "lon": float}`
- [x] Function has a 3-second timeout and returns `None` on failure
- [x] If `default_location` is set in config, geolocation is skipped
- [x] If `default_location` is not set, geolocation result is cached in memory for the session
- [x] No new heavy dependencies (uses stdlib `urllib.request` or existing `httpx`/`requests`)
- [x] Unit test with mocked HTTP response
- [x] Typecheck passes

### US-003: Expand `_resolve_timezone` to cover all major cities
**Description:** As a user, I want Rex to correctly resolve the timezone for any major city so that time queries are accurate worldwide.

**Acceptance Criteria:**
- [x] `_resolve_timezone` in `rex/tool_router.py` uses a curated dict of 200+ city-to-timezone mappings instead of the current 4-entry `city_aliases` dict
- [x] Falls back to `default_timezone` from config, then to geolocation result, then to UTC
- [x] `_execute_time_now` also returns the current date in its result dict (adds `"date"` key with `YYYY-MM-DD` format)
- [x] Existing `time_now` tests updated to cover at least 5 diverse cities (London, Tokyo, Sydney, New York, Mumbai)
- [x] Typecheck passes

### US-004: Implement `weather_now` tool with OpenWeatherMap
**Description:** As a user, I want Rex to give me current weather conditions when I ask, so I can plan my day.

**Acceptance Criteria:**
- [x] New module `rex/weather.py` with `async def get_weather(location: str, api_key: str) -> dict` that calls OpenWeatherMap Current Weather API
- [x] Returns `{"temp_f": float, "temp_c": float, "description": str, "humidity": int, "wind_mph": float, "city": str}`
- [x] Handles API errors gracefully, returning a clear error dict
- [x] `_execute_weather_now` added to `rex/tool_router.py`, wired into `execute_tool`
- [x] Weather tool health check in `tool_registry.py` checks for `openweathermap_api_key` credential
- [x] When no location is provided, uses `default_location` from config or geolocation fallback
- [x] Unit test with mocked API response
- [x] Typecheck passes

### US-005: Inject current date/time into LLM system prompt
**Description:** As a user, I want Rex to always know the current date and time without needing a tool call, so casual questions like "what day is it?" get instant answers.

**Acceptance Criteria:**
- [x] `rex/assistant.py` `_build_prompt()` (or equivalent system prompt builder) prepends `Current date and time: {YYYY-MM-DD HH:MM} {timezone}` and `User location: {city}` to the system message
- [x] Values sourced from config `default_timezone`/`default_location` or geolocation cache
- [x] LLM can answer "what time is it?" and "what's today's date?" without issuing a tool call
- [x] Existing assistant tests still pass
- [x] Typecheck passes

### US-006: Add TTS voice listing endpoint to Python backend
**Description:** As a developer, I need a backend function that enumerates available voices for the active TTS provider so the GUI can populate a dropdown.

**Acceptance Criteria:**
- [x] New module `rex/tts_voices.py` with `def list_voices(provider: str) -> list[dict]` returning `[{"id": str, "name": str, "language": str, "gender": str | None}]`
- [x] For `xtts`: lists speaker WAV files in the XTTS speakers directory
- [x] For `edge-tts`: calls `edge_tts.list_voices()` (async, cached)
- [x] For `pyttsx3`: enumerates `engine.getProperty('voices')`
- [x] Falls back to empty list if provider dependencies are missing
- [x] New IPC bridge script `rex_voices_bridge.py` that outputs JSON list to stdout (same pattern as `rex_chat_bridge.py`)
- [x] Unit test for each provider path (mocked)
- [x] Typecheck passes

### US-007: Add TTS sample playback endpoint to Python backend
**Description:** As a developer, I need a backend function that synthesizes a short sample phrase in a given voice so the GUI can play a preview.

**Acceptance Criteria:**
- [x] `rex/tts_voices.py` gains `async def synthesize_sample(provider: str, voice_id: str, text: str = "Hello, I'm Rex.") -> bytes` returning WAV audio bytes
- [x] For `xtts`: uses the selected speaker WAV as reference
- [x] For `edge-tts`: uses the selected voice name
- [x] For `pyttsx3`: uses the selected voice ID
- [x] Returns audio as base64-encoded WAV in a new bridge script `rex_voice_sample_bridge.py` (stdin: `{"voice_id": "...", "provider": "..."}`, stdout: `{"ok": true, "audio_base64": "..."}`)
- [x] Sample text limited to 50 characters max to keep synthesis fast
- [x] Typecheck passes

### US-008: Add voice selector dropdown and preview button to Electron Settings > Voice
**Description:** As a user, I want a dropdown in the Voice settings panel to pick my preferred TTS voice and hear a sample before committing.

**Acceptance Criteria:**
- [x] New IPC handlers `rex:listVoices` and `rex:previewVoice` registered in `gui/src/main/handlers/voice.ts`
- [x] `rex:listVoices` spawns `rex_voices_bridge.py` and returns the voice list
- [x] `rex:previewVoice` spawns `rex_voice_sample_bridge.py`, receives base64 WAV, sends it to renderer
- [x] Settings > Voice section in `SettingsPage.tsx` gains a `<select>` dropdown populated by `rex:listVoices` on mount
- [x] Each option shows voice name and language
- [x] A "Preview" button next to the dropdown calls `rex:previewVoice` and plays the returned audio via Web Audio API
- [x] Selecting a voice updates `tts_voice` in settings and persists to config
- [x] Loading state shown while voices are being fetched
- [x] Typecheck passes
- [x] Verify changes work in browser

### US-009: Fix `rex_voice_bridge.py` to use real VoiceLoop with proper constructor args
**Description:** As a user, I want the Voice tab's "Start Listening" button to actually activate the microphone, wake word detection, STT, LLM, and TTS pipeline.

**Acceptance Criteria:**
- [x] `rex_voice_bridge.py` `main()` constructs a `VoiceLoop` with all required args: `assistant`, `wake_listener`, `detection_source`, `record_phrase`, `transcribe`, `speak`
- [x] Uses `build_voice_loop()` from `rex/voice_loop.py` if available, or manually wires the components using the same pattern as `rex_loop.py`
- [x] State changes (`idle` / `listening` / `processing` / `speaking`) are emitted as NDJSON to stdout
- [x] User transcripts and Rex replies are emitted as NDJSON transcript events
- [x] Falls back to stub mode only when voice dependencies are genuinely missing (not due to wrong constructor signature)
- [x] Manual test: click "Start Listening", speak a phrase, verify transcript appears in UI
- [x] Typecheck passes

### US-010: Fix Chat-tab transcription latency
**Description:** As a user, I want speech-to-text in the Chat tab to complete quickly for short sentences instead of taking an unreasonably long time.

**Acceptance Criteria:**
- [x] Root cause identified and documented (likely: `rex_chat_bridge.py` spawns a new Python process per message, reloading Whisper model each time; or Whisper model size is `large` instead of `base`/`small`)
- [x] If cause is per-message process spawn: refactor to a persistent bridge process (NDJSON protocol like `rex_voice_bridge.py`) that keeps the Whisper model loaded
- [x] If cause is model size: ensure `whisper_model` config defaults to `base` and document recommended sizes for CPU vs GPU
- [x] Add a `rex:sendChatAudio` IPC handler that accepts audio data and returns the transcript, reusing the persistent process
- [x] Transcription of a 5-word sentence completes in under 3 seconds on CPU with `base` model
- [x] Existing Chat text input still works unchanged
- [x] Typecheck passes
- [ ] Verify changes work in browser

### US-011: Add mic button to Chat tab input area
**Description:** As a user, I want a microphone button in the Chat tab so I can dictate messages without switching to the Voice tab.

**Acceptance Criteria:**
- [ ] `ChatInput.tsx` gains a mic icon button to the left of the send button
- [ ] Clicking the mic button starts browser-based audio capture via `navigator.mediaDevices.getUserMedia`
- [ ] Visual indicator shows recording state (pulsing red dot or mic icon color change)
- [ ] On stop (click again or 5-second silence timeout), audio is sent to the backend via `rex:sendChatAudio` IPC
- [ ] Transcribed text is inserted into the text input field (user can review before sending)
- [ ] If microphone permission is denied, a toast notification explains the issue
- [ ] Typecheck passes
- [ ] Verify changes work in browser

### US-012: Integration test for time/weather/date accuracy
**Description:** As a developer, I want integration tests that verify Rex answers time, date, and weather questions correctly end-to-end.

**Acceptance Criteria:**
- [ ] Test file `tests/test_time_weather_integration.py`
- [ ] Test that `execute_tool({"tool": "time_now", "args": {"location": "London"}}, {})` returns a valid datetime in `Europe/London` timezone
- [ ] Test that `execute_tool({"tool": "time_now", "args": {}}, {"location": "Dallas"})` falls back to `America/Chicago`
- [ ] Test that `execute_tool({"tool": "weather_now", "args": {"location": "New York"}}, ctx)` returns temperature and description (mocked API)
- [ ] Test that the system prompt contains the current date string
- [ ] All tests pass with `pytest -q tests/test_time_weather_integration.py`
- [ ] Typecheck passes

## Non-Goals

- No support for custom/cloned XTTS voices in this iteration (only pre-existing speaker files).
- No real-time streaming STT (Whisper processes complete audio chunks, not live streams).
- No multi-language weather/time support beyond English.
- No GUI for editing `default_location` or `openweathermap_api_key` (use `rex_config.json` or `.env` for now; GUI settings for these can come later).
- No push notifications or proactive weather alerts.
- No offline weather data.

## Technical Considerations

- **VoiceLoop constructor mismatch**: The current `rex_voice_bridge.py` passes `on_state_change` and `on_transcript` kwargs to `VoiceLoop.__init__`, but `VoiceLoop` expects `assistant`, `wake_listener`, `detection_source`, `record_phrase`, `transcribe`, `speak`. The bridge must use `build_voice_loop()` or manually wire the pipeline like `rex_loop.py` does.
- **Whisper model loading cost**: Loading Whisper takes 2-10 seconds depending on model size. The chat bridge spawns a new Python process per message, paying this cost every time. A persistent process is essential.
- **TTS voice enumeration**: XTTS voices are WAV files in a directory; edge-tts provides an async `list_voices()` coroutine; pyttsx3 provides `engine.getProperty('voices')`. Each path needs different handling.
- **OpenWeatherMap free tier**: 1,000 calls/day, which is sufficient for personal use. API key goes in `.env` as `OPENWEATHERMAP_API_KEY`.
- **Timezone resolution**: A curated 200+ city dict avoids adding a heavy dependency (`timezonefinder`). If greater coverage is needed later, `timezonefinder` can be added as an optional enhancement.
- **Audio playback in Electron renderer**: The voice sample preview can use the Web Audio API to decode and play base64 WAV data returned from the backend. No additional Electron dependencies needed.
- **IPC bridge pattern**: All new Python bridges should follow the established NDJSON-over-stdio pattern used by `rex_voice_bridge.py` and `rex_chat_bridge.py`.
