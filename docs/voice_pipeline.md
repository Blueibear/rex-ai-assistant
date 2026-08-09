# Voice Pipeline Structured Log Contract

AskRex emits a stable set of structured timing records for each successful voice interaction. These records are additive to the existing human-readable voice logs and `VoiceLatencyTracker` summary; operators can use the canonical event names below without scraping log messages.

## Record fields

Every canonical voice-pipeline record includes:

- `event` ? one of the canonical event names below.
- `session_id` ? the process-scoped AskRex logging session identifier. It is stable for the lifetime of the running process.
- `interaction_id` ? the monotonically increasing voice interaction number within the `VoiceLoop` instance.
- `start_ns` ? an integer timestamp from `time.monotonic_ns()`. It is for elapsed-time calculations only and is not a wall-clock timestamp.
- `duration_ms` ? present on completed/ended stages where an elapsed interval is meaningful. It is a floating-point number of milliseconds.

When JSON logging is enabled, these fields appear under the record's `extra` object.

## Canonical events

| Event | Meaning | `duration_ms` |
|---|---|---|
| `wake_detected` | An activation was accepted and an interaction began. The historical event name is retained for compatibility; in default Hold-to-Talk mode this means manual activation, not wake-word detection. | No |
| `capture_started` | Command-audio capture started. | No |
| `capture_ended` | Command-audio capture returned. | Yes, capture interval |
| `stt_started` | The captured audio was handed to STT. | No |
| `stt_completed` | STT returned successfully. | Yes, STT interval |
| `llm_started` | Assistant response generation started. | No |
| `llm_completed` | Assistant response generation completed. | Yes, LLM interval; on the streaming path the scope overlaps TTS/playback and is labeled `timing_scope=streaming_llm_tts_playback` |
| `tts_started` | Spoken-response synthesis/playback handling began. | No |
| `playback_completed` | The configured speak/streaming-speak operation returned successfully, meaning AskRex finished the response-audio operation it can directly observe. | Yes, TTS/playback operation interval |

## Streaming overlap

The streaming path deliberately overlaps LLM token generation, sentence buffering, TTS, and playback. For that path `tts_started` may occur before `llm_completed`, and the `llm_completed` and `playback_completed` records carry `timing_scope=streaming_llm_tts_playback`. Those durations therefore describe the observable combined streaming interval rather than pretending the overlapping components were isolated.

## Failure behavior

A completion event is emitted only when that stage completes successfully. Existing `pipeline_timeout`, `stt_error`, `tts_error`, and other stage-specific error records remain the failure contract. AskRex must not emit a successful completion record for a timed-out or failed stage.


## Synthetic CI latency budgets

`tests/test_voice_latency_budget.py` exercises the real canonical `VoiceLoop` with synthetic,
non-hardware callbacks. The budgets below are regression guards for AskRex orchestration and
stage hand-offs; they are not end-user hardware, network, or model-provider SLAs. Real Whisper,
LLM, TTS, audio-driver, and speaker latency is tracked separately by runtime telemetry and the
[performance baseline](performance.md).

The first release uses intentionally wide margins so shared CI runners do not create false
failures while still catching accidental blocking calls, sleeps, or serial regressions.

| Synthetic stage | Budget key | Maximum |
|---|---|---|
| Activation accepted -> capture start | `activation_to_capture` | 250 ms |
| Capture callback | `capture` | 500 ms |
| STT callback | `stt` | 500 ms |
| LLM response callback | `llm` | 500 ms |
| TTS/playback callback | `tts_playback` | 500 ms |
| Activation -> playback complete | `total` | 2000 ms |

The budget test runs in the default pytest marker set because it uses synthetic callbacks and
completes in well under one second. If a budget is intentionally changed, update this table and
the enforced constants in the same change so documentation and CI cannot drift apart.

## Wake-word reliability evidence

US-046 adds a tracked synthesized acoustic fixture and report at `docs/voice/wakeword-report.md`. The built-in openWakeWord path is evaluated at the configured `0.50` activation threshold, with a `0.90` precision and `0.90` recall promotion threshold. The controlled fixture currently measures precision `0.800` and recall `1.000`, so wake-word remains beta. The fixture is a regression baseline, not a substitute for broader microphone, room-noise, distance, accent, and continuous-negative-audio deployment testing.

Longer Rex microphone windows are split into openWakeWord's native 1,280-sample (80 ms at 16 kHz) streaming chunks before peak aggregation so a wake phrase is not lost merely because its high-confidence frame occurs before the end of a one-second capture window.
