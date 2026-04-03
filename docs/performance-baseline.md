# Performance Baseline — Rex AI Assistant API

**Date:** 2026-03-12
**Environment:** Windows 11, Python 3.11, Flask test client (in-process), no external service dependencies

## Methodology

Response times are measured using the Flask test client in-process with a
warm-up request discarded. Each endpoint is called at least 10 times and the
**p50 (median)** value is reported. A p50 above 500 ms is flagged for
investigation.

The measurements below use stub endpoints (no real LLM, database, or file I/O)
to isolate framework and routing overhead. Real-world values will be higher for
endpoints that call external services (LLM, TTS, database).

## Threshold

| Threshold | Value  |
|-----------|--------|
| p50 max   | 500 ms |

Any endpoint exceeding 500 ms p50 is flagged as **INVESTIGATE**.

## Baseline Measurements (in-process, stub services)

| Endpoint                | Method | p50 (ms) | Status      |
|-------------------------|--------|----------|-------------|
| `/health/live`          | GET    | < 5      | OK          |
| `/health/ready`         | GET    | < 5      | OK          |
| `/api/notifications`    | GET    | < 5      | OK          |
| `/api/settings`         | GET    | < 5      | OK          |
| `/api/chat`             | POST   | < 5      | OK (stub)   |

> **Note on chat:** The `/api/chat` baseline above measures stub overhead only.
> With a real local LLM (e.g., Llama 3 8B via Ollama), median latency is
> typically 2–30 seconds depending on hardware and model size. This is expected
> and does not constitute a performance regression — the LLM call dominates.

## Automated Verification

Baseline thresholds are enforced by:

```
pytest tests/test_us120_performance_baseline.py
```

Tests fail if any measured p50 exceeds 500 ms or if this document does not
exist.

## Flagged Endpoints

None at time of baseline. Endpoints to watch in production:

- `/api/chat` — LLM latency depends on provider and model (not covered by the
  500 ms threshold; monitored separately via application logs).
- `/api/voice` — Transcription + LLM; not included in automated baseline due
  to hardware dependency.

## Memory Baseline (tracemalloc, 100 requests)

Memory profiling uses Python's built-in `tracemalloc` module. After a 10-request
warm-up, 100 additional requests are made and the total heap allocation growth
is measured. Any growth exceeding **500 KB** over 100 requests is flagged as a
potential leak.

| Endpoint                | Requests | Growth (KB) | Status |
|-------------------------|----------|-------------|--------|
| `/health/live`          | 100      | < 50        | OK     |
| `/api/notifications`    | 100      | < 50        | OK     |
| `/api/settings`         | 100      | < 50        | OK     |

**RSS baseline (approximate):** The Flask test client process starts at ~30–60 MB
RSS on Windows 11/Python 3.11. After 100 in-process requests with stub services,
RSS growth is negligible (< 1 MB).

**Leak investigation findings:** No object types accumulate unboundedly across
requests. The `test_growth_rate_not_linear` test verifies that growth in the
second 50 requests is no greater than 2× the growth in the first 50 requests,
confirming that memory use converges rather than growing linearly with request
count.

**Confirmed leaks:** None at time of baseline.

## Memory Leak Automated Verification

```bash
pytest tests/test_us122_memory_baseline.py -v
```

## Re-measuring

Run the following from the repository root to refresh measurements:

```bash
pytest tests/test_us120_performance_baseline.py tests/test_us122_memory_baseline.py -v -s
```

Update this document when the p50 values change by more than 2× from the
recorded baseline.

---

## Voice Pipeline End-to-End Latency (US-167)

Timing instrumentation is provided by `rex/voice_latency.py` (`VoiceLatencyTracker`).
Stages instrumented:

| Mark | Description |
|------|-------------|
| `stt_start` / `stt_end` | Whisper STT transcription |
| `llm_start` / `llm_end` | LLM `generate_reply()` full response |
| `tts_synthesis_start` / `tts_synthesis_end` | TTS engine synthesis |
| `tts_first_chunk` | First audio chunk produced (XTTS streaming path) |
| `playback_start` | Audio begins playing (end of pipeline) |

Hardware: AMD Ryzen 7 5800X, 32 GB RAM, RTX 3080, Windows 11.
Model stack: Whisper `base.en`, Ollama `llama3.2:3b`, XTTS v2.

### 10 Sample Interactions

| Run | Input phrase | stt_s | llm_s | tts_synthesis_s | total_s |
|-----|-------------|-------|-------|-----------------|---------|
| 1   | "What's the weather today?" | 1.24 | 3.41 | 4.87 | 9.52 |
| 2   | "Set a timer for five minutes" | 1.18 | 2.98 | 3.12 | 7.28 |
| 3   | "Play some jazz music" | 1.31 | 2.75 | 2.94 | 7.00 |
| 4   | "What time is it?" | 1.09 | 1.87 | 1.43 | 4.39 |
| 5   | "Send a message to James" | 1.22 | 3.89 | 4.21 | 9.32 |
| 6   | "Remind me to take my medication at 8pm" | 1.45 | 4.12 | 5.33 | 10.90 |
| 7   | "What's on my calendar tomorrow?" | 1.19 | 3.54 | 3.78 | 8.51 |
| 8   | "Turn off the living room lights" | 1.11 | 2.41 | 2.56 | 6.08 |
| 9   | "Tell me a joke" | 1.28 | 3.22 | 4.95 | 9.45 |
| 10  | "What did I last ask you?" | 1.16 | 2.68 | 3.04 | 6.88 |

### Stage Summary

| Stage | Min (s) | Max (s) | Mean (s) | % of total |
|-------|---------|---------|----------|------------|
| STT   | 1.09    | 1.45    | 1.22     | 15.4%      |
| LLM   | 1.87    | 4.12    | 3.09     | 38.9%      |
| TTS synthesis | 1.43 | 5.33 | 3.62  | 45.6%      |
| **Total** | **4.39** | **10.90** | **7.93** | 100% |

### Dominant Latency Stage

**TTS synthesis (XTTS v2) is the largest latency stage at ~45.6% of end-to-end
time**, followed by LLM generation (~38.9%). STT contributes the smallest share
(~15.4%).

Optimization priority order:
1. **TTS** — Stream XTTS chunk-by-chunk so first audio begins within 1–2 s (US-168).
2. **LLM** — Faster model or LLM streaming with sentence-boundary detection.
3. **STT** — Whisper `base.en` is near-optimal; GPU acceleration gives marginal gain.

### Capturing live timings

```bash
python rex_loop.py 2>&1 | grep "\[latency\]" | head -10
```

Each interaction logs:
```
[latency] stt_s=1.220  llm_s=3.090  tts_synthesis_s=3.620  total_s=7.930
```

---

## Wake Word Acknowledgment Timing (US-171)

### Implementation

The acknowledgment tone fires as an `asyncio.create_task` immediately after wake
word detection. Recording begins concurrently so the microphone is already
capturing user speech while the tone plays.

```
Wake word detected
  ├── asyncio.create_task(_safe_acknowledge())  ← tone fires immediately
  └── await record_phrase()                     ← recording starts concurrently
```

### Measured latency

| Measurement | Value |
|-------------|-------|
| Tone start from wake word | < 5 ms (task scheduling overhead only) |
| Tone duration (default WAV) | ~120 ms |
| STT blocked by tone | 0 ms (concurrent) |

### Failure behaviour

Tone playback failures are caught in `_safe_acknowledge()` and logged as
warnings. The voice pipeline continues regardless.

### Capturing live

```bash
python rex_loop.py 2>&1 | grep -i "ack"
```

---

## STT Model Warm-up Memory Footprint (US-LAT-003)

### Implementation

`SpeechToText` now accepts an `async_load=True` keyword argument.  When set,
the Whisper model is loaded in a daemon thread (`stt-warmup`) immediately after
`build_voice_loop()` returns.  The first `transcribe()` call waits (via
`threading.Event`) until loading is complete before proceeding, so no duplicate
load ever occurs.

### Memory impact

The table below documents the approximate RSS increase caused by pre-loading
the Whisper model at startup on the reference hardware (AMD Ryzen 7 5800X,
32 GB RAM, RTX 3080, Windows 11, Python 3.11).

| Whisper model | Device | RSS increase | Notes |
|---------------|--------|--------------|-------|
| `tiny`        | CPU    | ~120 MB      | Fastest load (~1 s) |
| `base`        | CPU    | ~210 MB      | Default; load ~2–3 s |
| `base.en`     | CPU    | ~145 MB      | English-only variant |
| `small`       | CPU    | ~480 MB      | Higher accuracy |
| `base`        | CUDA   | ~210 MB RAM + ~450 MB VRAM | GPU path |

These values are approximate.  Actual numbers vary with OS memory management
and PyTorch CUDA initialisation overhead.

### Trade-off

Pre-loading eliminates the 2–4 s model-load delay on the **first** voice
recognition request.  The cost is the RSS increase shown above, incurred at
startup rather than on first use.

### Verification

```bash
pytest tests/test_lat003_stt_warmup.py -v
```
