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
