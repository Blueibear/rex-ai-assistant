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

## Re-measuring

Run the following from the repository root to refresh measurements:

```bash
pytest tests/test_us120_performance_baseline.py -v -s
```

Update this document when the p50 values change by more than 2× from the
recorded baseline.
