# Hardening Rex for Reliable Operation

## Service supervision
Rex runs critical services under the `ServiceSupervisor` (scheduler, event bus, workflow runner, memory store, credential manager). The supervisor:
- Starts/stops services and provides automatic restarts with exponential backoff.
- Exposes health endpoints at `/health`, `/ready`, and `/metrics`.

### Health endpoints
- `GET /health`: JSON summary with service status, uptime, and restart counts.
- `GET /ready`: Returns 200 if all services are running.
- `GET /metrics`: JSON metrics for the supervisor and each service.

## Idempotency guarantees
Rex ensures repeated calls do not produce duplicate side effects:
- Scheduler jobs track scheduled runs and will not re-run the same scheduled slot.
- Workflow steps respect idempotency keys, skipping already executed steps.
- Executors skip workflows that are already finished.
- Notification delivery can use `idempotency_key` to prevent duplicate sends.

## Retry policies
Transient failures are retried with exponential backoff:
- GitHub API requests retry on `requests` exceptions.
- Notification dispatch retries on transient channel failures.

## Metrics and observability
The metrics endpoint aggregates:
- Scheduler job totals, successes, failures, and due job queue depth.
- Event bus publish counts and handler error counts.
- Memory store statistics.

## Recommended operations
- Use `rex-run` (or `python -m rex.app`) as the primary runtime entrypoint.
- For systemd deployments, rely on the generated service unit to keep the supervisor running.
- Monitor `/metrics` alongside log aggregation for uptime and reliability reporting.
