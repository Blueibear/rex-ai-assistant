"""Privacy-safe monotonic latency tracing for assistant request stages.

The trace intentionally stores only diagnostic metadata supplied through explicit
fields. User IDs, prompts, transcripts, tool payloads, and credentials are not
accepted by this API.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

ClockNs = Callable[[], int]


@dataclass
class LatencyTrace:
    """Collect monotonic stage timings without retaining request contents."""

    channel: str
    provider: str = "unknown"
    model: str = "unknown"
    settings_id: str = "default"
    clock_ns: ClockNs = time.perf_counter_ns
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _started_ns: int = field(init=False, repr=False)
    _finished_ns: int | None = field(default=None, init=False, repr=False)
    _open_stages: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _durations_ns: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._started_ns = self.clock_ns()

    def start(self, stage: str) -> None:
        """Start or restart a named stage."""
        self._open_stages[stage] = self.clock_ns()

    def end(self, stage: str) -> float:
        """Finish a stage and accumulate its duration in milliseconds."""
        started = self._open_stages.pop(stage, None)
        if started is None:
            raise ValueError(f"Latency stage was not started: {stage}")
        elapsed_ns = max(0, self.clock_ns() - started)
        self._durations_ns[stage] = self._durations_ns.get(stage, 0) + elapsed_ns
        return elapsed_ns / 1_000_000

    def add_duration_ms(self, stage: str, duration_ms: float) -> None:
        """Add an already-measured stage duration."""
        if duration_ms < 0:
            raise ValueError("Latency duration cannot be negative")
        self._durations_ns[stage] = self._durations_ns.get(stage, 0) + int(duration_ms * 1_000_000)

    def finish(self) -> None:
        """Seal total timing. Calling twice is harmless."""
        if self._finished_ns is None:
            self._finished_ns = self.clock_ns()

    def summary(self) -> dict[str, str | float]:
        """Return privacy-safe structured diagnostic fields."""
        end_ns = self._finished_ns if self._finished_ns is not None else self.clock_ns()
        result: dict[str, str | float] = {
            "trace_id": self.trace_id,
            "channel": self.channel,
            "provider": self.provider,
            "model": self.model,
            "settings_id": self.settings_id,
            "total_ms": round(max(0, end_ns - self._started_ns) / 1_000_000, 3),
        }
        for stage, duration_ns in sorted(self._durations_ns.items()):
            result[f"{stage}_ms"] = round(duration_ns / 1_000_000, 3)
        return result

    def log_summary(self, logger: logging.Logger, *, event: str = "latency_summary") -> None:
        """Emit one structured INFO record with no request payload data."""
        fields = self.summary()
        logger.info("[latency] %s", event, extra={"event": event, **fields})


__all__ = ["LatencyTrace"]
