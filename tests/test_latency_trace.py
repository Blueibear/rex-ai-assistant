from __future__ import annotations

import logging

import pytest

from rex.latency import LatencyTrace


def test_latency_trace_accumulates_segments_and_uses_safe_metadata() -> None:
    ticks = iter(
        [1_000_000_000, 1_000_000_000, 1_010_000_000, 1_025_000_000, 1_040_000_000, 1_070_000_000]
    )
    trace = LatencyTrace(
        channel="chat",
        provider="ollama",
        model="qwen-test",
        settings_id="local-default",
        clock_ns=lambda: next(ticks),
    )

    trace.start("routing")
    trace.end("routing")
    trace.start("tool")
    trace.end("tool")
    trace.finish()

    summary = trace.summary()
    assert summary["channel"] == "chat"
    assert summary["provider"] == "ollama"
    assert summary["model"] == "qwen-test"
    assert summary["settings_id"] == "local-default"
    assert summary["routing_ms"] == pytest.approx(10.0)
    assert summary["tool_ms"] == pytest.approx(15.0)
    assert summary["total_ms"] == pytest.approx(70.0)
    serialized = repr(summary).lower()
    assert "prompt" not in serialized
    assert "transcript" not in serialized
    assert "user_id" not in serialized


def test_latency_trace_logs_structured_summary_without_payload(caplog) -> None:
    ticks = iter([2_000_000_000, 2_005_000_000])
    trace = LatencyTrace(
        channel="chat", provider="local", model="test", clock_ns=lambda: next(ticks)
    )
    trace.finish()

    logger = logging.getLogger("tests.latency")
    with caplog.at_level(logging.INFO, logger="tests.latency"):
        trace.log_summary(logger, event="chat_latency")

    record = caplog.records[-1]
    assert record.event == "chat_latency"
    assert record.channel == "chat"
    assert record.total_ms == pytest.approx(5.0)
    assert not hasattr(record, "prompt")
    assert not hasattr(record, "transcript")
    assert not hasattr(record, "user_id")
