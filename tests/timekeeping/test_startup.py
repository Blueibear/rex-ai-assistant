from __future__ import annotations

from tests.test_assistant_latency import _assistant


def test_assistant_starts_canonical_timekeeping_runtime(monkeypatch) -> None:
    started = []
    sentinel = object()

    def ensure_runtime():
        started.append(True)
        return sentinel

    monkeypatch.setattr("rex.timekeeping.runtime.ensure_timekeeping_runtime", ensure_runtime)

    assistant = _assistant(monkeypatch)

    assert started == [True]
    assert assistant._timekeeping_runtime is sentinel
