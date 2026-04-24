from __future__ import annotations

import asyncio
import logging

import pytest

np = pytest.importorskip("numpy")

from rex.wakeword.listener import WakeWordListener, build_default_detector  # noqa: E402
from rex.wakeword.utils import WakeWordDetectionResult  # noqa: E402


@pytest.mark.unit
def test_wakeword_listener_resets_detector_after_accepted_wake():
    resets: list[bool] = []

    def detector(_frame):
        return WakeWordDetectionResult(
            triggered=True,
            threshold=0.03,
            predictions={"hey jarvis": 0.9},
            confidence=0.9,
            keyword="hey jarvis",
            reason="confidence_met_threshold",
        )

    async def source():
        return np.zeros(16000, dtype=np.float32)

    async def run_once():
        listener = WakeWordListener(
            detector,
            poll_interval=0,
            reset_detector=lambda: resets.append(True),
        )
        async for _frame in listener.listen(source):
            break

    asyncio.run(run_once())

    assert resets == [True]


@pytest.mark.unit
def test_wakeword_listener_resets_detection_buffer_after_accepted_wake():
    buffer_resets: list[str] = []

    def detector(_frame):
        return WakeWordDetectionResult(
            triggered=True,
            threshold=0.03,
            predictions={"hey jarvis": 0.9},
            confidence=0.9,
            keyword="hey jarvis",
            reason="confidence_met_threshold",
        )

    class SourceOwner:
        def reset_detection_buffer(self, *, reason: str = "manual") -> None:
            buffer_resets.append(reason)

        async def frame(self):
            return np.zeros(16000, dtype=np.float32)

    async def run_once():
        listener = WakeWordListener(detector, poll_interval=0)
        owner = SourceOwner()
        async for _frame in listener.listen(owner.frame):
            break

    asyncio.run(run_once())

    assert buffer_resets == ["accepted_wake"]


@pytest.mark.unit
def test_wakeword_listener_logs_attempt_details(caplog):
    calls = 0

    def detector(_frame):
        nonlocal calls
        calls += 1
        if calls > 1:
            return WakeWordDetectionResult(
                triggered=True,
                threshold=0.03,
                predictions={"hey jarvis": 0.9},
                confidence=0.9,
                keyword="hey jarvis",
                reason="confidence_met_threshold",
            )
        return WakeWordDetectionResult(
            triggered=False,
            threshold=0.03,
            predictions={"hey jarvis": 0.01},
            confidence=0.01,
            keyword="hey jarvis",
            reason="below_threshold",
        )

    async def source():
        return np.zeros(16000, dtype=np.float32)

    async def run_once():
        listener = WakeWordListener(detector, poll_interval=0)
        async for _frame in listener.listen(source):
            break

    with caplog.at_level(logging.DEBUG, logger="rex.wakeword.listener"):
        asyncio.run(run_once())

    records = [
        record for record in caplog.records if getattr(record, "event", None) == "wakeword_attempt"
    ]
    assert records
    assert records[0].confidence == 0.01
    assert records[0].threshold == 0.03
    assert records[0].accepted is False
    assert records[0].levelno == logging.INFO
    assert records[0].listening_cycle == 1
    assert records[0].time_since_wake_listening_start_s >= 0
    assert records[0].early_listening_attempt is True


@pytest.mark.unit
def test_wakeword_listener_logs_loop_enter_and_exit(caplog):
    def detector(_frame):
        return WakeWordDetectionResult(
            triggered=True,
            threshold=0.03,
            predictions={"hey jarvis": 0.9},
            confidence=0.9,
            keyword="hey jarvis",
            reason="confidence_met_threshold",
        )

    async def source():
        return np.zeros(16000, dtype=np.float32)

    async def run_once():
        listener = WakeWordListener(
            detector,
            poll_interval=0,
            threshold=0.03,
            keyword="hey jarvis",
            backend="test",
        )
        async for _frame in listener.listen(source):
            break

    with caplog.at_level(logging.INFO, logger="rex.wakeword.listener"):
        asyncio.run(run_once())

    events = [getattr(record, "event", None) for record in caplog.records]
    assert "wakeword_listener_loop_entered" in events
    assert "wakeword_listener_loop_exited" in events


@pytest.mark.unit
def test_wakeword_listener_rebuilds_detector_after_interaction_when_reset_missing(caplog):
    generations: list[int] = []

    def make_detector():
        generation = len(generations) + 1
        generations.append(generation)

        def detector(_frame):
            return WakeWordDetectionResult(
                triggered=False,
                threshold=0.03,
                predictions={f"gen-{generation}": 0.0},
                confidence=0.0,
                keyword=f"gen-{generation}",
                reason="below_threshold",
            )

        return detector, None, f"gen-{generation}"

    detector, reset_detector, keyword = make_detector()
    listener = WakeWordListener(
        detector,
        poll_interval=0,
        reset_detector=reset_detector,
        rebuild_detector=make_detector,
        threshold=0.03,
        keyword=keyword,
        backend="test",
    )

    with caplog.at_level(logging.INFO, logger="rex.wakeword.listener"):
        listener.reset(reason="accepted_wake")
        assert generations == [1]
        listener.reset(reason="post_interaction")

    assert generations == [1, 2]
    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "wakeword_detector_rebuilt"
    ]
    assert records
    assert records[-1].threshold == 0.03
    assert records[-1].detector_generation == 2


@pytest.mark.unit
def test_wakeword_listener_rebuilds_after_interaction_even_when_reset_supported(caplog):
    generations: list[int] = []
    resets: list[bool] = []

    def make_detector():
        generation = len(generations) + 1
        generations.append(generation)

        def detector(_frame):
            return WakeWordDetectionResult(
                triggered=False,
                threshold=0.03,
                predictions={f"gen-{generation}": 0.0},
                confidence=0.0,
                keyword=f"gen-{generation}",
                reason="below_threshold",
            )

        return detector, lambda: resets.append(True), f"gen-{generation}"

    detector, reset_detector, keyword = make_detector()
    listener = WakeWordListener(
        detector,
        poll_interval=0,
        reset_detector=reset_detector,
        rebuild_detector=make_detector,
        threshold=0.03,
        keyword=keyword,
        backend="test",
    )

    with caplog.at_level(logging.INFO, logger="rex.wakeword.listener"):
        listener.reset(reason="accepted_wake")
        listener.reset(reason="post_interaction")

    assert generations == [1, 2]
    assert resets == []
    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "wakeword_detector_rebuilt"
    ]
    assert records
    assert records[-1].detector_generation == 2


@pytest.mark.unit
def test_build_default_detector_uses_configured_threshold(monkeypatch, caplog):
    class FakeModel:
        def predict(self, _frame):
            return {"hey jarvis": 0.04}

    def fake_load_wakeword_model(**_kwargs):
        return FakeModel(), "hey jarvis"

    monkeypatch.setattr(
        "rex.wakeword.listener.load_wakeword_model",
        fake_load_wakeword_model,
    )

    async def source():
        return np.zeros(16000, dtype=np.float32)

    async def run_once():
        listener = build_default_detector(
            sample_rate=16000,
            chunk_duration=1.0,
            threshold=0.03,
            poll_interval=0,
        )
        async for _frame in listener.listen(source):
            break

    with caplog.at_level(logging.INFO, logger="rex.wakeword.listener"):
        asyncio.run(run_once())

    instance_records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "wakeword_detector_instance_ready"
    ]
    attempt_records = [
        record for record in caplog.records if getattr(record, "event", None) == "wakeword_attempt"
    ]
    assert instance_records[-1].threshold == 0.03
    assert attempt_records[-1].threshold == 0.03
    assert attempt_records[-1].accepted is True
