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


# ---------------------------------------------------------------------------
# Unreliable-model detection tests
# ---------------------------------------------------------------------------


def _quiet_result(
    triggered: bool = False,
    confidence: float = 0.90,
    rms: float = 0.003,
    peak: float = 0.012,
    threshold: float = 0.5,
) -> WakeWordDetectionResult:
    return WakeWordDetectionResult(
        triggered=triggered,
        threshold=threshold,
        predictions={"hey_rex": confidence},
        confidence=confidence,
        keyword="hey_rex",
        reason="confidence_met_threshold" if triggered else "below_threshold",
        audio_rms=rms,
        audio_peak=peak,
    )


@pytest.mark.unit
def test_wakeword_listener_unreliable_detected_after_min_frames():
    from rex.wakeword.listener import (
        _UNRELIABLE_CONFIDENCE_THRESHOLD,
        _UNRELIABLE_MIN_FRAMES,
        _UNRELIABLE_PEAK_MAX,
        _UNRELIABLE_RMS_MAX,
    )

    def detector(_frame):
        return _quiet_result(
            confidence=_UNRELIABLE_CONFIDENCE_THRESHOLD + 0.01,
            rms=_UNRELIABLE_RMS_MAX * 0.5,
            peak=_UNRELIABLE_PEAK_MAX * 0.5,
        )

    state: dict = {}

    async def run():
        call_count = 0
        listener = WakeWordListener(detector, poll_interval=0, unreliable_silence_enabled=True)
        state["listener"] = listener

        async def source():
            nonlocal call_count
            call_count += 1
            if call_count > _UNRELIABLE_MIN_FRAMES:
                listener.stop()
            return np.zeros(16000, dtype=np.float32)

        async for _ in listener.listen(source):
            break

    asyncio.run(run())
    assert state["listener"]._model_marked_unreliable is True


@pytest.mark.unit
def test_wakeword_listener_unreliable_not_triggered_below_min_frames():
    from rex.wakeword.listener import (
        _UNRELIABLE_CONFIDENCE_THRESHOLD,
        _UNRELIABLE_MIN_FRAMES,
        _UNRELIABLE_PEAK_MAX,
        _UNRELIABLE_RMS_MAX,
    )

    def detector(_frame):
        return _quiet_result(
            confidence=_UNRELIABLE_CONFIDENCE_THRESHOLD + 0.01,
            rms=_UNRELIABLE_RMS_MAX * 0.5,
            peak=_UNRELIABLE_PEAK_MAX * 0.5,
        )

    state: dict = {}

    async def run():
        call_count = 0
        # Stop on the (MIN_FRAMES - 1)th source call so the loop exits after processing
        # exactly MIN_FRAMES - 1 quiet frames (one short of activation).
        listener = WakeWordListener(detector, poll_interval=0, unreliable_silence_enabled=True)
        state["listener"] = listener

        async def source():
            nonlocal call_count
            call_count += 1
            if call_count >= _UNRELIABLE_MIN_FRAMES - 1:
                listener.stop()
            return np.zeros(16000, dtype=np.float32)

        async for _ in listener.listen(source):
            break

    asyncio.run(run())
    assert state["listener"]._model_marked_unreliable is False
    assert state["listener"]._high_confidence_quiet_frames == _UNRELIABLE_MIN_FRAMES - 1


@pytest.mark.unit
def test_wakeword_listener_loud_audio_does_not_increment_unreliable_counter():
    from rex.wakeword.listener import (
        _UNRELIABLE_CONFIDENCE_THRESHOLD,
        _UNRELIABLE_MIN_FRAMES,
        _UNRELIABLE_RMS_MAX,
    )

    def detector(_frame):
        # High confidence but loud audio — RMS well above the quiet threshold.
        return _quiet_result(
            confidence=_UNRELIABLE_CONFIDENCE_THRESHOLD + 0.01,
            rms=_UNRELIABLE_RMS_MAX * 10,  # loud
            peak=0.15,
        )

    state: dict = {}

    async def run():
        call_count = 0
        listener = WakeWordListener(detector, poll_interval=0, unreliable_silence_enabled=True)
        state["listener"] = listener

        async def source():
            nonlocal call_count
            call_count += 1
            if call_count > _UNRELIABLE_MIN_FRAMES + 2:
                listener.stop()
            return np.zeros(16000, dtype=np.float32)

        async for _ in listener.listen(source):
            break

    asyncio.run(run())
    assert state["listener"]._model_marked_unreliable is False
    assert state["listener"]._high_confidence_quiet_frames == 0


@pytest.mark.unit
def test_wakeword_listener_fallback_factory_called_on_activation():
    from rex.wakeword.listener import (
        _UNRELIABLE_CONFIDENCE_THRESHOLD,
        _UNRELIABLE_MIN_FRAMES,
        _UNRELIABLE_PEAK_MAX,
        _UNRELIABLE_RMS_MAX,
    )

    factory_calls: list[int] = []

    def fallback_detector(_frame):
        return _quiet_result(triggered=False, confidence=0.1)

    def fallback_factory():
        factory_calls.append(1)
        return fallback_detector, None, "hey jarvis"

    def detector(_frame):
        return _quiet_result(
            confidence=_UNRELIABLE_CONFIDENCE_THRESHOLD + 0.01,
            rms=_UNRELIABLE_RMS_MAX * 0.5,
            peak=_UNRELIABLE_PEAK_MAX * 0.5,
        )

    state: dict = {}

    async def run():
        call_count = 0
        listener = WakeWordListener(
            detector,
            poll_interval=0,
            unreliable_silence_enabled=True,
            fallback_detector_factory=fallback_factory,
            fallback_keyword="hey jarvis",
            fallback_backend="openwakeword",
        )
        state["listener"] = listener

        async def source():
            nonlocal call_count
            call_count += 1
            if call_count > _UNRELIABLE_MIN_FRAMES + 1:
                listener.stop()
            return np.zeros(16000, dtype=np.float32)

        async for _ in listener.listen(source):
            break

    asyncio.run(run())
    assert factory_calls == [1]
    assert state["listener"]._model_marked_unreliable is True
    assert state["listener"]._keyword == "hey jarvis"
    assert state["listener"]._backend == "openwakeword"


@pytest.mark.unit
def test_wakeword_listener_fallback_activation_events_emitted():
    from rex.wakeword.listener import (
        _UNRELIABLE_CONFIDENCE_THRESHOLD,
        _UNRELIABLE_MIN_FRAMES,
        _UNRELIABLE_PEAK_MAX,
        _UNRELIABLE_RMS_MAX,
    )

    events_emitted: list[str] = []

    def event_callback(payload: dict) -> None:
        extra = payload.get("extra", {})
        if isinstance(extra, dict):
            event_name = str(extra.get("event", ""))
            if event_name:
                events_emitted.append(event_name)

    def fallback_detector(_frame):
        return _quiet_result(triggered=False, confidence=0.1)

    def fallback_factory():
        return fallback_detector, None, "hey jarvis"

    def detector(_frame):
        return _quiet_result(
            confidence=_UNRELIABLE_CONFIDENCE_THRESHOLD + 0.01,
            rms=_UNRELIABLE_RMS_MAX * 0.5,
            peak=_UNRELIABLE_PEAK_MAX * 0.5,
        )

    async def run():
        call_count = 0
        listener = WakeWordListener(
            detector,
            poll_interval=0,
            unreliable_silence_enabled=True,
            fallback_detector_factory=fallback_factory,
            fallback_keyword="hey jarvis",
            fallback_backend="openwakeword",
            event_callback=event_callback,
        )

        async def source():
            nonlocal call_count
            call_count += 1
            if call_count > _UNRELIABLE_MIN_FRAMES + 1:
                listener.stop()
            return np.zeros(16000, dtype=np.float32)

        async for _ in listener.listen(source):
            break

    asyncio.run(run())
    assert "high_confidence_silence" in events_emitted
    assert "wakeword_backend_fallback_activated" in events_emitted


@pytest.mark.unit
def test_wakeword_listener_no_second_activation_after_unreliable():
    from rex.wakeword.listener import (
        _UNRELIABLE_CONFIDENCE_THRESHOLD,
        _UNRELIABLE_MIN_FRAMES,
        _UNRELIABLE_PEAK_MAX,
        _UNRELIABLE_RMS_MAX,
    )

    factory_calls: list[int] = []

    def fallback_detector(_frame):
        return _quiet_result(triggered=False, confidence=0.1)

    def fallback_factory():
        factory_calls.append(1)
        return fallback_detector, None, "hey jarvis"

    def detector(_frame):
        return _quiet_result(
            confidence=_UNRELIABLE_CONFIDENCE_THRESHOLD + 0.01,
            rms=_UNRELIABLE_RMS_MAX * 0.5,
            peak=_UNRELIABLE_PEAK_MAX * 0.5,
        )

    state: dict = {}

    async def run():
        call_count = 0
        # Run 3× MIN_FRAMES to ensure we'd see a second activation if the guard failed.
        listener = WakeWordListener(
            detector,
            poll_interval=0,
            unreliable_silence_enabled=True,
            fallback_detector_factory=fallback_factory,
            fallback_keyword="hey jarvis",
            fallback_backend="openwakeword",
        )
        state["listener"] = listener

        async def source():
            nonlocal call_count
            call_count += 1
            if call_count > _UNRELIABLE_MIN_FRAMES * 3:
                listener.stop()
            return np.zeros(16000, dtype=np.float32)

        async for _ in listener.listen(source):
            break

    asyncio.run(run())
    assert factory_calls == [1], "fallback factory must be called exactly once"


@pytest.mark.unit
def test_wakeword_listener_trigger_suppressed_on_activation_frame():
    from rex.wakeword.listener import (
        _UNRELIABLE_CONFIDENCE_THRESHOLD,
        _UNRELIABLE_MIN_FRAMES,
        _UNRELIABLE_PEAK_MAX,
        _UNRELIABLE_RMS_MAX,
    )

    # Detector: not triggered on frames 1 to N-1, triggered on frame N (activation frame).
    # The activation frame should be suppressed — no yield.
    frames_yielded: list[int] = []
    call_count_holder = [0]

    def detector(_frame):
        n = call_count_holder[0]
        is_activation_frame = n >= _UNRELIABLE_MIN_FRAMES
        return _quiet_result(
            triggered=is_activation_frame,
            confidence=_UNRELIABLE_CONFIDENCE_THRESHOLD + 0.01,
            rms=_UNRELIABLE_RMS_MAX * 0.5,
            peak=_UNRELIABLE_PEAK_MAX * 0.5,
        )

    def fallback_detector(_frame):
        return _quiet_result(triggered=False, confidence=0.1)

    def fallback_factory():
        return fallback_detector, None, "hey jarvis"

    async def run():
        call_count = 0
        listener = WakeWordListener(
            detector,
            poll_interval=0,
            unreliable_silence_enabled=True,
            fallback_detector_factory=fallback_factory,
            fallback_keyword="hey jarvis",
            fallback_backend="openwakeword",
        )

        async def source():
            nonlocal call_count
            call_count += 1
            call_count_holder[0] = call_count
            if call_count > _UNRELIABLE_MIN_FRAMES + 2:
                listener.stop()
            return np.zeros(16000, dtype=np.float32)

        async for _frame in listener.listen(source):
            frames_yielded.append(1)
            break  # stop after first yield if any

    asyncio.run(run())
    assert frames_yielded == [], "activation frame must be suppressed — no yield expected"


@pytest.mark.unit
def test_wakeword_listener_self_test_inactive_after_window_elapsed():
    """Quiet high-confidence frames outside the self-test window are ignored."""
    import time as _time

    from rex.wakeword.listener import (
        _UNRELIABLE_CONFIDENCE_THRESHOLD,
        _UNRELIABLE_MIN_FRAMES,
        _UNRELIABLE_PEAK_MAX,
        _UNRELIABLE_RMS_MAX,
        _UNRELIABLE_WINDOW_S,
    )

    factory_calls: list[int] = []

    def fallback_factory():
        factory_calls.append(1)
        return (lambda _frame: _quiet_result(confidence=0.1)), None, "hey jarvis"

    def detector(_frame):
        return _quiet_result(
            confidence=_UNRELIABLE_CONFIDENCE_THRESHOLD + 0.01,
            rms=_UNRELIABLE_RMS_MAX * 0.5,
            peak=_UNRELIABLE_PEAK_MAX * 0.5,
        )

    state: dict = {}

    async def run():
        call_count = 0
        listener = WakeWordListener(
            detector,
            poll_interval=0,
            unreliable_silence_enabled=True,
            fallback_detector_factory=fallback_factory,
            fallback_keyword="hey jarvis",
            fallback_backend="openwakeword",
        )
        state["listener"] = listener
        listener.mark_listening_started(reason="test")
        # Simulate a listening cycle that started well past the self-test window.
        listener._listening_started_at = _time.monotonic() - (_UNRELIABLE_WINDOW_S + 5.0)

        async def source():
            nonlocal call_count
            call_count += 1
            if call_count > _UNRELIABLE_MIN_FRAMES * 2:
                listener.stop()
            return np.zeros(16000, dtype=np.float32)

        async for _ in listener.listen(source):
            break

    asyncio.run(run())
    assert state["listener"]._model_marked_unreliable is False
    assert state["listener"]._high_confidence_quiet_frames == 0
    assert factory_calls == []


@pytest.mark.unit
def test_wakeword_listener_stays_on_fallback_across_cycles():
    """A model marked unreliable stays on the fallback in later cycles."""
    from rex.wakeword.listener import (
        _UNRELIABLE_CONFIDENCE_THRESHOLD,
        _UNRELIABLE_MIN_FRAMES,
        _UNRELIABLE_PEAK_MAX,
        _UNRELIABLE_RMS_MAX,
    )

    factory_calls: list[int] = []

    def fallback_detector(_frame):
        return _quiet_result(triggered=False, confidence=0.1)

    def fallback_factory():
        factory_calls.append(1)
        return fallback_detector, None, "hey jarvis"

    def detector(_frame):
        return _quiet_result(
            confidence=_UNRELIABLE_CONFIDENCE_THRESHOLD + 0.01,
            rms=_UNRELIABLE_RMS_MAX * 0.5,
            peak=_UNRELIABLE_PEAK_MAX * 0.5,
        )

    state: dict = {}

    async def run_cycle(listener, frames):
        call_count = 0

        async def source():
            nonlocal call_count
            call_count += 1
            if call_count > frames:
                listener.stop()
            return np.zeros(16000, dtype=np.float32)

        async for _ in listener.listen(source):
            break

    async def run():
        listener = WakeWordListener(
            detector,
            poll_interval=0,
            unreliable_silence_enabled=True,
            fallback_detector_factory=fallback_factory,
            fallback_keyword="hey jarvis",
            fallback_backend="openwakeword",
        )
        state["listener"] = listener
        # First cycle: enough quiet high-confidence frames to activate fallback.
        await run_cycle(listener, _UNRELIABLE_MIN_FRAMES + 1)
        assert listener._model_marked_unreliable is True
        assert factory_calls == [1]

        # New listening cycle: counter resets but the fallback stays active.
        listener.mark_listening_started(reason="post_interaction_reset")
        assert listener._model_marked_unreliable is True
        assert listener._high_confidence_quiet_frames == 0
        assert listener._backend == "openwakeword"
        assert listener._keyword == "hey jarvis"

        # Second cycle sees more quiet frames; no second activation may fire.
        await run_cycle(listener, _UNRELIABLE_MIN_FRAMES * 2)

    asyncio.run(run())
    assert state["listener"]._model_marked_unreliable is True
    assert factory_calls == [1]
    assert state["listener"]._backend == "openwakeword"
