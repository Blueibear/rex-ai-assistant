"""Tests for US-007: wake-to-capture stage logging and startup validation."""

from __future__ import annotations

import asyncio
import logging

import pytest

np = pytest.importorskip("numpy")

from rex.assistant_errors import WakeWordError  # noqa: E402
from rex.wakeword.listener import WakeWordListener, build_default_detector  # noqa: E402
from rex.wakeword.utils import detect_wakeword  # noqa: E402


class _AlwaysDetect:
    """Minimal model that always detects the wake word."""

    def predict(self, frame):
        return {"test": 1.0}


def _make_detector(model):
    def detector(frame):
        return detect_wakeword(model, frame, threshold=0.5)

    return detector


def _single_frame_source():
    frames = [np.ones(4, dtype=np.float32)]

    async def source():
        if frames:
            return frames.pop(0)
        return np.zeros(4, dtype=np.float32)

    return source


async def _collect_one(listener, source):
    """Collect the first yielded frame then stop the listener."""
    async for frame in listener.listen(source):
        listener.stop()
        return frame
    return None


def test_wake_word_detection_emits_structured_log(caplog):
    """AC 1: listener emits a structured log event on wake word detection."""
    detector = _make_detector(_AlwaysDetect())
    listener = WakeWordListener(detector, poll_interval=0)

    with caplog.at_level(logging.INFO, logger="rex.wakeword.listener"):
        asyncio.run(_collect_one(listener, _single_frame_source()))

    detected_records = [r for r in caplog.records if "Wake word detected" in r.getMessage()]
    assert detected_records, "Expected a structured log event for wake word detection"
    record = detected_records[0]
    assert getattr(record, "event", None) == "wakeword_detected"


def test_audio_capture_log_emitted_after_detection(caplog):
    """AC 2: audio_capture_start event is logged after wake word detection."""
    detector = _make_detector(_AlwaysDetect())
    listener = WakeWordListener(detector, poll_interval=0)

    with caplog.at_level(logging.DEBUG, logger="rex.wakeword.listener"):
        asyncio.run(_collect_one(listener, _single_frame_source()))

    events = [getattr(r, "event", None) for r in caplog.records]
    assert "audio_capture_start" in events, "Expected audio_capture_start log event"


def test_build_default_detector_raises_on_empty_keyword():
    """AC 3: empty keyword raises WakeWordError at startup (no silent hang)."""
    with pytest.raises(WakeWordError, match="must not be empty"):
        build_default_detector(sample_rate=16000, chunk_duration=0.1, keyword="")


def test_build_default_detector_raises_on_whitespace_keyword():
    """AC 3: whitespace-only keyword raises WakeWordError at startup."""
    with pytest.raises(WakeWordError, match="must not be empty"):
        build_default_detector(sample_rate=16000, chunk_duration=0.1, keyword="   ")


def test_wake_to_capture_transition_with_mock_stream(caplog):
    """AC 4: full wake -> capture transition with a mock audio stream."""
    detector = _make_detector(_AlwaysDetect())
    listener = WakeWordListener(detector, poll_interval=0)

    with caplog.at_level(logging.DEBUG, logger="rex.wakeword.listener"):
        frame = asyncio.run(_collect_one(listener, _single_frame_source()))

    assert frame is not None, "Expected a frame to be yielded after wake word detection"

    events = [getattr(r, "event", None) for r in caplog.records]
    assert "wakeword_detected" in events
    assert "audio_capture_start" in events

    det_idx = events.index("wakeword_detected")
    cap_idx = events.index("audio_capture_start")
    assert det_idx < cap_idx, "wakeword_detected must be logged before audio_capture_start"
