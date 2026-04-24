from __future__ import annotations

import asyncio

import pytest

np = pytest.importorskip("numpy")

from rex.wakeword.listener import WakeWordListener  # noqa: E402
from rex.wakeword.utils import WakeWordDetectionResult, detect_wakeword, evaluate_wakeword  # noqa: E402


class DummyModel:
    def __init__(self, scores):
        self._scores = scores
        self.last_frame = None

    def predict(self, frame):
        self.last_frame = frame
        return self._scores


def test_detect_wakeword_handles_multidimensional_audio():
    model = DummyModel({"rex": 0.6})
    audio = np.ones((2, 4), dtype=np.float32)

    assert detect_wakeword(model, audio, threshold=0.5)


def test_evaluate_wakeword_boosts_quiet_float_audio():
    model = DummyModel({"rex": 0.6})
    audio = np.ones(16000, dtype=np.float32) * 0.01

    result = evaluate_wakeword(model, audio, threshold=0.5)

    assert result.triggered
    assert result.applied_gain > 1.0
    assert model.last_frame is not None
    assert model.last_frame.dtype == np.int16
    assert int(np.max(np.abs(model.last_frame))) > 3000


def test_wakeword_listener_yields_on_detection():
    model = DummyModel({"rex": 0.8})

    def detector(frame):
        return detect_wakeword(model, frame, threshold=0.5)

    listener = WakeWordListener(detector, poll_interval=0)

    frames = [np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]

    async def source():
        return frames.pop(0)

    async def collect():
        output = []
        async for _ in listener.listen(source):
            output.append(True)
            listener.stop()
        return output

    results = asyncio.run(collect())
    assert results == [True]


def test_wakeword_listener_does_not_accept_borderline_confidence_cluster():
    results = [
        WakeWordDetectionResult(
            triggered=False,
            threshold=0.03,
            predictions={"hey jarvis": 0.012},
            confidence=0.012,
            keyword="hey jarvis",
            reason="below_threshold",
        ),
        WakeWordDetectionResult(
            triggered=False,
            threshold=0.03,
            predictions={"hey jarvis": 0.011},
            confidence=0.011,
            keyword="hey jarvis",
            reason="below_threshold",
        ),
        WakeWordDetectionResult(
            triggered=False,
            threshold=0.03,
            predictions={"hey jarvis": 0.026},
            confidence=0.026,
            keyword="hey jarvis",
            reason="below_threshold",
        ),
    ]

    listener = WakeWordListener(lambda _frame: results.pop(0), poll_interval=0)
    frames = [np.ones(4, dtype=np.float32) for _ in range(3)]

    async def source():
        frame = frames.pop(0)
        if not frames:
            listener.stop()
        return frame

    async def collect():
        output = []
        async for _ in listener.listen(source):
            output.append(True)
            listener.stop()
        return output

    collected = asyncio.run(collect())

    assert collected == []
