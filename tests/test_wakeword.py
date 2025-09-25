from __future__ import annotations

import asyncio

import pytest

np = pytest.importorskip("numpy")

from rex.wakeword.listener import WakeWordListener
from rex.wakeword.utils import detect_wakeword


class DummyModel:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, frame):
        return self._scores


def test_detect_wakeword_handles_multidimensional_audio():
    model = DummyModel({"rex": 0.6})
    audio = np.ones((2, 4), dtype=np.float32)

    assert detect_wakeword(model, audio, threshold=0.5)


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
