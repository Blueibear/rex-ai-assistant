from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from wakeword_utils import detect_wakeword


class DummyModel:
    def predict(self, audio):
        return {"dummy": 0.75}


def test_detect_wakeword_threshold():
    model = DummyModel()
    audio = np.zeros(10, dtype=np.float32)
    assert detect_wakeword(model, audio, threshold=0.5)
    assert not detect_wakeword(model, audio, threshold=0.9)
