from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from wakeword_utils import detect_wakeword


class DummyModel:
    def predict(self, audio):
        return {"dummy": 0.75}


class LowConfidenceModel:
    def predict(self, audio):
        return {"dummy": 0.3}


@pytest.mark.unit
def test_detect_wakeword_threshold():
    model = DummyModel()
    audio = np.zeros(10, dtype=np.float32)

    assert detect_wakeword(model, audio, threshold=0.5), "Should detect wakeword at 0.75 >= 0.5"
    assert not detect_wakeword(model, audio, threshold=0.9), "Should NOT detect at 0.75 < 0.9"


@pytest.mark.unit
def test_detect_wakeword_low_confidence():
    model = LowConfidenceModel()
    audio = np.zeros(10, dtype=np.float32)

    assert not detect_wakeword(
        model, audio, threshold=0.5
    ), "Wakeword should not trigger at 0.3 score"
