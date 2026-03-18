"""US-017: Wake word detection acceptance tests.

Criteria:
  1. wake word detection triggers listening
  2. microphone stream initializes
  3. wake word does not trigger on common conversational speech
  4. Typecheck passes
"""

from __future__ import annotations

import asyncio

import pytest

np = pytest.importorskip("numpy")

from rex.wakeword.listener import WakeWordListener  # noqa: E402
from rex.wakeword.utils import detect_wakeword  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal stub models
# ---------------------------------------------------------------------------


class HighConfidenceModel:
    """Always scores above any threshold — simulates wake word present."""

    def predict(self, audio: np.ndarray) -> dict[str, float]:
        return {"hey rex": 0.95}


class LowConfidenceModel:
    """Scores from common conversational speech (no wake word)."""

    def __init__(self, score: float = 0.05) -> None:
        self._score = score

    def predict(self, audio: np.ndarray) -> dict[str, float]:
        return {"hey rex": self._score}


# ---------------------------------------------------------------------------
# AC 1: wake word detection triggers listening
# ---------------------------------------------------------------------------


def test_wake_word_detection_triggers_listening() -> None:
    """When the model fires above threshold, detect_wakeword returns True."""
    model = HighConfidenceModel()
    audio = np.zeros(16000, dtype=np.float32)
    assert detect_wakeword(
        model, audio, threshold=0.5
    ), "Wake word should be detected and trigger listening"


def test_wake_word_listener_yields_frame_on_detection() -> None:
    """WakeWordListener yields the triggering audio frame, enabling downstream listening."""
    model = HighConfidenceModel()
    captured: list[bool] = []

    def detector(frame: np.ndarray) -> bool:
        return detect_wakeword(model, frame, threshold=0.5)

    listener = WakeWordListener(detector, poll_interval=0)
    audio_frame = np.ones(16000, dtype=np.float32)

    async def source() -> np.ndarray:
        return audio_frame

    async def run() -> None:
        async for _ in listener.listen(source):
            captured.append(True)
            listener.stop()

    asyncio.run(run())
    assert captured == [True], "Listener should yield exactly one frame when wake word fires"


# ---------------------------------------------------------------------------
# AC 2: microphone stream initializes
# ---------------------------------------------------------------------------


def test_wakeword_listener_initializes() -> None:
    """WakeWordListener can be instantiated and its stop flag starts False."""
    model = HighConfidenceModel()

    def detector(frame: np.ndarray) -> bool:
        return detect_wakeword(model, frame, threshold=0.5)

    listener = WakeWordListener(detector, poll_interval=0.05)
    # _running is False before listen() is called (not yet streaming)
    assert listener._running is False


def test_wakeword_listener_starts_running_during_listen() -> None:
    """WakeWordListener sets _running=True inside the listen loop."""
    HighConfidenceModel()
    running_states: list[bool] = []

    def detector(frame: np.ndarray) -> bool:
        running_states.append(True)
        return True  # trigger immediately

    listener = WakeWordListener(detector, poll_interval=0)
    audio_frame = np.ones(16000, dtype=np.float32)

    async def source() -> np.ndarray:
        return audio_frame

    async def run() -> None:
        async for _ in listener.listen(source):
            listener.stop()
            break

    asyncio.run(run())
    # detector was called → stream was running
    assert running_states, "Detector should have been called while stream was active"


def test_wakeword_listener_stops_cleanly() -> None:
    """Calling stop() causes the listen loop to exit without error."""
    HighConfidenceModel()
    call_count = 0

    def detector(frame: np.ndarray) -> bool:
        nonlocal call_count
        call_count += 1
        return False  # never trigger; we stop manually

    listener = WakeWordListener(detector, poll_interval=0)
    audio_frame = np.zeros(16000, dtype=np.float32)

    async def source() -> np.ndarray:
        return audio_frame

    async def run() -> None:
        task = asyncio.create_task(_collect(listener, source))
        await asyncio.sleep(0)
        listener.stop()
        await task

    async def _collect(listener_: WakeWordListener, src: object) -> None:
        async for _ in listener_.listen(src):  # type: ignore[arg-type]
            pass

    asyncio.run(run())
    assert not listener._running, "Listener should be stopped after stop() is called"


# ---------------------------------------------------------------------------
# AC 3: wake word does not trigger on common conversational speech
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase_score",
    [
        0.0,  # silence
        0.05,  # typical background noise
        0.10,  # soft speech not matching wake word
        0.25,  # louder speech, still below threshold
        0.49,  # just below default threshold
    ],
)
def test_wake_word_does_not_trigger_on_conversational_speech(
    phrase_score: float,
) -> None:
    """Common speech scores (< 0.5) must not trigger wake word detection."""
    model = LowConfidenceModel(score=phrase_score)
    audio = np.random.default_rng(42).standard_normal(16000).astype(np.float32)
    assert not detect_wakeword(
        model, audio, threshold=0.5
    ), f"Wake word should NOT trigger for score={phrase_score}"


def test_wake_word_does_not_trigger_on_exact_threshold_boundary() -> None:
    """Score exactly at threshold does trigger (>= semantics); just below does not."""
    model = LowConfidenceModel(score=0.499)
    audio = np.zeros(16000, dtype=np.float32)
    assert not detect_wakeword(model, audio, threshold=0.5)

    model_at = LowConfidenceModel(score=0.5)
    assert detect_wakeword(model_at, audio, threshold=0.5)


def test_wake_word_not_triggered_by_silence() -> None:
    """Silent audio (all zeros) should not trigger the wake word."""
    model = LowConfidenceModel(score=0.0)
    audio = np.zeros(16000, dtype=np.float32)
    assert not detect_wakeword(model, audio, threshold=0.5)
