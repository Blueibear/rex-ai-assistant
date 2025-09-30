from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

np = pytest.importorskip("numpy")

from config import AppConfig
from voice_loop import AsyncRexAssistant


class DummyDetector:
    sample_rate = 16000
    frame_length = 512

    def __init__(self) -> None:
        self.calls = 0

    def push(self, audio) -> bool:
        self.calls += 1
        return True


@pytest.mark.asyncio
async def test_audio_pipeline_end_to_end(monkeypatch, tmp_path):
    detector = DummyDetector()
    monkeypatch.setattr("voice_loop.load_wakeword_model", lambda **kwargs: (detector, "rex"))

    cfg = AppConfig(
        llm_backend="echo",
        gpu=False,
        command_duration=0.1,
        transcripts_enabled=False,
        conversation_export=False,
        memory_path=tmp_path / "memory.json",
        transcripts_dir=tmp_path / "transcripts",
        user_profiles_dir=tmp_path / "profiles",
    )

    assistant = AsyncRexAssistant(cfg)
    monkeypatch.setattr(assistant, "_play_wake_sound", lambda: None)
    monkeypatch.setattr(
        assistant,
        "_record_audio",
        lambda: np.zeros(detector.sample_rate // 10, dtype=np.float32),
    )
    monkeypatch.setattr(assistant, "_transcribe_audio", lambda audio: "wake run")

    class StubLM:
        def generate(self, prompt: str, *_, **__):
            return "acknowledged"

    assistant.language_model = StubLM()

    outputs: list[str] = []
    monkeypatch.setattr(assistant, "_speak_response", lambda text: outputs.append(text))

    await assistant._handle_interaction()

    assert outputs == ["acknowledged"]
