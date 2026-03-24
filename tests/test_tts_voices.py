"""Unit tests for rex/tts_voices.py — mocked provider paths."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import rex.tts_voices as tts_voices_module
from rex.tts_voices import clear_edge_tts_cache, list_voices, synthesize_sample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_speakers_dir(tmp_path: Path, names: list[str]) -> Path:
    """Create a temporary directory with dummy WAV files."""
    speakers_dir = tmp_path / "voices"
    speakers_dir.mkdir()
    for name in names:
        (speakers_dir / name).write_bytes(b"RIFF")
    return speakers_dir


# ---------------------------------------------------------------------------
# XTTS provider
# ---------------------------------------------------------------------------


class TestListVoicesXtts:
    def test_returns_voice_dicts_for_wav_files(self, tmp_path):
        speakers_dir = _make_speakers_dir(tmp_path, ["alice.wav", "bob_voice.wav"])
        voices = list_voices("xtts", speakers_dir=speakers_dir)

        assert len(voices) == 2
        ids = {v["id"] for v in voices}
        assert any("alice" in i for i in ids)
        assert any("bob_voice" in i for i in ids)

    def test_voice_dict_has_required_keys(self, tmp_path):
        speakers_dir = _make_speakers_dir(tmp_path, ["rex.wav"])
        voices = list_voices("xtts", speakers_dir=speakers_dir)

        assert len(voices) == 1
        v = voices[0]
        assert "id" in v
        assert "name" in v
        assert "language" in v
        assert "gender" in v

    def test_name_is_title_cased_stem(self, tmp_path):
        speakers_dir = _make_speakers_dir(tmp_path, ["hello_world.wav"])
        voices = list_voices("xtts", speakers_dir=speakers_dir)
        assert voices[0]["name"] == "Hello World"

    def test_missing_directory_returns_empty_list(self, tmp_path):
        nonexistent = tmp_path / "no_such_dir"
        voices = list_voices("xtts", speakers_dir=nonexistent)
        assert voices == []

    def test_empty_directory_returns_empty_list(self, tmp_path):
        speakers_dir = tmp_path / "empty"
        speakers_dir.mkdir()
        voices = list_voices("xtts", speakers_dir=speakers_dir)
        assert voices == []

    def test_non_wav_files_are_ignored(self, tmp_path):
        speakers_dir = _make_speakers_dir(tmp_path, ["voice.mp3", "voice.txt"])
        (speakers_dir / "good.wav").write_bytes(b"RIFF")
        voices = list_voices("xtts", speakers_dir=speakers_dir)
        assert len(voices) == 1
        assert "good" in voices[0]["id"]


# ---------------------------------------------------------------------------
# edge-tts provider
# ---------------------------------------------------------------------------


class TestListVoicesEdgeTts:
    def setup_method(self):
        clear_edge_tts_cache()

    def test_returns_voices_from_edge_tts(self):
        mock_raw = [
            {
                "ShortName": "en-US-AriaNeural",
                "FriendlyName": "Microsoft Aria Online (Natural) - English (United States)",
                "Locale": "en-US",
                "Gender": "Female",
            },
            {
                "ShortName": "en-GB-RyanNeural",
                "FriendlyName": "Microsoft Ryan Online (Natural) - English (United Kingdom)",
                "Locale": "en-GB",
                "Gender": "Male",
            },
        ]

        mock_edge_tts = MagicMock()
        mock_edge_tts.list_voices = AsyncMock(return_value=mock_raw)

        with patch.dict("sys.modules", {"edge_tts": mock_edge_tts}):
            clear_edge_tts_cache()
            voices = list_voices("edge-tts")

        assert len(voices) == 2
        assert voices[0]["id"] == "en-US-AriaNeural"
        assert voices[0]["gender"] == "Female"
        assert voices[1]["language"] == "en-GB"

    def test_result_is_cached(self):
        mock_raw = [
            {
                "ShortName": "en-US-AriaNeural",
                "FriendlyName": "Aria",
                "Locale": "en-US",
                "Gender": "Female",
            }
        ]
        mock_edge_tts = MagicMock()
        mock_edge_tts.list_voices = AsyncMock(return_value=mock_raw)

        with patch.dict("sys.modules", {"edge_tts": mock_edge_tts}):
            clear_edge_tts_cache()
            voices1 = list_voices("edge-tts")
            voices2 = list_voices("edge-tts")

        assert voices1 is voices2
        assert mock_edge_tts.list_voices.await_count == 1

    def test_missing_edge_tts_returns_empty_list(self):
        with patch.dict("sys.modules", {"edge_tts": None}):
            clear_edge_tts_cache()
            # ImportError path — use a real ImportError by removing from sys.modules
            import sys

            saved = sys.modules.pop("edge_tts", None)
            sys.modules["edge_tts"] = None  # type: ignore[assignment]
            clear_edge_tts_cache()
            voices = tts_voices_module._list_edge_tts_voices()
            sys.modules.pop("edge_tts", None)
            if saved is not None:
                sys.modules["edge_tts"] = saved
        assert voices == []

    def test_edge_tts_accepts_alias_spellings(self):
        mock_raw = [{"ShortName": "x", "FriendlyName": "X", "Locale": "en", "Gender": None}]
        mock_edge_tts = MagicMock()
        mock_edge_tts.list_voices = AsyncMock(return_value=mock_raw)

        with patch.dict("sys.modules", {"edge_tts": mock_edge_tts}):
            for alias in ("edge_tts", "edgetts"):
                clear_edge_tts_cache()
                voices = list_voices(alias)
                assert len(voices) == 1


# ---------------------------------------------------------------------------
# pyttsx3 provider
# ---------------------------------------------------------------------------


class TestListVoicesPyttsx3:
    def _make_voice(self, id_, name, languages=None):
        v = MagicMock()
        v.id = id_
        v.name = name
        v.languages = languages or []
        return v

    def test_returns_voices_from_pyttsx3(self):
        v1 = self._make_voice("com.apple.speech.synthesis.voice.Alex", "Alex", ["en_US"])
        v2 = self._make_voice("com.apple.speech.synthesis.voice.Samantha", "Samantha", ["en_US"])

        mock_engine = MagicMock()
        mock_engine.getProperty.return_value = [v1, v2]

        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        with patch.dict("sys.modules", {"pyttsx3": mock_pyttsx3}):
            voices = list_voices("pyttsx3")

        assert len(voices) == 2
        assert voices[0]["name"] == "Alex"
        assert voices[0]["language"] == "en_US"

    def test_missing_pyttsx3_returns_empty_list(self):
        import sys

        saved = sys.modules.pop("pyttsx3", None)
        sys.modules["pyttsx3"] = None  # type: ignore[assignment]
        voices = tts_voices_module._list_pyttsx3_voices()
        sys.modules.pop("pyttsx3", None)
        if saved is not None:
            sys.modules["pyttsx3"] = saved
        assert voices == []


# ---------------------------------------------------------------------------
# Unknown provider
# ---------------------------------------------------------------------------


class TestListVoicesUnknownProvider:
    def test_unknown_provider_returns_empty_list(self):
        voices = list_voices("unknown_provider_xyz")
        assert voices == []


# ---------------------------------------------------------------------------
# synthesize_sample — edge-tts provider
# ---------------------------------------------------------------------------


class TestSynthesizeSampleEdgeTts:
    def test_returns_audio_bytes_for_edge_tts(self, tmp_path):
        """edge-tts Communicate.save writes bytes to a temp file; we read them back."""
        fake_audio = b"FAKE_MP3_AUDIO"

        async def fake_save(path: str) -> None:
            with open(path, "wb") as f:
                f.write(fake_audio)

        mock_communicate = MagicMock()
        mock_communicate.save = fake_save

        mock_edge_tts = MagicMock()
        mock_edge_tts.Communicate.return_value = mock_communicate

        with patch.dict("sys.modules", {"edge_tts": mock_edge_tts}):
            result = asyncio.run(synthesize_sample("edge-tts", "en-US-AriaNeural"))

        assert result == fake_audio

    def test_alias_spellings_work_for_synthesis(self, tmp_path):
        fake_audio = b"FAKE_MP3"

        async def fake_save(path: str) -> None:
            with open(path, "wb") as f:
                f.write(fake_audio)

        mock_communicate = MagicMock()
        mock_communicate.save = fake_save
        mock_edge_tts = MagicMock()
        mock_edge_tts.Communicate.return_value = mock_communicate

        with patch.dict("sys.modules", {"edge_tts": mock_edge_tts}):
            result = asyncio.run(synthesize_sample("edge_tts", "en-US-AriaNeural"))

        assert result == fake_audio

    def test_missing_edge_tts_raises_runtime_error(self):
        import sys

        saved = sys.modules.pop("edge_tts", None)
        sys.modules["edge_tts"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError, match="edge-tts is not installed"):
                asyncio.run(synthesize_sample("edge-tts", "en-US-AriaNeural"))
        finally:
            sys.modules.pop("edge_tts", None)
            if saved is not None:
                sys.modules["edge_tts"] = saved

    def test_text_is_truncated_to_50_chars(self):
        long_text = "A" * 100
        captured: list[str] = []

        async def fake_save(path: str) -> None:
            with open(path, "wb") as f:
                f.write(b"x")

        mock_communicate = MagicMock()
        mock_communicate.save = fake_save
        mock_edge_tts = MagicMock()

        def capture_communicate(text: str, voice: str):
            captured.append(text)
            return mock_communicate

        mock_edge_tts.Communicate.side_effect = capture_communicate

        with patch.dict("sys.modules", {"edge_tts": mock_edge_tts}):
            asyncio.run(synthesize_sample("edge-tts", "en-US-AriaNeural", long_text))

        assert len(captured[0]) <= 50


# ---------------------------------------------------------------------------
# synthesize_sample — pyttsx3 provider
# ---------------------------------------------------------------------------


class TestSynthesizeSamplePyttsx3:
    def test_returns_audio_bytes_for_pyttsx3(self, tmp_path):
        fake_audio = b"FAKE_WAV"

        def fake_save_to_file(text: str, path: str) -> None:
            with open(path, "wb") as f:
                f.write(fake_audio)

        mock_engine = MagicMock()
        mock_engine.save_to_file.side_effect = fake_save_to_file

        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        with patch.dict("sys.modules", {"pyttsx3": mock_pyttsx3}):
            result = asyncio.run(synthesize_sample("pyttsx3", "com.apple.voice.Alex"))

        assert result == fake_audio
        mock_engine.setProperty.assert_called_with("voice", "com.apple.voice.Alex")

    def test_missing_pyttsx3_raises_runtime_error(self):
        import sys

        saved = sys.modules.pop("pyttsx3", None)
        sys.modules["pyttsx3"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError, match="pyttsx3 is not installed"):
                asyncio.run(synthesize_sample("pyttsx3", "some-voice-id"))
        finally:
            sys.modules.pop("pyttsx3", None)
            if saved is not None:
                sys.modules["pyttsx3"] = saved


# ---------------------------------------------------------------------------
# synthesize_sample — unknown provider
# ---------------------------------------------------------------------------


class TestSynthesizeSampleUnknown:
    def test_unknown_provider_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="Unsupported TTS provider"):
            asyncio.run(synthesize_sample("unknown_xyz", "some-voice-id"))
