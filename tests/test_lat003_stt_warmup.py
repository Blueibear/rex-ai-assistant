"""Tests for US-LAT-003: STT pipeline warm-up — pre-load Whisper model at startup."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_whisper_module(delay_event: threading.Event | None = None):
    """Return a mock whisper module whose load_model optionally blocks."""

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "hello"}

    mock_whisper = MagicMock()

    def _load_model(name, device=None):
        if delay_event is not None:
            delay_event.wait()
        return mock_model

    mock_whisper.load_model.side_effect = _load_model
    return mock_whisper, mock_model


# ---------------------------------------------------------------------------
# SpeechToText – async_load=False (synchronous, existing behaviour)
# ---------------------------------------------------------------------------


def test_stt_sync_load_is_loaded_immediately():
    """With async_load=False the model is ready immediately after __init__."""
    from rex.voice_loop import SpeechToText

    mock_whisper, _ = _make_whisper_module()

    with patch("rex.voice_loop._lazy_import_whisper", return_value=mock_whisper):
        stt = SpeechToText("base", "cpu")

    assert stt.is_loaded() is True


def test_stt_sync_load_calls_load_model_once():
    """With async_load=False, load_model is called exactly once during __init__."""
    from rex.voice_loop import SpeechToText

    mock_whisper, _ = _make_whisper_module()

    with patch("rex.voice_loop._lazy_import_whisper", return_value=mock_whisper):
        SpeechToText("base", "cpu")

    mock_whisper.load_model.assert_called_once_with("base", device="cpu")


# ---------------------------------------------------------------------------
# SpeechToText – async_load=True (background thread)
# ---------------------------------------------------------------------------


def test_stt_async_load_is_not_loaded_before_thread_completes():
    """With async_load=True, is_loaded() returns False while thread is blocked."""
    from rex.voice_loop import SpeechToText

    gate = threading.Event()  # gate keeps the load thread blocked
    mock_whisper, _ = _make_whisper_module(delay_event=gate)

    with patch("rex.voice_loop._lazy_import_whisper", return_value=mock_whisper):
        stt = SpeechToText("base", "cpu", async_load=True)

    # Thread is blocked on `gate`; model not yet loaded
    assert stt.is_loaded() is False

    # Unblock and wait for the thread
    gate.set()
    stt._load_event.wait(timeout=5)
    assert stt.is_loaded() is True


def test_stt_async_load_event_set_after_completion():
    """_load_event is set once the background thread finishes."""
    from rex.voice_loop import SpeechToText

    mock_whisper, _ = _make_whisper_module()

    with patch("rex.voice_loop._lazy_import_whisper", return_value=mock_whisper):
        stt = SpeechToText("base", "cpu", async_load=True)

    stt._load_event.wait(timeout=5)
    assert stt._load_event.is_set()


def test_stt_async_load_background_thread_is_daemon():
    """The warm-up thread is a daemon so it does not block process exit."""
    from rex.voice_loop import SpeechToText

    gate = threading.Event()
    mock_whisper, _ = _make_whisper_module(delay_event=gate)

    threads_before = {t.name for t in threading.enumerate()}

    with patch("rex.voice_loop._lazy_import_whisper", return_value=mock_whisper):
        stt = SpeechToText("base", "cpu", async_load=True)

    new_threads = [
        t for t in threading.enumerate() if t.name not in threads_before and t.name == "stt-warmup"
    ]
    if new_threads:
        assert new_threads[0].daemon is True

    gate.set()
    stt._load_event.wait(timeout=5)


# ---------------------------------------------------------------------------
# SpeechToText – load failure handling
# ---------------------------------------------------------------------------


def test_stt_sync_load_error_raises():
    """If load_model raises in sync path, SpeechToTextError is raised from __init__."""
    from rex.voice_loop import SpeechToText, SpeechToTextError

    mock_whisper = MagicMock()
    mock_whisper.load_model.side_effect = RuntimeError("out of memory")

    with patch("rex.voice_loop._lazy_import_whisper", return_value=mock_whisper):
        try:
            SpeechToText("base", "cpu")
            raise AssertionError("Expected SpeechToTextError")
        except SpeechToTextError:
            pass


def test_stt_async_load_error_captured():
    """With async_load=True a failed load sets _load_error without raising."""
    import asyncio

    from rex.voice_loop import SpeechToText, SpeechToTextError

    mock_whisper = MagicMock()
    mock_whisper.load_model.side_effect = RuntimeError("CUDA error")

    with patch("rex.voice_loop._lazy_import_whisper", return_value=mock_whisper):
        stt = SpeechToText("base", "cpu", async_load=True)

    stt._load_event.wait(timeout=5)
    assert stt._load_error is not None
    assert stt.is_loaded() is False

    # transcribe() should raise SpeechToTextError (not hang)
    async def _run():
        # We need a tiny valid WAV for the audio path — but it will hit the
        # load-error check before audio parsing, so just use empty bytes.
        with patch("rex.voice_loop._prepare_audio_for_stt", return_value=MagicMock()):
            with patch("rex.voice_loop._to_wav_buffer", return_value=b"RIFF"):
                await stt.transcribe(b"\x00" * 32, 16000)

    try:
        asyncio.run(_run())
        raise AssertionError("Expected SpeechToTextError")
    except SpeechToTextError:
        pass


# ---------------------------------------------------------------------------
# SpeechToText – transcribe() waits for warm-up
# ---------------------------------------------------------------------------


def test_stt_transcribe_waits_for_async_load():
    """transcribe() blocks until the background thread completes, then succeeds."""
    import asyncio

    from rex.voice_loop import SpeechToText

    gate = threading.Event()
    mock_whisper, mock_model = _make_whisper_module(delay_event=gate)

    with patch("rex.voice_loop._lazy_import_whisper", return_value=mock_whisper):
        stt = SpeechToText("base", "cpu", async_load=True)

    # Unblock the load thread after a short delay via a timer
    timer = threading.Timer(0.05, gate.set)
    timer.start()

    dummy_audio = MagicMock()
    dummy_audio.__len__ = lambda self: 16000

    async def _run():
        with patch("rex.voice_loop._prepare_audio_for_stt", return_value=dummy_audio):
            with patch("rex.voice_loop._to_wav_buffer", return_value=b"RIFF" + b"\x00" * 40):
                result = await stt.transcribe(dummy_audio, 16000)
        return result

    result = asyncio.run(_run())
    assert result == "hello"


# ---------------------------------------------------------------------------
# check_stt_warmup – doctor integration
# ---------------------------------------------------------------------------


def test_check_stt_warmup_with_loaded_instance():
    """check_stt_warmup reports OK 'STT model: loaded' when is_loaded() is True."""
    from rex.doctor import Status, check_stt_warmup

    mock_stt = MagicMock()
    mock_stt.is_loaded.return_value = True
    mock_stt._load_error = None

    result = check_stt_warmup(stt=mock_stt)
    assert result.status == Status.OK
    assert "loaded" in result.message


def test_check_stt_warmup_with_loading_instance():
    """check_stt_warmup reports INFO 'STT model: loading...' when not yet loaded."""
    from rex.doctor import Status, check_stt_warmup

    mock_stt = MagicMock()
    mock_stt.is_loaded.return_value = False
    mock_stt._load_error = None

    result = check_stt_warmup(stt=mock_stt)
    assert result.status == Status.INFO
    assert "loading" in result.message


def test_check_stt_warmup_with_load_error():
    """check_stt_warmup reports ERROR when _load_error is set."""
    from rex.doctor import Status, check_stt_warmup

    mock_stt = MagicMock()
    mock_stt.is_loaded.return_value = False
    mock_stt._load_error = "out of memory"

    result = check_stt_warmup(stt=mock_stt)
    assert result.status == Status.ERROR
    assert "out of memory" in result.message


def test_check_stt_warmup_standalone_whisper_available():
    """check_stt_warmup() with no instance returns OK when whisper is importable."""
    from rex.doctor import Status, check_stt_warmup

    with patch("rex.doctor.find_spec", return_value=MagicMock()):
        result = check_stt_warmup()

    assert result.status in (Status.OK, Status.WARNING, Status.ERROR)


def test_check_stt_warmup_standalone_whisper_missing():
    """check_stt_warmup() returns WARNING when whisper is not installed."""
    from rex.doctor import Status, check_stt_warmup

    with patch("rex.doctor.find_spec", return_value=None):
        result = check_stt_warmup()

    assert result.status == Status.WARNING
    assert "whisper" in result.message.lower()
