"""Tests for US-033: perceived speed system (post-STT acknowledgment)."""

from __future__ import annotations

import asyncio
from importlib.util import find_spec

import pytest

_numpy_available = find_spec("numpy") is not None
_skip_no_numpy = pytest.mark.skipif(not _numpy_available, reason="numpy not installed")


# ---------------------------------------------------------------------------
# Config tests — no numpy dependency
# ---------------------------------------------------------------------------


def test_acknowledgment_mode_in_appconfig():
    """AppConfig has acknowledgment_mode field with valid default."""
    from rex.config import AppConfig

    cfg = AppConfig()
    assert hasattr(cfg, "acknowledgment_mode")
    assert cfg.acknowledgment_mode in ("sound", "phrase", "none")


def test_acknowledgment_mode_values():
    """acknowledgment_mode accepts 'sound', 'phrase', and 'none'."""
    from rex.config import AppConfig

    for mode in ("sound", "phrase", "none"):
        cfg = AppConfig(acknowledgment_mode=mode)
        assert cfg.acknowledgment_mode == mode


# ---------------------------------------------------------------------------
# VoiceLoop tests — require numpy
# ---------------------------------------------------------------------------


class _FakeListener:
    async def listen(self, detection_source):
        yield None


async def _transcribe_hello(audio):
    return "hello"


async def _transcribe_empty(audio):
    return ""


async def _speak(text):
    pass


def _make_loop(*, post_stt_acknowledge=None, transcribe=None):
    from unittest.mock import AsyncMock, MagicMock

    import numpy as np

    from rex.voice_loop import VoiceLoop

    if transcribe is None:
        transcribe = _transcribe_hello

    assistant = MagicMock()
    del assistant.stream_reply

    async def _gen_reply(text, **kwargs):
        return "Done."

    assistant.generate_reply = _gen_reply

    async def _record():
        return np.zeros(16000, dtype="float32")

    return VoiceLoop(
        assistant,
        wake_listener=_FakeListener(),
        detection_source=AsyncMock(return_value=None),
        record_phrase=_record,
        transcribe=transcribe,
        speak=_speak,
        post_stt_acknowledge=post_stt_acknowledge,
    )


@_skip_no_numpy
def test_post_stt_ack_called_before_llm():
    """Acknowledgment fires after STT transcript is ready, before LLM generate_reply."""
    from unittest.mock import AsyncMock, MagicMock

    import numpy as np

    from rex.voice_loop import VoiceLoop

    call_order: list[str] = []

    async def _ack():
        call_order.append("ack")

    assistant = MagicMock()
    del assistant.stream_reply

    async def _gen_reply(text, **kwargs):
        call_order.append("llm")
        return "Done."

    assistant.generate_reply = _gen_reply

    async def _record():
        return np.zeros(16000, dtype="float32")

    loop = VoiceLoop(
        assistant,
        wake_listener=_FakeListener(),
        detection_source=AsyncMock(return_value=None),
        record_phrase=_record,
        transcribe=_transcribe_hello,
        speak=_speak,
        post_stt_acknowledge=_ack,
    )

    asyncio.run(loop.run(max_interactions=1))

    assert "ack" in call_order and "llm" in call_order
    assert call_order.index("ack") < call_order.index("llm"), "Post-STT ack must fire before LLM"


@_skip_no_numpy
def test_post_stt_ack_not_called_when_no_transcript():
    """Acknowledgment is NOT called when STT returns empty transcript."""
    ack_calls: list[int] = []

    async def _ack():
        ack_calls.append(1)

    loop = _make_loop(post_stt_acknowledge=_ack, transcribe=_transcribe_empty)

    asyncio.run(loop.run(max_interactions=1))

    assert ack_calls == [], "Ack should not fire on empty transcript"


@_skip_no_numpy
def test_post_stt_ack_none_mode_no_error():
    """When post_stt_acknowledge is None (mode='none'), pipeline runs without error."""
    loop = _make_loop(post_stt_acknowledge=None)
    asyncio.run(loop.run(max_interactions=1))


@_skip_no_numpy
def test_post_stt_ack_failure_does_not_hang_pipeline():
    """A failing post-STT ack does not stop the pipeline (LLM still called)."""
    from unittest.mock import AsyncMock, MagicMock

    import numpy as np

    from rex.voice_loop import VoiceLoop

    llm_calls: list[int] = []

    async def _bad_ack():
        raise RuntimeError("ack exploded")

    assistant = MagicMock()
    del assistant.stream_reply

    async def _gen_reply(text, **kwargs):
        llm_calls.append(1)
        return "Done."

    assistant.generate_reply = _gen_reply

    async def _record():
        return np.zeros(16000, dtype="float32")

    loop = VoiceLoop(
        assistant,
        wake_listener=_FakeListener(),
        detection_source=AsyncMock(return_value=None),
        record_phrase=_record,
        transcribe=_transcribe_hello,
        speak=_speak,
        post_stt_acknowledge=_bad_ack,
    )

    asyncio.run(loop.run(max_interactions=1))

    assert llm_calls == [1], "LLM should still be called even if ack fails"
