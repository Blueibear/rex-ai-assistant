from __future__ import annotations

import asyncio

import pytest

import rex_loop


def test_voice_loop_cli_defaults_to_hold_to_talk() -> None:
    args = rex_loop._create_parser().parse_args([])
    assert args.mode == "hold-to-talk"


def test_voice_loop_cli_allows_explicit_wake_word_mode() -> None:
    args = rex_loop._create_parser().parse_args(["--mode", "wake-word"])
    assert args.mode == "wake-word"


def test_manual_activation_listener_does_not_consume_detection_audio() -> None:
    from rex.voice.activation import ManualActivationListener

    trigger_calls = 0
    detection_calls = 0

    def trigger(_prompt: str) -> str:
        nonlocal trigger_calls
        trigger_calls += 1
        return ""

    async def detection_source():
        nonlocal detection_calls
        detection_calls += 1
        raise AssertionError("hold-to-talk mode must not consume wake-detection audio")

    async def first_activation() -> bytes:
        listener = ManualActivationListener(trigger=trigger)
        stream = listener.listen(detection_source)
        return await anext(stream)

    assert asyncio.run(first_activation()) == b""
    assert trigger_calls == 1
    assert detection_calls == 0


def test_manual_activation_listener_ends_cleanly_on_eof() -> None:
    from rex.voice.activation import ManualActivationListener

    def eof(_prompt: str) -> str:
        raise EOFError

    async def no_activation() -> None:
        listener = ManualActivationListener(trigger=eof)
        stream = listener.listen(lambda: pytest.fail("detection source should not run"))
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    asyncio.run(no_activation())
