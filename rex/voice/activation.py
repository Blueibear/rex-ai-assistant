"""Manual activation source for the source voice-loop hold-to-talk mode."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

DetectionSource = Callable[[], Awaitable[Any]]
Trigger = Callable[[str], str]


class ManualActivationListener:
    """Yield an interaction when the user explicitly starts a source-CLI turn.

    The packaged Electron application implements true press/hold recording in
    the renderer.  The source CLI uses an Enter-key trigger followed by the
    existing VAD-bounded phrase capture, which keeps wake-word detection fully
    out of the default path without adding a platform-specific keyboard hook.
    """

    def __init__(
        self,
        *,
        prompt: str = "Press Enter to talk (Ctrl+C to exit): ",
        trigger: Trigger = input,
    ) -> None:
        self._prompt = prompt
        self._trigger = trigger

    async def listen(self, _source: DetectionSource) -> AsyncIterator[bytes]:
        while True:
            try:
                await asyncio.to_thread(self._trigger, self._prompt)
            except EOFError:
                return
            # VoiceLoop treats bytes as an activation-only frame and therefore
            # will not prepend it to the separately captured command audio.
            yield b""

    def mark_listening_started(self, *, reason: str = "manual") -> None:
        """Compatibility no-op for the VoiceLoop activation lifecycle."""

    def reset(self, *, reason: str = "manual") -> None:
        """Compatibility no-op for the VoiceLoop activation lifecycle."""
