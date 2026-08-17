"""Music Assistant voice intent handler (US-022).

Detects music-control utterances before they reach the LLM so Rex can respond
immediately without spending tokens on a trivial task.

Supported patterns
------------------
Play:
  "play Shape of You"
  "play jazz in the kitchen"
  "play some music"

Pause:
  "pause"
  "pause the music"
  "pause in the living room"

Resume:
  "resume"
  "resume the music"
  "continue playing"

Skip:
  "skip"
  "next song"
  "skip this song"
  "next track"

Volume:
  "set volume to 50"
  "volume 70 in the kitchen"
  "turn volume up to 80"
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_PLAY_PATTERN = re.compile(
    r"^(?:please\s+)?play\s+(.+?)(?:\s+in\s+(?:the\s+)?(.+?))?$",
    re.IGNORECASE,
)

_PAUSE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^(?:please\s+)?pause(?:\s+(?:the\s+)?music)?(?:\s+in\s+(?:the\s+)?(.+?))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?stop\s+(?:the\s+)?music(?:\s+in\s+(?:the\s+)?(.+?))?$",
        re.IGNORECASE,
    ),
]

_RESUME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^(?:please\s+)?resume(?:\s+(?:the\s+|playing|music)?)?(?:\s+in\s+(?:the\s+)?(.+?))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?continue\s+(?:playing|music)(?:\s+in\s+(?:the\s+)?(.+?))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?unpause(?:\s+(?:the\s+)?music)?(?:\s+in\s+(?:the\s+)?(.+?))?$",
        re.IGNORECASE,
    ),
]

_SKIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^(?:please\s+)?skip(?:\s+(?:this\s+)?(?:song|track))?(?:\s+in\s+(?:the\s+)?(.+?))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?next\s+(?:song|track)(?:\s+in\s+(?:the\s+)?(.+?))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?next(?:\s+in\s+(?:the\s+)?(.+?))?$",
        re.IGNORECASE,
    ),
]

_VOLUME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^(?:please\s+)?(?:set\s+)?volume\s+(?:to\s+)?(\d{1,3})(?:\s+in\s+(?:the\s+)?(.+?))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?(?:turn\s+)?volume\s+(?:up\s+to|down\s+to|to)\s+(\d{1,3})(?:\s+in\s+(?:the\s+)?(.+?))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?set\s+(?:the\s+)?volume\s+to\s+(\d{1,3})(?:\s+in\s+(?:the\s+)?(.+?))?$",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Match helpers — return (matched: bool, ...) tuples
# ---------------------------------------------------------------------------


def _match_play(transcript: str) -> tuple[bool, str, str | None]:
    """Return (True, query, room) if transcript is a play command.

    Returns (False, '', None) when not matched.
    """
    m = _PLAY_PATTERN.match(transcript.strip())
    if m:
        query = m.group(1).strip()
        room_raw = m.group(2)
        room = room_raw.strip() if room_raw else None
        return True, query, room
    return False, "", None


def _match_room_command(
    patterns: list[re.Pattern[str]], transcript: str
) -> tuple[bool, str | None]:
    """Match any pattern that optionally captures a room.

    Returns (True, room_or_None) on match, (False, None) otherwise.
    """
    for pattern in patterns:
        m = pattern.match(transcript.strip())
        if m:
            room_raw = m.group(1) if m.lastindex and m.lastindex >= 1 else None
            room = room_raw.strip() if room_raw else None
            return True, room
    return False, None


def _match_volume(transcript: str) -> tuple[bool, int, str | None]:
    """Return (True, level, room) if transcript is a volume command.

    Returns (False, 0, None) when not matched.
    """
    for pattern in _VOLUME_PATTERNS:
        m = pattern.match(transcript.strip())
        if m:
            level = int(m.group(1))
            room_raw = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            room = room_raw.strip() if room_raw else None
            return True, level, room
    return False, 0, None


# ---------------------------------------------------------------------------
# Handler class
# ---------------------------------------------------------------------------


class MusicHandler:
    """Deprecated compatibility parser for legacy music phrases.

    The optional client is retained only for construction compatibility.  It is
    never called: all media execution belongs to the canonical tool lifecycle.
    """

    def __init__(self, client=None) -> None:  # noqa: ANN001
        self._client = client

    def handle(self, transcript: str):  # noqa: ANN201
        """Parse a legacy music utterance without executing provider mutations."""
        from rex.media.parser import parse_media_command

        return parse_media_command(transcript)


__all__ = ["MusicHandler"]
