"""Deterministic, authority-free parsing for canonical media commands."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_MAX_TRANSCRIPT_LENGTH = 2048
_MAX_FIELD_LENGTH = 512


class MediaCommandAction(StrEnum):
    PLAY = "play"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    NEXT = "next"
    PREVIOUS = "previous"
    SET_VOLUME = "set_volume"
    MUTE = "mute"
    UNMUTE = "unmute"
    QUERY_STATE = "query_state"
    TRANSFER = "transfer"


@dataclass(frozen=True, slots=True)
class MediaCommand:
    action: MediaCommandAction | str
    query: str | None = None
    target_text: str | None = None
    level: int | None = None

    def __post_init__(self) -> None:
        raw_action = "query_state" if self.action == "state" else self.action
        try:
            action = MediaCommandAction(raw_action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported media action: {self.action!r}") from exc
        object.__setattr__(self, "action", action)
        for field_name in ("query", "target_text"):
            value = getattr(self, field_name)
            if value is not None:
                if (
                    not isinstance(value, str)
                    or not value.strip()
                    or len(value) > _MAX_FIELD_LENGTH
                ):
                    raise ValueError(f"Invalid media command {field_name}")
                object.__setattr__(self, field_name, value.strip())
        if self.level is not None:
            if (
                isinstance(self.level, bool)
                or not isinstance(self.level, int)
                or not 0 <= self.level <= 100
            ):
                raise ValueError("Media command level must be between 0 and 100")


def _clean(text: str) -> str:
    return " ".join(text.strip().split()).rstrip(".!?").strip()


def _target_suffix(value: str) -> tuple[str, str | None]:
    match = re.search(r"\s+(?:in|on)\s+(?:the\s+)?(.+)$", value, flags=re.IGNORECASE)
    if match is None:
        return value.strip(), None
    return value[: match.start()].strip(), match.group(1).strip()


def _generic_target(value: str) -> str | None:
    value = value.strip()
    if value.casefold() in {
        "",
        "music",
        "the music",
        "speaker",
        "the speaker",
        "playback",
        "the playback",
    }:
        return None
    if value.casefold().startswith("the "):
        value = value[4:].strip()
    return value or None


def parse_media_command(text: str) -> MediaCommand | None:
    """Parse supported media grammar without resolving identity or authority."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if len(text) > _MAX_TRANSCRIPT_LENGTH:
        return None
    cleaned = _clean(text)
    if not cleaned:
        return None
    lower = cleaned.casefold()
    if lower.startswith("please "):
        cleaned = cleaned[7:].lstrip()
        lower = cleaned.casefold()
        if not cleaned:
            return None

    transfer = re.fullmatch(
        r"(?:move|send|transfer)\s+(?:it|this)(?:\s+(?:music|playback))?\s+to\s+(?:the\s+)?(.+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if transfer:
        return MediaCommand(MediaCommandAction.TRANSFER, target_text=transfer.group(1))
    transfer = re.fullmatch(r"play\s+it\s+in\s+(?:the\s+)?(.+)", cleaned, flags=re.IGNORECASE)
    if transfer:
        return MediaCommand(MediaCommandAction.TRANSFER, target_text=transfer.group(1))

    volume = re.fullmatch(
        r"(?:(?:set|turn)\s+)?volume\s+(?:(?:up|down)\s+to\s+|to\s+)?(-?\d+)(.*)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if volume:
        level = int(volume.group(1))
        if not 0 <= level <= 100:
            return None
        suffix = volume.group(2).strip()
        target = None
        if suffix:
            _, target = _target_suffix("x " + suffix)
        return MediaCommand(MediaCommandAction.SET_VOLUME, target_text=target, level=level)

    unmute = re.fullmatch(r"unmute(?:\s+to\s+(\d+))?(.*)", cleaned, flags=re.IGNORECASE)
    if unmute:
        unmute_level = int(unmute.group(1)) if unmute.group(1) else None
        if unmute_level is not None and not 0 <= unmute_level <= 100:
            return None
        suffix = unmute.group(2).strip()
        target = None
        if suffix:
            if suffix.casefold().startswith(("in ", "on ")):
                _, target = _target_suffix("x " + suffix)
            else:
                target = _generic_target(suffix)
        return MediaCommand(MediaCommandAction.UNMUTE, target_text=target, level=unmute_level)

    if lower.startswith("mute"):
        target = _generic_target(cleaned[4:])
        return MediaCommand(MediaCommandAction.MUTE, target_text=target)
    state_prefixes = (
        "what's playing",
        "what is playing",
        "is it playing",
        "is the music playing",
        "which song is playing",
        "what track is this",
    )
    for prefix in state_prefixes:
        if lower == prefix:
            return MediaCommand(MediaCommandAction.QUERY_STATE)
        if lower.startswith(prefix + " "):
            suffix = cleaned[len(prefix) :].strip()
            _, target = _target_suffix("x " + suffix)
            if target is not None:
                return MediaCommand(MediaCommandAction.QUERY_STATE, target_text=target)

    exact_actions = {
        "pause": MediaCommandAction.PAUSE,
        "pause the music": MediaCommandAction.PAUSE,
        "resume": MediaCommandAction.RESUME,
        "unpause": MediaCommandAction.RESUME,
        "continue": MediaCommandAction.RESUME,
        "continue playing": MediaCommandAction.RESUME,
        "continue music": MediaCommandAction.RESUME,
        "continue the playback": MediaCommandAction.RESUME,
        "stop": MediaCommandAction.STOP,
        "stop the music": MediaCommandAction.STOP,
        "next": MediaCommandAction.NEXT,
        "skip": MediaCommandAction.NEXT,
        "skip track": MediaCommandAction.NEXT,
        "skip this song": MediaCommandAction.NEXT,
        "skip this track": MediaCommandAction.NEXT,
        "next song": MediaCommandAction.NEXT,
        "next track": MediaCommandAction.NEXT,
        "previous": MediaCommandAction.PREVIOUS,
        "go back": MediaCommandAction.PREVIOUS,
        "back": MediaCommandAction.PREVIOUS,
    }
    if lower in exact_actions:
        return MediaCommand(exact_actions[lower])

    transport_patterns = (
        (r"pause\s+(.+)", MediaCommandAction.PAUSE),
        (r"resume\s+(.+)", MediaCommandAction.RESUME),
        (r"stop\s+(.+)", MediaCommandAction.STOP),
        (r"(?:next|skip|skip\s+track)\s+(.+)", MediaCommandAction.NEXT),
        (r"(?:previous|go\s+back|back)\s+(.+)", MediaCommandAction.PREVIOUS),
    )
    for pattern, action in transport_patterns:
        match = re.fullmatch(pattern, cleaned, flags=re.IGNORECASE)
        if match is None:
            continue
        remainder = match.group(1).strip()
        _, target = _target_suffix("x " + remainder)
        if target is None:
            target = _generic_target(remainder)
        return MediaCommand(action, target_text=target)

    if lower.startswith("play "):
        body = cleaned[5:].strip()
        query, target = _target_suffix(body)
        if query.casefold() in {"", "it", "the music", "music"}:
            return None
        return MediaCommand(MediaCommandAction.PLAY, query=query, target_text=target)

    return None


__all__ = ["MediaCommand", "MediaCommandAction", "parse_media_command"]
