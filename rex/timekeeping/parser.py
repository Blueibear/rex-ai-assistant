"""Deterministic natural-language parsing for common timer/alarm commands."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

Action = Literal[
    "create_timer",
    "list_timers",
    "query_timer",
    "pause_timer",
    "resume_timer",
    "cancel_timer",
    "rename_timer",
    "adjust_timer",
    "create_alarm",
    "edit_alarm",
    "list_alarms",
    "snooze_alarm",
    "dismiss_alarm",
    "enable_alarm",
    "disable_alarm",
    "cancel_alarm",
]


@dataclass(frozen=True, slots=True)
class TimekeepingCommand:
    action: Action
    reference: str | None = None
    duration_seconds: float | None = None
    delta_seconds: float | None = None
    new_name: str | None = None
    alarm_time: time | None = None
    alarm_date: date | None = None
    weekdays: tuple[int, ...] = ()
    target_text: str | None = None
    target_volume: int | None = None


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "forty-five": 45,
    "fifty": 50,
    "sixty": 60,
}
_NUMBER_PATTERN = "|".join(sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True))
_DURATION_RE = re.compile(
    rf"(?P<value>\d+(?:\.\d+)?|{_NUMBER_PATTERN})\s*(?:-|\s)?\s*"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
    re.IGNORECASE,
)
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().rstrip(".?!"))


def _extract_target_clause(text: str) -> tuple[str, str | None]:
    """Remove a trailing output-target clause without resolving device authority."""
    patterns = (
        re.compile(r"\s+and\s+play\s+it\s+on\s+(?:the\s+)?(?P<target>.+)$"),
        re.compile(r"(?P<prefix>\btimer)\s+(?:on|in)\s+(?:the\s+)?(?P<target>.+)$"),
        re.compile(r"(?P<prefix>\balarm)\s+on\s+(?:the\s+)?(?P<target>.+)$"),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match is None:
            continue
        target = match.group("target").strip(" ,")
        if not target:
            continue
        if "prefix" in match.groupdict() and match.group("prefix"):
            cleaned = text[: match.start()] + match.group("prefix")
        else:
            cleaned = text[: match.start()]
        return cleaned.strip(), target
    return text, None


def _number(value: str) -> float:
    return float(_NUMBER_WORDS[value]) if value in _NUMBER_WORDS else float(value)


def _duration_match(text: str) -> tuple[float, re.Match[str]] | None:
    match = _DURATION_RE.search(text)
    if match is None:
        return None
    value = _number(match.group("value").lower())
    unit = match.group("unit").lower()
    multiplier = (
        1
        if unit.startswith(("second", "sec"))
        else 60 if unit.startswith(("minute", "min")) else 3600
    )
    seconds = value * multiplier
    return (seconds, match) if seconds > 0 else None


def _clean_reference(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"^(?:the|my)\s+", "", value.strip())
    if cleaned in {"", "that", "this", "it", "my"}:
        return None
    return cleaned.strip()


def _parse_alarm_time(text: str) -> time | None:
    match = re.search(
        r"(?:\bfor\b|\bat\b|\bto\b)\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
        r"(?P<ampm>a\.?m\.?|p\.?m\.?)?",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    ampm = (match.group("ampm") or "").replace(".", "").lower()
    if not ampm:
        if "morning" in text and 1 <= hour <= 11:
            ampm = "am"
        elif any(word in text for word in ("afternoon", "evening", "tonight")) and 1 <= hour <= 11:
            ampm = "pm"
    if ampm:
        if not 1 <= hour <= 12:
            return None
        if ampm == "am":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    return time(hour, minute)


def _alarm_recurrence(text: str) -> tuple[int, ...]:
    if "every weekday" in text or "weekdays" in text:
        return (0, 1, 2, 3, 4)
    if "every day" in text or "daily" in text:
        return (0, 1, 2, 3, 4, 5, 6)
    if "every " not in text:
        return ()
    return tuple(sorted(day for name, day in _WEEKDAYS.items() if name in text))


def _relative_alarm_date(text: str, *, user_timezone: str, now: datetime) -> date | None:
    try:
        zone = ZoneInfo(user_timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown timezone: {user_timezone}") from exc
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local_today = now.astimezone(zone).date()
    if "tomorrow" in text:
        return local_today + timedelta(days=1)
    if "today" in text or "tonight" in text:
        return local_today
    return None


def _parse_timer_create(text: str) -> TimekeepingCommand | None:
    if "timer" not in text or not re.match(r"^(?:set|start|create)\b", text):
        return None
    duration = _duration_match(text)
    if duration is None:
        return None
    seconds, match = duration
    timer_pos = text.find("timer")
    name: str | None = None
    if timer_pos < match.start():
        prefix = text[:timer_pos]
        prefix = re.sub(r"^(?:set|start|create)\s+(?:a\s+)?", "", prefix).strip()
        name = _clean_reference(prefix)
    else:
        between = text[match.end() : timer_pos].strip(" -")
        name = _clean_reference(between)
    return TimekeepingCommand(action="create_timer", reference=name, duration_seconds=seconds)


def _parse_timer_management(text: str) -> TimekeepingCommand | None:
    rename = re.match(r"^rename\s+(?:the\s+|my\s+)?(.+?)\s+timer\s+to\s+(.+)$", text)
    if rename:
        return TimekeepingCommand(action="rename_timer", reference=_clean_reference(rename.group(1)), new_name=rename.group(2).strip())
    adjust = re.match(r"^(add|subtract|remove)\s+(.+?)\s+(?:to|from)\s+(?:the\s+|my\s+)?(.+?)\s+timer$", text)
    if adjust:
        duration = _duration_match(adjust.group(2))
        if duration is None:
            return None
        seconds = duration[0]
        if adjust.group(1) in {"subtract", "remove"}:
            seconds = -seconds
        return TimekeepingCommand(action="adjust_timer", reference=_clean_reference(adjust.group(3)), delta_seconds=seconds)
    action_match = re.match(r"^(pause|resume|cancel|stop)\s+(?:the\s+|my\s+)?(.+?)\s+timer$", text)
    if action_match:
        verb = action_match.group(1)
        action: Action = "cancel_timer" if verb in {"cancel", "stop"} else f"{verb}_timer"  # type: ignore[assignment]
        return TimekeepingCommand(action=action, reference=_clean_reference(action_match.group(2)))
    return None


def _parse_alarm_management(text: str) -> TimekeepingCommand | None:
    rename = re.match(r"^rename\s+(?:the\s+|my\s+)?(.+?)\s+alarm\s+to\s+(.+)$", text)
    if rename:
        return TimekeepingCommand(action="edit_alarm", reference=_clean_reference(rename.group(1)), new_name=rename.group(2).strip())
    move = re.match(r"^(?:change|move)\s+(?:the\s+|my\s+)?(.+?)\s+alarm\s+to\s+.+$", text)
    if move:
        alarm_time = _parse_alarm_time(text)
        if alarm_time is None:
            return None
        return TimekeepingCommand(action="edit_alarm", reference=_clean_reference(move.group(1)), alarm_time=alarm_time)
    if re.match(r"^(?:list|show)\s+(?:my\s+)?alarms$", text) or re.match(r"^what\s+alarms\b", text):
        return TimekeepingCommand(action="list_alarms")
    snooze = re.match(r"^snooze\s+(?:(?:the|my|that|this)\s+)?(?:(.*?)\s+)?alarm(?:\s+for\s+(.+))?$", text)
    if snooze:
        duration = _duration_match(snooze.group(2) or "10 minutes")
        if duration is None:
            return None
        return TimekeepingCommand(action="snooze_alarm", reference=_clean_reference(snooze.group(1)), duration_seconds=duration[0])
    match = re.match(r"^(dismiss|enable|disable|cancel|stop)\s+(?:the\s+|my\s+)?(.+?)\s+alarm$", text)
    if match:
        action_map: dict[str, Action] = {
            "dismiss": "dismiss_alarm",
            "enable": "enable_alarm",
            "disable": "disable_alarm",
            "cancel": "cancel_alarm",
            "stop": "cancel_alarm",
        }
        return TimekeepingCommand(action=action_map[match.group(1)], reference=_clean_reference(match.group(2)))
    return None


def _parse_alarm_create(text: str, *, user_timezone: str, now: datetime) -> TimekeepingCommand | None:
    if not (re.match(r"^(?:set|create)\b.*\balarm\b", text) or re.match(r"^wake\s+me\s+at\b", text)):
        return None
    alarm_time = _parse_alarm_time(text)
    if alarm_time is None:
        return None
    weekdays = _alarm_recurrence(text)
    alarm_date = None if weekdays else _relative_alarm_date(text, user_timezone=user_timezone, now=now)
    name: str | None = None
    named = re.match(r"^(?:set|create)\s+(?:an?\s+|my\s+)?(.+?)\s+alarm\b", text)
    if named:
        candidate = named.group(1).strip()
        if candidate not in {"a", "an", "my"}:
            name = _clean_reference(candidate)
    return TimekeepingCommand(action="create_alarm", reference=name, alarm_time=alarm_time, alarm_date=alarm_date, weekdays=weekdays)


def _parse_timer_query(text: str) -> TimekeepingCommand | None:
    if "timer" not in text:
        return None
    if re.match(r"^(?:list|show)\s+(?:my\s+)?timers$", text):
        return TimekeepingCommand(action="list_timers")
    query_phrases = ("how much time", "time is left", "time left", "remaining", "how long")
    if not any(phrase in text for phrase in query_phrases):
        return None
    if "timers" in text and not re.search(r"\bthe\s+.+?\s+timer\b", text):
        return TimekeepingCommand(action="list_timers")
    named = re.search(r"(?:the|my)\s+(.+?)\s+timer", text)
    if named:
        return TimekeepingCommand(action="query_timer", reference=_clean_reference(named.group(1)))
    return TimekeepingCommand(action="list_timers")


def parse_timekeeping_command(
    transcript: str,
    *,
    user_timezone: str,
    now: datetime | None = None,
) -> TimekeepingCommand | None:
    """Parse a supported timer/alarm command without model-dependent arithmetic."""
    text = _normalize(transcript)
    if not text:
        return None
    text, target_text = _extract_target_clause(text)
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    command: TimekeepingCommand | None = None
    for parser in (_parse_alarm_management, _parse_timer_management, _parse_timer_query):
        command = parser(text)
        if command is not None:
            break
    if command is None:
        command = _parse_alarm_create(text, user_timezone=user_timezone, now=current)
    if command is None:
        command = _parse_timer_create(text)
    if command is not None and target_text is not None:
        command = replace(command, target_text=target_text)
    return command


__all__ = ["TimekeepingCommand", "parse_timekeeping_command"]