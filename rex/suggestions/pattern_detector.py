"""Proactive pattern detection for Home Assistant command history.

Detects repeated command patterns (e.g., "user turns on kitchen light every
morning") and returns suggested automations.  Suggestions are *never* acted on
automatically; they are always surfaced to the user as questions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class PatternEntry:
    """A single command event recorded for pattern analysis.

    Args:
        entity_id: Home Assistant entity identifier (e.g. ``light.kitchen``).
        service: HA service called (e.g. ``turn_on``, ``turn_off``).
        timestamp: Unix epoch timestamp of when the command was issued.
    """

    entity_id: str
    service: str
    timestamp: float


def detect_patterns(
    history: list[PatternEntry],
    min_occurrences: int = 3,
    time_window_hours: float = 2.0,
) -> list[dict[str, Any]]:
    """Detect repeated command patterns in *history*.

    Commands are grouped by ``(entity_id, service, time_bucket)`` where
    *time_bucket* is ``floor(hour / time_window_hours)``.  Any combination that
    appears at least *min_occurrences* times is returned as a pattern.

    Args:
        history: Sequence of :class:`PatternEntry` objects to analyse.
        min_occurrences: Minimum number of matching events required before a
            combination is considered a pattern (default 3).
        time_window_hours: Size of the hourly bucket used for grouping events
            by approximate time-of-day (default 2.0 → 12 buckets per day).

    Returns:
        List of dicts, each with keys:

        * ``"pattern"`` – human-readable description of the detected pattern.
        * ``"frequency"`` – number of times the pattern was observed.
        * ``"suggested_automation"`` – plain-English automation suggestion.

        Results are sorted by *frequency* descending.

    Note:
        Suggestions are never acted on automatically.  They must always be
        presented to the user as a question before any action is taken.
    """
    if not history:
        return []

    window = max(time_window_hours, 0.5)
    buckets: dict[tuple[str, str, int], int] = defaultdict(int)

    for entry in history:
        dt = datetime.fromtimestamp(entry.timestamp)
        bucket = int(dt.hour / window)
        buckets[(entry.entity_id, entry.service, bucket)] += 1

    results: list[dict[str, Any]] = []
    for (entity_id, service, bucket), count in buckets.items():
        if count < min_occurrences:
            continue

        start_hour = int(bucket * window)
        end_hour = min(int(start_hour + window), 24)
        time_str = f"{start_hour:02d}:00–{end_hour:02d}:00"

        entity_name = entity_id.split(".")[-1].replace("_", " ")
        friendly_service = service.replace("_", " ")
        pattern = f"{friendly_service} {entity_name} around {time_str}"

        suggested_automation = f"Automate: {service} {entity_id} daily at {start_hour:02d}:00"

        results.append(
            {
                "pattern": pattern,
                "frequency": count,
                "suggested_automation": suggested_automation,
                "entity_id": entity_id,
                "service": service,
                "start_hour": start_hour,
            }
        )

    results.sort(key=lambda x: x["frequency"], reverse=True)
    return results
