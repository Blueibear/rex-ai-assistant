"""Free-time finder — returns available meeting slots from a busy schedule.

Given a date range, a meeting duration, and a list of busy blocks (obtained
from :meth:`CalendarStub.get_free_busy` or any ``CalendarBackend``), this
module finds contiguous free windows that fit the requested meeting duration
within configurable working hours.

Usage example::

    from datetime import datetime, timezone, timedelta
    from rex.calendar_backends import CalendarStub
    from rex.calendar_backends.free_time_finder import find_free_slots

    stub = CalendarStub()
    start = datetime(2026, 3, 9, tzinfo=timezone.utc)   # Monday
    end   = datetime(2026, 3, 13, 18, tzinfo=timezone.utc)  # Friday EOD

    busy = stub.get_free_busy(start, end)
    slots = find_free_slots(busy, start, end, duration_minutes=30)
    for slot in slots[:5]:
        print(slot.start, "->", slot.end)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from rex.calendar_backends.free_busy_stub import FreeBusyBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class TimeSlot:
    """A single available meeting slot.

    Attributes:
        start: Start of the slot (UTC-aware).
        end:   End of the slot (UTC-aware).
    """

    start: datetime
    end: datetime

    @property
    def duration_minutes(self) -> int:
        """Duration of the slot in whole minutes."""
        return int((self.end - self.start).total_seconds() // 60)

    def __repr__(self) -> str:
        return f"TimeSlot(start={self.start.isoformat()!r}, " f"end={self.end.isoformat()!r})"


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def find_free_slots(
    busy_blocks: list[FreeBusyBlock],
    range_start: datetime,
    range_end: datetime,
    duration_minutes: int,
    *,
    day_start_hour: int = 9,
    day_end_hour: int = 18,
    slot_step_minutes: int = 30,
    max_slots: int | None = None,
) -> list[TimeSlot]:
    """Return available meeting slots within *range_start* .. *range_end*.

    The finder walks the date range day by day, clips each day to the
    configured working hours, removes the provided busy blocks, and emits
    candidate slots of exactly *duration_minutes* stepped every
    *slot_step_minutes* through each free gap.

    Parameters
    ----------
    busy_blocks:
        Busy intervals to exclude (e.g. from
        :meth:`~rex.calendar_backends.CalendarStub.get_free_busy`).
    range_start:
        Inclusive start of the search window.
    range_end:
        Exclusive end of the search window.
    duration_minutes:
        Required slot length in minutes.
    day_start_hour:
        First hour of the working day (24-hour, default 9).
    day_end_hour:
        Last hour of the working day (24-hour, default 18).
    slot_step_minutes:
        How many minutes to advance the cursor on each step within a free gap
        (controls granularity of returned candidates).
    max_slots:
        If given, stop after collecting this many slots.

    Returns
    -------
    list[TimeSlot]
        Available slots ordered by start time; empty if none found.
    """
    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be positive")
    if day_start_hour >= day_end_hour:
        raise ValueError("day_start_hour must be less than day_end_hour")
    if slot_step_minutes <= 0:
        raise ValueError("slot_step_minutes must be positive")

    duration = timedelta(minutes=duration_minutes)
    step = timedelta(minutes=slot_step_minutes)

    range_start_utc = _ensure_utc(range_start)
    range_end_utc = _ensure_utc(range_end)

    slots: list[TimeSlot] = []

    # Walk one calendar day at a time
    current_day = range_start_utc.date()
    end_day = range_end_utc.date()

    while current_day <= end_day:
        day_open = datetime(
            current_day.year,
            current_day.month,
            current_day.day,
            day_start_hour,
            tzinfo=timezone.utc,
        )
        day_close = datetime(
            current_day.year,
            current_day.month,
            current_day.day,
            day_end_hour,
            tzinfo=timezone.utc,
        )

        # Clip to the caller-supplied range
        window_start = max(range_start_utc, day_open)
        window_end = min(range_end_utc, day_close)

        if window_start < window_end:
            # Collect busy blocks that touch this day's window
            day_busy = [b for b in busy_blocks if b.overlaps(window_start, window_end)]
            day_busy.sort(key=lambda b: b.start)

            _collect_day_slots(
                slots,
                day_busy,
                window_start,
                window_end,
                duration,
                step,
                max_slots,
            )

        if max_slots is not None and len(slots) >= max_slots:
            break

        current_day += timedelta(days=1)

    logger.debug(
        "find_free_slots: found %d slot(s) for %d-min meeting between %s and %s",
        len(slots),
        duration_minutes,
        range_start_utc.isoformat(),
        range_end_utc.isoformat(),
    )
    return slots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_day_slots(
    slots: list[TimeSlot],
    day_busy: list[FreeBusyBlock],
    window_start: datetime,
    window_end: datetime,
    duration: timedelta,
    step: timedelta,
    max_slots: int | None,
) -> None:
    """Append free-gap slots for a single day into *slots* (in-place)."""
    # Build a list of free segments by stepping around the busy blocks
    # free_segments is a list of (gap_start, gap_end) pairs
    gaps: list[tuple[datetime, datetime]] = []
    cursor = window_start

    for block in day_busy:
        block_start = _ensure_utc(block.start)
        block_end = _ensure_utc(block.end)

        # Clip block to window
        block_start = max(block_start, window_start)
        block_end = min(block_end, window_end)

        if cursor < block_start:
            # There is a free gap before this block
            gaps.append((cursor, block_start))

        # Advance cursor past this busy block (don't go backwards)
        if block_end > cursor:
            cursor = block_end

    # Check for a free gap after the last busy block
    if cursor < window_end:
        gaps.append((cursor, window_end))

    # Emit slots from each gap
    for gap_start, gap_end in gaps:
        candidate = gap_start
        while candidate + duration <= gap_end:
            slots.append(TimeSlot(start=candidate, end=candidate + duration))
            if max_slots is not None and len(slots) >= max_slots:
                return
            candidate += step


def _ensure_utc(dt: datetime) -> datetime:
    """Return *dt* as a UTC-aware datetime."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


__all__ = [
    "TimeSlot",
    "find_free_slots",
]
