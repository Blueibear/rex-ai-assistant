"""Canonical timer and alarm models for AskRex."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from rex.identity import validate_user_id

TimerStatus = Literal["active", "paused", "fired", "canceled"]
AlarmStatus = Literal["active", "ringing", "dismissed", "canceled"]


def utc_now() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timekeeping datetimes must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class DueEvent:
    kind: Literal["timer", "alarm"]
    record_id: str
    user_id: str
    name: str | None
    fired_at: datetime
    output_target_id: str | None = None


class TimerRecord(BaseModel):
    timer_id: str = Field(default_factory=lambda: f"tmr_{uuid.uuid4().hex[:12]}")
    user_id: str
    name: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    deadline_at: datetime | None
    original_duration_seconds: float
    paused_remaining_seconds: float | None = None
    status: TimerStatus = "active"
    fired_at: datetime | None = None
    canceled_at: datetime | None = None
    output_target_id: str | None = None

    @field_validator("user_id")
    @classmethod
    def validate_owner(cls, value: str) -> str:
        return validate_user_id(value)

    @field_validator("deadline_at", "created_at", "fired_at", "canceled_at")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value) if value is not None else None

    def remaining_seconds(self, now: datetime) -> float:
        if self.status == "paused":
            return max(0.0, float(self.paused_remaining_seconds or 0.0))
        if self.status != "active" or self.deadline_at is None:
            return 0.0
        return max(0.0, (self.deadline_at - ensure_utc(now)).total_seconds())


class AlarmRecord(BaseModel):
    alarm_id: str = Field(default_factory=lambda: f"alm_{uuid.uuid4().hex[:12]}")
    user_id: str
    name: str | None = None
    timezone_name: str
    local_time: str
    local_date: str | None = None
    weekdays: tuple[int, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)
    next_fire_at: datetime | None
    enabled: bool = True
    status: AlarmStatus = "active"
    last_fired_at: datetime | None = None
    dismissed_at: datetime | None = None
    canceled_at: datetime | None = None
    snooze_count: int = 0
    output_target_id: str | None = None

    @field_validator("user_id")
    @classmethod
    def validate_owner(cls, value: str) -> str:
        return validate_user_id(value)

    @field_validator("weekdays")
    @classmethod
    def validate_weekdays(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        normalized = tuple(sorted(set(value)))
        if any(day < 0 or day > 6 for day in normalized):
            raise ValueError("weekdays must use Monday=0 through Sunday=6")
        return normalized

    @field_validator(
        "created_at",
        "next_fire_at",
        "last_fired_at",
        "dismissed_at",
        "canceled_at",
    )
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value) if value is not None else None

    @property
    def recurring(self) -> bool:
        return bool(self.weekdays)


__all__ = [
    "AlarmRecord",
    "AlarmStatus",
    "DueEvent",
    "TimerRecord",
    "TimerStatus",
    "ensure_utc",
    "utc_now",
]
