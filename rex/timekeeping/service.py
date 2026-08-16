"""Persistent, identity-scoped timer and alarm service."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from rex.identity import validate_user_id
from rex.runtime_paths import household_data_path

from .models import AlarmRecord, DueEvent, TimerRecord, ensure_utc, utc_now


class TimekeepingService:
    """Own canonical timer/alarm state without owning a polling thread."""

    def __init__(
        self,
        storage_path: Path | str | None = None,
        *,
        now_func: Callable[[], datetime] | None = None,
    ) -> None:
        self.storage_path = Path(storage_path or household_data_path("timekeeping.json"))
        self._now = now_func or utc_now
        self._lock = threading.RLock()
        self._timers: dict[str, TimerRecord] = {}
        self._alarms: dict[str, AlarmRecord] = {}
        self._load()

    def _now_utc(self) -> datetime:
        return ensure_utc(self._now())

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        timers = raw.get("timers", []) if isinstance(raw, dict) else []
        alarms = raw.get("alarms", []) if isinstance(raw, dict) else []
        self._timers = {
            item.timer_id: item for item in (TimerRecord.model_validate(value) for value in timers)
        }
        self._alarms = {
            item.alarm_id: item for item in (AlarmRecord.model_validate(value) for value in alarms)
        }

    def _save(self) -> None:
        payload = {
            "version": 1,
            "timers": [record.model_dump(mode="json") for record in self._timers.values()],
            "alarms": [record.model_dump(mode="json") for record in self._alarms.values()],
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self.storage_path)

    @staticmethod
    def _owned(record: TimerRecord | AlarmRecord | None, user_id: str):
        owner = validate_user_id(user_id)
        if record is None or record.user_id != owner:
            return None
        return record

    def create_timer(
        self,
        user_id: str,
        duration_seconds: float,
        *,
        name: str | None = None,
        output_target_id: str | None = None,
    ) -> TimerRecord:
        owner = validate_user_id(user_id)
        duration = float(duration_seconds)
        if duration <= 0:
            raise ValueError("timer duration must be greater than zero")
        now = self._now_utc()
        timer = TimerRecord(
            user_id=owner,
            name=name.strip() if isinstance(name, str) and name.strip() else None,
            created_at=now,
            deadline_at=now + timedelta(seconds=duration),
            original_duration_seconds=duration,
            output_target_id=output_target_id,
        )
        with self._lock:
            self._timers[timer.timer_id] = timer
            self._save()
        return timer.model_copy(deep=True)

    def get_timer(self, timer_id: str, user_id: str) -> TimerRecord | None:
        with self._lock:
            record = self._owned(self._timers.get(timer_id), user_id)
            return record.model_copy(deep=True) if record is not None else None

    def list_timers(self, user_id: str, *, include_inactive: bool = False) -> list[TimerRecord]:
        owner = validate_user_id(user_id)
        with self._lock:
            records = [record for record in self._timers.values() if record.user_id == owner]
            if not include_inactive:
                records = [record for record in records if record.status in {"active", "paused"}]
            records.sort(key=lambda record: record.created_at)
            return [record.model_copy(deep=True) for record in records]

    def remaining_seconds(self, timer_id: str, user_id: str) -> float | None:
        with self._lock:
            record = self._owned(self._timers.get(timer_id), user_id)
            return record.remaining_seconds(self._now_utc()) if record is not None else None

    def pause_timer(self, timer_id: str, user_id: str) -> TimerRecord | None:
        with self._lock:
            record = self._owned(self._timers.get(timer_id), user_id)
            if record is None or record.status != "active":
                return None
            record.paused_remaining_seconds = record.remaining_seconds(self._now_utc())
            record.deadline_at = None
            record.status = "paused"
            self._save()
            return record.model_copy(deep=True)

    def resume_timer(self, timer_id: str, user_id: str) -> TimerRecord | None:
        with self._lock:
            record = self._owned(self._timers.get(timer_id), user_id)
            if record is None or record.status != "paused":
                return None
            remaining = max(0.0, float(record.paused_remaining_seconds or 0.0))
            record.deadline_at = self._now_utc() + timedelta(seconds=remaining)
            record.paused_remaining_seconds = None
            record.status = "active"
            self._save()
            return record.model_copy(deep=True)

    def adjust_timer(self, timer_id: str, user_id: str, delta_seconds: float) -> TimerRecord | None:
        with self._lock:
            record = self._owned(self._timers.get(timer_id), user_id)
            if record is None or record.status not in {"active", "paused"}:
                return None
            if record.status == "paused":
                current = float(record.paused_remaining_seconds or 0.0)
                record.paused_remaining_seconds = max(0.0, current + float(delta_seconds))
            else:
                assert record.deadline_at is not None
                record.deadline_at = max(
                    self._now_utc(), record.deadline_at + timedelta(seconds=float(delta_seconds))
                )
            self._save()
            return record.model_copy(deep=True)

    def rename_timer(self, timer_id: str, user_id: str, name: str) -> TimerRecord | None:
        normalized = name.strip()
        if not normalized:
            raise ValueError("timer name cannot be blank")
        with self._lock:
            record = self._owned(self._timers.get(timer_id), user_id)
            if record is None:
                return None
            record.name = normalized
            self._save()
            return record.model_copy(deep=True)

    def cancel_timer(self, timer_id: str, user_id: str) -> bool:
        with self._lock:
            record = self._owned(self._timers.get(timer_id), user_id)
            if record is None or record.status in {"fired", "canceled"}:
                return False
            record.status = "canceled"
            record.canceled_at = self._now_utc()
            record.deadline_at = None
            record.paused_remaining_seconds = None
            self._save()
            return True

    @staticmethod
    def _zone(timezone_name: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {timezone_name}") from exc

    @classmethod
    def _local_candidate(cls, local_date: date, local_time: time, timezone_name: str) -> datetime:
        zone = cls._zone(timezone_name)
        local = datetime.combine(local_date, local_time, tzinfo=zone)
        return local.astimezone(UTC)

    @classmethod
    def _next_recurring_occurrence(
        cls,
        *,
        local_time: time,
        timezone_name: str,
        weekdays: tuple[int, ...],
        after: datetime,
    ) -> datetime:
        if not weekdays:
            raise ValueError("recurring alarms require at least one weekday")
        threshold = ensure_utc(after)
        local_threshold = threshold.astimezone(cls._zone(timezone_name))
        for offset in range(8):
            candidate_date = local_threshold.date() + timedelta(days=offset)
            if candidate_date.weekday() not in weekdays:
                continue
            candidate = cls._local_candidate(candidate_date, local_time, timezone_name)
            if candidate > threshold:
                return candidate
        raise RuntimeError("could not calculate next recurring alarm occurrence")

    def create_alarm(
        self,
        user_id: str,
        *,
        local_time: time,
        timezone_name: str,
        local_date: date | None = None,
        weekdays: tuple[int, ...] = (),
        name: str | None = None,
        output_target_id: str | None = None,
    ) -> AlarmRecord:
        owner = validate_user_id(user_id)
        self._zone(timezone_name)
        weekdays = tuple(sorted(set(weekdays)))
        if any(day < 0 or day > 6 for day in weekdays):
            raise ValueError("weekdays must use Monday=0 through Sunday=6")
        if local_date is not None and weekdays:
            raise ValueError("alarm cannot be both one-shot dated and recurring")

        now = self._now_utc()
        resolved_date = local_date
        if weekdays:
            next_fire = self._next_recurring_occurrence(
                local_time=local_time,
                timezone_name=timezone_name,
                weekdays=weekdays,
                after=now,
            )
        else:
            if resolved_date is None:
                local_now = now.astimezone(self._zone(timezone_name))
                resolved_date = local_now.date()
                candidate = self._local_candidate(resolved_date, local_time, timezone_name)
                if candidate <= now:
                    resolved_date += timedelta(days=1)
            next_fire = self._local_candidate(resolved_date, local_time, timezone_name)
            if next_fire <= now:
                raise ValueError("one-shot alarm must be scheduled in the future")

        alarm = AlarmRecord(
            user_id=owner,
            name=name.strip() if isinstance(name, str) and name.strip() else None,
            timezone_name=timezone_name,
            local_time=local_time.isoformat(),
            local_date=resolved_date.isoformat() if resolved_date is not None else None,
            weekdays=weekdays,
            created_at=now,
            next_fire_at=next_fire,
            output_target_id=output_target_id,
        )
        with self._lock:
            self._alarms[alarm.alarm_id] = alarm
            self._save()
        return alarm.model_copy(deep=True)

    def get_alarm(self, alarm_id: str, user_id: str) -> AlarmRecord | None:
        with self._lock:
            record = self._owned(self._alarms.get(alarm_id), user_id)
            return record.model_copy(deep=True) if record is not None else None

    def list_alarms(self, user_id: str, *, include_inactive: bool = False) -> list[AlarmRecord]:
        owner = validate_user_id(user_id)
        with self._lock:
            records = [record for record in self._alarms.values() if record.user_id == owner]
            if not include_inactive:
                records = [
                    record
                    for record in records
                    if record.status not in {"dismissed", "canceled"} and record.enabled
                ]
            records.sort(key=lambda record: record.next_fire_at or datetime.max.replace(tzinfo=UTC))
            return [record.model_copy(deep=True) for record in records]

    def dismiss_alarm(self, alarm_id: str, user_id: str) -> AlarmRecord | None:
        with self._lock:
            record = self._owned(self._alarms.get(alarm_id), user_id)
            if record is None or record.status != "ringing":
                return None
            now = self._now_utc()
            record.dismissed_at = now
            if record.recurring:
                record.status = "active"
                record.next_fire_at = self._next_recurring_occurrence(
                    local_time=time.fromisoformat(record.local_time),
                    timezone_name=record.timezone_name,
                    weekdays=record.weekdays,
                    after=now,
                )
            else:
                record.status = "dismissed"
                record.enabled = False
                record.next_fire_at = None
            self._save()
            return record.model_copy(deep=True)

    def snooze_alarm(
        self, alarm_id: str, user_id: str, duration_seconds: float = 600
    ) -> AlarmRecord | None:
        duration = float(duration_seconds)
        if duration <= 0:
            raise ValueError("snooze duration must be greater than zero")
        with self._lock:
            record = self._owned(self._alarms.get(alarm_id), user_id)
            if record is None or record.status != "ringing":
                return None
            record.status = "active"
            record.enabled = True
            record.snooze_count += 1
            record.next_fire_at = self._now_utc() + timedelta(seconds=duration)
            self._save()
            return record.model_copy(deep=True)

    def disable_alarm(self, alarm_id: str, user_id: str) -> AlarmRecord | None:
        with self._lock:
            record = self._owned(self._alarms.get(alarm_id), user_id)
            if record is None or record.status in {"dismissed", "canceled"}:
                return None
            record.enabled = False
            self._save()
            return record.model_copy(deep=True)

    def enable_alarm(self, alarm_id: str, user_id: str) -> AlarmRecord | None:
        with self._lock:
            record = self._owned(self._alarms.get(alarm_id), user_id)
            if record is None or record.status in {"dismissed", "canceled"}:
                return None
            now = self._now_utc()
            if record.next_fire_at is None or record.next_fire_at <= now:
                if record.recurring:
                    record.next_fire_at = self._next_recurring_occurrence(
                        local_time=time.fromisoformat(record.local_time),
                        timezone_name=record.timezone_name,
                        weekdays=record.weekdays,
                        after=now,
                    )
                elif record.local_date is not None:
                    candidate = self._local_candidate(
                        date.fromisoformat(record.local_date),
                        time.fromisoformat(record.local_time),
                        record.timezone_name,
                    )
                    if candidate <= now:
                        return None
                    record.next_fire_at = candidate
            record.status = "active"
            record.enabled = True
            self._save()
            return record.model_copy(deep=True)

    def edit_alarm(
        self,
        alarm_id: str,
        user_id: str,
        *,
        local_time: time | None = None,
        timezone_name: str | None = None,
        local_date: date | None = None,
        weekdays: tuple[int, ...] | None = None,
        name: str | None = None,
    ) -> AlarmRecord | None:
        with self._lock:
            record = self._owned(self._alarms.get(alarm_id), user_id)
            if record is None or record.status in {"dismissed", "canceled"}:
                return None
            next_zone = timezone_name or record.timezone_name
            self._zone(next_zone)
            next_time = local_time or time.fromisoformat(record.local_time)
            next_weekdays = (
                tuple(sorted(set(weekdays))) if weekdays is not None else record.weekdays
            )
            if any(day < 0 or day > 6 for day in next_weekdays):
                raise ValueError("weekdays must use Monday=0 through Sunday=6")
            next_date = local_date
            if weekdays is None and local_date is None and not next_weekdays and record.local_date:
                next_date = date.fromisoformat(record.local_date)
            if next_date is not None and next_weekdays:
                raise ValueError("alarm cannot be both one-shot dated and recurring")
            now = self._now_utc()
            if next_weekdays:
                next_fire = self._next_recurring_occurrence(
                    local_time=next_time,
                    timezone_name=next_zone,
                    weekdays=next_weekdays,
                    after=now,
                )
                next_date = None
            else:
                if next_date is None:
                    local_now = now.astimezone(self._zone(next_zone))
                    next_date = local_now.date()
                    if self._local_candidate(next_date, next_time, next_zone) <= now:
                        next_date += timedelta(days=1)
                next_fire = self._local_candidate(next_date, next_time, next_zone)
                if next_fire <= now:
                    raise ValueError("one-shot alarm must be scheduled in the future")
            record.timezone_name = next_zone
            record.local_time = next_time.isoformat()
            record.local_date = next_date.isoformat() if next_date is not None else None
            record.weekdays = next_weekdays
            record.next_fire_at = next_fire
            record.status = "active"
            if name is not None:
                normalized_name = name.strip()
                if not normalized_name:
                    raise ValueError("alarm name cannot be blank")
                record.name = normalized_name
            self._save()
            return record.model_copy(deep=True)

    def cancel_alarm(self, alarm_id: str, user_id: str) -> bool:
        with self._lock:
            record = self._owned(self._alarms.get(alarm_id), user_id)
            if record is None or record.status == "canceled":
                return False
            record.status = "canceled"
            record.enabled = False
            record.canceled_at = self._now_utc()
            record.next_fire_at = None
            self._save()
            return True

    def claim_due_events(self, now: datetime | None = None) -> list[DueEvent]:
        check_time = ensure_utc(now) if now is not None else self._now_utc()
        events: list[DueEvent] = []
        with self._lock:
            for timer in self._timers.values():
                if (
                    timer.status == "active"
                    and timer.deadline_at is not None
                    and timer.deadline_at <= check_time
                ):
                    timer.status = "fired"
                    timer.fired_at = check_time
                    timer.deadline_at = None
                    events.append(
                        DueEvent(
                            kind="timer",
                            record_id=timer.timer_id,
                            user_id=timer.user_id,
                            name=timer.name,
                            fired_at=check_time,
                            output_target_id=timer.output_target_id,
                        )
                    )
            for alarm in self._alarms.values():
                if (
                    alarm.enabled
                    and alarm.status == "active"
                    and alarm.next_fire_at is not None
                    and alarm.next_fire_at <= check_time
                ):
                    alarm.status = "ringing"
                    alarm.last_fired_at = check_time
                    alarm.next_fire_at = None
                    events.append(
                        DueEvent(
                            kind="alarm",
                            record_id=alarm.alarm_id,
                            user_id=alarm.user_id,
                            name=alarm.name,
                            fired_at=check_time,
                            output_target_id=alarm.output_target_id,
                        )
                    )
            if events:
                self._save()
        events.sort(key=lambda event: (event.fired_at, event.kind, event.record_id))
        return events


__all__ = ["TimekeepingService"]
