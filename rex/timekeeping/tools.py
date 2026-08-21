"""Canonical ToolRegistry handlers for first-class timers and alarms."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from rex.identity import validate_user_id
from rex.output_routing.models import OutputKind
from rex.output_routing.runtime import get_output_routing_service

from .models import AlarmRecord, TimerRecord
from .parser import Action, TimekeepingCommand, parse_timekeeping_command
from .runtime import ensure_timekeeping_runtime, get_timekeeping_service

_MUTATION_ACTIONS = {
    "create_timer", "pause_timer", "resume_timer", "cancel_timer", "rename_timer", "adjust_timer",
    "create_alarm", "edit_alarm", "snooze_alarm", "dismiss_alarm", "enable_alarm", "disable_alarm", "cancel_alarm",
}
_READ_ACTIONS = {"list_timers", "query_timer", "list_alarms"}


def resolve_user_timezone(user_id: str) -> str:
    """Resolve a user's canonical timezone, falling back to household geolocation."""
    owner = validate_user_id(user_id)
    timezone_name: str | None = None
    try:
        from rex.user_profile_service import UserProfileService
        profile = UserProfileService().get_profile(owner)
        raw = profile.preferences.get("timezone")
        if isinstance(raw, str) and raw.strip():
            timezone_name = raw.strip()
    except Exception:
        timezone_name = None
    if not timezone_name:
        try:
            from rex.geolocation import get_cached_timezone
            timezone_name = get_cached_timezone()
        except Exception:
            timezone_name = None
    timezone_name = timezone_name or "UTC"
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return "UTC"
    return timezone_name


def _structured_command(
    *, action: str, reference: str | None = None, duration_seconds: float | None = None,
    delta_seconds: float | None = None, new_name: str | None = None,
    alarm_time: str | time | None = None, alarm_date: str | date | None = None,
    weekdays: list[int] | tuple[int, ...] | None = None, target_text: str | None = None,
    target_volume: int | None = None,
) -> TimekeepingCommand:
    if action not in _MUTATION_ACTIONS | _READ_ACTIONS:
        raise ValueError(f"unsupported timekeeping action: {action}")
    parsed_time = time.fromisoformat(alarm_time) if isinstance(alarm_time, str) else alarm_time
    parsed_date = date.fromisoformat(alarm_date) if isinstance(alarm_date, str) else alarm_date
    return TimekeepingCommand(
        action=cast(Action, action), reference=reference, duration_seconds=duration_seconds,
        delta_seconds=delta_seconds, new_name=new_name, alarm_time=parsed_time,
        alarm_date=parsed_date, weekdays=tuple(weekdays or ()), target_text=target_text,
        target_volume=target_volume,
    )


def _command_from_request(*, transcript: str, user_timezone: str, action: str | None = None, **kwargs: Any) -> TimekeepingCommand:
    if action:
        return _structured_command(action=action, **kwargs)
    command = parse_timekeeping_command(transcript, user_timezone=user_timezone)
    if command is None:
        raise ValueError("I couldn't identify a supported timer or alarm command")
    return command


def _timer_payload(record: TimerRecord) -> dict[str, Any]:
    service = get_timekeeping_service()
    return {
        "timer_id": record.timer_id, "name": record.name, "status": record.status,
        "remaining_seconds": service.remaining_seconds(record.timer_id, record.user_id),
        "deadline_at": record.deadline_at.isoformat() if record.deadline_at else None,
        "output_target_id": record.output_target_id,
    }


def _alarm_payload(record: AlarmRecord) -> dict[str, Any]:
    return {
        "alarm_id": record.alarm_id, "name": record.name, "status": record.status,
        "enabled": record.enabled, "timezone_name": record.timezone_name,
        "local_time": record.local_time, "local_date": record.local_date,
        "weekdays": list(record.weekdays),
        "next_fire_at": record.next_fire_at.isoformat() if record.next_fire_at else None,
        "snooze_count": record.snooze_count, "output_target_id": record.output_target_id,
    }


def _matching_timers(user_id: str, reference: str | None, statuses: set[str]) -> list[TimerRecord]:
    records = [record for record in get_timekeeping_service().list_timers(user_id, include_inactive=True) if record.status in statuses]
    if reference is None:
        return records
    ref = reference.casefold().strip()
    return [record for record in records if record.timer_id.casefold() == ref or (record.name is not None and record.name.casefold() == ref)]


def _matching_alarms(user_id: str, reference: str | None, statuses: set[str]) -> list[AlarmRecord]:
    records = [record for record in get_timekeeping_service().list_alarms(user_id, include_inactive=True) if record.status in statuses]
    if reference is None:
        return records
    ref = reference.casefold().strip()
    return [record for record in records if record.alarm_id.casefold() == ref or (record.name is not None and record.name.casefold() == ref)]


def _require_one_timer(user_id: str, reference: str | None, statuses: set[str]) -> TimerRecord:
    matches = _matching_timers(user_id, reference, statuses)
    if not matches:
        raise ValueError("No matching timer was found for this user")
    if len(matches) > 1:
        raise ValueError("Multiple timers match. Use the timer name or ID to choose one")
    return matches[0]


def _require_one_alarm(user_id: str, reference: str | None, statuses: set[str]) -> AlarmRecord:
    matches = _matching_alarms(user_id, reference, statuses)
    if not matches:
        raise ValueError("No matching alarm was found for this user")
    if len(matches) > 1:
        raise ValueError("Multiple alarms match. Use the alarm name or ID to choose one")
    return matches[0]


def timekeeping_read(*, transcript: str = "", action: str | None = None, reference: str | None = None, _user_id: str = "", timezone_name: str | None = None, **kwargs: Any) -> dict[str, Any]:
    owner = validate_user_id(_user_id)
    zone = timezone_name or resolve_user_timezone(owner)
    command = _command_from_request(transcript=transcript, user_timezone=zone, action=action, reference=reference)
    if command.action not in _READ_ACTIONS:
        raise ValueError("This request changes a timer or alarm; use timekeeping_manage")
    service = get_timekeeping_service()
    if command.action == "list_timers":
        return {"timers": [_timer_payload(record) for record in service.list_timers(owner)]}
    if command.action == "list_alarms":
        records = [record for record in service.list_alarms(owner, include_inactive=True) if record.status not in {"dismissed", "canceled"}]
        return {"alarms": [_alarm_payload(record) for record in records]}
    matches = _matching_timers(owner, command.reference, {"active", "paused"})
    if not matches:
        return {"found": False, "details": "No matching active timer was found."}
    if len(matches) > 1:
        return {"ambiguous": True, "message": "Multiple timers match. Choose one by name or ID.", "matches": [_timer_payload(record) for record in matches]}
    return {"found": True, "timer": _timer_payload(matches[0])}


def _mutation_output(record_type: str, record: TimerRecord | AlarmRecord, *field_names: str) -> dict[str, Any]:
    dumped = record.model_dump(mode="json")
    record_id = record.timer_id if isinstance(record, TimerRecord) else record.alarm_id
    payload = _timer_payload(record) if isinstance(record, TimerRecord) else _alarm_payload(record)
    return {
        "record_type": record_type, "record_id": record_id, "user_id": record.user_id,
        "name": record.name, record_type: payload,
        "verification": {"record_type": record_type, "record_id": record_id, "user_id": record.user_id, "fields": {name: dumped.get(name) for name in field_names}},
    }


def _explicit_output_target(owner: str, command: TimekeepingCommand, zone: str) -> str | None:
    if command.target_text is None:
        return None
    kind = OutputKind.TIMER if command.action == "create_timer" else OutputKind.ALARM
    route = get_output_routing_service().resolve(
        user_id=owner,
        output_kind=kind,
        explicit_target_text=command.target_text,
        origin_device_id=None,
        at=datetime.now(UTC).astimezone(ZoneInfo(zone)),
    )
    if route.target_id is None:
        raise ValueError(f"Output target is unavailable: {route.reason}")
    return route.target_id


def timekeeping_manage(
    *, transcript: str = "", action: str | None = None, reference: str | None = None,
    duration_seconds: float | None = None, delta_seconds: float | None = None,
    new_name: str | None = None, alarm_time: str | time | None = None,
    alarm_date: str | date | None = None, weekdays: list[int] | tuple[int, ...] | None = None,
    target_text: str | None = None, target_volume: int | None = None,
    _user_id: str = "", timezone_name: str | None = None, **kwargs: Any,
) -> dict[str, Any]:
    owner = validate_user_id(_user_id)
    zone = timezone_name or resolve_user_timezone(owner)
    command = _command_from_request(
        transcript=transcript, user_timezone=zone, action=action, reference=reference,
        duration_seconds=duration_seconds, delta_seconds=delta_seconds, new_name=new_name,
        alarm_time=alarm_time, alarm_date=alarm_date, weekdays=weekdays,
        target_text=target_text, target_volume=target_volume,
    )
    if command.action not in _MUTATION_ACTIONS:
        raise ValueError("This request only reads timer/alarm state; use timekeeping_read")
    ensure_timekeeping_runtime()
    service = get_timekeeping_service()

    if command.action == "create_timer":
        if command.duration_seconds is None:
            raise ValueError("timer duration is required")
        output_target_id = _explicit_output_target(owner, command, zone)
        record = service.create_timer(owner, command.duration_seconds, name=command.reference, output_target_id=output_target_id)
        return _mutation_output("timer", record, "status", "name", "deadline_at", "output_target_id")

    if command.action in {"pause_timer", "resume_timer", "cancel_timer", "rename_timer", "adjust_timer"}:
        statuses = {
            "pause_timer": {"active"}, "resume_timer": {"paused"},
            "cancel_timer": {"active", "paused"}, "rename_timer": {"active", "paused"},
            "adjust_timer": {"active", "paused"},
        }[command.action]
        timer_record = _require_one_timer(owner, command.reference, statuses)
        if command.action == "pause_timer":
            updated = service.pause_timer(timer_record.timer_id, owner)
        elif command.action == "resume_timer":
            updated = service.resume_timer(timer_record.timer_id, owner)
        elif command.action == "cancel_timer":
            service.cancel_timer(timer_record.timer_id, owner)
            updated = service.get_timer(timer_record.timer_id, owner)
        elif command.action == "rename_timer":
            if not command.new_name:
                raise ValueError("new timer name is required")
            updated = service.rename_timer(timer_record.timer_id, owner, command.new_name)
        else:
            if command.delta_seconds is None:
                raise ValueError("timer adjustment is required")
            updated = service.adjust_timer(timer_record.timer_id, owner, command.delta_seconds)
        if updated is None:
            raise RuntimeError("timer mutation did not produce updated state")
        return _mutation_output("timer", updated, "status", "name", "deadline_at", "paused_remaining_seconds")

    if command.action == "create_alarm":
        if command.alarm_time is None:
            raise ValueError("alarm time is required")
        output_target_id = _explicit_output_target(owner, command, zone)
        alarm_record = service.create_alarm(
            owner, local_time=command.alarm_time, timezone_name=zone,
            local_date=command.alarm_date, weekdays=command.weekdays,
            name=command.reference, output_target_id=output_target_id,
        )
        return _mutation_output("alarm", alarm_record, "status", "enabled", "name", "next_fire_at", "output_target_id")

    alarm_statuses = {
        "edit_alarm": {"active", "ringing"}, "snooze_alarm": {"ringing"},
        "dismiss_alarm": {"ringing"}, "enable_alarm": {"active"},
        "disable_alarm": {"active", "ringing"}, "cancel_alarm": {"active", "ringing"},
    }
    alarm_record = _require_one_alarm(owner, command.reference, alarm_statuses[command.action])
    if command.action == "edit_alarm":
        updated_alarm = service.edit_alarm(
            alarm_record.alarm_id, owner, local_time=command.alarm_time,
            timezone_name=timezone_name, local_date=command.alarm_date,
            weekdays=tuple(weekdays) if weekdays is not None else None, name=command.new_name,
        )
    elif command.action == "snooze_alarm":
        updated_alarm = service.snooze_alarm(alarm_record.alarm_id, owner, command.duration_seconds or 600)
    elif command.action == "dismiss_alarm":
        updated_alarm = service.dismiss_alarm(alarm_record.alarm_id, owner)
    elif command.action == "enable_alarm":
        updated_alarm = service.enable_alarm(alarm_record.alarm_id, owner)
    elif command.action == "disable_alarm":
        updated_alarm = service.disable_alarm(alarm_record.alarm_id, owner)
    else:
        service.cancel_alarm(alarm_record.alarm_id, owner)
        updated_alarm = service.get_alarm(alarm_record.alarm_id, owner)
    if updated_alarm is None:
        raise RuntimeError("alarm mutation did not produce updated state")
    return _mutation_output(
        "alarm", updated_alarm, "status", "enabled", "name", "local_time",
        "local_date", "weekdays", "next_fire_at", "snooze_count",
    )


def verify_timekeeping_mutation(args: dict[str, Any], output: Any) -> bool:
    """Re-read persisted state before allowing a mutation to be called verified."""
    if not isinstance(output, dict):
        return False
    verification = output.get("verification")
    if not isinstance(verification, dict):
        return False
    record_type = verification.get("record_type")
    record_id = verification.get("record_id")
    user_id = verification.get("user_id")
    expected = verification.get("fields")
    if not isinstance(record_type, str) or not record_type:
        return False
    if not isinstance(record_id, str) or not record_id:
        return False
    if not isinstance(user_id, str) or not user_id:
        return False
    if not isinstance(expected, dict):
        return False
    service = get_timekeeping_service()
    record: TimerRecord | AlarmRecord | None
    if record_type == "timer":
        record = service.get_timer(record_id, user_id)
    elif record_type == "alarm":
        record = service.get_alarm(record_id, user_id)
    else:
        return False
    if record is None:
        return False
    persisted = record.model_dump(mode="json")
    return all(persisted.get(name) == value for name, value in expected.items())


__all__ = ["resolve_user_timezone", "timekeeping_manage", "timekeeping_read", "verify_timekeeping_mutation"]
