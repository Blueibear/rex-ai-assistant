"""Typed per-user output-routing policy models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import StrEnum


class OutputKind(StrEnum):
    """Kinds of user-visible audio output governed by routing policy."""

    SPOKEN_RESPONSE = "spoken_response"
    TIMER = "timer"
    ALARM = "alarm"
    MEDIA = "media"


class FallbackMode(StrEnum):
    """Behavior when the preferred output target cannot be used."""

    NONE = "none"
    NAMED = "named"
    ASK = "ask"


def _validate_target_id(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty canonical target ID")
    return value


def _validate_volume(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError(f"{field_name} must be an integer from 0 through 100")
    return value


def _validate_days(days: tuple[int, ...], *, field_name: str) -> tuple[int, ...]:
    normalized = tuple(days)
    if any(
        isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6 for day in normalized
    ):
        raise ValueError(f"{field_name} values must be weekday integers 0 through 6")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicate weekdays")
    return normalized


def _time_in_window(value: time, start: time, end: time) -> bool:
    """Return whether *value* lies in a same-day or overnight local-time window."""
    if start == end:
        return True
    if start < end:
        return start <= value < end
    return value >= start or value < end


@dataclass(frozen=True, slots=True)
class QuietHours:
    """Optional local-time window that may suppress non-required audio."""

    enabled: bool = False
    start_local_time: time = time(22, 0)
    end_local_time: time = time(7, 0)
    days_of_week: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "days_of_week",
            _validate_days(self.days_of_week, field_name="quiet-hours days"),
        )

    def active_at(self, at: datetime) -> bool:
        if not self.enabled:
            return False
        if self.days_of_week and at.weekday() not in self.days_of_week:
            return False
        return _time_in_window(
            at.timetz().replace(tzinfo=None),
            self.start_local_time,
            self.end_local_time,
        )


@dataclass(frozen=True, slots=True)
class RoutingRule:
    """Conditional target rule evaluated before a per-kind default."""

    output_kind: OutputKind
    target_id: str
    days_of_week: tuple[int, ...] = ()
    start_local_time: time | None = None
    end_local_time: time | None = None
    target_volume: int | None = None
    fallback_mode: FallbackMode | None = None
    fallback_target_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_kind", OutputKind(self.output_kind))
        object.__setattr__(
            self,
            "target_id",
            _validate_target_id(self.target_id, field_name="rule target_id"),
        )
        object.__setattr__(
            self,
            "days_of_week",
            _validate_days(self.days_of_week, field_name="rule days"),
        )
        object.__setattr__(
            self,
            "target_volume",
            _validate_volume(self.target_volume, field_name="rule target_volume"),
        )
        if self.fallback_mode is not None:
            object.__setattr__(self, "fallback_mode", FallbackMode(self.fallback_mode))
        object.__setattr__(
            self,
            "fallback_target_id",
            _validate_target_id(
                self.fallback_target_id,
                field_name="rule fallback_target_id",
            ),
        )
        if (self.start_local_time is None) != (self.end_local_time is None):
            raise ValueError("rule start_local_time and end_local_time must be set together")
        if self.fallback_mode is FallbackMode.NAMED and self.fallback_target_id is None:
            raise ValueError("named rule fallback requires fallback_target_id")
        if self.fallback_mode is not FallbackMode.NAMED and self.fallback_target_id is not None:
            raise ValueError("rule fallback_target_id is only valid for named fallback")

    def matches(self, output_kind: OutputKind, at: datetime) -> bool:
        if self.output_kind is not OutputKind(output_kind):
            return False
        if self.days_of_week and at.weekday() not in self.days_of_week:
            return False
        if self.start_local_time is None or self.end_local_time is None:
            return True
        local_time = at.timetz().replace(tzinfo=None)
        return _time_in_window(local_time, self.start_local_time, self.end_local_time)


_POLICY_TARGET_FIELDS = (
    "spoken_response_target_id",
    "timer_target_id",
    "alarm_target_id",
    "media_target_id",
    "spoken_response_fallback_target_id",
    "timer_fallback_target_id",
    "alarm_fallback_target_id",
    "media_fallback_target_id",
)
_POLICY_FALLBACK_FIELDS = (
    "spoken_response_fallback",
    "timer_fallback",
    "alarm_fallback",
    "media_fallback",
)
_POLICY_VOLUME_FIELDS = (
    "spoken_response_volume",
    "timer_volume",
    "alarm_volume",
    "media_volume",
)


def _normalize_policy_targets(policy: UserOutputPolicy) -> None:
    for name in _POLICY_TARGET_FIELDS:
        object.__setattr__(
            policy,
            name,
            _validate_target_id(getattr(policy, name), field_name=name),
        )


def _normalize_policy_fallbacks(policy: UserOutputPolicy) -> None:
    for name in _POLICY_FALLBACK_FIELDS:
        object.__setattr__(policy, name, FallbackMode(getattr(policy, name)))


def _normalize_policy_volumes(policy: UserOutputPolicy) -> None:
    for name in _POLICY_VOLUME_FIELDS:
        object.__setattr__(
            policy,
            name,
            _validate_volume(getattr(policy, name), field_name=name),
        )


def _validate_policy_media_account(policy: UserOutputPolicy) -> None:
    if not isinstance(policy.prefer_media_request_origin, bool):
        raise ValueError("prefer_media_request_origin must be boolean")
    for name, label in (
        ("default_media_provider", "provider ID"),
        ("default_media_account_id", "account ID"),
    ):
        value = getattr(policy, name)
        if value is not None and (
            not isinstance(value, str) or not value.strip() or value != value.strip()
        ):
            raise ValueError(f"{name} must be a non-empty {label}")
    if (policy.default_media_provider is None) != (policy.default_media_account_id is None):
        raise ValueError("default_media_provider and default_media_account_id must be set together")


def _normalize_policy_collections(policy: UserOutputPolicy) -> None:
    object.__setattr__(
        policy,
        "quiet_hours",
        (
            policy.quiet_hours
            if isinstance(policy.quiet_hours, QuietHours)
            else QuietHours(**policy.quiet_hours)
        ),
    )
    object.__setattr__(
        policy,
        "rules",
        tuple(
            rule if isinstance(rule, RoutingRule) else RoutingRule(**rule) for rule in policy.rules
        ),
    )


def _validate_policy_fallback_targets(policy: UserOutputPolicy) -> None:
    for kind in OutputKind:
        mode, fallback_target = policy.fallback_for(kind)
        if mode is FallbackMode.NAMED and fallback_target is None:
            raise ValueError(f"named {kind.value} fallback requires a target")
        if mode is not FallbackMode.NAMED and fallback_target is not None:
            raise ValueError(f"{kind.value} fallback target is only valid for named fallback")


@dataclass(frozen=True, slots=True)
class UserOutputPolicy:
    """Persisted user-owned routing preferences.

    Target IDs are canonical IDs from ``AudioTargetRegistry``. Storing an ID
    does not grant authority; the registry is consulted again on every
    resolution.
    """

    spoken_response_target_id: str | None = None
    timer_target_id: str | None = None
    alarm_target_id: str | None = None
    media_target_id: str | None = None

    spoken_response_fallback: FallbackMode = FallbackMode.NONE
    timer_fallback: FallbackMode = FallbackMode.NONE
    alarm_fallback: FallbackMode = FallbackMode.NONE
    media_fallback: FallbackMode = FallbackMode.NONE

    spoken_response_fallback_target_id: str | None = None
    timer_fallback_target_id: str | None = None
    alarm_fallback_target_id: str | None = None
    media_fallback_target_id: str | None = None

    spoken_response_volume: int | None = None
    timer_volume: int | None = None
    alarm_volume: int | None = None
    media_volume: int | None = None

    prefer_media_request_origin: bool = True
    default_media_provider: str | None = None
    default_media_account_id: str | None = None
    quiet_hours: QuietHours = field(default_factory=QuietHours)
    rules: tuple[RoutingRule, ...] = ()

    def __post_init__(self) -> None:
        _normalize_policy_targets(self)
        _normalize_policy_fallbacks(self)
        _normalize_policy_volumes(self)
        _validate_policy_media_account(self)
        _normalize_policy_collections(self)
        _validate_policy_fallback_targets(self)

    def target_for(self, output_kind: OutputKind) -> str | None:
        return {
            OutputKind.SPOKEN_RESPONSE: self.spoken_response_target_id,
            OutputKind.TIMER: self.timer_target_id,
            OutputKind.ALARM: self.alarm_target_id,
            OutputKind.MEDIA: self.media_target_id,
        }[OutputKind(output_kind)]

    def fallback_for(self, output_kind: OutputKind) -> tuple[FallbackMode, str | None]:
        kind = OutputKind(output_kind)
        return {
            OutputKind.SPOKEN_RESPONSE: (
                self.spoken_response_fallback,
                self.spoken_response_fallback_target_id,
            ),
            OutputKind.TIMER: (self.timer_fallback, self.timer_fallback_target_id),
            OutputKind.ALARM: (self.alarm_fallback, self.alarm_fallback_target_id),
            OutputKind.MEDIA: (self.media_fallback, self.media_fallback_target_id),
        }[kind]

    def volume_for(self, output_kind: OutputKind) -> int | None:
        return {
            OutputKind.SPOKEN_RESPONSE: self.spoken_response_volume,
            OutputKind.TIMER: self.timer_volume,
            OutputKind.ALARM: self.alarm_volume,
            OutputKind.MEDIA: self.media_volume,
        }[OutputKind(output_kind)]


@dataclass(frozen=True, slots=True)
class ResolvedRoute:
    """Privacy-safe routing decision, separate from action authority."""

    output_kind: OutputKind
    target_id: str | None
    reason: str
    target_volume: int | None = None
    fallback_mode: FallbackMode | None = None
    fallback_from: str | None = None
    rule_index: int | None = None
    suppressed: bool = False
    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_kind", OutputKind(self.output_kind))
        object.__setattr__(
            self,
            "target_id",
            _validate_target_id(self.target_id, field_name="resolved target_id"),
        )
        object.__setattr__(
            self,
            "target_volume",
            _validate_volume(
                self.target_volume,
                field_name="resolved target_volume",
            ),
        )
        if self.fallback_mode is not None:
            object.__setattr__(self, "fallback_mode", FallbackMode(self.fallback_mode))


__all__ = [
    "FallbackMode",
    "OutputKind",
    "QuietHours",
    "ResolvedRoute",
    "RoutingRule",
    "UserOutputPolicy",
]
