"""Per-user output-routing policy persistence and deterministic resolution."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, time
from pathlib import Path
from typing import Any

from rex.identity import validate_user_id
from rex.media.registry import AudioTargetRegistry
from rex.runtime_paths import user_data_path

from .models import (
    FallbackMode,
    OutputKind,
    QuietHours,
    ResolvedRoute,
    RoutingRule,
    UserOutputPolicy,
)

_SCHEMA_VERSION = 1
_POLICY_LOCKS_GUARD = threading.Lock()
_POLICY_LOCKS: dict[Path, threading.RLock] = {}


def _path_lock(path: Path) -> threading.RLock:
    resolved = path.resolve(strict=False)
    with _POLICY_LOCKS_GUARD:
        lock = _POLICY_LOCKS.get(resolved)
        if lock is None:
            lock = threading.RLock()
            _POLICY_LOCKS[resolved] = lock
        return lock


def _parse_time(value: Any, *, field_name: str) -> time | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an HH:MM[:SS] string or null")
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid local time") from exc


def _quiet_hours_to_dict(value: QuietHours) -> dict[str, Any]:
    return {
        "enabled": value.enabled,
        "start_local_time": value.start_local_time.isoformat(timespec="minutes"),
        "end_local_time": value.end_local_time.isoformat(timespec="minutes"),
        "days_of_week": list(value.days_of_week),
    }


def _rule_to_dict(rule: RoutingRule) -> dict[str, Any]:
    return {
        "output_kind": rule.output_kind.value,
        "target_id": rule.target_id,
        "days_of_week": list(rule.days_of_week),
        "start_local_time": (
            rule.start_local_time.isoformat(timespec="minutes")
            if rule.start_local_time is not None
            else None
        ),
        "end_local_time": (
            rule.end_local_time.isoformat(timespec="minutes")
            if rule.end_local_time is not None
            else None
        ),
        "target_volume": rule.target_volume,
        "fallback_mode": (
            rule.fallback_mode.value if rule.fallback_mode is not None else None
        ),
        "fallback_target_id": rule.fallback_target_id,
    }


def _policy_to_dict(policy: UserOutputPolicy) -> dict[str, Any]:
    return {
        "spoken_response_target_id": policy.spoken_response_target_id,
        "timer_target_id": policy.timer_target_id,
        "alarm_target_id": policy.alarm_target_id,
        "media_target_id": policy.media_target_id,
        "spoken_response_fallback": policy.spoken_response_fallback.value,
        "timer_fallback": policy.timer_fallback.value,
        "alarm_fallback": policy.alarm_fallback.value,
        "media_fallback": policy.media_fallback.value,
        "spoken_response_fallback_target_id": policy.spoken_response_fallback_target_id,
        "timer_fallback_target_id": policy.timer_fallback_target_id,
        "alarm_fallback_target_id": policy.alarm_fallback_target_id,
        "media_fallback_target_id": policy.media_fallback_target_id,
        "spoken_response_volume": policy.spoken_response_volume,
        "timer_volume": policy.timer_volume,
        "alarm_volume": policy.alarm_volume,
        "media_volume": policy.media_volume,
        "prefer_media_request_origin": policy.prefer_media_request_origin,
        "default_media_provider": policy.default_media_provider,
        "default_media_account_id": policy.default_media_account_id,
        "quiet_hours": _quiet_hours_to_dict(policy.quiet_hours),
        "rules": [_rule_to_dict(rule) for rule in policy.rules],
    }


def _policy_from_dict(payload: Any) -> UserOutputPolicy:
    if not isinstance(payload, dict):
        raise ValueError("Output-routing policy must be an object")
    quiet_raw = payload.get("quiet_hours", {})
    if not isinstance(quiet_raw, dict):
        raise ValueError("quiet_hours must be an object")
    quiet = QuietHours(
        enabled=quiet_raw.get("enabled", False),
        start_local_time=_parse_time(
            quiet_raw.get("start_local_time", "22:00"),
            field_name="quiet_hours.start_local_time",
        )
        or time(22, 0),
        end_local_time=_parse_time(
            quiet_raw.get("end_local_time", "07:00"),
            field_name="quiet_hours.end_local_time",
        )
        or time(7, 0),
        days_of_week=tuple(quiet_raw.get("days_of_week", ())),
    )

    rules_raw = payload.get("rules", ())
    if not isinstance(rules_raw, list):
        raise ValueError("rules must be an array")
    rules: list[RoutingRule] = []
    for index, raw in enumerate(rules_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"rules[{index}] must be an object")
        rules.append(
            RoutingRule(
                output_kind=OutputKind(raw["output_kind"]),
                target_id=raw["target_id"],
                days_of_week=tuple(raw.get("days_of_week", ())),
                start_local_time=_parse_time(
                    raw.get("start_local_time"),
                    field_name=f"rules[{index}].start_local_time",
                ),
                end_local_time=_parse_time(
                    raw.get("end_local_time"),
                    field_name=f"rules[{index}].end_local_time",
                ),
                target_volume=raw.get("target_volume"),
                fallback_mode=(
                    FallbackMode(raw["fallback_mode"])
                    if raw.get("fallback_mode") is not None
                    else None
                ),
                fallback_target_id=raw.get("fallback_target_id"),
            )
        )

    known = {
        "spoken_response_target_id",
        "timer_target_id",
        "alarm_target_id",
        "media_target_id",
        "spoken_response_fallback",
        "timer_fallback",
        "alarm_fallback",
        "media_fallback",
        "spoken_response_fallback_target_id",
        "timer_fallback_target_id",
        "alarm_fallback_target_id",
        "media_fallback_target_id",
        "spoken_response_volume",
        "timer_volume",
        "alarm_volume",
        "media_volume",
        "prefer_media_request_origin",
        "default_media_provider",
        "default_media_account_id",
        "quiet_hours",
        "rules",
    }
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            f"Output-routing policy contains unknown fields: {sorted(unknown)!r}"
        )

    return UserOutputPolicy(
        spoken_response_target_id=payload.get("spoken_response_target_id"),
        timer_target_id=payload.get("timer_target_id"),
        alarm_target_id=payload.get("alarm_target_id"),
        media_target_id=payload.get("media_target_id"),
        spoken_response_fallback=FallbackMode(
            payload.get("spoken_response_fallback", FallbackMode.NONE.value)
        ),
        timer_fallback=FallbackMode(
            payload.get("timer_fallback", FallbackMode.NONE.value)
        ),
        alarm_fallback=FallbackMode(
            payload.get("alarm_fallback", FallbackMode.NONE.value)
        ),
        media_fallback=FallbackMode(
            payload.get("media_fallback", FallbackMode.NONE.value)
        ),
        spoken_response_fallback_target_id=payload.get(
            "spoken_response_fallback_target_id"
        ),
        timer_fallback_target_id=payload.get("timer_fallback_target_id"),
        alarm_fallback_target_id=payload.get("alarm_fallback_target_id"),
        media_fallback_target_id=payload.get("media_fallback_target_id"),
        spoken_response_volume=payload.get("spoken_response_volume"),
        timer_volume=payload.get("timer_volume"),
        alarm_volume=payload.get("alarm_volume"),
        media_volume=payload.get("media_volume"),
        prefer_media_request_origin=payload.get("prefer_media_request_origin", True),
        default_media_provider=payload.get("default_media_provider"),
        default_media_account_id=payload.get("default_media_account_id"),
        quiet_hours=quiet,
        rules=tuple(rules),
    )


class OutputRoutingService:
    """Persist and resolve user routing preferences without granting authority."""

    def __init__(
        self,
        registry: AudioTargetRegistry,
        *,
        root: Path | str | None = None,
    ) -> None:
        self._registry = registry
        self._root = Path(root) if root is not None else None

    def _policy_path(self, user_id: str) -> Path:
        user_id = validate_user_id(user_id)
        if self._root is None:
            return user_data_path(user_id, "output_routing", "policy.json")
        return self._root / "users" / user_id / "output_routing" / "policy.json"

    def get_policy(self, user_id: str) -> UserOutputPolicy:
        """Return the user's policy or safe defaults when no file exists."""
        user_id = validate_user_id(user_id)
        path = self._policy_path(user_id)
        with _path_lock(path):
            if not path.exists():
                return UserOutputPolicy()
            try:
                payload: Any = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Output-routing policy is unreadable: {exc}") from exc
            if (
                not isinstance(payload, dict)
                or set(payload) != {"version", "user_id", "policy"}
                or payload.get("version") != _SCHEMA_VERSION
                or payload.get("user_id") != user_id
            ):
                raise ValueError("Output-routing policy has invalid ownership or schema")
            return _policy_from_dict(payload["policy"])

    def save_policy(
        self,
        user_id: str,
        policy: UserOutputPolicy,
    ) -> UserOutputPolicy:
        """Atomically replace one user's routing policy."""
        user_id = validate_user_id(user_id)
        if not isinstance(policy, UserOutputPolicy):
            raise TypeError("policy must be UserOutputPolicy")
        path = self._policy_path(user_id)
        payload = {
            "version": _SCHEMA_VERSION,
            "user_id": user_id,
            "policy": _policy_to_dict(policy),
        }
        with _path_lock(path):
            self._atomic_write_json(path, payload)
        return policy

    def resolve(
        self,
        *,
        user_id: str,
        output_kind: OutputKind,
        explicit_target_text: str | None,
        origin_device_id: str | None,
        at: datetime,
    ) -> ResolvedRoute:
        """Resolve target precedence against current registry authorization/health."""
        user_id = validate_user_id(user_id)
        kind = OutputKind(output_kind)
        if not isinstance(at, datetime):
            raise TypeError("at must be a datetime")
        policy = self.get_policy(user_id)

        explicit_text = explicit_target_text.strip() if explicit_target_text else None
        if explicit_text:
            resolution = self._registry.resolve(
                explicit_text,
                user_id=user_id,
                origin_device_id=origin_device_id,
            )
            if resolution.target is not None:
                return ResolvedRoute(
                    output_kind=kind,
                    target_id=resolution.target.id,
                    reason="explicit_target",
                    target_volume=policy.volume_for(kind),
                )
            return self._fallback(
                user_id=user_id,
                kind=kind,
                policy=policy,
                preferred_reason=f"explicit_target_{resolution.reason}",
                preferred_target=explicit_text,
                target_volume=policy.volume_for(kind),
            )

        if self._quiet_hours_suppress(policy, kind, at):
            return ResolvedRoute(
                output_kind=kind,
                target_id=None,
                reason="quiet_hours",
                target_volume=None,
                suppressed=True,
            )

        if kind is OutputKind.MEDIA and policy.prefer_media_request_origin:
            origin = self._registry.resolve(
                None,
                user_id=user_id,
                origin_device_id=origin_device_id,
            )
            if origin.target is not None:
                return ResolvedRoute(
                    output_kind=kind,
                    target_id=origin.target.id,
                    reason="request_origin",
                    target_volume=policy.volume_for(kind),
                )

        for index, rule in enumerate(policy.rules):
            if not rule.matches(kind, at):
                continue
            resolution = self._registry.resolve(rule.target_id, user_id=user_id)
            if resolution.target is not None:
                return ResolvedRoute(
                    output_kind=kind,
                    target_id=resolution.target.id,
                    reason="conditional_rule",
                    target_volume=(
                        rule.target_volume
                        if rule.target_volume is not None
                        else policy.volume_for(kind)
                    ),
                    rule_index=index,
                )
            return self._fallback(
                user_id=user_id,
                kind=kind,
                policy=policy,
                preferred_reason=f"rule_target_{resolution.reason}",
                preferred_target=rule.target_id,
                target_volume=(
                    rule.target_volume
                    if rule.target_volume is not None
                    else policy.volume_for(kind)
                ),
                override_mode=rule.fallback_mode,
                override_target=rule.fallback_target_id,
                rule_index=index,
            )

        configured_target = policy.target_for(kind)
        if configured_target is not None:
            resolution = self._registry.resolve(configured_target, user_id=user_id)
            if resolution.target is not None:
                return ResolvedRoute(
                    output_kind=kind,
                    target_id=resolution.target.id,
                    reason="configured_default",
                    target_volume=policy.volume_for(kind),
                )
            return self._fallback(
                user_id=user_id,
                kind=kind,
                policy=policy,
                preferred_reason="configured_target_unavailable",
                preferred_target=configured_target,
                target_volume=policy.volume_for(kind),
            )

        return self._fallback(
            user_id=user_id,
            kind=kind,
            policy=policy,
            preferred_reason="target_required",
            preferred_target=None,
            target_volume=policy.volume_for(kind),
        )

    @staticmethod
    def _quiet_hours_suppress(
        policy: UserOutputPolicy,
        kind: OutputKind,
        at: datetime,
    ) -> bool:
        # Timer/alarm delivery is explicitly requested timekeeping output and must
        # not disappear merely because quiet hours are active. Task 3 may add
        # event-specific safety/override metadata; this layer already fails safe
        # by suppressing only optional spoken/media output.
        return kind in {
            OutputKind.SPOKEN_RESPONSE,
            OutputKind.MEDIA,
        } and policy.quiet_hours.active_at(at)

    def _fallback(
        self,
        *,
        user_id: str,
        kind: OutputKind,
        policy: UserOutputPolicy,
        preferred_reason: str,
        preferred_target: str | None,
        target_volume: int | None,
        override_mode: FallbackMode | None = None,
        override_target: str | None = None,
        rule_index: int | None = None,
    ) -> ResolvedRoute:
        policy_mode, policy_target = policy.fallback_for(kind)
        mode = override_mode if override_mode is not None else policy_mode
        fallback_target = override_target if override_mode is not None else policy_target

        if mode is FallbackMode.NONE:
            return ResolvedRoute(
                output_kind=kind,
                target_id=None,
                reason=preferred_reason,
                target_volume=None,
                fallback_mode=mode,
                fallback_from=preferred_target,
                rule_index=rule_index,
            )
        if mode is FallbackMode.ASK:
            return ResolvedRoute(
                output_kind=kind,
                target_id=None,
                reason="fallback_confirmation_required",
                target_volume=None,
                fallback_mode=mode,
                fallback_from=preferred_target,
                rule_index=rule_index,
                requires_confirmation=True,
            )
        if fallback_target is None:
            return ResolvedRoute(
                output_kind=kind,
                target_id=None,
                reason="fallback_target_missing",
                fallback_mode=mode,
                fallback_from=preferred_target,
                rule_index=rule_index,
            )

        resolution = self._registry.resolve(fallback_target, user_id=user_id)
        if resolution.target is None:
            return ResolvedRoute(
                output_kind=kind,
                target_id=None,
                reason=f"fallback_target_{resolution.reason}",
                fallback_mode=mode,
                fallback_from=preferred_target,
                rule_index=rule_index,
            )
        return ResolvedRoute(
            output_kind=kind,
            target_id=resolution.target.id,
            reason="named_fallback",
            target_volume=target_volume,
            fallback_mode=mode,
            fallback_from=preferred_target,
            rule_index=rule_index,
        )

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


__all__ = ["OutputRoutingService"]
