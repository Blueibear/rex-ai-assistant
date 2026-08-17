"""Authorization-aware, deterministic audio target resolution."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from types import MappingProxyType

from .models import AudioTarget, TargetKind, TargetResolution


def _normalize(value: str) -> str:
    """Normalize human-entered labels without introducing fuzzy matching."""
    return " ".join(value.casefold().split())


class AudioTargetRegistry:
    """Immutable target snapshot with explicit per-user authorization."""

    def __init__(
        self,
        targets: Iterable[AudioTarget],
        *,
        authorized_target_ids: Mapping[str, Collection[str]],
        origin_device_targets: Mapping[str, str] | None = None,
    ) -> None:
        targets_by_id: dict[str, AudioTarget] = {}
        for target in targets:
            if target.id in targets_by_id:
                raise ValueError(f"duplicate audio target id: {target.id}")
            targets_by_id[target.id] = target

        self._targets = tuple(sorted(targets_by_id.values(), key=lambda target: target.id))
        self._targets_by_id = MappingProxyType(targets_by_id)
        self._authorized_target_ids = MappingProxyType(
            {
                user_id: frozenset(target_ids)
                for user_id, target_ids in authorized_target_ids.items()
            }
        )
        self._origin_device_targets = MappingProxyType(dict(origin_device_targets or {}))

    @property
    def targets(self) -> tuple[AudioTarget, ...]:
        """Return the registry's immutable target snapshot."""
        return self._targets

    def resolve(
        self,
        query: str | None,
        *,
        user_id: str,
        origin_device_id: str | None = None,
    ) -> TargetResolution:
        """Resolve a query by the strict precedence defined for media routing."""
        normalized_query = _normalize(query) if query is not None else ""
        if not normalized_query:
            return self._resolve_origin(user_id, origin_device_id)

        exact_id = query.strip() if query is not None else ""
        target = self._targets_by_id.get(exact_id)
        if target is not None:
            return self._resolve_explicit_target(target, user_id)

        available = self._available_targets(user_id)

        named_targets = self._matching_targets(
            available,
            lambda candidate: candidate.kind is not TargetKind.GROUP
            and normalized_query
            in {
                _normalize(candidate.display_name),
                *(_normalize(alias) for alias in candidate.aliases),
            },
        )
        if named_targets:
            return self._unique_or_ambiguous(named_targets, "name_or_alias")

        room_targets = self._matching_targets(
            available,
            lambda candidate: candidate.room is not None
            and _normalize(candidate.room) == normalized_query,
        )
        if room_targets:
            return self._unique_or_ambiguous(room_targets, "room")

        group_targets = self._matching_targets(
            available,
            lambda candidate: candidate.kind is TargetKind.GROUP
            and normalized_query
            in {
                _normalize(candidate.display_name),
                *(_normalize(alias) for alias in candidate.aliases),
            },
        )
        if group_targets:
            return self._unique_or_ambiguous(group_targets, "persistent_group")

        return TargetResolution(target=None, reason="not_found")

    def _available_targets(self, user_id: str) -> tuple[AudioTarget, ...]:
        authorized_ids = self._authorized_target_ids.get(user_id, frozenset())
        return tuple(
            target for target in self._targets if target.id in authorized_ids and target.online
        )

    def _resolve_explicit_target(
        self,
        target: AudioTarget,
        user_id: str,
    ) -> TargetResolution:
        authorized_ids = self._authorized_target_ids.get(user_id, frozenset())
        if target.id not in authorized_ids:
            return TargetResolution(target=None, reason="not_authorized")
        if not target.online:
            return TargetResolution(target=None, reason="offline")
        return TargetResolution(target=target, reason="stable_id")

    def _resolve_origin(
        self,
        user_id: str,
        origin_device_id: str | None,
    ) -> TargetResolution:
        if origin_device_id is None:
            return TargetResolution(target=None, reason="target_required")

        target_id = self._origin_device_targets.get(origin_device_id)
        if target_id is None:
            return TargetResolution(target=None, reason="origin_not_mapped")

        target = self._targets_by_id.get(target_id)
        if target is None:
            return TargetResolution(target=None, reason="origin_not_found")

        authorized_ids = self._authorized_target_ids.get(user_id, frozenset())
        if target.id not in authorized_ids:
            return TargetResolution(target=None, reason="origin_not_authorized")
        if not target.online:
            return TargetResolution(target=None, reason="origin_offline")
        return TargetResolution(target=target, reason="origin_device")

    @staticmethod
    def _matching_targets(
        targets: Iterable[AudioTarget],
        predicate: Callable[[AudioTarget], bool],
    ) -> tuple[AudioTarget, ...]:
        return tuple(target for target in targets if predicate(target))

    @staticmethod
    def _unique_or_ambiguous(
        targets: tuple[AudioTarget, ...],
        reason: str,
    ) -> TargetResolution:
        if len(targets) == 1:
            return TargetResolution(target=targets[0], reason=reason)
        return TargetResolution(
            target=None,
            reason="ambiguous",
            ambiguous_ids=tuple(sorted(target.id for target in targets)),
        )


__all__ = ["AudioTargetRegistry"]
