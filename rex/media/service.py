"""Canonical media orchestration with fail-closed authorization and verification."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from rex.identity import validate_user_id

from .accounts import MediaAccountRef, MediaAccountStore
from .models import (
    AudioTarget,
    MediaAction,
    MediaCapability,
    MediaMutationOutcome,
    MediaMutationResult,
    MediaState,
    MediaStateSnapshot,
    TargetProviderAdapter,
)
from .parser import MediaCommand, MediaCommandAction
from .registry import AudioTargetRegistry
from .sessions import ActiveMediaSession, ActiveMediaSessionStore


@dataclass(frozen=True, slots=True)
class MediaServiceResult:
    outcome: str
    requested_target_id: str | None = None
    mutation: MediaMutationResult | None = None
    state: MediaStateSnapshot | None = None
    message: str | None = None
    ambiguous_ids: tuple[str, ...] = ()
    verification_expected: Mapping[str, object] | None = None


_ACTION_MAP: dict[MediaCommandAction, MediaAction] = {
    MediaCommandAction.PLAY: MediaAction.PLAY,
    MediaCommandAction.PAUSE: MediaAction.PAUSE,
    MediaCommandAction.RESUME: MediaAction.RESUME,
    MediaCommandAction.STOP: MediaAction.STOP,
    MediaCommandAction.NEXT: MediaAction.NEXT,
    MediaCommandAction.PREVIOUS: MediaAction.PREVIOUS,
    MediaCommandAction.SET_VOLUME: MediaAction.SET_VOLUME,
    MediaCommandAction.MUTE: MediaAction.MUTE,
    MediaCommandAction.UNMUTE: MediaAction.UNMUTE,
}
_CAPABILITY_MAP: dict[MediaAction, MediaCapability] = {
    MediaAction.PLAY: MediaCapability.PLAY,
    MediaAction.PAUSE: MediaCapability.PAUSE,
    MediaAction.RESUME: MediaCapability.RESUME,
    MediaAction.STOP: MediaCapability.STOP,
    MediaAction.NEXT: MediaCapability.NEXT,
    MediaAction.PREVIOUS: MediaCapability.PREVIOUS,
    MediaAction.SET_VOLUME: MediaCapability.SET_VOLUME,
    MediaAction.MUTE: MediaCapability.MUTE,
    MediaAction.UNMUTE: MediaCapability.MUTE,
}
_UNSUPPORTED_ACTIONS = {MediaCommandAction.TRANSFER}


class MediaService:
    def __init__(
        self,
        *,
        registry: AudioTargetRegistry,
        adapters: Mapping[str, TargetProviderAdapter],
        account_store: MediaAccountStore | None = None,
        session_store: ActiveMediaSessionStore | None = None,
        registry_refresher: Callable[[], AudioTargetRegistry] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._adapters = dict(adapters)
        self._account_store = account_store or MediaAccountStore()
        self._session_store = session_store or ActiveMediaSessionStore(clock=clock)
        self._registry_refresher = registry_refresher
        self._clock = clock

    def execute(
        self,
        command: MediaCommand,
        *,
        user_id: str,
        origin_device_id: str | None = None,
        account_ref: MediaAccountRef | None = None,
    ) -> MediaServiceResult:
        user_id = validate_user_id(user_id)
        if not isinstance(command, MediaCommand):
            raise TypeError("command must be a MediaCommand")

        self._refresh_registry()
        query = command.target_text
        if query is None and origin_device_id is None:
            session = self._session_store.get(user_id, now=self._clock())
            if session is not None:
                query = session.target_id
        resolution = self._registry.resolve(
            query, user_id=user_id, origin_device_id=origin_device_id
        )
        if resolution.target is None:
            return self._resolution_failure(resolution.reason, resolution.ambiguous_ids)
        target = resolution.target

        if command.action is MediaCommandAction.QUERY_STATE:
            adapter = self._adapters.get(target.provider)
            if adapter is None:
                return MediaServiceResult("unsupported", requested_target_id=target.id)
            try:
                state = adapter.get_state(target)
            except Exception as exc:
                return MediaServiceResult("failed", target.id, message=str(exc))
            return MediaServiceResult("read", target.id, state=state)

        if command.action in _UNSUPPORTED_ACTIONS:
            return MediaServiceResult("unsupported", requested_target_id=target.id)

        command_action = MediaCommandAction(command.action)
        action = _ACTION_MAP.get(command_action)
        if action is None:
            return MediaServiceResult("unsupported", requested_target_id=target.id)
        if _CAPABILITY_MAP[action] not in target.capabilities:
            return MediaServiceResult("unsupported", requested_target_id=target.id)

        account_error = self._validate_account_selection(user_id, target, account_ref)
        if account_error is not None:
            return MediaServiceResult(account_error, requested_target_id=target.id)

        adapter = self._adapters.get(target.provider)
        if adapter is None:
            return MediaServiceResult("unsupported", requested_target_id=target.id)
        value: str | int | float | None = None
        if action is MediaAction.PLAY:
            if command.query is None:
                return MediaServiceResult("failed", target.id, message="media query required")
            value = command.query
        elif action is MediaAction.SET_VOLUME:
            if command.level is None:
                return MediaServiceResult("failed", target.id, message="volume level required")
            value = command.level

        try:
            acknowledgement = adapter.execute_action(target, action, value=value)
        except Exception as exc:
            return MediaServiceResult("failed", target.id, message=str(exc))
        if not acknowledgement.accepted:
            return MediaServiceResult(
                "failed", target.id, message=acknowledgement.detail or "provider rejected"
            )

        expected = self._expected_state(action, value)
        observed = self._read_state(adapter, target)
        verified = (
            observed is not None
            and observed.target_id == target.id
            and bool(expected)
            and self._matches(observed, expected)
        )
        mutation_outcome = (
            MediaMutationOutcome.VERIFIED if verified else MediaMutationOutcome.ATTEMPTED_UNVERIFIED
        )
        mutation = MediaMutationResult(
            target_id=target.id,
            action=action,
            outcome=mutation_outcome,
            acknowledgement=acknowledgement,
            requested_value=value,
            observed_state=observed,
            verification_evidence=tuple(sorted(expected)),
        )
        outcome = mutation_outcome.value
        if verified:
            self._update_session(user_id, target, command, observed)
        return MediaServiceResult(
            outcome,
            requested_target_id=target.id,
            mutation=mutation,
            state=observed,
            verification_expected=expected,
        )

    def reverify(
        self,
        *,
        target_id: str,
        provider: str,
        user_id: str,
        expected: Mapping[str, object],
    ) -> bool:
        try:
            user_id = validate_user_id(user_id)
            resolution = self._registry.resolve(target_id, user_id=user_id)
        except (TypeError, ValueError):
            return False
        target = resolution.target
        if target is None or target.provider != provider or not expected:
            return False
        adapter = self._adapters.get(provider)
        if adapter is None:
            return False
        observed = self._read_state(adapter, target)
        return (
            observed is not None
            and observed.target_id == target.id
            and self._matches(observed, expected)
        )

    def _validate_account_selection(
        self,
        user_id: str,
        target: AudioTarget,
        account_ref: MediaAccountRef | None,
    ) -> str | None:
        if account_ref is not None:
            if not isinstance(account_ref, MediaAccountRef):
                return "account_not_authorized"
            if account_ref.user_id != user_id or account_ref.provider != target.provider:
                return "account_not_authorized"
            stored = self._account_store.get(user_id, account_ref.provider, account_ref.account_id)
            if stored != account_ref:
                return "account_not_authorized"
            return None
        accounts = tuple(
            account
            for account in self._account_store.list(user_id)
            if account.provider == target.provider
        )
        if len(accounts) > 1:
            return "account_ambiguous"
        return None

    def _refresh_registry(self) -> None:
        """Refresh dynamic target discovery before resolving a media command."""
        if self._registry_refresher is not None:
            self._registry = self._registry_refresher()

    @staticmethod
    def _resolution_failure(reason: str, ambiguous_ids: tuple[str, ...]) -> MediaServiceResult:
        mapped = {
            "not_authorized": "not_authorized",
            "origin_not_authorized": "not_authorized",
            "offline": "offline",
            "origin_offline": "offline",
            "ambiguous": "ambiguous",
            "target_required": "target_required",
        }.get(reason, "not_found")
        return MediaServiceResult(mapped, ambiguous_ids=ambiguous_ids)

    @staticmethod
    def _read_state(
        adapter: TargetProviderAdapter, target: AudioTarget
    ) -> MediaStateSnapshot | None:
        try:
            return adapter.get_state(target)
        except Exception:
            return None

    @staticmethod
    def _expected_state(action: MediaAction, value: str | int | float | None) -> dict[str, object]:
        if action is MediaAction.PLAY:
            expected: dict[str, object] = {"playback": [MediaState.PLAYING.value]}
            if isinstance(value, str) and value:
                expected["current_item"] = value
            return expected
        if action is MediaAction.RESUME:
            return {"playback": [MediaState.PLAYING.value]}
        if action is MediaAction.PAUSE:
            return {"playback": [MediaState.PAUSED.value]}
        if action is MediaAction.STOP:
            return {"playback": [MediaState.STOPPED.value]}
        if action is MediaAction.SET_VOLUME and isinstance(value, (int, float)):
            return {"volume_percent": float(value)}
        if action is MediaAction.MUTE:
            return {"muted": True}
        if action is MediaAction.UNMUTE:
            expected = {"muted": False}
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                expected["volume_percent"] = float(value)
            return expected
        return {}

    @staticmethod
    def _matches(snapshot: MediaStateSnapshot, expected: Mapping[str, object]) -> bool:
        for field, wanted in expected.items():
            if field == "playback":
                values = wanted if isinstance(wanted, (list, tuple, set, frozenset)) else [wanted]
                if snapshot.playback.value not in {str(value) for value in values}:
                    return False
            elif field == "current_item":
                actual = {
                    item
                    for item in (snapshot.current_item_id, snapshot.current_item_title)
                    if item is not None
                }
                if str(wanted) not in actual:
                    return False
            elif field == "muted":
                if snapshot.muted is not wanted:
                    return False
            elif field == "volume_percent":
                if snapshot.volume_percent is None:
                    return False
                if isinstance(wanted, bool) or not isinstance(wanted, (int, float)):
                    return False
                if abs(snapshot.volume_percent - float(wanted)) > 0.5:
                    return False
            else:
                return False
        return bool(expected)

    def _update_session(
        self,
        user_id: str,
        target: AudioTarget,
        command: MediaCommand,
        observed: MediaStateSnapshot | None,
    ) -> None:
        media_ref: str | None = None
        if observed is not None:
            media_ref = observed.current_item_id
        if media_ref is None and command.action is MediaCommandAction.PLAY and command.query:
            digest = hashlib.sha256(command.query.encode("utf-8")).hexdigest()[:24]
            media_ref = f"query:{digest}"
        if media_ref is None:
            existing = self._session_store.get(user_id, now=self._clock())
            if existing is not None and existing.target_id == target.id:
                media_ref = existing.media_ref
        if media_ref is None:
            return
        try:
            self._session_store.set(
                ActiveMediaSession(
                    user_id=user_id,
                    target_id=target.id,
                    provider=target.provider,
                    media_ref=media_ref,
                    updated_at=self._clock(),
                )
            )
        except ValueError:
            return


__all__ = ["MediaService", "MediaServiceResult"]
