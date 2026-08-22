"""Authorized, provenance-preserving situational context assembly."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeAlias

from rex.context.active import ActiveContextStore
from rex.context.source_policy import (
    AudienceScope,
    ContextSourcePolicyStore,
    ContextSourceType,
)
from rex.identity import validate_user_id

FactValue: TypeAlias = str | int | float | bool | None
CurrentInfoReader = Callable[[str, "SituationalSnapshot"], tuple["SituationalFact", ...]]


@dataclass(frozen=True, slots=True)
class SituationalFact:
    """One normalized fact plus source provenance and observation time."""

    key: str
    value: FactValue
    source_id: str
    observed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip() or len(self.key) > 128:
            raise ValueError("situational fact key is invalid")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("situational fact source_id is invalid")
        value = self.value
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("situational fact value must be a bounded scalar")
        if isinstance(value, str) and (
            len(value) > 512 or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("situational fact value must be a bounded scalar")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError("situational fact value must be a bounded scalar")
        if self.observed_at.tzinfo is None:
            raise ValueError("situational fact observed_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class SituationalSnapshot:
    """One user's bounded situational facts at a single assembly time."""

    user_id: str
    assembled_at: datetime
    facts: tuple[SituationalFact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", validate_user_id(self.user_id))
        if self.assembled_at.tzinfo is None:
            raise ValueError("situational snapshot assembled_at must be timezone-aware")
        keys = [fact.key for fact in self.facts]
        if len(keys) != len(set(keys)):
            raise ValueError("situational snapshot fact keys must be unique")

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(fact.source_id for fact in self.facts))

    def fact(self, key: str) -> SituationalFact | None:
        return next((fact for fact in self.facts if fact.key == key), None)

    def value(self, key: str, default: FactValue = None) -> FactValue:
        fact = self.fact(key)
        return default if fact is None else fact.value

    def freshness_seconds(self, key: str) -> float | None:
        fact = self.fact(key)
        if fact is None:
            return None
        age = (self.assembled_at - fact.observed_at).total_seconds()
        return max(0.0, age)

    def with_facts(self, facts: tuple[SituationalFact, ...]) -> SituationalSnapshot:
        return SituationalSnapshot(self.user_id, self.assembled_at, facts)

    def merged(self, facts: tuple[SituationalFact, ...]) -> SituationalSnapshot:
        by_key = {fact.key: fact for fact in self.facts}
        by_key.update({fact.key: fact for fact in facts})
        return SituationalSnapshot(self.user_id, self.assembled_at, tuple(by_key.values()))


class SituationalAssembler:
    """Build normalized context only from sources authorized for the current user."""

    def __init__(
        self,
        *,
        source_policy_store: ContextSourcePolicyStore,
        calendar_service: Any = None,
        knowledge_base: Any = None,
        active_context_store: ActiveContextStore | None = None,
        memory_reader: Callable[[str], Mapping[str, str]] | None = None,
        current_info_readers: Mapping[str, CurrentInfoReader] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._policy = source_policy_store
        self._calendar = calendar_service
        self._knowledge = knowledge_base
        self._active = active_context_store
        self._memory_reader = memory_reader
        self._current_info = dict(current_info_readers or {})
        self._clock = clock or (lambda: datetime.now(UTC))

    def _allows(self, source_id: str, source_type: ContextSourceType, user_id: str) -> bool:
        policy = self._policy.get(source_id, subject_user_id=user_id)
        if policy is None:
            policy = self._policy.register_source(
                source_id,
                source_type,
                owner_user_id=user_id,
                audience_scope=AudienceScope.PRIVATE,
            )
        return self._policy.is_context_eligible(
            source_id,
            subject_user_id=user_id,
            requester_user_id=user_id,
        )

    def _calendar_facts(self, user_id: str, now: datetime) -> tuple[SituationalFact, ...]:
        source_id = "integration:calendar"
        if self._calendar is None or not self._allows(
            source_id, ContextSourceType.INTEGRATION, user_id
        ):
            return ()
        events = self._calendar.get_upcoming_events(days=1, user_id=user_id)
        if not events:
            return ()
        event = events[0]
        minutes = max(0.0, (event.start_time - now).total_seconds() / 60.0)
        facts = [
            SituationalFact("calendar.next.title", str(event.title), source_id, now),
            SituationalFact("calendar.next.start_in_minutes", minutes, source_id, now),
        ]
        if getattr(event, "location", None):
            facts.append(
                SituationalFact("calendar.next.destination", str(event.location), source_id, now)
            )
        return tuple(facts)

    def _upload_facts(self, user_id: str) -> tuple[SituationalFact, ...]:
        if self._knowledge is None:
            return ()
        facts: list[SituationalFact] = []
        for doc in self._knowledge.list_documents_for_user(
            user_id,
            context_only=True,
            limit=5,
        ):
            if not self._policy.is_context_eligible(
                doc.source_id,
                subject_user_id=user_id,
                requester_user_id=user_id,
            ):
                continue
            facts.append(
                SituationalFact(
                    key=f"upload.{doc.doc_id}.title",
                    value=str(doc.title)[:256],
                    source_id=doc.source_id,
                    observed_at=doc.created_at,
                )
            )
        return tuple(facts)

    def _memory_facts(self, user_id: str, now: datetime) -> tuple[SituationalFact, ...]:
        source_id = f"memory:{user_id}"
        if self._memory_reader is None:
            return ()
        if not self._allows(source_id, ContextSourceType.MEMORY, user_id):
            return ()
        raw = self._memory_reader(user_id)
        facts: list[SituationalFact] = []
        for key, value in list(raw.items())[:10]:
            facts.append(
                SituationalFact(
                    key=f"memory.{str(key)[:64]}",
                    value=str(value)[:256],
                    source_id=source_id,
                    observed_at=now,
                )
            )
        return tuple(facts)

    def _active_facts(self, user_id: str, now: datetime) -> tuple[SituationalFact, ...]:
        source_id = "capability:active_context"
        if self._active is None or not self._allows(
            source_id, ContextSourceType.CAPABILITY, user_id
        ):
            return ()
        facts: list[SituationalFact] = []
        for ref in self._active.list_for_user(user_id)[:8]:
            facts.append(
                SituationalFact(
                    key=f"active.{ref.domain}.{ref.key}",
                    value=str(ref.payload.get("status") or ref.key)[:256],
                    source_id=source_id,
                    observed_at=now,
                )
            )
        return tuple(facts)

    def build(self, *, user_id: str) -> SituationalSnapshot:
        owner = validate_user_id(user_id)
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("situational assembler clock must be timezone-aware")
        facts = (
            self._calendar_facts(owner, now)
            + self._upload_facts(owner)
            + self._memory_facts(owner, now)
            + self._active_facts(owner, now)
        )
        return SituationalSnapshot(owner, now, facts)

    def enrich_current_info(
        self,
        snapshot: SituationalSnapshot,
        *,
        required: tuple[str, ...],
    ) -> SituationalSnapshot:
        owner = validate_user_id(snapshot.user_id)
        enriched = snapshot
        for name in dict.fromkeys(required):
            reader = self._current_info.get(name)
            if reader is None:
                continue
            source_id = f"integration:{name}"
            if not self._allows(source_id, ContextSourceType.INTEGRATION, owner):
                continue
            facts = tuple(reader(owner, enriched))
            authorized = tuple(
                fact
                for fact in facts
                if fact.source_id == source_id
                and self._policy.is_context_eligible(
                    source_id,
                    subject_user_id=owner,
                    requester_user_id=owner,
                )
            )
            enriched = enriched.merged(authorized)
        return enriched


__all__ = [
    "CurrentInfoReader",
    "FactValue",
    "SituationalAssembler",
    "SituationalFact",
    "SituationalSnapshot",
]
