from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rex.context.source_policy import (
    AudienceScope,
    ContextSourcePolicyStore,
    ContextSourceType,
)


def test_disabled_calendar_policy_prevents_source_read(tmp_path):
    from rex.context.situational import SituationalAssembler

    policy = ContextSourcePolicyStore(tmp_path / "policy")
    policy.register_source(
        "integration:calendar",
        ContextSourceType.INTEGRATION,
        owner_user_id="james",
        audience_scope=AudienceScope.PRIVATE,
        context_enabled=False,
    )

    class ForbiddenCalendar:
        def get_upcoming_events(self, days=7, *, user_id=None):
            raise AssertionError("disabled source must be filtered before retrieval")

    assembler = SituationalAssembler(
        source_policy_store=policy,
        calendar_service=ForbiddenCalendar(),
    )
    snapshot = assembler.build(user_id="james")
    assert "integration:calendar" not in snapshot.source_ids


def test_private_upload_for_other_user_never_enters_snapshot(tmp_path):
    from rex.context.situational import SituationalAssembler
    from rex.knowledge_base import KnowledgeBase

    policy = ContextSourcePolicyStore(tmp_path / "policy")
    kb = KnowledgeBase(
        tmp_path / "docs.json",
        tmp_path / "index.json",
        source_policy_store=policy,
    )
    james_doc = kb.ingest_text(
        "James-only travel plan",
        "Private Trip",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )
    kb.ingest_text(
        "Cole's notes",
        "Cole Notes",
        owner_user_id="cole",
        audience_scope="private",
        context_enabled=True,
    )

    snapshot = SituationalAssembler(
        source_policy_store=policy,
        knowledge_base=kb,
    ).build(user_id="cole")

    assert james_doc.source_id not in snapshot.source_ids
    assert all(fact.source_id != james_doc.source_id for fact in snapshot.facts)


def test_calendar_reader_is_user_bound_and_preserves_provenance(tmp_path):
    from rex.calendar_service import CalendarEvent
    from rex.context.situational import SituationalAssembler

    seen: list[str] = []
    now = datetime.now(UTC)

    class Calendar:
        def get_upcoming_events(self, days=7, *, user_id=None):
            seen.append(user_id)
            return [
                CalendarEvent(
                    event_id="evt-work",
                    title="Work",
                    start_time=now + timedelta(minutes=45),
                    end_time=now + timedelta(hours=2),
                    location="Office",
                )
            ]

    snapshot = SituationalAssembler(
        source_policy_store=ContextSourcePolicyStore(tmp_path / "policy"),
        calendar_service=Calendar(),
        clock=lambda: now,
    ).build(user_id="james")

    assert seen == ["james"]
    assert snapshot.value("calendar.next.destination") == "Office"
    fact = snapshot.fact("calendar.next.destination")
    assert fact is not None and fact.source_id == "integration:calendar"


def test_current_info_readers_are_lazy_and_explicit(tmp_path):
    from rex.context.situational import SituationalAssembler, SituationalFact

    calls: list[tuple[str, str]] = []
    now = datetime.now(UTC)

    def weather(user_id: str, snapshot):
        calls.append(("weather", user_id))
        return (
            SituationalFact(
                key="weather.storm_probability",
                value=0.8,
                source_id="integration:weather",
                observed_at=now,
            ),
        )

    assembler = SituationalAssembler(
        source_policy_store=ContextSourcePolicyStore(tmp_path / "policy"),
        current_info_readers={"weather": weather},
        clock=lambda: now,
    )
    base = assembler.build(user_id="james")
    assert calls == []

    enriched = assembler.enrich_current_info(base, required=("weather",))
    assert calls == [("weather", "james")]
    assert enriched.value("weather.storm_probability") == 0.8


def test_situational_fact_rejects_nested_values():
    import pytest

    from rex.context.situational import SituationalFact

    with pytest.raises(ValueError, match="scalar"):
        SituationalFact(
            key="weather.raw",
            value={"nested": "not allowed"},  # type: ignore[arg-type]
            source_id="integration:weather",
            observed_at=datetime.now(UTC),
        )
