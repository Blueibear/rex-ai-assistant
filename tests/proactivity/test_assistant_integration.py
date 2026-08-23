from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rex.assistant import Assistant
from rex.calendar_service import CalendarEvent
from rex.context.situational import SituationalAssembler, SituationalFact
from rex.context.source_policy import ContextSourcePolicyStore
from rex.proactivity.evaluator import ProactiveOpportunityEvaluator
from rex.suggestions.engine import SuggestionEngine


def test_assistant_builds_and_queues_contextual_candidate_lazily(tmp_path):
    now = datetime(2026, 8, 22, 18, 0, tzinfo=UTC)
    current_calls: list[str] = []

    class Calendar:
        def get_upcoming_events(self, days=7, *, user_id=None):
            return [
                CalendarEvent(
                    event_id="work",
                    title="Work",
                    start_time=now + timedelta(minutes=50),
                    end_time=now + timedelta(hours=2),
                    location="work",
                )
            ]

    def traffic(user_id, snapshot):
        current_calls.append("traffic")
        return (SituationalFact("traffic.delay_minutes", 18.0, "integration:traffic", now),)

    def weather(user_id, snapshot):
        current_calls.append("weather")
        return (SituationalFact("weather.storm_probability", 0.8, "integration:weather", now),)

    policy = ContextSourcePolicyStore(tmp_path / "policy")
    assembler = SituationalAssembler(
        source_policy_store=policy,
        calendar_service=Calendar(),
        current_info_readers={"traffic": traffic, "weather": weather},
        clock=lambda: now,
    )
    engine = SuggestionEngine(
        dismissed_path=tmp_path / "dismissed.json",
        automations_path=tmp_path / "automations.json",
    )
    assistant = Assistant.__new__(Assistant)
    assistant._situational_assembler = assembler
    assistant._proactive_evaluator = ProactiveOpportunityEvaluator()
    assistant._suggestion_engine = engine

    assistant._prepare_contextual_suggestion("james", response_text="Done.")

    assert current_calls == ["traffic", "weather"]
    pending = engine.pending_contextual_text("james")
    assert pending is not None and "leave" in pending.lower()


def test_assistant_does_not_queue_contextual_candidate_while_asking_question(tmp_path):
    assistant = Assistant.__new__(Assistant)
    engine = SuggestionEngine(
        dismissed_path=tmp_path / "dismissed.json",
        automations_path=tmp_path / "automations.json",
    )
    assistant._suggestion_engine = engine
    assistant._situational_assembler = object()
    assistant._proactive_evaluator = object()

    assistant._prepare_contextual_suggestion("james", response_text="Which timer did you mean?")

    assert not engine.has_pending("james")


def test_disabled_proactive_preference_prevents_context_reads():
    from types import SimpleNamespace

    assistant = Assistant.__new__(Assistant)
    assistant._suggestion_engine = SuggestionEngine()
    assistant._context_privacy_service = SimpleNamespace(
        preference_store=SimpleNamespace(
            get=lambda user_id: SimpleNamespace(proactive_assistance=False)
        )
    )

    reads: list[str] = []

    class ForbiddenAssembler:
        def build(self, *, user_id):
            reads.append(user_id)
            raise AssertionError("disabled proactivity must short-circuit before context reads")

    assistant._situational_assembler = ForbiddenAssembler()
    assistant._proactive_evaluator = ProactiveOpportunityEvaluator()

    assistant._prepare_contextual_suggestion("james", response_text="Done.")

    assert reads == []
    assert not assistant._suggestion_engine.has_pending("james")
