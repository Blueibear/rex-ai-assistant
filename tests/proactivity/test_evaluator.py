from __future__ import annotations

from datetime import UTC, datetime, timedelta


def _commute_snapshot(*, weather_age_seconds: float = 60.0, traffic_age_seconds: float = 45.0):
    from rex.context.situational import SituationalFact, SituationalSnapshot

    now = datetime(2026, 8, 22, 18, 0, tzinfo=UTC)
    return SituationalSnapshot(
        user_id="james",
        assembled_at=now,
        facts=(
            SituationalFact(
                key="calendar.next.destination",
                value="work",
                source_id="integration:calendar",
                observed_at=now - timedelta(seconds=120),
            ),
            SituationalFact(
                key="calendar.next.start_in_minutes",
                value=50.0,
                source_id="integration:calendar",
                observed_at=now - timedelta(seconds=120),
            ),
            SituationalFact(
                key="traffic.delay_minutes",
                value=18.0,
                source_id="integration:traffic",
                observed_at=now - timedelta(seconds=traffic_age_seconds),
            ),
            SituationalFact(
                key="weather.storm_probability",
                value=0.8,
                source_id="integration:weather",
                observed_at=now - timedelta(seconds=weather_age_seconds),
            ),
        ),
    )


def test_commute_weather_candidate_combines_authorized_sources():
    from rex.proactivity.evaluator import ProactiveOpportunityEvaluator

    candidate = ProactiveOpportunityEvaluator().evaluate(_commute_snapshot())[0]

    assert candidate.key == "commute:weather-delay"
    assert "leave" in candidate.spoken_text.lower()
    assert candidate.source_ids == (
        "integration:calendar",
        "integration:traffic",
        "integration:weather",
    )
    assert candidate.score >= 0.70


def test_stale_weather_disqualifies_commute_candidate():
    from rex.proactivity.evaluator import ProactiveOpportunityEvaluator

    result = ProactiveOpportunityEvaluator().evaluate(_commute_snapshot(weather_age_seconds=3600.0))

    assert result == ()


def test_low_signal_commute_does_not_cross_threshold():
    from rex.context.situational import SituationalFact
    from rex.proactivity.evaluator import ProactiveOpportunityEvaluator

    snapshot = _commute_snapshot()
    facts = tuple(
        SituationalFact(
            key=fact.key,
            value=(2.0 if fact.key == "traffic.delay_minutes" else fact.value),
            source_id=fact.source_id,
            observed_at=fact.observed_at,
        )
        for fact in snapshot.facts
    )
    snapshot = snapshot.with_facts(facts)

    assert ProactiveOpportunityEvaluator().evaluate(snapshot) == ()
