"""Deterministic high-signal proactive opportunity evaluation."""

from __future__ import annotations

from rex.context.situational import SituationalSnapshot

from .models import ProactiveCandidate

_SCORE_THRESHOLD = 0.70
_CURRENT_INFO_MAX_AGE_SECONDS = 900.0
_COMMUTE_WINDOW_MINUTES = 180.0


class ProactiveOpportunityEvaluator:
    """Turn authorized situational facts into ranked, truthful opportunities."""

    def required_current_info(self, snapshot: SituationalSnapshot) -> tuple[str, ...]:
        destination = snapshot.value("calendar.next.destination")
        starts_in = snapshot.value("calendar.next.start_in_minutes")
        if not isinstance(destination, str) or not isinstance(starts_in, (int, float)):
            return ()
        if not 0 <= float(starts_in) <= _COMMUTE_WINDOW_MINUTES:
            return ()
        required: list[str] = []
        if snapshot.fact("traffic.delay_minutes") is None:
            required.append("traffic")
        if snapshot.fact("weather.storm_probability") is None:
            required.append("weather")
        return tuple(required)

    def evaluate(self, snapshot: SituationalSnapshot) -> tuple[ProactiveCandidate, ...]:
        candidates: list[ProactiveCandidate] = []
        commute = self._commute_weather(snapshot)
        if commute is not None and commute.score >= _SCORE_THRESHOLD:
            candidates.append(commute)
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.key))
        return tuple(candidates)

    def _commute_weather(self, snapshot: SituationalSnapshot) -> ProactiveCandidate | None:
        destination = snapshot.value("calendar.next.destination")
        starts_in = snapshot.value("calendar.next.start_in_minutes")
        delay = snapshot.value("traffic.delay_minutes")
        storm = snapshot.value("weather.storm_probability")
        if not isinstance(destination, str):
            return None
        if isinstance(starts_in, bool) or not isinstance(starts_in, (int, float)):
            return None
        if isinstance(delay, bool) or not isinstance(delay, (int, float)):
            return None
        if isinstance(storm, bool) or not isinstance(storm, (int, float)):
            return None
        starts = float(starts_in)
        delay_minutes = float(delay)
        storm_probability = float(storm)
        if not 0 <= starts <= _COMMUTE_WINDOW_MINUTES:
            return None
        if delay_minutes < 10.0 or storm_probability < 0.60:
            return None
        traffic_age = snapshot.freshness_seconds("traffic.delay_minutes")
        weather_age = snapshot.freshness_seconds("weather.storm_probability")
        if traffic_age is None or weather_age is None:
            return None
        if max(traffic_age, weather_age) > _CURRENT_INFO_MAX_AGE_SECONDS:
            return None

        calendar_fact = snapshot.fact("calendar.next.destination")
        traffic_fact = snapshot.fact("traffic.delay_minutes")
        weather_fact = snapshot.fact("weather.storm_probability")
        if calendar_fact is None or traffic_fact is None or weather_fact is None:
            return None

        benefit = min(1.0, 0.55 + delay_minutes / 40.0 + storm_probability * 0.15)
        urgency = min(1.0, max(0.0, 1.0 - starts / _COMMUTE_WINDOW_MINUTES + 0.25))
        confidence = min(1.0, 0.80 + 0.10 * storm_probability)
        leave_early = max(10, round(delay_minutes / 5.0) * 5)
        spoken = (
            f"Traffic and the weather could slow your trip to {destination}. "
            f"You should leave about {leave_early} minutes earlier."
        )
        return ProactiveCandidate(
            key="commute:weather-delay",
            user_id=snapshot.user_id,
            spoken_text=spoken,
            source_ids=tuple(
                dict.fromkeys(
                    (
                        calendar_fact.source_id,
                        traffic_fact.source_id,
                        weather_fact.source_id,
                    )
                )
            ),
            freshness_seconds=max(
                snapshot.freshness_seconds("calendar.next.destination") or 0.0,
                traffic_age,
                weather_age,
            ),
            confidence=confidence,
            benefit=benefit,
            urgency=urgency,
            suggested_action="show_route",
        )


__all__ = ["ProactiveOpportunityEvaluator"]
