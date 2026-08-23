"""Typed proactive-assistance candidates with deterministic scoring."""

from __future__ import annotations

from dataclasses import dataclass

from rex.identity import validate_user_id


def _unit_interval(value: float, field: str) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return normalized


@dataclass(frozen=True, slots=True)
class ProactiveCandidate:
    key: str
    user_id: str
    spoken_text: str
    source_ids: tuple[str, ...]
    freshness_seconds: float
    confidence: float
    benefit: float
    urgency: float
    suggested_action: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip() or len(self.key) > 160:
            raise ValueError("proactive candidate key is invalid")
        object.__setattr__(self, "user_id", validate_user_id(self.user_id))
        if not isinstance(self.spoken_text, str) or not self.spoken_text.strip():
            raise ValueError("proactive candidate spoken_text is invalid")
        if self.freshness_seconds < 0:
            raise ValueError("freshness_seconds must not be negative")
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))
        object.__setattr__(self, "benefit", _unit_interval(self.benefit, "benefit"))
        object.__setattr__(self, "urgency", _unit_interval(self.urgency, "urgency"))

    @property
    def score(self) -> float:
        return 0.45 * self.benefit + 0.35 * self.urgency + 0.20 * self.confidence


__all__ = ["ProactiveCandidate"]
