"""Privacy-safe benchmark aggregation used by Rex production-readiness gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkSample:
    """One timing sample with no request payload or user identity."""

    request_class: str
    warm_state: str
    evidence_class: str
    stages_ms: dict[str, float]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty sample")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def build_report(samples: list[BenchmarkSample], *, profile: str) -> dict[str, Any]:
    """Aggregate p50/p95 timings by request class and cold/warm state."""
    grouped: dict[tuple[str, str], list[BenchmarkSample]] = {}
    for sample in samples:
        if sample.warm_state not in {"cold", "warm"}:
            raise ValueError(f"Unsupported warm state: {sample.warm_state}")
        if not sample.stages_ms or "total" not in sample.stages_ms:
            raise ValueError("Every benchmark sample must include total timing")
        if any(value < 0 for value in sample.stages_ms.values()):
            raise ValueError("Benchmark timings cannot be negative")
        grouped.setdefault((sample.request_class, sample.warm_state), []).append(sample)

    results: dict[str, dict[str, Any]] = {}
    for (request_class, warm_state), bucket in sorted(grouped.items()):
        evidence = {sample.evidence_class for sample in bucket}
        if len(evidence) != 1:
            raise ValueError("A benchmark bucket cannot mix evidence classes")
        stage_names = sorted({name for sample in bucket for name in sample.stages_ms})
        stage_report: dict[str, dict[str, float]] = {}
        for stage in stage_names:
            values = [sample.stages_ms[stage] for sample in bucket if stage in sample.stages_ms]
            stage_report[stage] = {
                "p50": round(_percentile(values, 0.50), 3),
                "p95": round(_percentile(values, 0.95), 3),
            }
        results.setdefault(request_class, {})[warm_state] = {
            "evidence_class": next(iter(evidence)),
            "sample_count": len(bucket),
            "stages_ms": stage_report,
        }

    return {
        "schema_version": 1,
        "profile": profile,
        "privacy": "timings_and_nonsecret_runtime_identifiers_only",
        "results": results,
    }


__all__ = ["BenchmarkSample", "build_report"]
