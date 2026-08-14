"""US-105 tests for identity-safe context cache primitives."""

from __future__ import annotations

import pytest

from rex.context.cache import (
    ContextArtifactCache,
    ContextCacheKey,
    ContextCacheVersions,
)
from rex.runtime.turn import TurnScope


def _versions(**overrides: str) -> ContextCacheVersions:
    values = {
        "identity": "identity-v1",
        "policy": "policy-v1",
        "permission": "permission-v1",
        "model": "model-v1",
        "capability": "capability-v1",
        "config": "config-v1",
        "memory": "memory-v1",
        "prompt_template": "prompt-v1",
    }
    values.update(overrides)
    return ContextCacheVersions(**values)


def test_private_key_requires_validated_user_and_differs_by_owner() -> None:
    versions = _versions()

    james = ContextCacheKey.private("james", versions)
    cole = ContextCacheKey.private("cole", versions)

    assert james != cole
    assert james.scope is TurnScope.USER
    assert james.user_id == "james"
    with pytest.raises(ValueError):
        ContextCacheKey.private("../james", versions)


def test_household_key_has_no_private_owner() -> None:
    key = ContextCacheKey.household(_versions())

    assert key.scope is TurnScope.HOUSEHOLD
    assert key.user_id is None
    assert key != ContextCacheKey.private("james", _versions())


def test_versions_reject_empty_revision_tokens() -> None:
    with pytest.raises(ValueError):
        _versions(memory="")


def test_repeated_key_reuses_cached_artifact() -> None:
    cache: ContextArtifactCache[str] = ContextArtifactCache(max_entries=4)
    key = ContextCacheKey.private("james", _versions())
    calls = 0

    def build() -> str:
        nonlocal calls
        calls += 1
        return "artifact"

    assert cache.get_or_build(key, build) == "artifact"
    assert cache.get_or_build(key, build) == "artifact"
    assert calls == 1

    metrics = cache.metrics_snapshot()["private_context"]
    assert metrics.hits == 1
    assert metrics.misses == 1
    assert metrics.builds == 1
    assert metrics.entries == 1
    assert metrics.build_seconds >= 0.0


def test_cache_evicts_least_recently_used_entry_when_bounded() -> None:
    cache: ContextArtifactCache[str] = ContextArtifactCache(max_entries=2)
    first = ContextCacheKey.private("james", _versions(model="one"))
    second = ContextCacheKey.private("james", _versions(model="two"))
    third = ContextCacheKey.private("james", _versions(model="three"))
    cache.get_or_build(first, lambda: "first")
    cache.get_or_build(second, lambda: "second")
    assert cache.get_or_build(first, lambda: "unexpected") == "first"
    cache.get_or_build(third, lambda: "third")

    rebuilt = 0

    def rebuild_second() -> str:
        nonlocal rebuilt
        rebuilt += 1
        return "second-rebuilt"

    assert cache.get_or_build(second, rebuild_second) == "second-rebuilt"
    assert rebuilt == 1
    metrics = cache.metrics_snapshot()["private_context"]
    assert metrics.evictions == 2
    assert metrics.entries == 2


def test_metrics_are_content_free_and_category_bounded() -> None:
    cache: ContextArtifactCache[str] = ContextArtifactCache(max_entries=2)
    cache.get_or_build(ContextCacheKey.private("james", _versions()), lambda: "James secret")

    rendered = repr(cache.metrics_snapshot())
    assert "james" not in rendered.lower()
    assert "secret" not in rendered.lower()
    assert set(cache.metrics_snapshot()) <= {"private_context", "household_context"}
