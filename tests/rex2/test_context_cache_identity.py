"""US-105 identity and concurrency isolation tests for context caching."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from rex.context.cache import ContextArtifactCache, ContextCacheKey, ContextCacheVersions
from rex.context.revisions import ContextCacheRequest
from rex.runtime.turn import AuthorizationSnapshotRef, TurnScope


def _versions() -> ContextCacheVersions:
    return ContextCacheVersions(
        identity="identity",
        policy="policy",
        permission="permission",
        model="model",
        capability="capability",
        config="config",
        memory="memory",
        prompt_template="prompt",
    )


def _authorization() -> AuthorizationSnapshotRef:
    return AuthorizationSnapshotRef(policy_ref="policy", permission_ref="permission")


def test_concurrent_private_builds_never_cross_users() -> None:
    cache: ContextArtifactCache[str] = ContextArtifactCache(max_entries=8)
    barrier = Barrier(2)

    def run(user_id: str) -> str:
        key = ContextCacheKey.private(user_id, _versions())

        def build() -> str:
            barrier.wait(timeout=5)
            return f"artifact-for-{user_id}"

        return cache.get_or_build(key, build)

    with ThreadPoolExecutor(max_workers=2) as pool:
        james_future = pool.submit(run, "james")
        cole_future = pool.submit(run, "cole")
        results = {
            "james": james_future.result(timeout=5),
            "cole": cole_future.result(timeout=5),
        }

    assert results["james"] == "artifact-for-james"
    assert results["cole"] == "artifact-for-cole"
    metrics = cache.metrics_snapshot()["private_context"]
    assert metrics.entries == 2
    assert metrics.builds == 2


def test_household_reuse_requires_explicit_household_key() -> None:
    cache: ContextArtifactCache[str] = ContextArtifactCache(max_entries=8)
    private_key = ContextCacheKey.private("james", _versions())
    household_key = ContextCacheKey.household(_versions())
    builds = {"private": 0, "household": 0}

    def private_build() -> str:
        builds["private"] += 1
        return "private-artifact"

    def household_build() -> str:
        builds["household"] += 1
        return "household-artifact"

    assert cache.get_or_build(private_key, private_build) == "private-artifact"
    assert cache.get_or_build(household_key, household_build) == "household-artifact"
    assert cache.get_or_build(household_key, household_build) == "household-artifact"
    assert builds == {"private": 1, "household": 1}
    assert cache.metrics_snapshot()["private_context"].entries == 1
    assert cache.metrics_snapshot()["household_context"].entries == 1


def test_private_cache_request_rejects_missing_or_invalid_identity() -> None:
    for user_id in (None, "../james"):
        with pytest.raises(ValueError):
            ContextCacheRequest(
                user_id=user_id,
                scope=TurnScope.USER,
                authorization=_authorization(),
                model_provider="local",
                model_name="model",
            )


def test_household_cache_request_rejects_private_owner() -> None:
    with pytest.raises(ValueError):
        ContextCacheRequest(
            user_id="james",
            scope=TurnScope.HOUSEHOLD,
            authorization=_authorization(),
            model_provider="local",
            model_name="model",
        )

    request = ContextCacheRequest(
        user_id=None,
        scope=TurnScope.HOUSEHOLD,
        authorization=_authorization(),
        model_provider="local",
        model_name="model",
    )
    assert request.scope is TurnScope.HOUSEHOLD
    assert request.user_id is None
