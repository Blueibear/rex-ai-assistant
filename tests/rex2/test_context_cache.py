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


# ---------------------------------------------------------------------------
# Deterministic revision snapshots
# ---------------------------------------------------------------------------


def _revision_request(
    *, policy: str = "policy-a", permission: str = "permission-a", model: str = "model-a"
):
    from rex.context.revisions import ContextCacheRequest
    from rex.runtime.turn import AuthorizationSnapshotRef

    return ContextCacheRequest(
        user_id="james",
        scope=TurnScope.USER,
        authorization=AuthorizationSnapshotRef(policy_ref=policy, permission_ref=permission),
        model_provider="local",
        model_name=model,
    )


def _revision_settings():
    from types import SimpleNamespace

    return SimpleNamespace(
        default_timezone="America/Chicago",
        default_location="Dallas, TX",
        personality="Friendly",
        openai_api_key="credential-value-must-never-be-fingerprinted",  # pragma: allowlist secret
    )


def _revision_registry():
    from rex.capabilities.registry import Capability, CapabilityRegistry

    registry = CapabilityRegistry()
    registry.register(
        Capability(
            name="test_read",
            description="Test read capability",
            enabled=True,
            health="unknown",
        )
    )
    return registry


def _seed_revision_files(tmp_path, monkeypatch):
    import json

    import rex.memory_utils as memory_utils

    identity_root = tmp_path / "identity"
    legacy_root = tmp_path / "legacy"
    (identity_root / "james").mkdir(parents=True)
    (legacy_root / "james").mkdir(parents=True)
    identity_profile = identity_root / "james" / "core.json"
    memory_profile = legacy_root / "james" / "core.json"
    facts_file = legacy_root / "james_facts.json"
    identity_profile.write_text(
        json.dumps({"role": "owner", "name": "James Secret"}), encoding="utf-8"
    )
    memory_profile.write_text(json.dumps({"preferences": {"tone": "concise"}}), encoding="utf-8")
    facts_file.write_text(json.dumps({"private_fact": "James Secret Fact"}), encoding="utf-8")
    monkeypatch.setenv("ASKREX_MEMORY_DIR", str(identity_root))
    monkeypatch.setattr(memory_utils, "MEMORY_ROOT", legacy_root)
    return identity_profile, memory_profile, facts_file


def test_policy_permission_and_model_revisions_change_deterministically(
    tmp_path, monkeypatch
) -> None:
    from rex.context.revisions import build_context_cache_versions

    _seed_revision_files(tmp_path, monkeypatch)
    settings = _revision_settings()
    registry = _revision_registry()
    baseline = build_context_cache_versions(_revision_request(), settings, registry)

    policy = build_context_cache_versions(_revision_request(policy="policy-b"), settings, registry)
    permission = build_context_cache_versions(
        _revision_request(permission="permission-b"), settings, registry
    )
    model = build_context_cache_versions(_revision_request(model="model-b"), settings, registry)

    assert policy.policy != baseline.policy
    assert permission.permission != baseline.permission
    assert model.model != baseline.model
    assert baseline == build_context_cache_versions(_revision_request(), settings, registry)


def test_context_source_policy_revision_invalidates_policy_version(tmp_path, monkeypatch) -> None:
    from rex.context.revisions import build_context_cache_versions

    _seed_revision_files(tmp_path, monkeypatch)
    settings = _revision_settings()
    registry = _revision_registry()
    baseline = build_context_cache_versions(
        _revision_request(), settings, registry, source_policy_revision="source-policy-a"
    )
    changed = build_context_cache_versions(
        _revision_request(), settings, registry, source_policy_revision="source-policy-b"
    )

    assert changed.policy != baseline.policy
    assert "source-policy" not in repr(changed)


def test_capability_and_config_changes_invalidate_versions(tmp_path, monkeypatch) -> None:
    from rex.context.revisions import build_context_cache_versions

    _seed_revision_files(tmp_path, monkeypatch)
    settings = _revision_settings()
    registry = _revision_registry()
    baseline = build_context_cache_versions(_revision_request(), settings, registry)
    registry.update_runtime_state("test_read", health="healthy")
    capability_changed = build_context_cache_versions(_revision_request(), settings, registry)
    settings.personality = "Formal"
    config_changed = build_context_cache_versions(_revision_request(), settings, registry)

    assert capability_changed.capability != baseline.capability
    assert config_changed.config != capability_changed.config


def test_identity_memory_and_prompt_template_changes_invalidate_versions(
    tmp_path, monkeypatch
) -> None:
    import rex.context.revisions as revisions

    identity_profile, _memory_profile, facts_file = _seed_revision_files(tmp_path, monkeypatch)
    settings = _revision_settings()
    registry = _revision_registry()
    baseline = revisions.build_context_cache_versions(_revision_request(), settings, registry)

    identity_profile.write_text('{"role":"guest","name":"James Secret"}', encoding="utf-8")
    identity_changed = revisions.build_context_cache_versions(
        _revision_request(), settings, registry
    )
    facts_file.write_text('{"private_fact":"Changed Secret Fact"}', encoding="utf-8")
    memory_changed = revisions.build_context_cache_versions(_revision_request(), settings, registry)
    monkeypatch.setattr(revisions, "CONTEXT_PROMPT_TEMPLATE_REVISION", "context-builder-v2")
    prompt_changed = revisions.build_context_cache_versions(_revision_request(), settings, registry)

    assert identity_changed.identity != baseline.identity
    assert memory_changed.memory != identity_changed.memory
    assert prompt_changed.prompt_template != memory_changed.prompt_template


def test_revision_tokens_do_not_expose_private_content_or_credentials(
    tmp_path, monkeypatch
) -> None:
    from rex.context.revisions import build_context_cache_versions

    _seed_revision_files(tmp_path, monkeypatch)
    versions = build_context_cache_versions(
        _revision_request(), _revision_settings(), _revision_registry()
    )

    rendered = repr(versions).lower()
    assert "james secret" not in rendered
    assert "secret fact" not in rendered
    assert "credential-value" not in rendered  # pragma: allowlist secret
    assert "core.json" not in rendered
    for token in (
        versions.identity,
        versions.policy,
        versions.permission,
        versions.model,
        versions.capability,
        versions.config,
        versions.memory,
        versions.prompt_template,
    ):
        assert len(token) == 64
        int(token, 16)
