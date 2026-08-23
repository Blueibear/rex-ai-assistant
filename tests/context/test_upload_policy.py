"""US-123 upload ownership, context eligibility, and provenance tests."""

from __future__ import annotations

import json

import pytest

from rex.context.source_policy import ContextSourcePolicyStore
from rex.knowledge_base import KnowledgeBase


def _kb(tmp_path) -> KnowledgeBase:
    return KnowledgeBase(
        docs_path=tmp_path / "docs.json",
        index_path=tmp_path / "index.json",
        source_policy_store=ContextSourcePolicyStore(tmp_path / "source-policy"),
    )


def test_context_disabled_upload_is_explicit_query_only(tmp_path) -> None:
    kb = _kb(tmp_path)
    kb.ingest_text(
        "secret recipe",
        title="Recipe",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=False,
    )
    assert kb.search_for_user("recipe", requester_user_id="james", context_only=True) == []
    assert [
        doc.title
        for doc in kb.search_for_user("recipe", requester_user_id="james", context_only=False)
    ] == ["Recipe"]


def test_private_upload_never_crosses_users_even_for_explicit_search(tmp_path) -> None:
    kb = _kb(tmp_path)
    kb.ingest_text(
        "private tax notes",
        title="Taxes",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )

    assert kb.search_for_user("tax", requester_user_id="cole", context_only=True) == []
    assert kb.search_for_user("tax", requester_user_id="cole", context_only=False) == []


def test_household_upload_is_visible_to_other_user_when_context_enabled(tmp_path) -> None:
    kb = _kb(tmp_path)
    kb.ingest_text(
        "household warranty details",
        title="Warranty",
        owner_user_id="james",
        audience_scope="household",
        context_enabled=True,
    )

    results = kb.search_for_user("warranty", requester_user_id="cole", context_only=True)
    assert [doc.title for doc in results] == ["Warranty"]


def test_legacy_unscoped_document_never_enters_background_context(tmp_path) -> None:
    docs_path = tmp_path / "docs.json"
    index_path = tmp_path / "index.json"
    docs_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "doc_legacy",
                        "title": "Legacy Notes",
                        "content": "legacy private material",
                        "source_path": None,
                        "tags": [],
                        "created_at": "2026-01-01T00:00:00Z",
                        "word_count": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

    legacy = kb.get_document("doc_legacy")
    assert legacy is not None
    assert legacy.audience_scope == "legacy_unassigned"
    assert legacy.context_enabled is False
    assert kb.search_for_user("legacy", requester_user_id="james", context_only=True) == []
    assert kb.search_for_user("legacy", requester_user_id="james", context_only=False) == []


def test_partial_new_upload_policy_is_rejected(tmp_path) -> None:
    kb = _kb(tmp_path)

    with pytest.raises(ValueError, match="audience_scope.*context_enabled"):
        kb.ingest_text("partial", title="Partial", owner_user_id="james")


def test_owner_can_assign_policy_to_legacy_document(tmp_path) -> None:
    kb = _kb(tmp_path)
    legacy = kb.ingest_text("claim me", title="Legacy")
    before = legacy.policy_revision

    updated = kb.assign_document_policy(
        legacy.doc_id,
        owner_user_id="james",
        actor_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )

    assert updated.owner_user_id == "james"
    assert updated.context_enabled is True
    assert updated.policy_revision != before
    assert [
        doc.title
        for doc in kb.search_for_user("claim", requester_user_id="james", context_only=True)
    ] == ["Legacy"]


def test_other_user_cannot_assign_or_reassign_document_policy(tmp_path) -> None:
    kb = _kb(tmp_path)
    doc = kb.ingest_text(
        "owned data",
        title="Owned",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )

    with pytest.raises(PermissionError, match="owner authorization required"):
        kb.assign_document_policy(
            doc.doc_id,
            owner_user_id="james",
            actor_user_id="cole",
            audience_scope="household",
            context_enabled=True,
        )


def test_upload_policy_updates_canonical_source_revision(tmp_path) -> None:
    policy_store = ContextSourcePolicyStore(tmp_path / "source-policy")
    kb = KnowledgeBase(
        docs_path=tmp_path / "docs.json",
        index_path=tmp_path / "index.json",
        source_policy_store=policy_store,
    )
    before = policy_store.revision_for_user("james")
    doc = kb.ingest_text(
        "contextual notes",
        title="Notes",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )
    after_ingest = policy_store.revision_for_user("james")

    assert after_ingest != before
    assert doc.policy_revision

    updated = kb.assign_document_policy(
        doc.doc_id,
        owner_user_id="james",
        actor_user_id="james",
        audience_scope="private",
        context_enabled=False,
    )
    after_disable = policy_store.revision_for_user("james")

    assert after_disable != after_ingest
    assert updated.policy_revision != doc.policy_revision
    assert kb.search_for_user("contextual", requester_user_id="james", context_only=True) == []


def test_context_results_carry_stable_upload_source_ids(tmp_path) -> None:
    kb = _kb(tmp_path)
    doc = kb.ingest_text(
        "project launch notes",
        title="Launch",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )

    results = kb.search_for_user("launch", requester_user_id="james", context_only=True)
    assert len(results) == 1
    assert results[0].source_id == f"upload:{doc.doc_id}"


def test_context_builder_formats_upload_provenance_without_credentials(tmp_path) -> None:
    from rex.context.builder import ContextBuilder

    kb = _kb(tmp_path)
    doc = kb.ingest_text(
        "meeting moved to Thursday",
        title="Schedule Notes",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )
    builder = ContextBuilder(settings=object(), history=[], user_id="james")

    rendered = builder.format_context_documents(
        kb.search_for_user("meeting", requester_user_id="james", context_only=True)
    )

    assert f"upload:{doc.doc_id}" in rendered
    assert "Schedule Notes" in rendered
    assert "meeting moved to Thursday" in rendered
    assert "policy_revision" not in rendered


def test_legacy_migration_persists_fail_closed_policy_fields(tmp_path) -> None:
    docs_path = tmp_path / "docs.json"
    index_path = tmp_path / "index.json"
    docs_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "doc_old",
                        "title": "Old",
                        "content": "legacy content",
                        "source_path": None,
                        "tags": [],
                        "created_at": "2026-01-01T00:00:00Z",
                        "word_count": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    KnowledgeBase(docs_path=docs_path, index_path=index_path)
    saved = json.loads(docs_path.read_text(encoding="utf-8"))["documents"][0]
    assert saved["owner_user_id"] is None
    assert saved["audience_scope"] == "legacy_unassigned"
    assert saved["context_enabled"] is False


def test_list_documents_for_user_filters_before_returning_metadata(tmp_path) -> None:
    kb = _kb(tmp_path)
    kb.ingest_text(
        "private one",
        title="James Private",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )
    kb.ingest_text(
        "shared one",
        title="House Shared",
        owner_user_id="james",
        audience_scope="household",
        context_enabled=True,
    )
    kb.ingest_text("legacy one", title="Legacy")

    james_titles = {doc.title for doc in kb.list_documents_for_user("james")}
    cole_titles = {doc.title for doc in kb.list_documents_for_user("cole")}

    assert james_titles == {"James Private", "House Shared"}
    assert cole_titles == {"House Shared"}


def test_memories_bridge_lists_only_safe_document_metadata(tmp_path) -> None:
    import importlib

    from rex.knowledge_base import set_knowledge_base

    kb = _kb(tmp_path)
    doc = kb.ingest_text(
        "do not expose this content in metadata",
        title="Private Upload",
        source_path="C:/private/secret.txt",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )
    bridge = importlib.import_module("rex_memories_bridge")
    set_knowledge_base(kb)
    try:
        result = bridge._handle_documents_list("james")
    finally:
        set_knowledge_base(None)

    assert result["ok"] is True
    assert len(result["documents"]) == 1
    metadata = result["documents"][0]
    assert metadata["id"] == doc.doc_id
    assert metadata["sourceId"] == doc.source_id
    assert metadata["title"] == "Private Upload"
    assert metadata["audienceScope"] == "private"
    assert metadata["contextEnabled"] is True
    assert "content" not in metadata
    assert "source_path" not in metadata
    assert "sourcePath" not in metadata
    assert "C:/private/secret.txt" not in repr(metadata)


def test_memories_bridge_cannot_change_another_users_upload_policy(tmp_path) -> None:
    import importlib

    from rex.knowledge_base import set_knowledge_base

    kb = _kb(tmp_path)
    doc = kb.ingest_text(
        "owned by james",
        title="Owned Upload",
        owner_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )
    bridge = importlib.import_module("rex_memories_bridge")
    set_knowledge_base(kb)
    try:
        with pytest.raises(PermissionError, match="owner authorization required"):
            bridge._handle_document_policy(
                "cole",
                doc.doc_id,
                {"audience_scope": "household", "context_enabled": True},
            )
    finally:
        set_knowledge_base(None)


def test_deleting_scoped_upload_invalidates_context_revision(tmp_path) -> None:
    policy_store = ContextSourcePolicyStore(tmp_path / "source-policy")
    kb = KnowledgeBase(
        docs_path=tmp_path / "docs.json",
        index_path=tmp_path / "index.json",
        source_policy_store=policy_store,
    )
    doc = kb.ingest_text(
        "temporary shared context",
        title="Temporary",
        owner_user_id="james",
        audience_scope="household",
        context_enabled=True,
    )
    cole_before = policy_store.revision_for_user("cole")

    assert kb.delete_document(doc.doc_id, requester_user_id="james") is True

    assert policy_store.revision_for_user("cole") != cole_before
    assert not policy_store.is_context_eligible(
        doc.source_id,
        subject_user_id="james",
        requester_user_id="cole",
    )


def test_legacy_compatibility_apis_cannot_expose_scoped_upload(tmp_path) -> None:
    kb = _kb(tmp_path)
    source = tmp_path / "private.txt"
    source.write_text("changed private content", encoding="utf-8")
    doc = kb.ingest_text(
        "private compatibility secret",
        title="Scoped",
        source_path=str(source),
        owner_user_id="james",
        audience_scope="private",
        context_enabled=True,
    )

    assert kb.get_document(doc.doc_id) is None
    assert kb.list_documents() == []
    assert kb.get_citations("compatibility") == []
    assert kb.refresh_document(doc.doc_id) is None
    assert kb.delete_document(doc.doc_id) is False
    owned = kb.get_document_for_user(doc.doc_id, requester_user_id="james")
    assert owned is not None and owned.content == "private compatibility secret"
    assert kb.get_document_for_user(doc.doc_id, requester_user_id="cole") is None
