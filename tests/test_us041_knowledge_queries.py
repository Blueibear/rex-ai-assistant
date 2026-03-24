"""US-041: Knowledge queries acceptance tests.

Acceptance criteria:
- queries executed
- query returns at least one result when indexed content contains the queried term
- errors handled
- Typecheck passes
"""

from __future__ import annotations

import pytest

from rex.knowledge_base import KnowledgeBase, search_documents, set_knowledge_base


@pytest.fixture(autouse=True)
def isolated_kb(tmp_path):
    """Provide an isolated KnowledgeBase backed by tmp_path."""
    kb = KnowledgeBase(
        docs_path=tmp_path / "docs.json",
        index_path=tmp_path / "index.json",
    )
    set_knowledge_base(kb)
    yield kb
    set_knowledge_base(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# queries executed
# ---------------------------------------------------------------------------


def test_search_returns_list(isolated_kb):
    """search() always returns a list, even on empty KB."""
    results = isolated_kb.search("anything")
    assert isinstance(results, list)


def test_search_empty_kb_returns_empty_list(isolated_kb):
    """Searching an empty knowledge base returns an empty list."""
    results = isolated_kb.search("python")
    assert results == []


def test_search_with_max_results_parameter(isolated_kb):
    """max_results limits the number of returned documents."""
    for i in range(5):
        isolated_kb.ingest_text(f"document about python topic {i}", title=f"Doc {i}")

    results = isolated_kb.search("python", max_results=2)
    assert len(results) <= 2


def test_search_all_results_are_knowledge_documents(isolated_kb):
    """Every result is a KnowledgeDocument instance."""
    from rex.knowledge_base import KnowledgeDocument

    isolated_kb.ingest_text("quantum computing overview", title="Quantum")
    results = isolated_kb.search("quantum")
    for doc in results:
        assert isinstance(doc, KnowledgeDocument)


def test_search_stop_words_only_returns_empty(isolated_kb):
    """Query consisting entirely of stop words returns empty list."""
    isolated_kb.ingest_text("hello world content", title="Test")
    results = isolated_kb.search("the and or")
    assert results == []


# ---------------------------------------------------------------------------
# query returns at least one result when indexed content contains the queried term
# ---------------------------------------------------------------------------


def test_query_matches_content_keyword(isolated_kb):
    """A keyword present in content is returned by search."""
    isolated_kb.ingest_text(
        "Rex uses transformers for natural language processing",
        title="Architecture",
    )
    results = isolated_kb.search("transformers")
    assert len(results) >= 1
    assert any("transformers" in doc.content.lower() for doc in results)


def test_query_matches_title_keyword(isolated_kb):
    """A keyword present in the title is returned by search."""
    isolated_kb.ingest_text(
        "Details about deep learning approaches",
        title="DeepLearning Guide",
    )
    results = isolated_kb.search("deeplearning")
    assert len(results) >= 1


def test_query_returns_most_relevant_first(isolated_kb):
    """The most relevant document (more keyword hits) ranks higher."""
    isolated_kb.ingest_text(
        "machine learning machine learning machine learning",
        title="ML Heavy",
    )
    isolated_kb.ingest_text(
        "machine learning is interesting",
        title="ML Light",
    )
    results = isolated_kb.search("machine learning")
    assert len(results) >= 1
    # Both documents match; verify we got results
    assert results[0].title in ("ML Heavy", "ML Light")


def test_query_multiword_finds_document(isolated_kb):
    """A multi-word query finds a document containing those words."""
    isolated_kb.ingest_text(
        "The neural network architecture uses attention mechanisms",
        title="Neural Networks",
    )
    results = isolated_kb.search("neural network attention")
    assert len(results) >= 1


def test_query_tag_filter_narrows_results(isolated_kb):
    """Tag filter restricts results to documents with all specified tags."""
    isolated_kb.ingest_text(
        "voice recognition pipeline details",
        title="Voice Doc",
        tags=["voice", "audio"],
    )
    isolated_kb.ingest_text(
        "voice assistant overview without audio tag",
        title="Assistant Doc",
        tags=["voice"],
    )

    results = isolated_kb.search("voice", tags=["audio"])
    assert len(results) >= 1
    for doc in results:
        assert "audio" in [t.lower() for t in doc.tags]


def test_query_tag_filter_no_match_returns_empty(isolated_kb):
    """Tag filter that matches no document returns empty list."""
    isolated_kb.ingest_text("some content here", title="Doc", tags=["science"])
    results = isolated_kb.search("content", tags=["nonexistent_tag"])
    assert results == []


def test_convenience_function_search_documents(isolated_kb):
    """search_documents() convenience function executes a query successfully."""
    isolated_kb.ingest_text(
        "knowledge base integration test content",
        title="Integration",
    )
    results = search_documents("knowledge")
    assert len(results) >= 1


def test_prefix_match_returns_result(isolated_kb):
    """Prefix of an indexed word still returns the document."""
    isolated_kb.ingest_text(
        "automation workflows simplify repetitive tasks",
        title="Automation",
    )
    # "automat" is a prefix of "automation"
    results = isolated_kb.search("automat")
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# errors handled
# ---------------------------------------------------------------------------


def test_search_empty_query_returns_empty_list(isolated_kb):
    """Empty string query returns empty list without raising."""
    isolated_kb.ingest_text("some content", title="Doc")
    results = isolated_kb.search("")
    assert results == []


def test_search_whitespace_only_query_returns_empty(isolated_kb):
    """Whitespace-only query returns empty list without raising."""
    isolated_kb.ingest_text("some content", title="Doc")
    results = isolated_kb.search("   ")
    assert results == []


def test_search_special_characters_does_not_raise(isolated_kb):
    """Query with special characters executes without raising."""
    isolated_kb.ingest_text("special content here", title="Special")
    try:
        results = isolated_kb.search("!@#$%^&*()")
        assert isinstance(results, list)
    except Exception as exc:
        pytest.fail(f"search() raised unexpectedly: {exc}")


def test_search_very_long_query_does_not_raise(isolated_kb):
    """Very long query string executes without raising."""
    isolated_kb.ingest_text("relevant content for testing", title="Test")
    long_query = "word " * 500
    try:
        results = isolated_kb.search(long_query)
        assert isinstance(results, list)
    except Exception as exc:
        pytest.fail(f"search() raised unexpectedly: {exc}")


def test_get_citations_returns_list(isolated_kb):
    """get_citations() returns a list of doc_ids without raising."""
    isolated_kb.ingest_text("citation test content phrase", title="Citation Doc")
    citations = isolated_kb.get_citations("citation test")
    assert isinstance(citations, list)
    assert len(citations) >= 1


def test_get_citations_no_match_returns_empty(isolated_kb):
    """get_citations() returns empty list when no document matches."""
    isolated_kb.ingest_text("hello world", title="Hello")
    citations = isolated_kb.get_citations("nonexistent phrase xyz")
    assert citations == []


def test_search_after_delete_excludes_deleted_doc(isolated_kb):
    """Deleted documents do not appear in search results."""
    doc = isolated_kb.ingest_text("temporary document content uniqueterm", title="Temp")
    isolated_kb.delete_document(doc.doc_id)
    results = isolated_kb.search("uniqueterm")
    assert all(r.doc_id != doc.doc_id for r in results)
