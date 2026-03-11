"""Tests for US-040: Knowledge ingestion.

Acceptance criteria:
- documents ingested
- data indexed
- query containing a keyword from an indexed document returns that document in results
- Typecheck passes
"""

from __future__ import annotations

import pytest

from rex.knowledge_base import KnowledgeBase, KnowledgeDocument, set_knowledge_base


@pytest.fixture(autouse=True)
def isolated_kb(tmp_path):
    """Use a temp-backed KnowledgeBase and reset the global after each test."""
    kb = KnowledgeBase(
        docs_path=tmp_path / "docs.json",
        index_path=tmp_path / "index.json",
    )
    set_knowledge_base(kb)
    yield kb
    set_knowledge_base(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Documents ingested
# ---------------------------------------------------------------------------


def test_ingest_text_returns_document(isolated_kb):
    """ingest_text() returns a KnowledgeDocument."""
    doc = isolated_kb.ingest_text("Hello world content", title="Test Doc")
    assert isinstance(doc, KnowledgeDocument)
    assert doc.title == "Test Doc"
    assert doc.content == "Hello world content"


def test_ingest_text_assigns_unique_id(isolated_kb):
    """Each ingested document gets a unique doc_id."""
    doc1 = isolated_kb.ingest_text("Content one", title="Doc 1")
    doc2 = isolated_kb.ingest_text("Content two", title="Doc 2")
    assert doc1.doc_id != doc2.doc_id


def test_ingest_text_stores_document(isolated_kb):
    """Ingested document is retrievable via get_document()."""
    doc = isolated_kb.ingest_text("Stored content", title="Stored")
    retrieved = isolated_kb.get_document(doc.doc_id)
    assert retrieved is not None
    assert retrieved.doc_id == doc.doc_id


def test_ingest_file(isolated_kb, tmp_path):
    """ingest_file() reads a file and creates a document."""
    txt = tmp_path / "notes.txt"
    txt.write_text("File based content here", encoding="utf-8")
    doc = isolated_kb.ingest_file(txt, title="Notes")
    assert doc.title == "Notes"
    assert "File based content here" in doc.content


def test_ingest_file_missing_raises(isolated_kb, tmp_path):
    """ingest_file() raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        isolated_kb.ingest_file(tmp_path / "nonexistent.txt")


def test_ingest_file_default_title_from_stem(isolated_kb, tmp_path):
    """ingest_file() uses the file stem as the title when title is omitted."""
    txt = tmp_path / "my_document.txt"
    txt.write_text("content", encoding="utf-8")
    doc = isolated_kb.ingest_file(txt)
    assert doc.title == "my_document"


def test_ingest_increments_len(isolated_kb):
    """len(kb) increases after each ingestion."""
    assert len(isolated_kb) == 0
    isolated_kb.ingest_text("A", title="A")
    assert len(isolated_kb) == 1
    isolated_kb.ingest_text("B", title="B")
    assert len(isolated_kb) == 2


def test_ingest_with_tags(isolated_kb):
    """Tags are stored on the ingested document."""
    doc = isolated_kb.ingest_text("Tagged content", title="Tagged", tags=["alpha", "beta"])
    assert "alpha" in doc.tags
    assert "beta" in doc.tags


# ---------------------------------------------------------------------------
# Data indexed
# ---------------------------------------------------------------------------


def test_ingest_indexes_content_keywords(isolated_kb):
    """After ingestion, keywords from content appear in the internal index."""
    isolated_kb.ingest_text("The quantum entanglement phenomenon", title="Physics")
    # 'quantum' and 'entanglement' are non-stop-words; they must be in the index
    assert "quantum" in isolated_kb._index
    assert "entanglement" in isolated_kb._index


def test_ingest_indexes_title_keywords(isolated_kb):
    """Keywords from the document title appear in the index."""
    isolated_kb.ingest_text("Some content", title="Chromatic Aberration")
    assert "chromatic" in isolated_kb._index
    assert "aberration" in isolated_kb._index


def test_ingest_indexes_tags(isolated_kb):
    """Tag words appear in the index."""
    isolated_kb.ingest_text("content", title="T", tags=["photosynthesis"])
    assert "photosynthesis" in isolated_kb._index


def test_index_maps_keyword_to_doc_id(isolated_kb):
    """The index maps a keyword to the correct doc_id."""
    doc = isolated_kb.ingest_text("Superconductivity demo", title="Physics")
    assert doc.doc_id in isolated_kb._index.get("superconductivity", set())


def test_persistence_reloads_index(tmp_path):
    """After creating a new KnowledgeBase from the same paths, the index is restored."""
    docs_path = tmp_path / "docs.json"
    index_path = tmp_path / "index.json"

    kb1 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
    kb1.ingest_text("Bioluminescence is fascinating", title="Biology")

    kb2 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
    assert "bioluminescence" in kb2._index


# ---------------------------------------------------------------------------
# Query containing keyword returns that document in results
# ---------------------------------------------------------------------------


def test_search_returns_document_by_content_keyword(isolated_kb):
    """search() returns a document whose content contains the queried keyword."""
    doc = isolated_kb.ingest_text("Photosynthesis converts sunlight into energy", title="Biology")
    results = isolated_kb.search("photosynthesis")
    assert any(r.doc_id == doc.doc_id for r in results)


def test_search_returns_document_by_title_keyword(isolated_kb):
    """search() returns a document whose title contains the queried keyword."""
    doc = isolated_kb.ingest_text("General content", title="Thermodynamics Overview")
    results = isolated_kb.search("thermodynamics")
    assert any(r.doc_id == doc.doc_id for r in results)


def test_search_returns_document_by_tag_keyword(isolated_kb):
    """search() returns a document whose tag contains the queried keyword."""
    doc = isolated_kb.ingest_text("content here", title="Doc", tags=["neuroscience"])
    results = isolated_kb.search("neuroscience")
    assert any(r.doc_id == doc.doc_id for r in results)


def test_search_no_results_for_unrelated_term(isolated_kb):
    """search() returns empty list when no document matches the query."""
    isolated_kb.ingest_text("Cats and dogs", title="Pets")
    results = isolated_kb.search("xylophone")
    assert results == []


def test_search_respects_max_results(isolated_kb):
    """search() limits results to max_results."""
    for i in range(10):
        isolated_kb.ingest_text(f"Gravity waves document {i}", title=f"Doc {i}")
    results = isolated_kb.search("gravity", max_results=3)
    assert len(results) <= 3


def test_search_empty_query_returns_empty(isolated_kb):
    """search() returns empty list for an empty query string."""
    isolated_kb.ingest_text("Some content", title="Doc")
    results = isolated_kb.search("")
    assert results == []


def test_search_returns_most_relevant_first(isolated_kb):
    """Document with keyword in both title and content ranks higher than content-only."""
    doc_high = isolated_kb.ingest_text("Magnetism content", title="Magnetism Guide")
    doc_low = isolated_kb.ingest_text("Magnetism mentioned once", title="Generic Title")
    results = isolated_kb.search("magnetism")
    # doc_high should appear at or before doc_low
    ids = [r.doc_id for r in results]
    assert ids.index(doc_high.doc_id) <= ids.index(doc_low.doc_id)
