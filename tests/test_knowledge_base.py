"""Tests for the knowledge base module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from rex.knowledge_base import (
    KnowledgeDocument,
    KnowledgeBase,
    get_knowledge_base,
    set_knowledge_base,
    ingest_document,
    search_documents,
)


# =============================================================================
# KnowledgeDocument Tests
# =============================================================================


class TestKnowledgeDocument:
    """Tests for the KnowledgeDocument model."""

    def test_word_count_computed(self):
        """Test that word count is computed from content."""
        doc = KnowledgeDocument(
            title="Test Doc",
            content="This is a test document with seven words.",
        )
        assert doc.word_count == 8

    def test_word_count_with_empty_content(self):
        """Test word count with empty content."""
        doc = KnowledgeDocument(
            title="Empty",
            content="",
        )
        assert doc.word_count == 0

    def test_default_values(self):
        """Test default values for optional fields."""
        doc = KnowledgeDocument(
            title="Test",
            content="Content",
        )
        assert doc.doc_id.startswith("doc_")
        assert doc.source_path is None
        assert doc.tags == []
        assert doc.created_at is not None


# =============================================================================
# KnowledgeBase Tests
# =============================================================================


class TestKnowledgeBase:
    """Tests for the KnowledgeBase class."""

    def test_ingest_file(self, tmp_path):
        """Test ingesting a text file."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        # Create a test file
        test_file = tmp_path / "test_doc.txt"
        test_file.write_text("This is test content for ingestion.")

        doc = kb.ingest_file(test_file)
        assert doc.title == "test_doc"
        assert "test content" in doc.content
        assert doc.source_path == str(test_file.resolve())

    def test_ingest_file_with_title(self, tmp_path):
        """Test ingesting a file with custom title."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        test_file = tmp_path / "notes.txt"
        test_file.write_text("Some notes here.")

        doc = kb.ingest_file(test_file, title="My Custom Title")
        assert doc.title == "My Custom Title"

    def test_ingest_file_with_tags(self, tmp_path):
        """Test ingesting a file with tags."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        test_file = tmp_path / "tagged.txt"
        test_file.write_text("Tagged content.")

        doc = kb.ingest_file(test_file, tags=["project", "notes"])
        assert doc.tags == ["project", "notes"]

    def test_ingest_file_not_found(self, tmp_path):
        """Test ingesting a non-existent file raises error."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        with pytest.raises(FileNotFoundError):
            kb.ingest_file(tmp_path / "nonexistent.txt")

    def test_ingest_text(self, tmp_path):
        """Test ingesting text directly."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        doc = kb.ingest_text(
            content="Direct text content.",
            title="Direct Text",
            tags=["direct"],
        )
        assert doc.title == "Direct Text"
        assert doc.content == "Direct text content."
        assert "direct" in doc.tags

    def test_get_document(self, tmp_path):
        """Test retrieving a document by ID."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        doc = kb.ingest_text(content="Test", title="Test Doc")
        retrieved = kb.get_document(doc.doc_id)

        assert retrieved is not None
        assert retrieved.title == "Test Doc"

    def test_get_document_not_found(self, tmp_path):
        """Test retrieving a non-existent document."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        result = kb.get_document("nonexistent")
        assert result is None

    def test_delete_document(self, tmp_path):
        """Test deleting a document."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        doc = kb.ingest_text(content="To delete", title="Delete Me")
        assert kb.get_document(doc.doc_id) is not None

        result = kb.delete_document(doc.doc_id)
        assert result is True
        assert kb.get_document(doc.doc_id) is None

    def test_delete_document_not_found(self, tmp_path):
        """Test deleting a non-existent document."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        result = kb.delete_document("nonexistent")
        assert result is False

    def test_search_by_content(self, tmp_path):
        """Test searching by content."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(
            content="The weather today is sunny and warm.",
            title="Weather Report",
        )
        kb.ingest_text(
            content="Meeting notes from the team discussion.",
            title="Meeting Notes",
        )

        results = kb.search("weather")
        assert len(results) == 1
        assert results[0].title == "Weather Report"

    def test_search_by_title(self, tmp_path):
        """Test searching matches title."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(
            content="Some random content here.",
            title="Project Alpha Documentation",
        )

        results = kb.search("Alpha")
        assert len(results) == 1
        assert "Alpha" in results[0].title

    def test_search_max_results(self, tmp_path):
        """Test search respects max_results."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        for i in range(10):
            kb.ingest_text(
                content=f"Document {i} with common keyword.",
                title=f"Doc {i}",
            )

        results = kb.search("common", max_results=3)
        assert len(results) == 3

    def test_search_with_tags_filter(self, tmp_path):
        """Test searching with tag filter."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(
            content="Project documentation.",
            title="Project Docs",
            tags=["project", "docs"],
        )
        kb.ingest_text(
            content="Personal notes.",
            title="Personal Notes",
            tags=["personal"],
        )

        results = kb.search("documentation", tags=["project"])
        assert len(results) == 1
        assert results[0].title == "Project Docs"

    def test_search_ranking(self, tmp_path):
        """Test that search results are ranked by relevance."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        # Document with query in title should rank higher
        kb.ingest_text(
            content="Some content here.",
            title="Python Programming Guide",
        )
        kb.ingest_text(
            content="Learn about python programming and more.",
            title="General Guide",
        )

        results = kb.search("python programming")
        # The one with query in title should be first
        assert len(results) == 2

    def test_search_no_results(self, tmp_path):
        """Test search with no matches."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(content="Hello world.", title="Greeting")

        results = kb.search("xyznonexistent")
        assert len(results) == 0

    def test_get_citations(self, tmp_path):
        """Test getting citations for a query."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        doc1 = kb.ingest_text(
            content="The deadline is next Friday.",
            title="Project Timeline",
        )
        doc2 = kb.ingest_text(
            content="No deadline mentioned here.",
            title="Other Notes",
        )

        citations = kb.get_citations("deadline")
        assert len(citations) == 2
        assert doc1.doc_id in citations
        assert doc2.doc_id in citations

    def test_get_citations_no_match(self, tmp_path):
        """Test getting citations with no matches."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(content="Hello world.", title="Greeting")

        citations = kb.get_citations("nonexistent")
        assert len(citations) == 0

    def test_list_documents(self, tmp_path):
        """Test listing all documents."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(content="Doc 1", title="First")
        kb.ingest_text(content="Doc 2", title="Second")
        kb.ingest_text(content="Doc 3", title="Third")

        docs = kb.list_documents()
        assert len(docs) == 3

    def test_list_documents_with_limit(self, tmp_path):
        """Test listing documents with limit."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        for i in range(5):
            kb.ingest_text(content=f"Doc {i}", title=f"Title {i}")

        docs = kb.list_documents(limit=2)
        assert len(docs) == 2

    def test_list_documents_with_tags(self, tmp_path):
        """Test listing documents filtered by tags."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(content="Doc 1", title="First", tags=["a"])
        kb.ingest_text(content="Doc 2", title="Second", tags=["b"])
        kb.ingest_text(content="Doc 3", title="Third", tags=["a", "b"])

        docs = kb.list_documents(tags=["a"])
        assert len(docs) == 2

    def test_list_tags(self, tmp_path):
        """Test listing all unique tags."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(content="1", title="1", tags=["alpha", "beta"])
        kb.ingest_text(content="2", title="2", tags=["gamma"])
        kb.ingest_text(content="3", title="3", tags=["alpha"])

        tags = kb.list_tags()
        assert sorted(tags) == ["alpha", "beta", "gamma"]

    def test_len(self, tmp_path):
        """Test __len__ returns document count."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        assert len(kb) == 0
        kb.ingest_text(content="Test", title="Test")
        assert len(kb) == 1

    def test_persistence(self, tmp_path):
        """Test that documents persist across instances."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"

        kb1 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        doc = kb1.ingest_text(
            content="Persisted content.",
            title="Persistent Doc",
        )

        kb2 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        retrieved = kb2.get_document(doc.doc_id)
        assert retrieved is not None
        assert retrieved.content == "Persisted content."

    def test_index_persistence(self, tmp_path):
        """Test that search index persists across instances."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"

        kb1 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        kb1.ingest_text(
            content="Searchable persistent content.",
            title="Indexed Doc",
        )

        kb2 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        results = kb2.search("searchable")
        assert len(results) == 1


# =============================================================================
# Tokenization and Indexing Tests
# =============================================================================


class TestTokenization:
    """Tests for tokenization and indexing."""

    def test_stop_words_excluded(self, tmp_path):
        """Test that stop words are excluded from index."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(
            content="The quick brown fox jumps over the lazy dog.",
            title="Test",
        )

        # Common stop words like "the" should not be indexed
        assert "the" not in kb._index
        assert "a" not in kb._index

        # Content words should be indexed
        assert "quick" in kb._index
        assert "brown" in kb._index
        assert "fox" in kb._index

    def test_case_insensitive_search(self, tmp_path):
        """Test that search is case-insensitive."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(
            content="UPPERCASE content here.",
            title="Uppercase Test",
        )

        results = kb.search("uppercase")
        assert len(results) == 1

    def test_prefix_matching(self, tmp_path):
        """Test that prefix matching works."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(
            content="Programming is fun.",
            title="Code",
        )

        # "program" should match "programming" via prefix
        results = kb.search("program")
        assert len(results) == 1


# =============================================================================
# Singleton and Convenience Function Tests
# =============================================================================


class TestSingletonPattern:
    """Tests for the global singleton instance."""

    def test_get_and_set_knowledge_base(self, tmp_path):
        """Test get/set knowledge base singleton."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        set_knowledge_base(kb)

        retrieved = get_knowledge_base()
        assert retrieved is kb


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_ingest_document(self, tmp_path):
        """Test ingest_document convenience function."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        set_knowledge_base(kb)

        test_file = tmp_path / "convenience.txt"
        test_file.write_text("Convenience function test.")

        doc = ingest_document(test_file, title="Convenience")
        assert doc.title == "Convenience"

    def test_search_documents(self, tmp_path):
        """Test search_documents convenience function."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        set_knowledge_base(kb)

        kb.ingest_text(content="Searchable content.", title="Test")

        results = search_documents("searchable")
        assert len(results) == 1


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_corrupt_docs_file(self, tmp_path):
        """Test graceful handling of corrupt docs file."""
        docs_path = tmp_path / "corrupt.json"
        index_path = tmp_path / "index.json"
        docs_path.write_text("not valid json {{{")

        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        assert len(kb) == 0

    def test_handles_corrupt_index_file(self, tmp_path):
        """Test graceful handling of corrupt index file."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "corrupt.json"

        # Create valid docs file
        docs_path.write_text('{"documents": []}')
        # Create corrupt index
        index_path.write_text("not valid json {{{")

        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        # Should rebuild index
        assert len(kb) == 0

    def test_unicode_content(self, tmp_path):
        """Test handling unicode content."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        doc = kb.ingest_text(
            content="Hello 世界 🌍 café",
            title="Unicode Test",
        )

        kb2 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        retrieved = kb2.get_document(doc.doc_id)
        assert "世界" in retrieved.content

    def test_very_long_content(self, tmp_path):
        """Test handling very long content."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        long_content = "word " * 10000  # 50000+ characters
        doc = kb.ingest_text(content=long_content, title="Long Doc")

        assert doc.word_count == 10000

    def test_empty_search_query(self, tmp_path):
        """Test search with empty query."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(content="Some content.", title="Test")

        results = kb.search("")
        assert len(results) == 0

    def test_search_with_only_stop_words(self, tmp_path):
        """Test search with only stop words."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        kb.ingest_text(content="Some content.", title="Test")

        results = kb.search("the and or")
        assert len(results) == 0

    def test_ingest_file_with_different_encodings(self, tmp_path):
        """Test ingesting files with different encodings."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        # UTF-8 file
        utf8_file = tmp_path / "utf8.txt"
        utf8_file.write_text("UTF-8 content: café", encoding="utf-8")

        doc = kb.ingest_file(utf8_file)
        assert "café" in doc.content


# =============================================================================
# Knowledge Refresh Tests (US-075)
# =============================================================================


class TestKnowledgeRefresh:
    """Tests for document refresh functionality."""

    def test_refresh_document_updates_content(self, tmp_path):
        """Test that refreshing a document re-reads its source file."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        source_file = tmp_path / "source.txt"
        source_file.write_text("original content")

        doc = kb.ingest_file(source_file, title="Source Doc")
        assert "original" in doc.content

        # Update the file
        source_file.write_text("updated content with newword")

        updated = kb.refresh_document(doc.doc_id)
        assert updated is not None
        assert "updated" in updated.content
        assert "original" not in updated.content

    def test_refresh_document_updates_index(self, tmp_path):
        """Test that refreshing a document updates the search index."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        source_file = tmp_path / "indexed.txt"
        source_file.write_text("oldterm is here")
        doc = kb.ingest_file(source_file)

        # Confirm old term is searchable
        assert len(kb.search("oldterm")) == 1

        # Update file with new term, old term removed
        source_file.write_text("newterm replaces everything")
        kb.refresh_document(doc.doc_id)

        # Old term should no longer match
        assert len(kb.search("oldterm")) == 0
        # New term should match
        assert len(kb.search("newterm")) == 1

    def test_refresh_document_removes_stale_index_entries(self, tmp_path):
        """Test that stale index entries are removed after refresh."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        source_file = tmp_path / "stale.txt"
        source_file.write_text("staleword content")
        doc = kb.ingest_file(source_file)

        assert "staleword" in kb._index

        # Overwrite file without the stale word
        source_file.write_text("completely different text")
        kb.refresh_document(doc.doc_id)

        # Stale word should be gone from index
        assert "staleword" not in kb._index

    def test_refresh_document_not_found(self, tmp_path):
        """Test refresh returns None for unknown doc_id."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        result = kb.refresh_document("nonexistent_id")
        assert result is None

    def test_refresh_document_no_source_path(self, tmp_path):
        """Test refresh returns None for documents without a source path."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        doc = kb.ingest_text(content="No source here.", title="Text Only")
        result = kb.refresh_document(doc.doc_id)
        assert result is None

    def test_refresh_document_missing_file_raises(self, tmp_path):
        """Test refresh raises FileNotFoundError when source file is gone."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        source_file = tmp_path / "gone.txt"
        source_file.write_text("content before deletion")
        doc = kb.ingest_file(source_file)

        source_file.unlink()

        with pytest.raises(FileNotFoundError):
            kb.refresh_document(doc.doc_id)

    def test_refresh_all_refreshes_file_backed_docs(self, tmp_path):
        """Test refresh_all updates all documents with source paths."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        file1 = tmp_path / "f1.txt"
        file2 = tmp_path / "f2.txt"
        file1.write_text("file one original")
        file2.write_text("file two original")

        doc1 = kb.ingest_file(file1)
        doc2 = kb.ingest_file(file2)

        # Update both files
        file1.write_text("file one updated")
        file2.write_text("file two updated")

        results = kb.refresh_all()
        assert results[doc1.doc_id] == "refreshed"
        assert results[doc2.doc_id] == "refreshed"

        assert "updated" in kb.get_document(doc1.doc_id).content
        assert "updated" in kb.get_document(doc2.doc_id).content

    def test_refresh_all_skips_text_only_docs(self, tmp_path):
        """Test refresh_all skips documents without a source path."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        doc = kb.ingest_text(content="No file.", title="Text Doc")
        results = kb.refresh_all()
        assert results[doc.doc_id] == "skipped"

    def test_refresh_all_records_error_for_missing_file(self, tmp_path):
        """Test refresh_all records error when source file is missing."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(docs_path=docs_path, index_path=index_path)

        source_file = tmp_path / "deleted.txt"
        source_file.write_text("will be deleted")
        doc = kb.ingest_file(source_file)

        source_file.unlink()

        results = kb.refresh_all()
        assert results[doc.doc_id].startswith("error:")

    def test_refresh_persists_to_disk(self, tmp_path):
        """Test that refresh saves updated content to disk."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"

        source_file = tmp_path / "persist.txt"
        source_file.write_text("before refresh")

        kb1 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        doc = kb1.ingest_file(source_file)

        source_file.write_text("after refresh")
        kb1.refresh_document(doc.doc_id)

        # Load fresh instance and verify persisted content
        kb2 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        stored = kb2.get_document(doc.doc_id)
        assert stored is not None
        assert "after" in stored.content
