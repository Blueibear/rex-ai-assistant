"""Tests for US-074: Document Indexing.

Acceptance Criteria:
- documents indexed
- index stored
- indexing failures logged
- Typecheck passes
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from rex.knowledge_base import KnowledgeBase


class TestDocumentsIndexed:
    """AC: documents indexed."""

    def test_ingest_indexes_document(self, tmp_path: Path) -> None:
        """Ingested document is added to the index."""
        kb = KnowledgeBase(
            docs_path=tmp_path / "docs.json",
            index_path=tmp_path / "index.json",
        )
        kb.ingest_text(content="Unique banana keyword here.", title="Fruit Doc")

        assert "banana" in kb._index
        assert len(kb._index["banana"]) == 1

    def test_ingest_file_indexes_content(self, tmp_path: Path) -> None:
        """File ingestion indexes the file content."""
        kb = KnowledgeBase(
            docs_path=tmp_path / "docs.json",
            index_path=tmp_path / "index.json",
        )
        test_file = tmp_path / "note.txt"
        test_file.write_text("Rocket science is complex.", encoding="utf-8")

        doc = kb.ingest_file(test_file)
        assert "rocket" in kb._index
        assert doc.doc_id in kb._index["rocket"]

    def test_index_contains_title_terms(self, tmp_path: Path) -> None:
        """Title words are added to the index."""
        kb = KnowledgeBase(
            docs_path=tmp_path / "docs.json",
            index_path=tmp_path / "index.json",
        )
        kb.ingest_text(content="Some content.", title="Quantum Physics Guide")

        assert "quantum" in kb._index
        assert "physics" in kb._index

    def test_search_returns_indexed_document(self, tmp_path: Path) -> None:
        """Search finds a document that was indexed."""
        kb = KnowledgeBase(
            docs_path=tmp_path / "docs.json",
            index_path=tmp_path / "index.json",
        )
        kb.ingest_text(content="Machine learning fundamentals.", title="ML Basics")

        results = kb.search("learning")
        assert len(results) == 1
        assert results[0].title == "ML Basics"

    def test_delete_removes_from_index(self, tmp_path: Path) -> None:
        """Deleting a document removes it from the index."""
        kb = KnowledgeBase(
            docs_path=tmp_path / "docs.json",
            index_path=tmp_path / "index.json",
        )
        doc = kb.ingest_text(content="Chocolate fudge recipe.", title="Desserts")
        assert "chocolate" in kb._index

        kb.delete_document(doc.doc_id)
        assert "chocolate" not in kb._index


class TestIndexStored:
    """AC: index stored."""

    def test_index_written_to_disk_after_ingest(self, tmp_path: Path) -> None:
        """After ingestion the index file exists on disk."""
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(
            docs_path=tmp_path / "docs.json",
            index_path=index_path,
        )
        kb.ingest_text(content="Persistent index test.", title="Test Doc")

        assert index_path.exists()

    def test_index_file_contains_valid_json(self, tmp_path: Path) -> None:
        """The stored index file is valid JSON."""
        index_path = tmp_path / "index.json"
        kb = KnowledgeBase(
            docs_path=tmp_path / "docs.json",
            index_path=index_path,
        )
        kb.ingest_text(content="Valid json index.", title="JSON Test")

        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "index" in data
        assert isinstance(data["index"], dict)

    def test_index_persists_across_instances(self, tmp_path: Path) -> None:
        """Search index survives creating a new KnowledgeBase instance."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"

        kb1 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        kb1.ingest_text(content="Ephemeral aurora borealis.", title="Lights")

        # New instance loads from disk
        kb2 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        results = kb2.search("aurora")
        assert len(results) == 1

    def test_index_rebuilds_when_index_file_missing(self, tmp_path: Path) -> None:
        """If only docs exist (no index), index is rebuilt from documents."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"

        # Populate via first instance (writes both files)
        kb1 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        kb1.ingest_text(content="Zebra migration patterns.", title="Wildlife")

        # Remove index file to simulate missing index
        index_path.unlink()

        # New instance should rebuild index from docs
        kb2 = KnowledgeBase(docs_path=docs_path, index_path=index_path)
        results = kb2.search("zebra")
        assert len(results) == 1


class TestIndexingFailuresLogged:
    """AC: indexing failures logged."""

    def test_corrupt_docs_file_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A corrupt docs file triggers a warning log."""
        docs_path = tmp_path / "docs.json"
        docs_path.write_text("not valid json {{{{", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="rex.knowledge_base"):
            KnowledgeBase(
                docs_path=docs_path,
                index_path=tmp_path / "index.json",
            )

        assert any("Failed to load documents" in r.message for r in caplog.records)

    def test_corrupt_index_file_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A corrupt index file triggers a warning log."""
        docs_path = tmp_path / "docs.json"
        index_path = tmp_path / "index.json"
        docs_path.write_text('{"documents": []}', encoding="utf-8")
        index_path.write_text("not valid json {{{{", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="rex.knowledge_base"):
            KnowledgeBase(docs_path=docs_path, index_path=index_path)

        assert any("Failed to load index" in r.message for r in caplog.records)

    def test_save_failure_logs_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An OSError during save is logged as an error."""
        kb = KnowledgeBase(
            docs_path=tmp_path / "docs.json",
            index_path=tmp_path / "index.json",
        )

        with caplog.at_level(logging.ERROR, logger="rex.knowledge_base"):
            with patch("builtins.open", side_effect=OSError("disk full")):
                kb.ingest_text(content="Failure test.", title="Fail Doc")

        assert any("Failed to save" in r.message for r in caplog.records)
