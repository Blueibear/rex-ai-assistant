"""Knowledge base for Rex AI Assistant.

This module provides a knowledge base for ingesting, indexing, and searching
text documents. It enables Rex to reference external documents and provide
citations when answering questions.

Features:
- Ingest text files with metadata (title, tags, source path)
- Full-text search with word-based indexing
- Citation retrieval for referencing documents
- Persistence to JSON files

Usage:
    from rex.knowledge_base import get_knowledge_base

    kb = get_knowledge_base()

    # Ingest a document
    doc = kb.ingest_file("/path/to/document.txt", title="My Doc", tags=["notes"])

    # Search for documents
    results = kb.search("search query", max_results=5)

    # Get citations
    citations = kb.get_citations("specific phrase")
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rex.context.source_policy import (
    AudienceScope,
    ContextSourcePolicyStore,
    ContextSourceType,
    DisclosurePolicy,
)
from rex.identity import validate_user_id

logger = logging.getLogger(__name__)

# Default data directory
_DATA_DIR = Path("data/knowledge_base")


def _ensure_data_dir() -> Path:
    """Ensure the data directory exists."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


# =============================================================================
# Knowledge Document Model
# =============================================================================


class KnowledgeDocument(BaseModel):
    """A document stored in the knowledge base.

    Attributes:
        doc_id: Unique identifier for this document.
        title: Human-readable title.
        content: Full text content of the document.
        source_path: Original file path where the document was ingested from.
        tags: List of tags for categorization.
        created_at: When the document was ingested.
        word_count: Number of words in the document.
    """

    doc_id: str = Field(
        default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this document",
    )
    title: str = Field(
        ...,
        description="Human-readable title",
    )
    content: str = Field(
        ...,
        description="Full text content of the document",
    )
    source_path: str | None = Field(
        default=None,
        description="Original file path where the document was ingested from",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="List of tags for categorization",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="When the document was ingested",
    )
    word_count: int = Field(
        default=0,
        description="Number of words in the document",
    )
    owner_user_id: str | None = Field(
        default=None,
        description="Validated Rex user who owns this upload",
    )
    audience_scope: str = Field(
        default="legacy_unassigned",
        description="private, household, or non-selectable legacy_unassigned",
    )
    context_enabled: bool = Field(
        default=False,
        description="Whether this document may enter broad/background context",
    )
    disclosure_policy: str = Field(
        default=DisclosurePolicy.OWNER_ONLY.value,
        description="Separate disclosure boundary for document-derived information",
    )
    policy_revision: str = Field(
        default="legacy-0",
        description="Content-free document-policy revision token",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize, validate policy metadata, and compute word count."""
        super().__init__(**data)
        if self.word_count == 0 and self.content:
            self.word_count = len(self.content.split())
        if self.owner_user_id is not None:
            self.owner_user_id = validate_user_id(self.owner_user_id)
        if self.audience_scope not in {
            AudienceScope.PRIVATE.value,
            AudienceScope.HOUSEHOLD.value,
            "legacy_unassigned",
        }:
            raise ValueError("Invalid knowledge-document audience scope")
        if self.audience_scope == "legacy_unassigned":
            if self.owner_user_id is not None or self.context_enabled:
                raise ValueError(
                    "Legacy-unassigned documents cannot have an owner or context access"
                )
        elif self.owner_user_id is None:
            raise ValueError("Scoped knowledge documents require an owner")
        if self.disclosure_policy not in {item.value for item in DisclosurePolicy}:
            raise ValueError("Invalid knowledge-document disclosure policy")
        if not isinstance(self.policy_revision, str) or not self.policy_revision.strip():
            raise ValueError("Knowledge-document policy_revision must be non-empty")

    @property
    def source_id(self) -> str:
        """Stable provenance identifier used by context policy and citations."""
        return f"upload:{self.doc_id}"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "doc_id": "doc_abc123def456",
                    "title": "Meeting Notes",
                    "content": "Discussion about project timeline...",
                    "source_path": "/docs/meeting_notes.txt",
                    "tags": ["meetings", "project"],
                    "created_at": "2024-01-15T10:30:00Z",
                    "word_count": 150,
                }
            ]
        }
    }


# =============================================================================
# Knowledge Base
# =============================================================================


class KnowledgeBase:
    """Knowledge base for storing and searching documents.

    Provides document ingestion, full-text indexing, and search capabilities.
    All data is persisted to JSON files.

    Example:
        kb = KnowledgeBase()

        # Ingest a document
        doc = kb.ingest_file(
            "/path/to/notes.txt",
            title="Project Notes",
            tags=["project", "notes"]
        )

        # Search documents
        results = kb.search("deadline", max_results=5)

        # Get citations
        citations = kb.get_citations("project deadline")
    """

    # Words to exclude from the index (common stop words)
    STOP_WORDS = frozenset(
        [
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "can",
            "just",
            "now",
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
        ]
    )

    def __init__(
        self,
        docs_path: Path | str | None = None,
        index_path: Path | str | None = None,
        *,
        source_policy_store: ContextSourcePolicyStore | None = None,
    ) -> None:
        """Initialize the knowledge base.

        Args:
            docs_path: Path to store documents JSON. Defaults to
                data/knowledge_base/docs.json
            index_path: Path to store the search index. Defaults to
                data/knowledge_base/index.json
        """
        if docs_path is None or index_path is None:
            _ensure_data_dir()

        self.docs_path = Path(docs_path) if docs_path else _DATA_DIR / "docs.json"
        self.index_path = Path(index_path) if index_path else _DATA_DIR / "index.json"
        self._source_policy_store = source_policy_store

        self._documents: dict[str, KnowledgeDocument] = {}
        self._index: dict[str, set[str]] = defaultdict(set)

        self._load()

    def _load(self) -> None:
        """Load documents/index and persist safe policy defaults for legacy rows."""
        migrated_legacy_policy = False
        policy_fields = {
            "owner_user_id",
            "audience_scope",
            "context_enabled",
            "disclosure_policy",
            "policy_revision",
        }
        # Load documents
        if self.docs_path.exists():
            try:
                with open(self.docs_path, encoding="utf-8") as f:
                    data = json.load(f)
                    for doc_data in data.get("documents", []):
                        if isinstance(doc_data, dict) and not policy_fields.issubset(doc_data):
                            migrated_legacy_policy = True
                        doc = KnowledgeDocument.model_validate(doc_data)
                        self._documents[doc.doc_id] = doc
                    logger.debug(f"Loaded {len(self._documents)} documents")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load documents: {e}")
                self._documents = {}

        # Load index
        if self.index_path.exists():
            try:
                with open(self.index_path, encoding="utf-8") as f:
                    data = json.load(f)
                    for word, doc_ids in data.get("index", {}).items():
                        self._index[word] = set(doc_ids)
                    logger.debug(f"Loaded index with {len(self._index)} terms")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load index: {e}")
                self._index = defaultdict(set)
                # Rebuild index from documents
                self._rebuild_index()

        if migrated_legacy_policy:
            # Persist fail-closed legacy defaults so later runs do not have to
            # infer whether an unscoped row is safe for background context.
            self._save()

    def _save(self) -> None:
        """Save documents and index to disk."""
        self.docs_path.parent.mkdir(parents=True, exist_ok=True)

        # Save documents
        try:
            docs_data = [doc.model_dump(mode="json") for doc in self._documents.values()]
            with open(self.docs_path, "w", encoding="utf-8") as f:
                json.dump({"documents": docs_data}, f, indent=2, default=str)
        except OSError as e:
            logger.error(f"Failed to save documents: {e}")

        # Save index
        try:
            index_data = {word: list(doc_ids) for word, doc_ids in self._index.items()}
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump({"index": index_data}, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save index: {e}")

    @staticmethod
    def _upload_policy_fields(
        *,
        owner_user_id: str | None,
        audience_scope: str | AudienceScope | None,
        context_enabled: bool | None,
        disclosure_policy: str | DisclosurePolicy | None,
    ) -> dict[str, Any]:
        supplied = any(
            value is not None
            for value in (owner_user_id, audience_scope, context_enabled, disclosure_policy)
        )
        if not supplied:
            # Compatibility path for existing explicit-query callers. These
            # rows are deliberately ineligible for user/background context.
            return {}
        if owner_user_id is None or audience_scope is None or context_enabled is None:
            raise ValueError(
                "New upload policy requires owner_user_id, audience_scope, and context_enabled"
            )
        owner = validate_user_id(owner_user_id)
        audience = AudienceScope(audience_scope)
        if not isinstance(context_enabled, bool):
            raise ValueError("context_enabled must be boolean")
        disclosure = (
            DisclosurePolicy(disclosure_policy)
            if disclosure_policy is not None
            else (
                DisclosurePolicy.HOUSEHOLD
                if audience is AudienceScope.HOUSEHOLD
                else DisclosurePolicy.OWNER_ONLY
            )
        )
        return {
            "owner_user_id": owner,
            "audience_scope": audience.value,
            "context_enabled": context_enabled,
            "disclosure_policy": disclosure.value,
            "policy_revision": f"policy-{uuid.uuid4().hex}",
        }

    def _sync_upload_source_policy(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        store = self._source_policy_store
        if store is None or doc.owner_user_id is None or doc.audience_scope == "legacy_unassigned":
            return doc
        saved = store.register_source(
            doc.source_id,
            ContextSourceType.UPLOAD,
            owner_user_id=doc.owner_user_id,
            audience_scope=AudienceScope(doc.audience_scope),
            context_enabled=doc.context_enabled,
            disclosure_policy=DisclosurePolicy(doc.disclosure_policy),
        )
        revision = f"{saved.audience_scope.value}:{saved.policy_revision}"
        return doc.model_copy(update={"policy_revision": revision})

    @staticmethod
    def _document_visible_to_user(
        doc: KnowledgeDocument,
        requester_user_id: str,
        *,
        context_only: bool,
    ) -> bool:
        if doc.audience_scope == "legacy_unassigned" or doc.owner_user_id is None:
            return False
        if context_only and not doc.context_enabled:
            return False
        if doc.audience_scope == AudienceScope.PRIVATE.value:
            return doc.owner_user_id == requester_user_id
        return doc.audience_scope == AudienceScope.HOUSEHOLD.value

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words, excluding stop words.

        Args:
            text: Text to tokenize.

        Returns:
            List of lowercase word tokens.
        """
        # Split on non-word characters, convert to lowercase
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        # Filter out stop words and very short words
        return [w for w in words if w not in self.STOP_WORDS and len(w) > 1]

    def _index_document(self, doc: KnowledgeDocument) -> None:
        """Add a document to the search index.

        Args:
            doc: The document to index.
        """
        # Index title
        for word in self._tokenize(doc.title):
            self._index[word].add(doc.doc_id)

        # Index content
        for word in self._tokenize(doc.content):
            self._index[word].add(doc.doc_id)

        # Index tags
        for tag in doc.tags:
            for word in self._tokenize(tag):
                self._index[word].add(doc.doc_id)

    def _unindex_document(self, doc_id: str) -> None:
        """Remove a document from the search index.

        Args:
            doc_id: The document ID to remove.
        """
        # Remove doc_id from all index entries
        empty_words = []
        for word, doc_ids in self._index.items():
            doc_ids.discard(doc_id)
            if not doc_ids:
                empty_words.append(word)

        # Clean up empty entries
        for word in empty_words:
            del self._index[word]

    def _rebuild_index(self) -> None:
        """Rebuild the entire search index from documents."""
        self._index = defaultdict(set)
        for doc in self._documents.values():
            self._index_document(doc)
        logger.info(f"Rebuilt index with {len(self._index)} terms")

    def ingest_file(
        self,
        path: str | Path,
        title: str | None = None,
        tags: list[str] | None = None,
        *,
        owner_user_id: str | None = None,
        audience_scope: str | AudienceScope | None = None,
        context_enabled: bool | None = None,
        disclosure_policy: str | DisclosurePolicy | None = None,
    ) -> KnowledgeDocument:
        """Ingest a text file into the knowledge base.

        Args:
            path: Path to the text file to ingest.
            title: Optional title (defaults to filename).
            tags: Optional list of tags.

        Returns:
            The created KnowledgeDocument.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file cannot be read as text.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with fallback encoding
            try:
                content = path.read_text(encoding="latin-1")
            except Exception as e:
                raise ValueError(f"Cannot read file as text: {path}") from e

        policy_fields = self._upload_policy_fields(
            owner_user_id=owner_user_id,
            audience_scope=audience_scope,
            context_enabled=context_enabled,
            disclosure_policy=disclosure_policy,
        )
        doc = KnowledgeDocument(
            title=title or path.stem,
            content=content,
            source_path=str(path.resolve()),
            tags=tags or [],
            **policy_fields,
        )
        doc = self._sync_upload_source_policy(doc)

        self._documents[doc.doc_id] = doc
        self._index_document(doc)
        self._save()

        logger.info(f"Ingested document: {doc.doc_id} - {doc.title}")
        return doc

    def ingest_text(
        self,
        content: str,
        title: str,
        tags: list[str] | None = None,
        source_path: str | None = None,
        *,
        owner_user_id: str | None = None,
        audience_scope: str | AudienceScope | None = None,
        context_enabled: bool | None = None,
        disclosure_policy: str | DisclosurePolicy | None = None,
    ) -> KnowledgeDocument:
        """Ingest text content directly into the knowledge base.

        Args:
            content: The text content to ingest.
            title: Title for the document.
            tags: Optional list of tags.
            source_path: Optional source path for reference.

        Returns:
            The created KnowledgeDocument.
        """
        policy_fields = self._upload_policy_fields(
            owner_user_id=owner_user_id,
            audience_scope=audience_scope,
            context_enabled=context_enabled,
            disclosure_policy=disclosure_policy,
        )
        doc = KnowledgeDocument(
            title=title,
            content=content,
            source_path=source_path,
            tags=tags or [],
            **policy_fields,
        )
        doc = self._sync_upload_source_policy(doc)

        self._documents[doc.doc_id] = doc
        self._index_document(doc)
        self._save()

        logger.info(f"Ingested text document: {doc.doc_id} - {doc.title}")
        return doc

    def get_document(self, doc_id: str) -> KnowledgeDocument | None:
        """Compatibility read for legacy-unassigned explicit-query rows only."""
        doc = self._documents.get(doc_id)
        if doc is None or doc.audience_scope != "legacy_unassigned":
            return None
        return doc

    def get_document_for_user(
        self,
        doc_id: str,
        *,
        requester_user_id: str,
        context_only: bool = False,
    ) -> KnowledgeDocument | None:
        """Return one document only after owner/audience/context authorization."""
        requester = validate_user_id(requester_user_id)
        doc = self._documents.get(doc_id)
        if doc is None or not self._document_visible_to_user(
            doc, requester, context_only=context_only
        ):
            return None
        return doc

    def assign_document_policy(
        self,
        doc_id: str,
        *,
        owner_user_id: str,
        actor_user_id: str,
        audience_scope: str | AudienceScope,
        context_enabled: bool,
        disclosure_policy: str | DisclosurePolicy | None = None,
    ) -> KnowledgeDocument:
        """Assign or update policy; only the asserted owner may authorize it."""
        owner = validate_user_id(owner_user_id)
        actor = validate_user_id(actor_user_id)
        if actor != owner:
            raise PermissionError("owner authorization required")
        doc = self._documents.get(doc_id)
        if doc is None:
            raise KeyError("Knowledge document was not found")
        if doc.owner_user_id is not None and doc.owner_user_id != owner:
            raise PermissionError("owner authorization required")
        fields = self._upload_policy_fields(
            owner_user_id=owner,
            audience_scope=audience_scope,
            context_enabled=context_enabled,
            disclosure_policy=disclosure_policy,
        )
        updated = doc.model_copy(update=fields)
        updated = self._sync_upload_source_policy(updated)
        self._documents[doc_id] = updated
        self._save()
        return updated

    def delete_document(
        self,
        doc_id: str,
        *,
        requester_user_id: str | None = None,
    ) -> bool:
        """Delete legacy rows directly; scoped uploads require their owner."""
        doc = self._documents.get(doc_id)
        if doc is None:
            return False
        if doc.audience_scope != "legacy_unassigned":
            if requester_user_id is None or doc.owner_user_id is None:
                return False
            requester = validate_user_id(requester_user_id)
            if requester != doc.owner_user_id:
                return False

        if (
            self._source_policy_store is not None
            and doc.owner_user_id is not None
            and doc.audience_scope != "legacy_unassigned"
        ):
            try:
                self._source_policy_store.set_context_enabled(
                    doc.owner_user_id,
                    doc.source_id,
                    False,
                )
            except KeyError:
                # Repair a missing policy record as explicitly disabled so the
                # relevant private/household cache revision still changes.
                self._source_policy_store.register_source(
                    doc.source_id,
                    ContextSourceType.UPLOAD,
                    owner_user_id=doc.owner_user_id,
                    audience_scope=AudienceScope(doc.audience_scope),
                    context_enabled=False,
                    disclosure_policy=DisclosurePolicy(doc.disclosure_policy),
                )

        self._unindex_document(doc_id)
        del self._documents[doc_id]
        self._save()

        logger.info(f"Deleted document: {doc_id}")
        return True

    def _score_index_matches(
        self,
        query_terms: list[str],
        allowed_ids: set[str],
    ) -> dict[str, int]:
        scores: dict[str, int] = defaultdict(int)
        for term in query_terms:
            for doc_id in self._index.get(term, set()).intersection(allowed_ids):
                scores[doc_id] += 2
            for indexed_term, doc_ids in self._index.items():
                if indexed_term != term and indexed_term.startswith(term):
                    for doc_id in doc_ids.intersection(allowed_ids):
                        scores[doc_id] += 1
        return scores

    def _add_substring_scores(
        self,
        query: str,
        allowed_ids: set[str],
        scores: dict[str, int],
    ) -> None:
        query_lower = query.lower()
        for doc_id in allowed_ids:
            doc = self._documents.get(doc_id)
            if doc is None:
                continue
            if query_lower in doc.content.lower():
                scores[doc_id] += 3
            if query_lower in doc.title.lower():
                scores[doc_id] += 5

    def _filter_scores_by_tags(
        self,
        scores: dict[str, int],
        tags: list[str] | None,
    ) -> dict[str, int]:
        if not tags:
            return scores
        tag_set = {tag.lower() for tag in tags}
        return {
            doc_id: score
            for doc_id, score in scores.items()
            if doc_id in self._documents
            and tag_set.issubset({tag.lower() for tag in self._documents[doc_id].tags})
        }

    def _search_with_allowed_ids(
        self,
        query: str,
        *,
        allowed_ids: set[str],
        max_results: int,
        tags: list[str] | None,
    ) -> list[KnowledgeDocument]:
        """Score only pre-authorized document IDs; never inspect other content."""
        query_terms = self._tokenize(query)
        if not query_terms or not allowed_ids:
            return []
        doc_scores = self._score_index_matches(query_terms, allowed_ids)
        self._add_substring_scores(query, allowed_ids, doc_scores)
        doc_scores = self._filter_scores_by_tags(doc_scores, tags)
        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        return [
            self._documents[doc_id]
            for doc_id, _score in sorted_docs[:max_results]
            if doc_id in self._documents
        ]

    def search(
        self,
        query: str,
        max_results: int = 5,
        tags: list[str] | None = None,
    ) -> list[KnowledgeDocument]:
        """Compatibility search limited to legacy-unassigned explicit-query rows."""
        allowed_ids = {
            doc.doc_id
            for doc in self._documents.values()
            if doc.audience_scope == "legacy_unassigned"
        }
        return self._search_with_allowed_ids(
            query,
            allowed_ids=allowed_ids,
            max_results=max_results,
            tags=tags,
        )

    def search_for_user(
        self,
        query: str,
        *,
        requester_user_id: str,
        context_only: bool,
        max_results: int = 5,
        tags: list[str] | None = None,
    ) -> list[KnowledgeDocument]:
        """Search only documents authorized for one validated user."""
        requester = validate_user_id(requester_user_id)
        allowed_ids = {
            doc.doc_id
            for doc in self._documents.values()
            if self._document_visible_to_user(doc, requester, context_only=context_only)
        }
        return self._search_with_allowed_ids(
            query,
            allowed_ids=allowed_ids,
            max_results=max_results,
            tags=tags,
        )

    def get_citations(self, query: str) -> list[str]:
        """Get document IDs that can be cited for a query.

        Returns doc_ids of documents containing the query text. This is
        a simple implementation that returns doc_ids only; future versions
        may include position information.

        Args:
            query: The text to find citations for.

        Returns:
            List of doc_ids that contain the query text.
        """
        query_lower = query.lower()
        citations = []

        for doc_id, doc in self._documents.items():
            if doc.audience_scope != "legacy_unassigned":
                continue
            if query_lower in doc.content.lower() or query_lower in doc.title.lower():
                citations.append(doc_id)

        return citations

    def list_documents_for_user(
        self,
        requester_user_id: str,
        *,
        context_only: bool = False,
        limit: int | None = None,
    ) -> list[KnowledgeDocument]:
        """List only documents visible to one user before exposing metadata."""
        requester = validate_user_id(requester_user_id)
        docs = [
            doc
            for doc in self._documents.values()
            if self._document_visible_to_user(doc, requester, context_only=context_only)
        ]
        docs.sort(key=lambda doc: doc.created_at, reverse=True)
        return docs[:limit] if limit is not None else docs

    def list_documents(
        self,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> list[KnowledgeDocument]:
        """List all documents, optionally filtered by tags.

        Args:
            tags: Optional tag filter.
            limit: Maximum number of documents to return.

        Returns:
            List of KnowledgeDocument objects.
        """
        docs = [
            doc for doc in self._documents.values() if doc.audience_scope == "legacy_unassigned"
        ]

        if tags:
            tag_set = {t.lower() for t in tags}
            docs = [d for d in docs if tag_set.issubset({t.lower() for t in d.tags})]

        # Sort by created_at descending
        docs.sort(key=lambda d: d.created_at, reverse=True)

        if limit:
            docs = docs[:limit]

        return docs

    def refresh_document(
        self,
        doc_id: str,
        *,
        requester_user_id: str | None = None,
    ) -> KnowledgeDocument | None:
        """Refresh legacy rows directly; scoped uploads require their owner.

        Reads the current content of the document's source file, updates the
        stored content, and rebuilds the index entries for that document.
        Stale index entries for the old content are removed.

        Args:
            doc_id: The document ID to refresh.

        Returns:
            The updated KnowledgeDocument, or None if the document does not
            exist or has no source path.

        Raises:
            FileNotFoundError: If the source file no longer exists.
        """
        doc = self._documents.get(doc_id)
        if doc is None:
            logger.warning(f"refresh_document: document not found: {doc_id}")
            return None
        if doc.audience_scope != "legacy_unassigned":
            if requester_user_id is None or doc.owner_user_id is None:
                return None
            requester = validate_user_id(requester_user_id)
            if requester != doc.owner_user_id:
                return None

        if not doc.source_path:
            logger.warning(f"refresh_document: no source path for document: {doc_id}")
            return None

        source = Path(doc.source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        try:
            new_content = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            new_content = source.read_text(encoding="latin-1")

        # Remove stale index entries for this document
        self._unindex_document(doc_id)

        # Update content and word count
        updated = doc.model_copy(
            update={
                "content": new_content,
                "word_count": len(new_content.split()),
            }
        )
        self._documents[doc_id] = updated

        # Re-index with new content
        self._index_document(updated)
        self._save()

        logger.info(f"Refreshed document: {doc_id} - {updated.title}")
        return updated

    def refresh_all(self) -> dict[str, str]:
        """Refresh all documents that have a source path.

        Re-reads each document's source file, updates content, and rebuilds
        index entries. Documents without a source path are skipped. Documents
        whose source file has been deleted are recorded as errors.

        Returns:
            Dict mapping doc_id to status string: "refreshed", "skipped",
            or "error: <reason>".
        """
        results: dict[str, str] = {}
        for doc_id, doc in list(self._documents.items()):
            if doc.audience_scope != "legacy_unassigned":
                continue
            if not doc.source_path:
                results[doc_id] = "skipped"
                continue
            try:
                self.refresh_document(doc_id)
                results[doc_id] = "refreshed"
            except FileNotFoundError as exc:
                logger.warning(f"refresh_all: {exc}")
                results[doc_id] = f"error: {exc}"
            except Exception as exc:  # pragma: no cover
                logger.error(f"refresh_all: unexpected error for {doc_id}: {exc}")
                results[doc_id] = f"error: {exc}"
        return results

    def list_tags(self) -> list[str]:
        """List all unique tags across all documents.

        Returns:
            Sorted list of tag names.
        """
        tags = set()
        for doc in self._documents.values():
            if doc.audience_scope == "legacy_unassigned":
                tags.update(doc.tags)
        return sorted(tags)

    def __len__(self) -> int:
        """Return the number of legacy documents visible to compatibility callers."""
        return sum(
            1 for doc in self._documents.values() if doc.audience_scope == "legacy_unassigned"
        )


# =============================================================================
# Global Instance (Singleton Pattern)
# =============================================================================

_knowledge_base: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """Get the process-warm mutable knowledge index singleton.

    The mutable index is intentionally excluded from warm-cache eviction/accounting:
    callers retain and mutate it, so evicting the manager's reference would neither be
    use-safe nor reclaim memory truthfully.
    """
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(source_policy_store=ContextSourcePolicyStore())
    return _knowledge_base


def set_knowledge_base(kb: KnowledgeBase | None) -> None:
    """Replace the process-warm mutable knowledge index (primarily for tests)."""
    global _knowledge_base
    _knowledge_base = kb


# =============================================================================
# Convenience Functions
# =============================================================================


def ingest_document(
    path: str | Path,
    title: str | None = None,
    tags: list[str] | None = None,
) -> KnowledgeDocument:
    """Convenience function to ingest a file.

    Args:
        path: Path to the text file.
        title: Optional title.
        tags: Optional tags.

    Returns:
        The created KnowledgeDocument.
    """
    kb = get_knowledge_base()
    return kb.ingest_file(path, title=title, tags=tags)


def search_documents(query: str, max_results: int = 5) -> list[KnowledgeDocument]:
    """Convenience function to search documents.

    Args:
        query: Search query.
        max_results: Maximum results.

    Returns:
        List of matching documents.
    """
    kb = get_knowledge_base()
    return kb.search(query, max_results=max_results)


__all__ = [
    "KnowledgeDocument",
    "KnowledgeBase",
    "get_knowledge_base",
    "set_knowledge_base",
    "ingest_document",
    "search_documents",
]
