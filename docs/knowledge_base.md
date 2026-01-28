# Knowledge Base

Rex includes a knowledge base for ingesting, indexing, and searching documents. This enables Rex to reference external documents and provide citations when answering questions.

## Overview

The knowledge base provides:

- **Document Ingestion** - Read and index text files
- **Full-Text Search** - Find documents by content, title, or tags
- **Citation Support** - Reference source documents
- **Persistence** - Documents survive restarts

## Getting Started

### Ingest a Document

```bash
# Ingest a text file
rex kb ingest /path/to/document.txt

# Ingest with custom title
rex kb ingest /path/to/notes.txt --title "Meeting Notes"

# Ingest with tags
rex kb ingest /path/to/project.txt --title "Project Docs" --tags project,docs
```

### Search Documents

```bash
# Search for documents
rex kb search "project deadline"

# Search with content snippets
rex kb search "project deadline" -v

# Limit results
rex kb search "project" --max-results 10
```

### View and Manage

```bash
# List all documents
rex kb list

# Show document content
rex kb show doc_abc123

# Delete a document
rex kb delete doc_abc123

# List all tags
rex kb tags
```

## Document Structure

Each document contains:

| Field | Description |
|-------|-------------|
| `doc_id` | Unique identifier (e.g., `doc_abc123`) |
| `title` | Human-readable title |
| `content` | Full text content |
| `source_path` | Original file path |
| `tags` | List of tags for categorization |
| `created_at` | When the document was ingested |
| `word_count` | Number of words |

## Python API

### Basic Usage

```python
from rex.knowledge_base import get_knowledge_base

kb = get_knowledge_base()

# Ingest a file
doc = kb.ingest_file("/path/to/file.txt", title="My Document")

# Ingest text directly
doc = kb.ingest_text(
    content="Full text content here...",
    title="Direct Text",
    tags=["notes", "project"]
)

# Search
results = kb.search("query", max_results=5)
for doc in results:
    print(f"{doc.doc_id}: {doc.title}")

# Get a specific document
doc = kb.get_document("doc_abc123")

# Delete a document
kb.delete_document("doc_abc123")
```

### Advanced Search

```python
# Search with tag filter
results = kb.search("deadline", tags=["project"])

# List all documents
docs = kb.list_documents()

# List with tag filter
docs = kb.list_documents(tags=["notes"])

# List with limit
docs = kb.list_documents(limit=10)
```

### Citations

Get document IDs that can be cited for a query:

```python
# Get citations
citations = kb.get_citations("project deadline")
# Returns: ["doc_abc123", "doc_def456"]

for doc_id in citations:
    doc = kb.get_document(doc_id)
    print(f"Source: {doc.title}")
```

## Search Behavior

### Indexing

Documents are indexed by:
- Title words
- Content words
- Tag words

Common stop words (the, and, or, etc.) are excluded from the index.

### Matching

Search uses multiple matching strategies:

1. **Exact word match** - Words matching index terms (2 points)
2. **Prefix match** - Words starting with query term (1 point)
3. **Substring match in content** - Query found in content (3 points)
4. **Substring match in title** - Query found in title (5 points)

Results are ranked by total score.

### Example

```
Query: "python programming"

Document A - Title: "Python Programming Guide"
  - "python" in title: +5 points
  - "programming" in title: +5 points
  - Total: 10 points

Document B - Title: "General Guide"
  - Content contains "python programming": +3 points
  - "python" in content index: +2 points
  - Total: 5 points

Result: Document A ranks higher
```

## CLI Commands

### Ingest

```bash
# Basic ingest
rex kb ingest /path/to/file.txt

# With title
rex kb ingest /path/to/file.txt --title "Custom Title"

# With tags (comma-separated)
rex kb ingest /path/to/file.txt --tags tag1,tag2,tag3
```

### Search

```bash
# Basic search
rex kb search "query"

# Verbose (show snippets)
rex kb search "query" -v

# Limit results
rex kb search "query" --max-results 3
```

### List

```bash
# List all documents
rex kb list

# Limit listing
rex kb list --limit 10
```

### Show

```bash
# Show full document content
rex kb show doc_abc123
```

### Delete

```bash
# Delete a document
rex kb delete doc_abc123
```

### Citations

```bash
# Get documents that cite a phrase
rex kb cite "specific phrase"
```

### Tags

```bash
# List all unique tags
rex kb tags
```

## Supported File Types

Currently, the knowledge base supports:

- **Text files** (.txt, .md, .rst, etc.)
- Any file that can be read as UTF-8 or Latin-1 text

Binary files and PDFs are not currently supported.

## Storage

Documents and index are persisted to JSON files:

- **Documents**: `data/knowledge_base/docs.json`
- **Search Index**: `data/knowledge_base/index.json`

### Document Storage Format

```json
{
  "documents": [
    {
      "doc_id": "doc_abc123",
      "title": "Meeting Notes",
      "content": "Full text content...",
      "source_path": "/path/to/original.txt",
      "tags": ["meetings", "project"],
      "created_at": "2024-01-15T10:30:00Z",
      "word_count": 150
    }
  ]
}
```

### Index Storage Format

```json
{
  "index": {
    "meeting": ["doc_abc123", "doc_def456"],
    "project": ["doc_abc123"],
    "notes": ["doc_abc123"]
  }
}
```

## Best Practices

### Organizing Documents

1. **Use descriptive titles** - Helps with search and identification
2. **Apply consistent tags** - Group related documents
3. **Keep documents focused** - One topic per document
4. **Update regularly** - Remove outdated content

### Tagging Strategy

| Tag Category | Examples |
|--------------|----------|
| Type | `notes`, `docs`, `reference`, `guide` |
| Project | `project-alpha`, `website`, `backend` |
| Status | `active`, `archived`, `draft` |
| Topic | `python`, `deployment`, `security` |

### Search Tips

1. **Use specific terms** - "python function" vs "code"
2. **Try variations** - "deploy" and "deployment"
3. **Filter by tags** - Use `--tags` for CLI filtering
4. **Check citations** - Use `cite` for exact phrase matching

## Integration Example

Integrating the knowledge base with Rex's assistant:

```python
from rex.knowledge_base import get_knowledge_base

class Assistant:
    def __init__(self):
        self.kb = get_knowledge_base()

    async def answer_with_sources(self, question: str) -> str:
        # Search knowledge base
        docs = self.kb.search(question, max_results=3)

        if not docs:
            return "I don't have information on that topic."

        # Build context from documents
        context = "\n\n".join([
            f"From '{doc.title}':\n{doc.content[:500]}..."
            for doc in docs
        ])

        # Get citations
        citations = [f"[{doc.doc_id}] {doc.title}" for doc in docs]

        # Generate answer using LLM with context
        answer = await self.generate_with_context(question, context)

        # Append sources
        sources = "\n\nSources:\n" + "\n".join(citations)
        return answer + sources
```

## API Reference

### KnowledgeBase

| Method | Description |
|--------|-------------|
| `ingest_file(path, title, tags)` | Ingest a text file |
| `ingest_text(content, title, tags, source_path)` | Ingest text directly |
| `get_document(doc_id)` | Get document by ID |
| `delete_document(doc_id)` | Delete document |
| `search(query, max_results, tags)` | Search documents |
| `get_citations(query)` | Get doc IDs containing query |
| `list_documents(tags, limit)` | List all documents |
| `list_tags()` | List all unique tags |

### KnowledgeDocument

| Attribute | Type | Description |
|-----------|------|-------------|
| `doc_id` | str | Unique identifier |
| `title` | str | Document title |
| `content` | str | Full text content |
| `source_path` | str | Original file path |
| `tags` | list[str] | List of tags |
| `created_at` | datetime | Creation timestamp |
| `word_count` | int | Word count |

### Convenience Functions

| Function | Description |
|----------|-------------|
| `ingest_document(path, title, tags)` | Ingest a file |
| `search_documents(query, max_results)` | Search documents |

## Limitations

- Text files only (no PDF, Word, etc.)
- Simple word-based indexing (no semantic search)
- No document versioning
- Single-user (no access control)

## Future Improvements

Planned enhancements:
- PDF and Office document support
- Semantic search with embeddings
- Document versioning and history
- Multi-user access control
- Web page ingestion
