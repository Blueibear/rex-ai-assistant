"""Memory, remember, and knowledge-base commands for the Rex CLI.

Extracted verbatim from ``rex/cli.py`` (US-REM-027). Handler behavior,
argument definitions, help text, defaults, and exit codes are unchanged.
"""

from __future__ import annotations

import argparse

from rex.commands._helpers import _parse_ttl


def _cli():
    """Return the ``rex.cli`` module at call time.

    ``rex.cli`` is the single patch point for service getters and command
    handlers (tests monkeypatch ``rex.cli.<name>``). Resolving through the
    module at call time preserves that behavior without creating an import
    cycle at module load time.
    """
    from rex import cli

    return cli


def cmd_remember(args: argparse.Namespace) -> int:
    """Store or display a remembered fact for the active user."""
    from rex.user_facts import recall_all, store

    user = getattr(args, "user", None) or "default"
    fact_text: str | None = getattr(args, "fact", None)

    if fact_text:
        # Parse free-form text like 'My dog is named Max' into a key-value pair.
        # If --key/--value are given explicitly, use those; otherwise auto-parse.
        key: str | None = getattr(args, "key", None)
        value: str | None = getattr(args, "value", None)

        if key and value:
            store(user, key, value)
            print(f"Remembered: {key} = {value} (for user '{user}')")
        else:
            # Auto-generate a key from a hash of the content and store full text.
            import hashlib

            key = "fact_" + hashlib.md5(fact_text.encode()).hexdigest()[:8]
            store(user, key, fact_text)
            print(f"Remembered: {fact_text!r} (for user '{user}')")
    else:
        # Display all stored facts.
        facts = recall_all(user)
        if not facts:
            print(f"No remembered facts for user '{user}'.")
        else:
            print(f"Remembered facts for user '{user}':")
            for k, v in facts.items():
                print(f"  {k} = {v}")
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    """Manage working and long-term memory."""
    import json

    from rex.memory import get_long_term_memory, get_working_memory

    subcommand = args.memory_command

    if subcommand == "recent":
        wm = get_working_memory()
        n = int(getattr(args, "count", 10))
        entries = wm.get_recent_with_timestamps(n)

        if not entries:
            print("No working memory entries.")
            return 0

        print("Recent Working Memory")
        print("=" * 60)
        print()

        for entry in entries:
            timestamp = entry.get("timestamp", "unknown")
            content = entry.get("content", "")
            print(f"[{timestamp}]")
            print(f"  {content}")
            print()

        print(f"Total: {len(entries)} entries shown (of {len(wm)})")
        return 0

    if subcommand == "add":
        ltm = get_long_term_memory()
        category = args.category
        try:
            content = json.loads(args.content)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON content: {e}")
            return 1

        expires_in = None
        if args.ttl:
            expires_in = _parse_ttl(args.ttl)
            if expires_in is None:
                print(f"Error: Invalid TTL format: {args.ttl}")
                print("Use formats like: 7d, 24h, 30m, 1w, 10s")
                return 1

        entry = ltm.add_entry(  # type: ignore[assignment]
            category=category,
            content=content,
            expires_in=expires_in,
            sensitive=args.sensitive,
        )

        print(f"Added memory entry: {entry.entry_id}")  # type: ignore[attr-defined]
        print(f"  Category: {entry.category}")  # type: ignore[attr-defined]
        print(f"  Expires: {entry.expires_at or 'never'}")  # type: ignore[attr-defined]
        print(f"  Sensitive: {entry.sensitive}")  # type: ignore[attr-defined]
        return 0

    if subcommand == "search":
        ltm = get_long_term_memory()
        keyword = args.keyword
        category = args.category
        show_sensitive = args.show_sensitive

        results = ltm.search(
            category=category,
            keyword=keyword,
            include_sensitive=True,
        )

        if not results:
            print("No matching memory entries found.")
            return 0

        print("Long-Term Memory Search Results")
        print("=" * 60)
        print()

        for entry in results:  # type: ignore[assignment]
            print(f"{entry.entry_id} [{entry.category}]")  # type: ignore[attr-defined]
            print(f"  Created: {entry.created_at}")  # type: ignore[attr-defined]
            if entry.expires_at:  # type: ignore[attr-defined]
                print(f"  Expires: {entry.expires_at}")  # type: ignore[attr-defined]
            if entry.sensitive:  # type: ignore[attr-defined]
                print("  [SENSITIVE]")
                if show_sensitive:
                    print(f"  Content: {json.dumps(entry.content, indent=4)}")  # type: ignore[attr-defined]
                else:
                    print("  Content: <hidden - use --show-sensitive>")
            else:
                print(f"  Content: {json.dumps(entry.content, indent=4)}")  # type: ignore[attr-defined]
            print()

        print(f"Total: {len(results)} entries found")
        return 0

    if subcommand == "forget":
        ltm = get_long_term_memory()
        entry_id = args.entry_id
        if ltm.forget(entry_id):
            print(f"Deleted memory entry: {entry_id}")
            return 0
        print(f"Error: Entry not found: {entry_id}")
        return 1

    if subcommand == "clear":
        wm = get_working_memory()
        count = len(wm)
        wm.clear()
        print(f"Cleared {count} working memory entries.")
        return 0

    if subcommand == "retention":
        ltm = get_long_term_memory()
        deleted = ltm.run_retention_policy()
        print(f"Retention policy deleted {deleted} expired entries.")
        return 0

    if subcommand == "stats":
        wm = get_working_memory()
        ltm = get_long_term_memory()

        print("Memory Statistics")
        print("=" * 60)
        print()
        print(f"Working Memory: {len(wm)} entries")
        print(f"Long-Term Memory: {len(ltm)} entries")
        print()

        counts = ltm.count_by_category()
        if counts:
            print("Long-Term Memory by Category:")
            for cat, count in sorted(counts.items()):
                print(f"  {cat}: {count}")

        return 0

    print("Unknown memory subcommand. Use 'rex memory --help'")
    return 1


def cmd_kb(args: argparse.Namespace) -> int:
    """Manage knowledge base."""
    from rex.knowledge_base import get_knowledge_base

    kb = get_knowledge_base()
    subcommand = args.kb_command

    if subcommand == "ingest":
        path = args.path
        title = args.title
        tags = args.tags.split(",") if args.tags else None

        try:
            doc = kb.ingest_file(path, title=title, tags=tags)
        except FileNotFoundError:
            print(f"Error: File not found: {path}")
            return 1
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        print(f"Ingested document: {doc.doc_id}")
        print(f"  Title: {doc.title}")
        print(f"  Words: {doc.word_count}")
        if doc.tags:
            print(f"  Tags: {', '.join(doc.tags)}")
        return 0

    if subcommand == "search":
        query = args.query
        max_results = args.max_results or 5
        results = kb.search(query, max_results=max_results)

        if not results:
            print("No matching documents found.")
            return 0

        print("Knowledge Base Search Results")
        print("=" * 60)
        print()

        for doc in results:
            print(f"{doc.doc_id}: {doc.title}")
            if doc.tags:
                print(f"  Tags: {', '.join(doc.tags)}")
            print(f"  Words: {doc.word_count}")
            if getattr(args, "verbose", False):
                snippet = doc.content[:200].replace("\n", " ")
                if len(doc.content) > 200:
                    snippet += "..."
                print(f"  Snippet: {snippet}")
            print()

        print(f"Total: {len(results)} documents found")
        return 0

    if subcommand == "list":
        limit = args.limit or 20
        docs = kb.list_documents(limit=limit)

        if not docs:
            print("No documents in knowledge base.")
            return 0

        print("Knowledge Base Documents")
        print("=" * 60)
        print()

        for doc in docs:
            print(f"{doc.doc_id}: {doc.title}")
            if doc.tags:
                print(f"  Tags: {', '.join(doc.tags)}")
            print(f"  Words: {doc.word_count}")
            print(f"  Created: {doc.created_at}")
            print()

        print(f"Total: {len(kb)} documents")
        return 0

    if subcommand == "show":
        doc_id = args.doc_id
        doc = kb.get_document(doc_id)  # type: ignore[assignment]

        if not doc:
            print(f"Error: Document not found: {doc_id}")
            return 1

        print(f"Document: {doc.doc_id}")
        print("=" * 60)
        print(f"Title: {doc.title}")
        if doc.source_path:
            print(f"Source: {doc.source_path}")
        if doc.tags:
            print(f"Tags: {', '.join(doc.tags)}")
        print(f"Words: {doc.word_count}")
        print(f"Created: {doc.created_at}")
        print()
        print("Content:")
        print("-" * 60)
        print(doc.content)
        return 0

    if subcommand == "delete":
        doc_id = args.doc_id
        if kb.delete_document(doc_id):
            print(f"Deleted document: {doc_id}")
            return 0
        print(f"Error: Document not found: {doc_id}")
        return 1

    if subcommand == "cite":
        query = args.query
        citations = kb.get_citations(query)

        if not citations:
            print("No citations found.")
            return 0

        print("Citations")
        print("=" * 60)
        print()

        for doc_id in citations:
            doc = kb.get_document(doc_id)  # type: ignore[assignment]
            if doc:
                print(f"{doc_id}: {doc.title}")

        print()
        print(f"Total: {len(citations)} documents can be cited")
        return 0

    if subcommand == "tags":
        tags = kb.list_tags()

        if not tags:
            print("No tags in knowledge base.")
            return 0

        print("Knowledge Base Tags")
        print("=" * 60)
        print()
        for tag in tags:
            print(f"  {tag}")
        print()
        print(f"Total: {len(tags)} tags")
        return 0

    print("Unknown kb subcommand. Use 'rex kb --help'")
    return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
    # memory
    memory_parser = subparsers.add_parser(
        "memory",
        help="Manage working and long-term memory",
        description="Manage Rex's working memory (recent context) and long-term structured memory.",
    )
    memory_subparsers = memory_parser.add_subparsers(
        title="memory commands",
        dest="memory_command",
        metavar="COMMAND",
    )

    memory_recent = memory_subparsers.add_parser(
        "recent",
        help="Show recent working memory entries",
        description="Display the most recent entries from working memory.",
    )
    memory_recent.add_argument(
        "count", type=int, nargs="?", default=10, help="Number of entries to show (default: 10)"
    )
    memory_recent.add_argument(
        "--show-sensitive", action="store_true", help="No-op (kept for compatibility)"
    )
    memory_recent.set_defaults(func=_cli().cmd_memory, memory_command="recent")

    memory_add = memory_subparsers.add_parser(
        "add",
        help="Add a long-term memory entry",
        description="Add a new entry to long-term memory with category and JSON content.",
    )
    memory_add.add_argument(
        "category", type=str, help="Category for the entry (e.g., preferences, facts)"
    )
    memory_add.add_argument(
        "content", type=str, help='JSON content for the entry (e.g., \'{"key": "value"}\')'
    )
    memory_add.add_argument("--ttl", type=str, help="Time to live (e.g., 7d, 24h, 30m, 1w, 10s)")
    memory_add.add_argument(
        "--sensitive", action="store_true", help="Mark this entry as containing sensitive data"
    )
    memory_add.set_defaults(func=_cli().cmd_memory, memory_command="add")

    memory_search = memory_subparsers.add_parser(
        "search",
        help="Search long-term memory",
        description="Search for entries in long-term memory by keyword or category.",
    )
    memory_search.add_argument(
        "keyword", type=str, nargs="?", help="Keyword to search for in content"
    )
    memory_search.add_argument("--category", type=str, help="Filter by category")
    memory_search.add_argument(
        "--show-sensitive", action="store_true", help="Show content of sensitive entries"
    )
    memory_search.set_defaults(func=_cli().cmd_memory, memory_command="search")

    memory_forget = memory_subparsers.add_parser(
        "forget",
        help="Delete a memory entry",
        description="Delete a specific entry from long-term memory.",
    )
    memory_forget.add_argument("entry_id", type=str, help="ID of the entry to delete")
    memory_forget.set_defaults(func=_cli().cmd_memory, memory_command="forget")

    memory_clear = memory_subparsers.add_parser(
        "clear",
        help="Clear all working memory",
        description="Remove all entries from working memory.",
    )
    memory_clear.set_defaults(func=_cli().cmd_memory, memory_command="clear")

    memory_retention = memory_subparsers.add_parser(
        "retention",
        help="Run retention policy",
        description="Delete all expired entries from long-term memory.",
    )
    memory_retention.set_defaults(func=_cli().cmd_memory, memory_command="retention")

    memory_stats = memory_subparsers.add_parser(
        "stats",
        help="Show memory statistics",
        description="Display statistics about working and long-term memory.",
    )
    memory_stats.set_defaults(func=_cli().cmd_memory, memory_command="stats")

    memory_parser.set_defaults(func=_cli().cmd_memory, memory_command="stats")

    # remember
    remember_parser = subparsers.add_parser(
        "remember",
        help="Store a fact for the active user",
        description=(
            "Store a remembered fact so Rex can recall it during conversations. "
            "Example: rex remember 'My dog is named Max'"
        ),
    )
    remember_parser.add_argument(
        "fact",
        nargs="?",
        default=None,
        help="Free-form fact to remember (e.g. 'My dog is named Max')",
    )
    remember_parser.add_argument(
        "--user",
        default="default",
        help="User identifier to store the fact for (default: 'default')",
    )
    remember_parser.add_argument("--key", default=None, help="Explicit key name")
    remember_parser.add_argument("--value", default=None, help="Explicit value (requires --key)")
    remember_parser.set_defaults(func=_cli().cmd_remember)

    # kb
    kb_parser = subparsers.add_parser(
        "kb",
        help="Manage knowledge base",
        description="Manage Rex's knowledge base for document storage and retrieval.",
    )
    kb_subparsers = kb_parser.add_subparsers(
        title="knowledge base commands",
        dest="kb_command",
        metavar="COMMAND",
    )

    kb_ingest = kb_subparsers.add_parser(
        "ingest",
        help="Ingest a file into the knowledge base",
        description="Read and index a text file for later search and retrieval.",
    )
    kb_ingest.add_argument("path", type=str, help="Path to the file to ingest")
    kb_ingest.add_argument(
        "--title", type=str, help="Title for the document (defaults to filename)"
    )
    kb_ingest.add_argument("--tags", type=str, help="Comma-separated list of tags")
    kb_ingest.set_defaults(func=_cli().cmd_kb, kb_command="ingest")

    kb_search = kb_subparsers.add_parser(
        "search",
        help="Search the knowledge base",
        description="Search for documents matching a query.",
    )
    kb_search.add_argument("query", type=str, help="Search query")
    kb_search.add_argument(
        "--max-results", type=int, default=5, help="Maximum number of results (default: 5)"
    )
    kb_search.add_argument("-v", "--verbose", action="store_true", help="Show content snippets")
    kb_search.set_defaults(func=_cli().cmd_kb, kb_command="search")

    kb_list = kb_subparsers.add_parser(
        "list",
        help="List all documents",
        description="List all documents in the knowledge base.",
    )
    kb_list.add_argument(
        "--limit", type=int, default=20, help="Maximum number of documents to list (default: 20)"
    )
    kb_list.set_defaults(func=_cli().cmd_kb, kb_command="list")

    kb_show = kb_subparsers.add_parser(
        "show",
        help="Show a document's content",
        description="Display the full content of a document.",
    )
    kb_show.add_argument("doc_id", type=str, help="Document ID")
    kb_show.set_defaults(func=_cli().cmd_kb, kb_command="show")

    kb_delete = kb_subparsers.add_parser(
        "delete",
        help="Delete a document",
        description="Remove a document from the knowledge base.",
    )
    kb_delete.add_argument("doc_id", type=str, help="Document ID to delete")
    kb_delete.set_defaults(func=_cli().cmd_kb, kb_command="delete")

    kb_cite = kb_subparsers.add_parser(
        "cite",
        help="Get citations for a query",
        description="Find documents that can be cited for a specific phrase or term.",
    )
    kb_cite.add_argument("query", type=str, help="Text to find citations for")
    kb_cite.set_defaults(func=_cli().cmd_kb, kb_command="cite")

    kb_tags = kb_subparsers.add_parser(
        "tags",
        help="List all tags",
        description="List all unique tags in the knowledge base.",
    )
    kb_tags.set_defaults(func=_cli().cmd_kb, kb_command="tags")

    kb_parser.set_defaults(func=_cli().cmd_kb, kb_command="list")
