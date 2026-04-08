"""Desktop file read/write/list operations for Rex (US-053 + US-056).

All operations are restricted to an allowlisted set of root directories.
Attempting to access a path outside the allowlist raises :class:`PermissionError`.

Cross-platform path normalisation is handled by :mod:`pathlib`.

Public API
----------
- :func:`read_file`   — read text content from a file
- :func:`write_file`  — write text content to a file
- :func:`list_dir`    — list entries in a directory
- :func:`summarize_file`  — summarise a text file via LLM (US-056)
- :func:`search_files`    — grep-like text search across a directory (US-056)
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowlist helpers
# ---------------------------------------------------------------------------


def _default_allowed_roots() -> list[str]:
    """Return the configured allowlist, falling back to the user's home dir."""
    try:
        from rex.config import load_config

        cfg = load_config()
        roots = getattr(cfg, "allowed_file_roots", None)
        if roots:
            return list(roots)
    except Exception:
        pass
    return [str(Path.home())]


def _resolve_and_check(
    path: "str | Path",
    allowed_roots: "list[str] | None",
) -> Path:
    """Resolve *path* to an absolute path and verify it is within *allowed_roots*.

    Args:
        path:          The file/directory path to check.
        allowed_roots: Absolute directory paths that are permitted.
                       If ``None``, the configured allowlist is used.

    Returns:
        The resolved :class:`~pathlib.Path`.

    Raises:
        PermissionError: If the resolved path is outside every allowed root.
    """
    resolved = Path(path).resolve()
    roots = allowed_roots if allowed_roots is not None else _default_allowed_roots()

    for root in roots:
        root_path = Path(root).resolve()
        try:
            resolved.relative_to(root_path)
            return resolved  # within this root — allowed
        except ValueError:
            continue  # not under this root, try the next one

    raise PermissionError(
        f"Access denied: '{resolved}' is outside the allowed directories: {roots}"
    )


# ---------------------------------------------------------------------------
# Public file operations
# ---------------------------------------------------------------------------


def read_file(
    path: "str | Path",
    allowed_roots: "list[str] | None" = None,
    encoding: str = "utf-8",
) -> str:
    """Read and return the text content of a file.

    Args:
        path:          Path to the file to read.
        allowed_roots: Allowed root directories (``None`` uses config defaults).
        encoding:      Text encoding (default UTF-8).

    Returns:
        The file content as a string.

    Raises:
        PermissionError: If *path* is outside the allowlist.
        FileNotFoundError: If *path* does not exist.
        IsADirectoryError: If *path* is a directory.
    """
    resolved = _resolve_and_check(path, allowed_roots)
    if resolved.is_dir():
        raise IsADirectoryError(f"'{resolved}' is a directory, not a file")
    logger.debug("read_file: %s", resolved)
    return resolved.read_text(encoding=encoding)


def write_file(
    path: "str | Path",
    content: str,
    allowed_roots: "list[str] | None" = None,
    encoding: str = "utf-8",
) -> None:
    """Write *content* to a file, creating parent directories as needed.

    Args:
        path:          Path to write to.
        content:       Text content to write.
        allowed_roots: Allowed root directories (``None`` uses config defaults).
        encoding:      Text encoding (default UTF-8).

    Raises:
        PermissionError: If *path* is outside the allowlist.
    """
    resolved = _resolve_and_check(path, allowed_roots)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("write_file: %s (%d bytes)", resolved, len(content.encode(encoding)))
    resolved.write_text(content, encoding=encoding)


def list_dir(
    path: "str | Path",
    allowed_roots: "list[str] | None" = None,
) -> list[str]:
    """Return a sorted list of entry names in *path*.

    Args:
        path:          Directory to list.
        allowed_roots: Allowed root directories (``None`` uses config defaults).

    Returns:
        Sorted list of file/directory names (not full paths).

    Raises:
        PermissionError: If *path* is outside the allowlist.
        NotADirectoryError: If *path* is not a directory.
    """
    resolved = _resolve_and_check(path, allowed_roots)
    if not resolved.is_dir():
        raise NotADirectoryError(f"'{resolved}' is not a directory")
    logger.debug("list_dir: %s", resolved)
    return sorted(entry.name for entry in resolved.iterdir())


# ---------------------------------------------------------------------------
# US-056: file summarization and search
# ---------------------------------------------------------------------------


def summarize_file(
    path: "str | Path",
    allowed_roots: "list[str] | None" = None,
    max_chars: int = 8000,
) -> str:
    """Summarise the text content of a file using the LLM.

    Reads at most *max_chars* characters from the file and passes them to the
    configured LLM with a summarise prompt.  The directory allowlist applies.

    Args:
        path:          Path to the file to summarise.
        allowed_roots: Allowed root directories (``None`` uses config defaults).
        max_chars:     Maximum characters from the file to include in the prompt.

    Returns:
        A human-readable summary string from the LLM, or a fallback message if
        the LLM is unavailable.

    Raises:
        PermissionError: If *path* is outside the allowlist.
        FileNotFoundError: If *path* does not exist.
    """
    content = read_file(path, allowed_roots)
    snippet = content[:max_chars]
    prompt = (
        f"Please summarise the following document concisely:\n\n{snippet}"
    )

    try:
        from rex.config import load_config
        from rex.llm_client import LanguageModel

        cfg = load_config()
        lm: Any = LanguageModel(cfg)
        result: str = lm.generate(prompt)
        return result
    except Exception as exc:
        logger.warning("summarize_file: LLM unavailable (%s); returning raw snippet", exc)
        # Fallback: return the first 300 chars as a preview
        preview = content[:300].strip()
        return f"[LLM unavailable] Preview:\n{preview}"


def search_files(
    directory: "str | Path",
    query: str,
    allowed_roots: "list[str] | None" = None,
    pattern: str = "*.txt",
) -> list[dict[str, Any]]:
    """Search text files in *directory* for lines matching *query*.

    Performs a case-insensitive substring search across all files matching
    *pattern* in *directory* (non-recursive).  Only plain text files are
    searched.  The directory allowlist applies to *directory*.

    Args:
        directory:     Directory to search.
        query:         Substring to look for (case-insensitive).
        allowed_roots: Allowed root directories (``None`` uses config defaults).
        pattern:       Glob pattern to select files (default ``"*.txt"``).

    Returns:
        List of ``{file, line_number, line}`` dicts for each matching line.

    Raises:
        PermissionError: If *directory* is outside the allowlist.
    """
    resolved_dir = _resolve_and_check(directory, allowed_roots)
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"'{resolved_dir}' is not a directory")

    query_lower = query.lower()
    results: list[dict[str, Any]] = []

    for file_path in sorted(resolved_dir.iterdir()):
        if not file_path.is_file():
            continue
        if not fnmatch.fnmatch(file_path.name, pattern):
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if query_lower in line.lower():
                results.append(
                    {
                        "file": str(file_path),
                        "line_number": lineno,
                        "line": line.rstrip(),
                    }
                )

    logger.debug("search_files: %d matches for %r in %s", len(results), query, resolved_dir)
    return results


__all__ = [
    "list_dir",
    "read_file",
    "search_files",
    "summarize_file",
    "write_file",
]
