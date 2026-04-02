"""Local file read/write capability for Rex (Phase 6 — US-WIN-001).

All operations are restricted to paths inside ``AppConfig.allowed_file_roots``
(defaults to the user's home directory).  Path traversal outside the allowlist
is blocked at the validation layer.

Functions are designed as tool handlers: they accept ``**kwargs`` so they can
be invoked uniformly by ``ToolDispatcher``.  The common calling convention is::

    result = read_file(path="/absolute/path/to/file")
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowlist resolution
# ---------------------------------------------------------------------------

_DEFAULT_ALLOWED_ROOTS: list[str] = [str(Path.home())]


def _resolve_allowed_roots(config: Any) -> list[Path]:
    """Return absolute ``Path`` objects for all configured allowlist roots.

    Falls back to the user home directory when the config carries no roots or
    has the attribute missing.
    """
    raw: list[str] = (
        getattr(config, "allowed_file_roots", _DEFAULT_ALLOWED_ROOTS) or _DEFAULT_ALLOWED_ROOTS
    )
    return [Path(r).resolve() for r in raw]


def _validate_path(raw_path: str, allowed_roots: list[Path]) -> Path:
    """Resolve *raw_path* and verify it is inside one of *allowed_roots*.

    Raises:
        PermissionError: if the resolved path is outside every allowlist root.
        ValueError: if *raw_path* is empty.
    """
    if not raw_path:
        raise ValueError("path must not be empty")
    resolved = Path(raw_path).resolve()
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise PermissionError(
        f"Path '{resolved}' is outside the allowed file roots: "
        + ", ".join(str(r) for r in allowed_roots)
    )


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


def read_file(
    *,
    path: str,
    config: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Read the text content of a file.

    Args:
        path: Absolute or relative path to the file.
        config: ``AppConfig`` instance (used to resolve allowlist roots).

    Returns:
        ``{"path": str, "content": str}`` on success, or
        ``{"error": str}`` on failure.
    """
    roots = _resolve_allowed_roots(config)
    try:
        safe = _validate_path(path, roots)
        content = safe.read_text(encoding="utf-8", errors="replace")
        logger.info("file_ops: read %s (%d bytes)", safe, len(content))
        return {"path": str(safe), "content": content}
    except (PermissionError, ValueError) as exc:
        logger.warning("file_ops: read blocked — %s", exc)
        return {"error": str(exc)}
    except OSError as exc:
        logger.warning("file_ops: read failed — %s", exc)
        return {"error": str(exc)}


def write_file(
    *,
    path: str,
    content: str = "",
    config: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Write *content* to a file, creating parent directories as needed.

    Args:
        path: Absolute or relative path to the target file.
        content: Text content to write (UTF-8).
        config: ``AppConfig`` instance (used to resolve allowlist roots).

    Returns:
        ``{"path": str, "bytes_written": int}`` on success, or
        ``{"error": str}`` on failure.
    """
    roots = _resolve_allowed_roots(config)
    try:
        safe = _validate_path(path, roots)
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        logger.info("file_ops: write %s (%d bytes)", safe, len(content))
        return {"path": str(safe), "bytes_written": len(content.encode())}
    except (PermissionError, ValueError) as exc:
        logger.warning("file_ops: write blocked — %s", exc)
        return {"error": str(exc)}
    except OSError as exc:
        logger.warning("file_ops: write failed — %s", exc)
        return {"error": str(exc)}


def list_directory(
    *,
    path: str,
    config: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """List the contents of a directory.

    Args:
        path: Absolute or relative path to the directory.
        config: ``AppConfig`` instance (used to resolve allowlist roots).

    Returns:
        ``{"path": str, "entries": list[str]}`` on success, or
        ``{"error": str}`` on failure.
    """
    roots = _resolve_allowed_roots(config)
    try:
        safe = _validate_path(path, roots)
        entries = [e.name for e in sorted(safe.iterdir())]
        logger.info("file_ops: list %s (%d entries)", safe, len(entries))
        return {"path": str(safe), "entries": entries}
    except (PermissionError, ValueError) as exc:
        logger.warning("file_ops: list blocked — %s", exc)
        return {"error": str(exc)}
    except OSError as exc:
        logger.warning("file_ops: list failed — %s", exc)
        return {"error": str(exc)}


def move_file(
    *,
    src: str,
    dst: str,
    config: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Move (rename) a file or directory.

    Both *src* and *dst* must be inside the allowlist roots.

    Args:
        src: Source path.
        dst: Destination path.
        config: ``AppConfig`` instance (used to resolve allowlist roots).

    Returns:
        ``{"src": str, "dst": str}`` on success, or ``{"error": str}`` on failure.
    """
    roots = _resolve_allowed_roots(config)
    try:
        safe_src = _validate_path(src, roots)
        safe_dst = _validate_path(dst, roots)
        safe_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(safe_src), str(safe_dst))
        logger.info("file_ops: move %s -> %s", safe_src, safe_dst)
        return {"src": str(safe_src), "dst": str(safe_dst)}
    except (PermissionError, ValueError) as exc:
        logger.warning("file_ops: move blocked — %s", exc)
        return {"error": str(exc)}
    except OSError as exc:
        logger.warning("file_ops: move failed — %s", exc)
        return {"error": str(exc)}


def delete_file(
    *,
    path: str,
    config: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Delete a file (not a directory).

    Args:
        path: Absolute or relative path to the file to remove.
        config: ``AppConfig`` instance (used to resolve allowlist roots).

    Returns:
        ``{"path": str, "deleted": True}`` on success, or
        ``{"error": str}`` on failure.
    """
    roots = _resolve_allowed_roots(config)
    try:
        safe = _validate_path(path, roots)
        safe.unlink()
        logger.info("file_ops: delete %s", safe)
        return {"path": str(safe), "deleted": True}
    except (PermissionError, ValueError) as exc:
        logger.warning("file_ops: delete blocked — %s", exc)
        return {"error": str(exc)}
    except OSError as exc:
        logger.warning("file_ops: delete failed — %s", exc)
        return {"error": str(exc)}
