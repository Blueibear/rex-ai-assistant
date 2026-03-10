"""User identity resolution for Rex AI Assistant.

Provides session-scoped active user selection for scenarios where voice
speaker recognition is unavailable or uncertain.

The active user can be set via:
1. Explicit ``--user <id>`` flag on any command
2. ``rex identify --user <id>`` or interactive ``rex identify``
3. ``runtime.active_user`` in ``config/rex_config.json``
4. ``runtime.user_id`` in ``config/rex_config.json`` (legacy fallback)

Session state is stored in a temporary file under the OS-appropriate
app data directory.  It is cleared on ``rex identify --clear`` or when
the session file is deleted.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _session_state_path() -> Path:
    """Return the path to the session state file."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(
            os.environ.get("XDG_RUNTIME_DIR", os.environ.get("TMPDIR", "/tmp"))  # noqa: S108
        )
    return base / "rex-ai" / "session.json"


def _known_user_ids() -> list[str]:
    """Discover known user IDs from Memory/ profiles."""
    memory_dir = Path(__file__).resolve().parent.parent / "Memory"
    if not memory_dir.is_dir():
        return []
    users = []
    for entry in sorted(memory_dir.iterdir()):
        if entry.is_dir() and (entry / "core.json").exists():
            users.append(entry.name)
    return users


def _load_session() -> dict:
    """Load the current session state from disk."""
    path = _session_state_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        except Exception:
            return {}
    return {}


def _save_session(data: dict) -> None:
    """Persist session state to disk."""
    path = _session_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save session state: %s", exc)


def get_session_user() -> str | None:
    """Return the active user from session state, or None."""
    session = _load_session()
    return session.get("active_user")


def set_session_user(user_id: str) -> None:
    """Set the active user in session state."""
    session = _load_session()
    session["active_user"] = user_id
    _save_session(session)
    logger.info("Session active user set to: %s", user_id)


def clear_session_user() -> None:
    """Clear the active user from session state."""
    session = _load_session()
    session.pop("active_user", None)
    _save_session(session)


def resolve_active_user(
    explicit_user: str | None = None,
    *,
    config: dict | None = None,
) -> str | None:
    """Resolve the active user through the priority chain.

    Priority:
    1. ``explicit_user`` (from ``--user`` flag)
    2. Session state (from ``rex identify``)
    3. ``runtime.active_user`` from config
    4. ``runtime.user_id`` from config

    Returns:
        User ID string, or ``None`` if no user could be resolved.
    """
    # 1. Explicit flag
    if explicit_user:
        return explicit_user

    # 2. Session state
    session_user = get_session_user()
    if session_user:
        return session_user

    # 3-4. Config values
    if config:
        runtime = config.get("runtime", {})
        active = runtime.get("active_user")
        if active:
            return active  # type: ignore[no-any-return]
        uid = runtime.get("user_id")
        if uid and uid != "default":
            return uid  # type: ignore[no-any-return]

    return None


def require_active_user(
    explicit_user: str | None = None,
    *,
    config: dict | None = None,
    action: str = "this command",
) -> str:
    """Resolve active user or raise an informative error.

    Args:
        explicit_user: User from ``--user`` flag.
        config: Config dict.
        action: Description of the action for error messages.

    Returns:
        Resolved user ID.

    Raises:
        SystemExit: If no user could be resolved.
    """
    user = resolve_active_user(explicit_user, config=config)
    if user:
        return user

    known = _known_user_ids()
    msg = f"Error: No active user for {action}.\n"
    if known:
        msg += f"Known users: {', '.join(known)}\n"
    msg += "Set one with: rex identify --user <id>\n"
    msg += "Or interactively: rex identify"
    raise SystemExit(msg)


def list_known_users() -> list[dict]:
    """Return info about known users from Memory/ profiles.

    Returns:
        List of dicts with ``id`` and ``name`` keys.
    """
    memory_dir = Path(__file__).resolve().parent.parent / "Memory"
    users = []  # type: ignore[var-annotated]
    if not memory_dir.is_dir():
        return users
    for entry in sorted(memory_dir.iterdir()):
        core = entry / "core.json"
        if entry.is_dir() and core.exists():
            try:
                data = json.loads(core.read_text(encoding="utf-8"))
                users.append(
                    {
                        "id": entry.name,
                        "name": data.get("name", entry.name),
                        "role": data.get("role", ""),
                    }
                )
            except Exception:
                users.append({"id": entry.name, "name": entry.name, "role": ""})
    return users


__all__ = [
    "clear_session_user",
    "get_session_user",
    "list_known_users",
    "require_active_user",
    "resolve_active_user",
    "set_session_user",
]
