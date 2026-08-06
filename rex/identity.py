# OPENCLAW-WRAP: This module will be wrapped around OpenClaw. Preserve public API.

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
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .config import settings
from .runtime_paths import memory_dir as get_memory_dir

logger = logging.getLogger(__name__)

_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_WINDOWS_RESERVED_DEVICE_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    "clock$",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


def validate_user_id(user_id: str) -> str:
    """Validate and return a filesystem-safe user identifier.

    User IDs are used as directory names under ``Memory/``. Restricting the
    accepted characters and length prevents traversal through values such as
    ``..`` while retaining common IDs containing letters, digits, dots,
    underscores, and hyphens.

    Raises:
        ValueError: If *user_id* is empty, reserved, or filesystem-unsafe.
    """
    if not isinstance(user_id, str) or not _USER_ID_PATTERN.fullmatch(user_id):
        raise ValueError(f"Invalid user_id: {user_id!r}")
    if user_id in {".", ".."}:
        raise ValueError(f"Invalid user_id: {user_id!r}")
    # Windows strips trailing spaces/periods and treats the portion before an
    # extension as a device name. Apply that rule everywhere so a profile is
    # never portable on one OS but unsafe on another.
    normalized = user_id.rstrip(" .")
    device_stem = normalized.split(".", 1)[0].rstrip(" .").casefold()
    if device_stem in _WINDOWS_RESERVED_DEVICE_NAMES:
        raise ValueError(f"Invalid user_id: {user_id!r}")
    return user_id


def _validated_candidate(user_id: object, *, source: str) -> str | None:
    """Return a valid candidate ID or fail closed for persisted configuration."""
    if not isinstance(user_id, str) or not user_id:
        return None
    try:
        return validate_user_id(user_id)
    except ValueError:
        logger.warning("Ignoring invalid %s user ID: %r", source, user_id)
        return None


def _session_state_path() -> Path:
    """Return the path to the session state file."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(
            os.environ.get("XDG_RUNTIME_DIR", os.environ.get("TMPDIR", "/tmp"))  # noqa: S108
        )
    return base / "rex-ai" / "session.json"


def _known_user_ids(*, memory_dir: Path | None = None) -> list[str]:
    """Discover known user IDs from canonical or test Memory storage."""
    root = memory_dir if memory_dir is not None else get_memory_dir()
    if not root.is_dir():
        return []
    users: list[str] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and (entry / "core.json").exists():
            if _validated_candidate(entry.name, source="Memory directory"):
                users.append(entry.name)
    return users

def _load_session() -> dict:
    """Load the current session state from disk."""
    path = _session_state_path()
    if path.exists():
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            age = datetime.now(UTC) - modified_at
            if age.total_seconds() > settings.session_ttl_hours * 3600:
                path.unlink(missing_ok=True)
                return {}
            return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            logger.warning("Corrupted session file %s, resetting: %s", path, e)
            return {}
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
    return _validated_candidate(session.get("active_user"), source="session")


def set_session_user(user_id: str) -> None:
    """Set the active user in session state."""
    user_id = validate_user_id(user_id)
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
    # 1. Explicit flag. Invalid explicit selection is a caller error and must
    # not silently fall back to another person's session.
    if explicit_user:
        return validate_user_id(explicit_user)

    # 2. Session state. Invalid persisted state fails closed.
    session_user = _load_session().get("active_user")
    if session_user:
        return _validated_candidate(session_user, source="session")

    # 3-4. Config values. Invalid configured identity fails closed rather than
    # unexpectedly selecting the legacy fallback.
    if config:
        runtime = config.get("runtime", {})
        active = runtime.get("active_user")
        if active:
            return _validated_candidate(active, source="runtime.active_user")
        uid = runtime.get("user_id")
        if uid and uid != "default":
            return _validated_candidate(uid, source="runtime.user_id")

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


def resolve_entrypoint_user_id(
    settings_obj: object | None = None,
    *,
    explicit_user: str | None = None,
) -> str:
    """Return the profile a first-party single-user entrypoint binds to.

    ``Assistant`` no longer invents an identity when none is supplied (issue
    #303), so entrypoints that intentionally serve one configured profile
    resolve it here — outside the assistant — and pass it explicitly to
    ``Assistant(user_id=...)``.

    Priority:
    1. ``explicit_user`` (e.g. a ``--user`` flag) — validated, caller errors
       do not fall back to another profile.
    2. The session/config active user (``rex identify`` chain).
    3. ``settings_obj.user_id`` (``runtime.user_id``) — validated.
    4. The explicit ``"default"`` profile.

    The final ``"default"`` is a deliberate selection of the profile named
    ``default`` by a trusted first-party entrypoint, not an automatic
    assistant-side fallback.

    Raises:
        ValueError: If an explicit or configured user ID fails validation.
    """
    if explicit_user:
        return validate_user_id(explicit_user)

    session_user = resolve_active_user()
    if session_user:
        return session_user

    configured = getattr(settings_obj, "user_id", None) if settings_obj is not None else None
    if isinstance(configured, str) and configured:
        return validate_user_id(configured)

    return "default"


def list_known_users(*, memory_dir: Path | None = None) -> list[dict]:
    """Return known users from canonical or test Memory storage."""
    root = memory_dir if memory_dir is not None else get_memory_dir()
    users: list[dict] = []
    if not root.is_dir():
        return users
    for entry in sorted(root.iterdir()):
        core = entry / "core.json"
        if not entry.is_dir() or not core.exists():
            continue
        if not _validated_candidate(entry.name, source="Memory directory"):
            continue
        try:
            data = json.loads(core.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("profile JSON is not an object")
            raw_name = data.get("name", entry.name)
            raw_role = data.get("role", "")
            name = raw_name if isinstance(raw_name, str) and raw_name else entry.name
            role = raw_role if isinstance(raw_role, str) else ""
            users.append({"id": entry.name, "name": name, "role": role})
        except Exception:
            users.append({"id": entry.name, "name": entry.name, "role": ""})
    return users

def _atomic_write_json(path: Path, data: dict) -> None:
    """Atomically write one JSON object beside its destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_user_profile(
    user_id: str,
    name: str,
    role: str = "",
    preferences: dict | None = None,
    *,
    memory_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a user profile in canonical or test Memory storage."""
    user_id = validate_user_id(user_id)
    base = memory_dir if memory_dir is not None else get_memory_dir()
    core_path = base / user_id / "core.json"
    if core_path.exists() and not overwrite:
        raise FileExistsError(f"User profile already exists: {core_path}")
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    data: dict = {
        "name": name,
        "role": role,
        "user": user_id,
        "preferences": preferences or {},
        "created_at": now,
        "last_updated": now,
    }
    _atomic_write_json(core_path, data)
    logger.info("Created user profile for %s", user_id)
    return core_path

def get_user_profile(
    user_id: str,
    *,
    memory_dir: Path | None = None,
) -> dict | None:
    """Load a user profile object from canonical or test Memory storage."""
    user_id = validate_user_id(user_id)
    base = memory_dir if memory_dir is not None else get_memory_dir()
    core_path = base / user_id / "core.json"
    if not core_path.exists():
        return None
    try:
        data = json.loads(core_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            logger.warning("Profile for %s is not a JSON object", user_id)
            return None
        return data
    except Exception as exc:
        logger.warning("Failed to load profile for %s: %s", user_id, exc)
        return None

def update_user_preferences(
    user_id: str,
    preferences: dict,
    *,
    memory_dir: Path | None = None,
) -> bool:
    """Merge preferences into an existing profile with atomic persistence."""
    user_id = validate_user_id(user_id)
    base = memory_dir if memory_dir is not None else get_memory_dir()
    core_path = base / user_id / "core.json"
    if not core_path.exists():
        return False
    try:
        data = json.loads(core_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        existing_prefs = data.get("preferences", {})
        if not isinstance(existing_prefs, dict):
            existing_prefs = {}
        merged = dict(existing_prefs)
        merged.update(preferences)
        data["preferences"] = merged
        data["last_updated"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        _atomic_write_json(core_path, data)
        logger.info("Updated preferences for user %s", user_id)
        return True
    except Exception as exc:
        logger.warning("Failed to update preferences for %s: %s", user_id, exc)
        return False

__all__ = [
    "clear_session_user",
    "create_user_profile",
    "get_session_user",
    "get_user_profile",
    "list_known_users",
    "require_active_user",
    "resolve_active_user",
    "resolve_entrypoint_user_id",
    "set_session_user",
    "update_user_preferences",
    "validate_user_id",
]
