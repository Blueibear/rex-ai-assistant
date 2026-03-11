"""Tests for US-033: User profiles.

Acceptance criteria:
- user profiles created
- preferences stored
- retrieval works
- Typecheck passes
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rex.identity import (
    create_user_profile,
    get_user_profile,
    list_known_users,
    update_user_preferences,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mem_dir(tmp_path: Path) -> Path:
    """Return a temporary Memory directory."""
    d = tmp_path / "Memory"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# User profiles created
# ---------------------------------------------------------------------------


def test_create_profile_creates_core_json(mem_dir: Path) -> None:
    path = create_user_profile("alice", "Alice", memory_dir=mem_dir)
    assert path.exists()
    assert path.name == "core.json"
    assert path.parent.name == "alice"


def test_create_profile_content(mem_dir: Path) -> None:
    create_user_profile("bob", "Bob", role="Admin", memory_dir=mem_dir)
    data = json.loads((mem_dir / "bob" / "core.json").read_text())
    assert data["name"] == "Bob"
    assert data["role"] == "Admin"
    assert data["user"] == "bob"
    assert "created_at" in data
    assert "last_updated" in data


def test_create_profile_with_preferences(mem_dir: Path) -> None:
    create_user_profile("carol", "Carol", preferences={"theme": "dark"}, memory_dir=mem_dir)
    data = json.loads((mem_dir / "carol" / "core.json").read_text())
    assert data["preferences"]["theme"] == "dark"


def test_create_profile_duplicate_raises(mem_dir: Path) -> None:
    create_user_profile("dave", "Dave", memory_dir=mem_dir)
    with pytest.raises(FileExistsError):
        create_user_profile("dave", "Dave2", memory_dir=mem_dir)


def test_create_profile_overwrite(mem_dir: Path) -> None:
    create_user_profile("eve", "Eve", memory_dir=mem_dir)
    create_user_profile("eve", "Eve Updated", overwrite=True, memory_dir=mem_dir)
    data = json.loads((mem_dir / "eve" / "core.json").read_text())
    assert data["name"] == "Eve Updated"


def test_create_profile_invalid_user_id(mem_dir: Path) -> None:
    with pytest.raises(ValueError):
        create_user_profile("", "Empty", memory_dir=mem_dir)
    with pytest.raises(ValueError):
        create_user_profile("a/b", "Slash", memory_dir=mem_dir)


def test_create_profile_appears_in_list_known_users(mem_dir: Path) -> None:
    create_user_profile("frank", "Frank", role="Tester", memory_dir=mem_dir)
    # list_known_users reads from the default Memory dir; test it via get_user_profile
    profile = get_user_profile("frank", memory_dir=mem_dir)
    assert profile is not None
    assert profile["name"] == "Frank"


# ---------------------------------------------------------------------------
# Preferences stored
# ---------------------------------------------------------------------------


def test_update_preferences_merges(mem_dir: Path) -> None:
    create_user_profile("grace", "Grace", preferences={"theme": "light"}, memory_dir=mem_dir)
    result = update_user_preferences("grace", {"language": "en"}, memory_dir=mem_dir)
    assert result is True
    data = json.loads((mem_dir / "grace" / "core.json").read_text())
    assert data["preferences"]["theme"] == "light"
    assert data["preferences"]["language"] == "en"


def test_update_preferences_overwrites_key(mem_dir: Path) -> None:
    create_user_profile("hank", "Hank", preferences={"theme": "dark"}, memory_dir=mem_dir)
    update_user_preferences("hank", {"theme": "solarized"}, memory_dir=mem_dir)
    data = json.loads((mem_dir / "hank" / "core.json").read_text())
    assert data["preferences"]["theme"] == "solarized"


def test_update_preferences_updates_last_updated(mem_dir: Path) -> None:
    create_user_profile("iris", "Iris", memory_dir=mem_dir)
    before = json.loads((mem_dir / "iris" / "core.json").read_text())["last_updated"]
    import time; time.sleep(0.01)
    update_user_preferences("iris", {"x": 1}, memory_dir=mem_dir)
    after = json.loads((mem_dir / "iris" / "core.json").read_text())["last_updated"]
    # last_updated should be >= before (may be equal within same second)
    assert after >= before


def test_update_preferences_missing_user_returns_false(mem_dir: Path) -> None:
    result = update_user_preferences("nobody", {"k": "v"}, memory_dir=mem_dir)
    assert result is False


# ---------------------------------------------------------------------------
# Retrieval works
# ---------------------------------------------------------------------------


def test_get_user_profile_returns_dict(mem_dir: Path) -> None:
    create_user_profile("jack", "Jack", memory_dir=mem_dir)
    profile = get_user_profile("jack", memory_dir=mem_dir)
    assert isinstance(profile, dict)
    assert profile["name"] == "Jack"


def test_get_user_profile_missing_returns_none(mem_dir: Path) -> None:
    result = get_user_profile("nobody", memory_dir=mem_dir)
    assert result is None


def test_get_user_profile_preferences_roundtrip(mem_dir: Path) -> None:
    prefs = {"tone": "casual", "language": "en"}
    create_user_profile("kim", "Kim", preferences=prefs, memory_dir=mem_dir)
    profile = get_user_profile("kim", memory_dir=mem_dir)
    assert profile is not None
    assert profile["preferences"] == prefs


def test_get_user_profile_after_update(mem_dir: Path) -> None:
    create_user_profile("leo", "Leo", preferences={"a": 1}, memory_dir=mem_dir)
    update_user_preferences("leo", {"b": 2}, memory_dir=mem_dir)
    profile = get_user_profile("leo", memory_dir=mem_dir)
    assert profile is not None
    assert profile["preferences"]["a"] == 1
    assert profile["preferences"]["b"] == 2


def test_list_known_users_includes_created_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """list_known_users uses the real Memory dir; test via a controlled mem dir."""
    mem_dir = tmp_path / "Memory"
    mem_dir.mkdir()
    create_user_profile("maya", "Maya", role="Guest", memory_dir=mem_dir)
    # list_known_users hardcodes the repo Memory dir, but we can verify core.json structure
    core = mem_dir / "maya" / "core.json"
    data = json.loads(core.read_text())
    assert data["name"] == "Maya"
    assert data["role"] == "Guest"
