"""Tests for US-324: Per-user memory -- minimal viable read/write."""

from __future__ import annotations

from pathlib import Path


def test_store_and_recall(tmp_path: Path) -> None:
    """store() persists a fact; recall() retrieves it."""
    from rex.user_facts import recall, store

    store("james", "dog_name", "Max", memory_root=tmp_path)
    assert recall("james", "dog_name", memory_root=tmp_path) == "Max"


def test_recall_missing_returns_none(tmp_path: Path) -> None:
    from rex.user_facts import recall

    assert recall("james", "nonexistent", memory_root=tmp_path) is None


def test_store_persists_across_calls(tmp_path: Path) -> None:
    """Facts written in one call are readable in a separate call."""
    from rex.user_facts import recall, store

    store("alice", "city", "Austin", memory_root=tmp_path)
    store("alice", "pet", "cat", memory_root=tmp_path)
    assert recall("alice", "city", memory_root=tmp_path) == "Austin"
    assert recall("alice", "pet", memory_root=tmp_path) == "cat"


def test_facts_isolated_per_user(tmp_path: Path) -> None:
    """Facts for different users do not bleed across."""
    from rex.user_facts import recall, store

    store("alice", "key", "alice_value", memory_root=tmp_path)
    store("bob", "key", "bob_value", memory_root=tmp_path)
    assert recall("alice", "key", memory_root=tmp_path) == "alice_value"
    assert recall("bob", "key", memory_root=tmp_path) == "bob_value"


def test_facts_stored_in_memory_dir(tmp_path: Path) -> None:
    """Facts are stored in a JSON file inside the Memory/ directory."""
    from rex.user_facts import store

    store("james", "dog", "Max", memory_root=tmp_path)
    facts_file = tmp_path / "james_facts.json"
    assert facts_file.exists()


def test_format_facts_for_prompt(tmp_path: Path) -> None:
    """format_facts_for_prompt returns a non-empty string when facts exist."""
    from rex.user_facts import format_facts_for_prompt, store

    store("james", "dog_name", "Max", memory_root=tmp_path)
    result = format_facts_for_prompt("james", memory_root=tmp_path)
    assert result is not None
    assert "dog_name" in result
    assert "Max" in result
    assert "james" in result


def test_format_facts_for_prompt_empty(tmp_path: Path) -> None:
    """format_facts_for_prompt returns None when no facts exist."""
    from rex.user_facts import format_facts_for_prompt

    assert format_facts_for_prompt("nobody", memory_root=tmp_path) is None


def test_cli_remember_stores_fact(tmp_path: Path, monkeypatch: object) -> None:
    """python -m rex remember 'fact' stores the fact for the default user."""
    import types

    from rex.cli import cmd_remember

    monkeypatch.setattr(
        "rex.user_facts._MEMORY_ROOT",
        tmp_path,
    )

    args = types.SimpleNamespace(fact="My dog is named Max", user="default", key=None, value=None)
    ret = cmd_remember(args)
    assert ret == 0

    from rex.user_facts import recall_all

    facts = recall_all("default", memory_root=tmp_path)
    assert any("My dog is named Max" in v for v in facts.values())


def test_cli_remember_explicit_key_value(tmp_path: Path, monkeypatch: object) -> None:
    """--key/--value stores under the given key."""
    import types

    from rex.cli import cmd_remember

    monkeypatch.setattr("rex.user_facts._MEMORY_ROOT", tmp_path)

    args = types.SimpleNamespace(
        fact="anything", user="default", key="dog_name", value="Max"
    )
    cmd_remember(args)

    from rex.user_facts import recall

    assert recall("default", "dog_name", memory_root=tmp_path) == "Max"
