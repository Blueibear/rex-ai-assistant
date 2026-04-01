"""Script-based skill auto-discovery.

Scans ``plugins/skills/`` for ``*.py`` files that expose a top-level
``SKILL_METADATA`` dict.  Valid files are automatically registered in the
provided :class:`~rex.skills.registry.SkillRegistry`.  Files without the
dict, or with an invalid dict, are logged and skipped.

Public API
----------
:func:`load_skills_from_directory`
    Scan a directory and register all valid skill scripts.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from rex.skills.registry import Skill, SkillRegistry

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_DIR = Path("plugins/skills")

# Required keys in SKILL_METADATA
_REQUIRED_KEYS = {"name", "description", "triggers"}


def _extract_metadata(source: str) -> dict | None:
    """Parse *source* and return the value of the ``SKILL_METADATA`` literal.

    Uses :mod:`ast` so the file is never executed.  Returns ``None`` if
    ``SKILL_METADATA`` is absent or its value is not a dict literal.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.debug("Syntax error while parsing skill file: %s", exc)
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "SKILL_METADATA":
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    return None
                if isinstance(value, dict):
                    return value
    return None


def _validate_metadata(meta: dict) -> str | None:
    """Return an error message if *meta* is invalid, else None."""
    missing = _REQUIRED_KEYS - meta.keys()
    if missing:
        return f"missing required keys: {missing}"

    if not isinstance(meta.get("name"), str) or not meta["name"].strip():
        return "name must be a non-empty string"

    triggers = meta.get("triggers")
    if not isinstance(triggers, list) or not triggers:
        return "triggers must be a non-empty list"

    return None


def load_skills_from_directory(
    registry: SkillRegistry,
    skills_dir: Path | str | None = None,
) -> list[Skill]:
    """Scan *skills_dir* and register every valid skill script.

    Files that are already registered (same handler path) are skipped so
    repeated calls are idempotent.

    Args:
        registry: The :class:`SkillRegistry` to register into.
        skills_dir: Directory to scan.  Defaults to ``plugins/skills/``.

    Returns:
        List of :class:`Skill` objects that were newly registered in this call.
    """
    resolved = Path(skills_dir) if skills_dir is not None else _DEFAULT_SKILLS_DIR

    if not resolved.is_dir():
        logger.debug("Skills directory %s does not exist; skipping scan", resolved)
        return []

    # Build set of already-registered handlers so we don't double-register.
    existing_handlers = {s.handler for s in registry.list_skills()}

    newly_registered: list[Skill] = []

    for path in sorted(resolved.glob("*.py")):
        if path.name.startswith("_"):
            # Skip __init__.py and private files.
            continue

        handler = str(path)
        if handler in existing_handlers:
            logger.debug("Skill %s already registered; skipping", path.name)
            continue

        source = path.read_text(encoding="utf-8")
        meta = _extract_metadata(source)

        if meta is None:
            logger.debug("No SKILL_METADATA found in %s; skipping", path.name)
            continue

        error = _validate_metadata(meta)
        if error:
            logger.warning("Invalid SKILL_METADATA in %s (%s); skipping", path.name, error)
            continue

        try:
            skill = registry.register(
                name=str(meta["name"]),
                description=str(meta.get("description", "")),
                trigger_patterns=[str(t) for t in meta["triggers"]],
                handler=handler,
            )
            newly_registered.append(skill)
            logger.info("Auto-registered skill %r from %s", skill.name, path.name)
        except Exception as exc:
            logger.warning("Failed to register skill from %s: %s", path.name, exc)

    return newly_registered
