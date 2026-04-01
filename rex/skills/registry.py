"""Skill registry — CRUD storage for custom Rex skills.

Skills are stored as JSON records in ``config/skills.json``.  Each record
describes a named capability that the assistant can dispatch to when a user
message matches one of its trigger patterns.

Public API
----------
:class:`Skill`
    Dataclass for a single skill record.
:class:`SkillRegistry`
    Load, save, and mutate the skills collection.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_PATH = Path("config/skills.json")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """A single registered skill.

    Attributes:
        id: Unique identifier (UUID string).
        name: Human-readable skill name.
        description: What the skill does (used for display and LLM context).
        trigger_patterns: List of regex or keyword patterns that activate the skill.
        handler: Module dotted path (``package.module:function``) or script
            file path that implements the skill.
        created_at: ISO-8601 timestamp when the skill was registered.
        enabled: Whether the skill is active.
    """

    id: str
    name: str
    description: str
    trigger_patterns: list[str]
    handler: str
    created_at: str
    enabled: bool = True

    @classmethod
    def new(
        cls,
        name: str,
        description: str,
        trigger_patterns: list[str],
        handler: str,
        *,
        enabled: bool = True,
    ) -> Skill:
        """Create a new skill with a generated ID and current timestamp."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            trigger_patterns=list(trigger_patterns),
            handler=handler,
            created_at=datetime.now(tz=UTC).isoformat(),
            enabled=enabled,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Skill:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            description=str(data.get("description", "")),
            trigger_patterns=list(data.get("trigger_patterns", [])),
            handler=str(data.get("handler", "")),
            created_at=str(data.get("created_at", "")),
            enabled=bool(data.get("enabled", True)),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Persistent store for custom Rex skills.

    Skills are persisted to a JSON file on every mutation.  Concurrent
    in-process access is safe; concurrent multi-process access is not
    guaranteed (no file locking).

    Parameters:
        skills_path: Path to the JSON storage file.  Created automatically
            if it does not exist.
    """

    def __init__(self, skills_path: Path | str | None = None) -> None:
        self._path = Path(skills_path) if skills_path is not None else _DEFAULT_SKILLS_PATH
        self._skills: dict[str, Skill] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        trigger_patterns: list[str],
        handler: str,
        *,
        enabled: bool = True,
    ) -> Skill:
        """Create and persist a new skill.

        Args:
            name: Human-readable skill name.
            description: What the skill does.
            trigger_patterns: Strings or regexes that activate the skill.
            handler: Module path or script path for the implementation.
            enabled: Whether the skill starts active (default True).

        Returns:
            The newly created :class:`Skill`.
        """
        if not name.strip():
            raise ValueError("Skill name must not be empty")
        if not trigger_patterns:
            raise ValueError("At least one trigger pattern is required")

        skill = Skill.new(
            name=name.strip(),
            description=description,
            trigger_patterns=trigger_patterns,
            handler=handler,
            enabled=enabled,
        )
        self._skills[skill.id] = skill
        self._save()
        logger.info("Registered skill %r (id=%s)", skill.name, skill.id)
        return skill

    def list_skills(self, *, include_disabled: bool = True) -> list[Skill]:
        """Return all skills, optionally filtering to enabled-only."""
        skills = list(self._skills.values())
        if not include_disabled:
            skills = [s for s in skills if s.enabled]
        return skills

    def get(self, skill_id: str) -> Skill | None:
        """Return the skill with the given ID, or None."""
        return self._skills.get(skill_id)

    def enable(self, skill_id: str) -> bool:
        """Enable a skill.  Returns True if found and updated."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return False
        skill.enabled = True
        self._save()
        return True

    def disable(self, skill_id: str) -> bool:
        """Disable a skill.  Returns True if found and updated."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return False
        skill.enabled = False
        self._save()
        return True

    def delete(self, skill_id: str) -> bool:
        """Remove a skill by ID.  Returns True if it existed."""
        if skill_id not in self._skills:
            return False
        del self._skills[skill_id]
        self._save()
        logger.info("Deleted skill id=%s", skill_id)
        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load skills from the JSON file (no-op if file absent)."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                logger.warning("skills.json root is not a list; ignoring")
                return
            for item in raw:
                try:
                    skill = Skill.from_dict(item)
                    self._skills[skill.id] = skill
                except Exception as exc:
                    logger.warning("Skipping malformed skill record: %s", exc)
        except Exception as exc:
            logger.warning("Failed to load skills from %s: %s", self._path, exc)

    def _save(self) -> None:
        """Persist all skills to the JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self._skills.values()]
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
