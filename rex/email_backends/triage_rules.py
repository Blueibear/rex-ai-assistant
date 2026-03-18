"""Config-driven email triage rules engine.

Rules are stored in a JSON config file and evaluated in declared priority
order; the first matching rule wins.  The engine auto-reloads the file
when its modification time changes so that adding or editing a rule takes
effect on the next :meth:`TriageRulesEngine.categorize` call without
restarting the process or modifying source code.

Rule file format (JSON)
-----------------------
.. code-block:: json

    {
      "rules": [
        {
          "name": "optional-human-label",
          "category": "urgent",
          "match": {
            "sender":  "regex-pattern",
            "subject": "regex-pattern",
            "body":    "regex-pattern"
          }
        }
      ]
    }

All three ``match`` keys (``sender``, ``subject``, ``body``) are optional.
A rule fires when **all** provided fields match.  An empty ``match`` dict
matches every email (useful as a final catch-all rule).

The ``category`` value must be one of: urgent, action_required, newsletter,
fyi.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rex.email_backends.base import EmailEnvelope
from rex.email_backends.triage import TRIAGE_CATEGORIES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriageRule:
    """A single triage rule loaded from config."""

    name: str
    category: str
    sender_pattern: str | None = None
    subject_pattern: str | None = None
    body_pattern: str | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_rules(raw: dict[str, Any]) -> list[TriageRule]:
    """Parse the raw JSON dict into an ordered list of :class:`TriageRule`."""
    rules: list[TriageRule] = []
    for i, entry in enumerate(raw.get("rules", [])):
        match_block = entry.get("match", {})
        category = entry.get("category", "fyi")
        if category not in TRIAGE_CATEGORIES:
            raise ValueError(
                f"Rule #{i} has invalid category {category!r}. "
                f"Valid values: {', '.join(TRIAGE_CATEGORIES)}"
            )
        rules.append(
            TriageRule(
                name=entry.get("name", f"rule-{i}"),
                category=category,
                sender_pattern=match_block.get("sender"),
                subject_pattern=match_block.get("subject"),
                body_pattern=match_block.get("body"),
            )
        )
    return rules


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _matches_rule(rule: TriageRule, envelope: EmailEnvelope) -> bool:
    """Return True when *envelope* satisfies every populated match field."""
    if rule.sender_pattern:
        if not re.search(rule.sender_pattern, envelope.from_addr, re.IGNORECASE):
            return False
    if rule.subject_pattern:
        if not re.search(rule.subject_pattern, envelope.subject, re.IGNORECASE):
            return False
    if rule.body_pattern:
        if not re.search(rule.body_pattern, envelope.snippet, re.IGNORECASE):
            return False
    return True


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TriageRulesEngine:
    """Evaluates config-driven triage rules against :class:`EmailEnvelope` objects.

    The engine watches the rules file's modification time and reloads
    automatically so that rule changes take effect without a restart.
    """

    def __init__(self, rules_path: str | Path) -> None:
        self._path = Path(rules_path)
        self._rules: list[TriageRule] = []
        self._mtime: float = -1.0
        self._load()

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Reload the rules file if it has changed since the last read."""
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            logger.warning("Triage rules file not found: %s", self._path)
            self._rules = []
            self._mtime = -1.0
            return
        if mtime == self._mtime:
            return  # file unchanged — skip reload
        try:
            raw: dict[str, Any] = json.loads(self._path.read_text(encoding="utf-8"))
            self._rules = _parse_rules(raw)
            self._mtime = mtime
            logger.info("Loaded %d triage rule(s) from %s", len(self._rules), self._path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load triage rules from %s: %s", self._path, exc)

    def reload(self) -> None:
        """Force an unconditional reload from the rules file."""
        self._mtime = -1.0
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def rules(self) -> list[TriageRule]:
        """Currently loaded rules (auto-reloads if file changed on disk)."""
        self._load()
        return list(self._rules)

    def categorize(self, envelope: EmailEnvelope) -> str | None:
        """Return the first matching rule's category, or ``None`` if no rule fires.

        ``None`` signals the caller to fall through to its own built-in logic
        (e.g. the pattern-based :func:`~rex.email_backends.triage.categorize`).
        """
        self._load()
        for rule in self._rules:
            if _matches_rule(rule, envelope):
                logger.debug(
                    "Rule %r matched subject=%r → %s",
                    rule.name,
                    envelope.subject,
                    rule.category,
                )
                return rule.category
        return None

    def categorize_with_fallback(self, envelope: EmailEnvelope) -> str:
        """Return the category from rules, falling back to built-in patterns.

        Calls :meth:`categorize`; if no rule fires, delegates to
        :func:`~rex.email_backends.triage.categorize`.
        """
        from rex.email_backends.triage import categorize as builtin_categorize

        result = self.categorize(envelope)
        if result is not None:
            return result
        return builtin_categorize(envelope)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def load_rules_from_file(path: str | Path) -> list[TriageRule]:
    """Parse and return rules from a JSON file (one-shot, no mtime tracking)."""
    p = Path(path)
    raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return _parse_rules(raw)


__all__ = [
    "TriageRule",
    "TriageRulesEngine",
    "load_rules_from_file",
]
