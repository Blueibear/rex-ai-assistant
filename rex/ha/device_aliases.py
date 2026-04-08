"""Device alias resolver with fuzzy matching and synonym support."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_ALIASES_PATH = Path("config/device_aliases.json")
_FUZZY_MAX_DISTANCE = 2


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a
    prev = list(range(len_b + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len_b
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len_b]


class AliasResolver:
    """Resolves natural-language device names to Home Assistant entity IDs.

    Alias file format::

        {
            "aliases": {
                "bedroom light": "light.bedroom_main",
                "kitchen light": "light.kitchen_ceiling"
            },
            "synonyms": {
                "lamp": "light",
                "telly": "tv"
            }
        }

    ``resolve(query)`` returns ``(entity_id, confidence)`` on a match or ``None``.
    Confidence is 1.0 for exact matches, and decreases with edit distance.
    """

    def __init__(self, aliases_path: Path | str | None = None) -> None:
        path = Path(aliases_path) if aliases_path is not None else _DEFAULT_ALIASES_PATH
        self._aliases: dict[str, str] = {}
        self._synonyms: dict[str, str] = {}
        self._load(path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.debug("device_aliases: no alias file at %s, starting empty", path)
            return
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("device_aliases: failed to load %s: %s", path, exc)
            return
        self._aliases = {k.lower(): v for k, v in raw.get("aliases", {}).items()}
        self._synonyms = {k.lower(): v.lower() for k, v in raw.get("synonyms", {}).items()}
        logger.debug(
            "device_aliases: loaded %d aliases, %d synonyms",
            len(self._aliases),
            len(self._synonyms),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_all(self, query: str, min_confidence: float = 0.7) -> list[tuple[str, str, float]]:
        """Return all ``(alias, entity_id, confidence)`` pairs above *min_confidence*.

        Results are sorted by confidence descending.  Each entity_id appears at
        most once (the highest-confidence match is kept).
        """
        normalised = query.strip().lower()
        expanded = self._apply_synonyms(normalised)
        results: list[tuple[str, str, float]] = []
        seen_entities: set[str] = set()

        for alias, entity_id in self._aliases.items():
            conf: float | None = None

            if normalised == alias:
                conf = 1.0
            elif expanded != normalised and expanded == alias:
                conf = 0.95
            else:
                d = _levenshtein(normalised, alias)
                if d <= _FUZZY_MAX_DISTANCE:
                    conf = 0.9 - (d - 1) * 0.1
                if expanded != normalised:
                    d2 = _levenshtein(expanded, alias)
                    if d2 <= _FUZZY_MAX_DISTANCE:
                        exp_conf = 0.9 - (d2 - 1) * 0.1 - 0.05
                        if conf is None or exp_conf > conf:
                            conf = exp_conf

            if conf is not None and conf >= min_confidence and entity_id not in seen_entities:
                results.append((alias, entity_id, round(conf, 4)))
                seen_entities.add(entity_id)

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def resolve(self, query: str) -> tuple[str, float] | None:
        """Return ``(entity_id, confidence)`` for *query*, or ``None`` if no match.

        Resolution order:
        1. Exact match (confidence 1.0)
        2. Synonym-expanded exact match (confidence 0.95)
        3. Fuzzy match within ``_FUZZY_MAX_DISTANCE`` edits (confidence scales with distance)
        """
        normalised = query.strip().lower()

        # 1. Exact match
        if normalised in self._aliases:
            return self._aliases[normalised], 1.0

        # 2. Apply synonyms and try exact again
        expanded = self._apply_synonyms(normalised)
        if expanded != normalised and expanded in self._aliases:
            return self._aliases[expanded], 0.95

        # 3. Fuzzy match (try both original and synonym-expanded)
        best_entity: str | None = None
        best_distance = _FUZZY_MAX_DISTANCE + 1
        best_expanded = False

        for alias, entity_id in self._aliases.items():
            dist = _levenshtein(normalised, alias)
            if dist <= _FUZZY_MAX_DISTANCE and dist < best_distance:
                best_distance = dist
                best_entity = entity_id
                best_expanded = False

            if expanded != normalised:
                dist2 = _levenshtein(expanded, alias)
                if dist2 <= _FUZZY_MAX_DISTANCE and dist2 < best_distance:
                    best_distance = dist2
                    best_entity = entity_id
                    best_expanded = True

        if best_entity is not None:
            # Confidence: 0.9 at distance 1, 0.8 at distance 2; slightly lower for expanded
            base = 0.9 - (best_distance - 1) * 0.1
            confidence = base - (0.05 if best_expanded else 0.0)
            return best_entity, round(confidence, 4)

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_synonyms(self, text: str) -> str:
        """Replace all synonym words in *text* with their canonical equivalents."""
        words = text.split()
        replaced = [self._synonyms.get(w, w) for w in words]
        return " ".join(replaced)
