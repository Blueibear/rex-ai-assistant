"""Error recovery for failed Home Assistant commands.

When a command fails (device offline, unreachable), this module generates
a human-friendly spoken message and suggests alternative devices from the
same room or from the recently-used history.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _extract_room(entity_id: str) -> str | None:
    """Extract the first word of the entity suffix as a room name.

    ``light.kitchen_ceiling`` → ``"kitchen"``
    ``switch.garage_door``    → ``"garage"``
    """
    parts = entity_id.split(".", 1)
    if len(parts) < 2:
        return None
    return parts[1].split("_")[0]


def _same_room_alternatives(
    failed_entity_id: str,
    domain: str,
    entity_map: dict[str, str],
    entity_cache: dict[str, str],
) -> list[tuple[str, str]]:
    """Return up to 2 *(friendly_name, entity_id)* pairs in the same room.

    Merges *entity_map* (static config) and *entity_cache* (live discovery).
    """
    room = _extract_room(failed_entity_id)
    if not room:
        return []

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    combined: dict[str, str] = {**entity_cache, **entity_map}

    for alias, eid in combined.items():
        if eid == failed_entity_id or eid in seen:
            continue
        eid_domain = eid.split(".", 1)[0] if "." in eid else ""
        if eid_domain != domain:
            continue
        if _extract_room(eid) == room:
            candidates.append((alias, eid))
            seen.add(eid)

    return candidates[:2]


def suggest_alternatives(
    failed_entity_id: str,
    domain: str,
    entity_map: dict[str, str],
    entity_cache: dict[str, str],
    recent_entity_ids: list[str] | None = None,
) -> str:
    """Build a user-friendly spoken error-recovery message.

    Tries same-room alternatives first, then recently-used devices of the
    same domain.  Falls back to a generic "device may be offline" message
    when no alternatives are available.

    Args:
        failed_entity_id: The entity that failed (e.g. ``light.kitchen_main``).
        domain: HA domain of the failed entity (e.g. ``"light"``).
        entity_map: Static alias → entity_id mapping from config.
        entity_cache: Live alias → entity_id cache from HA discovery.
        recent_entity_ids: Entity IDs used recently (oldest first).

    Returns:
        A string suitable for TTS output.
    """
    failed_friendly = failed_entity_id.split(".")[-1].replace("_", " ")
    base_msg = f"The {failed_friendly} is not responding."

    # --- same-room candidates ---
    same_room = _same_room_alternatives(failed_entity_id, domain, entity_map, entity_cache)

    # --- recently-used candidates of the same domain ---
    recent_alts: list[tuple[str, str]] = []
    if recent_entity_ids:
        combined: dict[str, str] = {**entity_cache, **entity_map}
        reverse_map: dict[str, str] = {v: k for k, v in combined.items()}
        for eid in reversed(recent_entity_ids):
            if eid == failed_entity_id:
                continue
            eid_domain = eid.split(".", 1)[0] if "." in eid else ""
            if eid_domain != domain:
                continue
            friendly = reverse_map.get(eid, eid.split(".")[-1].replace("_", " "))
            recent_alts.append((friendly, eid))
            if len(recent_alts) >= 2:
                break

    # --- merge, deduplicate ---
    all_candidates: list[tuple[str, str]] = []
    seen_eids: set[str] = set()
    for alias, eid in same_room + recent_alts:
        if eid not in seen_eids:
            all_candidates.append((alias, eid))
            seen_eids.add(eid)

    if not all_candidates:
        return f"{base_msg} I could not complete that. The device may be offline."

    if len(all_candidates) == 1:
        alt = all_candidates[0][0]
        return f"{base_msg} Would you like me to try the {alt} instead?"

    alt_names = " or ".join(c[0] for c in all_candidates[:2])
    return f"{base_msg} Would you like me to try the {alt_names} instead?"
