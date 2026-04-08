"""Capability query responder for AskRex assistant.

Detects "What can you do?" style intents and generates a human-readable
response listing enabled capabilities grouped by category.

Usage::

    from rex.capabilities.responder import is_capability_query, build_capability_response
    from rex.capabilities.registry import get_capability_registry

    if is_capability_query(transcript):
        registry = get_capability_registry()
        reply = build_capability_response(registry)
"""

from __future__ import annotations

import re
from collections import defaultdict

from .registry import CapabilityRegistry

# Intent patterns for "What can you do?" style queries.
_CAPABILITY_PATTERN = re.compile(
    r"""
    \b(
        what\s+can\s+you\s+do         # "what can you do"
        | what\s+are\s+your\s+capabilities?  # "what are your capabilities"
        | what\s+do\s+you\s+support   # "what do you support"
        | list\s+(your\s+)?capabilities?  # "list capabilities" / "list your capabilities"
        | list\s+(your\s+)?features   # "list your features"
        | what\s+features?\s+do\s+you\s+have  # "what features do you have"
        | what\s+can\s+rex\s+do       # "what can rex do"
        | help\s+me\s+understand\s+what\s+you\s+can\s+do  # verbose form
        | show\s+(me\s+)?your\s+(capabilities?|features)  # "show me your capabilities"
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_capability_query(transcript: str) -> bool:
    """Return ``True`` if *transcript* is asking about Rex's capabilities.

    Args:
        transcript: User's spoken or typed message.

    Returns:
        ``True`` when the message matches a known capability-query pattern.
    """
    return bool(_CAPABILITY_PATTERN.search(transcript.strip()))


def build_capability_response(registry: CapabilityRegistry) -> str:
    """Build a human-readable response listing enabled capabilities by category.

    Args:
        registry: The :class:`~rex.capabilities.registry.CapabilityRegistry`
            to query.

    Returns:
        A formatted string listing capabilities grouped by category, or a
        fallback message when no integrations are configured.
    """
    enabled = registry.list()  # only enabled, sorted by name

    # Filter out the always-on "chat" baseline for the integration check.
    integrations = [c for c in enabled if c.name != "chat"]

    if not integrations:
        return "I can chat with you, but no integrations are set up yet."

    # Group by category, preserving insertion order within each group.
    by_category: dict[str, list[str]] = defaultdict(list)
    for cap in enabled:
        by_category[cap.category].append(cap.description)

    lines: list[str] = ["Here's what I can do:"]
    # Sort categories so output is deterministic; put "General" first.
    categories = sorted(by_category.keys(), key=lambda c: (c != "General", c))
    for category in categories:
        items = by_category[category]
        bullet_lines = "\n".join(f"  - {item}" for item in items)
        lines.append(f"\n{category}:\n{bullet_lines}")

    return "\n".join(lines)


__all__ = ["build_capability_response", "is_capability_query"]
