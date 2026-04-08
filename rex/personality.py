"""Personality system for the Rex assistant.

Defines built-in personalities that control the assistant's tone and style.
The active personality's system prompt is injected into LLM calls via
``Assistant._build_prompt()``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Personality:
    """Describes an assistant personality."""

    name: str
    system_prompt: str
    tone_keywords: list[str]
    greeting: str


_BUILT_IN: dict[str, Personality] = {
    "Friendly": Personality(
        name="Friendly",
        system_prompt=(
            "You are Rex, a warm and approachable AI assistant. "
            "Use a conversational, upbeat tone. "
            "Be encouraging and empathetic in your responses."
        ),
        tone_keywords=["warm", "conversational", "upbeat", "encouraging"],
        greeting="Hey there! How can I help you today?",
    ),
    "Professional": Personality(
        name="Professional",
        system_prompt=(
            "You are Rex, a professional AI assistant. "
            "Use a clear, precise, and business-like tone. "
            "Be concise and focus on delivering accurate information efficiently."
        ),
        tone_keywords=["precise", "formal", "concise", "business-like"],
        greeting="Hello. How may I assist you?",
    ),
    "Minimal": Personality(
        name="Minimal",
        system_prompt=(
            "You are Rex. Be brief. Answer only what is asked. "
            "Omit pleasantries and filler words."
        ),
        tone_keywords=["brief", "terse", "direct"],
        greeting="Ready.",
    ),
}

DEFAULT_PERSONALITY = "Friendly"


def get_personality(name: str) -> Personality:
    """Return the personality with the given name.

    Falls back to the default personality if *name* is not recognised.
    """
    return _BUILT_IN.get(name, _BUILT_IN[DEFAULT_PERSONALITY])


def list_personalities() -> list[Personality]:
    """Return all built-in personalities."""
    return list(_BUILT_IN.values())


__all__ = [
    "DEFAULT_PERSONALITY",
    "Personality",
    "get_personality",
    "list_personalities",
]
