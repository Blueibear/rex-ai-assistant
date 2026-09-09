"""Compatibility exports for dependency-free wake-word selection helpers."""

from rex.wakeword_catalog import (
    DEFAULT_OPENWAKEWORD_KEYWORDS,
    has_text,
    list_openwakeword_keywords,
    normalize_keyword,
    resolve_keyword,
    select_fallback_keyword,
    split_keywords,
)

__all__ = [
    "DEFAULT_OPENWAKEWORD_KEYWORDS",
    "has_text",
    "list_openwakeword_keywords",
    "normalize_keyword",
    "resolve_keyword",
    "select_fallback_keyword",
    "split_keywords",
]
