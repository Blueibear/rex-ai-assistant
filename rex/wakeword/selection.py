"""Wake word selection helpers."""

from __future__ import annotations

from collections.abc import Iterable


def has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def normalize_keyword(value: str) -> str:
    return " ".join(value.strip().lower().split())


def split_keywords(value: str | None) -> list[str]:
    if not has_text(value):
        return []
    return [part.strip().replace("_", " ") for part in value.split(",") if part.strip()]  # type: ignore[union-attr]


def resolve_keyword(
    keyword: str | None,
    wakeword: str | None,
    *,
    default: str | None = None,
) -> str | None:
    for candidate in (keyword, wakeword, default):
        if has_text(candidate):
            return candidate.strip()  # type: ignore[union-attr]
    return None


def list_openwakeword_keywords(openwakeword_module: object | None) -> list[str]:
    if openwakeword_module is None:
        return []
    models = getattr(openwakeword_module, "MODELS", None)
    if isinstance(models, dict):
        keys = list(models.keys())
    elif isinstance(models, Iterable):
        keys = list(models)
    else:
        return []

    available: list[str] = []
    for item in keys:
        if not item:
            continue
        available.append(str(item).replace("_", " "))
    return available


def select_fallback_keyword(
    available: list[str],
    *,
    preferred: str = "hey jarvis",
    fallback_keyword: str | None = None,
) -> str:
    if not available:
        raise RuntimeError("No openwakeword models available.")

    normalized_map = {normalize_keyword(item): item for item in available}
    if has_text(fallback_keyword):
        selected = normalized_map.get(normalize_keyword(fallback_keyword))  # type: ignore[arg-type]
        if selected:
            return selected

    preferred_match = normalized_map.get(normalize_keyword(preferred))
    if preferred_match:
        return preferred_match

    return available[0]
