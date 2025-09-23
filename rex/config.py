"""Configuration management for the Rex assistant.

This module intentionally keeps its external dependencies optional so that
the unit test environment does not need to install heavy packages.  When
``python-dotenv`` is unavailable we fall back to light-weight helpers that can
read and persist values inside the project level ``.env`` file.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from functools import lru_cache
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional helper
    from dotenv import load_dotenv, set_key  # type: ignore
except ImportError:  # pragma: no cover - minimal fallback
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        """Best-effort stand-in for :func:`dotenv.load_dotenv`."""

        env_path = os.path.join(os.getcwd(), ".env")
        if not os.path.exists(env_path):
            return False
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line.lstrip().startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
        return True

    def set_key(dotenv_path: str, key: str, value: str) -> None:
        """Persist a key/value pair inside ``dotenv_path``."""

        lines: list[str] = []
        if os.path.exists(dotenv_path):
            with open(dotenv_path, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        prefix = f"{key}="
        for index, line in enumerate(lines):
            if line.startswith(prefix):
                lines[index] = f"{prefix}{value}"
                break
        else:
            lines.append(f"{prefix}{value}")
        with open(dotenv_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + ("\n" if lines else ""))


@dataclasses.dataclass
class Settings:
    whisper_model: str = "base"
    temperature: float = 0.8
    user_id: str = "default"
    llm_model: str = "distilgpt2"
    llm_backend: str = "transformers"
    max_memory_items: int = 50

    def dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


_ENV_FIELDS: Dict[str, tuple[str, Callable[[str], Any]]] = {
    "whisper_model": ("REX_WHISPER_MODEL", str),
    "temperature": ("REX_LLM_TEMPERATURE", float),
    "user_id": ("REX_ACTIVE_USER", str),
    "llm_model": ("REX_LLM_MODEL", str),
    "llm_backend": ("REX_LLM_BACKEND", str),
    "max_memory_items": ("REX_MEMORY_MAX_ITEMS", int),
}


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    load_dotenv()

    values: Dict[str, Any] = {}
    for field_name, (env_var, caster) in _ENV_FIELDS.items():
        raw = os.getenv(env_var)
        if raw is None or raw == "":
            continue
        try:
            values[field_name] = caster(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid value for %s: %s", env_var, raw)
    return Settings(**values)


settings = _load_settings()


def update_env_value(key: str, value: str) -> None:
    """Persist a configuration value into the ``.env`` file."""

    env_path = os.path.join(os.getcwd(), ".env")
    os.makedirs(os.path.dirname(env_path) or ".", exist_ok=True)
    set_key(env_path, key, value)
    _load_settings.cache_clear()
    globals()["settings"] = _load_settings()


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage Rex configuration values.")
    parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Persist a key/value pair to .env")
    parser.add_argument("--get", metavar="KEY", help="Print a single configuration value")
    parser.add_argument("--show", action="store_true", help="Print all resolved configuration values")

    args = parser.parse_args(argv)

    if args.set:
        key, value = args.set
        update_env_value(key, value)
        print(f"Updated {key} -> {value}")
        return 0

    if args.get:
        current = _load_settings().dict().get(args.get.lower())
        if current is None:
            print("<unset>")
        else:
            print(current)
        return 0

    if args.show:
        for key, value in _load_settings().dict().items():
            print(f"{key}: {value}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_cli())
