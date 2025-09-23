"""Configuration management for the Rex assistant."""

from __future__ import annotations

import argparse
import logging
import os
from functools import lru_cache
from typing import Any, Optional

# ``python-dotenv`` is an optional helper; provide a tiny fallback so the
# configuration module still imports when the dependency is absent (for
# example in constrained test environments).
try:  # pragma: no cover - exercised in environments without dotenv
    from dotenv import load_dotenv, set_key  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - defensive fallback
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

    def set_key(path: str, key: str, value: str) -> tuple[str, str, bool]:
        lines: list[str] = []
        updated = False
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith(f"{key}="):
                        lines.append(f"{key}={value}\n")
                        updated = True
                    else:
                        lines.append(line)
        if not updated:
            lines.append(f"{key}={value}\n")
        with open(path, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
        return (key, value, True)

try:  # pragma: no cover - optional dependency
    from pydantic import BaseSettings, Field, ValidationError  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - minimal shim for tests
    class ValidationError(Exception):
        """Fallback validation error used when pydantic is unavailable."""

    class _FieldInfo:
        __slots__ = ("default", "env")

        def __init__(self, default: Any, env: str | None = None) -> None:
            self.default = default
            self.env = env

    def Field(default: Any, *, env: str | None = None) -> _FieldInfo:
        return _FieldInfo(default, env)

    class BaseSettings:
        """Tiny subset of ``pydantic.BaseSettings`` used in tests."""

        def __init__(self, **overrides: Any) -> None:
            values: dict[str, Any] = {}
            annotations = getattr(self, "__annotations__", {})
            for name, annotation in annotations.items():
                field_info = getattr(self.__class__, name, None)
                if isinstance(field_info, _FieldInfo):
                    env_value = os.getenv(field_info.env or "") if field_info.env else None
                    raw = env_value if env_value not in (None, "") else field_info.default
                else:
                    raw = getattr(self.__class__, name, None)
                if name in overrides:
                    raw = overrides[name]
                values[name] = self._coerce(annotation, raw)
            for name, value in values.items():
                setattr(self, name, value)

        def _coerce(self, annotation: Any, value: Any) -> Any:
            try:
                if annotation in (float, Optional[float]):
                    return float(value)
                if annotation in (int, Optional[int]):
                    return int(value)
            except (TypeError, ValueError):
                raise ValidationError(f"Invalid value for {annotation}: {value}")
            return value

        def dict(self) -> dict[str, Any]:
            return {name: getattr(self, name) for name in getattr(self, "__annotations__", {})}

load_dotenv()


class Settings(BaseSettings):
    """Centralised configuration backed by environment variables."""

    whisper_model: str = Field("base", env="REX_WHISPER_MODEL")
    temperature: float = Field(0.8, env="REX_LLM_TEMPERATURE")
    user_id: str = Field("default", env="REX_ACTIVE_USER")
    llm_model: str = Field("distilgpt2", env="REX_LLM_MODEL")
    llm_backend: str = Field("transformers", env="REX_LLM_BACKEND")
    max_memory_items: int = Field(50, env="REX_MEMORY_MAX_ITEMS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:  # pragma: no cover - defensive guard
        logging.getLogger(__name__).error("Invalid configuration: %s", exc)
        raise


settings = _load_settings()


def update_env_value(key: str, value: str) -> None:
    """Persist a configuration value into the ``.env`` file."""

    env_path = os.path.join(os.getcwd(), ".env")
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
