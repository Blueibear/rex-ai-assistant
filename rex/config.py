"""Central configuration management for the Rex assistant.

Unified configuration that replaces both config.py and rex/config.py
"""

from __future__ import annotations

import argparse
import dataclasses
import hmac
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rex.assistant_errors import ConfigurationError

logger = logging.getLogger(__name__)

# Environment variables that must be provided for a functional deployment.
REQUIRED_ENV_KEYS: Sequence[str] = (
    "OPENAI_API_KEY",
)

# Optional dependencies
try:
    from dotenv import load_dotenv, set_key
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        env_path = Path.cwd() / ".env"
        if not env_path.exists():
            return False
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
        return True

    def set_key(dotenv_path: str, key: str, value: str) -> None:
        env_file = Path(dotenv_path)
        lines: list[str] = []
        if env_file.exists():
            lines = env_file.read_text(encoding="utf-8").splitlines()
        prefix = f"{key}="
        for index, line in enumerate(lines):
            if line.startswith(prefix):
                lines[index] = f"{prefix}{value}"
                break
        else:
            lines.append(f"{prefix}{value}")
        env_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

try:
    from pydantic import BaseSettings, Field
    _HAS_PYDANTIC = True
except ImportError:
    BaseSettings = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    _HAS_PYDANTIC = False

# -- The Settings dataclass would follow unchanged -- #
# Keep your original Settings class definition here.
# (Omitted here for brevity)

# -- ENV mapping (_ENV_MAPPING) and _cast_value function remain unchanged --

@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    """Load settings from environment with caching."""
    load_dotenv()
    values: Dict[str, Any] = {}
    for field, aliases in _ENV_MAPPING.items():
        for env_var in aliases:
            raw = os.getenv(env_var)
            if raw is None or raw == "":
                continue
            try:
                values[field] = _cast_value(field, raw)
                break
            except (TypeError, ValueError) as exc:
                logger.warning("Invalid value for %s: %s (%s)", env_var, raw, exc)

    # Handle allowed_origins specially
    if "REX_ALLOWED_ORIGINS" in os.environ:
        values["allowed_origins"] = [
            origin.strip().rstrip("/")
            for origin in os.environ["REX_ALLOWED_ORIGINS"].split(",")
            if origin.strip()
        ] or ["*"]

    return Settings(**values)

# Module-level settings instance
settings = _load_settings()

def reload_settings() -> Settings:
    """Force settings to be reloaded from the environment."""
    _load_settings.cache_clear()
    new_settings = _load_settings()
    globals()["settings"] = new_settings
    return new_settings

def load_config(*, env_path: Optional[Path] = None, reload: bool = False) -> Settings:
    """Load configuration (backward compatibility wrapper)."""
    if reload:
        return reload_settings()
    return settings

def validate_config(config: Settings) -> None:
    """Validate configuration (calls __post_init__ internally)."""
    config.__post_init__()

def update_env_value(key: str, value: str) -> None:
    """Persist a configuration value into the .env file."""
    env_path = Path.cwd() / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    set_key(str(env_path), key, value)
    reload_settings()

def show_config() -> None:
    """Print the current configuration values for debugging."""
    for key, value in sorted(settings.dict().items()):
        print(f"{key}: {value}")

def _cli(argv: Iterable[str] | None = None) -> int:
    """CLI for managing configuration."""
    parser = argparse.ArgumentParser(description="Manage Rex configuration values.")
    parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Persist a key/value pair to .env")
    parser.add_argument("--get", metavar="KEY", help="Print a single configuration value")
    parser.add_argument("--show", action="store_true", help="Print all resolved configuration values")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.set:
        key, value = args.set
        update_env_value(key, value)
        print(f"Updated {key} -> {value}")
        return 0

    if args.get:
        key = args.get.lower()
        value = settings.dict().get(key)
        print(value if value is not None else "<unset>")
        return 0

    if args.show:
        for key, value in sorted(settings.dict().items()):
            print(f"{key}: {value}")
        return 0

    parser.print_help()
    return 0

def cli(argv: Iterable[str] | None = None) -> int:
    """Public CLI entrypoint wrapper."""
    return _cli(argv)

# Backward compatibility aliases
AppConfig = Settings
ENV_PATH = Path.cwd() / ".env"
ENV_MAPPING = _ENV_MAPPING

__all__ = [
    "Settings",
    "AppConfig",
    "settings",
    "load_config",
    "reload_settings",
    "validate_config",
    "update_env_value",
    "show_config",
    "REQUIRED_ENV_KEYS",
    "ConfigurationError",
    "cli",
]

if __name__ == "__main__":
    raise SystemExit(_cli())

