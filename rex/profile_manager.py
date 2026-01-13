"""Profile manager for Rex assistant configuration."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from rex.logging_utils import get_logger

DEFAULT_PROFILES_DIR = "profiles"


logger = get_logger(__name__)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _basic_validate(profile: dict[str, Any], required: list[str]) -> None:
    for key in required:
        if key not in profile:
            raise ValueError(f"Profile missing required field: {key}")
    if not isinstance(profile["profile_version"], int):
        raise ValueError("profile_version must be an integer")
    if profile["profile_version"] != 1:
        raise ValueError("profile_version must be 1")
    if not isinstance(profile["name"], str):
        raise ValueError("name must be a string")
    if not isinstance(profile["description"], str):
        raise ValueError("description must be a string")
    if not isinstance(profile["capabilities"], list):
        raise ValueError("capabilities must be a list")
    if not isinstance(profile["overrides"], dict):
        raise ValueError("overrides must be an object")


def _validate_profile(profile: dict[str, Any], schema_path: Path) -> None:
    if not schema_path.exists():
        return

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []
    _basic_validate(profile, [str(item) for item in required])


def load_profile(name: str, profiles_dir: str = DEFAULT_PROFILES_DIR) -> dict[str, Any]:
    profile_path = Path(profiles_dir) / f"{name}.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            "Profile file not found: "
            f"{profile_path}. Create it or set active_profile in config/rex_config.json."
        )

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    schema_path = Path(profiles_dir) / "profile.schema.json"
    _validate_profile(profile, schema_path)
    return profile


def apply_profile(base_config: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base_config)
    overrides = profile.get("overrides", {})
    if overrides:
        merged = _deep_merge(merged, overrides)
    capabilities = profile.get("capabilities", [])
    if isinstance(capabilities, list):
        merged["capabilities"] = [item for item in capabilities if item]
    else:
        merged["capabilities"] = []
    return merged


def get_active_profile_name(config: dict[str, Any]) -> str:
    name = config.get("active_profile", "default")
    return name or "default"


__all__ = [
    "DEFAULT_PROFILES_DIR",
    "load_profile",
    "apply_profile",
    "get_active_profile_name",
]
