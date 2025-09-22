"""Lightweight fallback implementations of the bits of Pydantic we rely on."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, Union, get_args, get_origin

_MISSING: object = object()


class FieldInfo:
    """Container describing a settings field.

    Only a *very* small subset of :func:`pydantic.Field` is implemented – just
    enough for the configuration helper to express defaults and factories.
    """

    def __init__(
        self,
        default: Any = _MISSING,
        *,
        default_factory: Callable[[], Any] | None = None,
        **metadata: Any,
    ) -> None:
        if default is not _MISSING and default_factory is not None:
            raise TypeError("default and default_factory are mutually exclusive")
        self.default = default
        self.default_factory = default_factory
        self.metadata = metadata


def Field(default: Any = _MISSING, *, default_factory: Callable[[], Any] | None = None, **metadata: Any) -> FieldInfo:
    """Return a :class:`FieldInfo` mirroring :func:`pydantic.Field`."""

    return FieldInfo(default=default, default_factory=default_factory, **metadata)


def validator(*field_names: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator used by :class:`BaseSettings` to register validators."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "__validator_fields__", field_names)
        return func

    return decorator


class BaseSettingsMeta(type):
    """Collect annotations, defaults, and validators for settings classes."""

    def __new__(mcls, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]):
        annotations: Dict[str, Any] = {}
        fields: Dict[str, FieldInfo] = {}
        validators: List[Callable[..., Any]] = []

        for base in reversed(bases):
            annotations.update(getattr(base, "_settings_annotations", {}))
            fields.update(getattr(base, "_settings_fields", {}))
            validators.extend(getattr(base, "_settings_validators", []))

        own_annotations: Dict[str, Any] = namespace.get("__annotations__", {})
        annotations.update(own_annotations)

        for attr_name, attr_value in list(namespace.items()):
            if callable(attr_value) and hasattr(attr_value, "__validator_fields__"):
                validators.append(attr_value)

        for field_name in own_annotations:
            value = namespace.get(field_name, _MISSING)
            if isinstance(value, FieldInfo):
                field_info = value
                default_value = field_info.default if field_info.default is not _MISSING else None
                namespace[field_name] = default_value
            elif value is not _MISSING:
                field_info = FieldInfo(default=value)
            elif field_name in fields:
                # Inherit the field definition from the base class.
                field_info = fields[field_name]
            else:
                field_info = FieldInfo(default=_MISSING)
                namespace[field_name] = None
            fields[field_name] = field_info

        namespace["_settings_annotations"] = annotations
        namespace["_settings_fields"] = fields
        namespace["_settings_validators"] = validators

        return super().__new__(mcls, name, bases, namespace)


class BaseSettings(metaclass=BaseSettingsMeta):
    """Extremely small subset of :class:`pydantic.BaseSettings` behaviour."""

    class Config:
        env_file: str | None = None
        env_file_encoding: str = "utf-8"

    def __init__(self, **overrides: Any) -> None:
        env_file_values = self._load_env_file()
        for name, info in self._settings_fields.items():
            raw = self._select_value(name, info, overrides, env_file_values)
            converted = self._convert_value(name, raw)
            setattr(self, name, converted)
        self._run_validators()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _load_env_file(self) -> Dict[str, str]:
        config = getattr(self, "Config", None)
        if not config:
            return {}
        env_file = getattr(config, "env_file", None)
        if not env_file:
            return {}
        encoding = getattr(config, "env_file_encoding", "utf-8")

        try:
            from dotenv import dotenv_values  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            path = Path(env_file)
            if not path.exists():
                return {}
            values: Dict[str, str] = {}
            try:
                with path.open(encoding=encoding) as handle:
                    for line in handle:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        values[key.strip()] = value.strip().strip('"').strip("'")
            except OSError:
                return {}
            return values
        else:  # pragma: no cover - thin wrapper around python-dotenv
            loaded = dotenv_values(env_file, encoding=encoding)
            return {key: value for key, value in loaded.items() if value is not None}

    def _select_value(
        self,
        name: str,
        info: FieldInfo,
        overrides: Dict[str, Any],
        env_file_values: Dict[str, str],
    ) -> Any:
        if name in overrides:
            return overrides[name]
        env_key = name.upper()
        if env_key in os.environ:
            return os.environ[env_key]
        if name in env_file_values:
            return env_file_values[name]
        if info.default_factory is not None:
            return info.default_factory()
        if info.default is not _MISSING:
            return info.default
        return None

    def _convert_value(self, name: str, value: Any) -> Any:
        if value is None:
            return None
        annotation = self._settings_annotations.get(name)
        if annotation is None:
            return value

        origin = get_origin(annotation)
        if origin is Union:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if not args:
                return None
            return self._coerce(args[0], value)
        if origin in (list, List):
            args = get_args(annotation)
            inner = args[0] if args else Any
            if isinstance(value, str):
                items = [item.strip() for item in value.split(",") if item.strip()]
            elif isinstance(value, (list, tuple, set)):
                items = list(value)
            else:
                items = [value]
            return [self._coerce(inner, item) for item in items]
        return self._coerce(annotation, value)

    def _coerce(self, target: Type[Any], value: Any) -> Any:
        if target in (Any, object):
            return value
        if target is str:
            return str(value)
        if target is int:
            return int(value)
        if target is float:
            return float(value)
        if target is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "on", "y"}:
                return True
            if text in {"0", "false", "no", "off", "n"}:
                return False
            raise ValueError(f"Cannot interpret '{value}' as boolean")
        return value

    def _run_validators(self) -> None:
        for validator_func in self._settings_validators:
            for field in getattr(validator_func, "__validator_fields__", ()):  # pragma: no cover - defensive
                current = getattr(self, field)
                updated = validator_func(self.__class__, current)
                setattr(self, field, updated)


__all__ = ["BaseSettings", "Field", "validator"]
