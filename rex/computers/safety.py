"""Safety layer for desktop / computer control actions (US-055).

Actions are classified as ``safe`` or ``dangerous``.  Depending on the
configured ``computer_control_confirmation`` mode, dangerous (or all) actions
require confirmation from a callback before execution proceeds.

Action classification
---------------------
- **safe**:      ``read_file``, ``list_dir``, ``search_files``, ``summarize_file``
- **dangerous**: ``write_file``, ``delete_file``, ``execute_command``, ``launch_app``

Confirmation modes (``AppConfig.computer_control_confirmation``)
----------------------------------------------------------------
- ``"always"``        — confirmation required for every action
- ``"dangerous_only"``— confirmation required only for dangerous actions (default)
- ``"never"``         — no confirmation required; all actions proceed immediately

Usage example
-------------
::

    from rex.computers.safety import SafetyLayer, ActionType

    def ask_user(description: str) -> bool:
        answer = input(f"Allow: {description}? [y/N] ")
        return answer.strip().lower() == "y"

    layer = SafetyLayer(mode="dangerous_only")
    if layer.requires_confirmation("write_file"):
        if not ask_user("Write to file"):
            raise PermissionError("User denied write_file")
    write_file(path, content)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action classification
# ---------------------------------------------------------------------------

_SAFE_ACTIONS: frozenset[str] = frozenset(
    {
        "read_file",
        "list_dir",
        "search_files",
        "summarize_file",
    }
)

_DANGEROUS_ACTIONS: frozenset[str] = frozenset(
    {
        "write_file",
        "delete_file",
        "execute_command",
        "launch_app",
    }
)


class ActionType(str, Enum):
    """Classification of a computer-control action."""

    safe = "safe"
    dangerous = "dangerous"
    unknown = "unknown"


def classify_action(action_name: str) -> ActionType:
    """Return the :class:`ActionType` for *action_name*.

    Unknown action names are classified as ``dangerous`` for safety.

    Args:
        action_name: The name of the action (e.g. ``"write_file"``).

    Returns:
        :attr:`ActionType.safe`, :attr:`ActionType.dangerous`, or
        :attr:`ActionType.unknown` (treated as dangerous by callers).
    """
    key = action_name.lower().strip()
    if key in _SAFE_ACTIONS:
        return ActionType.safe
    if key in _DANGEROUS_ACTIONS:
        return ActionType.dangerous
    return ActionType.unknown


# ---------------------------------------------------------------------------
# Safety layer
# ---------------------------------------------------------------------------

_VALID_MODES = {"always", "dangerous_only", "never"}


class SafetyLayer:
    """Determines whether an action requires confirmation before execution.

    Args:
        mode: Confirmation mode — one of ``"always"``, ``"dangerous_only"``,
              ``"never"``.  Defaults to ``"dangerous_only"`` (matches
              ``AppConfig.computer_control_confirmation`` default).

    Raises:
        ValueError: If *mode* is not a recognised value.
    """

    def __init__(self, mode: str = "dangerous_only") -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid confirmation mode: {mode!r}. "
                f"Must be one of: {sorted(_VALID_MODES)}"
            )
        self._mode = mode

    @property
    def mode(self) -> str:
        """The active confirmation mode."""
        return self._mode

    def requires_confirmation(self, action_name: str) -> bool:
        """Return ``True`` if *action_name* requires confirmation in this mode.

        Args:
            action_name: The action to check (e.g. ``"write_file"``).

        Returns:
            ``True`` when confirmation is needed; ``False`` when the action
            may proceed without asking.
        """
        if self._mode == "never":
            return False
        if self._mode == "always":
            return True
        # "dangerous_only"
        action_type = classify_action(action_name)
        return action_type in (ActionType.dangerous, ActionType.unknown)

    def check(
        self,
        action_name: str,
        description: str = "",
        confirm_fn: "Callable[[str], bool] | None" = None,
    ) -> bool:
        """Check whether *action_name* is permitted to proceed.

        If confirmation is required and *confirm_fn* is provided, it is called
        with *description* and must return ``True`` to allow the action.

        If confirmation is required but *confirm_fn* is ``None``, the action is
        **denied** (returns ``False``) and a warning is logged.

        Args:
            action_name: Name of the action to check.
            description: Human-readable description passed to *confirm_fn*.
            confirm_fn:  Optional callable ``(description) -> bool`` that
                         performs the actual confirmation (voice, UI, etc.).

        Returns:
            ``True`` if the action is permitted; ``False`` if denied.
        """
        if not self.requires_confirmation(action_name):
            logger.debug("safety: %s is allowed (no confirmation needed)", action_name)
            return True

        if confirm_fn is None:
            logger.warning(
                "safety: %s requires confirmation but no confirm_fn provided — denying",
                action_name,
            )
            return False

        allowed = confirm_fn(description or action_name)
        if allowed:
            logger.info("safety: %s confirmed by user", action_name)
        else:
            logger.info("safety: %s denied by user", action_name)
        return allowed

    @classmethod
    def from_config(cls) -> "SafetyLayer":
        """Create a :class:`SafetyLayer` from the active ``AppConfig``.

        Falls back to ``"dangerous_only"`` if config cannot be loaded.
        """
        try:
            from rex.config import load_config

            cfg = load_config()
            mode = getattr(cfg, "computer_control_confirmation", "dangerous_only")
        except Exception:
            mode = "dangerous_only"
        return cls(mode=mode)


__all__ = [
    "ActionType",
    "SafetyLayer",
    "classify_action",
]
