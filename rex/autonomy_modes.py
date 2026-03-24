"""Autonomy modes for Rex - control automatic workflow execution.

This module defines autonomy modes that control how workflows are executed:
- OFF: No automatic execution, user must explicitly run each step
- SUGGEST: Generate plan and show it to user, require confirmation before execution
- AUTO: Automatically execute workflows that pass policy checks

The autonomy mode is determined by workflow category (e.g., "email.newsletter",
"os.command") and can be configured in config/autonomy.json.

Usage:
    from rex.autonomy_modes import get_mode, AutonomyMode
    from rex.workflow import Workflow

    workflow = Workflow(...)
    mode = get_mode(workflow)

    if mode == AutonomyMode.AUTO:
        # Execute immediately if policy allows
        executor.run()
    elif mode == AutonomyMode.SUGGEST:
        # Show plan and wait for approval
        print(f"Plan: {workflow.title}")
        # ... wait for user confirmation ...
    else:
        # Don't execute automatically
        print("Autonomy is OFF for this workflow")
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path

from rex.workflow import Workflow

logger = logging.getLogger(__name__)

# Default config path
DEFAULT_AUTONOMY_CONFIG = Path("config/autonomy.json")


class AutonomyMode(str, Enum):
    """Autonomy mode for workflow execution.

    - OFF: No automatic execution, user must explicitly trigger
    - SUGGEST: Generate plan, show to user, require confirmation
    - AUTO: Automatically execute if policy allows
    """

    OFF = "off"
    SUGGEST = "suggest"
    AUTO = "auto"


class AutonomyConfig:
    """Configuration for autonomy modes.

    Maps workflow categories to autonomy modes. Categories are hierarchical
    patterns like "email.newsletter" or "os.*".

    Attributes:
        config: Dict mapping categories to modes
        default_mode: Default mode for uncategorized workflows
    """

    def __init__(
        self,
        config: dict[str, str] | None = None,
        default_mode: AutonomyMode = AutonomyMode.SUGGEST,
    ):
        """Initialize autonomy config.

        Args:
            config: Dict mapping category patterns to mode strings
            default_mode: Default mode if no category matches
        """
        self.config = config or {}
        self.default_mode = default_mode

    def get_mode(self, category: str) -> AutonomyMode:
        """Get autonomy mode for a category.

        Supports exact matches and wildcard patterns (e.g., "email.*").

        Args:
            category: Category string (e.g., "email.newsletter")

        Returns:
            AutonomyMode for this category
        """
        # Try exact match first
        if category in self.config:
            return self._parse_mode(self.config[category])

        # Try wildcard matches (e.g., "email.*" matches "email.newsletter")
        parts = category.split(".")
        for i in range(len(parts), 0, -1):
            # Try progressively shorter prefixes with wildcards
            pattern = ".".join(parts[:i]) + ".*"
            if pattern in self.config:
                return self._parse_mode(self.config[pattern])

            # Try without wildcard
            pattern = ".".join(parts[:i])
            if pattern in self.config:
                return self._parse_mode(self.config[pattern])

        return self.default_mode

    def set_mode(self, category: str, mode: AutonomyMode | str) -> None:
        """Set autonomy mode for a category.

        Args:
            category: Category string
            mode: Mode to set (AutonomyMode or string)
        """
        if isinstance(mode, AutonomyMode):
            self.config[category] = mode.value
        else:
            # Validate it's a valid mode
            self._parse_mode(mode)
            self.config[category] = mode

    def _parse_mode(self, mode_str: str) -> AutonomyMode:
        """Parse mode string to AutonomyMode enum.

        Args:
            mode_str: Mode string ("off", "suggest", or "auto")

        Returns:
            AutonomyMode enum value

        Raises:
            ValueError: If mode string is invalid
        """
        mode_str = mode_str.lower()
        try:
            return AutonomyMode(mode_str)
        except ValueError:
            logger.warning("Invalid autonomy mode '%s', using default", mode_str)
            return self.default_mode

    def save(self, path: Path | str | None = None) -> None:
        """Save autonomy config to JSON file.

        Args:
            path: Path to save config. Defaults to config/autonomy.json.
        """
        if path is None:
            path = DEFAULT_AUTONOMY_CONFIG

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "default_mode": self.default_mode.value,
            "categories": self.config,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved autonomy config to %s", path)

    @classmethod
    def load(cls, path: Path | str | None = None) -> AutonomyConfig:
        """Load autonomy config from JSON file.

        Args:
            path: Path to load config from. Defaults to config/autonomy.json.

        Returns:
            AutonomyConfig instance
        """
        if path is None:
            path = DEFAULT_AUTONOMY_CONFIG

        path = Path(path)

        if not path.exists():
            logger.info("Autonomy config not found at %s, using defaults", path)
            return cls()

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            default_mode_str = data.get("default_mode", "suggest")
            default_mode = AutonomyMode(default_mode_str.lower())

            config = data.get("categories", {})

            logger.info("Loaded autonomy config from %s", path)
            return cls(config=config, default_mode=default_mode)

        except Exception as e:
            logger.warning("Failed to load autonomy config from %s: %s", path, e)
            return cls()


# Global config instance
_autonomy_config: AutonomyConfig | None = None


def get_autonomy_config() -> AutonomyConfig:
    """Get the global autonomy config instance.

    Loads from config/autonomy.json on first call.

    Returns:
        Global AutonomyConfig instance
    """
    global _autonomy_config
    if _autonomy_config is None:
        _autonomy_config = AutonomyConfig.load()
    return _autonomy_config


def set_autonomy_config(config: AutonomyConfig) -> None:
    """Set the global autonomy config instance.

    Useful for testing or custom configurations.

    Args:
        config: AutonomyConfig to set as global
    """
    global _autonomy_config
    _autonomy_config = config


def reset_autonomy_config() -> None:
    """Reset the global autonomy config instance.

    Forces reload from config file on next get_autonomy_config() call.
    """
    global _autonomy_config
    _autonomy_config = None


def get_mode(workflow: Workflow) -> AutonomyMode:
    """Get autonomy mode for a workflow.

    Determines the category from the workflow title and tool calls,
    then looks up the mode in the autonomy config.

    Args:
        workflow: Workflow to get mode for

    Returns:
        AutonomyMode for this workflow
    """
    config = get_autonomy_config()

    # Determine category from workflow
    category = _infer_category(workflow)

    return config.get_mode(category)


def _infer_category(workflow: Workflow) -> str:
    """Infer workflow category from its properties.

    Categories are hierarchical strings like:
    - "email.newsletter"
    - "email.send"
    - "calendar.event"
    - "home.control"
    - "os.command"
    - "web.search"

    Args:
        workflow: Workflow to categorize

    Returns:
        Category string
    """
    # Check tool calls to determine category
    tools_used = set()
    for step in workflow.steps:
        if step.tool_call:
            tools_used.add(step.tool_call.tool)

    # Email category
    if "send_email" in tools_used:
        title_lower = workflow.title.lower()
        if "newsletter" in title_lower:
            return "email.newsletter"
        elif "report" in title_lower:
            return "email.report"
        else:
            return "email.send"

    # Calendar category
    if "calendar_create_event" in tools_used:
        return "calendar.event"

    # Home automation category
    if "home_assistant_call_service" in tools_used:
        return "home.control"

    # OS command category
    if "execute_command" in tools_used:
        return "os.command"

    # Web search category
    if "web_search" in tools_used:
        return "web.search"

    # Weather category
    if "weather_now" in tools_used:
        return "info.weather"

    # Time category
    if "time_now" in tools_used:
        return "info.time"

    # File operations
    if any(tool in tools_used for tool in ("file_read", "file_write", "file_delete")):
        return "fs.operation"

    # Browser automation
    if any(
        tool in tools_used for tool in ("browser_navigate", "browser_click", "browser_screenshot")
    ):
        return "browser.automation"

    # Default: general workflow
    return "general.workflow"


def create_default_config() -> AutonomyConfig:
    """Create a default autonomy configuration.

    This sets up sensible defaults:
    - AUTO: Low-risk operations (weather, time, web search)
    - SUGGEST: Medium-risk operations (email, calendar, home automation)
    - OFF: High-risk operations (OS commands, file operations)

    Returns:
        AutonomyConfig with default settings
    """
    config = AutonomyConfig(default_mode=AutonomyMode.SUGGEST)

    # AUTO modes - low risk
    config.set_mode("info.*", AutonomyMode.AUTO)
    config.set_mode("web.search", AutonomyMode.AUTO)

    # SUGGEST modes - medium risk
    config.set_mode("email.*", AutonomyMode.SUGGEST)
    config.set_mode("calendar.*", AutonomyMode.SUGGEST)
    config.set_mode("home.*", AutonomyMode.SUGGEST)
    config.set_mode("browser.*", AutonomyMode.SUGGEST)

    # OFF modes - high risk
    config.set_mode("os.*", AutonomyMode.OFF)
    config.set_mode("fs.*", AutonomyMode.OFF)

    return config


__all__ = [
    "AutonomyMode",
    "AutonomyConfig",
    "get_autonomy_config",
    "set_autonomy_config",
    "reset_autonomy_config",
    "get_mode",
    "create_default_config",
]
