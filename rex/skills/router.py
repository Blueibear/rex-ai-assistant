"""Skill invocation router — match user messages to registered skills.

When the user sends a message, :class:`SkillRouter` checks all enabled skills'
``trigger_patterns`` against the message before falling through to the LLM.
If a pattern matches, the skill's handler is executed and the result is
returned directly to the user.

Public API
----------
:class:`SkillRouter`
    Match and execute registered skills from user messages.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rex.skills.registry import Skill, SkillRegistry

logger = logging.getLogger(__name__)


class SkillRouter:
    """Match user messages against registered skill trigger patterns and execute them.

    Parameters:
        registry: The :class:`~rex.skills.registry.SkillRegistry` to query for
            enabled skills.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, message: str) -> Skill | None:
        """Return the first enabled skill whose trigger matches *message*, or None.

        Trigger patterns are matched as case-insensitive regular expressions.
        Plain strings without regex special characters behave as substring
        searches.

        Args:
            message: The raw user utterance.

        Returns:
            The matched :class:`~rex.skills.registry.Skill`, or ``None`` if no
            enabled skill matched.
        """
        for skill in self._registry.list_skills(include_disabled=False):
            for pattern in skill.trigger_patterns:
                try:
                    if re.search(pattern, message, re.IGNORECASE):
                        logger.debug(
                            "skill_router: message matched skill %r via pattern %r",
                            skill.name,
                            pattern,
                        )
                        return skill
                except re.error as exc:
                    logger.warning(
                        "skill_router: invalid regex in skill %r pattern %r: %s",
                        skill.name,
                        pattern,
                        exc,
                    )
        return None

    def execute(self, skill: Skill, transcript: str) -> str:
        """Execute *skill* with *transcript* and return the response string.

        The skill's ``handler`` field is treated as either:

        * A file path to a Python script that exposes a ``run(transcript)``
          function (detected by the handler containing path separators or
          ending in ``.py``).
        * A dotted module path with an optional ``:function`` suffix
          (e.g. ``rex.skills.my_skill:run``).

        Execution errors are caught, logged, and returned as a human-readable
        error message so Rex does not crash.

        Args:
            skill: The skill to execute.
            transcript: The full user utterance that triggered the skill.

        Returns:
            The skill's response string.
        """
        logger.info(
            "skill_router: executing skill %r (id=%s) for transcript %r",
            skill.name,
            skill.id,
            transcript[:80],
        )
        try:
            result = self._invoke_handler(skill.handler, transcript)
        except Exception as exc:
            logger.exception("skill_router: error executing skill %r: %s", skill.name, exc)
            return f"I tried to run the '{skill.name}' skill but it encountered an error: {exc}"
        logger.info("skill_router: skill %r completed successfully", skill.name)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _invoke_handler(handler: str, transcript: str) -> str:
        """Load and call the handler, returning a string result.

        Args:
            handler: Handler path — either a file path (``/path/to/skill.py``)
                or a dotted module path (``package.module`` or
                ``package.module:function``).
            transcript: The user utterance passed to the handler.

        Returns:
            String response from the handler's ``run()`` (or the named
            function).

        Raises:
            Exception: Any exception raised during import or execution is
                propagated to the caller (which logs and formats it).
        """
        # Determine if handler is a file path or a module path
        handler_path = Path(handler)
        if handler.endswith(".py") or handler_path.is_file():
            return SkillRouter._invoke_script(handler_path, transcript)
        return SkillRouter._invoke_module(handler, transcript)

    @staticmethod
    def _invoke_script(path: Path, transcript: str) -> str:
        """Import a script file and call its ``run(transcript)`` function."""
        spec = importlib.util.spec_from_file_location("_rex_skill_dynamic", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load skill script from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run_fn = getattr(module, "run", None)
        if not callable(run_fn):
            raise AttributeError(f"Skill script {path} has no callable 'run' function")
        result = run_fn(transcript)
        return str(result)

    @staticmethod
    def _invoke_module(handler: str, transcript: str) -> str:
        """Import a module path handler and call its function.

        Handler format: ``package.module`` (calls ``run()``) or
        ``package.module:function_name``.
        """
        if ":" in handler:
            module_path, func_name = handler.rsplit(":", 1)
        else:
            module_path, func_name = handler, "run"

        module = importlib.import_module(module_path)
        func = getattr(module, func_name, None)
        if not callable(func):
            raise AttributeError(f"Module {module_path!r} has no callable {func_name!r}")
        result = func(transcript)
        return str(result)
