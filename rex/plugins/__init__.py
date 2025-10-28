"""Plugin loader for Rex with safety and sandboxing measures."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import os
import signal
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, List, Protocol

logger = logging.getLogger(__name__)

# Plugin safety configuration from environment
PLUGIN_TIMEOUT = int(os.getenv("REX_PLUGIN_TIMEOUT", "30"))
PLUGIN_OUTPUT_LIMIT = int(os.getenv("REX_PLUGIN_OUTPUT_LIMIT", "1048576"))  # 1MB
PLUGIN_RATE_LIMIT = int(os.getenv("REX_PLUGIN_RATE_LIMIT", "10"))  # requests per minute


class Plugin(Protocol):
    name: str

    def initialize(self) -> None:
        ...

    def process(self, *args, **kwargs):
        ...

    def shutdown(self) -> None:
        ...


@dataclass
class PluginSpec:
    name: str
    plugin: Plugin


class PluginSafetyWrapper:
    """Wrapper that adds safety measures to plugin execution."""

    def __init__(self, plugin: Plugin, name: str) -> None:
        self.plugin = plugin
        self.name = name
        self._call_times: deque = deque(maxlen=PLUGIN_RATE_LIMIT)

    def _check_rate_limit(self) -> bool:
        """Check if the plugin is being called too frequently."""
        now = time.time()
        # Remove calls older than 60 seconds
        while self._call_times and now - self._call_times[0] > 60:
            self._call_times.popleft()

        if len(self._call_times) >= PLUGIN_RATE_LIMIT:
            logger.warning("Plugin %s rate limit exceeded (%d/min)", self.name, PLUGIN_RATE_LIMIT)
            return False

        self._call_times.append(now)
        return True

    def _truncate_output(self, output: Any) -> Any:
        """Limit the size of plugin output to prevent memory exhaustion."""
        if output is None:
            return None

        if isinstance(output, str):
            if len(output) > PLUGIN_OUTPUT_LIMIT:
                logger.warning(
                    "Plugin %s output truncated from %d to %d bytes",
                    self.name, len(output), PLUGIN_OUTPUT_LIMIT
                )
                return output[:PLUGIN_OUTPUT_LIMIT] + "\n[...output truncated]"
            return output

        if isinstance(output, (list, dict)):
            import json
            try:
                serialized = json.dumps(output)
                if len(serialized) > PLUGIN_OUTPUT_LIMIT:
                    logger.warning("Plugin %s output exceeds size limit", self.name)
                    return "[Output too large]"
                return output
            except (TypeError, ValueError):
                return str(output)[:PLUGIN_OUTPUT_LIMIT]

        return str(output)[:PLUGIN_OUTPUT_LIMIT]

    def _sanitize_for_tts(self, output: Any) -> Any:
        """Sanitize output to prevent TTS injection attacks."""
        if not isinstance(output, str):
            return output

        # Remove potentially problematic characters that might affect TTS
        import re
        # Remove control characters except newlines and tabs
        sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', output)
        # Limit repeated characters (potential DoS via TTS)
        sanitized = re.sub(r'(.)\1{10,}', r'\1' * 10, sanitized)
        return sanitized

    def process(self, *args, **kwargs) -> Any:
        """Execute plugin with safety measures.

        NOTE: Timeout enforcement uses ThreadPoolExecutor with non-blocking shutdown.
        Threads cannot be forcefully terminated, so a truly hung plugin thread will
        remain running in the background. For mission-critical timeout enforcement,
        consider using multiprocessing.Process with terminate().
        """
        # Check rate limit
        if not self._check_rate_limit():
            raise RuntimeError(f"Plugin {self.name} rate limit exceeded")

        # Execute with timeout - DO NOT use context manager as it blocks on shutdown
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.plugin.process, *args, **kwargs)

        executor_shutdown = False
        try:
            result = future.result(timeout=PLUGIN_TIMEOUT)
        except concurrent.futures.TimeoutError:
            logger.error("Plugin %s timed out after %d seconds", self.name, PLUGIN_TIMEOUT)
            # Shutdown without waiting - abandon the hung thread to prevent blocking
            executor.shutdown(wait=False)
            executor_shutdown = True
            future.cancel()  # Attempt to cancel (may not work if already running)
            raise RuntimeError(f"Plugin {self.name} execution timed out")
        except Exception as exc:
            logger.exception("Plugin %s raised an exception", self.name)
            executor.shutdown(wait=False)
            executor_shutdown = True
            raise
        finally:
            # Clean shutdown for successful executions
            if not executor_shutdown:
                executor.shutdown(wait=False)

        # Truncate and sanitize output
        result = self._truncate_output(result)
        result = self._sanitize_for_tts(result)
        return result

    def __getattr__(self, name: str) -> Any:
        """Forward all other attributes to the wrapped plugin."""
        return getattr(self.plugin, name)


def load_plugins(path: str = "plugins") -> List[PluginSpec]:
    specs: List[PluginSpec] = []
    if not os.path.isdir(path):
        return specs

    for entry in os.listdir(path):
        if not entry.endswith(".py") or entry.startswith("_"):
            continue
        module_name = entry[:-3]
        if os.path.isabs(path):
            loader = importlib.machinery.SourceFileLoader(module_name, os.path.join(path, entry))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            if spec is None or spec.loader is None:
                logger.warning("Could not load plugin from %s", entry)
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Failed to exec plugin %s: %s", entry, exc)
                continue
            import_path = module_name
        else:
            module_path = f"{path.replace(os.sep, '.')}" if os.sep in path else path
            import_path = f"{module_path}.{module_name}" if module_path else module_name
            try:
                module = importlib.import_module(import_path)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Failed to import plugin %s: %s", import_path, exc)
                continue

        register = getattr(module, "register", None)
        if register is None or not callable(register):
            logger.warning("Plugin %s does not expose register(); skipping", import_path)
            continue

        plugin_obj = register()
        if not hasattr(plugin_obj, "process"):
            logger.warning("Plugin %s returned invalid object", import_path)
            continue

        try:
            plugin_obj.initialize()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Plugin %s failed to initialise: %s", plugin_obj, exc)
            continue

        # Wrap plugin with safety measures
        plugin_name = getattr(plugin_obj, "name", module_name)
        wrapped_plugin = PluginSafetyWrapper(plugin_obj, plugin_name)
        specs.append(PluginSpec(name=plugin_name, plugin=wrapped_plugin))
    return specs


def shutdown_plugins(specs: Iterable[PluginSpec]) -> None:
    for spec in specs:
        try:
            spec.plugin.shutdown()
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Error while shutting down plugin %s", spec.name)


__all__ = ["Plugin", "PluginSpec", "PluginSafetyWrapper", "load_plugins", "shutdown_plugins"]
