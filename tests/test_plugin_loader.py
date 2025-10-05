from __future__ import annotations

import textwrap
import pytest

# Codex-style plugin test (class-based with lifecycle methods)
from rex.plugins import load_plugins as load_rex_plugins, shutdown_plugins

# Master-style plugin loader test (dict-based plugin discovery)
from plugin_loader import load_plugins as load_dict_plugins


def test_class_based_plugin_loads_and_runs(tmp_path, monkeypatch):
    """Test that class-based plugins are loaded, initialized, and shut down correctly."""
    plugin_file = tmp_path / "plugins" / "demo.py"
    plugin_file.parent.mkdir(parents=True)

    plugin_file.write_text(
        textwrap.dedent(
            """
            from rex.plugins import Plugin

            class DemoPlugin:
                name = "demo"

                def __init__(self):
                    self.initialised = False
                    self.shut_down = False

                def initialize(self):
                    self.initialised = True

                def process(self, value):
                    return value.upper()

                def shutdown(self):
                    self.shut_down = True

            def register() -> Plugin:
                return DemoPlugin()
            """
        )
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_rex_plugins(str(plugin_file.parent))

    assert len(specs) == 1
    plugin = specs[0].plugin
    assert plugin.initialised
    assert plugin.process("hi") == "HI"

    shutdown_plugins(specs)
    assert plugin.shut_down


def test_dict_based_plugin_loader(tmp_path, monkeypatch):
    """Test that dict-based plugins are discovered and registered properly."""
    plugin_dir = tmp_path / "test_plugins"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("# marker")

    # Valid plugin 1
    (plugin_dir / "example.py").write_text(
        textwrap.dedent(
            """
            def register():
                return {"capability": "ok"}
            """
        ),
        encoding="utf-8",
    )

    # Valid plugin 2
    (plugin_dir / "second.py").write_text(
        textwrap.dedent(
            """
            def register():
                return {"feature": "active"}
            """
        ),
        encoding="utf-8",
    )

    # Invalid plugin (missing `register`)
    (plugin_dir / "broken.py").write_text(
        textwrap.dedent(
            """
            def init():
                return {"invalid": True}
            """
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    results = load_dict_plugins(str(plugin_dir))

    assert isinstance(results, dict)

    # Ensure valid plugins are loaded
    assert f"{plugin_dir.name}.example" in results
    assert results[f"{plugin_dir.name}.example"]["capability"] == "ok"

    assert f"{plugin_dir.name}.second" in results
    assert results[f"{plugin_dir.name}.second"]["feature"] == "active"

    # Ensure invalid plugin is skipped
    assert f"{plugin_dir.name}.broken" not in results
