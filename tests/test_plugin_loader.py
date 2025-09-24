from __future__ import annotations

import textwrap

from rex.plugins import load_plugins as load_plugin_specs, shutdown_plugins
from plugin_loader import load_plugins


def test_load_plugins_discovers_and_initialises(tmp_path, monkeypatch):
    plugin_file = tmp_path / "plugins" / "demo.py"
    plugin_file.parent.mkdir()
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

    specs = load_plugin_specs(str(plugin_file.parent))
    assert len(specs) == 1
    plugin = specs[0].plugin
    assert plugin.initialised
    assert plugin.process("hi") == "HI"

    shutdown_plugins(specs)
    assert plugin.shut_down


def test_load_dict_style_plugins(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "test_plugins"
    plugin_dir.mkdir()

    # Make it a Python package
    (plugin_dir / "__init__.py").write_text("# package marker\n", encoding="utf-8")

    # Add one valid plugin
    (plugin_dir / "example.py").write_text(
        textwrap.dedent(
            """
            def register():
                return {"capability": "ok"}
            """
        ),
        encoding="utf-8",
    )

    # Add a second plugin
    (plugin_dir / "second.py").write_text(
        textwrap.dedent(
            """
            def register():
                return {"feature": "active"}
            """
        ),
        encoding="utf-8",
    )

    # Add a broken plugin (no register function)
    (plugin_dir / "broken.py").write_text(
        textwrap.dedent(
            """
            def init():
                return {"invalid": True}
            """
        ),
        encoding="utf-8",
    )

    # Add to sys.path so it can be imported
    monkeypatch.syspath_prepend(str(tmp_path))

    # Load plugins
    capabilities = load_plugins(str(plugin_dir))

    # Assert valid plugins loaded
    assert isinstance(capabilities, dict), "Plugin loader should return a dict"
    assert f"{plugin_dir.name}.example" in capabilities
    assert capabilities[f"{plugin_dir.name}.example"]["capability"] == "ok"

    assert f"{plugin_dir.name}.second" in capabilities
    assert capabilities[f"{plugin_dir.name}.second"]["feature"] == "active"

    # Broken plugin should be skipped
    assert f"{plugin_dir.name}.broken" not in capabilities

