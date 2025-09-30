from __future__ import annotations

import textwrap

from plugin_loader import load_plugins as load_dict_plugins
from rex.plugins import load_plugins as load_rex_plugins, shutdown_plugins


def test_class_based_plugin_loads_and_runs(tmp_path, monkeypatch):
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

    specs = load_rex_plugins(str(plugin_file.parent))
    assert len(specs) == 1

    plugin = specs[0].plugin
    assert plugin.initialised
    assert plugin.process("hi") == "HI"

    shutdown_plugins(specs)
    assert plugin.shut_down


def test_dict_based_plugin_loader(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "test_plugins"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("# marker", encoding="utf-8")

    (plugin_dir / "example.py").write_text(
        textwrap.dedent(
            """
            def register():
                return {"capability": "ok"}
            """
        ),
        encoding="utf-8",
    )

    (plugin_dir / "second.py").write_text(
        textwrap.dedent(
            """
            def register():
                return {"feature": "active"}
            """
        ),
        encoding="utf-8",
    )

    (plugin_dir / "broken.py").write_text(
        textwrap.dedent(
            """
            def init():
                return {"invalid": True}
            """
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(plugin_dir.parent))
    results = load_dict_plugins(str(plugin_dir))

    assert isinstance(results, dict)
    assert f"{plugin_dir.name}.example" in results
    assert results[f"{plugin_dir.name}.example"]["capability"] == "ok"

    assert f"{plugin_dir.name}.second" in results
    assert results[f"{plugin_dir.name}.second"]["feature"] == "active"

    assert f"{plugin_dir.name}.broken" not in results
