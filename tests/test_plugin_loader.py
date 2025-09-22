from __future__ import annotations

from rex.plugins.base import PluginContext
from rex.plugins.loader import load_plugins


def test_load_plugins(tmp_path, monkeypatch):
    package_dir = tmp_path / "test_plugins"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("# package marker\n", encoding="utf-8")
    (package_dir / "example.py").write_text(
        "from rex.plugins.base import PluginContext\n"
        "class ExamplePlugin:\n"
        "    name = 'example'\n"
        "    def initialise(self):\n        return None\n"
        "    def process(self, context: PluginContext):\n        return context.text.upper()\n"
        "    def shutdown(self):\n        return None\n"
        "PLUGIN = ExamplePlugin()\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    plugins = load_plugins(package_dir.name)
    assert "example" in plugins
    context = PluginContext(user_id="user", text="hello")
    assert plugins["example"].process(context) == "HELLO"
