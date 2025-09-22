from __future__ import annotations

from plugin_loader import load_plugins


def test_load_plugins(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "test_plugins"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("# package marker\n", encoding="utf-8")
    (plugin_dir / "example.py").write_text(
        "def register():\n    return {'capability': 'ok'}\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    capabilities = load_plugins(str(plugin_dir))
    assert capabilities
    module_name = f"{plugin_dir.name}.example"
    assert module_name in capabilities
    assert capabilities[module_name]["capability"] == "ok"
