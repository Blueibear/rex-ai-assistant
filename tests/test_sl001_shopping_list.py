"""Tests for US-SL-001: Shopping list data model and storage."""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_list(tmp_path):
    from rex.shopping_list import ShoppingList

    return ShoppingList(path=tmp_path / "shopping_list.json")


# ---------------------------------------------------------------------------
# ShoppingItem dataclass
# ---------------------------------------------------------------------------


def test_item_has_required_fields(tmp_path):
    """ShoppingItem has id, name, quantity, unit, added_by, checked, added_at, checked_at."""
    import dataclasses

    from rex.shopping_list import ShoppingItem

    fields = {f.name for f in dataclasses.fields(ShoppingItem)}
    assert {
        "id",
        "name",
        "quantity",
        "unit",
        "added_by",
        "checked",
        "added_at",
        "checked_at",
    }.issubset(fields)


def test_item_checked_defaults_false(tmp_path):
    sl = _make_list(tmp_path)
    item = sl.add_item("milk")
    assert item.checked is False
    assert item.checked_at is None


def test_item_added_at_is_set(tmp_path):
    sl = _make_list(tmp_path)
    item = sl.add_item("eggs")
    assert item.added_at is not None and len(item.added_at) > 0


# ---------------------------------------------------------------------------
# add_item()
# ---------------------------------------------------------------------------


def test_add_item_returns_item(tmp_path):
    sl = _make_list(tmp_path)
    item = sl.add_item("bread")
    assert item.name == "bread"
    assert item.id is not None


def test_add_item_default_quantity(tmp_path):
    sl = _make_list(tmp_path)
    item = sl.add_item("butter")
    assert item.quantity == 1.0


def test_add_item_custom_quantity_and_unit(tmp_path):
    sl = _make_list(tmp_path)
    item = sl.add_item("milk", quantity=2.0, unit="litres")
    assert item.quantity == 2.0
    assert item.unit == "litres"


def test_add_item_added_by(tmp_path):
    sl = _make_list(tmp_path)
    item = sl.add_item("juice", added_by="alice")
    assert item.added_by == "alice"


def test_add_duplicate_unchecked_increments_quantity(tmp_path):
    """Adding the same unchecked item again increments quantity, not duplicates."""
    sl = _make_list(tmp_path)
    sl.add_item("apples", quantity=3.0)
    second = sl.add_item("Apples", quantity=2.0)
    items = sl.list_items()
    assert len(items) == 1
    assert second.quantity == 5.0


def test_add_duplicate_after_checked_creates_new(tmp_path):
    """Adding a checked item again creates a new unchecked entry."""
    sl = _make_list(tmp_path)
    item = sl.add_item("oranges")
    sl.check_item(item.id)
    sl.add_item("oranges")
    items = sl.list_items()
    assert len(items) == 2


# ---------------------------------------------------------------------------
# check_item() / uncheck_item()
# ---------------------------------------------------------------------------


def test_check_item_marks_checked(tmp_path):
    sl = _make_list(tmp_path)
    item = sl.add_item("tea")
    result = sl.check_item(item.id)
    assert result is True
    updated = sl.get_item(item.id)
    assert updated.checked is True
    assert updated.checked_at is not None


def test_check_item_returns_false_for_unknown(tmp_path):
    sl = _make_list(tmp_path)
    assert sl.check_item("nonexistent-id") is False


def test_uncheck_item_clears_checked(tmp_path):
    sl = _make_list(tmp_path)
    item = sl.add_item("coffee")
    sl.check_item(item.id)
    result = sl.uncheck_item(item.id)
    assert result is True
    updated = sl.get_item(item.id)
    assert updated.checked is False
    assert updated.checked_at is None


def test_uncheck_item_returns_false_for_unknown(tmp_path):
    sl = _make_list(tmp_path)
    assert sl.uncheck_item("no-such-id") is False


# ---------------------------------------------------------------------------
# remove_item()
# ---------------------------------------------------------------------------


def test_remove_item_deletes_it(tmp_path):
    sl = _make_list(tmp_path)
    item = sl.add_item("yogurt")
    result = sl.remove_item(item.id)
    assert result is True
    assert sl.get_item(item.id) is None
    assert len(sl.list_items()) == 0


def test_remove_item_returns_false_for_unknown(tmp_path):
    sl = _make_list(tmp_path)
    assert sl.remove_item("ghost") is False


# ---------------------------------------------------------------------------
# list_items()
# ---------------------------------------------------------------------------


def test_list_items_returns_all_by_default(tmp_path):
    sl = _make_list(tmp_path)
    sl.add_item("a")
    item = sl.add_item("b")
    sl.check_item(item.id)
    items = sl.list_items()
    assert len(items) == 2


def test_list_items_exclude_checked(tmp_path):
    sl = _make_list(tmp_path)
    sl.add_item("x")
    item = sl.add_item("y")
    sl.check_item(item.id)
    items = sl.list_items(include_checked=False)
    assert len(items) == 1
    assert items[0].name == "x"


def test_list_items_filter_by_added_by(tmp_path):
    sl = _make_list(tmp_path)
    sl.add_item("a", added_by="alice")
    sl.add_item("b", added_by="bob")
    items = sl.list_items(added_by="alice")
    assert len(items) == 1
    assert items[0].added_by == "alice"


# ---------------------------------------------------------------------------
# clear_checked()
# ---------------------------------------------------------------------------


def test_clear_checked_removes_checked_items(tmp_path):
    sl = _make_list(tmp_path)
    sl.add_item("keep")
    item = sl.add_item("remove")
    sl.check_item(item.id)
    count = sl.clear_checked()
    assert count == 1
    items = sl.list_items()
    assert len(items) == 1
    assert items[0].name == "keep"


def test_clear_checked_returns_zero_when_none_checked(tmp_path):
    sl = _make_list(tmp_path)
    sl.add_item("item")
    assert sl.clear_checked() == 0


# ---------------------------------------------------------------------------
# Persistence — round-trip to JSON
# ---------------------------------------------------------------------------


def test_persistence_survives_reload(tmp_path):
    """Items written to JSON are correctly loaded on next construction."""
    path = tmp_path / "sl.json"

    from rex.shopping_list import ShoppingList

    sl1 = ShoppingList(path=path)
    item = sl1.add_item("pasta", quantity=500.0, unit="g", added_by="bob")
    sl1.check_item(item.id)

    # Re-create from same file
    sl2 = ShoppingList(path=path)
    items = sl2.list_items()
    assert len(items) == 1
    loaded = items[0]
    assert loaded.name == "pasta"
    assert loaded.quantity == 500.0
    assert loaded.unit == "g"
    assert loaded.added_by == "bob"
    assert loaded.checked is True
    assert loaded.checked_at is not None


def test_persistence_creates_file_and_parent(tmp_path):
    """ShoppingList creates the data directory and JSON file automatically."""
    from rex.shopping_list import ShoppingList

    path = tmp_path / "nested" / "dir" / "sl.json"
    sl = ShoppingList(path=path)
    sl.add_item("pears")
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data["items"]) == 1


def test_persistence_empty_on_missing_file(tmp_path):
    """ShoppingList starts with an empty list when the file does not exist."""
    from rex.shopping_list import ShoppingList

    path = tmp_path / "does_not_exist.json"
    sl = ShoppingList(path=path)
    assert sl.list_items() == []


def test_persistence_handles_corrupt_file(tmp_path):
    """ShoppingList starts with an empty list when the JSON is corrupt."""
    from rex.shopping_list import ShoppingList

    path = tmp_path / "corrupt.json"
    path.write_text("not valid json", encoding="utf-8")
    sl = ShoppingList(path=path)
    assert sl.list_items() == []
