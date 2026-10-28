"""TASK-272: Settings > Tools (descape/tool_defaults_page.py), through
conftest.dialog_and_window(). Every store assertion reads the `tool_options`
key only; see tests/test_tool_options_viewer.py for why.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from descape import settings, tool_options

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _stored(tmp_path: Path):
    path = tmp_path / "config.yaml"
    return (yaml.safe_load(path.read_text()) or {}).get("tool_options") if path.is_file() else None


@pytest.fixture
def dialog():
    dialog, window = conftest.dialog_and_window()
    yield dialog, dialog.tool_defaults_page, window
    dialog.close()
    conftest.close_window(window)


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def test_one_row_per_option_grouped_in_groups_order(dialog) -> None:
    _dialog, page, _window = dialog
    assert [group_id for group_id, _header, _names in page.sections] == [g for g, _t in tool_options.GROUPS]
    assert [header.text() for _g, header, _n in page.sections] == [f"<b>{t}</b>" for _g, t in tool_options.GROUPS]
    in_tab_order = [name for _g, _h, names in page.sections for name in names]
    assert sorted(in_tab_order) == sorted(spec.name for spec in tool_options.OPTIONS)
    assert len(in_tab_order) == len(set(in_tab_order))
    assert list(page.controls) == in_tab_order and list(page.default_buttons) == in_tab_order


def test_the_tab_sits_between_appearance_and_saving(dialog) -> None:
    from PyQt5.QtWidgets import QTabWidget

    settings_dialog, _page, _window = dialog
    tabs = settings_dialog.findChild(QTabWidget)
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert titles[titles.index("Appearance") + 1] == "Tools"
    assert titles[titles.index("Tools") + 1] == "Saving"


def test_opening_and_closing_the_dialog_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    stored = {
        "brush_size": 5, "eraser_targets": ["trees"], "last_wall": 1, "beach_terrain": 999,
        "cliff_family": "no_such_cliff.sld", "cliff_frame": 3,
    }
    settings.set_tool_option_defaults(stored)
    monkeypatch.setattr(settings, "_tool_option_overrides", None)
    before = _stored(tmp_path)
    settings_dialog, window = conftest.dialog_and_window()
    try:
        page = settings_dialog.tool_defaults_page
        assert page.controls["last_wall"].currentText() == "Unavailable (1)"
        assert page.controls["beach_terrain"].currentText() == "Unavailable (999)"
        assert page.controls["cliff_family"].currentText() == "Unavailable (no_such_cliff.sld)"
        assert page.controls["cliff_piece"].count() == 0, "an unavailable family has no pieces to list"
        assert page.controls["brush_size"].value() == 5
        settings_dialog.show()
        settings_dialog.close()
    finally:
        conftest.close_window(window)
    assert _stored(tmp_path) == before


def test_a_row_edit_reaches_the_store_the_toolbar_and_the_status_line(tmp_path: Path, dialog) -> None:
    _dialog, page, window = dialog
    page.controls["brush_size"].setValue(5)
    assert _stored(tmp_path) == {"brush_size": 5}
    assert window.brush_size_spin.value() == 5
    assert _last_status(window) == "Brush size default: 5"

    page.controls["free_placement"].setChecked(True)
    assert window.unrestricted_check.isChecked()
    assert _last_status(window) == "Free placement default: on"

    combo = page.controls["brush_shape"]
    combo.setCurrentIndex(combo.findData("circle"))
    assert window.brush_shape_combo.currentData() == "circle"
    assert _last_status(window) == "Brush shape default: Circle"
    assert _stored(tmp_path) == {"brush_size": 5, "brush_shape": "circle", "free_placement": True}


def test_a_set_row_stores_its_ticked_members_in_schema_order(tmp_path: Path, dialog) -> None:
    _dialog, page, window = dialog
    button = page.controls["eraser_targets"]
    actions = {action.text(): action for action in button.menu().actions()}
    actions["Cliffs"].setChecked(True)
    actions["Trees"].setChecked(True)
    assert _stored(tmp_path) == {"eraser_targets": ["units", "buildings", "trees", "cliffs"]}
    assert window.eraser_target_actions["cliffs"].isChecked()
    assert button.text() == "Units, Buildings, Trees, Cliffs"
    assert _last_status(window) == "Eraser targets default: Units, Buildings, Trees, Cliffs"


def test_default_restores_the_control_and_prunes_the_entry(tmp_path: Path, dialog) -> None:
    _dialog, page, window = dialog
    page.controls["brush_size"].setValue(5)
    page.controls["paint_trees"].setChecked(False)
    page.default_buttons["brush_size"].click()
    assert page.controls["brush_size"].value() == 1
    assert window.brush_size_spin.value() == 1
    assert _stored(tmp_path) == {"paint_trees": False}
    page.default_buttons["paint_trees"].click()
    assert _stored(tmp_path) is None
    assert window.paint_trees_check.isChecked()


def test_default_clears_an_unavailable_id(tmp_path: Path, monkeypatch) -> None:
    settings.set_tool_option_defaults({"last_wall": 1})
    monkeypatch.setattr(settings, "_tool_option_overrides", None)
    settings_dialog, window = conftest.dialog_and_window()
    try:
        page = settings_dialog.tool_defaults_page
        page.default_buttons["last_wall"].click()
        assert page.controls["last_wall"].currentData() == tool_options.default("last_wall")
        assert _stored(tmp_path) is None
    finally:
        settings_dialog.close()
        conftest.close_window(window)


def test_load_from_toolbar_stores_exactly_the_values_that_differ(tmp_path: Path, dialog) -> None:
    _dialog, page, window = dialog
    window.brush_size_spin.setValue(6)
    window.off_grid_check.setChecked(True)
    window.convert_source_actions[0].setChecked(False)
    assert _stored(tmp_path) is None, "a toolbar edit writes nothing"
    page.load_from_toolbar_button.click()
    assert _stored(tmp_path) == {"brush_size": 6, "off_grid": True, "convert_sources": [1, 2, 3, 4, 5, 6, 7, 8]}
    assert page.controls["brush_size"].value() == 6
    assert page.controls["off_grid"].isChecked()
    assert _last_status(window) == "Tool defaults loaded from the toolbar"


def test_reset_all_empties_the_store_and_resets_the_row(tmp_path: Path, dialog) -> None:
    _dialog, page, window = dialog
    page.controls["brush_size"].setValue(4)
    page.controls["gate_walls"].setChecked(True)
    window.paste_units_check.setChecked(False)  # a session-only edit, reset too
    page.reset_all_button.click()
    assert _stored(tmp_path) is None
    values = window.tool_option_values()
    assert {name: value for name, value in values.items() if not tool_options.is_default(name, value)} == {}
    assert page.controls["brush_size"].value() == 1 and not page.controls["gate_walls"].isChecked()
    assert _last_status(window) == "All tool defaults reset"


def test_changing_the_cliff_family_clears_the_stored_piece_and_frame_on_the_row_too(tmp_path: Path, dialog) -> None:
    _dialog, page, window = dialog
    pieces = page.controls["cliff_piece"]
    pieces.setCurrentIndex(1)
    page.controls["cliff_frame"].setValue(2)
    piece = pieces.currentData()
    assert _stored(tmp_path) == {"cliff_piece": piece, "cliff_frame": 2}
    assert (window.cliff_piece_combo.currentData(), window.cliff_frame_spin.value()) == (piece, 2)

    families = page.controls["cliff_family"]
    families.setCurrentIndex(2)
    family = families.currentData()
    assert _stored(tmp_path) == {"cliff_family": family}
    assert window.cliff_family_combo.currentData() == family
    assert (window.cliff_piece_combo.currentIndex(), window.cliff_frame_spin.value()) == (0, 0)
    assert (pieces.currentIndex(), page.controls["cliff_frame"].value()) == (0, 0)
    assert [pieces.itemText(i) for i in range(pieces.count())] == [
        window.cliff_piece_combo.itemText(i) for i in range(window.cliff_piece_combo.count())
    ], "the tab's pieces follow its family"


def test_changing_the_cliff_piece_resets_the_frame_and_re_ranges_it(tmp_path: Path, dialog) -> None:
    from descape import cliff_catalog

    _dialog, page, window = dialog
    page.controls["cliff_frame"].setValue(3)
    pieces = page.controls["cliff_piece"]
    pieces.setCurrentIndex(1)
    assert _stored(tmp_path) == {"cliff_piece": pieces.currentData()}
    assert page.controls["cliff_frame"].value() == 0 and window.cliff_frame_spin.value() == 0
    assert page.controls["cliff_frame"].maximum() == cliff_catalog.frame_count(pieces.currentData()) - 1
