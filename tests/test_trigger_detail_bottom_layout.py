"""GH #387: Trigger Details docked above or below the map lays its form out
across the dock, in the game's cells (Game layout) or in Compact columns;
docked at a side or floating it stays one column.
"""

from __future__ import annotations

import pytest
from trigger_panel_helpers import (
    _no_install_env,  # noqa: F401 -- autouse fixture
    _record_count,
    _row_widget,
    _triggers_window,
)

from descape import settings
from descape import trigger_form_layout as tfl

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# The fixture's trigger 3 carries one entry of each of these.
_TRIGGER = 3


def _settle() -> None:
    from PyQt5.QtWidgets import QApplication

    for _ in range(3):
        QApplication.processEvents()


def _bottom_window(layout: str = "game", height: int = 300):
    """Triggers mode with Trigger Details docked under the map, `height` px tall."""
    from PyQt5.QtCore import Qt

    settings.set_trigger_details_bottom_layout(layout)
    window = _triggers_window()
    window.resize(1500, 900)
    _settle()
    dock = window.trigger_detail_dock
    window.addDockWidget(Qt.BottomDockWidgetArea, dock)
    window.resizeDocks([dock], [height], Qt.Vertical)
    _settle()
    return window


def _select(panel, name: str, trigger: int = _TRIGGER) -> None:
    """Select the entry row whose name reads `name`, or the trigger row for "Trigger"."""
    panel.select_trigger(trigger)
    _settle()
    tree = panel.entry_tree
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        rows = [top] if name == "Trigger" else [top.child(c) for c in range(top.childCount())]
        for row in rows:
            if row.text(0).startswith(name):
                tree.setCurrentItem(row)
                _settle()
                return
    raise AssertionError(f"no {name} row in trigger {trigger}")


def _holds(layout, widget) -> bool:
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is not None and (item.widget() is widget or item.widget().isAncestorOf(widget)):
            return True
        if item.layout() is not None and _holds(item.layout(), widget):
            return True
    return False


def _cell_of(panel, widget) -> tuple[int, int]:
    """(column, place in that column's stack), 1-based, of the cell holding `widget`."""
    for host in panel._grid_cells:
        if host is widget or host.isAncestorOf(widget):
            column = panel.grid_layout.getItemPosition(panel.grid_layout.indexOf(host))[1] + 1
            cells = [host.layout().itemAt(i).layout() for i in range(host.layout().count())]
            cells = [cell for cell in cells if cell is not None]
            return (column, next(n for n, cell in enumerate(cells, 1) if _holds(cell, widget)))
    raise AssertionError(f"{widget!r} is in no grid cell")


def _assert_in_grid(panel) -> None:
    assert panel.property_form.rowCount() == 0, "a row stayed in the column form"
    assert panel.grid_host.isVisible() and not panel.property_host.isVisible()
    for spec, _kind, _index, widget in panel._rows:
        assert panel.grid_host.isAncestorOf(widget), spec.name


def _assert_in_column(panel) -> None:
    assert panel.detail_layout() == "column"
    assert panel.property_form.rowCount() > 0
    assert panel.property_host.isVisible() and not panel.grid_host.isVisible()
    assert panel._grid_cells == []
    for spec, _kind, _index, widget in panel._rows:
        assert panel.property_host.isAncestorOf(widget), spec.name


def test_create_object_sits_in_the_games_cells() -> None:
    window = _bottom_window()
    try:
        panel = window.trigger_panel
        assert panel.detail_layout() == "game"
        _select(panel, "create object")
        _assert_in_grid(panel)
        cells = {name: _cell_of(panel, _row_widget(panel, name)) for name in (
            "object_list_unit_id", "source_player", "location_x", "location_y", "item_id", "facet", "disable_sound",
        )}
        assert cells == {
            "object_list_unit_id": (1, 3), "source_player": (2, 1), "location_x": (3, 1), "location_y": (3, 1),
            # The game's column 5 is empty here, so disable sound closes up to 5.
            "item_id": (2, 3), "facet": (2, 2), "disable_sound": (5, 1),
        }
        # Set Location's (3, 3), grown up through the free cells above it.
        assert _cell_of(panel, panel._group_rows["location"].set_button) == (3, 1)
    finally:
        conftest.close_window(window)


def test_a_locked_quantity_slot_yields_the_quantity_cell_to_the_live_one() -> None:
    window = _bottom_window()
    try:
        panel = window.trigger_panel
        _select(panel, "modify attribute", 1)
        _assert_in_grid(panel)
        live = [spec.name for spec, *_ in panel._rows if spec.name in ("quantity", "armour_attack_quantity") and not spec.read_only]
        assert len(live) == 1, "fixture: one live quantity slot"
        # The game's (4, 2) and (3, 2), each moved up into the free cell above.
        assert _cell_of(panel, _row_widget(panel, live[0])) == (4, 1)
        assert _cell_of(panel, _row_widget(panel, "message"))[0] == 3
        message = next(h for h in panel._grid_cells if h.isAncestorOf(_row_widget(panel, "message")))
        assert panel.grid_layout.getItemPosition(panel.grid_layout.indexOf(message))[2:] == (1, 1)
    finally:
        conftest.close_window(window)


def test_the_trigger_row_sits_in_the_games_cells() -> None:
    window = _bottom_window()
    try:
        panel = window.trigger_panel
        _select(panel, "Trigger")
        _assert_in_grid(panel)
        stacks: dict[int, list[tuple[int, int, str]]] = {}
        for spec, _kind, _index, widget in panel._rows:
            column, place = _cell_of(panel, widget)
            assert column == tfl.TRIGGER_SLOTS[spec.name][0], spec.name
            stacks.setdefault(column, []).append((tfl.TRIGGER_SLOTS[spec.name][1], place, spec.name))
        for stack in stacks.values():
            # Each column stacks its cells in the table's row order.
            assert [name for _r, _p, name in sorted(stack)] == [name for _r, _p, name in sorted(stack, key=lambda e: e[1])]
    finally:
        conftest.close_window(window)


def test_a_tall_cell_does_not_push_other_columns_down() -> None:
    """Task Object: the Objects / Area / Location column is tall, and column 1 still stacks tight."""
    window = _bottom_window("game", 400)
    try:
        panel = window.trigger_panel
        _select(panel, "task object")
        upper, lower = _row_widget(panel, "object_group"), _row_widget(panel, "object_type")
        assert _cell_of(panel, upper)[0] == _cell_of(panel, lower)[0], "fixture: one column"
        gap = lower.mapTo(panel.grid_host, lower.rect().topLeft()).y() - upper.mapTo(
            panel.grid_host, upper.rect().bottomLeft()
        ).y()
        assert gap < 3 * upper.height(), f"{gap} px between object group and object type"
    finally:
        conftest.close_window(window)


def test_coordinates_sit_beside_their_captions() -> None:
    from PyQt5.QtWidgets import QLabel

    window = _bottom_window("game", 400)
    try:
        panel = window.trigger_panel
        _select(panel, "create object")
        for name, caption in (("location_x", "X"), ("location_y", "Y")):
            box = _row_widget(panel, name)
            label = next(w for w in box.parentWidget().findChildren(QLabel) if w.buddy() is box)
            assert label.text() == caption and label.toolTip()
            assert abs(label.geometry().center().y() - box.geometry().center().y()) <= 2, "same line"
            assert label.geometry().right() < box.geometry().left()
        x, y = _row_widget(panel, "location_x"), _row_widget(panel, "location_y")
        assert x.geometry().top() == y.geometry().top(), "x and y on one line"
        _select(panel, "objects in area")
        grid = _row_widget(panel, "area_x1").parentWidget()
        assert grid.layout().getItemPosition(grid.layout().indexOf(_row_widget(panel, "area_x1")))[:2] == (0, 1)
        assert grid.layout().getItemPosition(grid.layout().indexOf(_row_widget(panel, "area_y2")))[:2] == (1, 3)
    finally:
        conftest.close_window(window)


def test_change_variable_is_one_row() -> None:
    window = _bottom_window("game", 300)
    try:
        panel = window.trigger_panel
        _select(panel, "change variable")
        cells = [_cell_of(panel, _row_widget(panel, name)) for name in ("variable", "message", "operation", "quantity")]
        assert cells == [(1, 1), (2, 1), (3, 1), (4, 1)]
    finally:
        conftest.close_window(window)


def test_docking_at_a_side_or_floating_gives_the_column_back() -> None:
    from PyQt5.QtCore import Qt

    window = _bottom_window()
    try:
        panel, dock = window.trigger_panel, window.trigger_detail_dock
        _select(panel, "create object")
        names = [spec.name for spec, *_ in panel._rows]
        window.addDockWidget(Qt.LeftDockWidgetArea, dock)
        _settle()
        _assert_in_column(panel)
        assert [spec.name for spec, *_ in panel._rows] == names
        window.addDockWidget(Qt.TopDockWidgetArea, dock)
        _settle()
        _assert_in_grid(panel)
        dock.setFloating(True)
        _settle()
        _assert_in_column(panel)
    finally:
        conftest.close_window(window)


def test_a_narrow_form_after_a_wide_one_keeps_no_old_column() -> None:
    """The trigger row's four columns after Create Object's five, Compact's fewer columns after more:
    a QGridLayout keeps its old rows and columns, which must hold no stretch."""
    from PyQt5.QtCore import Qt

    window = _bottom_window("game", 400)
    try:
        panel, dock = window.trigger_panel, window.trigger_detail_dock
        for _ in range(3):
            _select(panel, "create object")
            _select(panel, "Trigger", 0)
        rows = panel.grid_layout.rowCount()
        _select(panel, "create object")
        _select(panel, "Trigger", 0)
        assert panel.grid_layout.rowCount() == rows, "a rebuild adds no row"
        stretches = [panel.grid_layout.columnStretch(c) for c in range(panel.grid_layout.columnCount())]
        assert stretches[:5] == [0, 0, 0, 0, 1] and not any(stretches[5:]), "the spare width goes right of the four"
        settings.set_trigger_details_bottom_layout("compact")
        window._sync_trigger_detail_layout()
        window.resizeDocks([dock], [200], Qt.Vertical)
        _settle()
        _select(panel, "task object")
        window.resizeDocks([dock], [500], Qt.Vertical)
        _settle()
        columns = len(set(panel._compact_flow))
        assert columns == len(panel._grid_cells)
        stretches = [panel.grid_layout.columnStretch(c) for c in range(panel.grid_layout.columnCount())]
        assert stretches[columns] == 1 and sum(stretches) == 1
    finally:
        conftest.close_window(window)


def test_a_text_box_dragged_taller_keeps_its_grip_in_its_cell() -> None:
    """The cell must follow each grip step, or the box outgrows it and its grip is clipped away."""
    window = _bottom_window("game", 300)
    try:
        panel = window.trigger_panel
        _select(panel, "Trigger", 0)
        box = _row_widget(panel, "description")
        cell = next(h for h in panel._grid_cells if h.isAncestorOf(box))
        for lines in (10, 20, 40):
            box._drag_to(lines)
            _settle()
            grip = box.grip().geometry().translated(box.mapTo(cell, box.rect().topLeft()))
            assert cell.rect().contains(grip), (lines, grip, cell.rect())
            assert panel.grid_host.height() >= cell.geometry().bottom()
    finally:
        conftest.close_window(window)


def test_a_compact_reflow_keeps_the_focused_field() -> None:
    """Moving a widget into a new column takes its focus: the value commits first, and focus comes back."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = _bottom_window("compact", 260)
    try:
        panel, dock = window.trigger_panel, window.trigger_detail_dock
        _select(panel, "task object")
        window.activateWindow()
        QApplication.setActiveWindow(window)
        _settle()
        spin = _row_widget(panel, "max_units_affected")
        spin.setFocus(Qt.OtherFocusReason)
        _settle()
        assert QApplication.focusWidget() in (spin, spin.lineEdit()), "fixture: the field has focus"
        columns = len(panel._grid_cells)
        window.resizeDocks([dock], [500], Qt.Vertical)
        _settle()
        assert len(panel._grid_cells) < columns, "fixture: the split changed"
        assert QApplication.focusWidget() in (spin, spin.lineEdit())
        items = {panel.grid_layout.itemAt(i).widget() for i in range(panel.grid_layout.count())}
        assert items == set(panel._grid_cells), "no host left behind"
    finally:
        conftest.close_window(window)


def test_compact_fills_the_docks_height_and_reflows() -> None:
    from PyQt5.QtCore import Qt

    window = _bottom_window("compact", 260)
    try:
        panel, dock = window.trigger_panel, window.trigger_detail_dock
        _select(panel, "task object")
        _assert_in_grid(panel)
        spacing = max(panel.property_form.verticalSpacing(), 0)
        tallest = max(tfl.block_height(block, spacing) for block in panel._form_blocks)
        if tallest > panel._compact_height():
            pytest.skip(f"one block is {tallest} px at this font, taller than the dock")
        assert not panel.property_area.verticalScrollBar().isVisible(), "Compact fits the dock's height"
        columns = len(set(panel._compact_flow))
        assert columns > 1
        for host in panel._grid_cells:
            assert host.height() <= panel.property_area.viewport().height()
        window.resizeDocks([dock], [500], Qt.Vertical)
        _settle()
        assert len(set(panel._compact_flow)) < columns, "a taller dock needs fewer columns"
        assert len(panel._grid_cells) == len(set(panel._compact_flow))
        assert not panel.property_area.verticalScrollBar().isVisible()
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("layout", ["game", "compact"])
def test_a_value_typed_in_the_grid_records_one_edit(layout: str) -> None:
    window = _bottom_window(layout)
    try:
        panel = window.trigger_panel
        _select(panel, "create object")
        before = _record_count(window)
        (_kind, index), = panel._form_refs
        spin = _row_widget(panel, "facet")
        value = spin.value() + 3
        spin.setValue(value)
        _settle()
        assert _record_count(window) == before + 1
        assert panel._manager().triggers[_TRIGGER].effects[index].facet == value
        _assert_in_grid(panel)
        assert _row_widget(panel, "facet").value() == value
    finally:
        conftest.close_window(window)


def test_a_groups_set_button_still_arms_the_pick() -> None:
    window = _bottom_window()
    try:
        panel = window.trigger_panel
        _select(panel, "task object")
        _assert_in_grid(panel)
        row = panel._group_rows["objects"]
        row.set_button.click()
        _settle()
        target = next(t for t, b in panel._pick_buttons.items() if b is panel._group_rows["objects"].set_button)
        assert window._unit_picker == target
        assert panel._group_rows["objects"].set_button.isChecked()
    finally:
        conftest.close_window(window)


def test_the_setting_switches_a_docked_form_live() -> None:
    window = _bottom_window("game")
    try:
        panel = window.trigger_panel
        _select(panel, "create object")
        settings.set_trigger_details_bottom_layout("compact")
        window._sync_trigger_detail_layout()
        _settle()
        assert panel.detail_layout() == "compact"
        _assert_in_grid(panel)
        assert panel.grid_layout.indexOf(panel._grid_cells[0]) >= 0
        assert all(panel.grid_layout.getItemPosition(panel.grid_layout.indexOf(h))[0] == 0 for h in panel._grid_cells)
    finally:
        conftest.close_window(window)


def test_every_type_rehomes_every_row() -> None:
    """A row added past _add_form_row() would stay in the hidden column form."""
    window = _bottom_window()
    try:
        panel = window.trigger_panel
        vocabulary = panel._vocabulary
        for kind, entries in (("condition", vocabulary.conditions), ("effect", vocabulary.effects)):
            panel.select_trigger(0)
            start = len(getattr(panel._manager().triggers[0], f"{kind}s"))
            for type_id in entries:
                window.entry_structural_edit("new", 0, kind, -1, type_id)
            for offset in range(len(entries)):
                panel.select_entry(kind, start + offset)
                _settle()
                assert panel.property_form.rowCount() == 0, (kind, offset)
                for spec, _kind, _index, widget in panel._rows:
                    assert panel.grid_host.isAncestorOf(widget), (kind, offset, spec.name)
        window.edit_history.mark_saved()
    finally:
        conftest.close_window(window)


@pytest.mark.font_sensitive
@pytest.mark.parametrize("layout", ["game", "compact"])
@pytest.mark.parametrize("name", ["create object", "task object", "display instructions", "Trigger"])
def test_no_two_grid_cells_overlap(layout: str, name: str) -> None:
    window = _bottom_window(layout, 400)
    try:
        panel = window.trigger_panel
        _select(panel, name, 0 if name in ("display instructions", "Trigger") else _TRIGGER)
        _assert_in_grid(panel)
        rects = [host.geometry() for host in panel._grid_cells]
        for i, a in enumerate(rects):
            for b in rects[i + 1 :]:
                assert not a.intersects(b), (a, b)
            host = panel._grid_cells[i]
            assert a.height() >= host.minimumSizeHint().height()
            if host.layout().hasHeightForWidth():
                # The Create Objects tip wraps; its cell must take the wrapped height.
                assert a.height() >= host.layout().heightForWidth(a.width()), host
    finally:
        conftest.close_window(window)


def test_the_setting_defaults_round_trips_and_refuses(tmp_path, monkeypatch) -> None:
    import yaml

    ids = [lid for lid, _label in settings.TRIGGER_DETAILS_BOTTOM_LAYOUTS]
    assert settings.get_trigger_details_bottom_layout() == settings.TRIGGER_DETAILS_BOTTOM_LAYOUT_DEFAULT == ids[0]
    for value in ids:
        settings.set_trigger_details_bottom_layout(value)
        monkeypatch.setattr(settings, "_trigger_details_bottom_layout", None)
        assert settings.get_trigger_details_bottom_layout() == value
    with pytest.raises(ValueError):
        settings.set_trigger_details_bottom_layout("column")
    stored = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert stored["trigger_details_bottom_layout"] == ids[-1], "a refused value writes nothing"
    (tmp_path / "config.yaml").write_text("trigger_details_bottom_layout: bogus\n")
    monkeypatch.setattr(settings, "_trigger_details_bottom_layout", None)
    assert settings.get_trigger_details_bottom_layout() == settings.TRIGGER_DETAILS_BOTTOM_LAYOUT_DEFAULT


@pytest.mark.parametrize("close", ["Apply", "Cancel"])
def test_the_dialog_combo_switches_the_form_and_cancel_puts_it_back(close: str) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QDialogButtonBox

    dialog, window = conftest.dialog_and_window()
    try:
        window.addDockWidget(Qt.BottomDockWidgetArea, window.trigger_detail_dock)
        _settle()
        panel = window.trigger_panel
        assert panel.detail_layout() == "game"
        combo = dialog.trigger_details_layout_combo
        assert combo.currentData() == "game"
        combo.setCurrentIndex(combo.findData("compact"))
        assert settings.get_trigger_details_bottom_layout() == "compact"
        assert panel.detail_layout() == "compact", "shown at once"
        dialog._buttons.button(getattr(QDialogButtonBox, close)).click()
        _settle()
        expected = "compact" if close == "Apply" else "game"
        assert settings.get_trigger_details_bottom_layout() == expected
        assert panel.detail_layout() == expected
    finally:
        dialog.close()
        conftest.close_window(window)
