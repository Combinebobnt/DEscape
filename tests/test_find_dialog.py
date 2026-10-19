"""Edit > Find and Replace (GH #144): FindDialog and its ViewerWindow wiring.

The dialog is modeless, so nothing here calls exec_(). ViewerWindow._warn and
_confirm are stubbed in every test (a real box hangs an xdist worker), and
every window closes through conftest.close_window(), which marks the history
saved first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

_REF_OAK = 100
_REF_PINE = 101
_REF_WALL = 102
_REF_HOUSE = 200
_REF_ARCHER_P1 = 201
_REF_VILLAGER_P1 = 203
_REF_ARCHER_P2 = 300
_REF_VILLAGER_P2 = 301
_BARRACKS = 12
_OAK = 349
_STONE_WALL = 117

_close = conftest.close_window


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


class _Boxes:
    """Records every _warn/_confirm call; `answer` is what _confirm returns."""

    def __init__(self):
        self.warned: list[tuple[str, str]] = []
        self.confirmed: list[tuple] = []
        self.answer: dict | None = {}

    def warn(self, title, text):
        self.warned.append((title, text))

    def confirm(self, title, text, choices=(), actions=()):
        self.confirmed.append((title, text, tuple(choices), tuple(actions)))
        return self.answer


def _window(style: str | None = None):
    from PyQt5.QtWidgets import QApplication

    window = conftest.shown_window(1200, 800)
    boxes = _Boxes()
    window._warn = boxes.warn
    window._confirm = boxes.confirm
    window.load_scenario(UNITS_FIXTURE)
    assert window.scenario is not None
    if style is not None:
        window.terrain_style_combo.setCurrentText(style)
    QApplication.processEvents()
    return window, boxes


def _dialog(window):
    window._show_find_dialog()
    return window._find_dialog


def _settle():
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    QApplication.processEvents()


def _rows(dialog):
    return {row.reference_id: row for row in dialog.objects_tab.rows}


def _check_only(dialog, refs) -> None:
    tab = dialog.objects_tab
    tab._set_all_checked(False)
    from PyQt5.QtCore import Qt

    for i in range(tab.tree.topLevelItemCount()):
        item = tab.tree.topLevelItem(i)
        if item.data(0, Qt.UserRole).reference_id in refs:
            item.setCheckState(0, Qt.Checked)
    assert {key[1] for key in tab.checked} == set(refs)


def _unit(window, ref):
    return next(u for u in window.scenario.unit_manager.get_all_units() if u.reference_id == ref)


def _owner(window, ref) -> int:
    for player, units in enumerate(window.scenario.unit_manager.units):
        if any(u.reference_id == ref for u in units):
            return player
    raise AssertionError(f"{ref} not placed")


# -- wiring --------------------------------------------------------------------


def test_action_is_rebindable_unbound_and_enabled_without_building_the_unit_model(monkeypatch) -> None:
    from descape import settings
    from descape.viewer import ViewerWindow

    # The load's idle prebuild (TASK-031.61 A1) would build it; Find is what is under test.
    monkeypatch.setattr(ViewerWindow, "_schedule_unit_model_prebuild", lambda self: None)
    window, _boxes = _window()
    try:
        row = next(r for r in settings.REBINDABLE_ACTIONS if r[0] == "edit_find_replace")
        assert row[2] == ""
        assert window._keybind_actions["edit_find_replace"] is window.find_action
        assert window.find_action.shortcut().isEmpty()
        assert window.find_action.isEnabled()
        dialog = _dialog(window)
        assert not dialog.isModal()
        dialog.objects_tab.run_find()
        assert len(dialog.objects_tab.rows) == 8
        assert window.unit_edits is None, "opening Find or searching must not build the unit model"
    finally:
        _close(window)


def test_action_is_disabled_with_no_document() -> None:
    window = conftest.shown_window(1200, 800)
    try:
        assert not window.find_action.isEnabled()
        window._show_find_dialog()
        assert window._find_dialog is None
    finally:
        _close(window)


def test_closing_or_reopening_the_file_closes_the_dialog() -> None:
    window, _boxes = _window()
    try:
        _dialog(window)
        window.edit_history.mark_saved()
        window.load_scenario(UNITS_FIXTURE)
        assert window._find_dialog is None
        _dialog(window)
        window.close_scenario()
        assert window._find_dialog is None
    finally:
        _close(window)


def test_region_area_follows_the_selection() -> None:
    from descape.find_dialog import AREA_REGION

    window, _boxes = _window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        item = tab.area_combo.model().item(AREA_REGION)
        assert not item.isEnabled()
        window.on_region_selected((10, 10, 13, 13))
        assert item.isEnabled()
        tab.area_combo.setCurrentIndex(AREA_REGION)
        tab.run_find()
        assert sorted(_rows(dialog)) == [_REF_HOUSE, _REF_ARCHER_P1, _REF_VILLAGER_P1]
        window.on_region_selected(None)
        assert not item.isEnabled()
    finally:
        _close(window)


def test_invalid_regex_disables_find_inline() -> None:
    window, _boxes = _window()
    try:
        tab = _dialog(window).objects_tab
        tab.regex_box.setChecked(True)
        tab.name_edit.setText("(open")
        assert not tab.find_button.isEnabled()
        assert tab.pattern_error.isVisibleTo(tab)
        tab.name_edit.setText("arch")
        assert tab.find_button.isEnabled()
        tab.run_find()
        assert {r.unit_const for r in tab.rows} == {4}
    finally:
        _close(window)


# -- operations ----------------------------------------------------------------


def test_delete_is_one_undo_step_and_results_refresh_after_undo_from_terrain_mode() -> None:
    window, boxes = _window()
    try:
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        _check_only(dialog, {_REF_HOUSE, _REF_ARCHER_P2})
        before = window.edit_history.cursor
        dialog.objects_tab.delete_button.click()
        assert window.edit_history.cursor == before + 1
        assert "Delete 2 objects" in boxes.confirmed[-1][1]
        _settle()
        # The house's villager went with it.
        assert sorted(_rows(dialog)) == [_REF_OAK, _REF_PINE, _REF_WALL, _REF_ARCHER_P1, _REF_VILLAGER_P2]
        window.mode_combo.setCurrentText("Terrain")
        window.undo()
        _settle()
        assert len(dialog.objects_tab.rows) == 8
        window.redo()
        _settle()
        assert len(dialog.objects_tab.rows) == 5
    finally:
        _close(window)


def test_a_cancelled_confirm_changes_nothing() -> None:
    window, boxes = _window()
    try:
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        boxes.answer = None
        dialog.objects_tab.delete_button.click()
        assert window.edit_history.cursor == 0
        assert len(window.scenario.unit_manager.get_all_units()) == 8
    finally:
        _close(window)


def test_replace_is_one_undo_step_and_refused_rows_are_skipped_and_marked() -> None:
    from descape.find_dialog import _COL_NOTE

    window, boxes = _window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _check_only(dialog, {_REF_HOUSE, _REF_OAK, _REF_ARCHER_P2})
        tab.replace_edit.set_value(_BARRACKS)
        before = window.edit_history.cursor
        tab.replace_button.click()
        assert window.edit_history.cursor == before + 1
        assert "Replace 2 objects" in boxes.confirmed[-1][1]
        assert _unit(window, _REF_HOUSE).unit_const == 70, "a Barracks cannot hold the house's villager"
        assert _unit(window, _REF_OAK).unit_const == _BARRACKS
        assert _unit(window, _REF_ARCHER_P2).unit_const == _BARRACKS
        _settle()
        house_item = next(
            tab.tree.topLevelItem(i)
            for i in range(tab.tree.topLevelItemCount())
            if tab.tree.topLevelItem(i).text(1) == str(_REF_HOUSE)
        )
        assert "Garrison" in house_item.text(_COL_NOTE)
        assert "Replaced 2; 1 refused" in window.status_log.toPlainText()
        window.undo()
        assert _unit(window, _REF_OAK).unit_const == _OAK
        assert (_unit(window, _REF_OAK).x, _unit(window, _REF_OAK).y) == (5.5, 5.5)
    finally:
        _close(window)


def test_a_replaced_host_carries_its_garrison_to_the_new_anchor() -> None:
    window, _boxes = _window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _check_only(dialog, {_REF_HOUSE})
        tab.replace_edit.set_value(82)  # Castle: 4x4, so the 2x2 house's anchor moves
        tab.replace_button.click()
        house, villager = _unit(window, _REF_HOUSE), _unit(window, _REF_VILLAGER_P1)
        assert house.unit_const == 82
        assert (villager.x, villager.y) == (house.x, house.y)
        assert villager.garrisoned_in_id == _REF_HOUSE
    finally:
        _close(window)


def test_a_host_refused_only_for_its_dropped_occupants_type_is_replaced() -> None:
    """House + villager -> Watch Tower: the villager cannot sit in a tower,
    and once it drops the house can become one holding it (TASK-208 b)."""
    window, _boxes = _window()
    try:
        keys = [(1, _REF_HOUSE), (1, _REF_VILLAGER_P1)]
        refused = window._find_replace_preflight(keys, 79)
        assert refused == {(1, _REF_VILLAGER_P1): "Garrison: Watch Tower cannot go inside Watch Tower"}
        replaced, refused = window._find_replace_keys(keys, 79)
        assert replaced == 1 and set(refused) == {(1, _REF_VILLAGER_P1)}
        assert (_unit(window, _REF_HOUSE).unit_const, _unit(window, _REF_VILLAGER_P1).unit_const) == (79, 83)
        assert _unit(window, _REF_VILLAGER_P1).garrisoned_in_id == _REF_HOUSE
    finally:
        _close(window)


def test_change_owner_carries_occupants_and_refuses_an_orphan_occupant_row() -> None:
    window, _boxes = _window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _check_only(dialog, {_REF_VILLAGER_P1})
        tab.owner_combo.setCurrentIndex(3)
        tab.owner_apply.click()
        assert window.edit_history.cursor == 0, "an occupant without its host is refused"
        assert _owner(window, _REF_VILLAGER_P1) == 1
        assert "1 refused" in window.status_log.toPlainText()

        _check_only(dialog, {_REF_HOUSE})
        tab.owner_apply.click()
        assert window.edit_history.cursor == 1
        assert _owner(window, _REF_HOUSE) == 3
        assert _owner(window, _REF_VILLAGER_P1) == 3
        assert tab.checked == {(3, _REF_HOUSE)}, "the checked key follows the owner change"
        _settle()
        assert _rows(dialog)[_REF_HOUSE].player == 3
    finally:
        _close(window)


def test_select_on_map_reports_the_hidden_count() -> None:
    window, _boxes = _window()
    try:
        assert not window.show_garrisoned_action.isChecked()
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        _check_only(dialog, {_REF_ARCHER_P1, _REF_VILLAGER_P1})
        dialog.objects_tab.select_button.click()
        assert window.mode == "units"
        assert window._selection == [(1, _REF_ARCHER_P1)]
        assert "Selected 1 of 2; 1 hidden by filters / garrisoned / off-map" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_double_click_centres_on_the_row(monkeypatch) -> None:
    window, _boxes = _window()
    try:
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        centred = []
        monkeypatch.setattr(window.map_view, "center_on_tile", lambda x, y: centred.append((x, y)))
        tab = dialog.objects_tab
        item = next(
            tab.tree.topLevelItem(i)
            for i in range(tab.tree.topLevelItemCount())
            if tab.tree.topLevelItem(i).text(1) == str(_REF_ARCHER_P2)
        )
        tab.tree.itemDoubleClicked.emit(item, 0)
        assert centred == [(20, 20)]
    finally:
        _close(window)


def _object_item(tab, ref):
    return next(
        tab.tree.topLevelItem(i)
        for i in range(tab.tree.topLevelItemCount())
        if tab.tree.topLevelItem(i).text(1) == str(ref)
    )


def test_go_to_centres_on_the_highlighted_row_not_the_ticked_ones(monkeypatch) -> None:
    """GH #201: the button is double-click's centre step for the current row."""
    from descape.find_dialog import _GOTO_DISABLED_TIP, _GOTO_TIP

    window, _boxes = _window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        assert not tab.goto_button.isEnabled()
        assert tab.goto_button.toolTip() == _GOTO_DISABLED_TIP
        tab.run_find()
        assert not tab.goto_button.isEnabled(), "a Find highlights no row"
        centred = []
        monkeypatch.setattr(window.map_view, "center_on_tile", lambda x, y: centred.append((x, y)))
        tab.goto_button.click()
        assert centred == [], "a disabled button does nothing"

        _check_only(dialog, {_REF_OAK})
        tab.tree.setCurrentItem(_object_item(tab, _REF_ARCHER_P2))
        assert tab.goto_button.isEnabled()
        assert tab.goto_button.toolTip() == _GOTO_TIP
        tab.goto_button.click()
        assert centred == [(20, 20)], "the highlighted archer, not the ticked oak at (5, 5)"
        assert {key[1] for key in tab.checked} == {_REF_OAK}, "Go to leaves the ticks alone"

        tab.run_find()
        assert tab.tree.currentItem() is None
        assert not tab.goto_button.isEnabled(), "a refill drops the highlight"
        assert tab.goto_button.toolTip() == _GOTO_DISABLED_TIP
    finally:
        _close(window)


def test_go_to_follows_the_highlight_not_a_current_row_ctrl_clicked_off(monkeypatch) -> None:
    """TASK-352: a Ctrl+click unhighlights the current row; Go to must not still centre on it."""
    from PyQt5.QtCore import Qt

    from descape.find_dialog import _GOTO_DISABLED_TIP, _GOTO_TIP

    window, _boxes = _window()
    try:
        tab = _dialog(window).objects_tab
        tab.run_find()
        tree = tab.tree
        centred = []
        monkeypatch.setattr(window.map_view, "center_on_tile", lambda x, y: centred.append((x, y)))
        archer = _object_item(tab, _REF_ARCHER_P2)
        _click_row(tree, archer)
        assert tab.goto_button.isEnabled(), "a plain click highlights the row"
        _click_row(tree, archer, Qt.ControlModifier)
        assert tree.currentItem() is archer and tree.selectedItems() == []
        assert not tab.goto_button.isEnabled(), "nothing is highlighted"
        assert tab.goto_button.toolTip() == _GOTO_DISABLED_TIP
        tab.go_to_current()
        assert centred == []

        _click_row(tree, _object_item(tab, _REF_OAK))
        _click_row(tree, archer, Qt.ControlModifier)
        _click_row(tree, archer, Qt.ControlModifier)
        assert tree.currentItem() is archer and tree.selectedItems() == [_object_item(tab, _REF_OAK)]
        assert tab.goto_button.isEnabled()
        assert tab.goto_button.toolTip() == _GOTO_TIP
        tab.goto_button.click()
        assert centred == [(5, 5)], "the highlighted oak, not the unhighlighted current archer"
    finally:
        _close(window)


# -- ticking several rows at once (GH #144 follow-up) ----------------------------


def _click_row(tree, item, modifier=None) -> None:
    """A real click on the row's second column, clear of the checkbox."""
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest

    tree.scrollToItem(item)
    pos = QPoint(tree.columnViewportPosition(1) + 10, tree.visualItemRect(item).center().y())
    QTest.mouseClick(tree.viewport(), Qt.LeftButton, Qt.NoModifier if modifier is None else modifier, pos)


def _press_space(tree) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    QTest.keyClick(tree, Qt.Key_Space)


def _count_update_ops(tab, monkeypatch) -> list:
    calls = []
    real = tab._update_ops
    monkeypatch.setattr(tab, "_update_ops", lambda: (calls.append(1), real())[1])
    return calls


def _menu_action(menu, text: str):
    return next(a for a in menu.actions() if a.text() == text)


def _ticked_refs(tab) -> set:
    from PyQt5.QtCore import Qt

    items = [tab.tree.topLevelItem(i) for i in range(tab.tree.topLevelItemCount())]
    shown = {int(item.text(1)) for item in items if item.checkState(0) == Qt.Checked}
    assert shown == {key[1] for key in tab.checked} == {row.reference_id for row in tab.checked_rows()}
    return shown


_ALL_REFS = {_REF_OAK, _REF_PINE, _REF_WALL, _REF_HOUSE, _REF_ARCHER_P1, _REF_VILLAGER_P1, _REF_ARCHER_P2, _REF_VILLAGER_P2}


def test_space_flips_every_highlighted_object_row_to_the_opposite_of_the_current_one(monkeypatch) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QAbstractItemView

    window, _boxes = _window()
    try:
        tab = _dialog(window).objects_tab
        tab.run_find()
        tree = tab.tree
        assert tree.selectionMode() == QAbstractItemView.ExtendedSelection
        _click_row(tree, _object_item(tab, _REF_PINE))
        _click_row(tree, _object_item(tab, _REF_HOUSE), Qt.ShiftModifier)
        _click_row(tree, _object_item(tab, _REF_ARCHER_P2), Qt.ControlModifier)
        span = {_REF_PINE, _REF_WALL, _REF_HOUSE, _REF_ARCHER_P2}
        assert {int(item.text(1)) for item in tree.selectedItems()} == span
        assert _ticked_refs(tab) == _ALL_REFS, "highlighting rows ticks nothing"

        calls = _count_update_ops(tab, monkeypatch)
        _press_space(tree)
        assert _ticked_refs(tab) == _ALL_REFS - span
        assert len(calls) == 1, "one button refresh for the whole toggle, not one per row"

        # Mixed states follow the current row (the Ctrl+clicked archer), never a per-row flip.
        _object_item(tab, _REF_WALL).setCheckState(0, Qt.Checked)
        assert _ticked_refs(tab) == _ALL_REFS - span | {_REF_WALL}
        _press_space(tree)
        assert _ticked_refs(tab) == _ALL_REFS, "the current row was unticked, so all four tick"
        _object_item(tab, _REF_PINE).setCheckState(0, Qt.Unchecked)
        _press_space(tree)
        assert _ticked_refs(tab) == _ALL_REFS - span, "the current row was ticked, so all four untick"

        # Go to still follows the current row of a multi-row highlight.
        centred = []
        monkeypatch.setattr(window.map_view, "center_on_tile", lambda x, y: centred.append((x, y)))
        tab.goto_button.click()
        assert centred == [(20, 20)]
    finally:
        _close(window)


def test_space_on_one_object_row_and_qts_own_space_keys_still_work() -> None:
    from PyQt5.QtCore import QItemSelectionModel, Qt
    from PyQt5.QtTest import QTest

    window, _boxes = _window()
    try:
        tab = _dialog(window).objects_tab
        tab.run_find()
        tree = tab.tree
        oak, pine = _object_item(tab, _REF_OAK), _object_item(tab, _REF_PINE)
        # Clicked on the ID column: Qt's own Space only ever toggled a current Name cell.
        _click_row(tree, oak)
        _press_space(tree)
        assert _ticked_refs(tab) == _ALL_REFS - {_REF_OAK}

        # Nothing highlighted: Space falls through to Qt's toggle of the current Name cell.
        tree.clearSelection()
        tree.setCurrentItem(oak, 0, QItemSelectionModel.NoUpdate)
        _press_space(tree)
        assert _ticked_refs(tab) == _ALL_REFS

        # Ctrl+Space is still the keyboard's "add the focused row to the highlight".
        _click_row(tree, oak)
        tree.setCurrentItem(pine, 1, QItemSelectionModel.NoUpdate)
        QTest.keyClick(tree, Qt.Key_Space, Qt.ControlModifier)
        assert {int(item.text(1)) for item in tree.selectedItems()} == {_REF_OAK, _REF_PINE}
        assert _ticked_refs(tab) == _ALL_REFS, "Ctrl+Space ticked nothing"
    finally:
        _close(window)


def test_the_object_context_menu_ticks_and_unticks_the_highlighted_rows(monkeypatch) -> None:
    from PyQt5.QtCore import Qt

    window, _boxes = _window()
    try:
        tab = _dialog(window).objects_tab
        tab.run_find()
        tree = tab.tree
        menu = tree.check_menu()
        assert [a.text() for a in menu.actions()] == ["Tick selected", "Untick selected"]
        assert not any(a.isEnabled() for a in menu.actions()), "nothing is highlighted yet"

        _click_row(tree, _object_item(tab, _REF_OAK))
        _click_row(tree, _object_item(tab, _REF_WALL), Qt.ShiftModifier)
        span = {_REF_OAK, _REF_PINE, _REF_WALL}
        _object_item(tab, _REF_PINE).setCheckState(0, Qt.Unchecked)
        calls = _count_update_ops(tab, monkeypatch)
        _menu_action(tree.check_menu(), "Untick selected").trigger()
        assert _ticked_refs(tab) == _ALL_REFS - span
        assert len(calls) == 1
        assert tab.delete_button.isEnabled()
        _menu_action(tree.check_menu(), "Tick selected").trigger()
        assert _ticked_refs(tab) == _ALL_REFS

        # Check all / Check none and the buttons they gate are untouched by the highlight.
        tab.check_none.click()
        assert _ticked_refs(tab) == set() and not tab.delete_button.isEnabled()
        _menu_action(tree.check_menu(), "Tick selected").trigger()
        assert _ticked_refs(tab) == span and tab.delete_button.isEnabled()
    finally:
        _close(window)


def test_a_bulk_tick_skips_a_row_no_operation_accepts() -> None:
    """A duplicated reference id has no checkbox; highlighting it must not tick it."""
    from dataclasses import replace

    from PyQt5.QtCore import Qt

    from descape.find_dialog import FindCallbacks, FindDialog
    from descape.find_objects import FindRow

    conftest.ensure_qapp()
    base = FindRow((1, 10), 10, 4, "Archer", 1, 1.5, 1.5, 0.0, "", -1, 0, False, False)
    rows = [base, replace(base, key=(1, 11), reference_id=11, duplicate_ref=True), replace(base, key=(1, 12), reference_id=12)]
    dialog = FindDialog(callbacks=FindCallbacks(find_objects=lambda criteria: rows))
    try:
        dialog.show()
        tab = dialog.objects_tab
        tab.run_find()
        tree = tab.tree
        tree.selectAll()
        assert len(tree.selectedItems()) == 3
        assert [int(item.text(1)) for item in tree.highlighted_checkable()] == [10, 12]
        tree.setCurrentItem(_object_item(tab, 11), 0, tree.selectionModel().NoUpdate)
        _press_space(tree)
        assert tab.checked == set(), "no lead row: every highlighted row was ticked, so they untick"
        _press_space(tree)
        assert tab.checked == {(1, 10), (1, 12)}
        assert _object_item(tab, 11).data(0, Qt.CheckStateRole) is None
    finally:
        dialog.close()
        dialog.deleteLater()


def test_export_csv_writes_checked_rows_and_refuses_mods_subscribed(tmp_path) -> None:
    window, boxes = _window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _check_only(dialog, {_REF_OAK, _REF_ARCHER_P2})
        out = tmp_path / "out.csv"
        tab.ask_csv_path = lambda: str(out)
        tab.export_button.click()
        assert len(out.read_text(encoding="utf-8").splitlines()) == 3
        blocked = tmp_path / "mods" / "subscribed" / "out.csv"
        blocked.parent.mkdir(parents=True)
        tab.ask_csv_path = lambda: str(blocked)
        tab.export_button.click()
        assert not blocked.exists()
        assert boxes.warned and "mods/subscribed/" in boxes.warned[-1][1]
    finally:
        _close(window)


def test_a_mutating_op_is_refused_mid_stroke(monkeypatch) -> None:
    window, _boxes = _window()
    try:
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        monkeypatch.setattr(window, "_stroke_in_progress", lambda: True)
        dialog.objects_tab.delete_button.click()
        assert window.edit_history.cursor == 0
        assert len(window.scenario.unit_manager.get_all_units()) == 8
    finally:
        _close(window)


# -- render: a wall's neighbour after Replace, undo and redo ----------------------


@pytest.fixture
def wall_art(tmp_path, monkeypatch):
    """A synthetic install drawing the stone wall's five shapes in five colours,
    so a neighbour drawn with a stale shape is a pixel mismatch."""
    from descape import asset_source, unit_sprites

    from test_unit_sprites import build_sld

    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    name = "t_wall_stone_x1"
    (graphics / f"{name}.sld").write_bytes(build_sld(5, canvas=unit_sprites.NATIVE_TILE_W, playercolor=False))
    table = {
        _STONE_WALL: {
            "graphic_id": 1,
            "file_name": name,
            "angle_count": 5,
            "mirroring_mode": 0,
            "frame_count": 1,
            # Else a GAIA wall's stored rotation reads as 0.0 (render.stored_rotation).
            "rotation_is_variant": True,
        }
    }
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: table)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _fresh_canvas(window, style: str, sprites: bool = True) -> np.ndarray:
    from descape import render

    canvas_w, canvas_h = window._cache.canvas_dims(0)
    if style == "Stepped":
        full = render.render_terrain_iso_with_proj(window.scenario, with_units=True, with_sprites=sprites)[0]
    else:
        full = render.render_terrain_sloped_with_proj(window.scenario, with_units=True, with_sprites=sprites)[0]
    return full[:canvas_h, :canvas_w]


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_replacing_a_wall_beside_a_wall_repaints_like_a_fresh_render(style, wall_art) -> None:
    from descape import render

    window, _boxes = _window(style)
    try:
        # The fresh render's default filter shows the garrisoned villager.
        window.show_garrisoned_action.setChecked(True)
        assert window.show_sprites_action.isChecked()
        model = window._ensure_unit_edits()
        # A run at tiles 7..9: the middle wall is a straight run (index 0) only
        # while the fixture's wall at 7 stays a wall, then a tower (2).
        with window._unit_edit(model, "Add walls", [0]):
            model.add(0, _STONE_WALL, 8.5, 5.5, rotation=2.5132741928100586)
            model.add(0, _STONE_WALL, 9.5, 5.5, rotation=2.5132741928100586)
        assert render.wall_variant_rotation_overrides(window.scenario)[(0, 3)] == 0.0
        canvas_w, canvas_h = window._cache.canvas_dims(0)
        start = window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()
        assert np.array_equal(start, _fresh_canvas(window, style))

        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _check_only(dialog, {_REF_WALL})
        tab.replace_edit.set_value(_OAK)
        tab.replace_button.click()
        assert _unit(window, _REF_WALL).unit_const == _OAK
        assert render.wall_variant_rotation_overrides(window.scenario)[(0, 3)] == 2.0
        replaced =window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()
        assert np.array_equal(replaced, _fresh_canvas(window, style)), "stale pixels after Replace"
        assert not np.array_equal(replaced, start)

        window._cache.render_rect(0, 0, canvas_w, canvas_h)
        window.undo()
        undone = window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()
        assert np.array_equal(undone, _fresh_canvas(window, style)), "stale pixels after undo"
        assert np.array_equal(undone, start)

        window._cache.render_rect(0, 0, canvas_w, canvas_h)
        window.redo()
        redone = window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()
        assert np.array_equal(redone, _fresh_canvas(window, style)), "stale pixels after redo"
        assert np.array_equal(redone, replaced)
    finally:
        _close(window)


# == Part B: the Triggers tab =========================================================

FIND_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "find_triggers_120x120.aoe2scenario"
_KNIGHT = 38
_ARCHER = 4
_ARCHER_A = 500


def _trigger_window():
    from PyQt5.QtWidgets import QApplication

    window = conftest.shown_window(1200, 800)
    boxes = _Boxes()
    window._warn = boxes.warn
    window._confirm = boxes.confirm
    window.load_scenario(FIND_FIXTURE)
    assert window.scenario is not None
    QApplication.processEvents()
    return window, boxes


def _ttab(window):
    dialog = _dialog(window)
    dialog.tabs.setCurrentWidget(dialog.triggers_tab)
    assert dialog.triggers_tab.initialized
    return dialog.triggers_tab


def _tfind(tab, text="", **kwargs):
    tab.text_edit.setText(text)
    for key, box in tab.scope_boxes.items():
        box.setChecked(key in kwargs.get("scopes", ("names", "descriptions", "messages")))
    tab.run_find()


def _check_children(tab, predicate) -> None:
    from PyQt5.QtCore import Qt

    tab._set_all_checked(False)
    for i in range(tab.tree.topLevelItemCount()):
        parent = tab.tree.topLevelItem(i)
        for j in range(parent.childCount()):
            hit = parent.child(j).data(0, Qt.UserRole)
            if predicate(hit):
                parent.child(j).setCheckState(0, Qt.Checked)


def _check_parents(tab, indices) -> None:
    from PyQt5.QtCore import Qt

    tab._set_all_checked(False)
    for i in range(tab.tree.topLevelItemCount()):
        parent = tab.tree.topLevelItem(i)
        if parent.data(0, Qt.UserRole)[1] in indices:
            parent.setCheckState(0, Qt.Checked)
    assert tab.checked_parents == set(indices)


def _triggers(window):
    from descape.scenario_io import parse_triggers

    return parse_triggers(window.scenario).triggers


def test_text_replace_is_one_step_declaring_only_the_changed_trigger() -> None:
    window, boxes = _trigger_window()
    try:
        tab = _ttab(window)
        _tfind(tab, "bridge")
        assert len(tab.hits) == 1
        tab.replace_text_edit.setText("gate")
        before = window.edit_history.cursor
        tab.replace_text_button.click()
        assert "Replace in 1 fields across 1 triggers" in boxes.confirmed[-1][1]
        assert window.edit_history.cursor == before + 1
        assert window.edit_history.records[-1].touched == [0]
        assert _triggers(window)[0].effects[0].message == "Hold the gate.\r\nArchers to the walls."
        _settle()
        assert tab.hits == [], "the re-run no longer finds the replaced text"
    finally:
        _close(window)


def test_type_replace_enable_disable_and_delete_are_one_step_each() -> None:
    window, _boxes = _trigger_window()
    try:
        tab = _ttab(window)
        tab.add_type_const(_ARCHER)
        _tfind(tab)
        _check_children(tab, lambda h: h.attribute == "object_list")
        tab.replace_type_edit.set_value(_KNIGHT)
        tab.replace_type_button.click()
        assert window.edit_history.cursor == 1
        assert window.edit_history.records[-1].touched == [2]
        assert _triggers(window)[2].conditions[0].object_list == _KNIGHT
        assert _triggers(window)[2].effects[0].object_list_unit_id == _ARCHER, "an unchecked hit is not written"

        tab.clear_types()
        _tfind(tab)
        _check_parents(tab, {0, 3})
        tab.disable_button.click()
        assert window.edit_history.cursor == 2
        assert window.edit_history.records[-1].touched == [0], "trigger 3 was already disabled"
        _settle()
        assert tab.checked_parents == {0, 3}, "the dialog's own enable/disable keeps its checks"
        tab.enable_button.click()
        assert window.edit_history.cursor == 3
        assert window.edit_history.records[-1].touched == [0, 3]

        _settle()
        _check_parents(tab, {1})
        tab.delete_button.click()
        assert window.edit_history.cursor == 4
        assert [t.name for t in _triggers(window)] == ["[Find] Messages", "Retarget", "Find: disabled"]
        _settle()
        assert tab.checked_parents == set(), "a delete clears the checks"
    finally:
        _close(window)


def test_an_object_replace_following_a_paired_effect_is_one_composite_step() -> None:
    """Passes when the confirm offers the paired and filter-only checkboxes
    with their exact labels (counts, and the filter-only summary) and
    defaults (paired ticked, filter-only not), and the answered replace is one
    composite unit + trigger step that undoes and redoes."""
    from descape.edit_history import CompositeDiffRecord

    window, boxes = _trigger_window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _check_only(dialog, {_ARCHER_A})
        tab.replace_edit.set_value(_KNIGHT)
        boxes.answer = {"paired": True, "filter_only": False}
        tab.replace_button.click()
        _title, _text, choices, actions = boxes.confirmed[-1]
        assert [c[0] for c in choices] == ["paired", "filter_only"]
        assert [(c[1], c[2]) for c in choices] == [
            ("Retarget 1 paired trigger effects", True),
            ("Also retarget 1 type-filter-only effects (Create Object: 1 effect name Archer)", False),
        ]
        assert {a[0] for a in actions} == {"Show skipped", "Show others"}
        assert window.edit_history.cursor == 1
        record = window.edit_history.records[-1]
        assert isinstance(record, CompositeDiffRecord)
        assert record.kinds() == {"unit", "trigger"}
        assert _unit(window, _ARCHER_A).unit_const == _KNIGHT
        effects = _triggers(window)[2].effects
        assert effects[0].object_list_unit_id == _KNIGHT
        assert effects[1].object_list_unit_id == _ARCHER, "tier (b) stays off by default"
        window.undo()
        assert _unit(window, _ARCHER_A).unit_const == _ARCHER
        assert _triggers(window)[2].effects[0].object_list_unit_id == _ARCHER
        window.redo()
        assert _unit(window, _ARCHER_A).unit_const == _KNIGHT
        assert _triggers(window)[2].effects[0].object_list_unit_id == _KNIGHT
    finally:
        _close(window)


def test_a_replace_with_no_trigger_model_omits_the_follow_up(monkeypatch) -> None:
    from descape.edit_history import UnitDiffRecord

    window, boxes = _trigger_window()
    try:
        monkeypatch.setattr(window, "_ensure_trigger_edits", lambda: None)
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        _check_only(dialog, {_ARCHER_A})
        dialog.objects_tab.replace_edit.set_value(_KNIGHT)
        dialog.objects_tab.replace_button.click()
        _title, text, choices, _actions = boxes.confirmed[-1]
        assert choices == () and "not followed" in text
        assert isinstance(window.edit_history.records[-1], UnitDiffRecord)
        assert _triggers(window)[2].effects[0].object_list_unit_id == _ARCHER
    finally:
        _close(window)


def test_jump_lands_on_the_entry_even_with_a_panel_filter() -> None:
    window, _boxes = _trigger_window()
    try:
        window.mode_combo.setCurrentText("Triggers")
        window.trigger_panel.filter_edit.setText("Messages")
        tab = _ttab(window)
        tab.set_object_refs([_ARCHER_A])
        _tfind(tab)
        window.mode_combo.setCurrentText("Units")
        parent = next(
            tab.tree.topLevelItem(i) for i in range(tab.tree.topLevelItemCount()) if tab.tree.topLevelItem(i).text(0) == "2"
        )
        target = next(parent.child(j) for j in range(parent.childCount()) if parent.child(j).text(1).startswith("Effect #3"))
        tab.tree.itemDoubleClicked.emit(target, 0)
        panel = window.trigger_panel
        assert window.mode == "triggers"
        assert panel.filter_edit.text() == ""
        assert panel.current_trigger_index() == 2
        assert ("effect", 3) in panel.selected_entry_refs()
    finally:
        _close(window)


def _ticked_hits(tab) -> tuple[set, set]:
    """(ticked trigger indices, ticked child entry labels), checked against the tab's own sets."""
    from PyQt5.QtCore import Qt

    parents, children, locations = set(), set(), set()
    for i in range(tab.tree.topLevelItemCount()):
        parent = tab.tree.topLevelItem(i)
        if parent.checkState(0) == Qt.Checked:
            parents.add(parent.data(0, Qt.UserRole)[1])
        for j in range(parent.childCount()):
            child = parent.child(j)
            if child.checkState(0) == Qt.Checked:
                children.add(child.text(1).split()[0] + child.text(1).split()[1])
                locations.add(child.data(0, Qt.UserRole).location)
    assert parents == tab.checked_parents and locations == tab.checked_children
    return parents, children


def test_space_and_the_context_menu_tick_several_trigger_rows_at_once(monkeypatch) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QAbstractItemView

    window, _boxes = _trigger_window()
    try:
        tab = _ttab(window)
        tree = tab.tree
        assert tree.selectionMode() == QAbstractItemView.ExtendedSelection
        tab.add_type_const(_ARCHER)
        _tfind(tab)
        parent = tree.topLevelItem(0)
        every = {"Condition#0", "Effect#0", "Effect#1", "Effect#2", "Effect#3", "Effect#4", "Effect#5"}
        assert _ticked_hits(tab) == ({2}, every)

        _click_row(tree, parent.child(1))
        _click_row(tree, parent.child(4), Qt.ShiftModifier)
        span = {"Effect#0", "Effect#1", "Effect#2", "Effect#3"}
        calls = _count_update_ops(tab, monkeypatch)
        _press_space(tree)
        assert _ticked_hits(tab) == ({2}, every - span)
        assert len(calls) == 1, "_update_ops rescans every hit: once per toggle, not once per row"
        assert tab.delete_button.isEnabled() and tab.replace_type_button.isEnabled()

        # A trigger row and its hits highlight together; all follow the current (trigger) row.
        _click_row(tree, parent, Qt.ControlModifier)
        _press_space(tree)
        assert _ticked_hits(tab) == (set(), every - span), "the trigger row was ticked, so all five untick"
        assert not tab.delete_button.isEnabled(), "no trigger is ticked"

        del calls[:]
        _menu_action(tree.check_menu(), "Tick selected").trigger()
        assert _ticked_hits(tab) == ({2}, every)
        assert len(calls) == 1
        _click_row(tree, parent.child(0))
        _click_row(tree, parent.child(6), Qt.ShiftModifier)
        _menu_action(tree.check_menu(), "Untick selected").trigger()
        assert _ticked_hits(tab) == ({2}, set())
        assert not tab.replace_type_button.isEnabled() and tab.delete_button.isEnabled()
    finally:
        _close(window)


def test_bulk_unticked_trigger_rows_are_left_out_of_the_next_operation() -> None:
    from PyQt5.QtCore import Qt

    window, _boxes = _trigger_window()
    try:
        tab = _ttab(window)
        tree = tab.tree
        _tfind(tab)
        assert tab.checked_parents == {0, 1, 2, 3}
        assert not any(a.isEnabled() for a in tree.check_menu().actions()), "nothing is highlighted yet"
        _click_row(tree, tree.topLevelItem(0))
        _click_row(tree, tree.topLevelItem(2), Qt.ShiftModifier)
        _press_space(tree)
        assert _ticked_hits(tab) == ({3}, set())
        tab.check_all.click()
        assert _ticked_hits(tab) == ({0, 1, 2, 3}, set())
        _press_space(tree)
        tab.delete_button.click()
        assert [t.name for t in _triggers(window)] == ["[Find] Messages", "--- Section ---", "Retarget"]
    finally:
        _close(window)


def test_hits_hidden_under_a_collapsed_trigger_are_never_bulk_ticked() -> None:
    """TASK-352: like trigger_panel.selected_entry_refs(), a row the user cannot see is not swept in."""
    from PyQt5.QtCore import Qt

    window, _boxes = _trigger_window()
    try:
        tab = _ttab(window)
        tree = tab.tree
        tab.add_type_const(_ARCHER)
        _tfind(tab)
        parent = tree.topLevelItem(0)
        every = {"Condition#0", "Effect#0", "Effect#1", "Effect#2", "Effect#3", "Effect#4", "Effect#5"}
        tab._set_all_checked(False)
        _click_row(tree, parent)
        _click_row(tree, parent.child(3), Qt.ShiftModifier)
        assert len(tree.selectedItems()) == 5
        parent.setExpanded(False)
        assert len(tree.selectedItems()) == 5, "Qt keeps the hidden hits highlighted"
        assert tree.highlighted_checkable() == [parent]

        _press_space(tree)
        assert _ticked_hits(tab) == ({2}, set()), "Space ticks the trigger row only"
        _menu_action(tree.check_menu(), "Untick selected").trigger()
        tab.check_all.click()
        _menu_action(tree.check_menu(), "Untick selected").trigger()
        assert _ticked_hits(tab) == (set(), every), "the menu leaves the hidden hits ticked"
    finally:
        _close(window)


def test_results_refresh_after_trigger_undo_from_units_mode_and_a_panel_delete() -> None:
    window, _boxes = _trigger_window()
    try:
        tab = _ttab(window)
        _tfind(tab, "bridge")
        tab.replace_text_edit.setText("gate")
        tab.replace_text_button.click()
        _settle()
        assert tab.hits == []
        window.mode_combo.setCurrentText("Units")
        window.undo()
        _settle()
        assert len(tab.hits) == 1 and "bridge" in tab.hits[0].value_display

        _tfind(tab)
        assert len({h.trigger_index for h in tab.hits}) == 4
        window.trigger_structural_edit("delete", [1])
        _settle()
        assert [h.trigger_name for h in tab.hits] == ["[Find] Messages", "Retarget", "Find: disabled"]
    finally:
        _close(window)


def test_a_history_jump_across_delete_and_new_clears_checks_and_stale_hits_are_refused() -> None:
    window, _boxes = _trigger_window()
    try:
        tab = _ttab(window)
        _tfind(tab)
        assert tab.checked_parents == {0, 1, 2, 3}
        window.trigger_structural_edit("delete", [3])
        window.trigger_structural_edit("new", [])
        _settle()
        assert tab.checked_parents == set(), "a trigger change the dialog didn't make clears every check"

        _tfind(tab, "bridge")
        hits = list(tab.hits)
        model = window._ensure_trigger_edits()
        with window._trigger_edit(model, "Retype", content_touched=[0]) as m:
            m.manager().triggers[0].effects[0].message = "Hold the bridge now.\r\nArchers to the walls."
        before = window.edit_history.cursor
        notes = window._find_text_replace(hits, "bridge", False, False, "gate")
        assert window.edit_history.cursor == before, "a stale hit is refused, never written"
        assert list(notes.values()) == ["changed since Find; re-run Find"]
        assert _triggers(window)[0].effects[0].message.startswith("Hold the bridge now.")
    finally:
        _close(window)


def test_objects_triggers_column_is_blank_until_triggers_parse_then_follows_edits() -> None:
    window, _boxes = _trigger_window()
    try:
        dialog = _dialog(window)
        dialog.objects_tab.run_find()
        assert all(r.trigger_refs is None for r in dialog.objects_tab.rows)
        assert window.scenario._trigger_manager is None, "the Objects tab must not parse triggers"
        dialog.tabs.setCurrentWidget(dialog.triggers_tab)
        assert _rows(dialog)[_ARCHER_A].trigger_refs == 5
        assert _rows(dialog)[502].trigger_refs == 0
        window.trigger_structural_edit("delete", [2])
        _settle()
        assert _rows(dialog)[_ARCHER_A].trigger_refs == 0
    finally:
        _close(window)


def test_a_half_typed_panel_name_is_committed_before_an_op_which_then_refuses_the_stale_row() -> None:
    """Pins what happens: the Find op commits the focused panel field first (as
    a focus-out would), so the typed text is never lost, and the trigger it
    renamed no longer matches its Find row, so the op refuses it."""
    window, _boxes = _trigger_window()
    try:
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        panel.select_trigger(2)
        editor = next(w for spec, _k, _i, w in panel._rows if spec.name == "name")
        editor.setFocus()
        editor.setText("Retargeted")
        assert window.focusWidget() is editor
        tab = _ttab(window)
        _tfind(tab, "Retarget", scopes=("names",))
        _check_parents(tab, {2})
        before = window.edit_history.cursor
        tab.disable_button.click()
        assert _triggers(window)[2].name == "Retargeted"
        assert window.edit_history.cursor == before + 1, "only the name commit was recorded"
        assert _triggers(window)[2].enabled
        assert "results changed since Find" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_the_tab_is_disabled_when_triggers_do_not_parse(monkeypatch) -> None:
    from descape import viewer as viewer_module

    window, _boxes = _trigger_window()
    try:
        monkeypatch.setattr(viewer_module, "parse_triggers", lambda loaded: None)
        dialog = _dialog(window)
        dialog.tabs.setCurrentWidget(dialog.triggers_tab)
        tab = dialog.triggers_tab
        assert not tab.body.isEnabled()
        assert tab.unsupported.isVisibleTo(tab) and "can't be read" in tab.unsupported.text()
    finally:
        _close(window)


def test_a_read_only_trigger_file_finds_but_cannot_mutate() -> None:
    from descape.scenario_io import parse_triggers

    window, _boxes = _trigger_window()
    try:
        parse_triggers(window.scenario)
        window.scenario.trigger_write_supported = False
        tab = _ttab(window)
        _tfind(tab, "bridge")
        assert len(tab.hits) == 1
        for button in (tab.replace_text_button, tab.delete_button, tab.enable_button, tab.disable_button):
            assert not button.isEnabled()
            assert button.toolTip() == "Triggers are read-only for this file"
    finally:
        _close(window)


def test_save_reload_keeps_every_untouched_triggers_bytes(tmp_path) -> None:
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel

    window, _boxes = _trigger_window()
    try:
        original = load_map_and_units(FIND_FIXTURE)
        parse_triggers(original)
        original_blobs = TriggerEditModel(original)._blobs
        tab = _ttab(window)
        _tfind(tab, "bridge")
        tab.replace_text_edit.setText("gate")
        tab.replace_text_button.click()
        out = tmp_path / "replaced.aoe2scenario"
        write_scenario(window.scenario, out, **window._edit_model_kwargs())
        again = load_map_and_units(out)
        assert parse_triggers(again).triggers[0].effects[0].message.startswith("Hold the gate.")
        blobs = TriggerEditModel(again)._blobs
        assert blobs[1:] == original_blobs[1:]
        assert blobs[0] != original_blobs[0]
    finally:
        _close(window)


# -- GH #144 follow-up: the popup and the Objects tab's ops after a bulk tick ---


def test_a_right_click_on_the_result_list_pops_the_tick_menu_there(monkeypatch) -> None:
    """Pass conditions:
    - a real context-menu event on the Objects list's viewport reaches
      _show_check_menu through customContextMenuRequested, and pops exactly one
      menu, at the click's global position (QMenu.exec_ patched);
    - that menu holds Tick selected / Untick selected, both enabled with a
      highlight, and choosing Untick selected unticks only the highlighted rows."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QContextMenuEvent
    from PyQt5.QtWidgets import QApplication, QMenu

    window, _boxes = _window()
    try:
        tab = _dialog(window).objects_tab
        tab.run_find()
        tree = tab.tree
        assert tree.contextMenuPolicy() == Qt.CustomContextMenu
        _click_row(tree, _object_item(tab, _REF_OAK))
        _click_row(tree, _object_item(tab, _REF_WALL), Qt.ShiftModifier)
        popped = []

        def fake_exec(menu, *args):
            popped.append(([a.text() for a in menu.actions()], [a.isEnabled() for a in menu.actions()], args))
            _menu_action(menu, "Untick selected").trigger()

        monkeypatch.setattr(QMenu, "exec_", fake_exec)
        pos = tree.visualItemRect(_object_item(tab, _REF_PINE)).center()
        global_pos = tree.viewport().mapToGlobal(pos)
        QApplication.sendEvent(tree.viewport(), QContextMenuEvent(QContextMenuEvent.Mouse, pos, global_pos))

        assert len(popped) == 1, popped
        texts, enabled, args = popped[0]
        assert texts == ["Tick selected", "Untick selected"]
        assert enabled == [True, True]
        assert args[0] == global_pos
        assert _ticked_refs(tab) == _ALL_REFS - {_REF_OAK, _REF_PINE, _REF_WALL}
    finally:
        _close(window)


def _bulk_tick_two_archers(tab) -> None:
    """Check none, then Ctrl+click highlight two archers and tick both with one Space."""
    from PyQt5.QtCore import Qt

    tab.check_none.click()
    tree = tab.tree
    _click_row(tree, _object_item(tab, _REF_ARCHER_P1))
    _click_row(tree, _object_item(tab, _REF_ARCHER_P2), Qt.ControlModifier)
    _press_space(tree)
    assert _ticked_refs(tab) == {_REF_ARCHER_P1, _REF_ARCHER_P2}


def test_select_on_map_after_a_partial_bulk_tick_selects_only_the_ticked_rows() -> None:
    """Pass condition: after a bulk tick of two of the eight object rows,
    Select on Map leaves exactly those two units selected in Units mode."""
    window, _boxes = _window()
    try:
        tab = _dialog(window).objects_tab
        tab.run_find()
        _bulk_tick_two_archers(tab)
        tab.select_button.click()
        assert window.mode == "units"
        assert sorted(window._selection) == [(1, _REF_ARCHER_P1), (2, _REF_ARCHER_P2)]
    finally:
        _close(window)


def test_delete_after_a_partial_bulk_tick_deletes_only_the_ticked_rows() -> None:
    """Pass conditions: after a bulk tick of two of the eight object rows,
    Delete confirms "Delete 2 objects", removes exactly those two units in
    one undo step, and leaves the other six placed."""
    window, boxes = _window()
    try:
        dialog = _dialog(window)
        tab = dialog.objects_tab
        tab.run_find()
        _bulk_tick_two_archers(tab)
        before = window.edit_history.cursor
        tab.delete_button.click()
        assert "Delete 2 objects" in boxes.confirmed[-1][1]
        assert window.edit_history.cursor == before + 1
        placed = {u.reference_id for u in window.scenario.unit_manager.get_all_units()}
        assert placed == _ALL_REFS - {_REF_ARCHER_P1, _REF_ARCHER_P2}
    finally:
        _close(window)


def _value_cell_pixels(text: str, spans):
    """(cell, font): the trigger tab's Value cell for `text` and `spans`,
    painted by its own _SpanDelegate in a shown tree, as an (h, w, 3) array,
    and the tree's font it was drawn in."""
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem

    from descape import find_dialog as fd
    from testkit.qt_capture import qimage_rgb888_to_array

    conftest.ensure_qapp()
    tree = QTreeWidget()
    try:
        tree.setColumnCount(len(fd.TRIGGER_COLUMNS))
        tree.setItemDelegateForColumn(fd._TCOL_VALUE, fd._SpanDelegate(tree))
        tree.resize(1400, 160)
        for column in range(fd._TCOL_VALUE):
            tree.setColumnWidth(column, 40)
        tree.setColumnWidth(fd._TCOL_VALUE, 500)
        item = QTreeWidgetItem(tree)
        item.setText(fd._TCOL_VALUE, text)
        item.setData(fd._TCOL_VALUE, fd._SPANS_ROLE, list(spans))
        tree.show()
        _settle()
        rect = tree.visualRect(tree.indexFromItem(item, fd._TCOL_VALUE))
        assert rect.width() == 500
        shot = tree.viewport().grab().toImage().convertToFormat(QImage.Format_RGB888)
        pixels = qimage_rgb888_to_array(shot)
        return pixels[rect.top() : rect.bottom() + 1, rect.left() : rect.right() + 1].astype(int), tree.font()
    finally:
        tree.close()
        tree.deleteLater()


def _ink_columns(cell) -> set[int]:
    """Columns holding text ink: a pixel well below the background's own brightness."""
    background = cell[1, -1].sum()
    return {int(x) for x in np.nonzero((cell.sum(axis=2) < background - 150).any(axis=0))[0]}


@pytest.mark.font_sensitive
def test_the_value_highlight_covers_the_matched_glyphs() -> None:
    """GH #144 (C): the _HIGHLIGHT fill lands on the drawn match, not beside it.

    Passes when, in a Value cell reading "Send the Needle" with the span over
    "Needle" (the match last, so its ink is what the full text adds to the
    prefix's), the columns the fill changes on the cell's second row (the
    fill starts one pixel down) are one run whose ends sit within 1 px of the
    match's ink box widened by its end glyphs' own side bearings ("N"'s left,
    "e"'s right). The fill is advance-based, so the bearings are the measured
    gap between it and the ink; they grow with the DPI (2.7 px for "N" at
    168), which a bare 1 px ink-box tolerance would misread as misalignment.
    A zero-width span renders the same path with no fill, the unfilled
    baseline."""
    from PyQt5.QtGui import QFontMetricsF

    text = "Send the Needle"
    start, end = text.index("Needle"), len(text)
    filled, font = _value_cell_pixels(text, [(start, end)])
    bare, _font = _value_cell_pixels(text, [(start, start)])
    prefix, _font = _value_cell_pixels(text[:start], [(start, start)])
    metrics = QFontMetricsF(font)

    fill_columns = np.nonzero((filled[1] != bare[1]).any(axis=1))[0]
    assert fill_columns.size, "no highlight fill on the cell's second row"
    assert fill_columns[-1] - fill_columns[0] + 1 == fill_columns.size, "the fill is one run"
    match_ink = sorted(_ink_columns(bare) - _ink_columns(prefix))
    assert match_ink, "no ink for the match"
    want_first = match_ink[0] - metrics.leftBearing("N")
    want_last = match_ink[-1] + metrics.rightBearing("e")
    assert abs(fill_columns[0] - want_first) <= 1, (int(fill_columns[0]), want_first)
    assert abs(fill_columns[-1] - want_last) <= 1, (int(fill_columns[-1]), want_last)
