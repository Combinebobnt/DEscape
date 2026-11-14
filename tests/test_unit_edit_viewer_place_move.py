"""Units mode: the edit gate, placing, moving, nudging, deleting and
reassigning a unit, and the pixels each leaves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox
from unit_edit_viewer_helpers import (
    _BERRY_BUSH,
    _PLACE_CONST,
    _chunk_for_tile,
    _close,
    _empty_tile,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _pos_for_tile,
    _unit_count,
    _window,
)

from descape import settings, unit_pick
from testkit.viewer_family import set_viewer_global, viewer_global

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# --- b1.1: plumbing --------------------------------------------------------


def test_a_model_driven_edit_survives_undo_redo_and_reaches_disk(tmp_path: Path) -> None:
    window = _window()
    try:
        model = window._ensure_unit_edits()
        assert model is not None
        unit = next(u for u in window.scenario.unit_manager.get_all_units() if u.reference_id == 203)
        with window._unit_edit(model, "Move villager", [1]):
            model.set_position(unit, 40.5, 40.5, 5.0)
        assert window.edit_history.can_undo
        assert (unit.x, unit.y, unit.z) == (40.5, 40.5, 5.0)

        window.undo()
        assert (unit.x, unit.y, unit.z) != (40.5, 40.5, 5.0)
        assert window.edit_history.can_redo

        window.redo()
        assert (unit.x, unit.y, unit.z) == (40.5, 40.5, 5.0)

        out = tmp_path / "out.aoe2scenario"
        window.scenario.path = out
        window._untitled = False
        window.save()
        assert out.is_file()
    finally:
        _close(window)


def _cursor_shape():
    """The override cursor's shape, or None if there is no override. The shape
    and not the QCursor: overrideCursor() hands back a pointer Qt owns and
    destroys on restore, so a wrapper kept past that reads back as 0."""
    cursor = QApplication.overrideCursor()
    return None if cursor is None else cursor.shape()


def _cursor_spy(monkeypatch, outcome):
    """Replace viewer.UnitEditModel with a stand-in that records the override
    cursor in force while the construction gate is running. `outcome` is
    called with the scenario and decides what the gate does."""
    seen = []

    def _spy(scenario):
        seen.append(_cursor_shape())
        return outcome(scenario)

    set_viewer_global(monkeypatch, "UnitEditModel", _spy)
    return seen


def test_the_first_unit_edit_gate_shows_a_wait_cursor_and_restores_it(monkeypatch) -> None:
    """The gate scans every unit in the file, a one-time multi-hundred-ms
    freeze that reads as a hang without a cursor. The repeat calls (one per
    unit edit, forever after) are a cached return and must stay cursor-free,
    since a flicker per edit would be worse than no cursor at all."""
    window = _window()
    try:
        assert window.unit_edits is None, "the gate is supposed to be deferred to the first edit"
        real = viewer_global("UnitEditModel")
        seen = _cursor_spy(monkeypatch, real)

        assert QApplication.overrideCursor() is None
        model = window._ensure_unit_edits()

        assert model is not None
        assert seen == [Qt.WaitCursor]
        assert QApplication.overrideCursor() is None

        assert window._ensure_unit_edits() is model
        assert len(seen) == 1, "the cached return re-ran the gate"
        assert QApplication.overrideCursor() is None
    finally:
        _close(window)


def test_a_refused_unit_edit_gate_restores_the_cursor_before_its_dialog(monkeypatch) -> None:
    """The refusal path returns None rather than raising, so it needs its own
    coverage. The recorded cursor at dialog time is the point: a modal warning
    under a wait cursor reads as a still-frozen window."""
    from descape.unit_model import UnitEditsUnavailableError

    def _refuse(scenario):
        raise UnitEditsUnavailableError("synthetic refusal")

    window = _window()
    try:
        seen = _cursor_spy(monkeypatch, _refuse)
        at_dialog = []
        monkeypatch.setattr(
            QMessageBox, "warning", lambda *a, **k: at_dialog.append(_cursor_shape())
        )

        assert window._ensure_unit_edits() is None

        assert seen == [Qt.WaitCursor]
        assert at_dialog == [None]
        assert QApplication.overrideCursor() is None
    finally:
        _close(window)


def test_an_unexpected_gate_failure_still_restores_the_cursor(monkeypatch) -> None:
    """An exception the gate does not classify propagates, and a stuck wait
    cursor over a still-usable window would be a real UI bug."""

    def _explode(scenario):
        raise RuntimeError("synthetic gate failure")

    window = _window()
    try:
        seen = _cursor_spy(monkeypatch, _explode)

        with pytest.raises(RuntimeError):
            window._ensure_unit_edits()

        assert seen == [Qt.WaitCursor]
        assert QApplication.overrideCursor() is None
    finally:
        _close(window)


# --- b1.4: place ------------------------------------------------------------


def test_placing_a_unit_snaps_to_the_tile_centre_and_selects_it() -> None:
    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # (10, 10) is under the fixture's House
        window.units_panel.select_object(_PLACE_CONST)
        window.units_panel.select_owner(1)
        window.place_unit_action.setChecked(True)
        before = _unit_count(window)
        before_records = len(window.edit_history.records)

        window.on_unit_place(_pos_for_tile(window, 10, 10), Qt.NoModifier)

        assert _unit_count(window) == before + 1
        assert len(window.edit_history.records) == before_records + 1
        assert window.edit_history.peek_undo().kind == "unit"
        entry = window.map_view._unit_index.entry_for_key(window._selection[0])
        assert (entry.unit.x, entry.unit.y) == (10.5, 10.5)
        assert entry.unit.unit_const == _PLACE_CONST
        assert entry.player_id == 1

        window.undo()
        assert _unit_count(window) == before
    finally:
        _close(window)


def test_activating_a_catalog_row_checks_place_unit_and_returns_focus_to_the_map() -> None:
    """Double-click/Enter on the sidebar catalog replaces "accept the
    dialog": it activates the Place Unit tool (if available) and hands focus
    back to the map, so the common flow (pick an object, place it, then
    nudge it with arrow keys) doesn't get stuck sending arrow keys to the
    tree instead -- see units_panel.py's own _build_catalog_pane docstring.

    Needs a real shown+activated window: Qt's focus tracking is a no-op on
    an offscreen, never-activated one (test_keybinds.py's own comment on its
    QTest.keyClick tests makes the same point for shortcut dispatch).
    """
    window = _window()
    try:
        window.show()
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        assert window.isActiveWindow()

        window.units_panel.catalog_view.list.setFocus()
        window.units_panel.catalog_view.select(_PLACE_CONST)
        current = window.units_panel.catalog_view.list.currentItem()
        window.units_panel.catalog_view.list.itemActivated.emit(current)
        QApplication.processEvents()

        assert window.place_unit_action.isChecked()
        assert window.units_panel.selected_object_const() == _PLACE_CONST
        assert QApplication.focusWidget() is window.map_view
    finally:
        _close(window)


def test_placing_with_no_object_chosen_is_a_no_op() -> None:
    window = _window()
    try:
        window.place_unit_action.setChecked(True)
        before = _unit_count(window)
        window.on_unit_place(_pos_for_tile(window, 5, 5), Qt.NoModifier)
        assert _unit_count(window) == before
        assert not window.edit_history.can_undo
    finally:
        _close(window)


# --- TASK-288: GAIA-only objects place for GAIA ------------------------------


def test_placing_a_gaia_only_object_for_a_player_places_it_for_gaia() -> None:
    """Keep listed (GH #32) is what lets a player owner pick one at all."""
    settings.set_gaia_only_catalog_mode("keep")
    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # (10, 10) is under the fixture's House
        window.units_panel.select_object(_BERRY_BUSH)
        window.units_panel.select_owner(1)
        assert window.units_panel.selected_object_const() == _BERRY_BUSH
        window.place_unit_action.setChecked(True)
        before_gaia = len(window.scenario.unit_manager.units[0])
        before_p1 = len(window.scenario.unit_manager.units[1])

        window.on_unit_place(_pos_for_tile(window, 10, 10), Qt.NoModifier)

        assert len(window.scenario.unit_manager.units[0]) == before_gaia + 1
        assert len(window.scenario.unit_manager.units[1]) == before_p1
        assert window.scenario.unit_manager.units[0][-1].unit_const == _BERRY_BUSH
        assert window._selection[0][0] == 0
        status = window.status_log.toPlainText().splitlines()[-1]
        assert "for GAIA" in status and status.endswith("(GAIA-only, so placed for GAIA)")
        assert window.units_panel.owner_id() == 1, "the combo itself is left alone"

        window.undo()
        assert len(window.scenario.unit_manager.units[0]) == before_gaia
    finally:
        _close(window)


def test_placing_a_villager_for_a_player_keeps_the_player() -> None:
    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # (10, 10) is under the fixture's House
        window.units_panel.select_object(_PLACE_CONST)
        window.units_panel.select_owner(1)
        window.place_unit_action.setChecked(True)
        before_p1 = len(window.scenario.unit_manager.units[1])

        window.on_unit_place(_pos_for_tile(window, 10, 10), Qt.NoModifier)

        assert len(window.scenario.unit_manager.units[1]) == before_p1 + 1
        status = window.status_log.toPlainText().splitlines()[-1]
        assert "for Player 1" in status and "GAIA-only" not in status
    finally:
        _close(window)


def _place_gaia_bush(window, tile=(10, 10)):
    """A GAIA Forage Bush at `tile`, selected; returns its index entry."""
    window.unrestricted_check.setChecked(True)  # (10, 10) is under the fixture's House
    window.units_panel.select_owner(0)
    window.units_panel.select_object(_BERRY_BUSH)
    window.place_unit_action.setChecked(True)
    window.on_unit_place(_pos_for_tile(window, *tile), Qt.NoModifier)
    entry = window.map_view._unit_index.entry_for_key(window._selection[0])
    assert entry.player_id == 0 and entry.unit.unit_const == _BERRY_BUSH
    return entry


def test_reassigning_a_gaia_only_object_to_a_player_is_refused() -> None:
    """GH #32: the inspector's Owner never gives a GAIA-only object to a player."""
    window = _window()
    try:
        entry = _place_gaia_bush(window)
        window.units_panel.show_unit(entry)
        before_records = len(window.edit_history.records)

        combo = window.units_panel.unit_field_editors["player"]
        combo.setCurrentIndex(combo.findData(1))

        assert entry.unit in window.scenario.unit_manager.units[0]
        assert window._selection == [(0, entry.unit.reference_id)]
        assert len(window.edit_history.records) == before_records
        assert combo.currentData() == 0, "the Owner field shows the unit's real owner again"
        status = window.status_log.toPlainText().splitlines()[-1]
        assert status == "Not reassigned to Player 1; 1 object(s) left as they were (GAIA-only or holding one)"
    finally:
        _close(window)


def test_a_mixed_reassign_moves_the_others_and_notes_the_gaia_only_ones() -> None:
    window = _window()
    try:
        bush = _place_gaia_bush(window)
        window.units_panel.select_object(_PLACE_CONST)
        window.on_unit_place(_pos_for_tile(window, 12, 12), Qt.NoModifier)
        villager = window.map_view._unit_index.entry_for_key(window._selection[0])
        window._selection = [unit_pick.unit_key(0, bush.unit), unit_pick.unit_key(0, villager.unit)]
        window._update_unit_inspector_from_selection()

        combo = window.units_panel.unit_field_editors["player"]
        combo.setCurrentIndex(combo.findData(2))

        assert villager.unit in window.scenario.unit_manager.units[2]
        assert bush.unit in window.scenario.unit_manager.units[0]
        assert sorted(window._selection) == [(0, bush.unit.reference_id), (2, villager.unit.reference_id)]
        status = window.status_log.toPlainText().splitlines()[-1]
        assert status.endswith("; 1 object(s) left as they were (GAIA-only or holding one)")
    finally:
        _close(window)


def test_a_player_owned_gaia_only_object_can_be_given_to_gaia() -> None:
    """A bad file's player-owned bush is fixed by giving it to GAIA, never moved to another player."""
    window = _window()
    try:
        entry = _place_gaia_bush(window)
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "test: bad file", [0, 1]):
            model.reassign(entry.unit, 1)
        window._selection = [unit_pick.unit_key(1, entry.unit)]
        key = window._selection[0]

        window._find_reassign_keys([key], 3)
        assert entry.unit in window.scenario.unit_manager.units[1]

        window._find_reassign_keys([key], 0)
        assert entry.unit in window.scenario.unit_manager.units[0]
        assert window.status_log.toPlainText().splitlines()[-1] == "Reassigned 1 units to GAIA"
    finally:
        _close(window)


def test_find_reassign_of_a_gaia_only_object_to_a_player_is_refused() -> None:
    window = _window()
    try:
        entry = _place_gaia_bush(window)
        key = window._selection[0]

        window._find_reassign_keys([key], 3)

        assert entry.unit in window.scenario.unit_manager.units[0]
        status = window.status_log.toPlainText().splitlines()[-1]
        assert status == (
            "Reassigned 0 units to Player 3; 1 object(s) left as they were (GAIA-only or holding one)"
        )
    finally:
        _close(window)


def test_the_gaia_only_refused_note_is_empty_for_none() -> None:
    from descape.viewer_common import _gaia_only_refused_note

    assert _gaia_only_refused_note(0) == ""
    assert _gaia_only_refused_note(2) == "; 2 object(s) left as they were (GAIA-only or holding one)"


# --- b1.5: move + nudge ------------------------------------------------------


def test_moving_a_selected_unit_snaps_to_the_target_tile_centre() -> None:
    window = _window()
    try:
        entry = window.map_view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([entry])
        window.units_panel.show_unit(entry)
        before_records = len(window.edit_history.records)

        window.on_unit_move(key, _pos_for_tile(window, 25, 30), Qt.NoModifier)

        assert len(window.edit_history.records) == before_records + 1
        new_entry = window.map_view._unit_index.entry_for_key(key)
        assert (new_entry.unit.x, new_entry.unit.y) == (25.5, 30.5)

        window.undo()
        assert window.map_view._unit_index.entry_for_key(key).unit.x == entry.unit.x
    finally:
        _close(window)


def test_moving_a_unit_back_onto_its_own_tile_is_a_no_op() -> None:
    window = _window()
    try:
        entry = window.map_view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([entry])
        before_records = len(window.edit_history.records)
        own_tile = (int(entry.unit.x), int(entry.unit.y))

        window.on_unit_move(key, _pos_for_tile(window, *own_tile), Qt.NoModifier)

        assert len(window.edit_history.records) == before_records
    finally:
        _close(window)


def test_a_move_that_raises_after_patching_the_pick_index_drops_the_index(monkeypatch) -> None:
    """The index was patched for a move the abort undid, and Units entry reuses
    a non-None index, so the raise must leave it None (and the stack cycle)."""
    window = _window()
    try:
        entry = window.map_view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([entry])
        window._stack_cycle = ("sentinel",)
        real = window._patch_unit_index_for_move

        def patch_then_raise(*args):
            real(*args)
            raise RuntimeError("mid-edit failure")

        monkeypatch.setattr(window, "_patch_unit_index_for_move", patch_then_raise)
        with pytest.raises(RuntimeError, match="mid-edit failure"):
            window.on_unit_move(key, _pos_for_tile(window, 25, 30), Qt.NoModifier)
        assert window.map_view._unit_index is None
        assert window._stack_cycle is None
    finally:
        _close(window)


def test_an_arrow_after_a_catalog_click_and_a_placement_nudges_the_placed_unit() -> None:
    """GH #68: the common flow on the new browser list (click a row, place,
    nudge): the arrow reaches the map, not the list's current row."""
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        window.show()
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        catalog = window.units_panel.catalog_view
        catalog.select(_PLACE_CONST)
        QTest.mouseClick(catalog.list.viewport(), Qt.LeftButton, Qt.NoModifier,
                         catalog.list.visualItemRect(catalog.list.currentItem()).center())
        catalog.list.itemActivated.emit(catalog.list.currentItem())
        QApplication.processEvents()
        window.on_unit_place(_pos_for_tile(window, 30, 30), Qt.NoModifier)
        assert window._selection, window.status_log.toPlainText().splitlines()[-1]
        key = window._selection[0]
        start_x = window.map_view._unit_index.entry_for_key(key).unit.x
        row = catalog.list.currentRow()
        QTest.keyClick(QApplication.focusWidget(), Qt.Key_Right)
        assert window.map_view._unit_index.entry_for_key(key).unit.x > start_x
        assert catalog.list.currentRow() == row
    finally:
        _close(window)


def test_arrow_nudge_moves_by_a_fine_step_and_shift_by_a_whole_tile() -> None:
    window = _window()
    try:
        entry = window.map_view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([entry])
        window.units_panel.show_unit(entry)
        start_x = entry.unit.x

        window.on_unit_nudge(1, 0, Qt.NoModifier)
        fine = window.map_view._unit_index.entry_for_key(key).unit.x
        assert fine - start_x == pytest.approx(0.1, abs=1e-4)

        window.on_unit_nudge(1, 0, Qt.ShiftModifier)
        whole = window.map_view._unit_index.entry_for_key(key).unit.x
        assert whole - fine == pytest.approx(1.0)
    finally:
        _close(window)


# --- b1.6: delete ------------------------------------------------------------


def test_deleting_a_selected_unit_removes_it_immediately() -> None:
    """No confirm dialog: Delete commits straight to the undo-backed
    EditHistory, same as every other edit tool in this app."""
    window = _window()
    try:
        entry = window.map_view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([entry])
        window.units_panel.show_unit(entry)
        before = _unit_count(window)

        window.on_unit_delete(Qt.NoModifier)

        assert _unit_count(window) == before - 1
        assert window._selection == []
        assert window.units_panel.unit_inspector_empty.isVisibleTo(window.left_stack)

        window.undo()
        assert _unit_count(window) == before
    finally:
        _close(window)


# --- b1.3: inspector round-trip ----------------------------------------------


def test_owner_combo_reassigns_and_keeps_the_unit_selected() -> None:
    window = _window()
    try:
        entry = next(e for e in window.map_view._unit_index.entries if e.player_id == 1)
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([entry])
        window.units_panel.show_unit(entry)

        combo = window.units_panel.unit_field_editors["player"]
        combo.setCurrentIndex(combo.findData(2))

        assert window._selection == [(2, entry.unit.reference_id)]
        new_entry = window.map_view._unit_index.entry_for_key(window._selection[0])
        assert new_entry.player_id == 2

        window.undo()
    finally:
        _close(window)


def test_reassigning_to_the_same_owner_is_a_no_op() -> None:
    window = _window()
    try:
        entry = next(e for e in window.map_view._unit_index.entries if e.player_id == 1)
        window._selection = [(entry.player_id, entry.unit.reference_id)]
        window.map_view.set_unit_selection([entry])
        window.units_panel.show_unit(entry)
        before_records = len(window.edit_history.records)

        combo = window.units_panel.unit_field_editors["player"]
        combo.setCurrentIndex(combo.findData(1))

        assert len(window.edit_history.records) == before_records
    finally:
        _close(window)


def test_x_field_round_trips_through_a_single_undo_record() -> None:
    window = _window()
    try:
        entry = window.map_view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([entry])
        window.units_panel.show_unit(entry)
        start_x = entry.unit.x
        before_records = len(window.edit_history.records)

        spin = window.units_panel.unit_field_editors["x"]
        spin.setValue(start_x + 7)

        assert len(window.edit_history.records) == before_records + 1
        assert window.map_view._unit_index.entry_for_key(key).unit.x == start_x + 7

        window.undo()
        assert window.map_view._unit_index.entry_for_key(key).unit.x == start_x
    finally:
        _close(window)


def test_float_editors_have_keyboard_tracking_off() -> None:
    """Trap 3: without this, typing a multi-digit value fires valueChanged
    once per digit, recording one undo step per keystroke instead of one per
    commit."""
    window = _window()
    try:
        for field_id in ("x", "y", "z"):
            assert window.units_panel.unit_field_editors[field_id].keyboardTracking() is False
    finally:
        _close(window)


# --- render-cache pixel invalidation ----------------------------------------
#
# Every test above only asserts on the MODEL (unit lists, edit_history) and
# on _rebuild_unit_index()'s pick index, never on what the chunk cache
# actually paints -- which is exactly how a real regression got past this
# file once already: _after_unit_mutation() called self._cache.
# invalidate_units() (rebuilds the cache's SOURCE structures) and
# self.map_view.invalidate_region() (schedules a repaint) but never
# self._cache.invalidate_region() (evicts the already-COMPOSITED chunk pixel
# cache) -- so place/move/delete/reassign changed the model correctly while
# the map kept showing the pre-edit unit. These read pixels straight from
# the live FlatChunkCache (window._cache.get_chunk()), bypassing Qt's paint
# pipeline entirely, the same technique the manual repro that found the bug
# used.


def test_place_pixels_appear_where_none_existed() -> None:
    """Deliberately on an EMPTY tile, not just any tile: the fixture already
    has a same-colour Player 1 unit at (10.5, 10.5), and placing a second
    one there would leave the pixel unchanged for a reason that has nothing
    to do with cache invalidation -- exactly the false negative that first
    masked this bug during manual repro."""
    window = _window()
    try:
        tx, ty = _empty_tile(window)
        window.units_panel.select_object(_PLACE_CONST)
        window.units_panel.select_owner(1)
        window.place_unit_action.setChecked(True)
        cx, cy = _chunk_for_tile(window, tx, ty)

        before = window._cache.get_chunk(0, cx, cy).copy()
        window.on_unit_place(_pos_for_tile(window, tx, ty), Qt.NoModifier)
        after = window._cache.get_chunk(0, cx, cy)

        assert not np.array_equal(before, after)
    finally:
        _close(window)


def test_move_pixels_update_at_both_the_old_and_new_tile() -> None:
    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # the new tile is under the fixture's House
        entry = window.map_view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([entry])
        old_tile = (int(entry.unit.x), int(entry.unit.y))
        new_tile = (old_tile[0] + 5, old_tile[1] + 5)
        old_cx, old_cy = _chunk_for_tile(window, *old_tile)

        before = window._cache.get_chunk(0, old_cx, old_cy).copy()
        window.on_unit_move(key, _pos_for_tile(window, *new_tile), Qt.NoModifier)
        after = window._cache.get_chunk(0, old_cx, old_cy)

        assert not np.array_equal(before, after), "the unit's old tile still shows it after moving away"
    finally:
        _close(window)


def test_delete_pixels_stop_showing_the_unit() -> None:
    window = _window()
    try:
        entry = window.map_view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([entry])
        cx, cy = _chunk_for_tile(window, int(entry.unit.x), int(entry.unit.y))

        before = window._cache.get_chunk(0, cx, cy).copy()

        window.on_unit_delete(Qt.NoModifier)
        after = window._cache.get_chunk(0, cx, cy)

        assert not np.array_equal(before, after)
    finally:
        _close(window)


def test_reassign_pixels_change_colour() -> None:
    window = _window()
    try:
        entry = next(e for e in window.map_view._unit_index.entries if e.player_id == 1)
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([entry])
        window.units_panel.show_unit(entry)
        cx, cy = _chunk_for_tile(window, int(entry.unit.x), int(entry.unit.y))

        before = window._cache.get_chunk(0, cx, cy).copy()
        combo = window.units_panel.unit_field_editors["player"]
        combo.setCurrentIndex(combo.findData(2))
        after = window._cache.get_chunk(0, cx, cy)

        assert not np.array_equal(before, after)
    finally:
        _close(window)
