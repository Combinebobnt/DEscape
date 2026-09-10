"""Phase 3.5b's b1 slice: place/move/delete/reassign driven through a real
offscreen ViewerWindow, on top of phase 3's selection wiring
(test_unit_selection_viewer.py) and phase 3.5a's headless model
(test_units_undo.py).

Uses the real units_120x120.aoe2scenario fixture, not the blank template's
SyntheticUnit stand-ins: UnitEditModel's construction gate needs real,
byte-verified unit structs (see unit_model.py's docstring), which a
synthetic dataclass can't satisfy.

Every test forces Flat terrain style before entering Units mode: MapView.
_pick_tile()'s Flat branch is a plain pixel/tile_px division, so scene
positions are trivial to construct by hand. Stepped/Sloped's own inverse
maths is covered by test_unit_pick.py, not duplicated here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtWidgets import QMessageBox

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

# A real object const from object_catalog's Units category -- Villager
# (unit_const 83), same as tools/gen_units_fixture.py uses for its own
# placed-unit coverage.
_PLACE_CONST = 83


def _window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    # Unchecked BEFORE the style switch, while still in Stepped -- iso_action
    # defaults checked (MapView._isometric's own default), so Flat would
    # otherwise render through the Flat+Isometric plan's real-iso path
    # instead of the plain top-down canvas this module means to exercise.
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Units")
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _unit_count(window) -> int:
    return sum(len(u) for u in window.scenario.unit_manager.units)


def _pos_for_tile(window, tile_x: int, tile_y: int) -> QPointF:
    tp = window.map_view._tile_pixels
    return QPointF(tile_x * tp + tp // 2, tile_y * tp + tp // 2)


def _empty_tile(window) -> tuple[int, int]:
    """A tile with no unit on it in the fixture -- for a pixel check that
    must start from a genuinely blank chunk, not one that happens to already
    hold a same-colour unit (see test_place_pixels_appear_where_none_existed's
    own docstring for why "was there a unit here already" is load-bearing)."""
    occupied = {(int(u.x), int(u.y)) for units in window.scenario.unit_manager.units for u in units}
    return next((x, y) for x in range(60, 80) for y in range(60, 80) if (x, y) not in occupied)


def _chunk_for_tile(window, tile_x: int, tile_y: int) -> tuple[int, int]:
    tp = window.map_view._tile_pixels
    cp = window._cache.chunk_px
    return (tile_x * tp) // cp, (tile_y * tp) // cp


def _rect_for_tile(window, tile_x: int, tile_y: int) -> QRectF:
    """A scene rect covering exactly one tile -- Flat's own tile_px grid,
    same arithmetic _pos_for_tile uses for a single point."""
    tp = window.map_view._tile_pixels
    return QRectF(tile_x * tp, tile_y * tp, tp, tp)


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


# --- b1.4: place ------------------------------------------------------------


def test_placing_a_unit_snaps_to_the_tile_centre_and_selects_it() -> None:
    window = _window()
    try:
        window.place_object_edit.set_value(_PLACE_CONST)
        window.place_owner_combo.setCurrentIndex(window.place_owner_combo.findData(1))
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


# --- b1.5: move + nudge ------------------------------------------------------


def test_moving_a_selected_unit_snaps_to_the_target_tile_centre() -> None:
    window = _window()
    try:
        entry = window.map_view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([entry])
        window._update_unit_inspector(entry)
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


def test_arrow_nudge_moves_by_a_fine_step_and_shift_by_a_whole_tile() -> None:
    window = _window()
    try:
        entry = window.map_view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([entry])
        window._update_unit_inspector(entry)
        start_x = entry.unit.x

        window.on_unit_nudge(1, 0, Qt.NoModifier)
        fine = window.map_view._unit_index.entry_for_key(key).unit.x
        assert 0 < fine - start_x < 1

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
        window._update_unit_inspector(entry)
        before = _unit_count(window)

        window.on_unit_delete(Qt.NoModifier)

        assert _unit_count(window) == before - 1
        assert window._selection == []
        assert window.unit_inspector_empty.isVisibleTo(window.left_stack)

        window.undo()
        assert _unit_count(window) == before
    finally:
        _close(window)


def test_deleting_a_garrison_referenced_unit_is_refused(monkeypatch) -> None:
    """D4: the garrisoned_in_id guard is a hard refusal, unrelated to (and
    unaffected by the removal of) the confirm dialog."""
    window = _window()
    try:
        entries = window.map_view._unit_index.entries
        target, other = entries[0], entries[1]
        other.unit.garrisoned_in_id = target.unit.reference_id
        key = (target.player_id, target.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([target])
        before = _unit_count(window)

        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
        window.on_unit_delete(Qt.NoModifier)

        assert _unit_count(window) == before
        assert len(warned) == 1
        assert not window.edit_history.can_undo
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
        window._update_unit_inspector(entry)

        combo = window.unit_field_editors["player"]
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
        window._update_unit_inspector(entry)
        before_records = len(window.edit_history.records)

        combo = window.unit_field_editors["player"]
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
        window._update_unit_inspector(entry)
        start_x = entry.unit.x
        before_records = len(window.edit_history.records)

        spin = window.unit_field_editors["x"]
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
            assert window.unit_field_editors[field_id].keyboardTracking() is False
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
        window.place_object_edit.set_value(_PLACE_CONST)
        window.place_owner_combo.setCurrentIndex(window.place_owner_combo.findData(1))
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
        window._update_unit_inspector(entry)
        cx, cy = _chunk_for_tile(window, int(entry.unit.x), int(entry.unit.y))

        before = window._cache.get_chunk(0, cx, cy).copy()
        combo = window.unit_field_editors["player"]
        combo.setCurrentIndex(combo.findData(2))
        after = window._cache.get_chunk(0, cx, cy)

        assert not np.array_equal(before, after)
    finally:
        _close(window)


# --- b2: bulk operations (marquee, group move/delete, Convert) -------------
#
# The N-way selection highlight (MapView._draw_unit_selection unioning
# multiple entries into one QPainterPath) is new scene-item code with the
# same exposure b1's own postmortem flagged just above: every model-level
# assertion in this file's earlier tests would pass unchanged even if the
# cache invalidation for a GROUP edit were broken, since none of them read
# pixels for more than one unit at a time. The delete test below does, at
# two DIFFERENT chunks, so a regression that invalidated only one tile (or
# only the first selected unit) would fail it.


def test_group_delete_removes_every_selected_unit_and_repaints_both_tiles() -> None:
    window = _window()
    try:
        index = window.map_view._unit_index
        a, b = index.entries[0], index.entries[6]
        assert (int(a.unit.x), int(a.unit.y)) != (int(b.unit.x), int(b.unit.y))
        cx_a, cy_a = _chunk_for_tile(window, int(a.unit.x), int(a.unit.y))
        cx_b, cy_b = _chunk_for_tile(window, int(b.unit.x), int(b.unit.y))
        assert (cx_a, cy_a) != (cx_b, cy_b), "fixture must give two units in different chunks or this proves nothing"

        window._selection = [(a.player_id, a.unit.reference_id), (b.player_id, b.unit.reference_id)]
        window.map_view.set_unit_selection([a, b])
        before_a = window._cache.get_chunk(0, cx_a, cy_a).copy()
        before_b = window._cache.get_chunk(0, cx_b, cy_b).copy()
        before_count = _unit_count(window)

        window.on_unit_delete(Qt.NoModifier)

        assert _unit_count(window) == before_count - 2
        assert not np.array_equal(before_a, window._cache.get_chunk(0, cx_a, cy_a))
        assert not np.array_equal(before_b, window._cache.get_chunk(0, cx_b, cy_b))
        assert window._selection == []
        assert window.edit_history.can_undo
        # One record for the whole group, not one per unit.
        assert window.edit_history.peek_undo().kind == "unit"
        window.undo()
        assert _unit_count(window) == before_count
    finally:
        _close(window)


def test_group_nudge_moves_every_selected_unit_in_one_undo_record() -> None:
    window = _window()
    try:
        index = window.map_view._unit_index
        a, b = index.entries[0], index.entries[6]
        ax, ay = a.unit.x, a.unit.y
        bx, by = b.unit.x, b.unit.y
        window._selection = [(a.player_id, a.unit.reference_id), (b.player_id, b.unit.reference_id)]
        records_before = len(window.edit_history.records)

        window.on_unit_nudge(1, 0, Qt.NoModifier)

        assert len(window.edit_history.records) == records_before + 1
        assert (a.unit.x, a.unit.y) == (ax + 0.1, ay)
        assert (b.unit.x, b.unit.y) == (bx + 0.1, by)
    finally:
        _close(window)


def test_marquee_select_plain_replaces_the_selection() -> None:
    window = _window()
    try:
        index = window.map_view._unit_index
        a, b = index.entries[0], index.entries[3]
        window._selection = [(99, 99)]  # a stale key that must be discarded, not merged

        window.on_marquee_select([(a.player_id, a.unit.reference_id), (b.player_id, b.unit.reference_id)], Qt.NoModifier)

        assert set(window._selection) == {(a.player_id, a.unit.reference_id), (b.player_id, b.unit.reference_id)}
    finally:
        _close(window)


def test_marquee_select_ctrl_toggles_membership() -> None:
    window = _window()
    try:
        index = window.map_view._unit_index
        a, b = index.entries[0], index.entries[3]
        key_a, key_b = (a.player_id, a.unit.reference_id), (b.player_id, b.unit.reference_id)
        window._selection = [key_a]

        # A second marquee covering both A (already selected) and B toggles
        # each independently: A drops out, B is added.
        window.on_marquee_select([key_a, key_b], Qt.ControlModifier)

        assert set(window._selection) == {key_b}
    finally:
        _close(window)


def test_marquee_select_shift_only_adds() -> None:
    window = _window()
    try:
        index = window.map_view._unit_index
        a, b = index.entries[0], index.entries[3]
        key_a, key_b = (a.player_id, a.unit.reference_id), (b.player_id, b.unit.reference_id)
        window._selection = [key_a]

        window.on_marquee_select([key_a, key_b], Qt.ShiftModifier)

        assert set(window._selection) == {key_a, key_b}
    finally:
        _close(window)


def test_marquee_over_a_filtered_out_player_selects_nothing() -> None:
    """The marquee's own filter-respecting requirement (plan hard rule: a
    drag across a forest must not grab every hidden tree) -- exercised here
    against a hidden PLAYER rather than GAIA/trees, since this fixture's
    units are placed by player, not by tree-const. GAIA player 0's units
    sit at tiles (5,5)-(7,5); hiding GAIA and marqueeing exactly that strip
    must return nothing, even though units_in_rect's own tile math would
    otherwise find them."""
    window = _window()
    try:
        window.show_gaia_action.setChecked(False)
        rect = QRectF(5 * window.map_view._tile_pixels, 5 * window.map_view._tile_pixels, 3 * window.map_view._tile_pixels, window.map_view._tile_pixels)
        keys = window.map_view._units_in_scene_rect(rect)
        assert keys == []
    finally:
        _close(window)


def test_units_in_scene_rect_finds_units_under_the_marquee() -> None:
    window = _window()
    try:
        index = window.map_view._unit_index
        entry = index.entries[0]
        rect = _rect_for_tile(window, int(entry.unit.x), int(entry.unit.y))
        keys = window.map_view._units_in_scene_rect(rect)
        assert (entry.player_id, entry.unit.reference_id) in keys
    finally:
        _close(window)


# --- b3: rotate ---------------------------------------------------------------

_REF_ARCHER_P1 = 201  # type 70, angle_count 16 -- rotation IS an angle
_REF_VILLAGER_P1 = 203  # type 70, angle_count 16
_REF_WALL = 102  # GAIA wall -- rotation is a shape-variant index
_REF_HOUSE = 200  # angle_count 1 -- nothing to select, INERT


def _select(window, *reference_ids):
    """Selects the fixture units with these reference_ids, through the same
    reconciliation point every real selection path uses."""
    index = window.map_view._unit_index
    entries = [e for e in index.entries if e.unit.reference_id in reference_ids]
    assert len(entries) == len(reference_ids), "fixture unit missing from the index"
    window._selection = [(e.player_id, e.unit.reference_id) for e in entries]
    window._refresh_selection_view()
    return entries


def test_rotate_turns_the_selected_unit_by_one_stored_frame() -> None:
    window = _window()
    try:
        from descape import unit_rotation

        (entry,) = _select(window, _REF_ARCHER_P1)
        key = (entry.player_id, entry.unit.reference_id)
        before = entry.unit.rotation

        window.on_unit_rotate(1)

        after = window.map_view._unit_index.entry_for_key(key).unit.rotation
        assert after == pytest.approx(unit_rotation.rotate_step(before, 16, 1))

        window.undo()
        assert window.map_view._unit_index.entry_for_key(key).unit.rotation == before
    finally:
        _close(window)


def test_rotate_skips_a_wall_and_leaves_it_untouched() -> None:
    """A marquee over a village will always catch walls and doodads. They are
    skipped, not refused -- and the file must not change for them."""
    window = _window()
    try:
        (entry,) = _select(window, _REF_WALL)
        before = entry.unit.rotation

        window.on_unit_rotate(1)

        assert entry.unit.rotation == before
        assert not window.edit_history.is_dirty, "a fully-skipped rotate must record nothing"
        assert "not an angle" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_rotate_skips_an_inert_const_too() -> None:
    window = _window()
    try:
        (entry,) = _select(window, _REF_HOUSE)
        before = entry.unit.rotation
        window.on_unit_rotate(1)
        assert entry.unit.rotation == before
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_a_mixed_selection_rotates_the_angle_units_and_reports_the_skips() -> None:
    window = _window()
    try:
        entries = _select(window, _REF_ARCHER_P1, _REF_VILLAGER_P1, _REF_WALL)
        wall = next(e for e in entries if e.unit.reference_id == _REF_WALL)
        rotatables = [e for e in entries if e.unit.reference_id != _REF_WALL]
        before = {e.unit.reference_id: e.unit.rotation for e in entries}

        window.on_unit_rotate(1)

        assert wall.unit.rotation == before[_REF_WALL]
        for entry in rotatables:
            assert entry.unit.rotation != before[entry.unit.reference_id]
        assert "1 skipped" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_group_rotate_is_a_single_undo_record() -> None:
    window = _window()
    try:
        entries = _select(window, _REF_ARCHER_P1, _REF_VILLAGER_P1)
        before = [e.unit.rotation for e in entries]

        window.on_unit_rotate(1)
        assert [e.unit.rotation for e in entries] != before

        window.undo()
        assert [e.unit.rotation for e in entries] == before
    finally:
        _close(window)


def test_a_coarse_rotate_turns_a_quarter_of_the_stored_frames() -> None:
    """angle_count 16, so a quarter turn is 4 whole frames -- not an exact
    90 degrees written raw."""
    window = _window()
    try:
        from descape import unit_rotation

        (entry,) = _select(window, _REF_ARCHER_P1)
        before = entry.unit.rotation

        window.on_unit_rotate_coarse(1)

        assert entry.unit.rotation == pytest.approx(unit_rotation.rotate_step(before, 16, 4))
    finally:
        _close(window)


def test_rotate_does_nothing_outside_units_mode() -> None:
    """The keybind is a QAction shortcut, so it is live in every mode --
    unlike the arrow-key nudge, which MapView only injects in Units mode."""
    window = _window()
    try:
        (entry,) = _select(window, _REF_ARCHER_P1)
        before = entry.unit.rotation
        window.mode_combo.setCurrentText("Terrain")

        window.on_unit_rotate(1)

        assert entry.unit.rotation == before
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_the_rotate_actions_need_a_selection() -> None:
    window = _window()
    try:
        assert all(not a.isEnabled() for a in window._rotate_actions)
        _select(window, _REF_ARCHER_P1)
        assert all(a.isEnabled() for a in window._rotate_actions)
    finally:
        _close(window)


# --- b3: gate orientation cycling --------------------------------------------
#
# The fixture holds no gate, so each of these re-points the GAIA wall at a
# stone gate const and its own anchor first. Every path under test reads the
# live unit's fields, and _after_unit_mutation() is the same index rebuild a
# real edit already runs, so the index agrees with the swapped const.
_GATE_NE, _GATE_E, _GATE_SE = 64, 659, 88


def _make_gate(window, x: float = 10.0, y: float = 5.5):
    (entry,) = _select(window, _REF_WALL)
    entry.unit.unit_const, entry.unit.x, entry.unit.y = _GATE_NE, x, y
    window._after_unit_mutation()
    return _select(window, _REF_WALL)[0]


def _occupied(window, unit):
    from descape import render

    mm = window.scenario.map_manager
    return render.unit_occupied_tiles(unit, mm.map_width, mm.map_height)


def test_rotate_cycles_a_selected_gate_to_its_next_orientation() -> None:
    """A gate's orientation lives in its const, so "rotating" one swaps the
    const and re-anchors. The footprint really changes: (4, 1) along x becomes
    the sparse 6-tile diagonal, which is asserted on tiles rather than pixels.
    """
    window = _window()
    try:
        entry = _make_gate(window)
        before = _occupied(window, entry.unit)

        window.on_unit_rotate(1)

        assert (entry.unit.unit_const, entry.unit.x, entry.unit.y) == (_GATE_E, 10.0, 7.0)
        after = _occupied(window, entry.unit)
        assert len(before) == 4 and len(after) == 6
        assert before != after
    finally:
        _close(window)


def test_a_coarse_rotate_cycles_a_gate_by_two_orientations() -> None:
    """One gate step is 45 degrees, so a quarter turn is two of them: ne
    lands on se, not on the diagonal in between."""
    window = _window()
    try:
        entry = _make_gate(window)

        window.on_unit_rotate_coarse(1)

        assert (entry.unit.unit_const, entry.unit.x, entry.unit.y) == (_GATE_SE, 8.5, 7.0)
    finally:
        _close(window)


def test_a_gate_and_an_archer_rotate_together_in_one_undo_record() -> None:
    window = _window()
    try:
        _make_gate(window)
        entries = _select(window, _REF_WALL, _REF_ARCHER_P1)
        gate = next(e for e in entries if e.unit.reference_id == _REF_WALL)
        archer = next(e for e in entries if e.unit.reference_id == _REF_ARCHER_P1)
        before = (gate.unit.unit_const, archer.unit.rotation)

        window.on_unit_rotate(1)
        assert gate.unit.unit_const == _GATE_E
        assert archer.unit.rotation != before[1]

        window.undo()
        assert (gate.unit.unit_const, archer.unit.rotation) == before
        assert (gate.unit.x, gate.unit.y) == (10.0, 5.5)
    finally:
        _close(window)


def test_a_gate_with_no_room_at_the_map_edge_refuses_and_says_so() -> None:
    """Cycling ne to the diagonal grows the footprint three tiles down the y
    axis. Against the map's far edge that would write a hanging footprint, so
    the viewer, the layer that knows the map's size, skips it."""
    window = _window()
    try:
        entry = _make_gate(window, x=10.0, y=117.5)
        before = (entry.unit.unit_const, entry.unit.x, entry.unit.y)

        window.on_unit_rotate(1)

        assert (entry.unit.unit_const, entry.unit.x, entry.unit.y) == before
        assert not window.edit_history.is_dirty
        assert "map edge" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_the_inspector_shows_an_editor_only_for_an_angle_const() -> None:
    window = _window()
    try:
        editor = window.unit_field_editors["rotation"]
        label = window.unit_field_labels["rotation"]

        _select(window, _REF_ARCHER_P1)
        assert editor.isVisibleTo(window.unit_inspector_grid)
        assert not label.isVisibleTo(window.unit_inspector_grid)

        _select(window, _REF_WALL)
        assert not editor.isVisibleTo(window.unit_inspector_grid)
        assert label.isVisibleTo(window.unit_inspector_grid)
    finally:
        _close(window)


def test_typing_a_rotation_into_the_inspector_records_one_undo_step() -> None:
    window = _window()
    try:
        (entry,) = _select(window, _REF_ARCHER_P1)
        before = entry.unit.rotation

        window.unit_field_editors["rotation"].setValue(1.5)

        assert entry.unit.rotation == pytest.approx(1.5)
        window.undo()
        assert entry.unit.rotation == before
    finally:
        _close(window)
