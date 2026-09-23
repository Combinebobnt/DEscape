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
from PyQt5.QtWidgets import QApplication, QMessageBox

from descape import render

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
    _show_garrisoned(window)
    window.mode_combo.setCurrentText("Units")
    return window


def _window_with_style(style: str):
    """Like _window(), but for Stepped/Sloped instead of Flat -- Batch D's
    D5 scoped-patch wiring (dirty_screen_bbox_iso/_sloped via
    _patch_unit_edit_cache) has no Flat counterpart to exercise it against,
    since Flat's own branch there is patch_rects, not a bbox call."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.terrain_style_combo.setCurrentText(style)
    _show_garrisoned(window)
    window.mode_combo.setCurrentText("Units")
    return window


def _show_garrisoned(window) -> None:
    """GH #42 ships Filters > Show Garrisoned Units UNCHECKED, which would
    drop the fixture's garrisoned villager (reference_id 203, inside house
    200) out of the pick index and renumber every positional
    `index.entries[n]` in this module. This file tests edit mechanics, not
    filtering, so it opts the whole fixture back into view; the tests that
    are about the hidden state turn it off again themselves."""
    window.show_garrisoned_action.setChecked(True)


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
    import descape.viewer as viewer_module

    seen = []

    def _spy(scenario):
        seen.append(_cursor_shape())
        return outcome(scenario)

    monkeypatch.setattr(viewer_module, "UnitEditModel", _spy)
    return seen


def test_the_first_unit_edit_gate_shows_a_wait_cursor_and_restores_it(monkeypatch) -> None:
    """The gate scans every unit in the file, a one-time multi-hundred-ms
    freeze that reads as a hang without a cursor. The repeat calls (one per
    unit edit, forever after) are a cached return and must stay cursor-free,
    since a flicker per edit would be worse than no cursor at all."""
    import descape.viewer as viewer_module

    window = _window()
    try:
        assert window.unit_edits is None, "the gate is supposed to be deferred to the first edit"
        real = viewer_module.UnitEditModel
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

        window.units_panel.catalog_view.tree.setFocus()
        window.units_panel.catalog_view.select(_PLACE_CONST)
        current = window.units_panel.catalog_view.tree.currentItem()
        window.units_panel.catalog_view.tree.itemActivated.emit(current, 0)
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


# --- GH #42: a host's garrison follows it ------------------------------


def _hide_garrisoned(window) -> None:
    """Back to the app's own default (_show_garrisoned's opposite): the
    occupant is not in the index and cannot be selected, which is the state
    every carry-along below has to work in."""
    window.show_garrisoned_action.setChecked(False)


def _fixture_host_and_occupant(window):
    """The fixture's House 200 and the Villager 203 inside it, moved onto the
    house's own point first -- the file places the villager one tile over,
    where a real occupant always shares its host's coordinates."""
    units = window.scenario.unit_manager.units[1]
    host = next(u for u in units if u.reference_id == _REF_HOUSE)
    occupant = next(u for u in units if u.reference_id == _REF_VILLAGER_P1)
    assert occupant.garrisoned_in_id == host.reference_id
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Co-locate", [1], fields_only=True):
        model.set_position(occupant, host.x, host.y, host.z)
    return host, occupant


def test_dragging_a_host_carries_its_hidden_garrison(monkeypatch) -> None:
    window = _window()
    try:
        host, occupant = _fixture_host_and_occupant(window)
        _hide_garrisoned(window)
        key = (1, host.reference_id)
        window._selection = [key]
        window._refresh_selection_view()
        records_before = len(window.edit_history.records)

        window.on_unit_move(key, _pos_for_tile(window, 40, 44), Qt.NoModifier)

        assert (host.x, host.y) == (40.5, 44.5)
        assert (occupant.x, occupant.y) == (40.5, 44.5)
        assert len(window.edit_history.records) == records_before + 1

        window.undo()
        assert (occupant.x, occupant.y) == (10.5, 10.5)
    finally:
        _close(window)


def test_nudging_a_host_carries_its_hidden_garrison() -> None:
    window = _window()
    try:
        host, occupant = _fixture_host_and_occupant(window)
        _hide_garrisoned(window)
        window._selection = [(1, host.reference_id)]
        window._refresh_selection_view()
        records_before = len(window.edit_history.records)

        window.on_unit_nudge(1, 0, Qt.ShiftModifier)

        assert (occupant.x, occupant.y) == (host.x, host.y)
        assert host.x == 11.5
        assert len(window.edit_history.records) == records_before + 1
    finally:
        _close(window)


def test_typing_x_on_a_host_carries_its_hidden_garrison() -> None:
    window = _window()
    try:
        host, occupant = _fixture_host_and_occupant(window)
        _hide_garrisoned(window)
        window._selection = [(1, host.reference_id)]
        window._refresh_selection_view()
        records_before = len(window.edit_history.records)

        window.units_panel.unit_field_editors["x"].setValue(60.5)

        assert host.x == 60.5
        assert occupant.x == 60.5
        assert occupant.y == host.y
        assert len(window.edit_history.records) == records_before + 1

        window.undo()
        assert (host.x, occupant.x) == (10.5, 10.5)
    finally:
        _close(window)


def test_a_group_move_carries_every_selected_hosts_garrison() -> None:
    window = _window()
    try:
        host, occupant = _fixture_host_and_occupant(window)
        _hide_garrisoned(window)
        other = next(u for u in window.scenario.unit_manager.units[2] if u.reference_id == _REF_ARCHER_P2)
        window._selection = [(1, host.reference_id), (2, other.reference_id)]
        window._refresh_selection_view()
        records_before = len(window.edit_history.records)

        window.on_unit_nudge(0, 1, Qt.ShiftModifier)

        assert (occupant.x, occupant.y) == (host.x, host.y)
        assert host.y == 11.5
        assert other.y == 21.5
        assert len(window.edit_history.records) == records_before + 1
    finally:
        _close(window)


def _place_tower(window):
    """A Watch Tower (5 places, holds villagers/foot/monks) on an empty tile,
    selected -- the fixture's own host is a House, which the game gives no
    garrison places at all."""
    model = window._ensure_unit_edits()
    tile = _empty_tile(window)
    with window._unit_edit(model, "Place tower", [1]):
        tower = model.add(1, 79, tile[0] + 0.5, tile[1] + 0.5)
    window._rebuild_unit_index()
    window._selection = [(1, tower.reference_id)]
    window._refresh_selection_view()
    return tower


def test_selecting_a_host_lists_its_occupants_and_reports_an_impossible_one() -> None:
    """The fixture's villager sits in a House, which the game gives 0 places
    and no unit type -- shown as it is, with both notes, never corrected."""
    window = _window()
    try:
        _select(window, _REF_HOUSE)
        panel = window.units_panel
        assert panel.garrison_tree.topLevelItemCount() == 1
        assert "Garrison (1 / 0)" in panel.garrison_header.text()
        assert panel.garrison_note.text()
        assert not panel.garrison_add_button.isEnabled()
    finally:
        _close(window)


def test_a_plain_unit_has_no_garrison_block() -> None:
    window = _window()
    try:
        _select(window, _REF_ARCHER_P1)
        assert not window.units_panel.garrison_tree.isVisibleTo(window.units_panel)
    finally:
        _close(window)


def test_adding_an_occupant_puts_it_inside_the_host_and_undo_removes_it() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        _hide_garrisoned(window)
        before = _unit_count(window)
        records_before = len(window.edit_history.records)
        entry = window.map_view._unit_index.entry_for_key((1, tower.reference_id))

        window._garrison_add_const(entry, 4)  # Archer

        added = next(
            u for u in window.scenario.unit_manager.units[1] if u.garrisoned_in_id == tower.reference_id
        )
        assert (added.x, added.y) == (tower.x, tower.y)
        assert _unit_count(window) == before + 1
        assert len(window.edit_history.records) == records_before + 1
        # Hidden by the default filter, so it is not pickable either.
        assert window.map_view._unit_index.entry_for_key((1, added.reference_id)) is None
        assert window.units_panel.garrison_tree.topLevelItemCount() == 1

        window.undo()
        assert _unit_count(window) == before
        assert window.units_panel.garrison_tree.topLevelItemCount() == 0
    finally:
        _close(window)


def test_an_object_the_host_cannot_hold_is_refused() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        before = _unit_count(window)
        entry = window.map_view._unit_index.entry_for_key((1, tower.reference_id))

        window._garrison_add_const(entry, 280)  # Mangonel: no building holds siege

        assert _unit_count(window) == before
        assert "cannot go inside" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_full_host_refuses_a_further_occupant() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        entry = window.map_view._unit_index.entry_for_key((1, tower.reference_id))
        for _ in range(5):  # a Watch Tower's five places
            window._garrison_add_const(entry, 4)
        before = _unit_count(window)

        window._garrison_add_const(entry, 4)

        assert _unit_count(window) == before
        assert "is full" in window.status_log.toPlainText()
        assert not window.units_panel.garrison_add_button.isEnabled()
    finally:
        _close(window)


def test_deleting_from_the_garrison_list_removes_only_that_occupant() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        entry = window.map_view._unit_index.entry_for_key((1, tower.reference_id))
        window._garrison_add_const(entry, 4)
        window._garrison_add_const(entry, 83)
        occupants = [u for u in window.scenario.unit_manager.units[1] if u.garrisoned_in_id == tower.reference_id]
        assert len(occupants) == 2
        before = _unit_count(window)
        records_before = len(window.edit_history.records)

        window._on_garrison_delete([occupants[0].reference_id])

        assert _unit_count(window) == before - 1
        assert len(window.edit_history.records) == records_before + 1
        left = [u for u in window.scenario.unit_manager.units[1] if u.garrisoned_in_id == tower.reference_id]
        assert [u.reference_id for u in left] == [occupants[1].reference_id]
        assert window.units_panel.garrison_tree.topLevelItemCount() == 1

        window.undo()
        assert _unit_count(window) == before
    finally:
        _close(window)


def test_double_clicking_a_row_selects_the_occupant_only_when_it_is_shown() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        entry = window.map_view._unit_index.entry_for_key((1, tower.reference_id))
        window._garrison_add_const(entry, 4)
        occupant = next(
            u for u in window.scenario.unit_manager.units[1] if u.garrisoned_in_id == tower.reference_id
        )
        _hide_garrisoned(window)

        window._on_garrison_navigate(occupant.reference_id)
        assert window._selection == [(1, tower.reference_id)]
        assert "Show Garrisoned Units" in window.status_log.toPlainText()

        window.show_garrisoned_action.setChecked(True)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        window._on_garrison_navigate(occupant.reference_id)
        assert window._selection == [(1, occupant.reference_id)]
    finally:
        _close(window)


def test_a_unit_whose_own_reference_id_is_minus_one_holds_nothing(monkeypatch) -> None:
    """-1 is what every ungarrisoned unit carries, and the model's reverse map
    keeps that bucket on purpose. A unit whose OWN reference_id is -1 must
    therefore not read as the holder of every ungarrisoned unit in the file:
    no garrison block, and no cascade sweeping the map into one delete. The
    model still refuses to remove it (its own guard reads that bucket), which
    is the one live path left for the refusal warning."""
    window = _window()
    try:
        host = next(u for u in window.scenario.unit_manager.units[1] if u.reference_id == _REF_HOUSE)
        host.reference_id = -1
        window._rebuild_unit_index()
        window._selection = [(1, -1)]
        window._refresh_selection_view()
        assert not window.units_panel.garrison_tree.isVisibleTo(window.units_panel)
        before = _unit_count(window)

        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
        window.on_unit_delete(Qt.NoModifier)

        assert _unit_count(window) == before
        assert len(warned) == 1
    finally:
        _close(window)


def test_deleting_a_host_takes_its_garrison_with_it_in_one_undo_step(monkeypatch) -> None:
    """GH #42: this used to be the hard "referenced by a garrison" refusal.
    An occupant is hidden by default now, so it cannot be selected beside its
    host -- refusing would leave the host undeletable, and deleting only the
    host would dangle the link, so the delete cascades instead."""
    window = _window()
    try:
        entries = window.map_view._unit_index.entries
        target, other = entries[0], entries[1]
        other.unit.garrisoned_in_id = target.unit.reference_id
        key = (target.player_id, target.unit.reference_id)
        window._selection = [key]
        window.map_view.set_unit_selection([target])
        before = _unit_count(window)
        records_before = len(window.edit_history.records)

        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
        window.on_unit_delete(Qt.NoModifier)

        assert not warned
        assert _unit_count(window) == before - 2
        live = {id(u) for units in window.scenario.unit_manager.units for u in units}
        assert id(target.unit) not in live and id(other.unit) not in live
        assert len(window.edit_history.records) == records_before + 1
        assert "1 garrisoned" in window.status_log.toPlainText()

        window.undo()
        assert _unit_count(window) == before
        back = {u.reference_id for units in window.scenario.unit_manager.units for u in units}
        assert {target.unit.reference_id, other.unit.reference_id} <= back
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


def _unit_layout(window) -> list[list[int]]:
    return [[id(u) for u in units] for units in window.scenario.unit_manager.units]


def _refuse_single_remove(window, monkeypatch) -> None:
    """Makes the per-unit remove() fail loudly, so a group delete that slid
    back to looping it can't pass by accident."""
    model = window._ensure_unit_edits()

    def _no_loop(unit):
        raise AssertionError("a 2+ group delete must go through remove_many(), not remove()")

    monkeypatch.setattr(model, "remove", _no_loop)


def test_group_delete_across_three_players_is_one_record_and_undo_restores_exact_indices(monkeypatch) -> None:
    window = _window()
    try:
        _refuse_single_remove(window, monkeypatch)
        # Each target sits mid-list or at the head, so a survivor behind it shifts down.
        entries = _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2, _REF_TREE_PINE)
        assert len({e.player_id for e in entries}) == 3
        survivors = {
            id(u) for units in window.scenario.unit_manager.units for u in units
        } - {id(e.unit) for e in entries}
        layout_before = _unit_layout(window)
        records_before = len(window.edit_history.records)

        window.on_unit_delete(Qt.NoModifier)

        after = {id(u) for units in window.scenario.unit_manager.units for u in units}
        assert after == survivors
        assert len(window.edit_history.records) == records_before + 1
        assert window.edit_history.peek_undo().kind == "unit"
        assert window._selection == []
        for e in entries:
            assert window.map_view._unit_index.entry_for_key((e.player_id, e.unit.reference_id)) is None

        window.undo()
        assert _unit_layout(window) == layout_before
        for e in entries:
            assert window.map_view._unit_index.entry_for_key((e.player_id, e.unit.reference_id)) is not None

        window.redo()
        after_redo = {id(u) for units in window.scenario.unit_manager.units for u in units}
        assert after_redo == survivors
    finally:
        _close(window)


def test_a_group_delete_cascades_every_hosts_garrison_in_one_record(monkeypatch) -> None:
    """The group counterpart of the cascade: this file used to assert a
    partial refusal here (GH #42 removed the refusal), so what it now pins is
    that an unselected occupant of a selected host goes in the same record,
    through remove_many() rather than a remove() loop."""
    window = _window()
    try:
        _refuse_single_remove(window, monkeypatch)
        entries = _select(window, _REF_ARCHER_P1, _REF_VILLAGER_P1, _REF_ARCHER_P2)
        host = next(e for e in entries if e.unit.reference_id == _REF_ARCHER_P2)
        garrisoned = next(u for u in window.scenario.unit_manager.units[2] if u.reference_id == _REF_VILLAGER_P2)
        garrisoned.garrisoned_in_id = _REF_ARCHER_P2
        layout_before = _unit_layout(window)
        records_before = len(window.edit_history.records)
        before = _unit_count(window)

        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
        window.on_unit_delete(Qt.NoModifier)

        assert not warned
        assert _unit_count(window) == before - 4  # three selected, plus the occupant
        live = {id(u) for units in window.scenario.unit_manager.units for u in units}
        assert id(garrisoned) not in live
        assert not any(id(e.unit) in live for e in entries)
        assert id(host.unit) not in live
        assert len(window.edit_history.records) == records_before + 1
        assert "1 garrisoned" in window.status_log.toPlainText()

        window.undo()
        assert _unit_layout(window) == layout_before
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
        tile_px = window.map_view._tile_pixels
        rect = QRectF(5 * tile_px, 5 * tile_px, 3 * tile_px, tile_px)
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
_REF_TREE_PINE = 101
_REF_ARCHER_P2 = 300
_REF_VILLAGER_P2 = 301


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
        # GH #61: the status line speaks the inspector's facing, not radians.
        expected = unit_rotation.rotation_to_facing(after, 16)
        assert f"to facing {expected}/16" in window.status_log.toPlainText()

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


# --- Cycle Variant -------------------------------------------------------------
_REF_TREE_OAK = 100  # GAIA oak, 42 variants, stored 7.0


def test_cycle_variant_steps_the_selected_tree_and_undoes() -> None:
    window = _window()
    try:
        (entry,) = _select(window, _REF_TREE_OAK)
        key = (entry.player_id, entry.unit.reference_id)

        window.on_unit_variant(1)

        assert window.map_view._unit_index.entry_for_key(key).unit.rotation == 8.0
        assert "Cycled Tree Oak to variant 8/42" in window.status_log.toPlainText()
        window.undo()
        assert window.map_view._unit_index.entry_for_key(key).unit.rotation == 7.0
    finally:
        _close(window)


def test_cycle_variant_changes_the_resolved_frame() -> None:
    from descape import unit_rotation, unit_variant

    window = _window()
    try:
        (entry,) = _select(window, _REF_TREE_OAK)
        const = entry.unit.unit_const
        args = (unit_rotation.angle_count_for(const), unit_variant.variant_count_for(const))
        before = unit_variant.variant_of(entry.unit.rotation, *args)
        window.on_unit_variant(-1)
        assert unit_variant.variant_of(entry.unit.rotation, *args) == before - 1
    finally:
        _close(window)


def test_a_mixed_variant_selection_cycles_the_tree_and_reports_the_skips() -> None:
    window = _window()
    try:
        entries = _select(window, _REF_TREE_OAK, _REF_WALL, _REF_ARCHER_P1)
        before = {e.unit.reference_id: e.unit.rotation for e in entries}

        window.on_unit_variant(1)

        after = {e.unit.reference_id: e.unit.rotation for e in entries}
        assert after[_REF_TREE_OAK] != before[_REF_TREE_OAK]
        assert after[_REF_WALL] == before[_REF_WALL]
        assert after[_REF_ARCHER_P1] == before[_REF_ARCHER_P1]
        assert "2 skipped (this object has no graphic variants)" in window.status_log.toPlainText()
        window.undo()
        assert {e.unit.reference_id: e.unit.rotation for e in entries} == before
    finally:
        _close(window)


def test_a_fully_skipped_variant_cycle_records_nothing() -> None:
    window = _window()
    try:
        (entry,) = _select(window, _REF_WALL)
        before = entry.unit.rotation
        window.on_unit_variant(1)
        assert entry.unit.rotation == before
        assert not window.edit_history.is_dirty
        assert "no graphic variants" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_randomize_variant_is_seedable_and_always_changes() -> None:
    import random

    window = _window()
    try:
        (entry,) = _select(window, _REF_TREE_OAK)
        window._variant_rng = random.Random(5)
        expected = random.Random(5).randrange(41)
        expected = float(expected + 1 if expected >= 7 else expected)

        window.on_unit_variant(randomize=True)

        assert entry.unit.rotation == expected
        assert entry.unit.rotation != 7.0
        assert "Randomized Tree Oak to variant" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_cycle_variant_does_nothing_outside_units_mode() -> None:
    window = _window()
    try:
        (entry,) = _select(window, _REF_TREE_OAK)
        window.mode_combo.setCurrentText("Terrain")
        window.on_unit_variant(1)
        assert entry.unit.rotation == 7.0
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_the_variant_actions_need_a_selection() -> None:
    window = _window()
    try:
        assert len(window._variant_actions) == 3
        assert all(not a.isEnabled() for a in window._variant_actions)
        _select(window, _REF_TREE_OAK)
        assert all(a.isEnabled() for a in window._variant_actions)
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
        editor = window.units_panel.unit_field_editors["rotation"]
        label = window.units_panel.unit_field_labels["rotation"]

        _select(window, _REF_ARCHER_P1)
        assert editor.isVisibleTo(window.units_panel.unit_inspector_grid)
        assert not label.isVisibleTo(window.units_panel.unit_inspector_grid)

        _select(window, _REF_WALL)
        assert not editor.isVisibleTo(window.units_panel.unit_inspector_grid)
        assert label.isVisibleTo(window.units_panel.unit_inspector_grid)
    finally:
        _close(window)


def test_typing_a_rotation_into_the_inspector_records_one_undo_step() -> None:
    window = _window()
    try:
        import math

        (entry,) = _select(window, _REF_ARCHER_P1)
        before = entry.unit.rotation

        window.units_panel.unit_field_editors["rotation"].setValue(4)

        assert entry.unit.rotation == pytest.approx(math.pi / 2)
        window.undo()
        assert entry.unit.rotation == before
    finally:
        _close(window)


# --- GH #71: inspector field edits on a group ------------------------------


def _field(field_id):
    from descape import unit_fields

    return unit_fields.FIELDS_BY_ID[field_id]


def _owner_of(window, reference_id: int) -> int:
    units = window.scenario.unit_manager.units
    return next(p for p, us in enumerate(units) if any(u.reference_id == reference_id for u in us))


def test_typing_x_on_a_group_sets_every_unit_in_one_undo_record() -> None:
    window = _window()
    try:
        entries = _select(window, _REF_ARCHER_P1, _REF_VILLAGER_P1, _REF_ARCHER_P2)
        before = {e.unit.reference_id: (e.unit.x, e.unit.y, e.unit.z) for e in entries}
        spin = window.units_panel.unit_field_editors["x"]
        assert spin.text() == "(mixed)"
        records_before = len(window.edit_history.records)

        spin.setValue(50.0)

        assert len(window.edit_history.records) == records_before + 1
        for entry in entries:
            _x, y, z = before[entry.unit.reference_id]
            assert (entry.unit.x, entry.unit.y, entry.unit.z) == (50.0, y, z)
        # The refreshed group view now shows the common value.
        assert spin.text() == "50.00"
        assert len(window._selection) == 3
        assert "Set X on 3 units" in window.status_log.toPlainText()

        window.undo()
        for entry in entries:
            assert (entry.unit.x, entry.unit.y, entry.unit.z) == before[entry.unit.reference_id]
    finally:
        _close(window)


def test_a_group_move_via_the_field_keeps_the_pick_index_in_step() -> None:
    window = _window()
    try:
        _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        window.units_panel.unit_field_editors["y"].setValue(40.5)
        index = window.map_view._unit_index
        for key in window._selection:
            entry = index.entry_for_key(key)
            assert entry.own_y == 40
    finally:
        _close(window)


def test_typing_rotation_on_a_mixed_group_skips_the_tree_and_the_wall() -> None:
    import math

    window = _window()
    try:
        entries = _select(window, _REF_ARCHER_P1, _REF_TREE_OAK, _REF_WALL)
        by_ref = {e.unit.reference_id: e.unit for e in entries}
        tree_before = by_ref[_REF_TREE_OAK].rotation
        wall_before = by_ref[_REF_WALL].rotation
        assert tree_before == 7.0
        records_before = len(window.edit_history.records)

        window.units_panel.unit_field_editors["rotation"].setValue(4)

        assert by_ref[_REF_ARCHER_P1].rotation == pytest.approx(math.pi / 2)
        assert by_ref[_REF_TREE_OAK].rotation == tree_before
        assert by_ref[_REF_WALL].rotation == wall_before
        assert len(window.edit_history.records) == records_before + 1
        assert "(2 skipped: not rotatable)" in window.status_log.toPlainText()

        window.undo()
        assert by_ref[_REF_ARCHER_P1].rotation == 0.0
        assert by_ref[_REF_TREE_OAK].rotation == tree_before
    finally:
        _close(window)


def test_typing_a_facing_on_a_group_sets_every_member_to_it() -> None:
    import math

    window = _window()
    try:
        entries = _select(window, _REF_ARCHER_P1, _REF_VILLAGER_P2)
        window._on_unit_field_changed(_field("rotation"), 3)
        for entry in entries:
            assert entry.unit.rotation == pytest.approx(3 * 2 * math.pi / 16)
    finally:
        _close(window)


def test_committing_the_displayed_facing_leaves_an_off_grid_value_alone() -> None:
    """GH #61: 7.0 shows as facing 2; committing 2 must not snap it to 2*2pi/16."""
    import math

    window = _window()
    try:
        archer = next(e for e in window.map_view._unit_index.entries if e.unit.reference_id == _REF_ARCHER_P1)
        archer.unit.rotation = 7.0
        (entry,) = _select(window, _REF_ARCHER_P1)
        assert window.units_panel.unit_field_editors["rotation"].value() == 2

        window._on_unit_field_changed(_field("rotation"), 2)
        assert entry.unit.rotation == 7.0
        assert not window.edit_history.is_dirty

        window._on_unit_field_changed(_field("rotation"), 3)
        assert entry.unit.rotation == pytest.approx(3 * 2 * math.pi / 16)
    finally:
        _close(window)


def test_a_facing_on_a_mixed_direction_count_group_lands_each_on_its_own_frame() -> None:
    """Archer (16) + trebuchet (32): the panel's scale is 32, and facing 10
    of 32 is facing 5 of 16 for the archer, the same direction."""
    import math

    window = _window()
    try:
        treb = next(e for e in window.map_view._unit_index.entries if e.unit.reference_id == _REF_ARCHER_P2)
        treb.unit.unit_const = 42
        entries = _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        by_ref = {e.unit.reference_id: e.unit for e in entries}
        spin = window.units_panel.unit_field_editors["rotation"]
        assert spin.maximum() == 31
        records_before = len(window.edit_history.records)

        spin.setValue(10)

        assert by_ref[_REF_ARCHER_P2].rotation == pytest.approx(10 * 2 * math.pi / 32)
        assert by_ref[_REF_ARCHER_P1].rotation == pytest.approx(5 * 2 * math.pi / 16)
        assert len(window.edit_history.records) == records_before + 1
        assert spin.text() == "10"
    finally:
        _close(window)


def test_owner_change_across_two_players_is_one_record_and_follows_the_selection() -> None:
    window = _window()
    try:
        _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2, _REF_VILLAGER_P1)
        combo = window.units_panel.unit_field_editors["player"]
        assert combo.currentText() == "(mixed)"
        records_before = len(window.edit_history.records)

        combo.setCurrentIndex(combo.findData(3))

        assert len(window.edit_history.records) == records_before + 1
        assert sorted(window._selection) == [(3, _REF_ARCHER_P1), (3, _REF_VILLAGER_P1), (3, _REF_ARCHER_P2)]
        for ref in (_REF_ARCHER_P1, _REF_ARCHER_P2, _REF_VILLAGER_P1):
            assert _owner_of(window, ref) == 3
        assert combo.currentData() == 3
        assert combo.findText("(mixed)") < 0

        window.undo()
        assert _owner_of(window, _REF_ARCHER_P1) == 1
        assert _owner_of(window, _REF_VILLAGER_P1) == 1
        assert _owner_of(window, _REF_ARCHER_P2) == 2
    finally:
        _close(window)


def test_a_group_owner_change_skips_members_already_owned() -> None:
    window = _window()
    try:
        _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        before_p2 = [u.reference_id for u in window.scenario.unit_manager.units[2]]
        window._on_unit_field_changed(_field("player"), 2)
        # The P2 archer stays at its own index; only the P1 archer appends.
        assert [u.reference_id for u in window.scenario.unit_manager.units[2]] == [*before_p2, _REF_ARCHER_P1]
        assert sorted(window._selection) == [(2, _REF_ARCHER_P1), (2, _REF_ARCHER_P2)]
        assert "Reassigned 1 units to Player 2" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_no_op_group_edit_records_nothing() -> None:
    window = _window()
    try:
        _select(window, _REF_ARCHER_P1, _REF_VILLAGER_P1)  # same y, owner and rotation
        window._on_unit_field_changed(_field("y"), 10.5)
        window._on_unit_field_changed(_field("player"), 1)
        window._on_unit_field_changed(_field("rotation"), 0.0)
        window._on_unit_field_changed(_field("z"), 0.0)
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


def test_a_rotation_on_a_group_with_no_angle_member_records_nothing() -> None:
    window = _window()
    try:
        _select(window, _REF_TREE_OAK, _REF_WALL)
        window._on_unit_field_changed(_field("rotation"), 1.0)
        assert not window.edit_history.is_dirty
    finally:
        _close(window)


# --- Batch D's D5: scoped repaint on Stepped/Sloped -------------------------
#
# Every pixel test above runs on Flat (this module's own docstring), whose
# _patch_unit_edit_cache() branch is patch_rects(), not a dirty_screen_bbox_*
# call -- so Move's Stepped/Sloped branch has no coverage from them at all.
# These pin both: pixels move correctly, AND the splice actually engaged
# (render.sprite_draws_by_anchor(), the wholesale per-level rebuild D4's
# splice exists to avoid, is never called) -- a call-counted oracle in the
# same spirit as test_invalidate_units_splice.py's own, but through the real
# ViewerWindow wiring D5 added rather than a direct cache call.


def _move_without_wholesale_rebuild(window, monkeypatch) -> None:
    """Drives the Set-field ("x") call site rather than on_unit_move(): the
    latter goes through MapView._pick_tile()'s screen-to-tile inverse, which
    is Flat-only arithmetic in this module by design (its own docstring) --
    Stepped/Sloped's real inverse is test_unit_pick.py's job, not this
    file's. The inspector spin editor sets x directly, still through
    UnitEditModel.set_position() and the same _after_unit_mutation()/
    _patch_unit_edit_cache() tail on_unit_move() itself uses.

    Reads the whole canvas via render_rect(), not a single get_chunk(): a
    chunk's (cx, cy) index is a flat tile_px/chunk_px division in Flat, but
    Stepped/Sloped project a tile to an arbitrary screen diamond, so there
    is no cheap way to know which chunk a unit's sprite lands in without
    duplicating that projection here.

    _REF_VILLAGER_P1, not _REF_ARCHER_P1: the archer's own tile sits inside
    _REF_HOUSE's multi-tile footprint in this fixture, which is exactly
    _splice_eligible()'s documented shared-tile fallback (own-tile sharing
    is common enough that Batch D's D4 measured it directly, 5 of 16 corpus
    files) -- a real edit, correctly NOT spliced, but the wrong unit to
    prove the splice path with. The villager has no such neighbour."""
    (entry,) = _select(window, _REF_VILLAGER_P1)
    window.units_panel.show_unit(entry)
    old_x = entry.unit.x

    canvas_w, canvas_h = window._cache.canvas_dims(0)
    # Builds level 0 once, normally -- the edit itself must not pay for this
    # again (D4's splice deliberately never bumps _source_gen).
    before = window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()

    calls = []
    real = render.sprite_draws_by_anchor
    monkeypatch.setattr(render, "sprite_draws_by_anchor", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    window.units_panel.unit_field_editors["x"].setValue(old_x + 5)
    after = window._cache.render_rect(0, 0, canvas_w, canvas_h)

    assert not np.array_equal(before, after), "moving the unit should change the composited canvas"
    assert calls == [], "a spliced unit edit should not pay for a wholesale sprite-source rebuild"


def test_stepped_move_scopes_the_repaint_and_skips_the_wholesale_rebuild(monkeypatch) -> None:
    window = _window_with_style("Stepped")
    try:
        _move_without_wholesale_rebuild(window, monkeypatch)
    finally:
        _close(window)


def test_sloped_move_scopes_the_repaint_and_skips_the_wholesale_rebuild(monkeypatch) -> None:
    window = _window_with_style("Sloped")
    try:
        _move_without_wholesale_rebuild(window, monkeypatch)
    finally:
        _close(window)


# --- Batch D's D6: scoped undo/redo of a fields_only edit -------------------
#
# D5's own tests above pin the FORWARD edit's scoped path. Undo/redo goes
# through a different method (_move_history()), which D6 gives its own
# UnitSplice-building branch -- unpinned by anything above, since none of it
# calls window.undo()/redo().


def _move_then_undo_redo_without_wholesale_rebuild(window, monkeypatch) -> None:
    (entry,) = _select(window, _REF_VILLAGER_P1)
    window.units_panel.show_unit(entry)
    old_x = entry.unit.x

    window.units_panel.unit_field_editors["x"].setValue(old_x + 5)
    moved_x = entry.unit.x
    assert moved_x != old_x

    calls = []
    real = render.sprite_draws_by_anchor
    monkeypatch.setattr(render, "sprite_draws_by_anchor", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    window.undo()
    assert entry.unit.x == old_x
    window.redo()
    assert entry.unit.x == moved_x
    assert calls == [], "undo/redo of a fields_only edit should not pay for a wholesale sprite-source rebuild"


def test_stepped_undo_redo_of_a_move_scopes_the_repaint(monkeypatch) -> None:
    window = _window_with_style("Stepped")
    try:
        _move_then_undo_redo_without_wholesale_rebuild(window, monkeypatch)
    finally:
        _close(window)


def test_sloped_undo_redo_of_a_move_scopes_the_repaint(monkeypatch) -> None:
    window = _window_with_style("Sloped")
    try:
        _move_then_undo_redo_without_wholesale_rebuild(window, monkeypatch)
    finally:
        _close(window)


def test_undo_of_a_delete_still_takes_the_wholesale_path() -> None:
    """A membership edit (Delete) never gets fields_only=True, so its own
    UnitDiffRecord carries no unit_field_entries -- undo must still fall back
    to _after_unit_mutation(None) rather than crash trying to build splices
    from a whole-list record."""
    window = _window_with_style("Stepped")
    try:
        (entry,) = _select(window, _REF_VILLAGER_P1)
        window.on_unit_delete(Qt.NoModifier)
        assert window.map_view._unit_index.entry_for_key((entry.player_id, entry.unit.reference_id)) is None

        window.undo()
        assert window.map_view._unit_index.entry_for_key((entry.player_id, entry.unit.reference_id)) is not None
    finally:
        _close(window)


# --- stacked units: badges and repeat-click cycling ------------------------


def _make_stack(window, count: int = 3) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Places `count` villagers at one identical point on an empty tile, in
    one wholesale edit. Returns (tile, keys top-first)."""
    tile = _empty_tile(window)
    model = window._ensure_unit_edits()
    units = []
    with window._unit_edit(model, "Stack", [1]):
        units.extend(model.add(1, _PLACE_CONST, tile[0] + 0.5, tile[1] + 0.5) for _ in range(count))
    keys = [(1, u.reference_id) for u in reversed(units)]
    return tile, keys


def test_stack_groups_follow_the_index_and_badges_show_the_count() -> None:
    window = _window()
    try:
        tile, keys = _make_stack(window)
        members = window.map_view.stack_group_at(tile)
        assert [(m.player_id, m.unit.reference_id) for m in members] == keys
        item = window.map_view._stack_badge_item
        assert "3" in item.badge_texts()
        assert item.isVisible()
    finally:
        _close(window)


def test_repeat_clicks_walk_the_stack_and_wrap() -> None:
    window = _window()
    try:
        tile, keys = _make_stack(window)
        pos = _pos_for_tile(window, *tile)
        first = window.map_view.pick_unit_at(pos)

        selected = [window.on_click_select(pos, Qt.NoModifier) for _ in range(4)]

        assert selected[0] == (first.player_id, first.unit.reference_id) == keys[0]
        assert selected == [keys[0], keys[1], keys[2], keys[0]]
        assert window._selection == [keys[0]]
        assert window._stack_cycle[0] == tile
    finally:
        _close(window)


def test_a_modified_click_resets_the_cycle() -> None:
    window = _window()
    try:
        tile, keys = _make_stack(window)
        pos = _pos_for_tile(window, *tile)
        window.on_click_select(pos, Qt.NoModifier)
        window.on_click_select(pos, Qt.NoModifier)
        window.on_click_select(pos, Qt.ControlModifier)
        assert window._stack_cycle is None
        window._selection = []
        assert window.on_click_select(pos, Qt.NoModifier) == keys[0]
    finally:
        _close(window)


def test_a_selection_changed_elsewhere_restarts_the_cycle() -> None:
    window = _window()
    try:
        tile, keys = _make_stack(window)
        pos = _pos_for_tile(window, *tile)
        window.on_click_select(pos, Qt.NoModifier)
        window.on_click_select(pos, Qt.NoModifier)
        window.on_marquee_select([keys[2]], Qt.NoModifier)
        assert window.on_click_select(pos, Qt.NoModifier) == keys[0]
    finally:
        _close(window)


def test_a_click_on_an_unstacked_unit_is_unchanged() -> None:
    window = _window()
    try:
        entry = next(
            e for e in window.map_view._unit_index.entries
            if window.map_view.stack_group_at((e.own_x, e.own_y)) is None
            and e.unit.unit_const not in render.BUILDING_TILE_SPANS
        )
        pos = _pos_for_tile(window, entry.own_x, entry.own_y)
        picked = window.map_view.pick_unit_at(pos)
        key = (picked.player_id, picked.unit.reference_id)
        assert window.on_click_select(pos, Qt.NoModifier) == key
        assert window.on_click_select(pos, Qt.NoModifier) == key
    finally:
        _close(window)


def _press_release(map_view, viewport_pos) -> None:
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent

    for kind, button, buttons in (
        (QEvent.MouseButtonPress, Qt.LeftButton, Qt.LeftButton),
        (QEvent.MouseButtonRelease, Qt.LeftButton, Qt.NoButton),
    ):
        event = QMouseEvent(kind, viewport_pos, button, buttons, Qt.NoModifier)
        if kind == QEvent.MouseButtonPress:
            map_view.mousePressEvent(event)
        else:
            map_view.mouseReleaseEvent(event)


def test_after_cycling_the_drag_key_names_the_selected_unit_not_the_top_one() -> None:
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent

    window = _window()
    try:
        tile, keys = _make_stack(window)
        map_view = window.map_view
        viewport_pos = conftest.viewport_pos(map_view, *tile)
        _press_release(map_view, viewport_pos)
        map_view.mousePressEvent(
            QMouseEvent(QEvent.MouseButtonPress, viewport_pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        )
        assert window._selection == [keys[1]]
        assert map_view._unit_drag_key == keys[1]
    finally:
        _close(window)


def test_a_units_mode_press_picks_exactly_once(monkeypatch) -> None:
    """mousePressEvent used to call pick_unit_at() and then hand the same
    scene pos to on_click_select(), which picks again through
    pick_unit_cover_at(). Two full unit picks per click, on a path that
    already costs a plane lookup (Sloped) or a screen_to_tile (Stepped).

    Counted at unit_pick.pick_unit_cover, the single function both routes
    bottom out in, so this stays honest if either wrapper is renamed. Both
    the hit and the miss are covered: the miss is what decides whether the
    press arms a marquee, and it used to read `entry`, not the callback's
    return."""
    from PyQt5.QtCore import QEvent, QPointF
    from PyQt5.QtGui import QMouseEvent

    from descape import unit_pick

    window = _window()
    try:
        tile, _keys = _make_stack(window)
        map_view = window.map_view
        calls = []
        real = unit_pick.pick_unit_cover
        monkeypatch.setattr(
            unit_pick, "pick_unit_cover", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        )

        hit = conftest.viewport_pos(map_view, *tile)
        map_view.mousePressEvent(
            QMouseEvent(QEvent.MouseButtonPress, hit, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        )
        assert len(calls) == 1, f"a press on a unit picked {len(calls)} times, expected 1"
        assert map_view._unit_drag_key is not None
        assert map_view._marquee_start_pos is None

        calls.clear()
        # Far off the map, so the pick genuinely misses rather than landing
        # on some other unit.
        miss = QPointF(-500.0, -500.0)
        map_view.mousePressEvent(
            QMouseEvent(QEvent.MouseButtonPress, miss, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        )
        assert len(calls) == 1, f"a press on empty ground picked {len(calls)} times, expected 1"
        assert map_view._unit_drag_key is None
        assert map_view._marquee_start_pos is not None
    finally:
        _close(window)


def test_a_double_click_press_still_cycles() -> None:
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent

    window = _window()
    try:
        tile, keys = _make_stack(window)
        map_view = window.map_view
        viewport_pos = conftest.viewport_pos(map_view, *tile)
        _press_release(map_view, viewport_pos)
        map_view.mouseDoubleClickEvent(
            QMouseEvent(QEvent.MouseButtonDblClick, viewport_pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        )
        assert window._selection == [keys[1]]
    finally:
        _close(window)


def test_deleting_a_stack_member_updates_the_badge_and_restarts_the_cycle() -> None:
    window = _window()
    try:
        tile, keys = _make_stack(window)
        pos = _pos_for_tile(window, *tile)
        window.on_click_select(pos, Qt.NoModifier)
        window.on_click_select(pos, Qt.NoModifier)
        window.on_unit_delete(Qt.NoModifier)
        assert window._stack_cycle is None
        assert len(window.map_view.stack_group_at(tile)) == 2
        assert window.on_click_select(pos, Qt.NoModifier) == keys[0]
    finally:
        _close(window)


def test_the_badge_toggle_hides_the_layer_and_persists() -> None:
    from descape import settings

    window = _window()
    try:
        _make_stack(window)
        item = window.map_view._stack_badge_item
        assert item.isVisible()
        window.stack_badges_action.setChecked(False)
        assert not item.isVisible()
        assert settings.get_stack_badges() is False
        window.stack_badges_action.setChecked(True)
        assert item.isVisible()
    finally:
        _close(window)


def test_badges_only_show_in_units_mode() -> None:
    window = _window()
    try:
        _make_stack(window)
        item = window.map_view._stack_badge_item
        window.mode_combo.setCurrentText("View")
        assert not item.isVisible()
        window.mode_combo.setCurrentText("Units")
        assert window.map_view._stack_badge_item.isVisible()
    finally:
        _close(window)


# --- Convert brush: live reassignment during the drag -----------------------


def _convert_setup(window, count: int = 2):
    """Destination Player 2, brush size 1, and `count` player-1 units on
    distinct tiles whose tile holds nothing but player-1 units."""
    window.units_panel.select_owner(2)
    window.brush_size_spin.setValue(1)
    index = window.map_view._unit_index
    picked = []
    seen_tiles = set()
    for entry in index.entries:
        tile = (entry.own_x, entry.own_y)
        if entry.player_id != 1 or tile in seen_tiles:
            continue
        if entry.unit.unit_const in render.BUILDING_TILE_SPANS:
            continue
        bucket = [index.entries[o] for o in index.by_tile[tile]]
        if all(e.player_id == 1 for e in bucket):
            seen_tiles.add(tile)
            picked.append(entry)
        if len(picked) == count:
            break
    assert len(picked) == count
    return picked


def test_a_convert_stroke_reassigns_live_and_records_one_undo_step() -> None:
    window = _window()
    try:
        entries = _convert_setup(window)
        before_records = len(window.edit_history.records)
        units = [e.unit for e in entries]

        window._begin_convert_stroke()
        window._convert_stroke_tile(entries[0].own_x, entries[0].own_y)
        # Mutated before mouse-up, not batched to it.
        assert units[0] in window.scenario.unit_manager.units[2]
        window._convert_stroke_tile(entries[1].own_x, entries[1].own_y)
        window._end_convert_stroke()

        assert len(window.edit_history.records) == before_records + 1
        assert all(u in window.scenario.unit_manager.units[2] for u in units)
        assert window.unit_edits._pending is None

        window.undo()
        assert all(u in window.scenario.unit_manager.units[1] for u in units)
        assert len([u for u in window.scenario.unit_manager.units[2] if u in units]) == 0
    finally:
        _close(window)


def test_convert_pixels_change_before_the_stroke_ends() -> None:
    window = _window()
    try:
        (entry,) = _convert_setup(window, count=1)
        cx, cy = _chunk_for_tile(window, entry.own_x, entry.own_y)
        before = window._cache.get_chunk(0, cx, cy).copy()

        window._begin_convert_stroke()
        window._convert_stroke_tile(entry.own_x, entry.own_y)
        assert window._convert_refresh_timer.isActive()
        window._convert_refresh_timer.stop()
        window._flush_convert_refresh()
        mid = window._cache.get_chunk(0, cx, cy).copy()
        assert window._convert_model is not None  # still mid-stroke
        window._end_convert_stroke()

        assert not np.array_equal(before, mid)
    finally:
        _close(window)


def test_convert_coalesces_the_repaint_across_touched_tiles(monkeypatch) -> None:
    window = _window()
    try:
        entries = _convert_setup(window, count=2)
        calls = []
        original = window._after_unit_mutation

        def spy(*args, **kwargs):
            calls.append(kwargs.get("defer_index", False))
            return original(*args, **kwargs)

        monkeypatch.setattr(window, "_after_unit_mutation", spy)
        window._begin_convert_stroke()
        for entry in entries:
            window._convert_stroke_tile(entry.own_x, entry.own_y)
        assert calls == []  # no per-tile refresh; the timer hasn't fired
        window._end_convert_stroke()
        assert calls == [False]
        assert not window._convert_refresh_timer.isActive()
    finally:
        _close(window)


def test_a_convert_stroke_that_touches_nothing_records_nothing() -> None:
    window = _window()
    try:
        before_records = len(window.edit_history.records)
        tile = _empty_tile(window)
        window.units_panel.select_owner(2)
        window._begin_convert_stroke()
        window._convert_stroke_tile(*tile)
        window._end_convert_stroke()
        assert len(window.edit_history.records) == before_records
        assert window.unit_edits._pending is None
    finally:
        _close(window)


def test_a_stroke_left_open_is_aborted_by_the_next_one() -> None:
    window = _window()
    try:
        (entry,) = _convert_setup(window, count=1)
        window._begin_convert_stroke()
        window._begin_convert_stroke()
        window._convert_stroke_tile(entry.own_x, entry.own_y)
        window._end_convert_stroke()
        assert window.unit_edits._pending is None
        assert entry.unit in window.scenario.unit_manager.units[2]
    finally:
        _close(window)


def test_convert_flushes_splices_and_the_stroke_end_rebuilds_the_pick_index(monkeypatch) -> None:
    """Mid-stroke flushes splice the converted units rather than taking the
    wholesale path, and the stroke end still rebuilds the index Convert left
    stale all stroke, so picks report the new owner."""
    window = _window()
    try:
        entries = _convert_setup(window, count=2)
        calls = []
        original = window._after_unit_mutation

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(window, "_after_unit_mutation", spy)
        window._begin_convert_stroke()
        window._convert_stroke_tile(entries[0].own_x, entries[0].own_y)
        window._convert_refresh_timer.stop()
        window._flush_convert_refresh()
        window._convert_stroke_tile(entries[1].own_x, entries[1].own_y)
        window._end_convert_stroke()

        assert [c.get("defer_index", False) for c in calls] == [True, False]
        flushed, final = calls[0]["changed"], calls[1]["changed"]
        # A touched tile can hold more than the picked entry (a building over it).
        flushed_ids, final_ids = {id(s.unit) for s in flushed}, {id(s.unit) for s in final}
        assert id(entries[0].unit) in flushed_ids and id(entries[1].unit) in final_ids
        assert not flushed_ids & final_ids, "a unit was spliced twice"
        for splice in (*flushed, *final):
            assert (splice.player_id, splice.old_player_id) == (2, 1)
            assert window.scenario.unit_manager.units[2][splice.index] is splice.unit
        assert calls[1].get("rebuild_index") is True

        index = window.map_view._unit_index
        converted = {id(e.unit) for e in entries}
        owners = {entry.player_id for entry in index.entries if id(entry.unit) in converted}
        assert owners == {2}
    finally:
        _close(window)


# --- D2's free-placement toggle (free placement, Stage 3) -------------------


def _pos_in_tile(window, tile_x: int, tile_y: int, fx: float, fy: float) -> QPointF:
    """A scene position at a chosen sub-tile fraction. Flat only, which is
    what this module forces -- one continuous tile unit is tile_px pixels."""
    tp = window.map_view._tile_pixels
    return QPointF((tile_x + fx) * tp, (tile_y + fy) * tp)


def _place_at(window, pos, modifiers=Qt.NoModifier):
    window.units_panel.select_object(_PLACE_CONST)
    window.units_panel.select_owner(1)
    window.place_unit_action.setChecked(True)
    window.on_unit_place(pos, modifiers)
    return window.map_view._unit_index.entry_for_key(window._selection[0])


def test_the_free_placement_checkbox_shows_only_for_place_unit() -> None:
    """Asserted on the captured QAction handle, never on the widget's own
    isVisible(): offscreen, a never-shown widget reads False unconditionally."""
    window = _window()
    try:
        window._on_tool_selected("place_unit")
        assert window.free_place_param_action.isVisible()
        assert window.free_place_check.isEnabled()
        assert window.tool_param_separator_action.isVisible()

        window._on_tool_selected("convert")
        assert not window.free_place_param_action.isVisible()
        assert not window.free_place_check.isEnabled()
    finally:
        _close(window)


def test_the_toggle_is_session_only_and_starts_off() -> None:
    """Tool options are deliberately not persisted, so there is no settings
    key behind this and a fresh window must start snapped."""
    window = _window()
    try:
        assert not window.free_place_check.isChecked()
        assert not any("free_place" in key for key in dir(__import__("descape.settings", fromlist=["x"]))), (
            "free placement grew a settings accessor; it is supposed to be session-only"
        )
    finally:
        _close(window)


def test_placement_snaps_by_default_and_floats_with_the_toggle() -> None:
    window = _window()
    try:
        tx, ty = _empty_tile(window)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.15, 0.85))
        assert (entry.unit.x, entry.unit.y) == (tx + 0.5, ty + 0.5)

        window.free_place_check.setChecked(True)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.15, 0.85))
        assert (entry.unit.x, entry.unit.y) != (tx + 0.5, ty + 0.5)
        # Under the cursor, to within the pixel the click itself quantizes to.
        tp = window.map_view._tile_pixels
        assert abs(entry.unit.x - (tx + 0.15)) <= 1 / tp
        assert abs(entry.unit.y - (ty + 0.85)) <= 1 / tp
    finally:
        _close(window)


def test_held_alt_places_free_for_one_click_without_changing_the_toggle() -> None:
    window = _window()
    try:
        tx, ty = _empty_tile(window)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.15, 0.85), Qt.AltModifier)
        assert (entry.unit.x, entry.unit.y) != (tx + 0.5, ty + 0.5)
        assert not window.free_place_check.isChecked(), "the one-off modifier flipped the toggle"

        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.15, 0.85))
        assert (entry.unit.x, entry.unit.y) == (tx + 0.5, ty + 0.5)
    finally:
        _close(window)


def test_moving_a_unit_takes_the_same_branch_as_placing_one() -> None:
    """The choice lives in _placement_point, so Move and Place cannot
    disagree -- this is what pins that they really share it."""
    window = _window()
    try:
        tx, ty = _empty_tile(window)
        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.5, 0.5))
        key = window._selection[0]

        window.free_place_check.setChecked(True)
        window.on_unit_move(key, _pos_in_tile(window, tx + 1, ty + 1, 0.2, 0.7), Qt.NoModifier)
        moved = window.map_view._unit_index.entry_for_key(key)
        assert moved is not None
        assert (moved.unit.x, moved.unit.y) != (tx + 1.5, ty + 1.5)

        window.free_place_check.setChecked(False)
        window.on_unit_move(key, _pos_in_tile(window, tx + 2, ty + 2, 0.2, 0.7), Qt.NoModifier)
        moved = window.map_view._unit_index.entry_for_key(key)
        assert (moved.unit.x, moved.unit.y) == (tx + 2.5, ty + 2.5)
        assert entry is not None
    finally:
        _close(window)


def test_a_free_placed_unit_paints_where_it_stands() -> None:
    """The end-to-end check the whole feature exists for: model coordinates
    reaching the canvas. A model-only assertion is exactly what b1's own
    postmortem records as having missed the last bug here."""
    window = _window()
    try:
        tx, ty = _empty_tile(window)
        window.free_place_check.setChecked(True)
        _place_at(window, _pos_in_tile(window, tx, ty, 0.1, 0.1))
        cx, cy = _chunk_for_tile(window, tx, ty)
        low = np.array(window._cache.get_chunk(0, cx, cy), copy=True)

        window.undo()
        window.free_place_check.setChecked(False)
        _place_at(window, _pos_in_tile(window, tx, ty, 0.1, 0.1))
        snapped = np.array(window._cache.get_chunk(0, cx, cy), copy=True)

        assert not np.array_equal(low, snapped), (
            "a free-placed unit painted the same pixels as a snapped one"
        )
    finally:
        _close(window)


def test_a_refused_inverse_snaps_and_says_so(monkeypatch) -> None:
    """The screen -> map-point inverse can legitimately have no answer for a
    pixel: a degenerate denominator, or a round-trip that misses by more than
    a pixel, both reachable on a steep Sloped ramp. Snapping there is the
    right behaviour, but a SILENT snap is not -- a tester cannot tell it from
    a checkbox that never applied or an Alt the window manager ate.

    Driven by forcing _pick_map_point to refuse rather than by constructing a
    degenerate ramp: this module forces Flat on purpose, and the geometry's
    own degenerate case is pinned in tests/test_screen_to_map_point.py. What
    is untested without this is the viewer's REACTION to a refusal."""
    from descape.viewer import FREE_PLACE_FALLBACK_MESSAGE

    window = _window()
    try:
        tx, ty = _empty_tile(window)
        window.free_place_check.setChecked(True)
        monkeypatch.setattr(window.map_view, "_pick_map_point", lambda pos: None)

        entry = _place_at(window, _pos_in_tile(window, tx, ty, 0.15, 0.85))

        assert (entry.unit.x, entry.unit.y) == (tx + 0.5, ty + 0.5)
        assert FREE_PLACE_FALLBACK_MESSAGE in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_successful_free_placement_stays_quiet() -> None:
    """The other half: the fallback line must not fire on every free click,
    or it stops meaning anything."""
    from descape.viewer import FREE_PLACE_FALLBACK_MESSAGE

    window = _window()
    try:
        tx, ty = _empty_tile(window)
        window.free_place_check.setChecked(True)
        _place_at(window, _pos_in_tile(window, tx, ty, 0.15, 0.85))
        assert FREE_PLACE_FALLBACK_MESSAGE not in window.status_log.toPlainText()
    finally:
        _close(window)


# --- GH #75: group and stack move --------------------------------------------

_HOUSE_CONST = 70  # 2x2


def _add_units(window, placements):
    """[(const, x, y)] added for player 1 in one wholesale edit; returns keys."""
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Add", [1]):
        units = [model.add(1, const, x, y) for const, x, y in placements]
    return [(1, u.reference_id) for u in units]


def _free_tiles(window, tiles) -> None:
    occupied = {(int(u.x), int(u.y)) for units in window.scenario.unit_manager.units for u in units}
    assert not occupied & set(tiles), "the fixture already holds a unit on a test tile"


def _select_keys(window, keys) -> None:
    window._selection = list(keys)
    window._refresh_selection_view()


def _positions(window, keys):
    index = window.map_view._unit_index
    return [(index.entry_for_key(k).unit.x, index.entry_for_key(k).unit.y) for k in keys]


def _gesture(window, from_tile, to_tile=None, modifiers=Qt.NoModifier, to_scene=None) -> None:
    """A real press(-move)-release through MapView; no `to_tile` is a plain click."""
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent

    view = window.map_view
    press = conftest.viewport_pos(view, *from_tile)
    if to_scene is not None:
        end = QPointF(view.mapFromScene(to_scene))
    else:
        end = press if to_tile is None else conftest.viewport_pos(view, *to_tile)
    view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, press, Qt.LeftButton, Qt.LeftButton, modifiers))
    if end != press:
        view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton, modifiers))
    view.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton, modifiers))


def test_a_press_on_a_group_member_without_a_drag_collapses_on_release() -> None:
    window = _window()
    try:
        _free_tiles(window, [(62, 62), (64, 62)])
        keys = _add_units(window, [(_PLACE_CONST, 62.5, 62.5), (_PLACE_CONST, 64.5, 62.5)])
        _select_keys(window, keys)
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QMouseEvent

        view = window.map_view
        vp = conftest.viewport_pos(view, 64, 62)
        view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, vp, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        assert window._selection == keys, "the group must survive the press, or a drag can't move it"
        view.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, vp, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
        assert window._selection == [keys[1]]
    finally:
        _close(window)


def test_a_group_drag_moves_every_member_in_one_undo_record() -> None:
    window = _window()
    try:
        _free_tiles(window, [(62, 62), (64, 63), (66, 62)])
        keys = _add_units(
            window, [(_PLACE_CONST, 62.5, 62.5), (_PLACE_CONST, 64.3, 63.8), (_PLACE_CONST, 66.5, 62.5)]
        )
        _select_keys(window, keys)
        before = _positions(window, keys)
        depth = window.edit_history.cursor

        _gesture(window, (62, 62), (65, 64))

        after = _positions(window, keys)
        # Whole-tile (3, 2), and the off-centre member keeps its sub-tile offset.
        assert after == [(x + 3, y + 2) for x, y in before]
        assert window.edit_history.cursor == depth + 1
        assert window._selection == keys, "a group stays selected after the drag"
        window.undo()
        assert _positions(window, keys) == before
        window.redo()
        assert _positions(window, keys) == after
    finally:
        _close(window)


def test_the_edge_clamp_keeps_the_shape_even_on_an_off_map_release() -> None:
    window = _window()
    try:
        w = window.scenario.map_manager.map_width
        _free_tiles(window, [(w - 6, 60), (w - 4, 61)])
        keys = _add_units(window, [(_PLACE_CONST, w - 5.5, 60.5), (_PLACE_CONST, w - 3.5, 61.5)])
        _select_keys(window, keys)
        before = _positions(window, keys)

        tp = window.map_view._tile_pixels
        # Released well past the right edge: clamped, not cancelled.
        _gesture(window, (w - 6, 60), to_scene=QPointF((w + 10) * tp, 60.5 * tp))

        after = _positions(window, keys)
        assert after == [(x + 3, y) for x, y in before], "the outer member stops on the last column"
        assert "stopped at the map edge" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_fully_clamped_group_drag_writes_nothing() -> None:
    window = _window()
    try:
        w = window.scenario.map_manager.map_width
        _free_tiles(window, [(w - 3, 60), (w - 1, 60)])
        keys = _add_units(window, [(_PLACE_CONST, w - 2.5, 60.5), (_PLACE_CONST, w - 0.5, 60.5)])
        _select_keys(window, keys)
        before = _positions(window, keys)
        depth = window.edit_history.cursor

        _gesture(window, (w - 3, 60), (w - 1, 60))

        assert _positions(window, keys) == before
        assert window.edit_history.cursor == depth
    finally:
        _close(window)


def test_grabbing_a_houses_lower_tile_moves_the_group_exactly_one_tile() -> None:
    """A 2x2 house sits at tile + 1.0, so int(house.x) is its UPPER tile. The
    delta runs from the grabbed cover tile, not from that."""
    window = _window()
    try:
        _free_tiles(window, [(62, 62), (63, 62), (62, 63), (63, 63), (66, 62)])
        keys = _add_units(window, [(_HOUSE_CONST, 63.0, 63.0), (_PLACE_CONST, 66.5, 62.5)])
        _select_keys(window, keys)
        before = _positions(window, keys)
        # Zoomed in, so a one-tile drag clears UNIT_DRAG_THRESHOLD_PX.
        window.map_view.scale(4.0, 4.0)
        window.map_view.centerOn(QPointF(63 * window.map_view._tile_pixels, 63 * window.map_view._tile_pixels))

        _gesture(window, (62, 62), (63, 62))

        assert _positions(window, keys) == [(x + 1, y) for x, y in before]
    finally:
        _close(window)


def test_a_free_group_drag_keeps_every_offset(monkeypatch) -> None:
    window = _window()
    try:
        _free_tiles(window, [(62, 62), (64, 63)])
        keys = _add_units(window, [(_PLACE_CONST, 62.5, 62.5), (_PLACE_CONST, 64.3, 63.8)])
        _select_keys(window, keys)
        before = _positions(window, keys)
        monkeypatch.setattr(window.map_view, "_pick_map_point", lambda pos: (70.25, 66.75))

        _gesture(window, (62, 62), (70, 66), modifiers=Qt.AltModifier)

        dx, dy = 70.25 - before[0][0], 66.75 - before[0][1]
        after = _positions(window, keys)
        assert after[0] == (70.25, 66.75)
        assert after[1] == pytest.approx((before[1][0] + dx, before[1][1] + dy), abs=1e-4)
    finally:
        _close(window)


def test_a_click_on_a_stacked_group_member_carries_the_cycle_on_after_the_collapse() -> None:
    window = _window()
    try:
        tile, stack = _make_stack(window)
        other = (tile[0] + 3, tile[1])
        _free_tiles(window, [other])
        (loner,) = _add_units(window, [(_PLACE_CONST, other[0] + 0.5, other[1] + 0.5)])
        _select_keys(window, [stack[0], loner])

        _gesture(window, tile)
        assert window._selection == [stack[0]]
        assert "1 of 3 stacked here" in window.status_log.toPlainText()

        _gesture(window, tile)
        assert window._selection == [stack[1]]
    finally:
        _close(window)


def test_a_plain_drag_on_a_stack_still_pulls_one_unit_off() -> None:
    window = _window()
    try:
        tile, stack = _make_stack(window)
        before = _positions(window, stack)
        _gesture(window, tile, (tile[0] + 2, tile[1]))
        after = _positions(window, stack)
        assert sum(a != b for a, b in zip(after, before, strict=True)) == 1
    finally:
        _close(window)


def test_select_whole_stack_widens_the_selection_and_a_drag_moves_all_of_it() -> None:
    window = _window()
    try:
        tile, stack = _make_stack(window)
        other = (tile[0] + 3, tile[1])
        _free_tiles(window, [other])
        (loner,) = _add_units(window, [(_PLACE_CONST, other[0] + 0.5, other[1] + 0.5)])
        _select_keys(window, [loner, stack[1]])

        window.select_stack_action.trigger()
        assert window._selection == [loner, stack[1], stack[0], stack[2]]
        assert window._stack_cycle is None
        assert "4 units selected (stack)" in window.status_log.toPlainText()

        before = _positions(window, stack)
        _gesture(window, tile, (tile[0], tile[1] + 2))
        assert _positions(window, stack) == [(x, y + 2) for x, y in before]
    finally:
        _close(window)


def test_select_whole_stack_says_so_when_nothing_selected_is_stacked() -> None:
    window = _window()
    try:
        tile, _stack = _make_stack(window)
        other = (tile[0] + 3, tile[1])
        _free_tiles(window, [other])
        (loner,) = _add_units(window, [(_PLACE_CONST, other[0] + 0.5, other[1] + 0.5)])
        _select_keys(window, [loner])
        window.on_select_whole_stack()
        assert window._selection == [loner]
        assert "no selected unit is stacked" in window.status_log.toPlainText()
        _select_keys(window, [])
        window.on_select_whole_stack()
        assert "nothing is selected" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_clamp_group_delta_uses_each_members_own_span() -> None:
    from descape.viewer import clamp_group_delta
    from testkit import fakes

    house = fakes.SyntheticUnit(x=3.0, y=10.0, unit_const=_HOUSE_CONST)  # covers x 2..3
    villager = fakes.SyntheticUnit(x=15.5, y=10.5, unit_const=_PLACE_CONST)
    assert clamp_group_delta([house, villager], -5, 0, 20, 20, whole_tiles=True) == (-2, 0, True)
    assert clamp_group_delta([house, villager], 9, 0, 20, 20, whole_tiles=True) == (4, 0, True)
    assert clamp_group_delta([house, villager], 1, 1, 20, 20, whole_tiles=True) == (1, 1, False)
    # Free: the house's anchor may reach span/2 from the edge, the villager just short of it.
    dx, _dy, clamped = clamp_group_delta([house, villager], -5.0, 0.0, 20, 20, whole_tiles=False)
    assert (dx, clamped) == (-2.0, True)
    dx, _dy, _ = clamp_group_delta([villager], 9.0, 0.0, 20, 20, whole_tiles=False)
    assert int(villager.x + dx) == 19


def test_an_off_map_member_does_not_limit_the_clamp() -> None:
    from descape.viewer import clamp_group_delta
    from testkit import fakes

    stray = fakes.SyntheticUnit(x=-2.0, y=5.5, unit_const=_PLACE_CONST)
    villager = fakes.SyntheticUnit(x=5.5, y=5.5, unit_const=_PLACE_CONST)
    assert clamp_group_delta([stray, villager], -4, 0, 20, 20, whole_tiles=True) == (-4, 0, False)


# --- Edit > Scatter Units in Region… ----------------------------------------
#
# The dialog itself is tested headless in test_scatter_dialog.py; these drive
# the viewer around it, with exec_() monkeypatched so no modal box blocks the
# offscreen run.

_FISH_CONST = 456  # Fish (Salmon), the motivating GAIA case
_WATER_TERRAIN = 1  # TerrainId.WATER_SHALLOW
_POND = (60, 60, 70, 70)


def _paint_water(window, region) -> None:
    """The fixture is all grass, so a Water-only scatter needs a pond made
    first. Straight onto mm.terrain, the same flat index scatter_dialog
    reads."""
    mm = window.scenario.map_manager
    tx0, ty0, tx1, ty1 = region
    for y in range(ty0, ty1):
        for x in range(tx0, tx1):
            mm.terrain[y * mm.map_width + x].terrain_id = _WATER_TERRAIN


def _arm_scatter(window, monkeypatch, configure=None, accept=True):
    """Selects a region, picks the fish as GAIA, and auto-answers the dialog."""
    from PyQt5.QtWidgets import QDialog

    from descape import viewer as viewer_mod

    window.units_panel.select_object(_FISH_CONST)
    window.units_panel.select_owner(0)
    window.on_region_selected(_POND)

    def fake_exec(dialog):
        if configure is not None:
            configure(dialog)
        return QDialog.Accepted if accept else QDialog.Rejected

    monkeypatch.setattr(viewer_mod.ScatterDialog, "exec_", fake_exec)


def _water_40(dialog) -> None:
    dialog.water_radio.setChecked(True)
    dialog.count_spin.setValue(40)
    dialog.seed_spin.setValue(20260921)


def _gaia_fish(window) -> list:
    return [u for u in window.scenario.unit_manager.units[0] if u.unit_const == _FISH_CONST]


def test_the_scatter_action_is_gated_on_the_region_alone(monkeypatch) -> None:
    window = _window()
    try:
        assert window._region is None
        assert not window.scatter_action.isEnabled()
        window.on_region_selected(_POND)
        assert window.scatter_action.isEnabled()  # Units mode
        window.mode_combo.setCurrentText("Terrain")
        assert window.scatter_action.isEnabled()  # and Terrain mode
        window.deselect()
        assert not window.scatter_action.isEnabled()
    finally:
        _close(window)


def test_the_scatter_action_is_disabled_with_no_map() -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        assert window.scenario is None
        assert not window.scatter_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_scatter_with_no_object_chosen_opens_no_dialog(monkeypatch) -> None:
    window = _window()
    try:
        from PyQt5.QtWidgets import QDialog

        from descape import viewer as viewer_mod

        opened = []

        def record_exec(dialog):
            opened.append(dialog)
            return QDialog.Rejected

        monkeypatch.setattr(viewer_mod.ScatterDialog, "exec_", record_exec)
        window.on_region_selected(_POND)
        assert window.units_panel.selected_object_const() is None
        window.scatter_units_in_region()
        assert opened == []
        assert "choose an object in the Units panel first" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_scatter_places_every_unit_inside_the_region_and_on_water(monkeypatch) -> None:
    window = _window()
    try:
        _paint_water(window, _POND)
        _arm_scatter(window, monkeypatch, _water_40)
        before_records = len(window.edit_history.records)
        before_fish = len(_gaia_fish(window))

        window.scatter_units_in_region()

        fish = _gaia_fish(window)
        assert len(fish) == before_fish + 40
        mm = window.scenario.map_manager
        for unit in fish:
            tx, ty = int(unit.x), int(unit.y)
            assert 60 <= tx < 70 and 60 <= ty < 70
            assert mm.terrain[ty * mm.map_width + tx].terrain_id == _WATER_TERRAIN
            # AGENTS.md's hard rule: rotation is a variant index for most
            # GAIA objects, so scatter never invents one.
            assert unit.rotation == 0.0
            assert unit.initial_animation_frame == 0
        assert len({(int(u.x), int(u.y)) for u in fish}) == len(fish)  # one per tile
        assert len(window._selection) == 40
        assert set(window._selection) == {(0, u.reference_id) for u in fish}
        assert len(window.edit_history.records) == before_records + 1
        assert "Scattered 40 x" in window.status_log.toPlainText()

        window.undo()
        assert len(_gaia_fish(window)) == before_fish
    finally:
        _close(window)


def test_the_same_seed_scatters_to_the_same_coordinates(monkeypatch) -> None:
    window = _window()
    try:
        _paint_water(window, _POND)
        _arm_scatter(window, monkeypatch, _water_40)
        window.scatter_units_in_region()
        first = sorted((u.x, u.y) for u in _gaia_fish(window))
        window.undo()
        window.scatter_units_in_region()
        assert sorted((u.x, u.y) for u in _gaia_fish(window)) == first
    finally:
        _close(window)


def test_cancelling_the_scatter_dialog_records_nothing(monkeypatch) -> None:
    window = _window()
    try:
        _paint_water(window, _POND)
        _arm_scatter(window, monkeypatch, _water_40, accept=False)
        before_records = len(window.edit_history.records)
        before_fish = len(_gaia_fish(window))

        window.scatter_units_in_region()

        assert len(window.edit_history.records) == before_records
        assert len(_gaia_fish(window)) == before_fish
        # The dialog runs before _ensure_unit_edits(), so a cancel never pays
        # for the lazy model build.
        assert window.unit_edits is None
    finally:
        _close(window)


def test_asking_for_more_fish_than_the_pond_holds_says_so(monkeypatch) -> None:
    window = _window()
    try:
        _paint_water(window, (60, 60, 63, 63))  # 9 water tiles inside the region

        def configure(dialog):
            dialog.water_radio.setChecked(True)
            dialog.count_spin.setValue(50)

        _arm_scatter(window, monkeypatch, configure)
        window.scatter_units_in_region()
        assert len(_gaia_fish(window)) == 9
        log = window.status_log.toPlainText()
        assert "Scattered 9 x" in log
        assert "asked for 50" in log
    finally:
        _close(window)


def test_scatter_outside_units_mode_leaves_the_selection_alone(monkeypatch) -> None:
    window = _window()
    try:
        _paint_water(window, _POND)
        _arm_scatter(window, monkeypatch, _water_40)
        window.mode_combo.setCurrentText("Terrain")
        window._selection = []
        window.scatter_units_in_region()
        assert len(_gaia_fish(window)) == 40
        assert window._selection == []
    finally:
        _close(window)


def test_scatter_pixels_appear_and_one_undo_restores_them(monkeypatch) -> None:
    """The cache-invalidation check b1 and b2 both needed: a model-only
    assertion passes while the map keeps painting the pre-edit chunk."""
    window = _window()
    try:
        _arm_scatter(window, monkeypatch, lambda dialog: dialog.count_spin.setValue(40))
        cx, cy = _chunk_for_tile(window, 65, 65)

        before = window._cache.get_chunk(0, cx, cy).copy()
        window.scatter_units_in_region()
        after = window._cache.get_chunk(0, cx, cy).copy()
        assert not np.array_equal(before, after)

        window.undo()
        restored = window._cache.get_chunk(0, cx, cy)
        assert np.array_equal(before, restored)
    finally:
        _close(window)
