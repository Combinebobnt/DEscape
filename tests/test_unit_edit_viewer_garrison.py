"""Units mode: garrisons, from the inspector's block to drag drops, map picks,
unload and paste.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtWidgets import QMessageBox
from unit_edit_viewer_helpers import (
    _HOUSE_CONST,
    _PLACE_CONST,
    _REF_ARCHER_P1,
    _REF_ARCHER_P2,
    _REF_HOUSE,
    _REF_TREE_OAK,
    _REF_TREE_PINE,
    _REF_VILLAGER_P2,
    _V159_FIXTURE,
    _close,
    _empty_tile,
    _entry,
    _fixture_host_and_occupant,
    _hide_garrisoned,
    _live,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _owner_of,
    _place_tower,
    _pos_for_tile,
    _pos_in_tile,
    _rebuilt_canvas,
    _second_tower,
    _select,
    _select_rows,
    _show_garrisoned,
    _tower_with_two_inside,
    _unit_count,
    _unit_layout,
    _whole_canvas,
    _window,
    _window_with_style,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# --- GH #42: a host's garrison follows it ------------------------------


def test_dragging_a_host_carries_its_hidden_garrison(monkeypatch) -> None:
    window = _window()
    try:
        host, occupant = _fixture_host_and_occupant(window)
        _hide_garrisoned(window)
        key = (1, host.reference_id)
        window._selection = [key]
        window._refresh_selection_view()
        records_before = len(window.edit_history.records)

        # The House is 2x2: a snapped move takes the whole number nearest the cursor (GH #206).
        window.on_unit_move(key, _pos_in_tile(window, 40, 44, 0.3, 0.3), Qt.NoModifier)

        assert (host.x, host.y) == (40.0, 44.0)
        assert (occupant.x, occupant.y) == (40.0, 44.0)
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

        window.undo()
        assert (host.x, occupant.x) == (10.5, 10.5)
        assert occupant.y == host.y
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


def test_an_empty_house_has_no_garrison_block() -> None:
    """The game gives a House no places, so only an occupant already in the
    file (the fixture's own House) earns it a block."""
    window = _window()
    try:
        model = window._ensure_unit_edits()
        tile = _empty_tile(window)
        with window._unit_edit(model, "Place house", [1]):
            house = model.add(1, _HOUSE_CONST, tile[0] + 1.0, tile[1] + 1.0)
        window._rebuild_unit_index()
        window._selection = [(1, house.reference_id)]
        window._refresh_selection_view()

        assert window.units_panel.unit_field_labels["reference_id"].text() == str(house.reference_id)
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


def test_a_monastery_shows_a_garrison_block_and_takes_a_relic_but_not_a_monk() -> None:
    """TASK-467: mask 0 like a Market, but the game puts relics inside."""
    window = _window()
    try:
        model = window._ensure_unit_edits()
        tile = _empty_tile(window)
        with window._unit_edit(model, "Place monastery", [1]):
            monastery = model.add(1, 104, tile[0] + 1.5, tile[1] + 1.5)
        window._rebuild_unit_index()
        window._selection = [(1, monastery.reference_id)]
        window._refresh_selection_view()
        panel = window.units_panel
        assert panel.garrison_tree.isVisibleTo(panel)
        assert panel.garrison_add_button.isEnabled()
        entry = window.map_view._unit_index.entry_for_key((1, monastery.reference_id))

        window._garrison_add_const(entry, 125)  # Monk
        assert "cannot go inside" in window.status_log.toPlainText()
        window._garrison_add_const(entry, 285)  # Relic

        inside = [u for u in window.scenario.unit_manager.units[1] if u.garrisoned_in_id == monastery.reference_id]
        assert [u.unit_const for u in inside] == [285]
        assert panel.garrison_tree.topLevelItemCount() == 1
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
        assert window.units_panel.garrison_add_button.toolTip() == "Full: the game gives this one 5 places"
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


# --- GH #115: garrison and unload existing units ------------------------------


def test_garrisoning_an_existing_unit_links_co_locates_and_hides_it_in_one_record() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        _hide_garrisoned(window)
        archer = _live(window, _REF_ARCHER_P1)
        old = (archer.x, archer.y, archer.z)
        records_before = len(window.edit_history.records)

        assert window._garrison_units(_entry(window, 1, tower.reference_id), [_entry(window, 1, _REF_ARCHER_P1)])

        assert archer.garrisoned_in_id == tower.reference_id
        assert (archer.x, archer.y, archer.z) == (tower.x, tower.y, tower.z)
        assert _entry(window, 1, _REF_ARCHER_P1) is None
        assert window.units_panel.garrison_tree.topLevelItemCount() == 1
        assert len(window.edit_history.records) == records_before + 1
        assert window.edit_history.peek_undo().label == "Garrison units"

        window.undo()
        assert archer.garrisoned_in_id == -1
        assert (archer.x, archer.y, archer.z) == old
        assert _entry(window, 1, _REF_ARCHER_P1) is not None
    finally:
        _close(window)


def test_garrisoning_into_a_full_host_is_refused_with_no_change() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        host = _entry(window, 1, tower.reference_id)
        for _ in range(5):
            window._garrison_add_const(host, 4)
        archer = _live(window, _REF_ARCHER_P1)
        records_before = len(window.edit_history.records)

        assert not window._garrison_units(_entry(window, 1, tower.reference_id), [_entry(window, 1, _REF_ARCHER_P1)])

        assert archer.garrisoned_in_id == -1
        assert len(window.edit_history.records) == records_before
        assert "is full (5 places)" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_garrisoning_a_unit_that_holds_a_garrison_is_a_status_line_not_a_raise() -> None:
    """Step 2's no-nesting scope, pre-checked so the model never raises."""
    window = _window()
    try:
        tower = _place_tower(window)
        house = _live(window, _REF_HOUSE)
        records_before = len(window.edit_history.records)

        assert not window._garrison_units(_entry(window, 1, tower.reference_id), [_entry(window, 1, _REF_HOUSE)])

        assert house.garrisoned_in_id == -1
        assert len(window.edit_history.records) == records_before
        assert "holds" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_unload_makes_units_visible_and_pickable_beside_the_host() -> None:
    """Both land on the first in-bounds tile centre off the host's +y edge;
    a Watch Tower is 1x1, so that is the tile below its own."""
    window = _window()
    try:
        tower, occupants = _tower_with_two_inside(window)
        _hide_garrisoned(window)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        assert all(_entry(window, 1, u.reference_id) is None for u in occupants)
        records_before = len(window.edit_history.records)

        window._unload_units(_entry(window, 1, tower.reference_id), [(1, u) for u in occupants])

        expected = (int(tower.x) + 0.5, int(tower.y) + 1.5)
        for unit in occupants:
            assert unit.garrisoned_in_id == -1
            assert (unit.x, unit.y) == expected
            assert _entry(window, 1, unit.reference_id) is not None
        assert window.units_panel.garrison_tree.topLevelItemCount() == 0
        assert len(window.edit_history.records) == records_before + 1
        assert window.edit_history.peek_undo().label == "Unload units"

        window.undo()
        assert all(u.garrisoned_in_id == tower.reference_id for u in occupants)
        assert all((u.x, u.y) == (tower.x, tower.y) for u in occupants)
        assert all(_entry(window, 1, u.reference_id) is None for u in occupants)
    finally:
        _close(window)


def test_unload_with_show_garrisoned_on_moves_the_existing_index_entry() -> None:
    """With the filter on, an occupant is already in the index: unloading it
    is a move there, never a second entry."""
    window = _window()
    try:
        tower, occupants = _tower_with_two_inside(window)
        before = len(window.map_view._unit_index.entries)

        window._unload_units(_entry(window, 1, tower.reference_id), [(1, occupants[0])])

        assert len(window.map_view._unit_index.entries) == before
        moved = _entry(window, 1, occupants[0].reference_id)
        assert (moved.own_x, moved.own_y) == (int(tower.x), int(tower.y) + 1)
    finally:
        _close(window)


def _index_keys(index) -> list[tuple[int, int]]:
    return [(e.player_id, e.unit.reference_id) for e in index.entries]


def test_unload_with_show_garrisoned_off_keeps_the_index_in_paint_order() -> None:
    """An appended entry would outrank every later player's units on a
    shared tile, though paint order puts the unloaded unit before them."""
    from descape import unit_pick

    window = _window()
    try:
        tower, occupants = _tower_with_two_inside(window)
        _hide_garrisoned(window)
        later = [e for e in window.map_view._unit_index.entries if e.player_id > 1]
        assert later  # else an append and a rebuild agree by accident

        window._unload_units(_entry(window, 1, tower.reference_id), [(1, u) for u in occupants])

        rebuilt = unit_pick.build_index(window.scenario, window._unit_filter)
        assert _index_keys(window.map_view._unit_index) == _index_keys(rebuilt)
    finally:
        _close(window)


def _ranked_keys(index) -> list[tuple[int, int]]:
    return [(e.player_id, e.unit.reference_id) for e in sorted(index.entries, key=lambda e: e.rank)]


@pytest.mark.parametrize("op", ["place", "paste_inside", "garrison_add"])
def test_an_appending_edit_keeps_the_index_in_paint_order(op) -> None:
    """The same equality, in rank order, for the three edits that append
    (a rebuild per Place would cost the hot path), with Show Garrisoned on
    and a later player's units already in the index."""
    from descape import unit_pick

    window = _window()
    try:
        if op == "place":
            window.unrestricted_check.setChecked(True)  # (10, 10) is under the fixture's House
            window.units_panel.select_object(_PLACE_CONST)
            window.units_panel.select_owner(1)
            window.place_unit_action.setChecked(True)
            window.on_unit_place(_pos_for_tile(window, 10, 10), Qt.NoModifier)
        elif op == "paste_inside":
            tower, occupants = _tower_with_two_inside(window)
            window._rebuild_unit_index()  # so only the paste can drift
            window._selection = [(1, tower.reference_id)]
            window._refresh_selection_view()
            window._on_garrison_copy({u.reference_id for u in occupants})
            window._on_garrison_paste()
            inside = [u for u in window.scenario.unit_manager.units[1] if u.garrisoned_in_id == tower.reference_id]
            if len(inside) != 4:
                pytest.fail(f"premise: paste-inside left {len(inside)} occupants, not 4")
        else:
            tower = _place_tower(window)
            window._garrison_add_const(_entry(window, 1, tower.reference_id), 4)
        if not any(e.player_id > 1 for e in window.map_view._unit_index.entries):
            pytest.fail("premise: no later player's unit in the index")

        rebuilt = unit_pick.build_index(window.scenario, window._unit_filter)
        assert _ranked_keys(window.map_view._unit_index) == _ranked_keys(rebuilt)
    finally:
        _close(window)


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_garrison_and_unload_repaint_matches_a_fresh_render(style: str) -> None:
    """Both edits change a unit's visibility under the default filter. The
    scoped repaint must land where a from-scratch render does."""
    window = _window() if style == "Flat" else _window_with_style(style)
    try:
        tower = _place_tower(window)
        _hide_garrisoned(window)
        host = _entry(window, 1, tower.reference_id)
        before = _whole_canvas(window)

        assert window._garrison_units(host, [_entry(window, 2, _REF_ARCHER_P2)])
        garrisoned = _whole_canvas(window)
        assert not np.array_equal(before, garrisoned), "the garrisoned archer should leave its tile"
        assert np.array_equal(garrisoned, _rebuilt_canvas(window))

        archer = _live(window, _REF_ARCHER_P2)
        window._unload_units(_entry(window, 1, tower.reference_id), [(2, archer)])
        unloaded = _whole_canvas(window)
        assert not np.array_equal(garrisoned, unloaded), "the unloaded archer should appear"
        assert np.array_equal(unloaded, _rebuilt_canvas(window))
    finally:
        _close(window)


# --- GH #115 step 3: drop onto a host ----------------------------------------


def _drag_to_tower(window, keys, tower, modifiers=Qt.NoModifier) -> None:
    """Selects `keys` and releases a drag of the first one over the tower's tile."""
    window._selection = list(keys)
    window._refresh_selection_view()
    window.on_unit_move(keys[0], _pos_for_tile(window, int(tower.x), int(tower.y)), modifiers)


def test_dragging_onto_an_accepting_host_garrisons_and_one_undo_restores() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        archer = _live(window, _REF_ARCHER_P2)
        old = (archer.x, archer.y)
        records_before = len(window.edit_history.records)

        _drag_to_tower(window, [(2, _REF_ARCHER_P2)], tower)

        assert archer.garrisoned_in_id == tower.reference_id
        assert (archer.x, archer.y) == (tower.x, tower.y)
        assert len(window.edit_history.records) == records_before + 1

        window.undo()
        assert archer.garrisoned_in_id == -1
        assert (archer.x, archer.y) == old
    finally:
        _close(window)


def test_a_shift_drop_on_a_host_moves_instead() -> None:
    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # the move lands on the tower
        tower = _place_tower(window)
        archer = _live(window, _REF_ARCHER_P2)

        _drag_to_tower(window, [(2, _REF_ARCHER_P2)], tower, Qt.ShiftModifier)

        assert archer.garrisoned_in_id == -1
        assert (archer.x, archer.y) == (tower.x, tower.y)
    finally:
        _close(window)


def test_an_over_capacity_group_drop_is_refused_with_no_change() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        host = _entry(window, 1, tower.reference_id)
        for _ in range(4):
            window._garrison_add_const(host, 4)
        pair = [_live(window, _REF_ARCHER_P2), _live(window, _REF_VILLAGER_P2)]
        old = [(u.x, u.y, u.garrisoned_in_id) for u in pair]
        records_before = len(window.edit_history.records)

        _drag_to_tower(window, [(2, _REF_ARCHER_P2), (2, _REF_VILLAGER_P2)], tower)

        assert [(u.x, u.y, u.garrisoned_in_id) for u in pair] == old
        assert len(window.edit_history.records) == records_before
        assert "has room for 1 more" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_group_drop_with_a_wrong_type_member_is_refused_with_no_change() -> None:
    """The archer makes it a garrison drop; the tree then refuses the whole
    batch rather than garrisoning the archer and moving the tree."""
    window = _window()
    try:
        tower = _place_tower(window)
        archer, tree = _live(window, _REF_ARCHER_P2), _live(window, _REF_TREE_PINE)
        old = [(u.x, u.y) for u in (archer, tree)]
        records_before = len(window.edit_history.records)

        _drag_to_tower(window, [(2, _REF_ARCHER_P2), (0, _REF_TREE_PINE)], tower)

        assert [(u.x, u.y) for u in (archer, tree)] == old
        assert archer.garrisoned_in_id == -1
        assert len(window.edit_history.records) == records_before
        assert "cannot go inside Watch Tower" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_dropping_on_a_unit_that_holds_nothing_moves() -> None:
    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # the move lands on the tree
        oak = _live(window, _REF_TREE_OAK)
        archer = _live(window, _REF_ARCHER_P2)

        _drag_to_tower(window, [(2, _REF_ARCHER_P2)], oak)

        assert archer.garrisoned_in_id == -1
        assert (int(archer.x), int(archer.y)) == (int(oak.x), int(oak.y))
    finally:
        _close(window)


def test_a_drop_on_a_visible_occupant_garrisons_into_its_host() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        window._garrison_add_const(_entry(window, 1, tower.reference_id), 83)
        hit = window.map_view.pick_unit_cover_at(_pos_for_tile(window, int(tower.x), int(tower.y)))
        assert hit is not None and hit[0].unit.garrisoned_in_id == tower.reference_id, "the occupant must be on top"
        archer = _live(window, _REF_ARCHER_P2)

        _drag_to_tower(window, [(2, _REF_ARCHER_P2)], tower)

        assert archer.garrisoned_in_id == tower.reference_id
    finally:
        _close(window)


def test_dragging_a_unit_that_holds_a_garrison_onto_a_host_is_a_status_line() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        archer = _live(window, _REF_ARCHER_P2)
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Garrison units", [2]):
            model.set_garrisoned_in(_live(window, _REF_VILLAGER_P2), _REF_ARCHER_P2)
        old = (archer.x, archer.y)
        records_before = len(window.edit_history.records)

        _drag_to_tower(window, [(2, _REF_ARCHER_P2)], tower)

        assert (archer.x, archer.y) == old
        assert archer.garrisoned_in_id == -1
        assert len(window.edit_history.records) == records_before
        assert "holds a garrison of its own" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_the_drag_preview_outlines_a_host_only_when_the_drop_would_garrison() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        key = (2, _REF_ARCHER_P2)
        window._selection = [key]
        window._refresh_selection_view()
        over_tower = _pos_for_tile(window, int(tower.x), int(tower.y))

        window.on_unit_drag_preview(key, over_tower, Qt.NoModifier)
        assert window.map_view._unit_hover_key == (1, tower.reference_id)

        window.on_unit_drag_preview(key, over_tower, Qt.ShiftModifier)
        assert window.map_view._unit_hover_key is None

        host = _entry(window, 1, tower.reference_id)
        for _ in range(5):
            window._garrison_add_const(host, 4)
        window.on_unit_drag_preview(key, over_tower, Qt.NoModifier)
        assert window.map_view._unit_hover_key is None
    finally:
        _close(window)


def test_the_drag_preview_host_outline_uses_the_configured_hover_colour() -> None:
    """TASK-130 1: the outline is the hover pen (default white), not a fixed cyan.

    Passes when the drop-target outline drawn over an accepting host has the
    configured `unit_hover` colour, which by default is #ffffff, and after
    the setting changes to #00ffff and the overlay colours re-apply, the
    same preview draws in the new colour."""
    from PyQt5.QtGui import QColor

    from descape import settings

    window = _window()
    try:
        tower = _place_tower(window)
        key = (2, _REF_ARCHER_P2)
        window._selection = [key]
        window._refresh_selection_view()
        over_tower = _pos_for_tile(window, int(tower.x), int(tower.y))
        mv = window.map_view

        window.on_unit_drag_preview(key, over_tower, Qt.NoModifier)
        assert mv._unit_hover_key == (1, tower.reference_id)
        assert settings.get_overlay_color("unit_hover") == "#ffffff"
        assert mv._unit_hover_item.pen().color() == QColor(settings.get_overlay_color("unit_hover"))

        window.on_unit_drag_preview(key, over_tower, Qt.ShiftModifier)
        assert mv._unit_hover_item is None
        settings.set_overlay_color("unit_hover", "#00ffff")
        mv.apply_overlay_colors()
        window.on_unit_drag_preview(key, over_tower, Qt.NoModifier)
        assert mv._unit_hover_item.pen().color() == QColor("#00ffff")
    finally:
        _close(window)


def test_an_off_map_group_drag_outlines_the_edge_host_its_clamped_release_garrisons_into() -> None:
    """A group release clamps to the map rect, so the preview must test the clamped point too."""
    window = _window()
    try:
        occupied = {(int(u.x), int(u.y)) for units in window.scenario.unit_manager.units for u in units}
        ty = next(y for y in range(10, 80) if (0, y) not in occupied)
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Place tower", [1]):
            tower = model.add(1, 79, 0.5, ty + 0.5)
        window._rebuild_unit_index()
        keys = [(2, _REF_ARCHER_P2), (2, _REF_VILLAGER_P2)]
        window._selection = list(keys)
        window._refresh_selection_view()
        off_map = QPointF(-3 * window.map_view._tile_pixels, _pos_for_tile(window, 0, ty).y())

        window.on_unit_drag_preview(keys[0], off_map, Qt.NoModifier)
        assert window.map_view._unit_hover_key == (1, tower.reference_id)
        window.on_unit_move(keys[0], window.map_view._clamped_to_map_rect(off_map), Qt.NoModifier)
        assert _live(window, _REF_ARCHER_P2).garrisoned_in_id == tower.reference_id
    finally:
        _close(window)


# --- GH #115 step 4: an owner change carries the garrison --------------------


def _tower_with_mixed_garrison(window):
    """A Player 1 tower holding two Player 1 occupants and Player 2's archer,
    all hidden by the default filter, with the tower selected."""
    tower, occupants = _tower_with_two_inside(window)
    assert window._garrison_units(_entry(window, 1, tower.reference_id), [_entry(window, 2, _REF_ARCHER_P2)])
    _hide_garrisoned(window)
    window._selection = [(1, tower.reference_id)]
    window._refresh_selection_view()
    return tower, [*occupants, _live(window, _REF_ARCHER_P2)]


def test_a_host_owner_change_carries_every_occupant_in_one_record() -> None:
    window = _window()
    try:
        tower, inside = _tower_with_mixed_garrison(window)
        layout = _unit_layout(window)
        records_before = len(window.edit_history.records)

        combo = window.units_panel.unit_field_editors["player"]
        combo.setCurrentIndex(combo.findData(3))

        assert _owner_of(window, tower.reference_id) == 3
        assert [_owner_of(window, u.reference_id) for u in inside] == [3, 3, 3]
        assert all(u.garrisoned_in_id == tower.reference_id for u in inside)
        assert len(window.edit_history.records) == records_before + 1
        assert window._selection == [(3, tower.reference_id)]

        window.undo()
        assert _unit_layout(window) == layout
    finally:
        _close(window)


def test_a_convert_stroke_over_a_host_carries_its_hidden_occupants() -> None:
    window = _window()
    try:
        tower, inside = _tower_with_mixed_garrison(window)
        window.units_panel.select_owner(4)
        window.brush_size_spin.setValue(1)
        layout = _unit_layout(window)
        records_before = len(window.edit_history.records)

        window._begin_convert_stroke()
        window._convert_stroke_tile(int(tower.x), int(tower.y))
        window._end_convert_stroke()

        assert _owner_of(window, tower.reference_id) == 4
        assert [_owner_of(window, u.reference_id) for u in inside] == [4, 4, 4]
        assert len(window.edit_history.records) == records_before + 1
        assert "Converted 4 unit(s)" in window.status_log.toPlainText()

        window.undo()
        assert _unit_layout(window) == layout
    finally:
        _close(window)


# --- GH #115 step 5: the Garrison block's manager buttons ---------------------


def test_map_pick_garrisons_each_click_and_stays_armed_until_esc() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        _hide_garrisoned(window)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        panel = window.units_panel

        panel.garrison_pick_button.click()
        assert window.map_view.unit_picker_active()
        assert window._selection == [(1, tower.reference_id)], "arming must keep the host selected"
        assert panel.garrison_tree.isVisibleTo(panel)

        window.map_view.on_unit_pick((2, _REF_ARCHER_P2))
        window.map_view.on_unit_pick((2, _REF_VILLAGER_P2))

        assert _live(window, _REF_ARCHER_P2).garrisoned_in_id == tower.reference_id
        assert _live(window, _REF_VILLAGER_P2).garrisoned_in_id == tower.reference_id
        assert window.map_view.unit_picker_active()
        assert panel.garrison_pick_button.isChecked()
        assert panel.garrison_tree.topLevelItemCount() == 2

        window.map_view.on_unit_pick((1, tower.reference_id))
        assert "host itself" in window.status_log.toPlainText()

        window.map_view.on_unit_picker_cancel()
        assert not window.map_view.unit_picker_active()
        assert not panel.garrison_pick_button.isChecked()
    finally:
        _close(window)


def test_a_map_pick_refused_by_the_rule_changes_nothing_and_stays_armed() -> None:
    window = _window()
    try:
        _place_tower(window)
        window.units_panel.garrison_pick_button.click()
        oak = _live(window, _REF_TREE_OAK)
        records_before = len(window.edit_history.records)

        window.map_view.on_unit_pick((0, _REF_TREE_OAK))

        assert oak.garrisoned_in_id == -1
        assert len(window.edit_history.records) == records_before
        assert "cannot go inside Watch Tower" in window.status_log.toPlainText()
        assert window.map_view.unit_picker_active()
    finally:
        _close(window)


def test_selecting_another_unit_disarms_the_garrison_pick() -> None:
    window = _window()
    try:
        tower = _place_tower(window)
        window.units_panel.garrison_pick_button.click()
        window._selection = [(2, _REF_ARCHER_P2)]
        window._refresh_selection_view()
        assert not window.map_view.unit_picker_active()
        assert window._garrison_picker is None
        assert tower.garrisoned_in_id == -1
    finally:
        _close(window)


def test_the_unload_button_makes_the_selected_rows_visible_and_pickable() -> None:
    window = _window()
    try:
        tower, occupants = _tower_with_two_inside(window)
        _hide_garrisoned(window)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        _select_rows(window, {occupants[0].reference_id})

        window.units_panel.garrison_unload_button.click()

        assert occupants[0].garrisoned_in_id == -1
        assert _entry(window, 1, occupants[0].reference_id) is not None
        assert occupants[1].garrisoned_in_id == tower.reference_id
        assert window.units_panel.garrison_tree.topLevelItemCount() == 1
    finally:
        _close(window)


def test_copy_then_paste_into_a_second_host_keeps_each_occupants_owner() -> None:
    window = _window()
    try:
        tower, occupants = _tower_with_two_inside(window)
        archer = _live(window, _REF_ARCHER_P2)
        assert window._garrison_units(_entry(window, 1, tower.reference_id), [_entry(window, 2, _REF_ARCHER_P2)])
        other = _second_tower(window)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        _select_rows(window, {occupants[0].reference_id, archer.reference_id})
        panel = window.units_panel
        assert not panel.garrison_paste_button.isEnabled()

        panel.garrison_copy_button.click()
        assert panel.garrison_paste_button.isEnabled()
        assert sorted(panel.garrison_selection()) == sorted([occupants[0].reference_id, archer.reference_id]), (
            "Copy must leave the copied rows selected"
        )
        window._selection = [(2, other.reference_id)]
        window._refresh_selection_view()
        records_before = len(window.edit_history.records)
        panel.garrison_paste_button.click()

        pasted = sorted(
            (p, u.unit_const)
            for p, units in enumerate(window.scenario.unit_manager.units)
            for u in units
            if u.garrisoned_in_id == other.reference_id
        )
        assert pasted == [(1, 4), (2, 4)]
        assert len(window.edit_history.records) == records_before + 1
        assert panel.garrison_tree.topLevelItemCount() == 2

        window.undo()
        assert not any(u.garrisoned_in_id == other.reference_id for us in window.scenario.unit_manager.units for u in us)
    finally:
        _close(window)


def test_copy_then_paste_into_a_second_host_keeps_each_occupants_own_fields() -> None:
    """TASK-130 3, on the 1.59 fixture so the capture flag is a real field.

    Passes when two occupants of one tower that differ from each other and
    from add()'s defaults in rotation, status, caption (id and text) and
    capture flag are copied through the panel's Copy and pasted into a
    second tower through its Paste, and each pasted copy carries its own
    source's const, owner, rotation, status, caption and capture flag."""
    from descape.unit_model import capture_flag_of

    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(_V159_FIXTURE)
    try:
        assert window.scenario is not None and window.scenario.scenario_version == "1.59"
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        _show_garrisoned(window)
        window.mode_combo.setCurrentText("Units")
        model = window._ensure_unit_edits()
        tile = _empty_tile(window)
        with window._unit_edit(model, "Place towers", [1, 2]):
            tower = model.add(1, 79, tile[0] + 0.5, tile[1] + 0.5)
            other = model.add(2, 79, tile[0] + 3.5, tile[1] + 0.5)
            inside = [
                model.add(1, 4, tower.x, tower.y, rotation=1.25, status=4, garrisoned_in_id=tower.reference_id,
                          caption_string_id=7, caption_string="First", capture_flag=1),
                model.add(2, 83, tower.x, tower.y, rotation=2.5, status=3, garrisoned_in_id=tower.reference_id,
                          caption_string_id=9, caption_string="Second", capture_flag=3),
            ]
        window._rebuild_unit_index()

        def fields(player, unit):
            return (player, unit.unit_const, unit.rotation, unit.status,
                    unit.caption_string_id, unit.caption_string, capture_flag_of(unit))

        source = sorted(fields(p, u) for p, u in zip((1, 2), inside, strict=True))
        assert len({row[2:] for row in source}) == 2
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        _select_rows(window, {u.reference_id for u in inside})
        panel = window.units_panel
        panel.garrison_copy_button.click()
        window._selection = [(2, other.reference_id)]
        window._refresh_selection_view()
        panel.garrison_paste_button.click()

        pasted = sorted(
            fields(p, u)
            for p, units in enumerate(window.scenario.unit_manager.units)
            for u in units
            if u.garrisoned_in_id == other.reference_id
        )
        assert pasted == source
    finally:
        _close(window)


def test_paste_is_refused_whole_when_it_would_overfill_the_host() -> None:
    window = _window()
    try:
        tower, occupants = _tower_with_two_inside(window)
        other = _second_tower(window)
        for _ in range(4):
            window._garrison_add_const(_entry(window, 2, other.reference_id), 4)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        _select_rows(window, {u.reference_id for u in occupants})
        window.units_panel.garrison_copy_button.click()
        window._selection = [(2, other.reference_id)]
        window._refresh_selection_view()
        before = _unit_count(window)
        records_before = len(window.edit_history.records)

        window.units_panel.garrison_paste_button.click()

        assert _unit_count(window) == before
        assert len(window.edit_history.records) == records_before
        assert "has room for 1 more" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_the_per_row_owner_reassigns_only_the_selected_occupant() -> None:
    window = _window()
    try:
        tower, occupants = _tower_with_two_inside(window)
        _hide_garrisoned(window)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        _select_rows(window, {occupants[1].reference_id})
        records_before = len(window.edit_history.records)

        combo = window.units_panel.garrison_owner_combo
        combo.setCurrentIndex(combo.findData(3))
        combo.activated.emit(combo.currentIndex())

        assert _owner_of(window, occupants[1].reference_id) == 3
        assert _owner_of(window, occupants[0].reference_id) == 1
        assert _owner_of(window, tower.reference_id) == 1
        assert occupants[1].garrisoned_in_id == tower.reference_id
        assert len(window.edit_history.records) == records_before + 1
        assert window._selection == [(1, tower.reference_id)]
        owners = [window.units_panel.garrison_tree.topLevelItem(i).text(1) for i in range(2)]
        assert sorted(owners) == ["Player 1", "Player 3"]
    finally:
        _close(window)


def _wheel_notch(combo) -> None:
    """One notch down the list, delivered as Qt would (test_unit_choice_viewer.py's _wheel)."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    centre = QPointF(combo.rect().center())
    event = QWheelEvent(centre, centre, QPoint(0, 0), QPoint(0, -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(combo, event)


def test_a_wheel_over_the_unfocused_garrison_owner_combo_reassigns_nothing() -> None:
    """TASK-480: Qt 5 emits activated for a wheel notch, and Owner... acts on it."""
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        tower, occupants = _tower_with_two_inside(window)
        _hide_garrisoned(window)
        window.show()
        QApplication.setActiveWindow(window)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        _select_rows(window, {u.reference_id for u in occupants})
        combo = window.units_panel.garrison_owner_combo
        assert combo.isEnabled()
        window.map_view.setFocus()
        QApplication.processEvents()
        assert not combo.hasFocus()
        records_before = len(window.edit_history.records)

        _wheel_notch(combo)

        assert [_owner_of(window, u.reference_id) for u in occupants] == [1, 1]
        assert len(window.edit_history.records) == records_before
        assert not combo.hasFocus(), "the wheel must not hand the combo focus either"
    finally:
        _close(window)
