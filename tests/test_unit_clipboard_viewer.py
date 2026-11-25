"""GH #192: Copy, Cut and Paste of the selected units, through a real
offscreen ViewerWindow on the units_120x120 fixture.

The clipboard entry itself (region_clipboard.copy_units(), a RegionBlock
without tiles) is covered without Qt in tests/test_region_clipboard.py. This
file is the window half: the copy, the paste write path and where the group
lands, Cut as one undo step, and which focus the three Edit actions answer.

Window helpers come from tests/unit_edit_viewer_helpers.py, whose _window() is
Flat with Show Garrisoned Units on. Every window is closed through _close().
"""

from __future__ import annotations

import math

import pytest
from unit_edit_viewer_helpers import (
    _BARRACKS_CONST,
    _HOUSE_CONST,
    _PLACE_CONST,
    _REF_ARCHER_P1,
    _REF_ARCHER_P2,
    _REF_HOUSE,
    _REF_TREE_OAK,
    _REF_TREE_PINE,
    _REF_VILLAGER_P1,
    _REF_WALL,
    FIXTURE_PATH,
    _add_units,
    _close,
    _hide_garrisoned,
    _last_status_lines,
    _no_install_env,  # noqa: F401 (an autouse fixture)
    _orphan_reference_id,
    _select,
    _select_keys,
    _unit_count,
    _warnings,
    _window,
    _window_with_style,
)

from descape.scenario_io import load_map_and_units
from descape.terrain_palette import tile_span
from descape.unit_model import span_low_corner

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _shown_units_window():
    """_window(), but shown and active: a key press only reaches the window's
    shortcuts, and a widget only takes focus, on a real active window."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = conftest.shown_window(1500, 900)
    # A modifier an earlier test's keyClick left held would turn a click into a toggle.
    for key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta):
        QTest.keyRelease(window, key)
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.show_garrisoned_action.setChecked(True)
    window.mode_combo.setCurrentText("Units")
    window.activateWindow()
    QApplication.setActiveWindow(window)
    QApplication.processEvents()
    return window


def _focus(widget) -> None:
    from PyQt5.QtWidgets import QApplication

    widget.setFocus()
    QApplication.processEvents()
    assert QApplication.focusWidget() is widget


def _ctrl(widget, key) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    QTest.keyClick(widget, key, Qt.ControlModifier)
    QApplication.processEvents()


def _units(window) -> list:
    return [u for units in window.scenario.unit_manager.units for u in units]


def _layout(window) -> list[list[int]]:
    """Every player list as reference ids, in list order."""
    return [[u.reference_id for u in units] for units in window.scenario.unit_manager.units]


def _terrain_ids(window) -> list[int]:
    return [t.terrain_id for t in window.scenario.map_manager.terrain]


def _paste_at(window, tile) -> list:
    """Hovers `tile` and presses Paste; returns the units the paste created."""
    before = {id(u) for u in _units(window)}
    window.on_hover(tile)
    # Qt 5's trigger() fires a disabled action too, so the gate is asserted here.
    assert window.paste_action.isEnabled()
    window.paste_action.trigger()
    return [u for u in _units(window) if id(u) not in before]


def _keys(units) -> set[tuple[int, int]]:
    return {(int(u.player), u.reference_id) for u in units}


# -- copy --------------------------------------------------------------------


def test_copy_units_pushes_one_entry_and_edits_nothing() -> None:
    window = _window()
    try:
        _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        layout = _layout(window)
        records = len(window.edit_history.records)

        assert window.copy_units()

        history = window._clipboard_history
        assert len(history.entries) == 1
        assert history.active is history.entries[0]
        assert not history.active_block.has_tiles
        assert len(history.active_block.units) == 2
        assert history.entries[0].label == "2 units"
        assert len(window.edit_history.records) == records, "a copy is not an edit"
        assert _layout(window) == layout
        assert _status(window) == "Copied 2 units (entry 1 of 10)"
    finally:
        _close(window)


def test_copy_units_with_nothing_selected_pushes_nothing() -> None:
    window = _window()
    try:
        assert not window.copy_units()
        assert window._clipboard_history.entries == []
    finally:
        _close(window)


def test_clipboard_history_lists_a_units_entry() -> None:
    window = _window()
    try:
        _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        window.copy_units()
        window._show_clipboard_history()

        tree = window._clipboard_dialog.tree
        assert tree.topLevelItemCount() == 1
        row = tree.topLevelItem(0)
        assert row.text(1) == "2 units"
        # The archers stand on (11, 10) and (20, 20): the Size column is that bounding box.
        assert row.text(2) == "10x11"
        assert row.text(3) == "2"
        assert row.icon(0).isNull(), "no terrain, so no thumbnail"
    finally:
        _close(window)


# -- paste -------------------------------------------------------------------


def _copy_two_free_units(window) -> list[tuple[int, int]]:
    """Two villagers off their tile centres, selected and copied; returns their keys."""
    keys = _add_units(window, [(_PLACE_CONST, 30.25, 40.75), (_PLACE_CONST, 34.5, 43.125)])
    _select_keys(window, keys)
    assert window.copy_units()
    return keys


def test_paste_units_centres_the_group_on_the_hovered_tile() -> None:
    window = _window()
    try:
        _copy_two_free_units(window)
        window.on_region_selected((2, 2, 5, 5))
        count = _unit_count(window)
        records = len(window.edit_history.records)
        terrain = _terrain_ids(window)

        pasted = _paste_at(window, (60, 70))

        # Anchor tiles span x 30..34 and y 40..43, so the anchor is (60 - 2, 70 - 1).
        assert sorted((u.x, u.y) for u in pasted) == [(58.25, 69.75), (62.5, 72.125)]
        assert _unit_count(window) == count + 2
        assert len(window.edit_history.records) == records + 1
        record = window.edit_history.records[-1]
        assert record.label == "Paste Units"
        assert record.kinds() == {"unit"}
        assert _terrain_ids(window) == terrain
        assert set(window._selection) == _keys(pasted)
        assert window._region == (2, 2, 5, 5)
        assert window._paste_move is None
        assert _status(window) == "Pasted 2 units at (60, 70)"

        window.undo()
        assert _unit_count(window) == count
        assert not {(58.25, 69.75), (62.5, 72.125)} & {(u.x, u.y) for u in _units(window)}
        window.redo()
        assert _unit_count(window) == count + 2
        assert {(58.25, 69.75), (62.5, 72.125)} <= {(u.x, u.y) for u in _units(window)}
    finally:
        _close(window)


def test_paste_units_selects_the_pasted_units_in_stepped() -> None:
    """The units-only branch of the paste write path: Flat has its own."""
    window = _window_with_style("Stepped")
    try:
        _copy_two_free_units(window)
        count = _unit_count(window)

        pasted = _paste_at(window, (60, 70))

        assert _unit_count(window) == count + 2
        assert set(window._selection) == _keys(pasted)
        assert sorted((u.x, u.y) for u in pasted) == [(58.25, 69.75), (62.5, 72.125)]
    finally:
        _close(window)


def test_paste_units_ignores_the_paste_filter_checkboxes() -> None:
    window = _window()
    try:
        _copy_two_free_units(window)
        window.paste_units_check.setChecked(False)
        window.paste_terrain_check.setChecked(True)
        window.paste_elevation_check.setChecked(True)
        count = _unit_count(window)
        terrain = _terrain_ids(window)

        pasted = _paste_at(window, (60, 70))

        assert len(pasted) == 2, "an unchecked Units box must not make Paste a no-op"
        assert _unit_count(window) == count + 2
        assert _terrain_ids(window) == terrain
        assert window.edit_history.records[-1].kinds() == {"unit"}
    finally:
        _close(window)


def test_the_paste_write_path_never_reads_tiles_from_a_units_entry() -> None:
    """Every caller is covered, not only Paste Units: the tile half would
    index the block's empty tuples."""
    window = _window()
    try:
        _copy_two_free_units(window)
        block = window._clipboard_history.active_block
        terrain = _terrain_ids(window)

        record, parts, _skipped, added, _reshaped = window._paste_block_at(block, 58, 69, True, True, True)

        assert parts == ["units"]
        assert record.label == "Paste Units"
        assert len(added) == 2
        assert _terrain_ids(window) == terrain
    finally:
        _close(window)


def test_paste_units_centres_a_host_whose_occupant_is_stored_elsewhere() -> None:
    """The fixture's villager 203 is stored two tiles from house 200, which
    holds it: it rides along and must not pull the house off the cursor."""
    window = _window()
    try:
        _select(window, _REF_HOUSE)
        assert window.copy_units()
        assert _status(window) == "Copied 1 unit (1 garrisoned) (entry 1 of 10)"

        pasted = _paste_at(window, (60, 70))

        by_const = {u.unit_const: u for u in pasted}
        assert set(by_const) == {_HOUSE_CONST, _PLACE_CONST}
        house, villager = by_const[_HOUSE_CONST], by_const[_PLACE_CONST]
        assert (house.x, house.y) == (60.5, 70.5)
        assert (villager.x, villager.y) == (62.5, 70.5)
        assert _status(window) == "Pasted 1 unit (1 garrisoned) at (60, 70)"
    finally:
        _close(window)


def _footprint(unit) -> tuple[int, int, int, int]:
    """(low_x, low_y, high_x, high_y), half-open, never clamped to the map."""
    span_x, span_y = tile_span(unit.unit_const, (1, 1))
    low_x, low_y = span_low_corner(unit)
    return low_x, low_y, low_x + span_x, low_y + span_y


def _offsets(units) -> list[tuple[float, float]]:
    """Each unit's position relative to the group's first, by const."""
    ordered = sorted(units, key=lambda u: u.unit_const)
    return [(u.x - ordered[0].x, u.y - ordered[0].y) for u in ordered]


@pytest.mark.parametrize("hover", [(0, 0), (119, 119), (0, 119)])
def test_paste_units_at_the_map_edge_keeps_every_footprint_on_the_map(hover) -> None:
    window = _window()
    try:
        mm = window.scenario.map_manager
        # A 3x3 Barracks, a 2x2 House and a villager: three different spans.
        keys = _add_units(
            window, [(_BARRACKS_CONST, 40.5, 40.5), (_HOUSE_CONST, 37.0, 44.0), (_PLACE_CONST, 44.5, 42.5)]
        )
        _select_keys(window, keys)
        sources = [window.map_view._unit_index.entry_for_key(k).unit for k in keys]
        assert window.copy_units()

        pasted = _paste_at(window, hover)

        assert len(pasted) == 3, "nothing is dropped: the group is pulled inside instead"
        boxes = [_footprint(u) for u in pasted]
        for low_x, low_y, high_x, high_y in boxes:
            assert low_x >= 0 and low_y >= 0
            assert high_x <= mm.map_width and high_y <= mm.map_height
        assert _offsets(pasted) == _offsets(sources), "the group keeps its shape"
        # Pulled in no further than needed: on each axis some footprint touches the edge.
        if hover[0] == 0:
            assert min(box[0] for box in boxes) == 0
        else:
            assert max(box[2] for box in boxes) == mm.map_width
        if hover[1] == 0:
            assert min(box[1] for box in boxes) == 0
        else:
            assert max(box[3] for box in boxes) == mm.map_height
    finally:
        _close(window)


def _copy_two_spread_villagers(window) -> None:
    """Two villagers four tiles apart on both axes: a span-1 unit is the
    group's outermost member on every edge."""
    keys = _add_units(window, [(_PLACE_CONST, 30.5, 30.5), (_PLACE_CONST, 34.5, 34.5)])
    _select_keys(window, keys)
    assert window.copy_units()


@pytest.mark.parametrize(
    ("hover", "expected", "centre"),
    [
        ((0, 50), [(0.5, 48.5), (4.5, 52.5)], (2, 50)),
        # One tile in: the low villager would land on x = -0.5, which int() reads as tile 0.
        ((1, 50), [(0.5, 48.5), (4.5, 52.5)], (2, 50)),
        ((50, 0), [(48.5, 0.5), (52.5, 4.5)], (50, 2)),
        ((119, 50), [(115.5, 48.5), (119.5, 52.5)], (117, 50)),
        ((50, 119), [(48.5, 115.5), (52.5, 119.5)], (50, 117)),
    ],
)
def test_paste_units_at_the_map_edge_keeps_a_free_unit_on_the_map(hover, expected, centre) -> None:
    """Measured with floor, never with the truncating footprint helper, which
    cannot tell a coordinate in (-1, 0) from tile 0."""
    window = _window()
    try:
        mm = window.scenario.map_manager
        _copy_two_spread_villagers(window)

        pasted = _paste_at(window, hover)

        for unit in pasted:
            assert unit.x >= 0 and unit.y >= 0, f"pasted off the low edge at ({unit.x}, {unit.y})"
            assert math.floor(unit.x) < mm.map_width and math.floor(unit.y) < mm.map_height
        assert len(pasted) == 2, "nothing is dropped: the group is pulled inside instead"
        assert sorted((u.x, u.y) for u in pasted) == expected
        assert _status(window) == f"Pasted 2 units at ({centre[0]}, {centre[1]})"
    finally:
        _close(window)


def test_a_low_edge_paste_is_on_the_map_after_save_and_reload(tmp_path) -> None:
    window = _window()
    try:
        _copy_two_spread_villagers(window)
        pasted = _paste_at(window, (1, 50))
        positions = sorted((u.x, u.y) for u in pasted)

        out = tmp_path / "out.aoe2scenario"
        window.scenario.path = out
        window._untitled = False
        window.save()

        reloaded = [u for units in load_map_and_units(out).unit_manager.units for u in units]
        assert len(reloaded) == _unit_count(window)
        assert [(u.x, u.y) for u in reloaded if u.x < 0 or u.y < 0] == []
        assert set(positions) <= {(u.x, u.y) for u in reloaded}
        assert positions == [(0.5, 48.5), (4.5, 52.5)]
    finally:
        _close(window)


def test_paste_units_lands_an_occupant_stored_past_the_low_edge_on_its_host() -> None:
    """An occupant rides with its host and does not limit the clamp, so one
    stored on the host's low side would land in (-1, 0): it lands on the
    host's own x, y and z instead (TASK-341), never off the map."""
    window = _window()
    try:
        model = window._ensure_unit_edits()
        occupant = next(u for u in _units(window) if u.reference_id == _REF_VILLAGER_P1)
        with window._unit_edit(model, "Move villager", [1]):
            model.set_position(occupant, 8.5, 10.5, occupant.z)
        _select(window, _REF_HOUSE)
        assert window.copy_units()

        pasted = _paste_at(window, (1, 50))

        # The house anchors on x = 1.5, so its occupant's own offset would put it on x = -0.5.
        by_const = {u.unit_const: u for u in pasted}
        house, villager = by_const[_HOUSE_CONST], by_const[_PLACE_CONST]
        assert (house.x, house.y) == (1.5, 50.5)
        assert (villager.x, villager.y, villager.z) == (house.x, house.y, house.z)
        assert villager.garrisoned_in_id == house.reference_id
        assert _status(window) == "Pasted 1 unit (1 garrisoned) at (1, 50)"
    finally:
        _close(window)


@pytest.mark.parametrize("hover", [(-3, 5), (5, 120), None])
def test_paste_units_with_no_tile_on_the_map_pastes_nothing(hover) -> None:
    """Flat reports the tile under the cursor even past the map's edge."""
    window = _window()
    try:
        _copy_two_free_units(window)
        count = _unit_count(window)
        records = len(window.edit_history.records)
        selection = list(window._selection)

        assert _paste_at(window, hover) == []

        assert _unit_count(window) == count
        assert len(window.edit_history.records) == records
        assert window._selection == selection
        assert _status(window) == "Paste Units: no target tile, hover the map first"
    finally:
        _close(window)


@pytest.mark.parametrize("show_garrisoned", [True, False])
def test_paste_units_relinks_a_garrison_to_the_new_host(show_garrisoned) -> None:
    window = _window()
    try:
        if not show_garrisoned:
            _hide_garrisoned(window)  # the shipping default: the occupant cannot be selected
        _select(window, _REF_HOUSE)
        assert window.copy_units()

        pasted = _paste_at(window, (60, 70))

        by_const = {u.unit_const: u for u in pasted}
        house, villager = by_const[_HOUSE_CONST], by_const[_PLACE_CONST]
        assert house.reference_id not in (_REF_HOUSE, _REF_VILLAGER_P1)
        assert villager.garrisoned_in_id == house.reference_id
        assert house.garrisoned_in_id == -1
        assert window._selection == [(1, house.reference_id)], "only the host is selected"
    finally:
        _close(window)


def test_paste_units_carries_variant_rotations_verbatim() -> None:
    """A tree's and a wall's rotation is a variant index, not an angle."""
    window = _window()
    try:
        entries = _select(window, _REF_TREE_OAK, _REF_TREE_PINE, _REF_WALL)
        stored = {e.unit.unit_const: (e.unit.rotation, e.unit.initial_animation_frame) for e in entries}
        assert stored[349][0] == 7.0, "the fixture's oak stores a variant index"
        assert window.copy_units()

        pasted = _paste_at(window, (60, 70))

        assert {u.unit_const: (u.rotation, u.initial_animation_frame) for u in pasted} == stored
        assert all(u.player == 0 for u in pasted), "the owner is kept"
    finally:
        _close(window)


def test_paste_units_notes_a_unit_the_filter_hides() -> None:
    window = _window()
    try:
        _select(window, _REF_TREE_OAK)
        assert window.copy_units()
        window.show_trees_action.setChecked(False)

        pasted = _paste_at(window, (60, 70))

        assert len(pasted) == 1
        assert window._selection == []
        assert _status(window) == (
            "Pasted 1 unit at (60, 70) (a Filters toggle is hiding it, so it won't be visible)"
        )
    finally:
        _close(window)


def test_a_units_entry_pastes_in_terrain_mode_without_selecting() -> None:
    window = _window()
    try:
        _copy_two_free_units(window)
        window.mode_combo.setCurrentText("Terrain")
        window.on_region_selected((2, 2, 5, 5))
        selection = list(window._selection)
        count = _unit_count(window)

        pasted = _paste_at(window, (60, 70))

        assert len(pasted) == 2
        assert _unit_count(window) == count + 2
        assert window._selection == selection
        assert window._region == (2, 2, 5, 5)
        assert window._paste_move is None
        # Exact: outside Units mode nothing is selected, so nothing reads as hidden by a filter.
        assert _status(window) == "Pasted 2 units at (60, 70)"
    finally:
        _close(window)


def test_paste_units_on_read_only_units_reports_the_skip(monkeypatch) -> None:
    window = _window()
    try:
        _copy_two_free_units(window)
        count = _unit_count(window)
        records = len(window.edit_history.records)
        selection = list(window._selection)
        monkeypatch.setattr(window, "_ensure_unit_edits", lambda: None)

        assert _paste_at(window, (60, 70)) == []

        assert _unit_count(window) == count
        assert len(window.edit_history.records) == records
        assert window._selection == selection
        assert _status(window).startswith("Units were skipped")
    finally:
        _close(window)


# -- cut ---------------------------------------------------------------------


def test_cut_units_is_one_undo_step_that_restores_exact_list_indices() -> None:
    window = _window()
    try:
        # One unit from each of three player lists, each ahead of a survivor.
        entries = _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2, _REF_TREE_PINE)
        assert len({e.player_id for e in entries}) == 3
        layout = _layout(window)
        count = _unit_count(window)
        records = len(window.edit_history.records)

        window.cut_units()

        assert _unit_count(window) == count - 3
        assert len(window.edit_history.records) == records + 1, "Cut is one undo step"
        record = window.edit_history.peek_undo()
        assert record.label == "Cut 3 units"
        assert record.kinds() == {"unit"}
        assert window._selection == []
        history = window._clipboard_history
        assert len(history.entries) == 1
        assert not history.active_block.has_tiles
        assert sorted(u.unit_const for u in history.active_block.units) == sorted(
            e.unit.unit_const for e in entries
        )
        assert _last_status_lines(window, 2) == ["Cut 3 units", "Cut 3 units to the clipboard"]

        window.undo()
        assert _layout(window) == layout
        assert len(history.entries) == 1, "undo puts the units back and keeps the copy"
        window.redo()
        assert _unit_count(window) == count - 3
    finally:
        _close(window)


def test_cutting_one_unit_is_one_record() -> None:
    from descape.viewer_common import _unit_name

    window = _window()
    try:
        (entry,) = _select(window, _REF_ARCHER_P1)
        name = _unit_name(entry.unit.unit_const)
        layout = _layout(window)
        records = len(window.edit_history.records)

        window.cut_units()

        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.peek_undo().label == "Cut unit"
        assert _REF_ARCHER_P1 not in {u.reference_id for u in _units(window)}
        assert len(window._clipboard_history.active_block.units) == 1
        assert _last_status_lines(window, 2) == ["Cut unit", f"Cut {name} to the clipboard"]

        window.undo()
        assert _layout(window) == layout
    finally:
        _close(window)


def test_cutting_a_host_takes_its_hidden_garrison_in_one_record() -> None:
    from descape.viewer_common import _unit_name

    window = _window()
    try:
        _hide_garrisoned(window)  # the shipping default: only the host can be selected
        (entry,) = _select(window, _REF_HOUSE)
        name = _unit_name(entry.unit.unit_const)
        layout = _layout(window)
        count = _unit_count(window)
        records = len(window.edit_history.records)

        window.cut_units()

        assert _unit_count(window) == count - 2
        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.peek_undo().label == "Cut 2 units"
        assert _last_status_lines(window, 2) == ["Cut 2 units", f"Cut {name} (1 garrisoned) to the clipboard"]
        block = window._clipboard_history.active_block
        by_const = {u.unit_const: slot for slot, u in enumerate(block.units)}
        assert block.units[by_const[_PLACE_CONST]].garrison_slot == by_const[_HOUSE_CONST]

        # The clipboard holds both: pasting brings back a house with its villager inside.
        pasted = {u.unit_const: u for u in _paste_at(window, (60, 70))}
        assert pasted[_PLACE_CONST].garrisoned_in_id == pasted[_HOUSE_CONST].reference_id

        window.undo()
        window.undo()
        assert _layout(window) == layout
    finally:
        _close(window)


def test_cut_units_on_read_only_units_copies_and_removes_nothing(monkeypatch) -> None:
    window = _window()
    try:
        _select(window, _REF_ARCHER_P1)
        count = _unit_count(window)
        records = len(window.edit_history.records)
        monkeypatch.setattr(window, "_ensure_unit_edits", lambda: None)

        window.cut_units()

        assert _unit_count(window) == count
        assert len(window.edit_history.records) == records
        assert window._clipboard_history.entries == []
        assert _status(window) == "Cut Units: units are read-only for this file"
    finally:
        _close(window)


def test_a_refused_cut_copies_nothing_and_warns_as_a_cut() -> None:
    window = _window()
    try:
        warned = _warnings(window)
        house_key = _orphan_reference_id(window, 1, _REF_HOUSE)
        _select_keys(window, [house_key])
        count = _unit_count(window)
        records = len(window.edit_history.records)

        window.cut_units()

        assert _unit_count(window) == count
        assert len(window.edit_history.records) == records
        assert window._clipboard_history.entries == [], "nothing removable, so nothing copied"
        assert _status(window) == "Cut failed: referenced by a garrison"
        assert warned == [("Cannot cut unit", "Referenced by another unit's garrison.")]
    finally:
        _close(window)


def test_a_partly_refused_cut_copies_only_what_it_removes() -> None:
    window = _window()
    try:
        warned = _warnings(window)
        house_key = _orphan_reference_id(window, 1, _REF_HOUSE)
        _select_keys(window, [house_key, (1, _REF_ARCHER_P1)])
        count = _unit_count(window)

        window.cut_units()

        assert _unit_count(window) == count - 1
        assert window.edit_history.peek_undo().label == "Cut unit"
        assert [u.unit_const for u in window._clipboard_history.active_block.units] == [4], "the archer alone"
        assert _last_status_lines(window, 2) == ["Cut unit", "Cut 1 to the clipboard; 1 refused (garrison reference)"]
        assert warned == [
            ("Some units not cut", "1 unit(s) are referenced by another unit's garrison and were not cut.")
        ]
    finally:
        _close(window)


# -- cut keeps reference_ids (TASK-341) ----------------------------------------


def _ids(window) -> list[int]:
    return [u.reference_id for u in _units(window)]


def _cut_house_and_archer(window) -> dict[int, int]:
    """Cuts the house (its villager rides along hidden) and a free archer;
    returns const -> the cut unit's reference_id."""
    _hide_garrisoned(window)
    _select(window, _REF_HOUSE, _REF_ARCHER_P1)
    window.cut_units()
    assert not {_REF_HOUSE, _REF_VILLAGER_P1, _REF_ARCHER_P1} & set(_ids(window))
    return {_HOUSE_CONST: _REF_HOUSE, _PLACE_CONST: _REF_VILLAGER_P1, 4: _REF_ARCHER_P1}


def test_the_first_paste_after_a_cut_keeps_the_ids_and_a_second_mints() -> None:
    window = _window()
    try:
        cut = _cut_house_and_archer(window)

        first = {u.unit_const: u for u in _paste_at(window, (60, 70))}

        assert {const: u.reference_id for const, u in first.items()} == cut
        assert first[_PLACE_CONST].garrisoned_in_id == _REF_HOUSE
        assert _status(window).endswith(", IDs kept")

        second = _paste_at(window, (80, 70))

        assert len(second) == 3
        assert not {u.reference_id for u in second} & set(cut.values())
        assert len(set(_ids(window))) == len(_ids(window)), "ids stay unique document-wide"
        assert "kept" not in _status(window)
    finally:
        _close(window)


def test_undoing_the_paste_frees_the_ids_again_and_undoing_the_cut_does_not() -> None:
    window = _window()
    try:
        cut = _cut_house_and_archer(window)
        _paste_at(window, (60, 70))
        window.undo()  # the paste: the wrong spot

        again = _paste_at(window, (62, 72))

        assert {u.unit_const: u.reference_id for u in again} == cut
        window.undo()  # the second paste
        window.undo()  # the cut: the originals are back with their ids
        assert set(cut.values()) <= set(_ids(window))

        minted = _paste_at(window, (64, 74))

        assert not {u.reference_id for u in minted} & set(cut.values())
        assert len(set(_ids(window))) == len(_ids(window))
        assert "kept" not in _status(window)
    finally:
        _close(window)


def test_a_cut_pasted_into_another_document_mints() -> None:
    import uuid

    window = _window()
    try:
        cut = _cut_house_and_archer(window)
        window._doc_id = uuid.uuid4().hex

        pasted = _paste_at(window, (60, 70))

        assert len(pasted) == 3
        assert not {u.reference_id for u in pasted} & set(cut.values())
        assert "kept" not in _status(window)
    finally:
        _close(window)


def test_a_cut_of_a_unit_placed_then_undone_twice_mints_without_raising() -> None:
    """The counter rewinds below the cut id, so add()'s guard would refuse it."""
    window = _window()
    try:
        [key] = _add_units(window, [(_PLACE_CONST, 30.5, 40.5)])
        _select_keys(window, [key])
        window.cut_units()
        window.undo()  # the cut
        window.undo()  # the place
        model = window._ensure_unit_edits()
        assert key[1] >= model.next_unit_id

        pasted = _paste_at(window, (60, 70))

        assert len(pasted) == 1
        assert len(set(_ids(window))) == len(_ids(window))
    finally:
        _close(window)


def test_a_read_only_paste_of_a_cut_warns_once(monkeypatch) -> None:
    window = _window()
    try:
        _cut_house_and_archer(window)
        warned = _warnings(window)

        def read_only():
            window._warn("Units are read-only", "for this file")

        monkeypatch.setattr(window, "_ensure_unit_edits", read_only)

        assert _paste_at(window, (60, 70)) == []
        assert len(warned) == 1
    finally:
        _close(window)


def test_a_kept_id_is_at_the_new_place_once_after_save_and_reload(tmp_path) -> None:
    window = _window()
    try:
        _cut_house_and_archer(window)
        archer = next(u for u in _paste_at(window, (60, 70)) if u.unit_const == 4)
        assert archer.reference_id == _REF_ARCHER_P1

        out = tmp_path / "out.aoe2scenario"
        window.scenario.path = out
        window._untitled = False
        window.save()

        reloaded = [u for units in load_map_and_units(out).unit_manager.units for u in units]
        same = [u for u in reloaded if u.reference_id == _REF_ARCHER_P1]
        assert [(u.x, u.y) for u in same] == [(archer.x, archer.y)]
    finally:
        _close(window)


# -- the Edit actions: titles, enablement, dispatch ---------------------------


def _edit_state(window) -> list[tuple[str, bool]]:
    return [(a.text(), a.isEnabled()) for a in (window.cut_action, window.copy_action, window.paste_action)]


def test_units_mode_titles_and_enablement_follow_the_unit_selection() -> None:
    window = _window()
    try:
        assert _edit_state(window) == [("Cu&t Units", False), ("&Copy Region", False), ("&Paste Region", False)]

        _select(window, _REF_ARCHER_P1)
        assert _edit_state(window) == [("Cu&t Units", True), ("&Copy Units", True), ("&Paste Region", False)]
        assert window.cut_action.toolTip() == "Cut the selected units and what they garrison, as one undo step"
        assert window.copy_action.toolTip() == "Copy the selected units and what they garrison"

        _select(window)
        assert _edit_state(window) == [("Cu&t Units", False), ("&Copy Region", False), ("&Paste Region", False)]
        assert window.copy_action.toolTip() == "Copy the selected region's terrain, elevation and units"

        window.mode_combo.setCurrentText("Terrain")
        assert _edit_state(window)[0] == ("Cu&t Triggers", False), "Cut is still Triggers and Units only"
    finally:
        _close(window)


def test_with_no_units_selected_copy_is_still_copy_region() -> None:
    window = _window()
    try:
        window.on_region_selected((2, 2, 5, 5))
        assert _edit_state(window)[1] == ("&Copy Region", True)

        window.copy_action.trigger()

        block = window._clipboard_history.active_block
        assert block.has_tiles
        assert (block.width, block.height) == (3, 3)
    finally:
        _close(window)


def test_a_unit_selection_wins_copy_over_a_selected_region() -> None:
    window = _window()
    try:
        window.on_region_selected((2, 2, 5, 5))
        _select(window, _REF_ARCHER_P1)
        assert _edit_state(window)[1] == ("&Copy Units", True)

        window.copy_action.trigger()

        block = window._clipboard_history.active_block
        assert not block.has_tiles
        assert len(block.units) == 1
        assert _status(window) == "Copied 1 unit (entry 1 of 10)"
    finally:
        _close(window)


def test_paste_is_titled_for_the_active_entrys_kind() -> None:
    window = _window()
    try:
        window.on_region_selected((2, 2, 5, 5))
        window.copy_action.trigger()
        region_entry = window._clipboard_history.active
        assert _edit_state(window)[2] == ("&Paste Region", True)

        _select(window, _REF_ARCHER_P1)
        window.copy_action.trigger()
        assert _edit_state(window)[2] == ("&Paste Units", True)
        assert window.paste_action.toolTip() == (
            "Paste the copied units centred on the hovered tile, as one undo step"
        )

        window._on_clipboard_activate(region_entry.entry_id)
        assert _edit_state(window)[2] == ("&Paste Region", True)
        assert window.paste_action.toolTip() == "Paste the copied region at the hovered tile, as one undo step"
    finally:
        _close(window)


def test_the_cut_action_cuts_the_selected_units() -> None:
    window = _window()
    try:
        _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        count = _unit_count(window)

        window.cut_action.trigger()

        assert _unit_count(window) == count - 2
        assert window.edit_history.peek_undo().label == "Cut 2 units"
        assert _edit_state(window) == [("Cu&t Units", False), ("&Copy Region", False), ("&Paste Units", True)]
    finally:
        _close(window)


# -- focus: who answers Ctrl+X and Ctrl+C --------------------------------------


def test_ctrl_x_cuts_from_the_map_and_stays_with_a_focused_spin_box() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    window = _shown_units_window()
    try:
        _select(window, _REF_ARCHER_P1)
        count = _unit_count(window)

        # The control: the key does reach the window's Cut from the map.
        _focus(window.map_view)
        _ctrl(window.map_view, Qt.Key_X)
        assert _unit_count(window) == count - 1
        assert window.edit_history.peek_undo().label == "Cut unit"

        window.undo()
        (entry,) = _select(window, _REF_ARCHER_P1)
        records = len(window.edit_history.records)
        fired = []
        window.cut_action.triggered.connect(lambda *_: fired.append("cut"))
        spin = window.units_panel.unit_field_editors["x"]
        _focus(spin)
        spin.selectAll()
        _ctrl(spin, Qt.Key_X)

        assert fired == [], "a spin box keeps Ctrl+X for its own text"
        assert _unit_count(window) == count
        _focus(window.map_view)
        assert len(window.edit_history.records) == records
        assert entry.unit.x == 11.5
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        _close(window)


def test_the_focused_garrison_list_takes_cut_copy_and_paste_from_the_window() -> None:
    """TASK-341 item 7: with the Garrison list focused, all three keys act on
    its rows through its own buttons, and Paste never reaches the map."""
    import unit_edit_viewer_helpers as helpers
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    window = _shown_units_window()
    try:
        panel = window.units_panel
        tree = panel.garrison_tree
        tower, occupants = helpers._tower_with_two_inside(window)
        other = helpers._second_tower(window)
        _hide_garrisoned(window)
        _select(window, _REF_ARCHER_P1)
        assert window.copy_units(), "a units entry is active, so a map paste could happen"
        window.on_hover((60, 70))
        _select_keys(window, [(1, tower.reference_id)])
        count = _unit_count(window)
        records = len(window.edit_history.records)

        # A click focuses the list first and selects the row after.
        _focus(tree)
        assert _edit_state(window) == [
            ("Cu&t Garrisoned Units", False), ("&Copy Garrisoned Units", False), ("&Paste Garrisoned Units", False),
        ]
        assert window.copy_action.toolTip() == panel.garrison_copy_button.toolTip()
        window._dispatch_paste()
        assert _unit_count(window) == count, "nothing to paste in the list, and no map paste"

        helpers._select_rows(window, {occupants[0].reference_id})
        assert _edit_state(window)[:2] == [("Cu&t Garrisoned Units", True), ("&Copy Garrisoned Units", True)]
        _ctrl(tree, Qt.Key_C)
        assert window._garrison_clipboard, "the list's Copy filled the Garrison clipboard"
        assert len(window.edit_history.records) == records
        assert _edit_state(window)[2] == ("&Paste Garrisoned Units", True)

        _ctrl(tree, Qt.Key_V)
        assert _unit_count(window) == count + 1
        assert window.edit_history.peek_undo().label == "Paste into garrison"
        assert len([u for u in _units(window) if u.garrisoned_in_id == tower.reference_id]) == 3

        # The paste's refresh hides and re-shows the Garrison block (show_unit()), which drops focus.
        _focus(tree)
        helpers._select_rows(window, {occupants[1].reference_id})
        _ctrl(tree, Qt.Key_X)
        assert window._garrison_cut is not None, "Ctrl+X marked the row"
        _select_keys(window, [(2, other.reference_id)])
        _focus(tree)
        _ctrl(tree, Qt.Key_V)
        assert window.edit_history.peek_undo().label == "Move into garrison"
        assert occupants[1].garrisoned_in_id == other.reference_id
        assert _unit_count(window) == count + 1
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        _close(window)


def test_opening_the_edit_menu_rereads_text_selected_in_a_focused_label() -> None:
    """Selecting text inside an already focused label moves no focus, so only
    the Edit menu's aboutToShow brings Cut and Copy up to date (TASK-341)."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication, QMenu

    window = _shown_units_window()
    edit_menu = next(m for m in window.menuBar().findChildren(QMenu) if m.title() == "&Edit")
    try:
        _select(window, _REF_ARCHER_P1)
        label = window.units_panel.unit_field_labels["name"]
        _focus(label)
        label.setSelection(0, len(label.text()))
        assert label.selectedText()
        assert _edit_state(window)[:2] == [("Cu&t Units", True), ("&Copy Units", True)], "stale until the menu opens"

        edit_menu.popup(window.mapToGlobal(QPoint(10, 10)))
        QApplication.processEvents()

        assert _edit_state(window)[:2] == [("Cu&t Units", False), ("&Copy", True)]
    finally:
        edit_menu.hide()
        QApplication.processEvents()
        _close(window)


def test_copy_with_a_labels_text_selected_copies_that_text() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    window = _shown_units_window()
    try:
        _select(window, _REF_ARCHER_P1)
        count = _unit_count(window)
        label = window.units_panel.unit_field_labels["name"]
        text = label.text()
        assert text
        label.setSelection(0, len(text))
        _focus(label)
        assert label.selectedText() == text
        assert _edit_state(window)[:2] == [("Cu&t Units", False), ("&Copy", True)]
        assert window.copy_action.toolTip() == "Copy the selected text"

        QApplication.clipboard().setText("stale")
        _ctrl(label, Qt.Key_C)

        assert QApplication.clipboard().text() == text
        assert window._clipboard_history.entries == [], "the units were not copied"

        _ctrl(label, Qt.Key_X)
        # As above: the dispatch itself refuses, whatever the action's enabled state says.
        window._dispatch_cut()
        assert _unit_count(window) == count
    finally:
        QTest.keyRelease(window, Qt.Key_Control)
        _close(window)
