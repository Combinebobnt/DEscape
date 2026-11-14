"""Units mode: the inspector's fields on a single unit and a group: typed
rotation, x/y/z steps and the map-edge clamp, owner changes.
"""

from __future__ import annotations

import pytest
from PyQt5.QtCore import Qt
from unit_edit_viewer_helpers import (
    _HOUSE_CONST,
    _PLACE_CONST,
    _REF_ARCHER_P1,
    _REF_ARCHER_P2,
    _REF_HOUSE,
    _REF_TREE_OAK,
    _REF_VILLAGER_P1,
    _REF_VILLAGER_P2,
    _REF_WALL,
    _add_units,
    _close,
    _field,
    _fixture_host_and_occupant,
    _free_tiles,
    _hide_garrisoned,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _owner_of,
    _positions,
    _select,
    _select_keys,
    _window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


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


# --- GH #154: a group's X/Y/Z steps shift every member ----------------------


def _last_status_line(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _reposition(window, reference_id: int, x: float, y: float, z: float) -> None:
    """Puts a fixture unit somewhere for a test's premise, outside the record under test."""
    unit = next(u for us in window.scenario.unit_manager.units for u in us if u.reference_id == reference_id)
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Premise", [_owner_of(window, reference_id)], fields_only=True):
        model.set_position(unit, x, y, z)


def test_a_step_on_a_mixed_x_group_moves_every_unit_in_one_undo_record() -> None:
    window = _window()
    try:
        entries = _select(window, _REF_ARCHER_P1, _REF_VILLAGER_P1, _REF_ARCHER_P2)
        assert len({e.player_id for e in entries}) == 2
        before = {e.unit.reference_id: (e.unit.x, e.unit.y, e.unit.z) for e in entries}
        spin = window.units_panel.unit_field_editors["x"]
        assert spin.text() == "(mixed)"
        records_before = len(window.edit_history.records)

        spin.stepBy(1)

        assert len(window.edit_history.records) == records_before + 1
        for entry in entries:
            x, y, z = before[entry.unit.reference_id]
            assert (entry.unit.x, entry.unit.y, entry.unit.z) == (x + 1, y, z)
        assert spin.text() == "(mixed)"
        assert len(window._selection) == 3
        assert _last_status_line(window) == "Moved 3 units by +1 on X"

        window.undo()
        for entry in entries:
            assert (entry.unit.x, entry.unit.y, entry.unit.z) == before[entry.unit.reference_id]
    finally:
        _close(window)


def test_a_step_on_a_shared_value_shows_the_new_common_value() -> None:
    window = _window()
    try:
        entries = _select(window, _REF_ARCHER_P1, _REF_VILLAGER_P1)
        assert {e.unit.y for e in entries} == {10.5}
        spin = window.units_panel.unit_field_editors["y"]
        assert spin.text() == "10.50"
        spin.stepBy(-1)
        assert {e.unit.y for e in entries} == {9.5}
        assert spin.text() == "9.50"
    finally:
        _close(window)


def test_a_real_click_on_the_down_arrow_moves_a_mixed_group() -> None:
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QStyle, QStyleOptionSpinBox

    window = _window()
    try:
        entries = _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        before = [e.unit.x for e in entries]
        spin = window.units_panel.unit_field_editors["x"]
        assert spin.text() == "(mixed)"
        option = QStyleOptionSpinBox()
        spin.initStyleOption(option)
        down = spin.style().subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxDown, spin)
        assert down.isValid()

        QTest.mouseClick(spin, Qt.LeftButton, Qt.NoModifier, down.center())

        assert [e.unit.x for e in entries] == [x - 1 for x in before]
        assert spin.text() == "(mixed)"
    finally:
        _close(window)


def test_typed_text_then_up_sets_the_group_then_shifts_it_in_two_records() -> None:
    """interpretText() runs the typed edit and its repopulate inside stepBy(); the shift follows."""
    from PyQt5.QtTest import QTest

    window = _window()
    try:
        entries = _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        spin = window.units_panel.unit_field_editors["x"]
        records_before = len(window.edit_history.records)
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, "50")
        assert [e.unit.x for e in entries] == [11.5, 20.5]

        QTest.keyClick(spin, Qt.Key_Up)

        assert [e.unit.x for e in entries] == [51.0, 51.0]
        assert len(window.edit_history.records) == records_before + 2
        assert spin.text() == "51.00"
        window.undo()
        assert [e.unit.x for e in entries] == [50.0, 50.0]
    finally:
        _close(window)


def test_a_step_stops_at_the_map_edge() -> None:
    window = _window()
    try:
        _reposition(window, _REF_ARCHER_P2, 119.5, 20.5, 0.0)
        entries = _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        spin = window.units_panel.unit_field_editors["x"]
        records_before = len(window.edit_history.records)

        spin.stepBy(1)

        assert [e.unit.x for e in entries] == [11.5, 119.5]
        assert len(window.edit_history.records) == records_before
        assert _last_status_line(window) == "2 units already at the map edge; nothing moved"

        _reposition(window, _REF_ARCHER_P2, 116.5, 20.5, 0.0)
        records_before = len(window.edit_history.records)
        spin.stepBy(10)

        assert [e.unit.x for e in entries] == [14.5, 119.5]
        assert len(window.edit_history.records) == records_before + 1
        assert _last_status_line(window) == "Moved 2 units by +3 on X, stopped at the map edge"
    finally:
        _close(window)


def test_an_arrow_nudge_stops_the_whole_selection_at_the_map_edge() -> None:
    """TASK-468: the nudge never clamped, so an edge unit slid off the map
    and the next nudge of it crashed the index patch."""
    window = _window()
    try:
        _free_tiles(window, [(118, 50), (119, 50), (30, 50)])
        keys = _add_units(window, [(_HOUSE_CONST, 119.0, 50.0), (_PLACE_CONST, 30.5, 50.5)])
        _select_keys(window, keys)
        before = _positions(window, keys)
        records, cursor = len(window.edit_history.records), window.edit_history.cursor

        for _ in range(2):
            assert window.on_unit_nudge(1, 0, Qt.ShiftModifier) is True
            assert _positions(window, keys) == before
            assert (len(window.edit_history.records), window.edit_history.cursor) == (records, cursor)
            assert _last_status_line(window) == "2 units already at the map edge; nothing moved"

        assert window.on_unit_nudge(-1, 0, Qt.ShiftModifier) is True
        assert _positions(window, keys) == [(x - 1, y) for x, y in before]
        assert len(window.edit_history.records) == records + 1
    finally:
        _close(window)


def test_a_fine_nudge_travels_inside_the_edge_tile_but_never_past_it() -> None:
    from descape.viewer_common import _unit_name

    window = _window()
    try:
        _free_tiles(window, [(119, 52), (119, 54)])
        inside, edge = _add_units(window, [(_PLACE_CONST, 119.5, 52.5), (_PLACE_CONST, 119.95, 54.5)])
        _select_keys(window, [inside])
        window.on_unit_nudge(1, 0, Qt.NoModifier)
        assert _positions(window, [inside])[0][0] == pytest.approx(119.6, abs=1e-4)

        _select_keys(window, [edge])
        start = _positions(window, [edge])
        assert window.on_unit_nudge(1, 0, Qt.NoModifier) is True
        assert _positions(window, [edge]) == start
        assert _last_status_line(window) == f"{_unit_name(_PLACE_CONST)} already at the map edge; nothing moved"
    finally:
        _close(window)


def test_a_nudge_brings_an_off_map_member_back_without_raising() -> None:
    """The reported crash path: an entry the index patch left off-map moves
    again before any rebuild, and is pickable once back on the map."""
    window = _window()
    try:
        _free_tiles(window, [(119, 56), (30, 56)])
        keys = _add_units(window, [(_PLACE_CONST, 119.5, 56.5), (_PLACE_CONST, 30.5, 56.5)])
        index = window.map_view._unit_index
        stray = index.entry_for_key(keys[0])
        window._move_units(window._ensure_unit_edits(), [stray], 1.0, 0.0, "Premise")
        assert window.map_view._unit_index is index and stray.unit.x == 120.5
        _select_keys(window, keys)

        assert window.on_unit_nudge(-1, 0, Qt.ShiftModifier) is True
        assert _positions(window, keys) == [(119.5, 56.5), (29.5, 56.5)]
        entry = window.map_view._unit_index.entry_for_key(keys[0])
        assert entry.order in window.map_view._unit_index.by_tile[(119, 56)]
    finally:
        _close(window)


def test_a_z_step_moves_every_level_and_clamps_at_the_limit() -> None:
    window = _window()
    try:
        _reposition(window, _REF_ARCHER_P1, 11.5, 10.5, 13.0)
        entries = _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        spin = window.units_panel.unit_field_editors["z"]

        spin.stepBy(1)
        assert [(e.unit.x, e.unit.y, e.unit.z) for e in entries] == [(11.5, 10.5, 14.0), (20.5, 20.5, 1.0)]
        assert _last_status_line(window) == "Moved 2 units by +1 on Z"

        spin.stepBy(10)
        assert [e.unit.z for e in entries] == [15.0, 2.0]
        assert _last_status_line(window) == "Moved 2 units by +1 on Z, stopped at the Z limit"

        records_before = len(window.edit_history.records)
        spin.stepBy(1)
        assert [e.unit.z for e in entries] == [15.0, 2.0]
        assert len(window.edit_history.records) == records_before
        assert _last_status_line(window) == "2 units already at the Z limit; nothing moved"
    finally:
        _close(window)


def test_a_member_stored_above_the_z_limit_does_not_hold_the_rest_back() -> None:
    window = _window()
    try:
        _reposition(window, _REF_ARCHER_P1, 11.5, 10.5, 20.0)
        entries = _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        window.units_panel.unit_field_editors["z"].stepBy(2)
        assert [e.unit.z for e in entries] == [22.0, 2.0]
    finally:
        _close(window)


def test_a_step_carries_a_selected_hosts_hidden_garrison_on_x_and_z() -> None:
    window = _window()
    try:
        host, occupant = _fixture_host_and_occupant(window)
        _hide_garrisoned(window)
        other = next(u for u in window.scenario.unit_manager.units[2] if u.reference_id == _REF_ARCHER_P2)
        window._selection = [(1, host.reference_id), (2, other.reference_id)]
        window._refresh_selection_view()

        window.units_panel.unit_field_editors["x"].stepBy(1)
        assert (host.x, occupant.x, other.x) == (11.5, 11.5, 21.5)

        window.units_panel.unit_field_editors["z"].stepBy(1)
        assert (host.z, occupant.z, other.z) == (1.0, 1.0, 1.0)
        assert (occupant.x, occupant.y) == (host.x, host.y)
    finally:
        _close(window)


def test_a_group_step_holding_a_gate_takes_the_gate_plan() -> None:
    from test_wall_run_viewer import _NEW, _ROW, VILLAGER, _add_unit, _gate_window, _select_units, _walls_on
    from test_wall_run_viewer import _close as _close_wall_window

    window, gate = _gate_window()
    try:
        other = _add_unit(window, VILLAGER, 60.5, 60.5)
        _select_units(window, [gate, other])
        records = len(window.edit_history.records)

        window.units_panel.unit_field_editors["x"].stepBy(1)

        assert (gate.x, other.x) == (28.0, 61.5)
        assert _walls_on(window, {(29, _ROW)}) == []
        assert len(window.edit_history.records) == records + 1
        assert _last_status_line(window) == "Moved 2 units by +1 on X, replacing 1 wall, filling 1"

        window.undo()
        assert (gate.x, other.x) == (27.0, 60.5)
        assert len(_walls_on(window, _NEW)) == 4
    finally:
        _close_wall_window(window)


def test_the_pick_index_follows_a_group_step() -> None:
    window = _window()
    try:
        _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        spin = window.units_panel.unit_field_editors["y"]
        spin.stepBy(3)
        index = window.map_view._unit_index
        assert sorted(index.entry_for_key(key).own_y for key in window._selection) == [13, 23]
    finally:
        _close(window)


def test_a_stored_negative_zero_z_keeps_its_sign_through_a_step_and_a_nudge() -> None:
    import math

    window = _window()
    try:
        _reposition(window, _REF_ARCHER_P1, 11.5, 10.5, -0.0)
        entries = _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
        archer = entries[0].unit
        assert archer.reference_id == _REF_ARCHER_P1
        assert math.copysign(1.0, archer.z) == -1.0, "premise: -0.0 stored"

        window.units_panel.unit_field_editors["x"].stepBy(1)
        assert archer.x == 12.5
        assert math.copysign(1.0, archer.z) == -1.0

        window.on_unit_nudge(1, 0, Qt.NoModifier)
        assert math.copysign(1.0, archer.z) == -1.0
    finally:
        _close(window)


@pytest.mark.parametrize(
    ("values", "delta", "expected"),
    [
        pytest.param([0.0, 3.0], 2.0, (2.0, False), id="in-range"),
        pytest.param([0.0, 14.0], 5.0, (1.0, True), id="clamped-up"),
        pytest.param([1.0, 6.0], -3.0, (-1.0, True), id="clamped-down"),
        pytest.param([20.0, 2.0], 4.0, (4.0, False), id="outside-does-not-limit"),
        pytest.param([20.0, -3.0], 4.0, (4.0, False), id="nothing-in-range"),
        pytest.param([14.5], 1.0, (0.0, True), id="whole-steps-only"),
    ],
)
def test_clamp_level_delta(values, delta, expected) -> None:
    from descape.viewer_unit_inspector import clamp_level_delta

    assert clamp_level_delta(values, delta, 0.0, 15.0) == expected


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


def test_a_rotation_on_a_group_with_no_angle_member_writes_the_tree_and_leaves_the_wall() -> None:
    """GH #123: no ANGLE member makes the field a variant number, so the tree
    takes it and the wall, which the game re-derives, stays verbatim."""
    window = _window()
    try:
        entries = _select(window, _REF_TREE_OAK, _REF_WALL)
        by_ref = {e.unit.reference_id: e.unit for e in entries}
        wall_before = by_ref[_REF_WALL].rotation

        window._on_unit_field_changed(_field("rotation"), 1.0)

        assert by_ref[_REF_TREE_OAK].rotation == 1.0
        assert by_ref[_REF_WALL].rotation == wall_before
        assert "(1 skipped: not rotatable)" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_rotation_on_a_wall_only_group_records_nothing() -> None:
    window = _window()
    try:
        _select(window, _REF_WALL, _REF_HOUSE)
        window._on_unit_field_changed(_field("rotation"), 1.0)
        assert not window.edit_history.is_dirty
    finally:
        _close(window)
