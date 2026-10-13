"""GH #204: the Garrison block's Cut button.

Cut marks the selected occupant rows and edits nothing. Paste inside another
host then moves those same units there through ViewerWindow._garrison_units(),
the path a drop on a host takes, so each keeps its reference_id and the type
and capacity checks still run. The panel half is standalone, on
test_units_panel.py's convention; the viewer half drives a real offscreen
window on test_unit_edit_viewer.py's fixture and helpers.
"""

from __future__ import annotations

import pytest
import test_unit_edit_viewer as viewer_tests
import test_units_panel as panel_tests
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from test_unit_edit_viewer import _no_install_env  # noqa: F401 -- autouse fixture

from descape.units_panel import UnitsPanel

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_ROWS = [("Archer", "Player 1", 11), ("Villager", "Player 1", 12)]
_CUT_TIP = "Cut: Paste inside another host moves it there"
_GONE = "Paste: the cut units are no longer inside the host they were cut from"


# --- the panel ------------------------------------------------------------------


def _host_panel(**callbacks):
    panel = panel_tests._panel(**callbacks)
    panel.show_unit(panel_tests._entry(unit_const=panel_tests._TOWER_CONST, reference_id=10))
    return panel


def test_cut_sits_before_copy_and_reports_the_selected_rows() -> None:
    cut = []
    panel = _host_panel(on_garrison_cut=cut.append)
    panel.show_garrison(_ROWS, 5)
    layout = panel.garrison_button_rows[1].layout()
    assert [layout.itemAt(i).widget() for i in range(3)] == [
        panel.garrison_cut_button,
        panel.garrison_copy_button,
        panel.garrison_paste_button,
    ]
    assert panel.garrison_cut_button.text() == "Cut"
    assert not panel.garrison_cut_button.isEnabled()

    panel.garrison_tree.topLevelItem(1).setSelected(True)
    assert panel.garrison_cut_button.isEnabled()
    panel.garrison_cut_button.click()
    assert cut == [[12]]

    panel.garrison_tree.clearSelection()
    assert not panel.garrison_cut_button.isEnabled()


def test_show_garrison_draws_the_cut_rows_italic_and_dimmed() -> None:
    panel = _host_panel()
    panel.show_garrison(_ROWS, 5, cut=[12])
    plain, marked = (panel.garrison_tree.topLevelItem(i) for i in range(2))
    dimmed = panel.garrison_tree.palette().color(QPalette.Disabled, QPalette.Text)
    assert dimmed != panel.garrison_tree.palette().color(QPalette.Active, QPalette.Text)
    for column in range(2):
        assert marked.font(column).italic()
        assert marked.foreground(column).color() == dimmed
        assert marked.toolTip(column) == _CUT_TIP
        assert not plain.font(column).italic()
        assert plain.foreground(column).style() == Qt.NoBrush
        assert plain.toolTip(column) == ""

    panel.show_garrison(_ROWS, 5)
    assert not panel.garrison_tree.topLevelItem(1).font(0).italic(), "a refresh with no cut draws every row plain"


def test_set_garrison_cut_restyles_in_place_and_keeps_the_selection() -> None:
    panel = _host_panel()
    panel.show_garrison(_ROWS, 5)
    row = panel.garrison_tree.topLevelItem(0)
    row.setSelected(True)

    panel.set_garrison_cut([11])
    assert row.font(0).italic()
    assert not panel.garrison_tree.topLevelItem(1).font(0).italic()
    assert panel.garrison_selection() == [11]
    assert panel.garrison_cut_button.isEnabled()

    panel.set_garrison_cut(())
    assert not row.font(0).italic()
    assert row.foreground(0).style() == Qt.NoBrush
    assert row.toolTip(0) == ""
    assert panel.garrison_selection() == [11]


@pytest.mark.font_sensitive
def test_every_garrison_button_row_fits_beside_the_scroll_bar_at_the_floor() -> None:
    """test_units_panel.py's garrison floor check allows the whole floor; a
    row also shares it with the inspector's scroll bar and margins."""
    panel = _host_panel()
    panel.show_garrison(_ROWS, 5)
    area = panel.inspector_area
    margins = panel.inspector_layout.contentsMargins()
    room = (
        UnitsPanel.MIN_USEFUL_WIDTH - 2 * area.frameWidth() - area.verticalScrollBar().sizeHint().width()
        - margins.left() - margins.right()
    )
    for index, row in enumerate(panel.garrison_button_rows):
        width = row.minimumSizeHint().width()
        assert width <= room, f"garrison button row {index} demands {width} px of {room}"


# --- the viewer -----------------------------------------------------------------


def _select_host(window, player_id: int, host) -> None:
    window._selection = [(player_id, host.reference_id)]
    window._refresh_selection_view()


def _two_towers(window):
    """Player 1's tower holding an Archer and a Villager, hidden as the app
    ships, and Player 2's empty one. The first is selected."""
    tower, occupants = viewer_tests._tower_with_two_inside(window)
    other = viewer_tests._second_tower(window)
    viewer_tests._hide_garrisoned(window)
    _select_host(window, 1, tower)
    return tower, other, occupants


def _cut(window, reference_ids) -> None:
    viewer_tests._select_rows(window, set(reference_ids))
    window.units_panel.garrison_cut_button.click()


def _marked(window) -> list[int]:
    """The reference_ids of the rows the Garrison block draws as cut."""
    tree = window.units_panel.garrison_tree
    rows = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    return [row.data(0, Qt.UserRole) for row in rows if row.font(0).italic()]


def _inside(window, host) -> list:
    return [u for units in window.scenario.unit_manager.units for u in units if u.garrisoned_in_id == host.reference_id]


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def test_cut_marks_the_rows_and_edits_nothing() -> None:
    window = viewer_tests._window()
    try:
        tower, _other, occupants = _two_towers(window)
        panel = window.units_panel
        ref = occupants[0].reference_id
        records_before = len(window.edit_history.records)
        assert not panel.garrison_paste_button.isEnabled()

        _cut(window, [ref])

        assert _marked(window) == [ref]
        assert panel.garrison_selection() == [ref], "Cut must leave the cut rows selected"
        assert all(u.garrisoned_in_id == tower.reference_id for u in occupants)
        assert len(window.edit_history.records) == records_before
        assert panel.garrison_paste_button.isEnabled()
        assert _last_status(window) == "Cut 1 garrisoned unit(s); Paste inside another host moves them"
    finally:
        viewer_tests._close(window)


def test_cut_then_paste_moves_the_same_units_in_one_undo_step() -> None:
    window = viewer_tests._window()
    try:
        tower, occupants = viewer_tests._tower_with_two_inside(window)
        archer = viewer_tests._live(window, viewer_tests._REF_ARCHER_P2)
        host = viewer_tests._entry(window, 1, tower.reference_id)
        assert window._garrison_units(host, [viewer_tests._entry(window, 2, viewer_tests._REF_ARCHER_P2)])
        other = viewer_tests._second_tower(window)
        viewer_tests._hide_garrisoned(window)
        _select_host(window, 1, tower)
        panel = window.units_panel
        moving = [occupants[0], archer]
        refs = [u.reference_id for u in moving]
        owners = [viewer_tests._owner_of(window, ref) for ref in refs]
        assert owners == [1, 2]
        count = viewer_tests._unit_count(window)

        _cut(window, refs)
        _select_host(window, 2, other)
        assert panel.garrison_paste_button.isEnabled()
        assert _marked(window) == []
        records_before = len(window.edit_history.records)
        panel.garrison_paste_button.click()

        assert [id(u) for u in _inside(window, other)] == [id(u) for u in moving]
        assert [u.reference_id for u in moving] == refs
        assert [viewer_tests._owner_of(window, ref) for ref in refs] == owners
        assert all((u.x, u.y, u.z) == (other.x, other.y, other.z) for u in moving)
        assert [id(u) for u in _inside(window, tower)] == [id(occupants[1])]
        assert viewer_tests._unit_count(window) == count, "a move, never a copy"
        assert all(viewer_tests._entry(window, owner, ref) is None for owner, ref in zip(owners, refs, strict=True))
        assert len(window.edit_history.records) == records_before + 1
        assert window.edit_history.peek_undo().label == "Move into garrison"
        assert _last_status(window) == "Moved 2 unit(s) inside Watch Tower"
        assert panel.garrison_tree.topLevelItemCount() == 2
        assert _marked(window) == []
        assert not panel.garrison_paste_button.isEnabled(), "a pasted cut leaves nothing to paste"

        window.undo()
        assert sorted(id(u) for u in _inside(window, tower)) == sorted(id(u) for u in [*occupants, archer])
        assert [u.reference_id for u in moving] == refs
        assert all((u.x, u.y, u.z) == (tower.x, tower.y, tower.z) for u in moving)
        _select_host(window, 1, tower)
        assert _marked(window) == [], "undo puts the units back, not the cut"
        assert not panel.garrison_paste_button.isEnabled()

        window.redo()
        assert [id(u) for u in _inside(window, other)] == [id(u) for u in moving]
    finally:
        viewer_tests._close(window)


def test_a_pasted_cut_is_saved_as_the_same_units_inside_the_new_host(tmp_path) -> None:
    from descape.scenario_io import load_map_and_units

    window = viewer_tests._window()
    try:
        tower, other, occupants = _two_towers(window)
        refs = [u.reference_id for u in occupants]
        count = viewer_tests._unit_count(window)
        _cut(window, refs)
        _select_host(window, 2, other)
        window.units_panel.garrison_paste_button.click()

        out = tmp_path / "out.aoe2scenario"
        window.scenario.path = out
        window._untitled = False
        window.save()

        saved = [u for units in load_map_and_units(out).unit_manager.units for u in units]
        assert len(saved) == count
        by_ref = {u.reference_id: u for u in saved}
        assert len(by_ref) == count, "no reference_id was duplicated or lost"
        for ref, unit in zip(refs, occupants, strict=True):
            assert by_ref[ref].garrisoned_in_id == other.reference_id
            assert by_ref[ref].unit_const == unit.unit_const
            assert (by_ref[ref].x, by_ref[ref].y) == (other.x, other.y)
        assert not any(u.garrisoned_in_id == tower.reference_id for u in saved)
    finally:
        viewer_tests._close(window)


def test_a_cut_paste_with_show_garrisoned_on_moves_the_pick_index_entries() -> None:
    window = viewer_tests._window()
    try:
        tower, other, occupants = _two_towers(window)
        viewer_tests._show_garrisoned(window)
        _select_host(window, 1, tower)
        ref = occupants[0].reference_id
        entries_before = len(window.map_view._unit_index.entries)

        _cut(window, [ref])
        _select_host(window, 2, other)
        window.units_panel.garrison_paste_button.click()

        moved = viewer_tests._entry(window, 1, ref)
        assert (moved.own_x, moved.own_y) == (int(other.x), int(other.y))
        assert len(window.map_view._unit_index.entries) == entries_before
    finally:
        viewer_tests._close(window)


def test_a_cut_paste_into_a_host_without_the_room_is_refused_whole_and_keeps_the_cut() -> None:
    window = viewer_tests._window()
    try:
        tower, other, occupants = _two_towers(window)
        for _ in range(4):
            window._garrison_add_const(viewer_tests._entry(window, 2, other.reference_id), 4)
        refs = [u.reference_id for u in occupants]
        panel = window.units_panel

        _cut(window, refs)
        _select_host(window, 2, other)
        records_before = len(window.edit_history.records)
        panel.garrison_paste_button.click()

        assert all(u.garrisoned_in_id == tower.reference_id for u in occupants)
        assert len(window.edit_history.records) == records_before
        assert _last_status(window) == "Garrison: Watch Tower has room for 1 more (5 places)"
        assert panel.garrison_paste_button.isEnabled(), "a refused Paste keeps the cut for another host"
        _select_host(window, 1, tower)
        assert sorted(_marked(window)) == sorted(refs)
    finally:
        viewer_tests._close(window)


def test_a_cut_paste_into_a_host_that_cannot_take_the_type_is_refused() -> None:
    """A Dock holds ships and nothing else."""
    window = viewer_tests._window()
    try:
        tower, _other, occupants = _two_towers(window)
        model = window._ensure_unit_edits()
        tile = viewer_tests._empty_tile(window)
        with window._unit_edit(model, "Place dock", [2]):
            dock = model.add(2, 45, tile[0] + 0.5, tile[1] + 0.5)
        window._rebuild_unit_index()
        _select_host(window, 1, tower)

        _cut(window, [occupants[0].reference_id])
        _select_host(window, 2, dock)
        assert window.units_panel.garrison_paste_button.isEnabled()
        records_before = len(window.edit_history.records)
        window.units_panel.garrison_paste_button.click()

        assert occupants[0].garrisoned_in_id == tower.reference_id
        assert len(window.edit_history.records) == records_before
        assert _last_status(window) == "Garrison: Archer cannot go inside Dock"
    finally:
        viewer_tests._close(window)


def test_paste_into_the_host_the_rows_were_cut_from_changes_nothing_and_keeps_the_cut() -> None:
    window = viewer_tests._window()
    try:
        tower, _other, occupants = _two_towers(window)
        refs = [u.reference_id for u in occupants]
        panel = window.units_panel

        _cut(window, refs)
        records_before = len(window.edit_history.records)
        panel.garrison_paste_button.click()

        assert all(u.garrisoned_in_id == tower.reference_id for u in occupants)
        assert len(window.edit_history.records) == records_before
        assert _last_status(window) == "Garrison: already inside Watch Tower"
        assert sorted(_marked(window)) == sorted(refs)
        assert panel.garrison_paste_button.isEnabled()
    finally:
        viewer_tests._close(window)


def test_a_copy_replaces_a_pending_cut() -> None:
    window = viewer_tests._window()
    try:
        tower, other, occupants = _two_towers(window)
        refs = [u.reference_id for u in occupants]
        panel = window.units_panel

        _cut(window, refs)
        viewer_tests._select_rows(window, {refs[0]})
        panel.garrison_copy_button.click()
        assert _marked(window) == []

        count = viewer_tests._unit_count(window)
        _select_host(window, 2, other)
        panel.garrison_paste_button.click()

        assert all(u.garrisoned_in_id == tower.reference_id for u in occupants), "the cut rows never moved"
        assert viewer_tests._unit_count(window) == count + 1
        assert [u.unit_const for u in _inside(window, other)] == [occupants[0].unit_const]
        assert panel.garrison_paste_button.isEnabled(), "copies paste again"
    finally:
        viewer_tests._close(window)


def test_a_cut_replaces_the_copied_rows() -> None:
    window = viewer_tests._window()
    try:
        tower, other, occupants = _two_towers(window)
        refs = [u.reference_id for u in occupants]
        panel = window.units_panel

        viewer_tests._select_rows(window, {refs[0]})
        panel.garrison_copy_button.click()
        _cut(window, [refs[1]])
        count = viewer_tests._unit_count(window)
        _select_host(window, 2, other)
        panel.garrison_paste_button.click()

        assert [id(u) for u in _inside(window, other)] == [id(occupants[1])]
        assert viewer_tests._unit_count(window) == count, "a move: nothing was added"
        _select_host(window, 1, tower)
        assert not panel.garrison_paste_button.isEnabled(), "the copies went when the Cut was made"
    finally:
        viewer_tests._close(window)


@pytest.mark.parametrize("leave", ["unload", "delete"])
def test_a_cut_row_that_left_its_host_is_skipped_by_the_paste(leave: str) -> None:
    window = viewer_tests._window()
    try:
        tower, other, occupants = _two_towers(window)
        refs = [u.reference_id for u in occupants]
        panel = window.units_panel

        _cut(window, refs)
        viewer_tests._select_rows(window, {refs[0]})
        (panel.garrison_unload_button if leave == "unload" else panel.garrison_delete_button).click()
        assert _marked(window) == [refs[1]]

        _select_host(window, 2, other)
        records_before = len(window.edit_history.records)
        panel.garrison_paste_button.click()

        assert [id(u) for u in _inside(window, other)] == [id(occupants[1])]
        assert occupants[0].garrisoned_in_id == (-1 if leave == "unload" else tower.reference_id)
        live = {id(u) for units in window.scenario.unit_manager.units for u in units}
        assert (id(occupants[0]) in live) == (leave == "unload")
        assert len(window.edit_history.records) == records_before + 1
        assert _last_status(window) == "Moved 1 unit(s) inside Watch Tower"
    finally:
        viewer_tests._close(window)


def test_undoing_the_unload_of_a_cut_row_marks_it_again() -> None:
    window = viewer_tests._window()
    try:
        _tower, _other, occupants = _two_towers(window)
        refs = [u.reference_id for u in occupants]

        _cut(window, refs)
        viewer_tests._select_rows(window, {refs[0]})
        window.units_panel.garrison_unload_button.click()
        assert _marked(window) == [refs[1]]

        window.undo()
        assert sorted(_marked(window)) == sorted(refs)
    finally:
        viewer_tests._close(window)


def test_a_cut_row_moved_into_another_host_is_no_longer_cut() -> None:
    """With Show Garrisoned Units on, a cut occupant can be dropped on another host."""
    window = viewer_tests._window()
    try:
        _tower, other, occupants = _two_towers(window)
        viewer_tests._show_garrisoned(window)
        ref = occupants[0].reference_id
        panel = window.units_panel

        _cut(window, [ref])
        assert window._garrison_units(viewer_tests._entry(window, 2, other.reference_id), [viewer_tests._entry(window, 1, ref)])
        _select_host(window, 2, other)
        assert panel.garrison_tree.topLevelItemCount() == 1
        assert _marked(window) == []

        records_before = len(window.edit_history.records)
        panel.garrison_paste_button.click()
        assert len(window.edit_history.records) == records_before
        assert _last_status(window) == _GONE
    finally:
        viewer_tests._close(window)


def test_a_paste_whose_cut_rows_are_all_gone_says_so_and_clears_the_cut() -> None:
    window = viewer_tests._window()
    try:
        tower, other, occupants = _two_towers(window)
        refs = [u.reference_id for u in occupants]
        panel = window.units_panel

        _cut(window, refs)
        panel.garrison_delete_button.click()
        _select_host(window, 2, other)
        assert panel.garrison_paste_button.isEnabled()
        records_before = len(window.edit_history.records)
        panel.garrison_paste_button.click()

        assert len(window.edit_history.records) == records_before
        assert _last_status(window) == _GONE
        assert not panel.garrison_paste_button.isEnabled()
        _select_host(window, 1, tower)
        assert not panel.garrison_paste_button.isEnabled(), "the cut is dropped, not just greyed here"
    finally:
        viewer_tests._close(window)


def test_a_cut_does_not_outlive_its_document() -> None:
    """The reloaded fixture hands out the same reference_ids again, so only
    the document check keeps the old cut off the new file's units."""
    window = viewer_tests._window()
    try:
        tower, _other, occupants = _two_towers(window)
        old_refs = (tower.reference_id, [u.reference_id for u in occupants])
        _cut(window, old_refs[1])

        window.edit_history.mark_saved()
        window.load_scenario(viewer_tests.FIXTURE_PATH)
        window.mode_combo.setCurrentText("Units")
        tower, _other, occupants = _two_towers(window)
        assert (tower.reference_id, [u.reference_id for u in occupants]) == old_refs

        assert _marked(window) == []
        assert not window.units_panel.garrison_paste_button.isEnabled()
    finally:
        viewer_tests._close(window)
