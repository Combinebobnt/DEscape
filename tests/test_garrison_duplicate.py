"""TASK-549: the Garrison block's Duplicate button.

Duplicate adds copies of the selected rows inside the same host, through
Paste's own checks and writes (ViewerWindow._add_garrison_copies()), and
leaves the garrison clipboard and a pending Cut alone. Driven on
test_unit_edit_viewer.py's fixture and helpers, as test_garrison_cut.py is.
"""

from __future__ import annotations

import pytest
import test_unit_edit_viewer as viewer_tests
from test_unit_edit_viewer import _no_install_env  # noqa: F401 -- autouse fixture

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _v159_window():
    """The 1.59 fixture, so the capture flag is a real field."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(viewer_tests._V159_FIXTURE)
    assert window.scenario is not None and window.scenario.scenario_version == "1.59"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Units")
    return window


def _fields(player, unit):
    from descape.unit_model import capture_flag_of

    return (player, unit.unit_const, unit.rotation, unit.status,
            unit.caption_string_id, unit.caption_string, capture_flag_of(unit))


def _inside(window, host) -> list:
    return [
        (p, u)
        for p, units in enumerate(window.scenario.unit_manager.units)
        for u in units
        if u.garrisoned_in_id == host.reference_id
    ]


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _tower_with_two_owners(window):
    model = window._ensure_unit_edits()
    tile = viewer_tests._empty_tile(window)
    with window._unit_edit(model, "Place tower", [1, 2]):
        tower = model.add(1, 79, tile[0] + 0.5, tile[1] + 0.5)
        inside = [
            model.add(1, 4, tower.x, tower.y, rotation=1.25, status=4, garrisoned_in_id=tower.reference_id,
                      caption_string_id=7, caption_string="First", capture_flag=1),
            model.add(2, 83, tower.x, tower.y, rotation=2.5, status=3, garrisoned_in_id=tower.reference_id,
                      caption_string_id=9, caption_string="Second", capture_flag=3),
        ]
    window._rebuild_unit_index()
    window._selection = [(1, tower.reference_id)]
    window._refresh_selection_view()
    return tower, inside


def test_duplicate_copies_the_selected_rows_inside_the_same_host_in_one_record() -> None:
    window = _v159_window()
    try:
        tower, inside = _tower_with_two_owners(window)
        source = sorted(_fields(p, u) for p, u in _inside(window, tower))
        records = len(window.edit_history.records)
        viewer_tests._select_rows(window, {u.reference_id for u in inside})
        window.units_panel.garrison_duplicate_button.click()

        rows = _inside(window, tower)
        assert len(rows) == 4
        assert sorted(_fields(p, u) for p, u in rows) == sorted(source + source)
        new_refs = {u.reference_id for _p, u in rows} - {u.reference_id for u in inside}
        assert len(new_refs) == 2, "copies are new units"
        assert all((u.x, u.y) == (tower.x, tower.y) for _p, u in rows)
        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.peek_undo().label == "Duplicate in garrison"
        assert _last_status(window) == "Duplicated 2 unit(s) inside Watch Tower"
        assert window.units_panel.garrison_tree.topLevelItemCount() == 4

        window.undo()
        assert len(_inside(window, tower)) == 2
    finally:
        viewer_tests._close(window)


def test_duplicate_at_capacity_changes_nothing_and_says_why() -> None:
    window = viewer_tests._window()
    try:
        tower, occupants = viewer_tests._tower_with_two_inside(window)
        host = viewer_tests._entry(window, 1, tower.reference_id)
        for _ in range(3):
            window._garrison_add_const(host, 4)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        records = len(window.edit_history.records)
        count = viewer_tests._unit_count(window)
        viewer_tests._select_rows(window, {occupants[0].reference_id})

        window.units_panel.garrison_duplicate_button.click()

        assert viewer_tests._unit_count(window) == count
        assert len(window.edit_history.records) == records
        assert _last_status(window) == "Garrison: Watch Tower is full (5 places)"
    finally:
        viewer_tests._close(window)


def test_duplicate_leaves_the_garrison_clipboard_and_a_pending_cut() -> None:
    window = viewer_tests._window()
    try:
        tower, occupants = viewer_tests._tower_with_two_inside(window)
        viewer_tests._hide_garrisoned(window)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        panel = window.units_panel

        viewer_tests._select_rows(window, {occupants[0].reference_id})
        panel.garrison_copy_button.click()
        clipboard = window._garrison_clipboard
        viewer_tests._select_rows(window, {occupants[1].reference_id})
        panel.garrison_duplicate_button.click()
        assert window._garrison_clipboard == clipboard

        viewer_tests._select_rows(window, {occupants[0].reference_id})
        panel.garrison_cut_button.click()
        cut = window._garrison_cut
        assert cut is not None
        viewer_tests._select_rows(window, {occupants[1].reference_id})
        panel.garrison_duplicate_button.click()
        assert window._garrison_cut == cut
        assert panel.garrison_paste_button.isEnabled()
    finally:
        viewer_tests._close(window)


def test_duplicate_needs_a_selected_row() -> None:
    window = viewer_tests._window()
    try:
        tower, occupants = viewer_tests._tower_with_two_inside(window)
        window._selection = [(1, tower.reference_id)]
        window._refresh_selection_view()
        button = window.units_panel.garrison_duplicate_button
        assert button.text() == "Duplicate"
        assert button.toolTip() == "Copy the selected rows inside this same host"
        assert not button.isEnabled()
        viewer_tests._select_rows(window, {occupants[0].reference_id})
        assert button.isEnabled()
    finally:
        viewer_tests._close(window)
