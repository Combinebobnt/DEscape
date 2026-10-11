"""Track B Stage 1 + 2 of the 2026-09-02 cliffs plan: the Cliff tool's
Terrain-mode placement, driven through a real offscreen ViewerWindow -- same
style as test_unit_edit_viewer.py's b1.4 (Place Unit) tests, but through the
on_edit_stroke_start/tile/end trio, which is Cliff's own dispatch route: it
is a Terrain-mode tool, not a Units-mode one, so MapView never calls
on_unit_place for it.

Stage 2 widened the tool from click_only to a drag stroke, so a click is now
a one-tile stroke rather than an on_fill() dispatch. The placement assertions
below are unchanged by that, which is the check that the widening left
Stage 1's gesture alone: a lone node has no neighbours, so it takes the
picker's own piece and frame rather than the connectivity table's.

The chain resolution itself (lattice stepping, dirset lookup, face
inheritance) is tested headless in test_cliff_chain.py; what's here is the
wiring -- one undo record per stroke, and no stray EditHistory terrain
stroke opened alongside it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5.QtCore import Qt

from descape import cliff_catalog

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

# n_cliff_default_x1's own 01 piece: 3x3 span (A4's table), so span_anchor
# lands its centre at tile + 1.5 on both axes -- an easy value to assert on.
_DEFAULT_CLIFF = 264
GAIA_PLAYER_ID = 0


def _window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Terrain")
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _unit_count(window) -> int:
    return sum(len(u) for u in window.scenario.unit_manager.units)


def _stroke(window, tiles) -> None:
    """The three calls MapView makes for a press-drag-release. A single tile
    is a plain click -- MapView opens a stroke for a press even when the
    mouse never moves."""
    window.on_edit_stroke_start()
    for tile_x, tile_y in tiles:
        window.on_edit_stroke_tile(tile_x, tile_y, Qt.NoModifier)
    window.on_edit_stroke_end()


def _select_piece(window, unit_const: int) -> None:
    families = cliff_catalog.families()
    file_name = next(name for name, pieces in families.items() if any(p.unit_const == unit_const for p in pieces))
    family_index = window.cliff_family_combo.findData(file_name)
    assert family_index != -1
    window.cliff_family_combo.setCurrentIndex(family_index)
    piece_index = window.cliff_piece_combo.findData(unit_const)
    assert piece_index != -1
    window.cliff_piece_combo.setCurrentIndex(piece_index)


def test_cliff_button_only_shows_in_terrain_mode():
    """QAction.isVisible(), never the widget's -- offscreen unshown widgets
    report False unconditionally and would pass vacuously either way."""
    window = _window()
    try:
        assert window.cliff_action.isVisible()
        window.mode_combo.setCurrentText("Units")
        assert not window.cliff_action.isVisible()
    finally:
        _close(window)


def test_placing_a_cliff_anchors_on_the_clicked_tile_and_selects_it():
    window = _window()
    try:
        _select_piece(window, _DEFAULT_CLIFF)
        window.cliff_frame_spin.setValue(3)
        window.cliff_action.setChecked(True)
        before = _unit_count(window)
        before_records = len(window.edit_history.records)

        _stroke(window, [(10, 10)])

        assert _unit_count(window) == before + 1
        assert len(window.edit_history.records) == before_records + 1
        assert window.edit_history.peek_undo().kind == "unit"
        # Not window.map_view._unit_index: that pick index is only built in
        # Units mode (_after_unit_mutation()'s own mode gate), and Cliff is
        # Terrain-mode-only, so it stays None here -- read the placed unit
        # straight off the scenario instead.
        unit = window.scenario.unit_manager.units[GAIA_PLAYER_ID][-1]
        assert unit.unit_const == _DEFAULT_CLIFF
        assert unit.rotation == 3.0
        assert unit.initial_animation_frame == 3
        # 264 is 3x3 (odd, odd) -- span_anchor(10, 10, 3, 3) == (11.5, 11.5).
        assert (unit.x, unit.y) == (11.5, 11.5)

        window.undo()
        assert _unit_count(window) == before
    finally:
        _close(window)


def test_placing_with_no_piece_chosen_is_a_no_op():
    window = _window()
    try:
        window.cliff_piece_combo.clear()  # currentData() now None
        window.cliff_action.setChecked(True)
        before = _unit_count(window)
        _stroke(window, [(5, 5)])
        assert _unit_count(window) == before
        assert not window.edit_history.can_undo
    finally:
        _close(window)


def test_a_drag_places_a_connected_chain_as_one_undo_record():
    """The Convert brush's accumulate-then-commit shape: many placements,
    exactly one record, and the whole run undone together."""
    window = _window()
    try:
        _select_piece(window, _DEFAULT_CLIFF)
        window.cliff_action.setChecked(True)
        before = _unit_count(window)
        before_records = len(window.edit_history.records)

        _stroke(window, [(10, y) for y in range(10, 25)])

        placed = _unit_count(window) - before
        assert placed == 5  # a 3x3 piece stepping its own span over 15 tiles
        assert len(window.edit_history.records) == before_records + 1
        assert window.edit_history.peek_undo().kind == "unit"
        units = window.scenario.unit_manager.units[GAIA_PLAYER_ID][-placed:]
        assert [u.y for u in units] == [11.5, 14.5, 17.5, 20.5, 23.5]
        assert all(u.x == 11.5 for u in units)
        # A straight south run resolves to 3x3 s1 throughout (the {S} cap's
        # prior, then size-keeping pairs), with the rotations the measured
        # pairs give (test_cliff_chain.py's four-node run, one node longer).
        assert all(u.unit_const == _DEFAULT_CLIFF for u in units)
        assert [u.rotation for u in units] == [18.0, 2.0, 1.0, 2.0, 19.0]
        assert all(u.rotation == u.initial_animation_frame for u in units)

        window.undo()
        assert _unit_count(window) == before
    finally:
        _close(window)


def _mark_boxes(item, tile_px):
    """Each Flat footprint mark's tile box (x0, x1, y0, y1)."""
    out = []
    for polygon, _brush, _pen in item._marks:
        rect = polygon.boundingRect()
        out.append(tuple(round(v / tile_px) for v in (rect.left(), rect.right(), rect.top(), rect.bottom())))
    return sorted(out)


def test_a_drag_previews_the_run_release_places_and_release_clears_it():
    """GH #128's deferred half: mid-drag, the ghost is the whole run a
    release would place right now (Flat draws each node as its footprint
    mark), re-resolved per event; release places exactly that and clears it."""
    window = _window()
    try:
        _select_piece(window, _DEFAULT_CLIFF)
        window.cliff_action.setChecked(True)
        view = window.map_view
        window.on_edit_stroke_start()
        for y in range(10, 25):
            window.on_edit_stroke_tile(10, y, Qt.NoModifier)
        nodes = window._cliff_stroke.nodes()
        assert len(nodes) == 5
        assert _mark_boxes(view._unit_ghost_item, view._tile_pixels) == sorted(n.bounds for n in nodes)
        # A turn re-resolves the run: the vertical leg's pieces change size.
        for x in range(11, 20):
            window.on_edit_stroke_tile(x, 24, Qt.NoModifier)
        nodes = window._cliff_stroke.nodes()
        assert _mark_boxes(view._unit_ghost_item, view._tile_pixels) == sorted(n.bounds for n in nodes)
        window.on_edit_stroke_end()
        assert view._unit_ghost_item is None
        placed = window.scenario.unit_manager.units[GAIA_PLAYER_ID][-len(nodes):]
        assert [(u.unit_const, u.x, u.y) for u in placed] == [(n.unit_const, n.x, n.y) for n in nodes]
        last = window.status_log.toPlainText().splitlines()[-1]
        assert last.startswith(f"Placed a run of {len(nodes)} cliffs, piece sizes chosen per junction"), last
    finally:
        _close(window)


def _shown_window():
    from PyQt5.QtWidgets import QApplication

    window = _window()
    window.show()
    QApplication.processEvents()
    _select_piece(window, _DEFAULT_CLIFF)
    window.cliff_frame_spin.setValue(3)
    window.cliff_action.setChecked(True)
    assert window.map_view._tool == "cliff"
    return window


def _hover_tile(window, tile_x: int, tile_y: int) -> None:
    from PyQt5.QtCore import QEvent

    view = window.map_view
    pos = conftest.viewport_pos(view, tile_x, tile_y)
    view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, pos, Qt.NoButton, Qt.NoButton))


def test_hovering_ghosts_the_piece_a_click_places_and_a_tool_switch_clears_it():
    window = _shown_window()
    try:
        view = window.map_view
        _hover_tile(window, 30, 40)
        assert _mark_boxes(view._unit_ghost_item, view._tile_pixels) == [(30, 33, 40, 43)]
        # A picker change with no mouse move re-ghosts the new piece.
        _select_piece(window, cliff_catalog.piece_for_suffix(_DEFAULT_CLIFF, 2))
        assert _mark_boxes(view._unit_ghost_item, view._tile_pixels) == [(30, 31, 40, 43)]
        window.pan_action.setChecked(True)
        assert view._unit_ghost_item is None
        view.refresh_cliff_ghost()
        assert view._unit_ghost_item is None
    finally:
        _close(window)


def test_the_hover_ghost_is_gone_off_the_map_edge_and_on_leaving_the_view():
    """A piece that would not fit is not placed by a click, so it is not ghosted."""
    from PyQt5.QtCore import QEvent

    window = _shown_window()
    try:
        view = window.map_view
        _hover_tile(window, 118, 40)  # a 3x3 at x=118 overhangs a 120-wide map
        assert view._unit_ghost_item is None
        _hover_tile(window, 40, 40)
        assert view._unit_ghost_item is not None
        view.leaveEvent(QEvent(QEvent.Leave))
        assert view._unit_ghost_item is None
    finally:
        _close(window)


def test_a_cliff_stroke_opens_no_terrain_stroke_alongside_it():
    """The plan's own third Stage 1 invariant, live now that the tool drags:
    falling through to edit_history.begin_stroke() would leave a terrain
    stroke open and commit an empty one at release."""
    window = _window()
    try:
        _select_piece(window, _DEFAULT_CLIFF)
        window.cliff_action.setChecked(True)
        before_records = len(window.edit_history.records)
        _stroke(window, [(20, y) for y in range(20, 30)])
        assert len(window.edit_history.records) == before_records + 1
        assert window.edit_history.peek_undo().kind == "unit"
    finally:
        _close(window)
