"""Track B Stage 1 + 2 of the 2026-09-02 cliffs plan: the Cliff tool's
Terrain-mode placement, driven through a real offscreen ViewerWindow -- same
style as test_unit_edit_viewer.py's b1.4 (Place Unit) tests, but through the
on_edit_stroke_start/tile/end trio, which is Cliff's own dispatch route: it
is a Terrain-mode tool, not a Units-mode one, so MapView never calls
on_unit_place for it.

Stage 2 widened the tool from click_only to a drag stroke, so a click is now
a one-tile stroke rather than an on_fill() dispatch. A lone node away from
other cliffs still takes the picker's own piece and frame, centred on the
clicked tile; one beside an existing cliff is shaped by that contact.

The chain resolution itself (cardinal steps, the facing the drag direction
picks, the shape table lookup and the measured offsets) is tested headless
in test_cliff_chain.py; what's here is the wiring: one undo record per
stroke and its joins, the hover and drag previews, and no stray
EditHistory terrain stroke opened alongside it.
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

# n_cliff_default_x1's own 01 piece: 3x3 span (A4's table), centred on the
# clicked tile, so its anchor is tile + 0.5 on both axes.
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
        # 264 is 3x3, centred on the clicked tile: span_anchor(9, 9, 3, 3).
        assert (unit.x, unit.y) == (10.5, 10.5)

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

        _stroke(window, [(10, y) for y in range(10, 24)])

        placed = _unit_count(window) - before
        assert placed == 5  # a 3x3 piece stepping its own span over 14 tiles
        assert len(window.edit_history.records) == before_records + 1
        assert window.edit_history.peek_undo().kind == "unit"
        units = window.scenario.unit_manager.units[GAIA_PLAYER_ID][-placed:]
        assert [u.y for u in units] == [10.5, 13.5, 16.5, 19.5, 22.5]
        assert all(u.x == 10.5 for u in units)
        # Drawn south the run is front-facing: 3x3 s1 throughout, the tall
        # south and north caps (18, 19) around the straight's twins, picked
        # by footprint position (test_cliff_chain.py's straights).
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
        for y in range(10, 24):
            window.on_edit_stroke_tile(10, y, Qt.NoModifier)
        nodes = window._cliff_stroke.nodes()
        assert len(nodes) == 5
        assert _mark_boxes(view._unit_ghost_item, view._tile_pixels) == sorted(n.bounds for n in nodes)
        # A turn makes the run's old end a corner and adds the next leg.
        for x in range(11, 20):
            window.on_edit_stroke_tile(x, 23, Qt.NoModifier)
        nodes = window._cliff_stroke.nodes()
        assert _mark_boxes(view._unit_ghost_item, view._tile_pixels) == sorted(n.bounds for n in nodes)
        window.on_edit_stroke_end()
        assert view._unit_ghost_item is None
        placed = window.scenario.unit_manager.units[GAIA_PLAYER_ID][-len(nodes):]
        assert [(u.unit_const, u.x, u.y) for u in placed] == [(n.unit_const, n.x, n.y) for n in nodes]
        last = window.status_log.toPlainText().splitlines()[-1]
        assert last == (
            f"Placed a run of {len(nodes)} cliffs from the picked family, shaped by the drag direction and their "
            "neighbours"
        ), last
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


def _hover_within(window, tile_x: int, tile_y: int, fx: float, fy: float) -> None:
    """A hover at fraction `(fx, fy)` across the tile, (0.5, 0.5) being its
    centre: another mouse position inside the same tile."""
    from PyQt5.QtCore import QEvent, QPointF

    view = window.map_view
    tile_px = view._tile_pixels
    pos = QPointF(view.mapFromScene(QPointF((tile_x + fx) * tile_px, (tile_y + fy) * tile_px)))
    view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, pos, Qt.NoButton, Qt.NoButton))
    assert view._cliff_hover == (tile_x, tile_y), "the offset must stay inside the tile"


def _box_tiles(x0: int, x1: int, y0: int, y1: int) -> tuple:
    return tuple((x, y) for x in range(x0, x1) for y in range(y0, y1))


def _highlight_box(view):
    """The edit highlight's tile box (x0, x1, y0, y1) and how many tile
    outlines it holds, read off the painted path (Flat)."""
    from PyQt5.QtGui import QPainterPath

    item = view._highlight_outline_item
    if item is None:
        return None
    path = item.path()
    rect, tp = path.boundingRect(), view._tile_pixels
    tiles = sum(1 for i in range(path.elementCount()) if path.elementAt(i).type == QPainterPath.MoveToElement)
    return tuple(round(v / tp) for v in (rect.left(), rect.right(), rect.top(), rect.bottom())), tiles


def test_hovering_ghosts_the_piece_a_click_places_and_a_tool_switch_clears_it():
    window = _shown_window()
    try:
        view = window.map_view
        _hover_tile(window, 30, 40)
        # Centred on the hovered tile, where a click places it.
        assert _mark_boxes(view._unit_ghost_item, view._tile_pixels) == [(29, 32, 39, 42)]
        # A picker change with no mouse move re-ghosts the new piece.
        _select_piece(window, cliff_catalog.piece_for_suffix(_DEFAULT_CLIFF, 2))
        assert _mark_boxes(view._unit_ghost_item, view._tile_pixels) == [(30, 31, 39, 42)]
        assert _highlight_box(view) == ((30, 31, 39, 42), 3)
        window.pan_action.setChecked(True)
        assert view._unit_ghost_item is None
        assert view._cliff_footprint is None
        view.refresh_cliff_ghost()
        assert view._unit_ghost_item is None
    finally:
        _close(window)


def test_the_hover_highlight_is_the_piece_s_footprint_under_the_cursor():
    """GH #44 round 2: the gold highlight covers the nine tiles a 3x3 would
    land on, not the one cursor tile, and a second move inside the same tile
    (which repaints the highlight per pixel) leaves it and the ghost alone."""
    window = _shown_window()
    try:
        view = window.map_view
        _hover_tile(window, 30, 40)
        ghost = view._unit_ghost_item
        assert _mark_boxes(ghost, view._tile_pixels) == [(29, 32, 39, 42)]
        assert view._cliff_footprint == _box_tiles(29, 32, 39, 42)
        assert _highlight_box(view) == ((29, 32, 39, 42), 9)

        _hover_within(window, 30, 40, 0.65, 0.35)
        assert _highlight_box(view) == ((29, 32, 39, 42), 9)
        assert view._unit_ghost_item is ghost
        # The brush widgets' own rebuild (a size or shape change with no move) keeps it too.
        view._clear_highlight()
        view.refresh_highlight((30, 40))
        assert _highlight_box(view) == ((29, 32, 39, 42), 9)

        _hover_tile(window, 31, 40)
        assert _highlight_box(view) == ((30, 33, 39, 42), 9)
        # Another tool's hover is the plain brush diamond again.
        window.draw_action.setChecked(True)
        window.brush_size_spin.setValue(1)
        _hover_tile(window, 31, 40)
        assert _highlight_box(view) == ((31, 32, 40, 41), 1)
    finally:
        _close(window)


def test_mid_drag_the_highlight_stays_on_the_run_s_head():
    """Each move repaints the highlight at the cursor tile, and a revisited
    tile sends no stroke event: neither may put the one-tile diamond back."""
    window = _shown_window()
    try:
        view = window.map_view
        _hover_tile(window, 30, 40)
        _mouse_press(window, (30, 40))
        assert _highlight_box(view) == ((29, 32, 39, 42), 9)
        _mouse_move_held(window, (30, 43))
        head = window._cliff_stroke.head_bounds()
        assert head == (29, 32, 42, 45)
        assert _highlight_box(view) == (head, 9)
        # Back over a tile the drag already crossed, then onto another one.
        _mouse_move_held(window, (30, 41))
        assert _highlight_box(view) == (head, 9)
        _mouse_move_held(window, (31, 41))
        assert _highlight_box(view) == (head, 9)
        _mouse_release(window, (31, 41))
        # Released over the run just placed: a press there would continue it, a click places nothing.
        assert view._unit_ghost_item is None
        assert _highlight_box(view) == ((31, 32, 41, 42), 1)
    finally:
        _close(window)


def test_a_hover_at_the_map_edge_ghosts_the_clamped_piece():
    """A 3x3 hovered at x=119 on a 120-wide map is clamped onto it, as the
    click is, and its highlight with it."""
    window = _shown_window()
    try:
        view = window.map_view
        _hover_tile(window, 119, 40)
        assert _mark_boxes(view._unit_ghost_item, view._tile_pixels) == [(117, 120, 39, 42)]
        assert _highlight_box(view) == ((117, 120, 39, 42), 9)
        before = _unit_count(window)
        _mouse_drag(window, [(119, 40)])
        assert _unit_count(window) == before + 1
        placed = _gaia(window)[-1]
        assert (placed.x, placed.y) == (118.5, 40.5)
    finally:
        _close(window)


def _thin_piece_at_10_7(window):
    """A lone 1x3 on (10,11,7,10), placed by a click at its centre tile."""
    _select_piece(window, cliff_catalog.piece_for_suffix(_DEFAULT_CLIFF, 2))
    _stroke(window, [(10, 8)])
    thin = _gaia(window)[-1]
    assert (thin.x, thin.y, thin.rotation) == (10.5, 8.5, 3.0)
    _select_piece(window, _DEFAULT_CLIFF)
    return thin


def test_hover_beside_an_existing_cliff_is_what_the_click_places():
    """The hover used to ghost the picked piece regardless of the cliffs
    around it, while the click resolved against them. With a 1x3 on
    (10,11,7,10) and the 3x3 picked, both now show and place one 1x3 flush
    under it, and the old piece becomes a south-running cap (frame 22)."""
    window = _shown_window()
    try:
        view = window.map_view
        thin = _thin_piece_at_10_7(window)
        _hover_tile(window, 10, 10)
        # The new piece, and the old one's ghost at its new frame.
        assert _mark_boxes(view._unit_ghost_item, view._tile_pixels) == [(10, 11, 7, 10), (10, 11, 10, 13)]
        assert _highlight_box(view) == ((10, 11, 10, 13), 3)
        before = _unit_count(window)

        _mouse_drag(window, [(10, 10)])

        assert _unit_count(window) == before + 1, "the click must place a cliff"
        placed = _gaia(window)[-1]
        assert (placed.unit_const, placed.x, placed.y, placed.rotation) == (
            cliff_catalog.piece_for_suffix(_DEFAULT_CLIFF, 2), 10.5, 11.5, 23.0,
        )
        assert (thin.rotation, thin.initial_animation_frame) == (22.0, 22)
        assert _last_status(window) == (
            "Placed a cliff from the picked family, shaped to join the cliff beside it, "
            "1 existing cliff reshaped to join"
        )
    finally:
        _close(window)


def test_hover_on_an_existing_cliff_shows_nothing_and_builds_no_stroke_per_pixel(monkeypatch):
    from descape import cliff_chain

    window = _shown_window()
    try:
        view = window.map_view
        _thin_piece_at_10_7(window)
        built = []
        real = cliff_chain.ChainStroke
        monkeypatch.setattr(cliff_chain, "ChainStroke", lambda **kwargs: built.append(1) or real(**kwargs))
        _hover_tile(window, 10, 8)
        assert view._unit_ghost_item is None
        assert view._cliff_footprint is None
        # Nothing placeable: the cursor tile, as for any brush.
        assert _highlight_box(view) == ((10, 11, 8, 9), 1)
        assert len(built) == 1
        _hover_within(window, 10, 8, 0.65, 0.65)
        _hover_within(window, 10, 8, 0.35, 0.6)
        assert len(built) == 1, "the empty result is memoized too"
        _hover_tile(window, 10, 9)
        assert len(built) == 2 and view._unit_ghost_item is None
    finally:
        _close(window)


def test_the_ghost_returns_at_the_cursor_after_an_undo_without_a_mouse_move():
    """A drag anchored on an end cap, released over the run it just laid:
    nothing to ghost there. Undo takes the run away again, and the hover
    ghost is back under the cursor with no move."""
    window = _shown_window()
    try:
        view = window.map_view
        cap = _tall_run(window)
        _hover_tile(window, 41, 41)
        assert view._unit_ghost_item is None  # on the cap
        _mouse_drag(window, [(41, 41), (41, 44), (41, 46)])
        assert (cap.rotation, [(u.x, u.y) for u in _gaia(window)[-2:]]) == (1.0, [(41.5, 44.5), (41.5, 47.5)])
        assert view._unit_ghost_item is None
        window.undo()
        assert cap.rotation == 19.0
        assert _mark_boxes(view._unit_ghost_item, view._tile_pixels) == [(40, 43, 45, 48)]
        assert _highlight_box(view) == ((40, 43, 45, 48), 9)
    finally:
        _close(window)


def test_the_ghost_returns_at_the_cursor_after_a_history_window_jump():
    """TASK-358: the History window's jump moves the cursor without
    _move_history(), so it refreshes the Cliff ghost itself."""
    window = _shown_window()
    try:
        view = window.map_view
        cap = _tall_run(window)
        _hover_tile(window, 41, 41)
        _mouse_drag(window, [(41, 41), (41, 44), (41, 46)])
        assert view._unit_ghost_item is None
        window._on_history_jump(window.edit_history.cursor - 1)
        assert cap.rotation == 19.0
        assert _mark_boxes(view._unit_ghost_item, view._tile_pixels) == [(40, 43, 45, 48)]
        assert _highlight_box(view) == ((40, 43, 45, 48), 9)
    finally:
        _close(window)


def test_the_map_s_cliffs_are_scanned_once_per_unit_change(monkeypatch):
    """Every hovered tile builds a one-tile stroke; the cliff scan and its
    tile index behind them are per unit_gen, and a press reuses them."""
    from descape import cliff_chain

    window = _shown_window()
    try:
        scans = []
        real = cliff_chain.existing_cliffs
        monkeypatch.setattr(cliff_chain, "existing_cliffs", lambda *args: scans.append(1) or real(*args))
        for tile in ((30, 40), (31, 40), (50, 60)):
            _hover_tile(window, *tile)
        assert len(scans) == 1
        _mouse_press(window, (50, 60))
        assert len(scans) == 1, "the press reuses the hover's scan"
        _mouse_release(window, (50, 60))
        # The placed cliff changed the units: the release's own re-preview rescans, once.
        assert len(scans) == 2
        _hover_tile(window, 70, 70)
        _hover_tile(window, 71, 70)
        assert len(scans) == 2
        assert [c.bounds for c in window._cliff_existing()[0]] == [(49, 52, 59, 62)]
    finally:
        _close(window)


def test_the_hover_s_ghost_parts_start_over_when_the_look_changes():
    window = _shown_window()
    try:
        _hover_tile(window, 30, 40)
        _hover_tile(window, 50, 60)
        assert len(window._cliff_hover_parts) == 2
        assert window._cliff_ghost_parts == {}, "the stroke's parts are its own"
        window.show_sprites_action.setChecked(not window.show_sprites_action.isChecked())
        _hover_tile(window, 70, 70)
        assert list(window._cliff_hover_parts) == [(_DEFAULT_CLIFF, 3, 70.5, 70.5)]
    finally:
        _close(window)


def test_the_hover_s_ghost_parts_start_over_after_an_elevation_edit():
    """An in-place elevation edit keeps id(_iso_elevations), so the look
    cannot tell: _apply_dirty() drops the cached draws along with the memo."""
    window = _shown_window()
    try:
        _hover_tile(window, 30, 40)
        stale = window._cliff_hover_parts
        assert len(stale) == 1
        mm = window.scenario.map_manager
        index = 40 * mm.map_width + 30
        mm.terrain[index].elevation += 1
        window._apply_dirty([index])
        _hover_tile(window, 30, 40)
        assert window._cliff_hover_parts is not stale
        assert len(window._cliff_hover_parts) == 1
    finally:
        _close(window)


def _mouse_press(window, tile) -> None:
    from PyQt5.QtCore import QEvent

    view = window.map_view
    pos = conftest.viewport_pos(view, *tile)
    view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))


def _mouse_move_held(window, tile) -> None:
    from PyQt5.QtCore import QEvent

    view = window.map_view
    pos = conftest.viewport_pos(view, *tile)
    view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, pos, Qt.NoButton, Qt.LeftButton))


def _mouse_release(window, tile) -> None:
    from PyQt5.QtCore import QEvent

    view = window.map_view
    pos = conftest.viewport_pos(view, *tile)
    view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton))


def _mouse_drag(window, tiles) -> None:
    """A real press, one held move per further tile, and a release on the last."""
    _mouse_press(window, tiles[0])
    for tile in tiles[1:]:
        _mouse_move_held(window, tile)
    _mouse_release(window, tiles[-1])


def test_the_hover_preview_reruns_at_the_cursor_after_release_undo_and_redo():
    """Without a mouse move: the preview is asked again for the tile the drag
    ended on (not the one it started on), and again after undo and redo, since
    what a click would place depends on the cliffs now on the map."""
    window = _shown_window()
    try:
        view = window.map_view
        asked = []

        def preview(tile):
            asked.append(tile)
            window.on_cliff_preview(tile)

        view.set_cliff_preview(preview)
        _hover_tile(window, 30, 40)
        assert asked == [(30, 40)]
        before = _unit_count(window)
        _mouse_drag(window, [(30, 40), (30, 43), (30, 46)])
        assert _unit_count(window) > before
        # No preview mid-stroke (the run is the ghost), one at release.
        assert asked == [(30, 40), (30, 46)]
        window.undo()
        assert asked[2:] == [(30, 46)] and _unit_count(window) == before
        window.redo()
        assert asked[3:] == [(30, 46)]
    finally:
        _close(window)


def test_the_hover_ghost_and_footprint_are_gone_on_leaving_the_view():
    from PyQt5.QtCore import QEvent

    window = _shown_window()
    try:
        view = window.map_view
        _hover_tile(window, 40, 40)
        assert view._unit_ghost_item is not None
        assert view._cliff_footprint is not None
        view.leaveEvent(QEvent(QEvent.Leave))
        assert view._unit_ghost_item is None
        assert view._cliff_footprint is None
        assert _highlight_box(view) is None
    finally:
        _close(window)


def test_a_stroke_closed_by_leaving_the_view_leaves_no_highlight_or_ghost():
    """TASK-358: the stroke's close repaints the hover's highlight and ghost,
    so leaveEvent clears them after closing it, not before."""
    from PyQt5.QtCore import QEvent

    window = _shown_window()
    try:
        view = window.map_view
        _hover_tile(window, 30, 40)
        before = _unit_count(window)
        _mouse_press(window, (30, 40))
        _mouse_move_held(window, (30, 46))
        assert view._stroke_active

        view.leaveEvent(QEvent(QEvent.Leave))

        assert not view._stroke_active
        assert _unit_count(window) > before
        assert _highlight_box(view) is None
        assert view._unit_ghost_item is None
        assert view._cliff_footprint is None
    finally:
        _close(window)


def _gaia(window):
    return window.scenario.unit_manager.units[GAIA_PLAYER_ID]


def _tall_run(window):
    """Three 3x3 cliffs drawn south from a press at (41, 35), so they sit on
    (40,43,34,37), (40,43,37,40) and (40,43,40,43): a tall south cap (18),
    a straight, and a tall north cap (19) at the bottom. Returns that cap."""
    _stroke(window, [(41, y) for y in range(35, 41)])
    cap = _gaia(window)[-1]
    assert (cap.unit_const, cap.x, cap.y, cap.rotation) == (_DEFAULT_CLIFF, 41.5, 41.5, 19.0)
    return cap


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def test_a_drag_anchored_on_an_end_cap_extends_the_run_and_reshapes_the_cap_in_one_undo_record():
    """GH #44 round 2: press on the end of an existing run and drag on in
    line. The run continues at the same size, and the old end cap becomes a
    straight (its frame only: same const, same position), all in one record."""
    window = _window()
    try:
        _select_piece(window, _DEFAULT_CLIFF)
        window.cliff_action.setChecked(True)
        cap = _tall_run(window)
        before = _unit_count(window)
        before_records = len(window.edit_history.records)
        placed_at = (cap.unit_const, cap.x, cap.y, cap.z, cap.reference_id)

        _stroke(window, [(41, y) for y in range(41, 47)])

        assert _unit_count(window) == before + 2
        assert len(window.edit_history.records) == before_records + 1
        assert [(u.unit_const, u.x, u.y, u.rotation) for u in _gaia(window)[-2:]] == [
            (_DEFAULT_CLIFF, 41.5, 44.5, 2.0),
            (_DEFAULT_CLIFF, 41.5, 47.5, 19.0),
        ]
        # The cap at (40,43,40,43): (40 + 40) // 3 is even, so the straight's lower twin.
        assert (cap.rotation, cap.initial_animation_frame) == (1.0, 1)
        assert (cap.unit_const, cap.x, cap.y, cap.z, cap.reference_id) == placed_at
        assert _last_status(window) == (
            "Placed a run of 2 cliffs from the picked family, shaped by the drag direction and their neighbours, "
            "1 existing cliff reshaped to join"
        )

        window.undo()
        assert _unit_count(window) == before
        assert (cap.rotation, cap.initial_animation_frame) == (19.0, 19)
        window.redo()
        assert _unit_count(window) == before + 2
        assert _gaia(window)[-3] is cap
        assert (cap.rotation, cap.initial_animation_frame) == (1.0, 1)
    finally:
        _close(window)


def test_a_join_that_needs_a_different_size_piece_is_left_alone_and_counted():
    """Sideways off the same cap, to the west: no all-3x3 corner faces that
    way, so the cap keeps its frame and the status line says so."""
    window = _window()
    try:
        _select_piece(window, _DEFAULT_CLIFF)
        window.cliff_action.setChecked(True)
        cap = _tall_run(window)
        before = _unit_count(window)

        _stroke(window, [(x, 41) for x in range(41, 35, -1)])

        assert _unit_count(window) == before + 2
        assert (cap.rotation, cap.initial_animation_frame) == (19.0, 19)
        assert _last_status(window) == (
            "Placed a run of 2 cliffs from the picked family, shaped by the drag direction and their neighbours, "
            "1 left as it was (a join there needs a different-size piece)"
        )
    finally:
        _close(window)


def test_a_unit_change_between_press_and_release_drops_the_stroke():
    """An undo mid-drag changes the units the stroke scanned at press, and a
    join writes to those very objects: the stroke places nothing instead."""
    window = _window()
    try:
        _select_piece(window, _DEFAULT_CLIFF)
        window.cliff_action.setChecked(True)
        _tall_run(window)
        window.on_edit_stroke_start()
        for y in range(41, 47):
            window.on_edit_stroke_tile(41, y, Qt.NoModifier)
        assert len(window._cliff_stroke.nodes()) == 2
        window.undo()
        after_undo = _unit_count(window)
        records = len(window.edit_history.records)

        window.on_edit_stroke_end()

        assert _unit_count(window) == after_undo
        assert len(window.edit_history.records) == records
        assert _last_status(window) == "Cliff: the map's units changed during the drag, so nothing was placed"
        assert window._cliff_stroke is None
    finally:
        _close(window)


def test_a_join_reshapes_a_player_owned_cap_and_undo_restores_it():
    """TASK-358: the edit declares the cap owner's list as well as GAIA's, so
    the undo record holds that cap's old frame."""
    window = _window()
    try:
        _select_piece(window, _DEFAULT_CLIFF)
        window.cliff_action.setChecked(True)
        gaia_cap = _tall_run(window)
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Give the cap to player 1", [GAIA_PLAYER_ID, 1]):
            model.remove(gaia_cap)
            model.add(1, _DEFAULT_CLIFF, 41.5, 41.5, rotation=19.0, initial_animation_frame=19)
        cap = window.scenario.unit_manager.units[1][-1]
        assert (cap.unit_const, cap.x, cap.y, cap.rotation) == (_DEFAULT_CLIFF, 41.5, 41.5, 19.0)
        before = _unit_count(window)

        _stroke(window, [(41, y) for y in range(41, 47)])

        assert _unit_count(window) == before + 2
        assert (cap.rotation, cap.initial_animation_frame) == (1.0, 1)
        window.undo()
        assert _unit_count(window) == before
        assert (cap.rotation, cap.initial_animation_frame) == (19.0, 19)
    finally:
        _close(window)


def _press_a_join_on_a_reopened_run(window, tmp_path):
    """Saves the tall run and reopens it, so the press sees an unedited file
    (unit_gen 0), then holds a drag that would extend it and reshape its cap."""
    _select_piece(window, _DEFAULT_CLIFF)
    window.cliff_action.setChecked(True)
    _tall_run(window)
    out = tmp_path / "run.aoe2scenario"
    window.scenario.path = out
    window._untitled = False
    window.save()
    window.load_scenario(out)
    window.mode_combo.setCurrentText("Terrain")
    _select_piece(window, _DEFAULT_CLIFF)
    window.cliff_action.setChecked(True)
    assert window.scenario.unit_gen == 0
    window.on_edit_stroke_start()
    for y in range(41, 47):
        window.on_edit_stroke_tile(41, y, Qt.NoModifier)
    assert len(window._cliff_stroke.nodes()) == 2 and len(window._cliff_stroke.rewrites()) == 1


def test_a_document_swap_between_press_and_release_drops_the_stroke(tmp_path):
    """TASK-358: unit_gen restarts at 0 per loaded file, so between two
    unedited files only the scenario's identity tells the swap apart."""
    window = _window()
    try:
        _press_a_join_on_a_reopened_run(window, tmp_path)
        window.load_scenario(FIXTURE_PATH)
        assert window.scenario.unit_gen == 0
        before = _unit_count(window)
        records = len(window.edit_history.records)

        window.on_edit_stroke_end()

        assert _unit_count(window) == before
        assert len(window.edit_history.records) == records
        assert _last_status(window) == "Cliff: the map's units changed during the drag, so nothing was placed"
    finally:
        _close(window)


def test_a_join_rewrite_that_raises_leaves_the_map_untouched(tmp_path, monkeypatch):
    """TASK-358: the rewrites run before the adds, since an aborted unit edit
    does not roll back: a raise there must not have placed any node yet."""
    from descape.unit_model import UnitEditModel

    window = _window()
    try:
        _press_a_join_on_a_reopened_run(window, tmp_path)
        cap = _gaia(window)[-1]
        before = _unit_count(window)
        records = len(window.edit_history.records)

        def refuse(self, unit, frame):
            raise ValueError("refused")

        monkeypatch.setattr(UnitEditModel, "set_cliff_frame", refuse)
        with pytest.raises(ValueError, match="refused"):
            window.on_edit_stroke_end()

        assert _unit_count(window) == before
        assert (cap.rotation, cap.initial_animation_frame) == (19.0, 19)
        assert len(window.edit_history.records) == records
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
