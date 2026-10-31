"""The mid-drag move preview: a translucent ghost of the dragged unit that
follows the cursor, snapped to the destination tile and sitting at that
tile's elevation.

**Which layer each test asserts on is the point here.** b1's postmortem is
this module's trap, inverted: that slice asserted only on the model, shipped
a cache-invalidation bug, and the repo answered with pixel-level tests reading
window._cache.get_chunk() directly (tests/test_unit_edit_viewer.py). Those
tests cannot see this feature at all -- the ghost is a scene item, not cache
pixels, and it is the design that it never becomes cache pixels. So:

- geometry, through the real unit_at()/unit_polygons()/sprite path;
- scene-item presence and clearing, mirroring the marquee and selection-
  outline tests in tests/test_unit_selection_viewer.py;
- the gesture itself, synthesized press-move-release into MapView -- which
  nothing in tests/ did for a unit drag before this module (every existing
  move test calls window.on_unit_move() directly, and nothing anywhere
  referenced UNIT_DRAG_THRESHOLD_PX);
- and the design intent: no cache invalidation between press and release.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication
from test_place_preview import _ghost_draws, _placed_draws

from descape import iso_geometry, object_catalog, placement_rules, render, unit_pick, unit_rotation
from testkit import fakes

import conftest

# A 4x4 building const, for the map-edge clamp the units fixture's own
# almost-entirely-1x1 placements cannot exercise.
_MULTI_TILE_CONST = 33

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


def _window(style: str = "Flat"):
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    # Unchecked BEFORE the style switch, matching every other module here:
    # iso_action defaults checked, so Flat would otherwise render through the
    # real-iso path rather than the plain top-down canvas.
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText(style)
    window.mode_combo.setCurrentText("Units")
    # Every gesture here is a unit drag, which is the Move tool's (TASK-012).
    window.move_action.setChecked(True)
    window.show()
    QApplication.processEvents()
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _an_entry(window):
    """Some unit in the fixture, as the live index's own entry -- the shape
    every callable here resolves through."""
    index = window.map_view._unit_index
    assert index is not None and index.entries
    return index.entries[0]


def _viewport_pos(window, tile_x: float, tile_y: float) -> QPointF:
    """Viewport-space centre of a (possibly fractional) tile. Flat only: its
    _pick_tile branch is a plain pixel/tile_px division, so a scene point is
    trivial to construct. Stepped/Sloped drive the resolver directly instead
    of through synthesized pixels."""
    tile_px = window.map_view._tile_pixels
    scene = QPointF((tile_x + 0.5) * tile_px, (tile_y + 0.5) * tile_px)
    return QPointF(window.map_view.mapFromScene(scene))


def _drag(window, entry, from_tile, to_tile, release: bool = True):
    """A real press-move-release on a unit, through MapView's own handlers."""
    view = window.map_view
    press = _viewport_pos(window, *from_tile)
    move = _viewport_pos(window, *to_tile)
    view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, press, Qt.LeftButton, Qt.LeftButton))
    view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, move, Qt.NoButton, Qt.LeftButton))
    if release:
        view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, move, Qt.LeftButton, Qt.NoButton))
    return move


# --- S1: geometry at a destination, not at the stored position -------------


def test_a_stand_in_recomputes_its_bounds_at_a_map_edge_rather_than_translating() -> None:
    """The clamping trap, and the reason unit_at() exists rather than a tile
    delta added to a computed box. unit_tile_bounds() CLAMPS to the map, so
    translating a clamped box by the drag delta is wrong for any unit dragged
    near an edge -- the translated box would still carry the clamp taken at
    the source.

    Pure geometry on a synthetic multi-tile unit rather than a fixture one:
    this needs a span > 1 to have any clamp to bite, and the units fixture is
    almost entirely 1x1 (a skip here would leave the trap untested, which is
    the whole reason this test exists)."""
    w = h = 120
    unit = fakes.SyntheticUnit(x=60.0, y=60.0, unit_const=_MULTI_TILE_CONST)
    span_x, span_y = render.tile_span(_MULTI_TILE_CONST, render.NON_BUILDING_SPAN)
    assert span_x > 1 and span_y > 1, "this const must be multi-tile for the clamp to bite"

    middle = render.unit_tile_bounds(render.unit_at(unit, 60.0, 60.0), w, h)
    edge = render.unit_tile_bounds(render.unit_at(unit, w - 1.0, h - 1.0), w, h)
    assert middle is not None and edge is not None
    # The clamped box at the edge is genuinely narrower than the same unit's
    # box in open ground; translating the open-ground box could not produce
    # that, and translating the EDGE box back inward could not undo it.
    assert (edge[1] - edge[0]) < (middle[1] - middle[0])
    assert (edge[3] - edge[2]) < (middle[3] - middle[2])
    assert edge[1] == w and edge[3] == h


def test_a_stand_in_leaves_the_real_unit_untouched() -> None:
    """unit_at() must never be a view onto the real unit: a preview that
    moved the unit it previews would commit the move on the first mouse
    move, and every hard rule about write paths would be running a frame at
    a time."""
    window = _window()
    try:
        unit = _an_entry(window).unit
        before = (unit.x, unit.y)
        ghost = render.unit_at(unit, 12.5, 34.5)
        assert (ghost.x, ghost.y) == (12.5, 34.5)
        assert (unit.x, unit.y) == before
        assert ghost is not unit
    finally:
        _close(window)


def test_a_sloped_ghost_sits_at_its_destinations_height_not_its_sources() -> None:
    """The floating-ghost trap: a ghost hanging at the SOURCE elevation over a
    ramp is the obvious bug in this slice, and it is invisible to any test
    that only drags across flat ground.

    Pure geometry against a hand-built corner_rise rather than the units
    fixture, which is uniformly elevation 0 and so cannot discriminate this at
    all. The claim is specifically that the RISE TERM moved: two different
    tiles produce two different polygons even at one height, purely from their
    own screen origins, so "the polygons differ" would pass a frozen ghost."""
    tile_px = 16
    proj = iso_geometry.canvas_size_and_origin(5, 5, tile_px, 0, 3)
    corner_rise = np.zeros((6, 6), dtype=np.int32)
    corner_rise[3:, :] = 12  # a ramp across the lower half

    unit = fakes.SyntheticUnit(x=1.5, y=1.5, unit_const=83)

    def entry_at(tx: int, ty: int):
        ghost = render.unit_at(unit, tx + 0.5, ty + 0.5)
        return unit_pick.UnitEntry(1, ghost, tx, ty, 0)

    low, high = entry_at(1, 1), entry_at(1, 4)
    assert unit_pick.unit_rise_px_for(low, corner_rise) == 0
    assert unit_pick.unit_rise_px_for(high, corner_rise) == 12

    # And the rise actually reaches the drawn shape: same tile ROW position in
    # screen space would put these at the same height without it.
    low_poly = unit_pick.unit_polygons(low, "sloped", tile_px, 5, 5, None, proj, corner_rise=corner_rise)
    high_poly = unit_pick.unit_polygons(high, "sloped", tile_px, 5, 5, None, proj, corner_rise=corner_rise)
    assert low_poly and high_poly
    flat_high = unit_pick.unit_polygons(
        high, "sloped", tile_px, 5, 5, None, proj, corner_rise=np.zeros_like(corner_rise)
    )
    assert flat_high
    assert min(py for pts in high_poly for _px, py in pts) == (
        min(py for pts in flat_high for _px, py in pts) - 12
    )


# --- S2: the preview and the commit agree about where the unit lands -------


def test_the_preview_and_the_commit_resolve_the_same_point() -> None:
    """If these diverge the preview lies about where the unit lands, which is
    worse than no preview at all. One resolver, asserted rather than
    documented."""
    window = _window()
    try:
        pos = window.map_view.mapToScene(_viewport_pos(window, 40, 40).toPoint())
        point, fell_back = window._resolve_placement_point(pos, Qt.NoModifier)
        assert window._placement_point(pos, Qt.NoModifier) == point
        assert fell_back is False
        assert point == (40.5, 40.5)
    finally:
        _close(window)


def test_the_previews_resolver_does_not_log_an_off_grid_fallback() -> None:
    """The status-log spam trap: the committing callers log a refused off-grid
    placement inverse, and this resolver runs once per mouse-move. On a steep
    Sloped ramp the inverse can refuse pixel after pixel, so logging here
    would fill the log with one line per frame."""
    window = _window()
    try:
        logged = []
        window._log_status = lambda text, *a, **k: logged.append(text)
        window.off_grid_check.setChecked(True)
        # Forces the fallback branch without needing a degenerate ramp.
        window.map_view._pick_map_point = lambda pos: None
        pos = window.map_view.mapToScene(_viewport_pos(window, 40, 40).toPoint())

        point, fell_back = window._resolve_placement_point(pos, Qt.NoModifier)
        assert (point, fell_back) == ((40.5, 40.5), True)
        assert logged == []

        window._placement_point(pos, Qt.NoModifier)
        assert len(logged) == 1
    finally:
        _close(window)


# --- S3/S4: the ghost item, the gesture, and the clears --------------------


def test_a_real_drag_past_the_threshold_shows_a_ghost_at_the_destination() -> None:
    """Not "a ghost exists" -- WHERE it is. A ghost resolved from the real
    unit rather than the stand-in would still be a non-empty item with sane
    geometry; it would just sit on the source tile, which is both the most
    likely wiring mistake here and completely invisible to a presence check.

    Flat, so the expected rect is the tile's own pixel box exactly."""
    window = _window()
    try:
        view = window.map_view
        entry = _an_entry(window)
        unit = entry.unit
        assert unit.x % 1 == 0.5 and unit.y % 1 == 0.5, "the exact-rect form needs a centred unit"
        src = (int(unit.x), int(unit.y))
        dst = (src[0] + 6, src[1] + 6)
        _drag(window, entry, src, dst, release=False)

        item = view._unit_ghost_item
        assert item is not None
        tp = view._tile_pixels
        assert item.sceneBoundingRect() == QRectF(dst[0] * tp, dst[1] * tp, tp, tp)
    finally:
        _close(window)


def test_a_snapped_4x4_drag_ghost_lands_on_whole_numbers() -> None:
    """GH #206: an even-span building snaps to the whole number nearest the
    cursor, in the ghost exactly as in the commit. Flat, so the ghost's rect
    is its owned tiles' pixel box."""
    window = _window()
    try:
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add", [1]):
            unit = model.add(1, _MULTI_TILE_CONST, 60.0, 60.0)
        key = (1, unit.reference_id)
        window._selection = [key]
        window._refresh_selection_view()
        tp = window.map_view._tile_pixels
        pos = QPointF(64.3 * tp, 61.8 * tp)

        window.on_unit_drag_preview(key, pos, Qt.NoModifier)
        item = window.map_view._unit_ghost_item
        assert item is not None
        assert item.sceneBoundingRect() == QRectF(62 * tp, 60 * tp, 4 * tp, 4 * tp)

        window.on_unit_move(key, pos, Qt.NoModifier)
        assert (unit.x, unit.y) == (64.0, 62.0)
    finally:
        _close(window)


def test_a_plain_click_never_flickers_a_ghost() -> None:
    """Below UNIT_DRAG_THRESHOLD_PX the gesture is still a selection click, so
    a ghost appearing on 1px of hand jitter would read as a flicker bug. The
    threshold this gates on is the same one mouseReleaseEvent uses to tell a
    click from a drag."""
    window = _window()
    try:
        view = window.map_view
        entry = _an_entry(window)
        src = (int(entry.unit.x), int(entry.unit.y))
        press = _viewport_pos(window, *src)
        jitter = QPointF(press.x() + 1.0, press.y() + 1.0)
        assert (jitter - press).manhattanLength() <= view.UNIT_DRAG_THRESHOLD_PX

        view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, press, Qt.LeftButton, Qt.LeftButton))
        view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, jitter, Qt.NoButton, Qt.LeftButton))
        assert view._unit_ghost_item is None
    finally:
        _close(window)


def test_the_ghost_clears_on_release() -> None:
    window = _window()
    try:
        view = window.map_view
        entry = _an_entry(window)
        src = (int(entry.unit.x), int(entry.unit.y))
        _drag(window, entry, src, (src[0] + 6, src[1] + 6))
        assert view._unit_ghost_item is None
    finally:
        _close(window)


def test_escape_mid_drag_clears_the_ghost_and_commits_nothing() -> None:
    """Escape is not optional polish here: a visible preview invites a cancel
    gesture, and before this a started drag could only be completed. Dropping
    _unit_drag_key is what makes the eventual release a no-op."""
    from PyQt5.QtGui import QKeyEvent

    window = _window()
    try:
        view = window.map_view
        entry = _an_entry(window)
        unit = entry.unit
        before = (unit.x, unit.y)
        src = (int(unit.x), int(unit.y))
        move = _drag(window, entry, src, (src[0] + 6, src[1] + 6), release=False)
        assert view._unit_ghost_item is not None

        view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
        assert view._unit_ghost_item is None
        assert view._unit_drag_key is None

        view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, move, Qt.LeftButton, Qt.NoButton))
        assert (unit.x, unit.y) == before
    finally:
        _close(window)


def test_leaving_the_widget_clears_the_ghost_but_not_the_drag() -> None:
    """Qt's implicit grab still routes the release back here, so a drag that
    merely crosses the viewport edge must still be able to commit. What must
    not survive is a preview drawn where the cursor no longer is."""
    window = _window()
    try:
        view = window.map_view
        entry = _an_entry(window)
        src = (int(entry.unit.x), int(entry.unit.y))
        _drag(window, entry, src, (src[0] + 6, src[1] + 6), release=False)
        assert view._unit_ghost_item is not None

        view.leaveEvent(QEvent(QEvent.Leave))
        assert view._unit_ghost_item is None
        assert view._unit_drag_key is not None
    finally:
        _close(window)


def test_the_hover_outline_no_longer_freezes_over_the_source_unit() -> None:
    """The gap the Context section of this feature's plan opens on: the drag
    branch used to `return` without clearing the hover cue, unlike every
    sibling branch, so the cyan outline sat frozen on the source unit for the
    whole drag."""
    window = _window()
    try:
        view = window.map_view
        entry = _an_entry(window)
        src = (int(entry.unit.x), int(entry.unit.y))
        # Hover it first, so there is a live outline for the drag to clear.
        view.mouseMoveEvent(
            conftest.mouse_event(QEvent.MouseMove, _viewport_pos(window, *src), Qt.NoButton, Qt.NoButton)
        )
        _drag(window, entry, src, (src[0] + 6, src[1] + 6), release=False)
        assert view._unit_hover_item is None
        assert view._unit_hover_key is None
    finally:
        _close(window)


# --- The design intent, pinned ---------------------------------------------


def test_a_drag_invalidates_no_cache_until_the_release_commits() -> None:
    """THE assertion that stops a future session "simplifying" the ghost into
    per-move mutation.

    Every unit mutation routes through _after_unit_mutation(), which is
    whole-canvas on all three of its steps (~15-20ms for invalidate_units'
    walk alone at 11k units, +34-92ms per resident mip with sprites on, and a
    whole-canvas invalidate_region whose next paint is the COLD composite
    path). That is one to three orders of magnitude past a frame, which is
    why the preview is an overlay and not a re-render."""
    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # the drop is under the fixture's House
        view = window.map_view
        entry = _an_entry(window)
        unit = entry.unit
        src = (int(unit.x), int(unit.y))
        calls = {"units": 0, "region": 0}
        cache = window._cache
        real_units, real_region = cache.invalidate_units, cache.invalidate_region

        def counted_units(*a, **k):
            calls["units"] += 1
            return real_units(*a, **k)

        def counted_region(*a, **k):
            calls["region"] += 1
            return real_region(*a, **k)

        cache.invalidate_units = counted_units
        cache.invalidate_region = counted_region

        press = _viewport_pos(window, *src)
        view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, press, Qt.LeftButton, Qt.LeftButton))
        for step in range(1, 7):
            view.mouseMoveEvent(
                conftest.mouse_event(
                    QEvent.MouseMove, _viewport_pos(window, src[0] + step, src[1] + step), Qt.NoButton, Qt.LeftButton
                )
            )
        assert calls == {"units": 0, "region": 0}, "the ghost must touch no cache state"

        end = _viewport_pos(window, src[0] + 6, src[1] + 6)
        view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))
        assert (int(unit.x), int(unit.y)) == (src[0] + 6, src[1] + 6), "the release must still commit"
        assert calls["units"] + calls["region"] > 0, "the commit must still invalidate"
    finally:
        _close(window)


def test_the_wall_connectivity_override_is_resolved_once_per_drag() -> None:
    """Per-move it would be a walk over every unit in the file (the memo
    behind wall_variant_rotation_overrides) plus _unit_list_index's linear
    scan by identity -- exactly the per-frame cost the preview exists to
    avoid."""
    window = _window()
    try:
        entry = _an_entry(window)
        scans = []
        real = window._unit_list_index
        window._unit_list_index = lambda pid, unit: (scans.append(pid), real(pid, unit))[1]
        for _ in range(5):
            window._ghost_rotation_override(entry)
        assert len(scans) == 1
    finally:
        _close(window)


# --- GH #75: a group drag ghosts every member ------------------------------


def _group(window, tiles):
    """Villagers for player 1 centred on `tiles`, selected as one group."""
    occupied = {(int(u.x), int(u.y)) for units in window.scenario.unit_manager.units for u in units}
    assert not occupied & set(tiles), "the fixture already holds a unit on a test tile"
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Add", [1]):
        units = [model.add(1, 83, tx + 0.5, ty + 0.5) for tx, ty in tiles]
    keys = [(1, u.reference_id) for u in units]
    window._selection = list(keys)
    window._refresh_selection_view()
    return keys, units


def test_a_group_drag_draws_one_ghost_per_member_at_its_destination() -> None:
    window = _window()
    try:
        view = window.map_view
        tiles = [(62, 62), (65, 64)]
        _keys, units = _group(window, tiles)
        _drag(window, None, tiles[0], (68, 66), release=False)

        item = view._unit_ghost_item
        assert item is not None
        assert len(item._marks) == 2
        tp = view._tile_pixels
        expected = QRectF(68 * tp, 66 * tp, tp, tp).united(QRectF(71 * tp, 68 * tp, tp, tp))
        assert item.sceneBoundingRect() == expected
        assert [(u.x, u.y) for u in units] == [(62.5, 62.5), (65.5, 64.5)], "a preview writes nothing"
    finally:
        _close(window)


def test_past_the_cap_only_the_grabbed_unit_is_ghosted_and_the_status_says_so(monkeypatch) -> None:
    from descape import viewer

    window = _window()
    try:
        monkeypatch.setattr(viewer, "GROUP_GHOST_CAP", 1)
        logged = []
        real_log = window._log_status
        window._log_status = lambda text, *a, **k: (logged.append(text), real_log(text, *a, **k))
        tiles = [(62, 62), (65, 64)]
        _group(window, tiles)
        view = window.map_view
        _drag(window, None, tiles[0], (68, 66), release=False)
        view.mouseMoveEvent(
            conftest.mouse_event(QEvent.MouseMove, _viewport_pos(window, 69, 66), Qt.NoButton, Qt.LeftButton)
        )

        assert len(view._unit_ghost_item._marks) == 1
        assert sum("Moving 2 units" in line for line in logged) == 1, "logged once per drag, not per move"
    finally:
        _close(window)


def test_escape_mid_group_drag_clears_every_ghost_and_writes_nothing() -> None:
    from PyQt5.QtGui import QKeyEvent

    window = _window()
    try:
        view = window.map_view
        tiles = [(62, 62), (65, 64)]
        keys, units = _group(window, tiles)
        cursor = window.edit_history.cursor
        move = _drag(window, None, tiles[0], (68, 66), release=False)
        assert view._unit_ghost_item is not None

        view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
        assert view._unit_ghost_item is None
        view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, move, Qt.LeftButton, Qt.NoButton))

        assert [(u.x, u.y) for u in units] == [(62.5, 62.5), (65.5, 64.5)]
        assert window.edit_history.cursor == cursor
        assert window._selection == keys, "a cancelled drag does not collapse the group either"
    finally:
        _close(window)


def test_a_group_ghost_follows_the_cursor_off_the_map_clamped_at_the_edge() -> None:
    window = _window()
    try:
        view = window.map_view
        w = window.scenario.map_manager.map_width
        tiles = [(w - 6, 60), (w - 4, 60)]
        _group(window, tiles)
        _drag(window, None, tiles[0], (w + 8, 60), release=False)

        item = view._unit_ghost_item
        assert item is not None, "the single-unit path clears off-map; a group clamps instead"
        tp = view._tile_pixels
        assert item.sceneBoundingRect() == QRectF((w - 3) * tp, 60 * tp, tp, tp).united(
            QRectF((w - 1) * tp, 60 * tp, tp, tp)
        )
    finally:
        _close(window)


def test_a_group_drag_scans_each_members_list_index_once_not_per_frame() -> None:
    window = _window()
    try:
        tiles = [(62, 62), (65, 64)]
        _group(window, tiles)
        index = window.map_view._unit_index
        entries = [index.entry_for_key(k) for k in window._selection]
        scans = []
        real = window._unit_list_index
        window._unit_list_index = lambda pid, unit: (scans.append(pid), real(pid, unit))[1]
        for _ in range(4):
            for entry in entries:
                window._ghost_rotation_override(entry)
        assert len(scans) == 2
    finally:
        _close(window)


# --- TASK-327: a drag ghost shows a drop the release would refuse --------------------

_VILLAGER = 83
_ARCHER = 4
_TOWER = 79  # Watch Tower: 5 places, takes foot archers
_WATER_DEEP = 22


def _add(window, const: int, tiles, player: int = 1) -> list:
    """Units added through the model, which never asks the placement rules."""
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Add", [player]):
        return [model.add(player, const, tx + 0.5, ty + 0.5) for tx, ty in tiles]


def _set_terrain(window, tiles, terrain_id: int) -> None:
    mm = window.scenario.map_manager
    for x, y in tiles:
        mm.terrain[y * mm.map_width + x].terrain_id = terrain_id


def _move(window, tile) -> None:
    window.map_view.mouseMoveEvent(
        conftest.mouse_event(QEvent.MouseMove, _viewport_pos(window, *tile), Qt.NoButton, Qt.LeftButton)
    )


def _scene_tile(window, tile) -> QPointF:
    tp = window.map_view._tile_pixels
    return QPointF((tile[0] + 0.5) * tp, (tile[1] + 0.5) * tp)


def _refused(window) -> bool:
    item = window.map_view._unit_ghost_item
    assert item is not None
    return item.refused


def test_a_single_drag_ghost_is_refused_onto_a_blocker_or_bad_terrain_and_not_back_on_its_own_point() -> None:
    window = _window()
    try:
        # The unit already stands on water: its own point is a no-op drop, not a refusal.
        _set_terrain(window, [(62, 62), (68, 62)], _WATER_DEEP)
        (unit,) = _add(window, _VILLAGER, [(62, 62)])
        _add(window, _ARCHER, [(74, 62)], player=2)
        key = (1, unit.reference_id)
        window._selection = [key]
        window._refresh_selection_view()

        _drag(window, None, (62, 62), (62, 68), release=False)
        assert not _refused(window)
        _move(window, (68, 62))
        assert _refused(window)
        _move(window, (74, 62))
        assert _refused(window)
        # Back on the press pixel MapView drops the gesture to a click, so the point is asked directly.
        window.on_unit_drag_preview(key, _scene_tile(window, (62, 62)), Qt.NoModifier)
        assert not _refused(window)
        assert (unit.x, unit.y) == (62.5, 62.5), "a preview writes nothing"
    finally:
        _close(window)


def test_an_accepting_host_is_not_refused_and_a_refusing_one_is_even_with_free_placement() -> None:
    window = _window()
    try:
        (tower,) = _add(window, _TOWER, [(70, 66)])
        (archer,) = _add(window, _ARCHER, [(66, 66)])
        key = (1, archer.reference_id)
        window._selection = [key]
        window._refresh_selection_view()
        over_tower = _scene_tile(window, (70, 66))

        window.on_unit_drag_preview(key, over_tower, Qt.NoModifier)
        assert window.map_view._unit_hover_key == (1, tower.reference_id)
        assert not _refused(window), "the drop garrisons; no placement rule applies"

        host = window.map_view._unit_index.entry_for_key((1, tower.reference_id))
        for _ in range(5):
            window._garrison_add_const(host, _ARCHER)
        window.unrestricted_check.setChecked(True)
        window.on_unit_drag_preview(key, over_tower, Qt.NoModifier)
        assert window.map_view._unit_hover_key is None
        assert _refused(window), "the release refuses on the host's message before it reads Free placement"
        # Shift makes it a plain move, which Free placement lets through.
        window.on_unit_drag_preview(key, over_tower, Qt.ShiftModifier)
        assert not _refused(window)
    finally:
        _close(window)


def test_a_group_ghost_draws_only_the_blocked_member_refused() -> None:
    window = _window()
    try:
        view = window.map_view
        tiles = [(62, 62), (65, 64)]
        _group(window, tiles)
        # +6, +4 puts the second member on this archer.
        _add(window, _ARCHER, [(71, 68)], player=2)
        _drag(window, None, tiles[0], (68, 66), release=False)

        item = view._unit_ghost_item
        assert item._refused_marks == [False, True]
        tp = view._tile_pixels
        refused = [polygon for (polygon, _b, _p), r in zip(item._marks, item._refused_marks, strict=True) if r]
        assert [p.boundingRect() for p in refused] == [QRectF(71 * tp, 68 * tp, tp, tp)]
    finally:
        _close(window)


def test_a_zero_delta_group_ghost_is_never_refused_even_over_a_non_member() -> None:
    """The release is a no-op at a zero delta, before the rule."""
    window = _window()
    try:
        tiles = [(62, 62), (65, 64)]
        keys, _units = _group(window, tiles)
        _add(window, _ARCHER, [(65, 64)], player=2)  # on the second member, which the rule would refuse
        _drag(window, None, tiles[0], (62, 68), release=False)
        # Back on the press pixel MapView drops the gesture to a click, so the delta is asked directly.
        window.on_unit_drag_preview(keys[0], _scene_tile(window, tiles[0]), Qt.NoModifier)
        assert len(window.map_view._unit_ghost_item._marks) == 2
        assert not _refused(window)
    finally:
        _close(window)


def test_one_blocker_grid_serves_every_move_and_the_drop_and_a_rebound_selection_builds_another(monkeypatch) -> None:
    from descape import placement_rules

    window = _window()
    try:
        view = window.map_view
        first, _units = _group(window, [(62, 62), (65, 64)])
        second, _units = _group(window, [(62, 80), (65, 82)])
        # The second member lands on water at +10, +6, so the drop is refused and edits nothing.
        _set_terrain(window, [(75, 70)], _WATER_DEEP)
        builds = []
        real = placement_rules.BlockerGrid

        class _Counting(real):
            def __init__(self, *args, **kwargs):
                builds.append(1)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(placement_rules, "BlockerGrid", _Counting)
        window._selection = list(first)
        window._refresh_selection_view()
        gen = window.scenario.unit_gen

        end = _drag(window, None, (62, 62), (66, 64), release=False)
        for tile in ((68, 66), (72, 68)):
            _move(window, tile)
        end = _viewport_pos(window, 72, 68)
        view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))
        assert "can't go there" in window.status_log.toPlainText().splitlines()[-1]
        assert window.scenario.unit_gen == gen
        assert len(builds) == 1

        # The same size and no unit edit: the grid is keyed on the selection list itself.
        window._selection = list(second)
        window._refresh_selection_view()
        for _ in range(2):
            window.on_unit_drag_preview(second[0], _scene_tile(window, (70, 90)), Qt.NoModifier)
        assert len(builds) == 2
    finally:
        _close(window)


@pytest.mark.parametrize("blocked, tinted", [(1, True), (2, False)], ids=["inside_cap", "outside_cap"])
def test_past_the_cap_the_grabbed_ghost_tints_for_a_blocked_member_inside_the_cap(monkeypatch, blocked, tinted) -> None:
    from descape import viewer

    window = _window()
    try:
        monkeypatch.setattr(viewer, "GROUP_GHOST_CAP", 2)
        logged = []
        real_log = window._log_status
        window._log_status = lambda text, *a, **k: (logged.append(text), real_log(text, *a, **k))
        tiles = [(62, 62), (65, 64), (68, 64)]
        _group(window, tiles)
        # +6, +4: the checked members are the grabbed one and the next in the selection.
        dest = [(tx + 6, ty + 4) for tx, ty in tiles]
        _add(window, _ARCHER, [dest[blocked]], player=2)
        _drag(window, None, tiles[0], dest[0], release=False)

        assert len(window.map_view._unit_ghost_item._marks) == 1
        assert _refused(window) is tinted
        assert any("placement is checked in full on release" in line for line in logged)
    finally:
        _close(window)


# --- GH #280: Rotate during a live drag --------------------------------------------------

_OAK = 349  # cyclable, 42 variants
_WALL = 117
_GATE = 64  # (4, 1); one step is a 4x4 diagonal, two are (1, 4)
_GATE_DIAGONAL = 659
_GATE_Y = 88  # (1, 4)
_CORNER_Y = 95  # _GATE_Y's own corner piece
_RAM = 35  # ANGLE, and holds a garrison
_TURN = 2 * math.pi

# MapView.leaveEvent reads QApplication.mouseButtons() and a synthetic press never sets it (TASK-406).
_HELD = {"value": False}


@pytest.fixture(autouse=True)
def _hold_the_synthetic_button(monkeypatch):
    monkeypatch.setattr(
        QApplication, "mouseButtons", staticmethod(lambda: Qt.LeftButton if _HELD["value"] else Qt.NoButton)
    )
    _HELD["value"] = False
    yield
    _HELD["value"] = False


def _start(window, from_tile, to_tile):
    """Press on `from_tile`, move to `to_tile`, button still down."""
    _HELD["value"] = True
    _drag(window, None, from_tile, to_tile, release=False)


def _release(window, tile) -> None:
    pos = _viewport_pos(window, *tile)
    window.map_view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton))
    _HELD["value"] = False


def _escape(window) -> None:
    window.map_view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    _HELD["value"] = False


def _cw(window, presses: int = 1) -> None:
    """The fine Rotate key, by the QAction it fires, which refuses when disabled."""
    for _ in range(presses):
        window.rotate_cw_action.trigger()


def _ccw(window, presses: int = 1) -> None:
    for _ in range(presses):
        window.rotate_ccw_action.trigger()


def _at(window, const: int, x: float, y: float, player: int = 1, rotation: float | None = None, **fields):
    """One unit at an exact point, which _add()'s tile centres can't give a gate."""
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Add", [player]):
        unit = model.add(player, const, x, y, **fields)
    if rotation is not None:
        unit.rotation = rotation
    return unit


def _select_units(window, *units, player: int = 1) -> list:
    window._selection = [(player, u.reference_id) for u in units]
    window._refresh_selection_view()
    return list(window._selection)


def _fields(unit) -> tuple:
    return (unit.x, unit.y, unit.z, unit.unit_const, unit.rotation, unit.initial_animation_frame)


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _undo_label(window) -> str:
    return window.edit_history.peek_undo().label


def _spy_positions(window, monkeypatch) -> list:
    model = window._ensure_unit_edits()
    calls: list = []
    real = model.set_position
    monkeypatch.setattr(model, "set_position", lambda unit, *a, **k: (calls.append(unit), real(unit, *a, **k))[1])
    return calls


def _count_invalidations(window) -> dict[str, int]:
    calls = {"units": 0, "region": 0}
    cache = window._cache
    real_units, real_region = cache.invalidate_units, cache.invalidate_region

    def counted_units(*a, **k):
        calls["units"] += 1
        return real_units(*a, **k)

    def counted_region(*a, **k):
        calls["region"] += 1
        return real_region(*a, **k)

    cache.invalidate_units, cache.invalidate_region = counted_units, counted_region
    return calls


def _zoomed(window):
    """A view where one tile spans well past UNIT_DRAG_THRESHOLD_PX, so a real
    drag can end on the tile it began on."""
    window.map_view.scale(12, 12)
    assert window.map_view.transform().m11() * window.map_view._tile_pixels > 2 * window.map_view.UNIT_DRAG_THRESHOLD_PX
    return window


def test_rotate_mid_drag_writes_nothing_and_the_release_lands_it_in_one_undo_record() -> None:
    window = _window()
    try:
        (unit,) = _add(window, _ARCHER, [(60, 60)])
        before = _fields(unit)
        _start(window, (60, 60), (66, 66))
        gen, records = window.scenario.unit_gen, len(window.edit_history.records)
        calls = _count_invalidations(window)

        _cw(window)
        window.rotate_cw_coarse_action.trigger()

        assert _fields(unit) == before
        assert window.scenario.unit_gen == gen and len(window.edit_history.records) == records
        assert calls == {"units": 0, "region": 0}
        assert _last_status(window) == "Move: Archer facing 5/16"

        _release(window, (66, 66))
        assert (unit.x, unit.y) == (66.5, 66.5)
        assert unit.rotation == pytest.approx(5 * _TURN / 16)
        assert len(window.edit_history.records) == records + 1
        assert _undo_label(window) == "Move and rotate unit"
        assert _last_status(window) == "Moved Archer to (66.5, 66.5), facing 5/16"

        window.undo()
        assert _fields(unit) == before
    finally:
        _close(window)


def test_a_coarse_key_mid_drag_is_a_quarter_of_the_first_rotatable_members_frames() -> None:
    window = _window()
    try:
        (eight,) = _add(window, 1956, [(60, 60)])
        (archer,) = _add(window, _ARCHER, [(62, 60)])
        (oak,) = _add(window, _OAK, [(64, 60)], player=1)
        _select_units(window, eight, archer, oak)
        _start(window, (60, 60), (60, 66))
        window.rotate_cw_coarse_action.trigger()
        _release(window, (60, 66))
        # Eight frames give two steps for every rotatable member, as on_unit_rotate_coarse.
        assert eight.rotation == pytest.approx(2 * _TURN / 8)
        assert archer.rotation == pytest.approx(2 * _TURN / 16)
        assert oak.rotation == 11.0, "a variant steps a quarter of its own count"
    finally:
        _close(window)


def test_escape_after_a_turn_writes_nothing() -> None:
    window = _window()
    try:
        (unit,) = _add(window, _ARCHER, [(60, 60)])
        before = _fields(unit)
        gen = window.scenario.unit_gen
        _start(window, (60, 60), (66, 66))
        _cw(window, 3)
        _escape(window)
        _release(window, (66, 66))
        assert _fields(unit) == before and window.scenario.unit_gen == gen
    finally:
        _close(window)


def test_a_refused_drop_after_a_turn_writes_nothing() -> None:
    window = _window()
    try:
        (unit,) = _add(window, _ARCHER, [(60, 60)])
        _add(window, _ARCHER, [(66, 66)], player=2)
        before = _fields(unit)
        gen, records = window.scenario.unit_gen, len(window.edit_history.records)
        _start(window, (60, 60), (66, 66))
        _cw(window)
        assert _refused(window)
        _release(window, (66, 66))
        assert "Tick Free placement" in _last_status(window)
        assert _fields(unit) == before
        assert (window.scenario.unit_gen, len(window.edit_history.records)) == (gen, records)
    finally:
        _close(window)


@pytest.mark.parametrize(
    ("const", "press", "x", "y", "presses", "landed"),
    [
        pytest.param(83, (60, 60), 60.5, 60.5, 6, 83, id="angle"),
        pytest.param(349, (60, 60), 60.5, 60.5, 5, 349, id="tree"),
        pytest.param(64, (60, 60), 60.0, 60.5, 2, 88, id="gate"),
    ],
)
def test_the_turned_ghost_is_the_unit_the_drop_lands(rotating_install, const, press, x, y, presses, landed) -> None:
    window = _window("Stepped")
    try:
        window.show_sprites_action.setChecked(True)
        assert window._sprites_enabled
        view = window.map_view
        unit = _at(window, const, x, y, player=0 if const == 349 else 1)
        key = (0 if const == 349 else 1, unit.reference_id)
        _select_units(window, unit, player=key[0])
        start = conftest.polygon_viewport_pos(view, *press)
        end = conftest.polygon_viewport_pos(view, 70, 74)
        _HELD["value"] = True
        view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton))
        view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton))
        unturned = _ghost_draws(view)

        _cw(window, presses)
        ghost = _ghost_draws(view)
        assert ghost != unturned, "the turn must show with no mouse move"

        view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))
        _HELD["value"] = False
        assert unit.unit_const == landed
        assert ghost == _placed_draws(window, view._unit_index.entry_for_key(key))
    finally:
        _close(window)


def test_a_turned_group_ghost_is_the_units_the_drop_lands(rotating_install) -> None:
    window = _window("Stepped")
    try:
        window.show_sprites_action.setChecked(True)
        view = window.map_view
        villager = _at(window, _VILLAGER, 60.5, 60.5)
        gate = _at(window, _GATE, 66.0, 60.5)
        keys = _select_units(window, villager, gate)
        start = conftest.polygon_viewport_pos(view, 60, 60)
        end = conftest.polygon_viewport_pos(view, 70, 74)
        _HELD["value"] = True
        view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton))
        view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton))
        unturned = _ghost_draws(view)

        _cw(window, 2)
        ghost = _ghost_draws(view)
        assert ghost != unturned

        view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))
        _HELD["value"] = False
        assert gate.unit_const == _GATE_Y
        landed = [draw for key in keys for draw in _placed_draws(window, view._unit_index.entry_for_key(key))]
        assert sorted(ghost) == sorted(landed)
    finally:
        _close(window)


def test_a_turned_gate_resolves_its_landing_point_by_its_turned_span() -> None:
    window = _window()
    try:
        gate = _at(window, _GATE, 60.0, 60.5)
        view = window.map_view
        # Off the even axis' rounding tie (GH #206), as the real cursor pixel is.
        cursor = (69.8, 69.7)
        _start(window, (60, 60), cursor)
        scene = view.mapToScene(_viewport_pos(window, *cursor).toPoint())
        tp = view._tile_pixels
        unturned = view._unit_ghost_item.sceneBoundingRect()
        assert (unturned.width(), unturned.height()) == (4 * tp, tp)

        _cw(window, 2)
        point = window._resolve_placement_point(scene, Qt.NoModifier, unit_const=_GATE_Y)[0]
        assert point != window._resolve_placement_point(scene, Qt.NoModifier, unit_const=_GATE)[0]
        turned = view._unit_ghost_item.sceneBoundingRect()
        assert (turned.width(), turned.height()) == (tp, 4 * tp), "the ghost redraws with no mouse move"

        _release(window, cursor)
        assert (gate.unit_const, gate.x, gate.y) == (_GATE_Y, *point)
        assert point[0] == 70.5 and point[1].is_integer()
    finally:
        _close(window)


def test_the_refusal_tint_and_the_drop_read_the_turned_footprint() -> None:
    window = _window()
    try:
        gate = _at(window, _GATE, 60.0, 60.5)
        cursor = (69.8, 69.7)  # lands (1, 4) on rows 68..71
        # On the (1, 4) footprint's top tile, two rows off what (4, 1) covers at that point or its own.
        _add(window, _ARCHER, [(70, 68)], player=2)
        before = _fields(gate)
        _start(window, (60, 60), cursor)
        assert not _refused(window)
        _cw(window, 2)
        assert _refused(window)
        _release(window, cursor)
        assert "Tick Free placement" in _last_status(window)
        assert _fields(gate) == before
    finally:
        _close(window)


def test_a_quarter_turn_at_one_point_is_a_new_question_for_the_tint() -> None:
    """Two diagonals resolve the same point from one cursor pixel, so only the const tells their answers apart."""
    window = _window()
    try:
        _at(window, _GATE_DIAGONAL, 60.0, 60.0)
        cursor = (69.5, 69.5)  # the whole-number point (70, 70)
        # (71, 68) is a corner of the anti-diagonal's sparse footprint, and the main diagonal's 4x4 leaves it empty.
        _add(window, _ARCHER, [(71, 68)], player=2)
        _start(window, (59, 59), cursor)
        assert not _refused(window)
        window.rotate_cw_coarse_action.trigger()
        assert _refused(window)
    finally:
        _escape(window)
        _close(window)


def test_a_turn_at_one_delta_is_a_new_question_for_the_group_tint() -> None:
    window = _window()
    try:
        gate = _at(window, _GATE, 60.0, 60.5)
        villager = _at(window, _VILLAGER, 66.5, 60.5)
        # The delta is (0, +10); this is on the (1, 4) turn's low end, re-anchored on (58, 60).
        _add(window, _ARCHER, [(58, 72)], player=2)
        _select_units(window, gate, villager)
        _start(window, (66, 60), (66, 70))
        assert not _refused(window)
        _cw(window, 2)
        assert window.map_view._unit_ghost_item._refused_marks == [False, True], "the villager's, then the gate's"
    finally:
        _escape(window)
        _close(window)


def test_a_turned_gates_own_corner_pieces_never_block_it() -> None:
    """The rule keeps the real unit, since gate_companions() finds it by identity."""
    window = _window()
    try:
        gate = _at(window, _GATE_Y, 60.5, 60.0)
        piece = _at(window, _CORNER_Y, 60.5, 58.5)
        assert placement_rules.gate_companions(window.scenario, gate) == [piece]
        point = window._resolve_placement_point(_scene_tile(window, (60, 58)), Qt.NoModifier, unit_const=_GATE)[0]
        assert placement_rules.refused(window.scenario, _GATE, *point, exclude=[gate])
        _select_units(window, gate)
        _start(window, (60, 61), (60, 58))
        _cw(window, 2)
        assert not _refused(window)
        _release(window, (60, 58))
        assert (gate.unit_const, gate.x, gate.y) == (_GATE, *point)
    finally:
        _close(window)


def test_a_key_with_the_cursor_off_the_map_stays_pending() -> None:
    window = _window()
    try:
        view = window.map_view
        (unit,) = _add(window, _ARCHER, [(60, 60)])
        before = _fields(unit)
        _start(window, (60, 60), (66, 66))
        _move(window, (-6, 66))
        assert view._unit_ghost_item is None
        gen = window.scenario.unit_gen

        _cw(window, 3)
        assert view._unit_ghost_item is None, "off the map there is still nothing to draw"
        assert _fields(unit) == before and window.scenario.unit_gen == gen

        _move(window, (70, 66))
        _release(window, (70, 66))
        assert unit.rotation == pytest.approx(3 * _TURN / 16)
    finally:
        _close(window)


def test_below_the_threshold_a_key_rotates_at_once() -> None:
    window = _window()
    try:
        view = window.map_view
        (unit,) = _add(window, _ARCHER, [(60, 60)])
        press = _viewport_pos(window, 60, 60)
        _HELD["value"] = True
        view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, press, Qt.LeftButton, Qt.LeftButton))
        assert not view.unit_drag_live()
        records = len(window.edit_history.records)

        _cw(window)
        assert unit.rotation == pytest.approx(_TURN / 16)
        assert len(window.edit_history.records) == records + 1
        assert _undo_label(window) == "Rotate unit"

        _release(window, (60, 60))
        assert (unit.x, unit.y) == (60.5, 60.5), "a click"
    finally:
        _close(window)


def test_back_inside_the_threshold_a_key_stays_pending_and_the_release_writes_nothing() -> None:
    window = _window()
    try:
        (unit,) = _add(window, _ARCHER, [(60, 60)])
        before = _fields(unit)
        gen = window.scenario.unit_gen
        _start(window, (60, 60), (66, 66))
        _move(window, (60, 60))
        assert window.map_view.unit_drag_live()
        assert window.map_view._unit_ghost_item is None

        _cw(window)
        assert window.map_view._unit_ghost_item is None, "no replay of the point before it came back"
        assert _fields(unit) == before
        _release(window, (60, 60))
        assert _fields(unit) == before and window.scenario.unit_gen == gen
    finally:
        _close(window)


def test_a_turn_dropped_on_its_own_point_is_a_rotate_with_no_position_write(monkeypatch) -> None:
    window = _zoomed(_window())
    try:
        ram = _at(window, _RAM, 60.5, 60.5)
        rider = _at(window, _ARCHER, 60.5, 60.5, garrisoned_in_id=ram.reference_id)
        _select_units(window, ram)
        positions = _spy_positions(window, monkeypatch)
        rider_before = _fields(rider)
        records = len(window.edit_history.records)

        _start(window, (59.7, 59.7), (60.3, 60.3))
        _cw(window, 2)
        _release(window, (60.3, 60.3))

        assert positions == [], "nobody's position is rewritten, occupants included"
        assert (ram.x, ram.y) == (60.5, 60.5)
        assert _fields(rider) == rider_before
        assert ram.rotation != 0.0
        assert len(window.edit_history.records) == records + 1
        assert _undo_label(window) == "Rotate unit"
        assert _last_status(window).startswith("Rotated ")
    finally:
        _close(window)


def test_a_gate_turned_onto_its_own_re_anchored_point_only_swaps_its_const(monkeypatch) -> None:
    """Its own point here is the stand-in's: the sibling's anchor on the same low corner."""
    window = _zoomed(_window())
    try:
        gate = _at(window, _GATE, 60.0, 60.5)
        _select_units(window, gate)
        positions = _spy_positions(window, monkeypatch)
        _start(window, (59.5, 59.8), (59.5, 60.2))
        _cw(window, 1)
        (x, y) = (gate.x, gate.y)
        assert (x, y) == (60.0, 60.5)
        entry = window.map_view._unit_index.entry_for_key(window._selection[0])
        stand_in = window._drag_stand_in(entry, False)
        assert (stand_in.x, stand_in.y) == (60.0, 62.0)
        # (60, 62) on the sparse diagonal is the tile the cursor lands the turned gate on.
        _move(window, (59.2, 61.5))
        _release(window, (59.2, 61.5))
        assert positions == []
        assert (gate.unit_const, gate.x, gate.y) == (_GATE_DIAGONAL, 60.0, 62.0)
        assert _undo_label(window) == "Rotate unit"
    finally:
        _close(window)


def test_a_garrison_drop_discards_the_turn_and_says_so() -> None:
    window = _window()
    try:
        (tower,) = _add(window, _TOWER, [(66, 66)])
        (archer,) = _add(window, _ARCHER, [(60, 60)])
        rotation = archer.rotation
        _start(window, (60, 60), (66, 66))
        _cw(window, 3)
        _release(window, (66, 66))
        assert archer.garrisoned_in_id == tower.reference_id
        assert archer.rotation == rotation
        assert "the turn was dropped" in _last_status(window)
    finally:
        _close(window)


def test_a_second_drag_after_escape_gets_its_own_stand_in() -> None:
    """Esc bumps no unit_gen. The first unit's stand-in would hand its rotation to the second."""
    window = _window()
    try:
        first = _at(window, _ARCHER, 60.5, 60.5, rotation=1.0)
        second = _at(window, _ARCHER, 80.5, 60.5, rotation=2.0)
        _start(window, (60, 60), (66, 66))
        _cw(window)
        _escape(window)
        _release(window, (66, 66))

        _start(window, (80, 60), (86, 66))
        _cw(window)
        _release(window, (86, 66))
        assert first.rotation == 1.0 and (first.x, first.y) == (60.5, 60.5)
        assert second.rotation == unit_rotation.rotate_step(2.0, 16, 1)
    finally:
        _close(window)


def test_a_second_drag_after_a_refused_drop_starts_at_no_turn() -> None:
    window = _window()
    try:
        (unit,) = _add(window, _ARCHER, [(60, 60)])
        _add(window, _ARCHER, [(66, 66)], player=2)
        _start(window, (60, 60), (66, 66))
        _cw(window, 3)
        _release(window, (66, 66))
        assert unit.rotation == 0.0

        _start(window, (60, 60), (70, 70))
        _release(window, (70, 70))
        assert (unit.x, unit.y) == (70.5, 70.5)
        assert unit.rotation == 0.0, "each drag starts at zero"
        assert _undo_label(window) == "Move unit"
    finally:
        _close(window)


def test_a_group_drags_failed_fit_gate_does_not_carry_into_a_later_single_drag() -> None:
    window = _window()
    try:
        gate = _at(window, _GATE, 60.0, 119.5)
        villager = _at(window, _VILLAGER, 70.5, 110.5)
        _select_units(window, gate, villager)
        _start(window, (70, 110), (70, 100))
        _cw(window)
        entry = window.map_view._unit_index.entry_for_key((1, gate.reference_id))
        assert window._drag_stand_in(entry, True) is gate
        _escape(window)
        _release(window, (70, 100))

        _select_units(window, gate)
        _start(window, (60, 119), (60, 100))
        _cw(window)
        assert window._drag_stand_in(entry, False).unit_const == _GATE_DIAGONAL
        _release(window, (60, 100))
        assert gate.unit_const == _GATE_DIAGONAL, "a single drag's fit is its landing's"
    finally:
        _close(window)


def test_a_single_drag_that_becomes_a_group_asks_the_fit_it_skipped() -> None:
    """A Select All mid-drag widens the drag with no press and no unit edit, so
    only `group` in the memo key tells the two answers apart."""
    window = _window()
    try:
        gate = _at(window, _GATE, 60.0, 119.5)
        villager = _at(window, _VILLAGER, 70.5, 110.5)
        _select_units(window, gate)
        _start(window, (60, 119), (60, 100))
        _cw(window)
        entry = window.map_view._unit_index.entry_for_key((1, gate.reference_id))
        assert window._drag_stand_in(entry, False).unit_const == _GATE_DIAGONAL

        _select_units(window, gate, villager)
        _move(window, (60, 96))
        assert window._drag_stand_in(entry, True) is gate
        _release(window, (60, 96))
        assert gate.unit_const == _GATE, "it doesn't fit at its source, so it keeps its const"
        assert (gate.x, gate.y) == (60.0, 96.5)
    finally:
        _close(window)


def test_a_group_turns_each_member_on_its_spot_in_one_record() -> None:
    window = _window()
    try:
        low = _at(window, _ARCHER, 60.5, 60.5, rotation=1.0)
        high = _at(window, _ARCHER, 62.5, 60.5, rotation=2.0)
        oak = _at(window, _OAK, 64.5, 60.5, rotation=5.0)
        wall = _at(window, _WALL, 66.5, 60.5)
        gate = _at(window, _GATE, 60.0, 64.5)
        members = (low, high, oak, wall, gate)
        before = [_fields(u) for u in members]
        _select_units(window, *members)
        records = len(window.edit_history.records)

        _start(window, (60, 60), (63, 70))
        _cw(window)
        assert _last_status(window) == "Move: turning 4 of 5 units; 1 skipped (no facing or variant to change)"
        assert [_fields(u) for u in members] == before
        _release(window, (63, 70))

        assert (low.x, low.y, low.rotation) == (63.5, 70.5, unit_rotation.rotate_step(1.0, 16, 1))
        assert (high.x, high.y, high.rotation) == (65.5, 70.5, unit_rotation.rotate_step(2.0, 16, 1))
        assert (oak.x, oak.y, oak.rotation) == (67.5, 70.5, 6.0)
        assert (wall.x, wall.y) == (69.5, 70.5)
        # Rotate, then the delta: the diagonal's anchor on the (4, 1) low corner (58, 64).
        assert (gate.unit_const, gate.x, gate.y) == (_GATE_DIAGONAL, 63.0, 76.0)
        assert len(window.edit_history.records) == records + 1
        assert _undo_label(window) == "Move and rotate 5 units"
        assert ", turning 4" in _last_status(window)

        window.undo()
        assert [_fields(u) for u in members] == before
    finally:
        _close(window)


def test_the_group_clamp_uses_a_turned_gates_span() -> None:
    window = _window()
    try:
        gate = _at(window, _GATE, 60.0, 110.5)
        villager = _at(window, _VILLAGER, 60.5, 100.5)
        _select_units(window, gate, villager)
        _start(window, (60, 100), (60, 118))
        unturned = window._group_delta(
            window.map_view._unit_index.entry_for_key((1, villager.reference_id)),
            _scene_tile(window, (60, 118)),
            Qt.NoModifier,
            commit=False,
        )
        assert unturned[2] == 9, "(4, 1) reaches row 119 from row 110"
        _cw(window, 2)
        _release(window, (60, 118))
        # (1, 4) from the same low corner: rows 110..113, so 6 rows of room.
        assert gate.unit_const == _GATE_Y
        assert (gate.x, gate.y) == (58.5, 118.0)
        assert (villager.x, villager.y) == (60.5, 106.5)
        assert "stopped at the map edge" in _last_status(window)
    finally:
        _close(window)


def test_a_group_gate_that_does_not_fit_at_its_source_keeps_its_const_and_is_counted() -> None:
    window = _window()
    try:
        gate = _at(window, _GATE, 60.0, 119.5)
        archer = _at(window, _ARCHER, 70.5, 110.5)
        _select_units(window, gate, archer)
        _start(window, (70, 110), (70, 100))
        _cw(window)
        assert _last_status(window) == "Move: turning 1 of 2 units; 1 gate(s) skipped (would hang off the map edge)"
        _release(window, (70, 100))
        assert (gate.unit_const, gate.x, gate.y) == (_GATE, 60.0, 109.5)
        assert archer.rotation == pytest.approx(_TURN / 16)
    finally:
        _close(window)


def test_a_whole_cycle_counts_no_gate_as_skipped_even_one_already_overhanging() -> None:
    window = _window()
    try:
        gate = _at(window, _GATE, 119.0, 60.5)
        archer = _at(window, _ARCHER, 70.5, 110.5)
        _select_units(window, gate, archer)
        _start(window, (70, 110), (70, 100))
        for _ in range(4):
            _cw(window)
        assert _last_status(window) == "Move: turning 1 of 2 units"
    finally:
        _close(window)


def test_a_hidden_occupant_rides_a_group_drag_with_its_own_rotation_untouched() -> None:
    window = _window()
    try:
        ram = _at(window, _RAM, 60.5, 60.5)
        rider = _at(window, _ARCHER, 60.5, 60.5, rotation=1.0, garrisoned_in_id=ram.reference_id)
        other = _at(window, _ARCHER, 64.5, 60.5)
        _select_units(window, ram, other)
        _start(window, (64, 60), (64, 70))
        _cw(window)
        _release(window, (64, 70))
        assert (ram.y, other.y) == (70.5, 70.5)
        assert ram.rotation != 0.0 and other.rotation != 0.0
        assert (rider.x, rider.y, rider.rotation) == (60.5, 70.5, 1.0)
    finally:
        _close(window)


def test_a_zero_delta_turn_writes_only_the_turned_members(monkeypatch) -> None:
    window = _zoomed(_window())
    try:
        archer = _at(window, _ARCHER, 60.5, 60.5)
        ram = _at(window, _RAM, 64.5, 60.5)
        rider = _at(window, _ARCHER, 64.5, 60.5, rotation=1.0, garrisoned_in_id=ram.reference_id)
        wall = _at(window, _WALL, 68.5, 60.5, rotation=3.0)
        before = _fields(wall), _fields(rider)
        _select_units(window, archer, ram, wall)
        positions = _spy_positions(window, monkeypatch)
        records = len(window.edit_history.records)

        _start(window, (59.7, 59.7), (60.3, 60.3))
        _cw(window)
        _release(window, (60.3, 60.3))

        assert positions == []
        assert archer.rotation == pytest.approx(_TURN / 16) and ram.rotation != 0.0
        assert (_fields(wall), _fields(rider)) == before
        assert len(window.edit_history.records) == records + 1
        assert _undo_label(window) == "Rotate 2 units"
        assert _last_status(window).startswith("Rotated 2 units")
    finally:
        _close(window)


def test_the_tint_and_the_drop_agree_on_a_gate_turned_in_place_onto_a_blocker() -> None:
    window = _zoomed(_window())
    try:
        gate = _at(window, _GATE, 60.0, 60.5)
        villager = _at(window, _VILLAGER, 66.5, 60.5)
        _add(window, _ARCHER, [(58, 63)], player=2)  # on the (1, 4) turn's low end
        before = _fields(gate), _fields(villager)
        _select_units(window, gate, villager)

        _start(window, (65.7, 59.7), (66.3, 60.3))
        assert not _refused(window), "unturned, a zero delta is no-op"
        _cw(window, 2)
        assert _refused(window)
        _release(window, (66.3, 60.3))
        assert "can't go there" in _last_status(window)
        assert (_fields(gate), _fields(villager)) == before
    finally:
        _close(window)


def test_a_member_already_standing_refused_does_not_veto_a_facing_turn() -> None:
    window = _zoomed(_window())
    try:
        _set_terrain(window, [(60, 60)], _WATER_DEEP)
        drowned = _at(window, _ARCHER, 60.5, 60.5)
        dry = _at(window, _ARCHER, 64.5, 60.5)
        _select_units(window, drowned, dry)
        _start(window, (63.7, 59.7), (64.3, 60.3))
        _cw(window)
        assert not _refused(window)
        _release(window, (64.3, 60.3))
        assert drowned.rotation == dry.rotation == pytest.approx(_TURN / 16)
    finally:
        _close(window)


def test_a_single_unit_already_standing_refused_can_still_turn_in_place() -> None:
    window = _zoomed(_window())
    try:
        _set_terrain(window, [(60, 60)], _WATER_DEEP)
        drowned = _at(window, _ARCHER, 60.5, 60.5)
        _start(window, (59.7, 59.7), (60.3, 60.3))
        _cw(window)
        assert not _refused(window)
        _release(window, (60.3, 60.3))
        assert drowned.rotation == pytest.approx(_TURN / 16)
    finally:
        _close(window)


@pytest.mark.parametrize("presses", [(1, 1), (4, 0)], ids=["there_and_back", "whole_cycle"])
def test_a_turn_undone_in_the_drag_leaves_an_off_parity_gate_as_a_plain_move(presses) -> None:
    window = _window()
    try:
        gate = _at(window, _GATE, 60.25, 60.5)
        wall = _at(window, _WALL, 70.5, 60.5)
        _select_units(window, gate, wall)
        entries = [window.map_view._unit_index.entry_for_key(k) for k in window._selection]
        _start(window, (70, 60), (70, 66))
        anchor = entries[1]
        delta = window._group_delta(anchor, _scene_tile(window, (70, 66)), Qt.NoModifier, commit=False)[1:3]

        _cw(window, presses[0])
        _ccw(window, presses[1])

        assert window._drag_stand_in(entries[0], True) is gate
        assert window._group_delta(anchor, _scene_tile(window, (70, 66)), Qt.NoModifier, commit=False)[1:3] == delta
        _release(window, (70, 66))
        assert (gate.unit_const, gate.x, gate.y) == (_GATE, 60.25, 66.5), "no re-anchor"
        assert _undo_label(window) == "Move 2 units"
    finally:
        _close(window)


def test_a_key_after_a_keybound_switch_to_place_unit_turns_the_drag_not_the_place_ghost() -> None:
    window = _window()
    try:
        (unit,) = _add(window, _ARCHER, [(60, 60)])
        window.units_panel._set_pending_object(_VILLAGER)
        window.units_panel.select_owner(1)
        _start(window, (60, 60), (66, 66))
        window.place_unit_action.setChecked(True)
        assert window._current_tool == "place_unit"
        assert window.map_view.unit_drag_live()
        place = window._place_rotation

        _cw(window, 2)
        assert window._place_rotation == place
        _release(window, (66, 66))
        assert (unit.x, unit.y, unit.rotation) == (66.5, 66.5, pytest.approx(2 * _TURN / 16))
    finally:
        _close(window)


def test_a_wall_says_it_has_nothing_to_turn_and_moves_as_it_would() -> None:
    window = _window()
    try:
        wall = _at(window, _WALL, 60.5, 60.5, rotation=2.0)
        _select_units(window, wall)
        _start(window, (60, 60), (66, 66))
        _cw(window)
        assert _last_status(window) == "Move: Stone Wall has no facing or variant to rotate"
        _release(window, (66, 66))
        assert (wall.x, wall.y) == (66.5, 66.5)
        assert _undo_label(window) == "Move unit"
    finally:
        _close(window)


def test_a_gate_turned_mid_drag_names_its_sibling_and_the_status_keeps_it() -> None:
    window = _window()
    try:
        _at(window, _GATE, 60.0, 60.5)
        _start(window, (60, 60), (70, 70))
        _cw(window, 2)
        assert _last_status(window) == f"Move: {object_catalog.display_name(_GATE_Y)}"
        _release(window, (70, 70))
        assert f"now {object_catalog.display_name(_GATE_Y)}" in _last_status(window)
    finally:
        _close(window)
