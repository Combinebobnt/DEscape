"""Track C4's Step 3 -- Sloped's incremental edit path, the piece that makes
Sloped editable rather than view-only. Covers descape.render.
dirty_screen_bbox_sloped(), SlopedChunkCache's corner_rise rebuild, and
ViewerWindow._apply_dirty()'s sloped branch.

Checks 1-6 drive _apply_dirty() and the bbox function directly, which is how
they were written: at Step 3 the Terrain tools were still greyed out in
Sloped, so there was no stroke to drive them through. They stay direct
because that is the level they measure at.

Checks 7 and 8 are the two UI routes. Check 7 is Undo, which reached
_apply_dirty() in Sloped even at Step 3 -- _update_tool_enabled()'s gate
never covered it, since _update_edit_actions() enables it from edit_history
alone -- which is what made Step 3 a fix for a reachable wrong-pixels bug
rather than wiring for a dormant one. Check 8 is Step 4's own route, a real
Elevate press/release through MapView, which only became reachable once that
gate came off.

The load-bearing check is check 1: patch-to-byte-identity against a fresh
full render. It is the one assertion that fails for a wrong dirty ring, a
stale corner_rise and a wrong canvas clamp alike, so the narrower checks
below exist to say WHICH of those broke, not to add reach. Checks 7 and 8
reuse that same oracle at the end of a real UI path.

Checks:
  1. _apply_dirty() in Sloped leaves the cache byte-identical to a fresh
     full render_terrain_sloped() of the post-edit scenario.
  2. The bbox covers a one-tile ring around every CHANGED tile, not just
     the clicked one (with_units=False, so the ring is the radius rather
     than being subsumed by UNIT_FOOTPRINT_MAX_RADIUS).
  3. The elevations array is mutated IN PLACE -- same object, new values
     (Risk #6: the pick plane and the pixels must not drift).
  4. corner_rise is re-derived by patch(), not carried over from
     construction.
  5. The bbox clamps to SlopedChunkCache's own tight canvas, not Stepped's
     skirt-padded one -- the single divergence between the two bbox
     functions.
  6. Sloped carries an elevation snapshot on ViewerWindow, so _apply_dirty()
     can never fall through to the Flat branch.
  7. Undo, taken while Sloped is showing, leaves the canvas byte-identical to
     a fresh full render -- the pre-Step-4 UI path, end to end.
  8. A real Elevate click in Sloped edits the tile the pick plane reports
     under the cursor, and leaves the canvas byte-identical the same way --
     Step 4's own route, end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

import conftest
from descape import iso_geometry
from descape.elevation_tools import set_tile_elevation
from descape.render import (
    SLOPE_CORNER_RULE,
    dirty_screen_bbox_iso,
    dirty_screen_bbox_sloped,
    render_terrain_sloped,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import SlopedChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units


def _load_sloped_scenario():
    """Same ramped fixture tests/test_sloped_chunks.py uses -- a one-level
    climb across the map's width, non-flat so the resample branch actually
    runs, and gentle enough to stay inside the +-1-between-neighbours
    invariant."""
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    mm = scenario.map_manager
    for tile in mm.terrain:
        tile.elevation = 1 if tile.x >= mm.map_width // 2 else 0
    return scenario


def _elevation_grid(mm) -> np.ndarray:
    return np.array(
        [[int(mm.get_tile(x, y).elevation) for x in range(mm.map_width)] for y in range(mm.map_height)]
    )


def _edit_and_dirty(mm, ex: int, ey: int, elevation: int):
    """Runs the real single-tile elevation edit and returns the terrain
    INDICES it actually changed -- what a stroke hands _apply_dirty(). Index,
    not (x, y): one click propagates through _elevation_tile_recursion to
    many tiles, and the propagated set is exactly what the dirty ring has to
    be measured against."""
    before = _elevation_grid(mm)
    set_tile_elevation(mm, ex, ey, elevation)
    after = _elevation_grid(mm)
    changed = {(int(x), int(y)) for y, x in zip(*np.nonzero(before != after))}
    assert changed, "set_tile_elevation produced no change -- fixture is broken"
    return [i for i, tile in enumerate(mm.terrain) if (tile.x, tile.y) in changed], changed


def _make_cache(scenario, **kwargs):
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
    cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, **kwargs)
    return cache, elevations, proj


def test_patch_after_edit_matches_a_fresh_full_render():
    """Check 1. The whole-canvas oracle: whatever the bbox and the rebuilt
    corner_rise came out as, the cached pixels must be indistinguishable
    from rendering the post-edit scenario from scratch."""
    scenario = _load_sloped_scenario()
    mm = scenario.map_manager
    cache, elevations, proj = _make_cache(scenario)
    canvas_w, canvas_h = cache.canvas_dims()
    cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk

    dirty, _changed = _edit_and_dirty(mm, mm.map_width // 4, mm.map_height // 4, 3)
    bbox = dirty_screen_bbox_sloped(scenario, dirty, elevations, proj, with_units=True)
    assert bbox is not None, "a legal in-range edit must not decline the incremental path"
    cache.patch(bbox)

    stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
    full = render_terrain_sloped(scenario, with_units=True)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


def test_bbox_covers_a_one_tile_ring_around_every_changed_tile():
    """Check 2. The failure mode is a stale one-tile fringe, not a crash, and
    it is invisible on a byte-identity run whose edit happens to sit mid-chunk
    -- so assert the ring directly. with_units=False pins the ring itself:
    with units on, UNIT_FOOTPRINT_MAX_RADIUS is larger and would cover a
    dropped ring for the wrong reason."""
    scenario = _load_sloped_scenario()
    mm = scenario.map_manager
    _cache, elevations, proj = _make_cache(scenario)

    dirty, changed = _edit_and_dirty(mm, mm.map_width // 4, mm.map_height // 4, 3)
    bbox = dirty_screen_bbox_sloped(scenario, dirty, elevations, proj, with_units=False)
    assert bbox is not None
    x0, y0, x1, y1 = bbox

    canvas_w, canvas_h = proj.canvas_w, proj.canvas_h
    for cx, cy in sorted(changed):
        for nx in range(max(0, cx - 1), min(mm.map_width, cx + 2)):
            for ny in range(max(0, cy - 1), min(mm.map_height, cy + 2)):
                tx0, ty0, tx1, ty1 = iso_geometry.tile_screen_bounds_swept(nx, ny, proj)
                # Clamped the same way the bbox itself is: a tile hanging off
                # the canvas can only be required to be covered where the
                # canvas actually exists.
                tx0, ty0 = max(0, tx0), max(0, ty0)
                tx1, ty1 = min(canvas_w, tx1), min(canvas_h, ty1)
                if tx1 <= tx0 or ty1 <= ty0:
                    continue
                assert x0 <= tx0 and y0 <= ty0 and x1 >= tx1 and y1 >= ty1, (
                    f"tile ({nx}, {ny}), in the ring around changed tile ({cx}, {cy}), "
                    f"has swept bounds {(tx0, ty0, tx1, ty1)} outside bbox {bbox}"
                )


def test_elevations_are_mutated_in_place():
    """Check 3. Risk #6 -- the caller's snapshot, the cache's and MapView's
    are one array; returning a fresh one instead would silently desync the
    pick plane from the pixels."""
    scenario = _load_sloped_scenario()
    mm = scenario.map_manager
    _cache, elevations, proj = _make_cache(scenario)
    ex, ey = mm.map_width // 4, mm.map_height // 4
    assert int(elevations[ey, ex]) != 3, "fixture already sits at the target elevation"

    dirty, changed = _edit_and_dirty(mm, ex, ey, 3)
    same_object = elevations
    dirty_screen_bbox_sloped(scenario, dirty, elevations, proj, with_units=True)

    assert elevations is same_object
    for cx, cy in changed:
        assert int(elevations[cy, cx]) == int(mm.get_tile(cx, cy).elevation)


def test_patch_rebuilds_corner_rise():
    """Check 4. The blocker Step 3 existed to clear: corner_rise used to be
    received once at construction and never refreshed, so every post-edit
    repaint used the pre-edit height field."""
    scenario = _load_sloped_scenario()
    mm = scenario.map_manager
    cache, elevations, proj = _make_cache(scenario)
    canvas_w, canvas_h = cache.canvas_dims()
    cache.render_rect(0, 0, canvas_w, canvas_h)
    stale = cache.corner_rise.copy()

    dirty, _changed = _edit_and_dirty(mm, mm.map_width // 4, mm.map_height // 4, 3)
    bbox = dirty_screen_bbox_sloped(scenario, dirty, elevations, proj, with_units=True)
    cache.patch(bbox)

    expected = iso_geometry.corner_rise_px(elevations, proj, rule=SLOPE_CORNER_RULE)
    assert not np.array_equal(stale, expected), "edit changed no corner heights -- fixture is broken"
    assert np.array_equal(cache.corner_rise, expected)


def test_bbox_clamps_to_the_sloped_canvas_not_the_padded_one():
    """Check 5. The one divergence from dirty_screen_bbox_iso: Sloped's cache
    canvas carries no skirt headroom, so a bbox clamped to Stepped's padded
    height would hand patch()/invalidate_region() rows the Sloped canvas does
    not have.

    Edited at the DEEPEST tile on screen, swept for rather than assumed to
    be (w-1, h-1) -- it is the west corner (0, h-1) in this projection, and
    picking the wrong one silently tests nothing, since the two clamps agree
    everywhere the sweep stops short of canvas_h."""
    scenario = _load_sloped_scenario()
    mm = scenario.map_manager
    cache, elevations, proj = _make_cache(scenario)
    canvas_w, canvas_h = cache.canvas_dims()

    _deepest, ex, ey = max(
        (iso_geometry.tile_screen_bounds_swept(x, y, proj)[3], x, y)
        for y in range(mm.map_height)
        for x in range(mm.map_width)
    )
    dirty, _changed = _edit_and_dirty(mm, ex, ey, 3)
    sloped = dirty_screen_bbox_sloped(scenario, dirty, elevations.copy(), proj, with_units=True)
    stepped = dirty_screen_bbox_iso(scenario, dirty, elevations.copy(), proj, with_units=True)
    assert sloped is not None and stepped is not None

    assert sloped[2] <= canvas_w and sloped[3] <= canvas_h
    assert stepped[3] > sloped[3], (
        "the two clamps agree here, so this fixture no longer exercises the divergence"
    )


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_sloped_window_carries_an_elevation_snapshot():
    """Check 6. _apply_dirty()'s Flat branch opens with an assert that the
    snapshot is None. Sloped used to satisfy it, so a Sloped edit would have
    silently taken the Flat path and patched axis-aligned rects onto an
    isometric canvas -- a wrong-pixels bug, not an exception."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        window.terrain_style_combo.setCurrentText("Sloped")
        assert window._terrain_style == "sloped"
        assert window._iso_elevations is not None
        assert window._iso_proj is not None
        # The same array object the cache holds, not a copy -- the in-place
        # mutation contract only works if every holder shares one.
        assert window._iso_elevations is window._cache.elevations
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_undo_while_sloped_is_showing_repaints_correctly():
    """Check 7 -- a UI path end to end through ViewerWindow.undo().
    Pre-Step-3 this took _apply_dirty()'s Flat branch and patched axis-aligned
    (x*tile_px, y*tile_px) squares onto an isometric canvas, so the assertion
    below is a real regression guard, not a wiring check.

    Still worth keeping now that Step 4 has released the tools and check 8
    covers a real stroke: Undo enters _apply_dirty() from
    ViewerWindow._move_history() rather than from a stroke commit, and it is
    the only route that was ever reachable with the tool gate on -- so it is
    the one that must keep working if that gate is ever reinstated."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        scenario = window.scenario
        mm = scenario.map_manager
        ex, ey = mm.map_width // 4, mm.map_height // 4

        # A real stroke through the history, as the Elevate tool commits one.
        window.edit_history.begin_stroke(mm.terrain)
        set_tile_elevation(mm, ex, ey, 3)
        window.edit_history.commit_stroke("Elevate", mm.terrain)
        window._update_edit_actions()

        window.terrain_style_combo.setCurrentText("Sloped")
        assert window._terrain_style == "sloped"
        assert window.undo_action.isEnabled(), (
            "Undo is gated in Sloped now -- this test's whole premise is that it isn't"
        )

        cache = window._cache
        canvas_w, canvas_h = cache.canvas_dims()
        cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk

        window.undo()
        assert int(mm.get_tile(ex, ey).elevation) != 3, "undo did not revert the edit"

        stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
        full = render_terrain_sloped(scenario, with_units=True)
        assert np.array_equal(stitched, full[:canvas_h, :canvas_w])
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_real_elevate_click_in_sloped_edits_the_hovered_tile():
    """Check 8 -- Step 4's own route, and the only check in this file that
    runs the whole chain the user's hand actually drives: a QMouseEvent into
    MapView, _pos_on_map/_pick_tile answering from the pick plane,
    _touch_tile -> on_edit_stroke_tile -> _apply_dirty's sloped branch, and
    the same byte-identity oracle as check 1 at the end of it.

    Driven through real events rather than window.on_edit_stroke_tile(),
    matching tests/test_brush_stroke.py's own
    test_elevate_drag_raises_each_tile_at_most_once_per_stroke: calling the
    handler directly would skip exactly the half Step 4 released, since the
    handler was reachable all along and the click was not.

    The elevations are ramped BEFORE the style switch on purpose. On a flat
    map sloped_quad_indices returns diamond_indices' arrays verbatim, so a
    click there would exercise zero resample code -- the same trap
    tests/test_sloped_pick.py's delegation guard is explicit about.

    What this does NOT prove: that the tile under the user's eye is the tile
    the plane reports. That is visual fidelity (Track A/C6), and no
    off-engine oracle reaches it -- which is why C4 lands with a "ready for
    testing" bullet rather than a closed one."""
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication

    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        scenario = window.scenario
        mm = scenario.map_manager
        for tile in mm.terrain:
            tile.elevation = 1 if tile.x >= mm.map_width // 2 else 0

        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.terrain_style_combo.setCurrentText("Sloped")
        assert window.elevation_action.isEnabled(), "Step 4 has not released Elevate"

        # A real paint cycle, so map_view carries the fit-to-view transform
        # mapFromScene() below needs -- see tests/test_sloped_viewer.py's
        # own _show_and_settle() for why showing map_view alone is not
        # enough.
        window.resize(300, 300)
        window.show()
        QApplication.processEvents()
        QApplication.processEvents()

        map_view, cache = window.map_view, window._cache
        canvas_w, canvas_h = cache.canvas_dims()
        scene_pt = QPointF(canvas_w // 2, canvas_h // 2)
        tile = map_view._pick_tile(scene_pt)
        assert tile is not None, "the canvas centre must land on a tile"
        before = int(mm.get_tile(*tile).elevation)

        cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk
        view_pt = QPointF(map_view.mapFromScene(scene_pt))
        map_view.mousePressEvent(
            QMouseEvent(QEvent.MouseButtonPress, view_pt, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        )
        map_view.mouseReleaseEvent(
            QMouseEvent(QEvent.MouseButtonRelease, view_pt, Qt.LeftButton, Qt.NoButton, Qt.NoModifier)
        )

        assert int(mm.get_tile(*tile).elevation) == before + 1, (
            f"the click did not raise the tile the pick plane reported, {tile}"
        )
        assert len(window.edit_history.records) == 1, "the stroke did not commit exactly one record"

        stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
        full = render_terrain_sloped(scenario, with_units=True)
        assert np.array_equal(stitched, full[:canvas_h, :canvas_w])
    finally:
        window.edit_history.mark_saved()
        window.close()
