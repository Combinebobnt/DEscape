"""The geometry <-> render <-> LIVE VIEWER link tests/test_seam_line.py's
own docstring admits is missing: every test there checks index arrays or
an off-engine render.py call. This is a real, shown ViewerWindow driven
headlessly (QT_QPA_PLATFORM=offscreen, via conftest.ensure_qapp())
and captured through an actual QGraphicsScene.render() paint dispatch, not
descape.render.render_terrain_iso() called standalone.

Same technique tools/verify_iso_viewer_pick.py established, written
natively as pytest per tests/README.md's stated convention ("New tools/
verify_*.py scripts are not the convention going forward") -- mirroring
tests/test_lazy_viewport.py/test_fill_tool.py/test_brush_stroke.py, not
the corpus-marked legacy-adapter pattern. Runs against the shipped blank
120x120 template only -- no examples/ corpus needed, so this stays in the
default tier.

Every ViewerWindow() constructed here (via conftest.stepped_window) must
call edit_history.mark_saved() before close() -- see that helper's own
docstring for why a dirty document's close() would otherwise hang on a
modal QMessageBox with nothing offscreen to click it.
"""

from __future__ import annotations

import numpy as np
import pytest

import conftest
from descape import iso_geometry as ig
from descape import render
from descape import settings
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# Interior anchor for every fixture in this file -- matches tests/
# test_lazy_viewport.py's own PROBE_ORIGIN=(64, 64) on this SAME 120x120
# blank template, not a coincidence: MapView.set_source() draws a
# decorative "map extent" outline (ground_outline_corners(), a Qt-scene-
# only QPen(200, 200, 200) polygon tracing the whole map's ground-level
# silhouette, drawn regardless of Terrain Style or actual elevation
# content) that render_terrain_iso() never draws at all. A probe placed
# near tile (10, 10) on a 120x120 map sits close enough to the map's own
# (0, 0) corner that this outline's edge cuts straight through it --
# confirmed by capturing that exact case: 161 stray grey (200, 200, 200)
# pixels, a thin diagonal line, where the live capture disagreed with an
# independent render_terrain_iso() crop. Not a rendering bug -- the
# outline is real, intentional Qt-side decoration; the fix is keeping
# every probe well inside the map, away from any edge, like this file's
# only sibling that already does the same comparison.
_ANCHOR = (60, 60)


def _set_pyramid(scenario, center=_ANCHOR, step: int = 1) -> None:
    """Same shape as tests/test_seam_line.py's own _pyramid_scenario, but
    applied to an already-loaded live scenario's tiles in place rather than
    building a fresh LoadedScenario -- conftest.stepped_window only takes a
    path, and reloading from a second on-disk fixture just to get a hill
    would be a second, unrelated code path for no benefit."""
    cx, cy = center
    for tile in scenario.map_manager.terrain:
        rings = max(0, 3 - max(abs(tile.x - cx), abs(tile.y - cy)))
        tile.elevation = min(ig.MAX_ELEVATION, step * rings)


def _bbox_for_tiles(proj, tile_px: int, tiles, max_elev: int, pad_tiles: int = 1):
    """A capture rect covering `tiles`' diamonds at both elevation 0 and
    `max_elev`, independent of anything the app itself computes.

    Deliberately NOT derived from descape.render.dirty_screen_bbox_iso --
    that is exactly the function under test in
    test_a_real_mouse_drag_elevation_edit_keeps_the_seam_byte_identical
    below, and a capture rect sized by the same function it's supposed to
    be catching bugs in would hide an under-sized dilation instead of
    exposing it (see that test's own docstring). tile_screen_origin's only
    elevation-dependent term is `-elevation * proj.elev_step`, so a tile's
    screen position only ever moves UP as elevation rises -- checking just
    the two endpoints 0 and max_elev is enough to bound every intermediate
    elevation's extent too."""
    xs, ys = [], []
    for tx, ty in tiles:
        for e in (0, max_elev):
            sx, sy = ig.tile_screen_origin(tx, ty, e, proj)
            xs += [sx, sx + 2 * proj.half_w]
            ys += [sy, sy + 2 * proj.half_h]
    pad = pad_tiles * tile_px
    return max(0, min(xs) - pad), max(0, min(ys) - pad), max(xs) + pad, max(ys) + pad


def test_stepped_scene_render_matches_a_full_render_at_pct_200():
    """The render-touching test test_seam_line.py's own module docstring
    says nothing here mechanically guards: byte-identity between a real
    QGraphicsScene.render() capture (the actual paint dispatch a user
    sees) and an independent crop of render_terrain_iso() -- so the
    geometry could be perfect and the live viewer could still draw it at
    the wrong origin, on the wrong tiles, or not at all, and this is what
    would catch it.

    pct=200 is the case that matters: the contact-shadow band is empty
    there, so the seam is the only up-screen cue and any mismatch is
    unambiguously its own -- test_seam_line.py's own render test picked
    the same pct for the same reason.
    """
    from PyQt5.QtCore import QRectF

    settings.set_graphics_quality(settings.GRAPHICS_QUALITY_DEFAULT)
    window = conftest.stepped_window(
        BLANK_TEMPLATE_PATH, elev_step_pct=200, graphics_quality=settings.GRAPHICS_QUALITY_DEFAULT
    )
    try:
        _set_pyramid(window.scenario)
        window._render_current()

        proj = window.map_view._iso_proj
        assert proj is not None, "Stepped mode produced no projection"
        tile_px = proj.tile_px
        cx, cy = _ANCHOR
        tiles = [(x, y) for x in range(cx - 3, cx + 4) for y in range(cy - 3, cy + 4)]
        x0, y0, x1, y1 = _bbox_for_tiles(proj, tile_px, tiles, max_elev=3)
        probe = QRectF(x0, y0, x1 - x0, y1 - y0)

        got = conftest.scene_rect_to_array(window.map_view.scene(), probe)
        full = render.render_terrain_iso(window.scenario, with_units=False)
        px0, py0 = int(probe.left()), int(probe.top())
        px1, py1 = min(px0 + got.shape[1], full.shape[1]), min(py0 + got.shape[0], full.shape[0])
        expected = full[py0:py1, px0:px1]
        got = got[: py1 - py0, : px1 - px0]

        assert got.shape == expected.shape, f"crop shape mismatch: {got.shape} vs {expected.shape}"
        if not np.array_equal(got, expected):
            diff = int(np.count_nonzero(np.any(got != expected, axis=2)))
            raise AssertionError(
                f"{diff}/{got.shape[0] * got.shape[1]} pixels differ from an independent full render"
            )

        last_mip = window.map_view._canvas_item._last_mip
        assert last_mip == 0, f"capture wasn't 1:1 -- selected mip {last_mip}, byte-identity is meaningless"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_real_mouse_drag_elevation_edit_keeps_the_seam_byte_identical():
    """The untested hop: real QMouseEvents so MapView._touch_tile ->
    ViewerWindow.on_edit_stroke_tile -> _apply_dirty ->
    descape.render.dirty_screen_bbox_iso -> IsoChunkCache.patch ->
    MapCanvasItem.paint all run -- not a direct on_edit_stroke_tile() call,
    which every existing brush test uses and which never reaches
    dirty_screen_bbox_iso's own dilation math at all.

    Target tile T=(10,10) starts at elevation 0; its down-screen neighbour
    D=(10,11) starts one level higher, so D's up_left seam (which faces T)
    is drawn BEFORE the edit. Raising T by one level (a single left-click,
    brush size 1, no modifiers) makes T >= D, so D's up_left seam must
    disappear -- while T's OWN up_left/up_right/apex seams newly appear,
    since T is now higher than ITS neighbours (T.y-1) and (T.x+1, T.y),
    which stayed at 0. One edit, one erased seam and one newly-drawn one,
    both inside the same dirty region the app's own bbox has to cover.

    If the incrementally-patched capture doesn't match a fresh full render
    of the same post-edit scenario, that is the payoff this test exists
    for, not a test bug: diagnose _apply_dirty/IsoChunkCache.patch, do not
    widen the captured rect to make the mismatch fall outside it.

    Confirmed discriminating, not just plausible: skipping
    ViewerWindow._apply_dirty's `self._cache.patch(bbox)` call entirely
    (leaving only `invalidate_region`, so Qt repaints from stale chunks)
    turns this test red. Shrinking dirty_screen_bbox_iso's own lateral
    dilation radius to 0 does NOT -- tile_screen_bounds_swept() already
    sweeps the tile's full legal elevation range (0..15 here), which alone
    produces a bbox tall enough to spatially cover the one-tile-away
    neighbour this test uses regardless of that radius. A brush stroke
    wide enough to need the lateral term itself would need its own test.
    """
    from PyQt5.QtCore import QEvent, QRectF, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication

    from descape.brush import BRUSH_SHAPE_SQUARE

    settings.set_graphics_quality(settings.GRAPHICS_QUALITY_DEFAULT)
    window = conftest.stepped_window(
        BLANK_TEMPLATE_PATH, elev_step_pct=200, graphics_quality=settings.GRAPHICS_QUALITY_DEFAULT
    )
    try:
        mm = window.scenario.map_manager
        tx, ty = _ANCHOR
        dx, dy = tx, ty + 1
        mm.get_tile(tx, ty).elevation = 0
        mm.get_tile(dx, dy).elevation = 1
        window._render_current()

        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        assert window.elevation_action.isEnabled(), "elevation tool unexpectedly disabled in Stepped mode"
        window.brush_size_spin.setValue(1)
        window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(BRUSH_SHAPE_SQUARE))

        map_view = window.map_view
        proj = map_view._iso_proj
        assert proj is not None

        # T's diamond CENTER at its current (pre-edit) elevation -- where a
        # user's cursor would actually land to click this tile as displayed.
        ox, oy = ig.tile_screen_origin(tx, ty, 0, proj)
        scene_pt_press = ox + proj.half_w, oy + proj.half_h
        from PyQt5.QtCore import QPointF

        press_pos = QPointF(map_view.mapFromScene(QPointF(*scene_pt_press)))

        tile_px = proj.tile_px
        watch_tiles = [(tx, ty), (dx, dy), (tx, ty - 1), (tx + 1, ty)]
        x0, y0, x1, y1 = _bbox_for_tiles(proj, tile_px, watch_tiles, max_elev=2)
        watch_rect = QRectF(x0, y0, x1 - x0, y1 - y0)

        before = conftest.scene_rect_to_array(map_view.scene(), watch_rect)

        press = QMouseEvent(QEvent.MouseButtonPress, press_pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        map_view.mousePressEvent(press)
        release = QMouseEvent(
            QEvent.MouseButtonRelease, press_pos, Qt.LeftButton, Qt.NoButton, Qt.NoModifier
        )
        map_view.mouseReleaseEvent(release)
        QApplication.processEvents()

        assert mm.get_tile(tx, ty).elevation == 1, "the click didn't raise T -- fixture assumption broke"
        assert mm.get_tile(dx, dy).elevation == 1, "D moved too -- fixture assumption broke"

        after = conftest.scene_rect_to_array(map_view.scene(), watch_rect)
        assert not np.array_equal(before, after), "the edit changed nothing in the watched region"

        full = render.render_terrain_iso(window.scenario, with_units=False)
        px0, py0 = int(watch_rect.left()), int(watch_rect.top())
        px1, py1 = min(px0 + after.shape[1], full.shape[1]), min(py0 + after.shape[0], full.shape[0])
        expected = full[py0:py1, px0:px1]
        got = after[: py1 - py0, : px1 - px0]

        assert got.shape == expected.shape, f"crop shape mismatch: {got.shape} vs {expected.shape}"
        if not np.array_equal(got, expected):
            diff = int(np.count_nonzero(np.any(got != expected, axis=2)))
            raise AssertionError(
                f"{diff}/{got.shape[0] * got.shape[1]} pixels differ between the incrementally-patched "
                "live view and a fresh full render of the same post-edit scenario"
            )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_flat_ground_draws_no_seam_through_the_viewer():
    """The live-viewer counterpart to tests/test_seam_line.py's own
    test_flat_ground_draws_no_seam_at_all, which only ever calls
    render_terrain_iso() directly. Same A/B: capture once with SEAM_SHADE
    at its real value, neutralise it to an exact no-op and force a fresh
    IsoChunkCache (a new one, not the memoized old one -- _render_current()
    rebuilds it), capture again. On perfectly flat terrain the `delta <= 0`
    guard in render._render_tile_iso should make the seam a no-op
    regardless of SEAM_SHADE's value, so the two captures must be
    byte-identical.
    """
    settings.set_graphics_quality(settings.GRAPHICS_QUALITY_DEFAULT)
    window = conftest.stepped_window(
        BLANK_TEMPLATE_PATH, elev_step_pct=100, graphics_quality=settings.GRAPHICS_QUALITY_DEFAULT
    )
    try:
        for tile in window.scenario.map_manager.terrain:
            tile.elevation = 0
        window._render_current()

        proj = window.map_view._iso_proj
        assert proj is not None
        cx, cy = _ANCHOR
        watch_tiles = [(cx, cy), (cx, cy + 1), (cx + 1, cy)]
        watch_rect_args = _bbox_for_tiles(proj, proj.tile_px, watch_tiles, max_elev=0)
        from PyQt5.QtCore import QRectF

        x0, y0, x1, y1 = watch_rect_args
        watch_rect = QRectF(x0, y0, x1 - x0, y1 - y0)

        with_seam = conftest.scene_rect_to_array(window.map_view.scene(), watch_rect)

        original = render.SEAM_SHADE
        try:
            render.SEAM_SHADE = 1.0
            render._seam_factors.cache_clear()
            window._render_current()
            without_seam = conftest.scene_rect_to_array(window.map_view.scene(), watch_rect)
        finally:
            render.SEAM_SHADE = original
            render._seam_factors.cache_clear()

        assert np.array_equal(with_seam, without_seam), "a flat map drew a seam through the live viewer"
    finally:
        window.edit_history.mark_saved()
        window.close()
