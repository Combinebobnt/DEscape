"""TASK-029's owed in-app pass, offscreen: one-level raises and lowers of a
tile on flat ground, through a real ViewerWindow's _apply_dirty, at the
elev_step_pct stops the bbox-y0-slack plan names (Stepped 25/50/200, Sloped 25).

tests/test_bbox_elev_sweep.py's one-level cases call the bbox function and
patch() directly; these travel the viewer's own branch (its with_sprites and
flatten arguments) against chunks warmed before the edit, which is where a
too-short bbox top leaves a stale strip above the apex.
Dropping shadow_reach_px alone stays green (the 1-tile dilation ring covers
the wedge). Dropping it and a tile height more turns the raises red at
Stepped pct 25/50 and Sloped pct 25; the lowers and Stepped pct 200 stay green.
The level sibling repeats it at mips -1 and -2 and the finer +1 (both styles
since TASK-146), the visible level patched in place, and the same break turns
the same cases red there.

Every ViewerWindow here closes through conftest.close_window().
"""

from __future__ import annotations

import numpy as np
import pytest
from test_bbox_elev_sweep import EDIT, ONE_LEVEL_BASE, WINDOW_RADIUS, _make_cache, _window

from descape import iso_geometry, settings, viewer
from descape.elevation_tools import set_tile_elevation

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _edit_tiles(style, delta):
    # Sloped's "max" corner rule: a lone lowered tile paints nothing, so lower a 2x2 block.
    ex, ey = EDIT
    if style == "sloped" and delta < 0:
        return [(ex + dx, ey + dy) for dx in (0, 1) for dy in (0, 1)]
    return [EDIT]


CASES = [
    ("stepped", 25),
    ("stepped", 50),
    ("stepped", 200),  # Guard: green under every bbox break so far.
    ("sloped", 25),
]
# Guard: lower_1 is green under every bbox break so far.
DELTAS = [1, -1]


@pytest.mark.parametrize("delta", DELTAS, ids=["raise_1", "lower_1"])
@pytest.mark.parametrize("style, pct", CASES, ids=[f"{s}_pct{p}" for s, p in CASES])
def test_viewer_one_level_edit_patches_to_a_fresh_render(monkeypatch, style, pct, delta):
    """Pass conditions, at mip 0 in a window around the edit:
    - mip 0 is the level viewport_chunk_target() picks (else _apply_dirty evicts it, not patches)
    - the viewer's _apply_dirty reached its style's dirty_screen_bbox function exactly once
    - the edit changed pixels in the window (not vacuous)
    - the changed pixels sit strictly inside the window (the window does not bound the diff)
    - the viewer's bbox contains every changed pixel, including the rows above the apex
    - every chunk warmed over the diff is still cached after the edit (patched in place, not evicted)
    - the live cache, warmed before the edit and patched by _apply_dirty, equals a fresh render
    """
    from PyQt5.QtGui import QTransform

    from testkit.qt_window import show_and_settle

    # Never the setter: it writes the real ini.
    monkeypatch.setattr(settings, "_elev_step_pct", pct)
    window = conftest.blank_window()
    try:
        mm = window.scenario.map_manager
        for tile in mm.terrain:
            tile.elevation = ONE_LEVEL_BASE
        window.terrain_style_combo.setCurrentText(style.capitalize())
        window.refresh_map()
        assert window._render_style == style
        cache = window._cache
        proj = window._iso_proj
        assert proj.elev_step == max(1, round(proj.half_h * pct / 100)), "pct did not reach the projection"
        show_and_settle(window, 800)
        window.map_view.setTransform(QTransform())
        window.map_view.centerOn(*iso_geometry.tile_screen_origin(*EDIT, ONE_LEVEL_BASE, proj))
        target = window.map_view.viewport_chunk_target()
        assert target is not None and target[0] == 0, f"viewport picks {target}, not mip 0"

        wx0, wy0, wx1, wy1 = _window(proj, cache.canvas_dims(), WINDOW_RADIUS)
        before = cache.render_rect(wx0, wy0, wx1, wy1).copy()  # warms the chunks the patch must fix
        warmed = cache.chunk_index_range(0, wx0, wy0, wx1, wy1)

        bboxes: list = []
        name = "dirty_screen_bbox_iso" if style == "stepped" else "dirty_screen_bbox_sloped"
        real = getattr(viewer, name)

        def recording(*args, **kwargs):
            bboxes.append(real(*args, **kwargs))
            return bboxes[-1]

        monkeypatch.setattr(viewer, name, recording)

        def mutate():
            for x, y in _edit_tiles(style, delta):
                set_tile_elevation(mm, x, y, ONE_LEVEL_BASE + delta)

        dirty = window.edit_history.apply("Elevate" if delta > 0 else "Lower", mm.terrain, mutate)
        assert dirty, "fixture edit changed nothing"
        window._apply_dirty(dirty)
        assert len(bboxes) == 1 and bboxes[0] is not None, f"viewer bbox calls: {bboxes}"
        cx0, cy0, cx1, cy1 = warmed
        resident = {(cx, cy) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1) if cache.has_chunk(0, cx, cy)}
        patched = cache.render_rect(wx0, wy0, wx1, wy1)

        fresh_cache, _elev, _proj = _make_cache(style, window.scenario, cache.sprites_enabled)
        fresh = fresh_cache.render_rect(wx0, wy0, wx1, wy1)
    finally:
        conftest.close_window(window)

    ys, xs = np.nonzero((before != fresh).any(axis=2))
    assert ys.size, "the edit changed no pixels in the window -- fixture is vacuous"
    dx0, dy0 = int(xs.min()) + wx0, int(ys.min()) + wy0
    dx1, dy1 = int(xs.max()) + wx0 + 1, int(ys.max()) + wy0 + 1
    assert dx0 > wx0 and dy0 > wy0 and dx1 < wx1 and dy1 < wy1, (
        f"diff {dx0, dy0, dx1, dy1} touches window {wx0, wy0, wx1, wy1}; widen the window radius"
    )
    bx0, by0, bx1, by1 = bboxes[0]
    assert bx0 <= dx0 and by0 <= dy0 and bx1 >= dx1 and by1 >= dy1, (
        f"changed pixels {dx0, dy0, dx1, dy1} escape the viewer's bbox {bboxes[0]}"
    )
    cx0, cy0, cx1, cy1 = cache.chunk_index_range(0, dx0, dy0, dx1, dy1)
    evicted = [(cx, cy) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1) if (cx, cy) not in resident]
    assert not evicted, f"mip 0 chunks {evicted} were evicted, not patched -- the patch went untested"
    stale = np.nonzero((patched != fresh).any(axis=2))[0]
    assert not stale.size, f"stale rows {stale.min() + wy0}..{stale.max() + wy0} after the viewer's patch"


# Sloped has Stepped's ladder, +1 included, since TASK-146.
LEVEL_CASES = [(s, p, mip) for s, p in CASES for mip in (-1, -2, 1)]


@pytest.mark.parametrize("delta", DELTAS, ids=["raise_1", "lower_1"])
@pytest.mark.parametrize("style, pct, mip", LEVEL_CASES, ids=[f"{s}_pct{p}-{m}" for s, p, m in LEVEL_CASES])
def test_viewer_one_level_edit_patches_a_coarse_level_to_a_fresh_render(monkeypatch, style, pct, delta, mip):
    """The same edit with the window shown and zoomed to another level (out to
    -1/-2, or in to the finer +1 the viewer picks at max zoom), so _apply_dirty
    passes that level as rebuild_levels, as when panning zoomed out.
    Pass conditions, at `mip` in the edit window converted to that level:
    - the level is on the ladder and is the one viewport_chunk_target() picks
    - the viewer's _apply_dirty reached its style's dirty_screen_bbox function exactly once
    - the edit changed pixels in the window (not vacuous)
    - the changed pixels sit strictly inside the window (the window does not bound the diff)
    - the viewer's bbox, converted to the level, contains every changed pixel
    - every chunk warmed over the diff is still cached after the edit (patched in place, not evicted)
    - the live cache, warmed at `mip` before the edit, equals a fresh render at `mip`
    """
    from PyQt5.QtGui import QTransform

    from testkit.qt_window import show_and_settle

    monkeypatch.setattr(settings, "_elev_step_pct", pct)
    window = conftest.blank_window()
    try:
        mm = window.scenario.map_manager
        for tile in mm.terrain:
            tile.elevation = ONE_LEVEL_BASE
        window.terrain_style_combo.setCurrentText(style.capitalize())
        window.refresh_map()
        assert window._render_style == style
        show_and_settle(window, 800)
        cache = window._cache
        assert mip in cache.mip_levels(), cache.mip_levels()
        window.map_view.setTransform(QTransform.fromScale(2.0**mip, 2.0**mip))
        window.map_view.centerOn(*iso_geometry.tile_screen_origin(*EDIT, ONE_LEVEL_BASE, window._iso_proj))
        target = window.map_view.viewport_chunk_target()
        assert target is not None and target[0] == mip, f"viewport picks {target}, not mip {mip}"

        wx0, wy0, wx1, wy1 = cache._bbox_to_level(mip, _window(window._iso_proj, cache.canvas_dims(), WINDOW_RADIUS))
        before = cache.render_rect(wx0, wy0, wx1, wy1, mip=mip).copy()  # warms the chunks the patch must fix

        bboxes: list = []
        name = "dirty_screen_bbox_iso" if style == "stepped" else "dirty_screen_bbox_sloped"
        real = getattr(viewer, name)

        def recording(*args, **kwargs):
            bboxes.append(real(*args, **kwargs))
            return bboxes[-1]

        monkeypatch.setattr(viewer, name, recording)

        def mutate():
            for x, y in _edit_tiles(style, delta):
                set_tile_elevation(mm, x, y, ONE_LEVEL_BASE + delta)

        dirty = window.edit_history.apply("Elevate" if delta > 0 else "Lower", mm.terrain, mutate)
        assert dirty, "fixture edit changed nothing"
        window._apply_dirty(dirty)
        assert len(bboxes) == 1 and bboxes[0] is not None, f"viewer bbox calls: {bboxes}"
        fresh_cache, _elev, _proj = _make_cache(style, window.scenario, cache.sprites_enabled)
        fresh = fresh_cache.render_rect(wx0, wy0, wx1, wy1, mip=mip)

        ys, xs = np.nonzero((before != fresh).any(axis=2))
        assert ys.size, "the edit changed no pixels in the window -- fixture is vacuous"
        dx0, dy0 = int(xs.min()) + wx0, int(ys.min()) + wy0
        dx1, dy1 = int(xs.max()) + wx0 + 1, int(ys.max()) + wy0 + 1
        cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, dx0, dy0, dx1, dy1)
        evicted = [(cx, cy) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1) if not cache.has_chunk(mip, cx, cy)]
        patched = cache.render_rect(wx0, wy0, wx1, wy1, mip=mip)
    finally:
        conftest.close_window(window)

    assert dx0 > wx0 and dy0 > wy0 and dx1 < wx1 and dy1 < wy1, (
        f"diff {dx0, dy0, dx1, dy1} touches window {wx0, wy0, wx1, wy1}; widen the window radius"
    )
    bx0, by0, bx1, by1 = cache._bbox_to_level(mip, bboxes[0])
    assert bx0 <= dx0 and by0 <= dy0 and bx1 >= dx1 and by1 >= dy1, (
        f"changed pixels {dx0, dy0, dx1, dy1} escape the viewer's bbox {bx0, by0, bx1, by1} at mip {mip}"
    )
    assert not evicted, f"mip {mip} chunks {evicted} were evicted, not patched -- the patch went untested"
    stale = np.nonzero((patched != fresh).any(axis=2))[0]
    assert not stale.size, f"mip {mip}: stale rows {stale.min() + wy0}..{stale.max() + wy0} after the viewer's patch"
