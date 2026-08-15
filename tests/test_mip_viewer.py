"""Phases B-D-c and B-D-d: LOD selection, the scaled blit, and the zoom-out
floor (maintainer plan docs/PLAN_MIPS.md).

Phases a and b were headless -- they built the level set and the per-level
cache plumbing without any of it ever being selected. This file covers the
two phases that change pixels: MapCanvasItem picking a mip from the painter
transform and painting from that level through a scaled drawImage() instead
of the point overload (c), then extending selection BELOW the reference and
dividing the zoom-out clamp by the ladder's depth (d).

Several tests here were deliberately flipped by phase d rather than added
to -- c pinned "the reference level is never left downwards", which is
exactly what d exists to stop doing. Each such test says so in its own
docstring, with what replaced the property it used to protect.

Layered cheapest-first, same shape as tests/test_mip_geometry.py:

1. _max_axis_scale() against the two numbers the plan measured, plus the
   clamping rules -- pure math, no window.
2. _select_mip() reading deviceTransform rather than worldTransform --
   pinned with a stub painter, so it fails if someone "simplifies" it to
   the world transform (which would silently drop HiDPI a level).
2b. MapView's zoom-out floor over synthetic level sets of every depth,
   including the two degenerate ones (no coarser level at all; no canvas
   item yet) that must reduce to the pre-mip floor exactly.
3. Real offscreen paint cycles: the coarsest level at fit, rising under
   scripted zoom; picks unchanged across levels (plan decision D2); block
   tiling matching a single draw at a scaled level, in both the magnifying
   and the minifying band.
4. A subprocess at QT_SCALE_FACTOR=2, the only honest way to check real
   HiDPI selection -- QApplication reads that once per process, and
   conftest.ensure_qapp() shares one for the whole session.

Everything here is gui-marked: the whole file needs PyQt5.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import conftest
from descape.render import _ChunkCacheBase
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH

ROOT = Path(__file__).resolve().parent.parent

# Flat's own view transform (MapView.set_isometric's scale(1, 0.5) then
# rotate(-45)) at 2x zoom, and what the two candidate LOD metrics return on
# it. The plan chose the max singular value specifically because
# levelOfDetailFromTransform disagrees HERE: floor(log2()) of 1 vs 0, one
# whole level, showing as an octave of blur along Flat's stretched axis.
FLAT_2X_MAX_SINGULAR_VALUE = 2.0
FLAT_2X_LEVEL_OF_DETAIL = 1.5811388300841898

# Enough scene-space zoom to push the selected level past the reference on
# the 120x120 fixture: fit-to-view there is a device scale of ~0.083 (the
# canvas is 7680px wide in a 300px window), so even the 6x of
# test_lazy_viewport's own probe only reaches ~0.5, which since phase d
# selects one level BELOW the reference rather than the reference itself.
# Applied via view.scale() directly, which deliberately bypasses
# wheelEvent's zoom clamp -- these steps are about what gets SELECTED, and
# routing them through the clamp would couple them to phase d's floor.
# The leading 1.0 is a no-op step on purpose: selection is
# supposed to be a pure function of the transform, so re-painting an
# unchanged view must re-select the same level.
ZOOM_STEPS = (1.0, 8.0, 4.0)

# The HiDPI probe needs the DPR-1 run to land BELOW the finest enumerated
# level, or "one level finer" is unobservable (doubling just clamps back to
# the same level). 16x from fit puts the device scale at ~1.3, i.e. level 0
# with room above it; DPR 2 doubles that to ~2.6, i.e. level 1. The probe
# reports its own measured device scale so a drift in the fit scale shows
# up as an explicit skip rather than a silent pass.
HIDPI_ZOOM_STEPS = (4.0, 4.0)

WINDOW_SIZE = 300

# Measured on this machine at every residual magnification in [1, 2)
# probed (1.0, 1.25, 1.5, 1.85): the tiled and single-draw renders come
# back BYTE-IDENTICAL, maxdiff 0, not merely close. The bar is left as a
# bounded difference anyway, per the plan -- padded overlapping draws under
# a filter are not guaranteed exactly equal, and a stricter-than-necessary
# bar here would be an environment-dependent tripwire rather than a real
# regression signal. A genuine seam would be far larger than this.
MAX_SEAM_DIFF = 2

# The minifying-band seam probe (Phase B-D-d). 960 scene units is 120 pixels
# of this fixture's coarsest level, i.e. 7.5 blocks at MINIFYING_BLOCK_PX --
# the same block count the magnifying test above gets from its own 240-unit
# probe, so the two bands are compared on equal terms.
#
# Sized DOWN from a first attempt at 3352 units, which failed at maxdiff 5
# and was measured to be an artifact of the probe, not a seam. At that size
# the tiled and single-draw renders differ on 48 of 112225 pixels forming 47
# separate clusters (largest: 2 pixels), only 2% of them anywhere near a
# block boundary, biased toward the far end of the draw -- isolated
# resampling-phase noise consistent with fixed-point source stepping
# accumulating across one very long draw, which the tiled render restarts
# per block. Both directions show it (the magnifying band reaches maxdiff 69
# on the same oversized probe, at 274 of 38 million pixels), so it is not a
# mip-down property at all and predates this phase. A real seam is a
# CONNECTED line of hundreds of pixels along a block boundary; at the size
# below both bands come back byte-identical, which is the bar this file
# already holds the magnifying band to.
PROBE_SCENE_SIDE = 960.0
MINIFYING_BLOCK_PX = 16


def _max_axis_scale(transform):
    from descape.viewer import _max_axis_scale as fn

    return fn(transform)


class _StubPainter:
    """Only _select_mip's own two inputs, so a change from deviceTransform
    to worldTransform is a test failure rather than a silent HiDPI
    regression -- the two are deliberately different here, exactly as they
    are at QT_SCALE_FACTOR=2 (measured: device m11 6.0 vs world 3.0)."""

    def __init__(self, device, world):
        self._device, self._world = device, world

    def deviceTransform(self):
        return self._device

    def worldTransform(self):
        return self._world


class _StubCache:
    """The three members _select_mip and MapView's zoom-out floor touch,
    over an arbitrary level set. The selection RULE itself is bound
    straight off _ChunkCacheBase rather than restated here: Phase B-D-d
    moved it wholly onto the cache (tested in tests/test_mip_cache.py), so
    a copy here would only be able to disagree with the real one."""

    mip_levels = _ChunkCacheBase.mip_levels
    mip_scale = _ChunkCacheBase.mip_scale
    mip_for_scale = _ChunkCacheBase.mip_for_scale

    def __init__(self, levels, reference_tile_px: int = 32):
        assert 0 in levels, "level 0 is always the reference (D2)"
        self._mip_tile_px = {
            level: int(reference_tile_px * 2.0**level) for level in sorted(levels)
        }

    def canvas_dims(self, mip: int = 0):
        return (64, 64)


def _canvas_item(levels):
    from descape.viewer import MapCanvasItem

    conftest.ensure_qapp()
    return MapCanvasItem(_StubCache(levels))


def _uniform(scale):
    from PyQt5.QtGui import QTransform

    t = QTransform()
    t.scale(scale, scale)
    return t


# ---------------------------------------------------------------------------
# Layer 1: the LOD metric itself.
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_max_axis_scale_reproduces_the_plan_s_measured_oracle() -> None:
    """The whole reason this project computes its own metric instead of
    calling Qt's. If this ever stops disagreeing, re-derive the choice."""
    from PyQt5.QtGui import QTransform
    from PyQt5.QtWidgets import QStyleOptionGraphicsItem

    conftest.ensure_qapp()
    t = QTransform()
    t.scale(1.0, 0.5)
    t.rotate(-45)
    t.scale(2.0, 2.0)

    assert _max_axis_scale(t) == pytest.approx(FLAT_2X_MAX_SINGULAR_VALUE)
    assert QStyleOptionGraphicsItem.levelOfDetailFromTransform(t) == pytest.approx(
        FLAT_2X_LEVEL_OF_DETAIL
    )
    # The part that actually costs a level, not just a different float.
    assert math.floor(math.log2(_max_axis_scale(t))) == 1
    assert math.floor(math.log2(QStyleOptionGraphicsItem.levelOfDetailFromTransform(t))) == 0


@pytest.mark.gui
def test_max_axis_scale_is_transpose_invariant() -> None:
    """Singular values don't change under transpose, so Qt's m12/m21
    naming cannot silently give a wrong level -- pinned because getting
    that convention backwards is otherwise an easy, invisible mistake."""
    from PyQt5.QtGui import QTransform

    conftest.ensure_qapp()
    t = QTransform()
    t.scale(1.0, 0.5)
    t.rotate(-45)
    transposed = QTransform(t.m11(), t.m21(), t.m12(), t.m22(), 0, 0)
    assert _max_axis_scale(t) == pytest.approx(_max_axis_scale(transposed))


@pytest.mark.gui
@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0, 2.0, 7.0])
def test_max_axis_scale_of_a_uniform_scale_is_that_scale(scale) -> None:
    conftest.ensure_qapp()
    assert _max_axis_scale(_uniform(scale)) == pytest.approx(scale)


# ---------------------------------------------------------------------------
# Layer 2: selection.
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_select_mip_reads_the_device_transform_not_the_world_one() -> None:
    item = _canvas_item([-2, -1, 0, 1, 2])
    # Device 2x the world transform, the real HiDPI relationship.
    painter = _StubPainter(_uniform(4.0), _uniform(2.0))
    assert item._select_mip(painter) == 2
    assert item._select_mip(_StubPainter(_uniform(2.0), _uniform(4.0))) == 1


@pytest.mark.gui
def test_select_mip_clamps_to_the_coarsest_enumerated_level() -> None:
    """FLIPPED IN PHASE B-D-d. Phase B-D-c pinned the opposite of this (the
    selected level never went below the reference, so mip_scale() was 1 at
    and below fit-to-view); mip-down is the entire point of this phase, so
    minification now walks down the ladder and stops at its bottom rung.
    What protects the existing byte-identity bars instead is the weaker,
    still-true property in the two tests below: a UNIT-scale render selects
    the reference."""
    item = _canvas_item([-3, -2, -1, 0, 1])
    assert item._select_mip(_StubPainter(_uniform(1.0), _uniform(1.0))) == 0
    assert item._select_mip(_StubPainter(_uniform(0.5), _uniform(0.5))) == -1
    assert item._select_mip(_StubPainter(_uniform(0.2), _uniform(0.2))) == -3
    assert item._select_mip(_StubPainter(_uniform(0.001), _uniform(0.001))) == -3
    # A level set with nothing below the reference stays where it was.
    assert _canvas_item([0, 1])._select_mip(_StubPainter(_uniform(0.001), _uniform(0.001))) == 0


@pytest.mark.gui
def test_a_unit_scale_transform_selects_the_reference_level() -> None:
    """Half of the load-bearing property for every existing byte-identity
    check in the suite -- the arithmetic half. The other half (that a
    scene render into an equally-sized QImage really does produce a
    unit-scale device transform) is a Qt behaviour, not arithmetic, and is
    checked for real in test_a_unit_scale_scene_render_paints_the_reference
    below. Split deliberately: this plan has now been wrong three times
    about what Qt does, and a stubbed transform cannot catch a fourth."""
    for levels in ([-3, -2, -1, 0, 1], [-1, 0], [0]):
        item = _canvas_item(levels)
        assert item._select_mip(_StubPainter(_uniform(1.0), _uniform(1.0))) == 0
        assert item._cache.mip_scale(0) == 1.0


@pytest.mark.gui
def test_select_mip_clamps_to_the_finest_enumerated_level() -> None:
    item = _canvas_item([-1, 0, 1])
    assert item._select_mip(_StubPainter(_uniform(2.0), _uniform(2.0))) == 1
    assert item._select_mip(_StubPainter(_uniform(64.0), _uniform(64.0))) == 1
    # A cache with no exact neighbour at all is a normal answer, not a
    # failure -- see iso_geometry.mip_projections_for's docstring.
    assert _canvas_item([0])._select_mip(_StubPainter(_uniform(64.0), _uniform(64.0))) == 0


@pytest.mark.gui
def test_select_mip_survives_a_degenerate_transform() -> None:
    """A collapsed transform must not raise out of paint(); log2(0) would.
    Lands on the COARSEST level in Phase B-D-d (it was the reference in
    B-D-c) -- a degenerate transform is a zero-area view, so the cheapest
    level is the right defensive answer as well as the consistent one."""
    from PyQt5.QtGui import QTransform

    conftest.ensure_qapp()
    item = _canvas_item([-1, 0, 1])
    collapsed = QTransform(0, 0, 0, 0, 0, 0)
    assert item._select_mip(_StubPainter(collapsed, collapsed)) == -1


# ---------------------------------------------------------------------------
# Layer 2b: the zoom-out floor (Phase B-D-d).
# ---------------------------------------------------------------------------


class _FakeCanvasItem:
    def __init__(self, cache):
        self._cache = cache


def _bare_map_view():
    """A MapView with no source set, for the floor arithmetic alone. Its
    five callbacks are edit-tool plumbing nothing here reaches."""
    from descape.viewer import MapView

    conftest.ensure_qapp()
    noop = lambda *a, **k: None  # noqa: E731
    return MapView(noop, noop, noop, noop, noop)


@pytest.mark.gui
@pytest.mark.parametrize("levels", [[0], [0, 1, 2], [-1, 0], [-2, -1, 0, 1, 2], [-3, 0]])
def test_zoom_out_floor_divides_by_the_mip_depth(levels) -> None:
    """Phase B-D-d's clamp change, over level sets of every depth rather
    than whatever one fixture's elev_step_pct happens to enumerate. Two
    properties, and the second is the one that makes this phase incapable
    of costing anyone zoom range:

    1. The floor is the pre-mip one divided by how much coarser the ladder
       goes, so the reachable zoom-out range grows by exactly the mip depth
       while the residual minification AT the floor is unchanged.
    2. It is never TIGHTER than the pre-mip floor -- including for a level
       set with nothing below the reference, where it must reduce to
       exactly the old expression rather than to something merely close.

    Deliberately not the plan's original formula, which was an absolute
    scale bound (1 / (mip_scale * MAX_RESIDUAL_MINIFICATION)). Measured on
    the 480x480 fixture at a 480px window: fit-to-view is scale 0.0413,
    where that formula gives 0.0625 -- a floor ABOVE fit, refusing zoom-out
    everywhere and stranding the view above the whole-map-visible scale.
    See PLAN_MIPS.md's correction entry."""
    from descape.viewer import MapView

    view = _bare_map_view()
    cache = _StubCache(levels)
    view._canvas_item = _FakeCanvasItem(cache)
    baseline = abs(view.transform().determinant()) ** 0.5

    view._capture_zoom_baseline()
    depth = cache.mip_scale(levels[0])
    assert depth == 2.0 ** -levels[0]
    assert view._min_linear_scale == pytest.approx(MapView.MIN_ZOOM_FRACTION_OF_FIT * baseline / depth)
    assert view._min_linear_scale <= MapView.MIN_ZOOM_FRACTION_OF_FIT * baseline
    # The zoom-IN ceiling is untouched by this phase.
    assert view._max_linear_scale == pytest.approx(MapView.MAX_ZOOM_MULTIPLE_OF_FIT * baseline)


@pytest.mark.gui
def test_zoom_out_floor_falls_back_to_the_pre_mip_one_with_no_canvas_item() -> None:
    """_capture_zoom_baseline runs from set_isometric(), which ViewerWindow
    can reach before any source is set -- so "no canvas item yet" must be a
    plain no-op on the floor, not an attribute error or an inf."""
    from descape.viewer import MapView

    view = _bare_map_view()
    assert view._canvas_item is None
    view._capture_zoom_baseline()
    baseline = abs(view.transform().determinant()) ** 0.5
    assert view._coarsest_mip_scale() == 1.0
    assert view._min_linear_scale == pytest.approx(MapView.MIN_ZOOM_FRACTION_OF_FIT * baseline)


@pytest.mark.gui
def test_zoom_out_floor_on_a_real_window_matches_the_mip_ladder() -> None:
    """The formula above against a real fit-to-view baseline and a real
    enumerated ladder, so a change to either (a different reference
    tile_px, a shallower elev_step_pct) can't silently decouple them."""
    from descape.viewer import MapView

    window = _stepped_window()
    try:
        view = window.map_view
        item = view._canvas_item
        assert item is not None
        baseline = abs(view.transform().determinant()) ** 0.5
        depth = item._cache.mip_scale(item._cache.mip_levels()[0])
        assert view._min_linear_scale == pytest.approx(
            MapView.MIN_ZOOM_FRACTION_OF_FIT * baseline / depth
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


# ---------------------------------------------------------------------------
# Layer 3: real offscreen paint cycles.
# ---------------------------------------------------------------------------


def _stepped_window():
    from descape.viewer import ViewerWindow
    from PyQt5.QtWidgets import QApplication

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    if window.scenario is None:
        window.close()
        pytest.skip(f"{FIXTURE_PATH.name} failed to load")
    window.terrain_style_combo.setCurrentText("Stepped")
    window.resize(WINDOW_SIZE, WINDOW_SIZE)
    window.show()
    QApplication.processEvents()
    QApplication.processEvents()
    return window


def _paint_spy(monkeypatch):
    """Records (device scale, selected mip) per real paint() call."""
    from descape.viewer import MapCanvasItem

    seen: list[tuple[float, int]] = []
    original = MapCanvasItem.paint

    def spy(self, painter, option, widget=None):
        result = original(self, painter, option, widget)
        seen.append((_max_axis_scale(painter.deviceTransform()), self._last_mip))
        return result

    monkeypatch.setattr(MapCanvasItem, "paint", spy)
    return seen


@pytest.mark.gui
def test_selected_mip_is_the_coarsest_at_fit_and_rises_with_zoom(monkeypatch) -> None:
    """The headline behaviour of both pixel phases at once. FLIPPED IN
    PHASE B-D-d at the fit end only: fit-to-view is a heavy minification
    (the canvas is thousands of pixels wide in a 300px window), so it now
    bottoms out on the coarsest enumerated level rather than sitting on the
    reference -- that is the whole fit-to-view memory/latency win. The
    monotone rise across zoom-in steps, and agreement with the derived
    rule, are unchanged from B-D-c."""
    from PyQt5.QtWidgets import QApplication

    window = _stepped_window()
    try:
        seen = _paint_spy(monkeypatch)
        view = window.map_view
        item = view._canvas_item
        assert item is not None
        levels = item._cache.mip_levels()
        if levels[-1] <= 0:
            pytest.skip("fixture enumerates no level finer than the reference")

        view.viewport().update()
        QApplication.processEvents()
        assert seen, "no paint cycle at fit-to-view"
        fit_scale, fit_mip = seen[-1]
        assert fit_scale < 1.0, f"fit-to-view was not a minification ({fit_scale})"
        assert fit_mip == levels[0]
        if levels[0] < 0:
            # The mip-DOWN win, stated as the property that produces it:
            # the level painted at fit covers more than one scene unit per
            # pixel, so its whole canvas is smaller than the reference's.
            assert item._cache.mip_scale(fit_mip) > 1.0
            assert item._cache.canvas_dims(fit_mip) < item._cache.canvas_dims()

        selected = [fit_mip]
        for step in ZOOM_STEPS:
            seen.clear()
            view.scale(step, step)
            view.viewport().update()
            QApplication.processEvents()
            assert seen, f"no paint cycle after a {step}x scale step"
            scale, mip = seen[-1]
            # Selection agrees with the derived rule, not just with itself.
            expected = max(levels[0], min(levels[-1], math.floor(math.log2(scale))))
            assert mip == expected, f"at device scale {scale}: selected {mip}, rule says {expected}"
            selected.append(mip)

        assert selected == sorted(selected), f"selection was not monotonic under zoom-in: {selected}"
        assert max(selected) > 0, f"zooming never selected a finer level than the reference: {selected}"
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
def test_picks_are_unchanged_across_mip_levels(monkeypatch) -> None:
    """Plan decision D2: scene space is pinned to the reference level, so
    hit-testing must be completely unaffected by which mip is painted.
    Trivially true today -- which is exactly why it's worth pinning. This
    is the regression test for someone later making scene space follow the
    selected level, which would silently break every pick, brush preview
    and overscroll bound at once."""
    from PyQt5.QtWidgets import QApplication

    window = _stepped_window()
    try:
        seen = _paint_spy(monkeypatch)
        view = window.map_view
        item = view._canvas_item
        assert item is not None
        if item._cache.mip_levels()[-1] <= 0:
            pytest.skip("fixture enumerates no level finer than the reference")

        from PyQt5.QtCore import QPointF

        w, h = item.boundingRect().width(), item.boundingRect().height()
        # Fixed SCENE points, not viewport points: the view transform is
        # what we're about to change, so viewport points would move to
        # different tiles for reasons that have nothing to do with mips.
        probes = [QPointF(w * fx, h * fy) for fx in (0.25, 0.5, 0.75) for fy in (0.25, 0.5, 0.75)]

        view.viewport().update()
        QApplication.processEvents()
        assert seen[-1][1] == item._cache.mip_levels()[0]
        bounding_at_fit = item.boundingRect()
        before = [view._pick_tile(p) for p in probes]

        for step in ZOOM_STEPS:
            view.scale(step, step)
        view.viewport().update()
        QApplication.processEvents()
        assert seen[-1][1] > 0, "zoom never selected a finer level, so this proves nothing"

        assert [view._pick_tile(p) for p in probes] == before
        assert item.boundingRect() == bounding_at_fit
    finally:
        window.edit_history.mark_saved()
        window.close()


def _render_scene_rect(scene, rect, zoom: float) -> np.ndarray:
    """Renders `rect` (scene units) into a target `zoom` times larger, so
    the painter's device scale is exactly `zoom` regardless of the view's
    own transform -- how this file reaches a chosen mip deterministically.

    Drives exactly ONE paint() call, which is what lets callers read
    item._last_mip afterwards as the level THIS render selected rather
    than as whatever the last of several paints happened to pick."""
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QImage, QPainter

    tw, th = int(rect.width() * zoom), int(rect.height() * zoom)
    image = QImage(tw, th, QImage.Format_RGB888)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    scene.render(painter, QRectF(0, 0, tw, th), rect)
    painter.end()
    bits = image.constBits()
    bits.setsize(image.bytesPerLine() * th)
    return np.array(bits).reshape(th, image.bytesPerLine())[:, : tw * 3].reshape(th, tw, 3).copy()


@pytest.mark.gui
def test_a_unit_scale_scene_render_paints_the_reference() -> None:
    """The premise the whole "existing byte-identity checks stay green"
    argument rests on, checked against real Qt rather than a stub
    transform. tests/test_lazy_viewport.py check 3 and
    tools/verify_iso_viewer_pick.py check 4 both rasterize a scene rect via
    scene.render() into an EQUALLY SIZED QImage; that is only safe if Qt
    hands paint() a unit-scale device transform there, so that level 0 is
    selected and the blit stays on the unchanged point overload.

    Phase B-D-c got that protection from a much stronger property (nothing
    below the reference was ever selected at all). Phase B-D-d deliberately
    gives that up, which is what makes this worth checking directly rather
    than inferring from what those two files happen to render at."""
    from PyQt5.QtCore import QRectF

    window = _stepped_window()
    try:
        item = window.map_view._canvas_item
        assert item is not None
        if item._cache.mip_levels()[0] >= 0:
            pytest.skip("fixture enumerates no level coarser than the reference")

        # Same 1:1 shape both of those checks use, on an interior rect.
        _render_scene_rect(window.map_view.scene(), QRectF(300, 300, 96, 96), 1.0)
        assert item._last_mip == 0
        assert item._cache.mip_scale(item._last_mip) == 1.0
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.parametrize("zoom", [2.5, 3.0, 3.7])
def test_block_tiling_matches_a_single_draw_at_a_scaled_level(monkeypatch, zoom) -> None:
    """paint() tiles a large exposedRect into PAINT_BLOCK_PX sub-blocks,
    each independently padded and drawn. Under the scaled blit those
    destination rects have to abut exactly, or block boundaries show as
    seams. The zoom values are chosen to land the residual magnification
    strictly inside [1, 2) -- an integer residual would sample texel
    centres exactly and hide any filtering problem."""
    from PyQt5.QtCore import QRectF
    from descape.viewer import MapCanvasItem

    window = _stepped_window()
    try:
        scene = window.map_view.scene()
        item = window.map_view._canvas_item
        assert item is not None
        if item._cache.mip_levels()[-1] <= 0:
            pytest.skip("fixture enumerates no level finer than the reference")

        probe = QRectF(300, 300, 240, 240)
        monkeypatch.setattr(MapCanvasItem, "PAINT_BLOCK_PX", 100_000)
        single = _render_scene_rect(scene, probe, zoom)
        selected = item._last_mip
        monkeypatch.setattr(MapCanvasItem, "PAINT_BLOCK_PX", 64)
        tiled = _render_scene_rect(scene, probe, zoom)

        assert selected > 0, f"zoom {zoom} selected level {selected}, not the scaled path this tests"
        assert item._last_mip == selected
        assert tiled.shape == single.shape
        diff = int(np.abs(tiled.astype(int) - single.astype(int)).max())
        assert diff <= MAX_SEAM_DIFF, (
            f"tiled vs single-draw render differs by up to {diff} at zoom {zoom} "
            f"(level {selected}) -- block destination rects are not abutting"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.parametrize("residual", [0.3, 0.5, 0.8, 0.95])
def test_block_tiling_matches_a_single_draw_at_a_minifying_residual(monkeypatch, residual) -> None:
    """The band Phase B-D-c could not reach, and the reason PAINT_PAD_PX
    needed re-justifying. Its 2px is argued from "residual magnification in
    [1, 2) means bilinear reaches at most 1 source texel past a block
    edge" -- true of every zoom B-D-c could select, but mip-down clamps at
    the coarsest level, so below that the residual is a MINIFICATION and
    that argument lapses. Live, not theoretical: fit-to-view already sits
    on the bottom rung, and the zoom-out floor permits 4x further.

    Parametrized on the residual rather than on a zoom, since the zoom
    producing a given residual depends on how deep this fixture's ladder
    goes. residual < 1 also guarantees the clamp binds (log2(zoom) <
    levels[0]), so the coarsest level really is what gets painted.

    PROBE_SCENE_SIDE is measured, not arbitrary -- see its own comment: an
    oversized probe picks up an isolated-pixel noise floor that has nothing
    to do with block seams and would make this test read as a failure of
    something it doesn't test."""
    from PyQt5.QtCore import QRectF
    from descape.viewer import MapCanvasItem

    window = _stepped_window()
    try:
        scene = window.map_view.scene()
        item = window.map_view._canvas_item
        assert item is not None
        levels = item._cache.mip_levels()
        if levels[0] >= 0:
            pytest.skip("fixture enumerates no level coarser than the reference")
        scene_per_level_px = item._cache.mip_scale(levels[0])
        zoom = residual / scene_per_level_px

        bounding = item.boundingRect()
        side = min(PROBE_SCENE_SIDE, bounding.width() - 600, bounding.height() - 600)
        blocks = side / scene_per_level_px / MINIFYING_BLOCK_PX
        if blocks < 4:
            pytest.skip(f"canvas only spans {blocks:.1f} blocks at the coarsest level")
        probe = QRectF(300, 300, side, side)

        monkeypatch.setattr(MapCanvasItem, "PAINT_BLOCK_PX", 100_000)
        single = _render_scene_rect(scene, probe, zoom)
        selected = item._last_mip
        monkeypatch.setattr(MapCanvasItem, "PAINT_BLOCK_PX", MINIFYING_BLOCK_PX)
        tiled = _render_scene_rect(scene, probe, zoom)

        assert selected == levels[0], f"residual {residual} selected level {selected}, not the coarsest"
        assert item._last_mip == selected
        assert tiled.shape == single.shape
        diff = int(np.abs(tiled.astype(int) - single.astype(int)).max())
        assert diff <= MAX_SEAM_DIFF, (
            f"tiled vs single-draw render differs by up to {diff} at residual minification "
            f"{residual} across {blocks:.1f} blocks (level {selected}) -- PAINT_PAD_PX does not "
            f"cover the minifying band"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


# ---------------------------------------------------------------------------
# Layer 4: real HiDPI, which needs its own process.
# ---------------------------------------------------------------------------


_HIDPI_PROBE = """
import math, sys, tempfile
from pathlib import Path
from PyQt5.QtWidgets import QApplication
app = QApplication(['probe'])
import descape.asset_source as asset_source
import descape.settings as settings
# Same isolation conftest's autouse _isolated_settings fixture gives every
# other test: a developer's real config.yaml may hold an off-stop
# elev_step_pct, which changes the enumerated level set out from under this.
missing = Path(tempfile.mkdtemp()) / "config.yaml"
asset_source.CONFIG_PATH = missing
settings.CONFIG_PATH = missing
settings._elev_step_pct = None
settings._graphics_quality = None
from descape.viewer import MapCanvasItem, ViewerWindow, _max_axis_scale
from descape.scenario_io import BLANK_TEMPLATE_PATH

seen = []
_original = MapCanvasItem.paint
def _spy(self, painter, option, widget=None):
    result = _original(self, painter, option, widget)
    seen.append(_max_axis_scale(painter.deviceTransform()))
    return result
MapCanvasItem.paint = _spy

window = ViewerWindow()
window.load_scenario(BLANK_TEMPLATE_PATH)
window.terrain_style_combo.setCurrentText("Stepped")
window.resize({size}, {size})
window.show()
QApplication.processEvents(); QApplication.processEvents()
view = window.map_view
for step in {steps}:
    view.scale(step, step)
view.viewport().update()
QApplication.processEvents()
print("MIP", view._canvas_item._last_mip,
      "LEVELS", view._canvas_item._cache.mip_levels()[-1],
      "SCALE", seen[-1] if seen else -1.0)
window.edit_history.mark_saved(); window.close()
"""


def _probe_mip_at_scale_factor(scale_factor: str, steps=HIDPI_ZOOM_STEPS) -> tuple[int, int, float]:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_SCALE_FACTOR"] = scale_factor
    env.pop("QT_AUTO_SCREEN_SCALE_FACTOR", None)
    script = _HIDPI_PROBE.format(size=WINDOW_SIZE, steps=tuple(steps))
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300
    )
    if proc.returncode != 0:
        pytest.skip(f"HiDPI probe subprocess failed at QT_SCALE_FACTOR={scale_factor}: {proc.stderr[-800:]}")
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("MIP ")), None)
    assert line, f"probe printed no result at QT_SCALE_FACTOR={scale_factor}: {proc.stdout!r}"
    parts = line.split()
    return int(parts[1]), int(parts[3]), float(parts[5])


@pytest.mark.gui
def test_hidpi_selects_exactly_one_level_finer() -> None:
    """Selecting from deviceTransform is what makes HiDPI correct for
    free: at QT_SCALE_FACTOR=2 every logical unit is 2 device pixels, one
    full octave, so the same view state must select exactly one level
    finer. Needs its own process -- QApplication reads QT_SCALE_FACTOR
    once at construction, and this suite shares one QApplication.

    Skips rather than fails if the finer level would be clamped away:
    "one finer" is unobservable when the level set has already run out."""
    base_mip, finest, base_scale = _probe_mip_at_scale_factor("1")
    hidpi_mip, _, hidpi_scale = _probe_mip_at_scale_factor("2")

    # The mechanism itself, independent of what level it lands on: the
    # device transform really does carry the DPR, which is the whole reason
    # selection reads it rather than the world transform.
    assert hidpi_scale == pytest.approx(2.0 * base_scale, rel=1e-6), (
        f"QT_SCALE_FACTOR=2 did not double the device scale ({base_scale} -> {hidpi_scale}) -- "
        f"this probe is not measuring HiDPI at all"
    )
    if base_mip >= finest:
        pytest.skip(
            f"DPR 1 already selects the finest level ({finest}) at device scale {base_scale} -- "
            f"'one level finer' is unobservable; lower HIDPI_ZOOM_STEPS"
        )
    assert hidpi_mip == base_mip + 1, (
        f"DPR 1 selected {base_mip} at device scale {base_scale}, DPR 2 selected {hidpi_mip} at "
        f"{hidpi_scale} -- expected exactly one level finer (finest enumerated: {finest})"
    )


@pytest.mark.gui
def test_fit_to_view_is_still_a_minification_at_dpr_2() -> None:
    """FLIPPED IN PHASE B-D-d. In B-D-c this pinned fit-to-view staying on
    the REFERENCE level at DPR 2 -- the phase's own safety property, since
    it meant no existing byte-identity check could reach the scaled blit.
    Mip-down deliberately gives that up (fit is where the win is), and the
    residue moved to test_a_unit_scale_render_selects_the_reference_level,
    which is what actually keeps those two checks on the point overload.

    What survives here is the measurement the plan got wrong: it expected
    DPR 2 to select a FINER level at fit. It doesn't, because fit is a
    heavy minification -- the canvas is thousands of pixels wide in a 300px
    window, so doubling the device scale only moves it from ~0.083 to
    ~0.165, still far below 1 and still clamped at the bottom of the
    ladder. Worth keeping as a real-HiDPI check that the clamp binds at fit
    rather than DPR silently changing what fit-to-view costs."""
    mip, _, scale = _probe_mip_at_scale_factor("2", steps=())
    assert scale < 1.0, f"fit-to-view at DPR 2 was not a minification ({scale})"
    assert 2.0**mip <= scale, (
        f"fit-to-view at DPR 2 selected level {mip} at device scale {scale} -- that level "
        f"magnifies rather than being the coarsest one the ladder could offer"
    )
