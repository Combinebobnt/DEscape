"""Phases B-D-c and B-D-d: LOD selection, the scaled blit, and the zoom-out
floor.

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

2026-08-28: mip_for_scale's rule flipped from floor(log2(scale)) to ceil
(descape/render.py), so residual magnification lands in (0.5, 1.0] instead
of [1, 2) -- every level at or below native density instead of above it.
A handful of tests below flip again for the same reason phase d's did:
each says so in its own docstring.

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
from descape.render_cache import _ChunkCacheBase
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from testkit import qt_capture

ROOT = Path(__file__).resolve().parent.parent

# Flat's own view transform (MapView.set_isometric's scale(1, 0.5) then
# rotate(-45)) at 2x zoom, and what the two candidate LOD metrics return on
# it. The plan chose the max singular value specifically because
# levelOfDetailFromTransform disagrees HERE: under the floor(log2()) rule
# this project used until 2026-08-28, that was 1 vs 0, one whole level,
# showing as an octave of blur along Flat's stretched axis. Kept as
# historical record of that derivation -- under the current ceil(log2())
# rule both metrics land on level 1 here, so this pair no longer
# straddles; see FLAT_2P4X_* below for a pair that still does.
FLAT_2X_MAX_SINGULAR_VALUE = 2.0
FLAT_2X_LEVEL_OF_DETAIL = 1.5811388300841898

# The same transform family at 2.4x zoom instead of 2x -- measured (not
# derived) the same way as the pair above, chosen because it still
# straddles a level boundary under ceil(log2()): max_axis_scale=2.4 ->
# ceil(log2(2.4)) == 2, but levelOfDetailFromTransform=1.897... ->
# ceil(log2(1.897...)) == 1, still a whole level and still an octave of
# blur along Flat's stretched axis if Qt's own metric were used instead.
FLAT_2P4X_MAX_SINGULAR_VALUE = 2.4
FLAT_2P4X_LEVEL_OF_DETAIL = 1.8973665961010273

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
# the same level). 2026-08-28: under mip_for_scale's ceil(log2(scale)) rule
# (was floor), 16x from fit -- device scale ~1.3 -- already selects level 1
# (the finest one on this fixture, tile_px=64 -> ladder {-2,-1,0,1}), which
# left no room for DPR-2 to go finer and made the test skip instead of
# assert. 8x from fit (device scale ~0.65) selects level 0 with room above
# it; DPR 2 doubles that to ~1.3, i.e. level 1. The probe reports its own
# measured device scale so a drift in the fit scale shows up as an
# explicit skip rather than a silent pass.
HIDPI_ZOOM_STEPS = (4.0, 2.0)

WINDOW_SIZE = 300

# Measured on this machine at every residual magnification probed (1.0,
# 1.25, 1.5, 1.85 -- all reachable today only via the finest-level clamp,
# see test_block_tiling_matches_a_single_draw_at_a_scaled_level): the tiled
# and single-draw renders come back BYTE-IDENTICAL, maxdiff 0, not merely
# close. The bar is left as a bounded difference anyway, per the plan --
# padded overlapping draws under a filter are not guaranteed exactly equal,
# and a stricter-than-necessary bar here would be an environment-dependent
# tripwire rather than a real regression signal. A genuine seam would be
# far larger than this.
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
    from descape.viewer_canvas import _max_axis_scale as fn

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
    from descape.viewer_canvas import MapCanvasItem

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
    calling Qt's. If this ever stops disagreeing, re-derive the choice.

    2x zoom no longer straddles a level boundary under mip_for_scale's
    current ceil(log2(scale)) rule (both metrics land on level 1) -- kept
    as historical record of the floor-era derivation. The assertion that
    actually matters now (still disagreeing by a whole level) moved to
    test_max_axis_scale_reproduces_the_plan_s_measured_oracle_at_2p4x
    below."""
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
    # Under floor, this used to be the part that actually cost a level (1
    # vs 0). Under ceil both now land on 1 -- pinned so a future rule
    # change that silently restores the disagreement here doesn't go
    # unnoticed.
    assert math.ceil(math.log2(_max_axis_scale(t))) == 1
    assert math.ceil(math.log2(QStyleOptionGraphicsItem.levelOfDetailFromTransform(t))) == 1


@pytest.mark.gui
def test_max_axis_scale_reproduces_the_plan_s_measured_oracle_at_2p4x() -> None:
    """2026-08-28: the pair above stopped straddling a level boundary when
    mip_for_scale's rule flipped to ceil(log2(scale)). Same transform
    family at 2.4x zoom instead of 2x still straddles under the current
    rule -- this is the live version of the property the test above used
    to pin."""
    from PyQt5.QtGui import QTransform
    from PyQt5.QtWidgets import QStyleOptionGraphicsItem

    conftest.ensure_qapp()
    t = QTransform()
    t.scale(1.0, 0.5)
    t.rotate(-45)
    t.scale(2.4, 2.4)

    assert _max_axis_scale(t) == pytest.approx(FLAT_2P4X_MAX_SINGULAR_VALUE)
    assert QStyleOptionGraphicsItem.levelOfDetailFromTransform(t) == pytest.approx(
        FLAT_2P4X_LEVEL_OF_DETAIL
    )
    # The part that actually costs a level, not just a different float.
    assert math.ceil(math.log2(_max_axis_scale(t))) == 2
    assert math.ceil(math.log2(QStyleOptionGraphicsItem.levelOfDetailFromTransform(t))) == 1


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
    the reference.

    2026-08-28: the 0.2 probe moved from -3 to -2 (ceil(log2(0.2)) == -2,
    was floor's -3) now that mip_for_scale's rule flipped -- it no longer
    reaches the bottom rung on its own, so 0.1 was added to keep that walk-
    to-bottom behavior exercised (ceil(log2(0.1)) == -3, still the ladder's
    coarsest)."""
    item = _canvas_item([-3, -2, -1, 0, 1])
    assert item._select_mip(_StubPainter(_uniform(1.0), _uniform(1.0))) == 0
    assert item._select_mip(_StubPainter(_uniform(0.5), _uniform(0.5))) == -1
    assert item._select_mip(_StubPainter(_uniform(0.2), _uniform(0.2))) == -2
    assert item._select_mip(_StubPainter(_uniform(0.1), _uniform(0.1))) == -3
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
    about what Qt does, and a stubbed transform cannot catch a fourth.

    2026-08-28: under mip_for_scale's floor-era rule this property held
    with a full octave of margin -- any scale in [1.0, 2.0) selected level
    0. Under the current ceil(log2(scale)) rule scale=1.0 sits exactly ON
    the level-0/level-1 boundary (ceil(log2(1.0)) == 0 but
    ceil(log2(1.0000001)) == 1), with no margin either side. Verified safe:
    _max_axis_scale returns exactly 1.0 for a unit transform (no float
    slop in the metric itself), and the real byte-identity callers this
    property protects always render integer-dimensioned rects (see
    testkit.qt_capture.scene_rect_to_array's own docstring)."""
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
    eight callbacks are edit-tool, unit-selection and ruler plumbing nothing
    here reaches."""
    from descape.map_view import MapView

    conftest.ensure_qapp()
    noop = lambda *a, **k: None  # noqa: E731
    return MapView(noop, noop, noop, noop, noop, noop, noop, noop, noop, noop, noop, noop, noop, noop, noop)


@pytest.mark.gui
@pytest.mark.parametrize("levels", [[0], [0, 1, 2], [-1, 0], [-2, -1, 0, 1, 2], [-3, 0]])
def test_zoom_bounds_stay_capped_regardless_of_mip_depth(levels) -> None:
    """Phase B-D-d's clamp divided the floor/ceiling by the mip ladder's
    depth so wheel-zoom could always reach the sharpest/coarsest enumerated
    level, growing the reachable range with the ladder rather than leaving
    the extra levels dead behind a fixed cap.

    2026-08-30: that widening is now itself capped back to the pre-mip
    MIN_ZOOM_FRACTION_OF_FIT/MAX_ZOOM_MULTIPLE_OF_FIT bound (see
    _capture_zoom_baseline's own comment for why -- a fixed 50%-6400%-of-fit
    range was wanted more than the reachability guarantee). Given
    _coarsest_mip_scale() >= 1 and _finest_mip_scale() <= 1 always, the
    widened value is never tighter than the raw bound, so max()/min()-ing
    against it always lands exactly on the raw bound -- this test's real
    claim is that this holds at every ladder depth, not just a shallow one.
    """
    from descape.map_view import MapView

    view = _bare_map_view()
    cache = _StubCache(levels)
    view._canvas_item = _FakeCanvasItem(cache)
    baseline = abs(view.transform().determinant()) ** 0.5

    view._capture_zoom_baseline()
    assert view._min_linear_scale == pytest.approx(MapView.MIN_ZOOM_FRACTION_OF_FIT * baseline)
    assert view._max_linear_scale == pytest.approx(MapView.MAX_ZOOM_MULTIPLE_OF_FIT * baseline)


@pytest.mark.gui
def test_zoom_out_floor_falls_back_to_the_pre_mip_one_with_no_canvas_item() -> None:
    """_capture_zoom_baseline runs from set_isometric(), which ViewerWindow
    can reach before any source is set -- so "no canvas item yet" must be a
    plain no-op on the floor, not an attribute error or an inf."""
    from descape.map_view import MapView

    view = _bare_map_view()
    assert view._canvas_item is None
    view._capture_zoom_baseline()
    baseline = abs(view.transform().determinant()) ** 0.5
    assert view._coarsest_mip_scale() == 1.0
    assert view._min_linear_scale == pytest.approx(MapView.MIN_ZOOM_FRACTION_OF_FIT * baseline)


@pytest.mark.gui
def test_zoom_out_floor_on_a_real_window_stays_capped_despite_a_real_ladder() -> None:
    """The 2026-08-30 cap (_capture_zoom_baseline's own comment) against a
    real fit-to-view baseline and a real enumerated ladder, not just the
    synthetic ladders in test_zoom_bounds_stay_capped_regardless_of_mip_depth
    -- so a change to either (a different reference tile_px, a shallower
    elev_step_pct) can't silently decouple the two.

    depth > 1 here is load-bearing: it proves the cap actually engaged
    (min_linear_scale landed on the raw floor despite a ladder that would
    have divided it lower pre-cap), not that this ladder happens to be
    depth 1 where capped and uncapped give the same answer.

    The baseline reads _fit_baseline_scale() rather than the transform, which
    is what it used before the bounds stopped going stale on resize. This
    fixture resizes AFTER its fit, so the transform is deliberately stale here
    -- that staleness is the bug _fit_baseline_scale() exists to fix, and
    tests/test_zoom_bounds.py is where the two are pinned to each other."""
    from descape.map_view import MapView

    window = _stepped_window()
    try:
        view = window.map_view
        item = view._canvas_item
        assert item is not None
        baseline = view._fit_baseline_scale()
        depth = item._cache.mip_scale(item._cache.mip_levels()[0])
        assert depth > 1
        assert view._min_linear_scale == pytest.approx(MapView.MIN_ZOOM_FRACTION_OF_FIT * baseline)
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
    from descape.viewer_canvas import MapCanvasItem

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
            expected = max(levels[0], min(levels[-1], math.ceil(math.log2(scale))))
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
    than as whatever the last of several paints happened to pick.

    Deliberately NOT testkit.qt_capture.scene_rect_to_array with a zoom
    argument: sizing here truncates via int(), that one rounds up via
    math.ceil(), so on a fractional rect the two disagree by a pixel even
    at zoom 1.0. Only the stride extraction is shared."""
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QImage, QPainter

    tw, th = int(rect.width() * zoom), int(rect.height() * zoom)
    image = QImage(tw, th, QImage.Format_RGB888)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    scene.render(painter, QRectF(0, 0, tw, th), rect)
    painter.end()
    return qt_capture.qimage_rgb888_to_array(image)


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
    than inferring from what those two files happen to render at.

    2026-08-28: this now pins an EXACT-boundary property, not one with an
    octave of margin -- see test_a_unit_scale_transform_selects_the_
    reference_level's own docstring for why that's safe."""
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
    seams. The zoom values are chosen to land on a non-integer residual --
    an integer residual would sample texel centres exactly and hide any
    filtering problem.

    2026-08-28: under mip_for_scale's floor-era rule these zoom values
    landed the residual directly, unclamped, strictly inside [1, 2). Under
    the current ceil(log2(scale)) rule they instead land it via the
    finest-level CLAMP (ceil(log2(2.5/3.0/3.7)) == 2, clamped down to this
    fixture's finest enumerated level, same as before) -- same selected
    level, same residuals (1.25/1.5/1.85), same assertions; only how they're
    reached changed."""
    from PyQt5.QtCore import QRectF
    from descape.viewer_canvas import MapCanvasItem

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
    needed re-justifying. Its 2px is argued from "bilinear reaches at most 1
    source texel past a block edge regardless of ratio" (see PAINT_PAD_PX's
    own comment) -- true of every residual this test or the scaled-level one
    above can reach, magnifying or minifying alike.

    Parametrized on the residual rather than on a zoom, since the zoom
    producing a given residual depends on how deep this fixture's ladder
    goes.

    2026-08-28: under mip_for_scale's current ceil(log2(scale)) rule,
    residual <= 1 no longer uniformly means "the clamp binds" -- 0.3 and
    0.5 still reach the coarsest level via the clamp (ceil(log2(residual))
    == -1 relative to it), but 0.8 and 0.95 now reach it EXACTLY, by the
    rule itself (ceil(log2(residual)) == 0 relative to it), no clamping
    needed. `assert selected == levels[0]` below holds either way; only
    which mechanism produces it differs per residual.

    PROBE_SCENE_SIDE is measured, not arbitrary -- see its own comment: an
    oversized probe picks up an isolated-pixel noise floor that has nothing
    to do with block seams and would make this test read as a failure of
    something it doesn't test."""
    from PyQt5.QtCore import QRectF
    from descape.viewer_canvas import MapCanvasItem

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
from descape.viewer import ViewerWindow
from descape.viewer_canvas import MapCanvasItem, _max_axis_scale
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
      "COARSEST", view._canvas_item._cache.mip_levels()[0],
      "SCALE", seen[-1] if seen else -1.0)
window.edit_history.mark_saved(); window.close()
"""


def _probe_mip_at_scale_factor(scale_factor: str, steps=HIDPI_ZOOM_STEPS) -> tuple[int, int, int, float]:
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
    return int(parts[1]), int(parts[3]), int(parts[5]), float(parts[7])


@pytest.mark.gui
def test_hidpi_selects_exactly_one_level_finer() -> None:
    """Selecting from deviceTransform is what makes HiDPI correct for
    free: at QT_SCALE_FACTOR=2 every logical unit is 2 device pixels, one
    full octave, so the same view state must select exactly one level
    finer. Needs its own process -- QApplication reads QT_SCALE_FACTOR
    once at construction, and this suite shares one QApplication.

    Skips rather than fails if the finer level would be clamped away:
    "one finer" is unobservable when the level set has already run out."""
    base_mip, finest, _, base_scale = _probe_mip_at_scale_factor("1")
    hidpi_mip, _, _, hidpi_scale = _probe_mip_at_scale_factor("2")

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
    ~0.165, still far below 1. Worth keeping as a real-HiDPI check that the
    clamp binds at fit rather than DPR silently changing what fit-to-view
    costs.

    FLIPPED AGAIN when MIP_MIN_TILE_PIXELS rose 8->16 (2026-08-27): that
    shrank the ladder by exactly one level, so fit-to-view at DPR 2 now
    falls in the beyond-the-ladder regime render.py's _select_mip
    docstring already describes -- the clamp binds at the (now shallower)
    coarsest level, and the residual there is a genuine minification of
    that level's own texture rather than the <2x magnification the deeper
    ladder used to guarantee (at the time). That's the accepted cost of the
    floor raise (see MIP_MIN_TILE_PIXELS's own comment): benign for
    quality -- a minified blit downsamples smoothly, it does not blockify
    -- it only means Qt does a real minifying composite at this one
    extreme scale instead of a near-1:1 blit. What this test still guards
    is the mechanism: the clamp must bind at the cache's own actual
    coarsest enumerated level, not silently pick something else.

    2026-08-28: "genuine minification instead of <2x magnification" is no
    longer specific to the floor raise -- mip_for_scale's rule flipped to
    ceil(log2(scale)), so minification at the clamped bottom (and
    everywhere else in the unclamped range too) is the norm now, not an
    accepted cost unique to this floor. The property this test checks
    (clamp binds at the cache's actual coarsest level) is unaffected."""
    mip, _, coarsest, scale = _probe_mip_at_scale_factor("2", steps=())
    assert scale < 1.0, f"fit-to-view at DPR 2 was not a minification ({scale})"
    assert mip == coarsest, (
        f"fit-to-view at DPR 2 selected level {mip} at device scale {scale}, but the cache's "
        f"coarsest enumerated level is {coarsest} -- the clamp did not bind where expected"
    )
