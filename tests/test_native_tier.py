"""TASK-599: the 96 px native-sprite tier in the iso mip ladders, and
TASK-601's 192 tier at Default. The ladder table, the rank-based level
selection, and the wrap-padded terrain crops a tile_px that does not divide
the texture needs. Sprite identity at 96 is in
tests/test_x2_sprites.py; backend agreement at 96 in test_native_composite.py."""

from __future__ import annotations

import numpy as np
import pytest

from descape import asset_source, iso_geometry, render, settings
from descape.render_cache import _ChunkCacheBase

NATIVE = iso_geometry.NATIVE_TILE_PIXELS


def test_the_ladder_table() -> None:
    """Every (base, quality finest, tier, tier limit) a real cache can be built
    with; a tier only where the finest and the quality's tier limit reach it and
    it is the base times 3/2 times a power of two (TASK-599, TASK-601)."""
    table = {
        # Default (finest 192, tiers to 192): small and large maps.
        (64, 192, True, 192): {-2: 16, -1: 32, 0: 64, 1: 96, 2: 128, 3: 192},
        (32, 192, True, 192): {-1: 16, 0: 32, 1: 64, 2: 96, 3: 128, 4: 192},
        # Enhanced (finest 128, tiers to 96), until TASK-441.
        (64, 128, True, 96): {-2: 16, -1: 32, 0: 64, 1: 96, 2: 128},
        (32, 128, True, 96): {-1: 16, 0: 32, 1: 64, 2: 96, 3: 128},
        # Top-down Flat: no tiers, so the 192 ceiling stops the power-of-two loop at 128.
        (64, 192, False, 192): {-2: 16, -1: 32, 0: 64, 1: 128},
        (32, 192, False, 192): {-1: 16, 0: 32, 1: 64, 2: 128},
        # Potato (finest 64) and Potatest (32) never reach 96.
        (32, 64, True, 96): {-1: 16, 0: 32, 1: 64},
        (16, 64, True, 96): {0: 16, 1: 32, 2: 64},
        (16, 32, True, 96): {0: 16, 1: 32},
        # No real cache: base 16 only comes with a finest of 64 or less. 96 is 16 x 3/2 x 4, so it would be a tier.
        (16, 128, True, 96): {0: 16, 1: 32, 2: 64, 3: 96, 4: 128},
        # The tier limit alone keeps 192 out under a ceiling that admits it (Decision 2's back door).
        (64, 192, True, 96): {-2: 16, -1: 32, 0: 64, 1: 96, 2: 128},
    }
    bad = {}
    for (base, finest, tier, tier_max), want in table.items():
        got = iso_geometry.mip_tile_px_candidates(base, finest, native_tier=tier, native_tier_max=tier_max)
        if got != want:
            bad[(base, finest, tier, tier_max)] = got
    assert not bad, bad


def test_a_raised_ceiling_still_keeps_192_off_enhanced(monkeypatch) -> None:
    """Decision 2: TASK-202's 256 ceiling must not bring 192 (and x2 at 1:1, TASK-441) to Enhanced."""
    monkeypatch.setattr(iso_geometry, "MIP_MAX_TILE_PIXELS", 256)
    monkeypatch.setitem(render.FINEST_TILE_PIXELS, settings.GRAPHICS_QUALITY_MAX, 256)
    monkeypatch.setattr(settings, "get_graphics_quality", lambda: settings.GRAPHICS_QUALITY_MAX)
    got = iso_geometry.mip_tile_px_candidates(64, render.finest_tile_pixels(), native_tier=True,
                                              native_tier_max=render.native_tier_max())
    assert 256 in got.values(), "the patched ceiling did not reach the ladder -- vacuous"
    assert 192 not in got.values() and NATIVE in got.values()


@pytest.mark.parametrize("quality", sorted(render.FINEST_TILE_PIXELS))
@pytest.mark.parametrize("large", [False, True], ids=["small", "large"])
def test_the_tier_exists_exactly_at_default_and_enhanced(monkeypatch, quality, large) -> None:
    monkeypatch.setattr(settings, "get_graphics_quality", lambda: quality)
    side = render.LARGE_MAP_TILE_THRESHOLD + 1 if large else render.LARGE_MAP_TILE_THRESHOLD
    base = render.tile_pixels_for_map(side, side)
    proj = iso_geometry.canvas_size_and_origin(side, side, base, 0, 7, elev_step_pct=100)
    projs = iso_geometry.mip_projections_for(
        side, side, proj, 100, max_tile_px=render.finest_tile_pixels(), native_tier_max=render.native_tier_max()
    )
    iso = {p.tile_px for p in projs.values()}
    flat = set(iso_geometry.mip_tile_px_candidates(base, render.finest_tile_pixels()).values())
    assert (NATIVE in iso) == (quality >= settings.GRAPHICS_QUALITY_DEFAULT)
    assert (2 * NATIVE in iso) == (quality == settings.GRAPHICS_QUALITY_DEFAULT), "192 at Default only (TASK-601)"
    assert NATIVE not in flat and 2 * NATIVE not in flat, "top-down Flat keeps the power-of-two ladder (Decision 5)"


class _Ladder:
    mip_levels = _ChunkCacheBase.mip_levels
    mip_scale = _ChunkCacheBase.mip_scale
    mip_for_scale = _ChunkCacheBase.mip_for_scale

    def __init__(self, tile_px_by_level):
        self._mip_tile_px = dict(sorted(tile_px_by_level.items()))


def test_mip_for_scale_decision_table() -> None:
    pow2 = _Ladder(iso_geometry.mip_tile_px_candidates(64, 128))
    tier64 = _Ladder(iso_geometry.mip_tile_px_candidates(64, 128, native_tier=True))
    tier32 = _Ladder(iso_geometry.mip_tile_px_candidates(32, 128, native_tier=True))
    default64 = _Ladder(iso_geometry.mip_tile_px_candidates(64, 192, native_tier=True, native_tier_max=192))
    rows = [
        # TASK-601: (1.5, 2.0] picks 128, (2.0, 3.0] picks 192, past that 192 still.
        (default64, 1.5 + 1e-9, 128), (default64, 2.0, 128), (default64, 2.0 + 1e-9, 192), (default64, 3.0, 192),
        (default64, 4.0, 192), (default64, 192 / 64 * (1 + 4e-16), 192),
        # (ladder, scale, expected tile_px)
        (pow2, 0.2, 16), (pow2, 0.25, 16), (pow2, 0.3, 32), (pow2, 1.0, 64), (pow2, 1.2, 128), (pow2, 9.0, 128),
        (tier64, 1.0, 64), (tier64, 1.0 + 1e-9, 96), (tier64, 1.5, 96), (tier64, 1.5 + 1e-9, 128),
        (tier64, 2.0, 128), (tier64, 0.5, 32), (tier64, 0.0, 16), (tier64, -1.0, 16),
        (tier32, 2.0, 64), (tier32, 2.5, 96), (tier32, 3.0, 96), (tier32, 3.1, 128),
        # A view landed on 96 by a float multiply an ulp high stays on it.
        (tier64, 96 / 64 * (1 + 4e-16), 96), (tier32, 96 / 32 * (1 + 4e-16), 96),
    ]
    bad = [(id(lad), s, want, lad._mip_tile_px[lad.mip_for_scale(s)]) for lad, s, want in rows
           if lad._mip_tile_px[lad.mip_for_scale(s)] != want]
    assert not bad, bad
    # On a power-of-two ladder the rule is still clamp(ceil(log2(scale))).
    import math

    for s in np.geomspace(0.05, 20, 400):
        want = max(pow2.mip_levels()[0], min(pow2.mip_levels()[-1], math.ceil(math.log2(s))))
        assert pow2.mip_for_scale(float(s)) == want, s


def _texture(size: int = asset_source.LOADED_TEXTURE_SIZE, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(599)
    shape = (size, size, channels) if channels else (size, size)
    return rng.integers(0, 256, shape, dtype=np.uint8)


@pytest.mark.parametrize("tile_px", [NATIVE, 2 * NATIVE])
@pytest.mark.parametrize("channels", [3, 0], ids=["texture", "mask"])
def test_a_padded_crop_equals_the_modulo_crop(channels, tile_px) -> None:
    tex = _texture(channels=channels)
    size = tex.shape[0]
    straddled = 0
    for x in range(size // 32):
        for y in (0, 5, 16):
            ox, oy = render._crop_offset(x, y, size, tile_px)
            got = asset_source.crop_source(tex, tile_px)[oy : oy + tile_px, ox : ox + tile_px]
            rows, cols = (oy + np.arange(tile_px)) % size, (ox + np.arange(tile_px)) % size
            assert np.array_equal(got, tex[np.ix_(rows, cols)]), (x, y)
            straddled += ox + tile_px > size
    assert straddled, "no crop straddled the wrap -- vacuous"


def test_crop_source_pads_only_where_tile_px_does_not_divide() -> None:
    tex = _texture()
    for tile_px in (16, 32, 64, 128):
        assert asset_source.crop_source(tex, tile_px) is tex
    assert asset_source.crop_source(tex, 2 * NATIVE).shape[:2] == (tex.shape[0] + 2 * NATIVE,) * 2
    padded = asset_source.crop_source(tex, NATIVE)
    assert padded.shape[:2] == (tex.shape[0] + NATIVE,) * 2
    assert not padded.flags.writeable
    assert asset_source.crop_source(tex, NATIVE) is padded, "padded copy is not cached"
    other = tex.copy()
    other[0, 0] ^= 1
    assert asset_source.crop_source(other, NATIVE)[0, 0, 0] == other[0, 0, 0], "a cache hit across arrays"


def test_padded_copies_live_only_while_a_native_tier_cache_does(monkeypatch) -> None:
    import gc

    gc.collect()  # a finalizer from an earlier test's cache must not fire mid-test
    monkeypatch.setattr(asset_source, "_wrap_holders", 0)
    monkeypatch.setattr(asset_source, "_wrap_padded", {})

    class _Cache:
        _init_mip_levels = _ChunkCacheBase._init_mip_levels
        tile_px = 64

    tex = _texture()
    pow2, tiered = _Cache(), _Cache()
    pow2._init_mip_levels(iso_geometry.mip_tile_px_candidates(64, 128))
    assert asset_source._wrap_holders == 0, "a power-of-two ladder holds the copies"
    tiered._init_mip_levels(iso_geometry.mip_tile_px_candidates(64, 128, native_tier=True))
    second = _Cache()
    second._init_mip_levels(iso_geometry.mip_tile_px_candidates(64, 128, native_tier=True))
    padded = asset_source.crop_source(tex, NATIVE)
    del tiered
    gc.collect()
    assert asset_source.crop_source(tex, NATIVE) is padded, "dropped while a holder lives"
    del second
    gc.collect()
    assert not asset_source._wrap_padded
    assert asset_source.crop_source(tex, NATIVE) is not padded
    asset_source.clear_install_caches()
    assert not asset_source._wrap_padded


# --- the zoom stop (Step 5) -------------------------------------------------

import conftest  # noqa: E402

WINDOW_SIZE = 700


def _wheel(view, up: bool) -> None:
    """One real 120-unit notch at the viewport centre, a fresh gesture."""
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent

    point = QPointF(view.viewport().rect().center())
    event = QWheelEvent(point, point, QPoint(0, 0), QPoint(0, 120 if up else -120), Qt.NoButton, Qt.NoModifier,
                        Qt.NoScrollPhase, False)
    view._end_wheel_gesture()
    view.wheelEvent(event)


def _at_native(view) -> bool:
    return view.at_native_stop()


def _assert_on_native_grid(view, dpr: float) -> None:
    """The native level drawn 1:1 with its origin on a whole device px of
    the window: the viewport's own offset there included, since the backing
    store is the window's."""
    from descape import zoom_preframe

    item, cache = view._canvas_item, view._canvas_item._cache
    mip = view._mip_for_zoom_factor(1.0)
    assert cache.mip_tile_px(mip) == NATIVE
    assert cache.mip_scale(mip) * view.devicePixelRatioF() * _max_scale(view) == pytest.approx(1.0, abs=1e-12)
    dt = item.deviceTransform(view.viewportTransform())
    offset = zoom_preframe.viewport_offset(view.viewport())
    for phys in ((offset.x() + dt.dx()) * dpr, (offset.y() + dt.dy()) * dpr):
        assert phys == pytest.approx(round(phys), abs=1e-6), phys


def _offset_viewport_off_grid(view) -> None:
    """Viewport margins that put the viewport at an offset of 1 mod 4 in its
    window: a fractional device px at DPR 1.25 and 1.5."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    from descape import zoom_preframe

    view.setViewportMargins(0, 0, 0, 0)
    QApplication.processEvents()
    offset = zoom_preframe.viewport_offset(view.viewport())
    view.setViewportMargins((1 - offset.x()) % 4, (1 - offset.y()) % 4, 0, 0)
    QApplication.processEvents()
    offset = view.viewport().mapTo(view.window(), QPoint())
    assert (offset.x() % 4, offset.y() % 4) == (1, 1), offset


def _max_scale(view) -> float:
    from descape.viewer_canvas import _max_axis_scale

    return _max_axis_scale(view._canvas_item.deviceTransform(view.viewportTransform()))


@pytest.mark.test_map
@pytest.mark.gui
@pytest.mark.parametrize("off_grid", [False, True], ids=["layout-offset", "off-grid-offset"])
@pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0])
def test_zoom_to_native_lands_on_the_native_level_on_whole_device_px(monkeypatch, dpr, off_grid) -> None:
    window = conftest.stepped_window(conftest.TEST_MAP_PATH, size=WINDOW_SIZE)
    try:
        view = window.map_view
        monkeypatch.setattr(view, "devicePixelRatioF", lambda: dpr)
        if off_grid:
            _offset_viewport_off_grid(view)
        assert not _at_native(view), "fit already at the stop -- vacuous"
        assert window.zoom_native_action.isEnabled()
        window.zoom_native_action.trigger()
        assert _at_native(view)
        _assert_on_native_grid(view, dpr)
        # A pan at the stop keeps the level and, at a whole DPR, the grid.
        view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() + 37)
        assert _at_native(view)
        if dpr == int(dpr):
            _assert_on_native_grid(view, dpr)
    finally:
        conftest.close_window(window)


@pytest.mark.test_map
@pytest.mark.gui
@pytest.mark.parametrize("fit_multiple", [None, 30.0], ids=["real-ceiling", "ceiling-below-stop"])
def test_the_wheel_lands_on_the_native_stop_from_both_directions(monkeypatch, fit_multiple) -> None:
    """A big map in a small window puts the fit-relative ceiling below the stop
    (30x fit does on the test map); the ceiling is raised to the finest
    level's stop there (TASK-600), so the native stop stays reachable."""
    from test_wheel_zoom import _notch_scale, _scale

    window = conftest.stepped_window(conftest.TEST_MAP_PATH, size=WINDOW_SIZE)
    try:
        view = window.map_view
        if fit_multiple is not None:
            monkeypatch.setattr(view, "MAX_ZOOM_MULTIPLE_OF_FIT", fit_multiple)
            view._capture_zoom_baseline()
            at_stop = _scale(view) * view.native_zoom_factor()
            finest = _scale(view) * view.level_zoom_factor(view._canvas_item._cache.mip_levels()[-1])
            assert fit_multiple * view._fit_baseline_scale() < at_stop, "ceiling not below the stop -- vacuous"
            assert view._max_linear_scale == pytest.approx(finest, rel=1e-12)
        for up in (True, False):
            # Start on the far side of the stop: fit for zooming in, two notches past it for zooming out.
            if not up:
                view.zoom_to_native()
                _wheel(view, True)
                _wheel(view, True)
                assert view.native_zoom_factor() < 1.0
            landed = False
            for _ in range(40):
                before = view.native_zoom_factor()
                _wheel(view, up)
                after = view.native_zoom_factor()
                if _at_native(view):
                    landed = True
                    break
                assert (before > 1.0) == (after > 1.0), f"{'in' if up else 'out'}: skipped the stop {before} -> {after}"
            assert landed, f"{'in' if up else 'out'}: never reached the stop"
            _assert_on_native_grid(view, 1.0)
            # The next notch carries on to the stop's lattice neighbour (TASK-600).
            want = _notch_scale(view, 1 if up else -1)
            _wheel(view, up)
            assert _scale(view) == pytest.approx(want, rel=1e-9)
    finally:
        conftest.close_window(window)


@pytest.mark.test_map
@pytest.mark.gui
@pytest.mark.parametrize("dpr", [1.0, 1.25])
def test_the_twin_predicts_the_snapped_notch_onto_the_stop(monkeypatch, dpr) -> None:
    """The pre-rendered frame's key is the exact transform, so the twin's
    prediction of a notch onto the stop must include the wheel's snap."""
    from descape import zoom_preframe

    window = conftest.stepped_window(conftest.TEST_MAP_PATH, size=WINDOW_SIZE)
    try:
        view = window.map_view
        view._zoom_on_cursor = False
        monkeypatch.setattr(view, "devicePixelRatioF", lambda: dpr)
        _offset_viewport_off_grid(view)
        twin = zoom_preframe.ZoomTwin(view)
        anchor = view.viewport().rect().center()
        snapped = []
        # A pan one notch in leaves the notch back off the grid for some pan sizes.
        for pan in range(4):
            view.zoom_to_native()
            _wheel(view, True)
            assert not _at_native(view)
            view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() + pan)
            view._wheel_gesture_mip = view._mip_for_zoom_factor(1.0)
            factor = view._wheel_zoom_factor(-1)
            predicted, _rect = twin.predict(factor, anchor, view._twin_snap(factor))
            unsnapped, _rect = twin.predict(factor, anchor)
            assert view._twin_snap(0.8 * factor) is None, "a notch off the stop snaps"
            _wheel(view, False)
            assert _at_native(view)
            _assert_on_native_grid(view, dpr)
            assert predicted == view.viewportTransform(), pan
            snapped.append(unsnapped != predicted)
        assert any(snapped) == (dpr != 1.0), snapped
    finally:
        conftest.close_window(window)


def test_the_tier_is_the_sprite_arts_native_tile() -> None:
    from descape import unit_sprites

    assert NATIVE == unit_sprites.NATIVE_TILE_W


@pytest.mark.test_map
@pytest.mark.gui
def test_the_command_greys_where_the_ladder_has_no_native_level() -> None:
    window = conftest.stepped_window(conftest.TEST_MAP_PATH, size=WINDOW_SIZE, graphics_quality=2)
    try:
        assert not window.zoom_native_action.isEnabled()
        assert "Potato" in window.zoom_native_action.toolTip()
        assert not window.map_view.zoom_to_native()
    finally:
        conftest.close_window(window)

    # Pinned: the window above left the settings global at Potato.
    window = conftest.stepped_window(conftest.TEST_MAP_PATH, size=WINDOW_SIZE, graphics_quality=3)
    try:
        assert window.zoom_native_action.isEnabled()
        window.terrain_style_combo.setCurrentText("Flat")
        assert window.zoom_native_action.isEnabled(), "Flat + Isometric View renders through the iso ladder"
        window.iso_action.setChecked(False)
        assert window.map_view.native_zoom_factor() is None
        assert not window.zoom_native_action.isEnabled()
        assert "top-down Flat" in window.zoom_native_action.toolTip()
    finally:
        conftest.close_window(window)
