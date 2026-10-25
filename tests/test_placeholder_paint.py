"""Placeholder paints across a level change (TASK-031.25).

While a level the view just crossed onto builds in warm ticks, a live paint
draws a resident neighbour scaled instead, and the viewer refines once the
view's chunks at the new level are cached. Cache-side predicates first
(Qt-free, real caches), then the canvas item's ranking on a duck-typed cache,
then real offscreen windows opted in with stepped_window(placeholders=True).
Every other window in the suite runs with placeholders off (D8).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from test_level_warm import MAP_W, Unit, _scenario, _warm

from descape import render
from descape.render_cache import _ChunkCacheBase
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from descape.unit_filter import UnitFilter

import conftest
from test_unit_sprites import CONST

# Wide enough that a level's view spans 3+ chunks (TASK-031.25 T3b, T3i), and short, so
# two levels' views fit the blank fixture's small chunk byte budget (its level-0 canvas area x 3).
WIDE = 2200
WIDE_HEIGHT = 1100
NARROW = 1600


# --- Step 1: missing_chunks() and level_warmable() -------------------------


@pytest.fixture
def _sprite_install(tmp_path, monkeypatch):
    """test_margin_warm.py's copy of test_level_warm.py's fixture, for the
    F811 reason it records."""
    from descape import asset_source, unit_sprites

    from test_unit_sprites import FILE_NAME, build_sld

    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4, canvas=4 * unit_sprites.NATIVE_TILE_W))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
                         "mirroring_mode": 6, "frame_count": 1}},
    )
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _cache(style: str, scn, *, units: bool = True, sprites: bool = True):
    from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache

    tile_px = render.tile_pixels_for_map(MAP_W, MAP_W)
    if style == "stepped":
        elevations, proj = render.elevations_and_proj(scn)
        return IsoChunkCache(scn, elevations, proj, tile_px, sprites=sprites, with_units=units)
    if style == "sloped":
        elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
        return SlopedChunkCache(scn, elevations, corner_rise, proj, tile_px, sprites=sprites, with_units=units)
    return FlatChunkCache(scn, tile_px, sprites=sprites, with_units=units)


def test_missing_chunks_counts_uncached_keys_under_a_clipped_rect() -> None:
    cache = _cache("flat", _scenario([[]]), units=False, sprites=False)
    size = cache.chunk_px
    canvas_w, canvas_h = cache.canvas_dims(0)
    assert canvas_w > size and canvas_h > size, "a one-chunk canvas -- vacuous"
    full = cache.missing_chunks(0, 0, 0, canvas_w, canvas_h)
    assert full == math.ceil(canvas_w / size) * math.ceil(canvas_h / size)

    cache.get_chunk(0, 0, 0)
    assert cache.missing_chunks(0, 0, 0, 2 * size, size) == 1
    assert cache.missing_chunks(0, 0, 0, size, size) == 0
    assert cache.missing_chunks(0, 0, 0, canvas_w, canvas_h) == full - 1
    # Clipped to the canvas as render_rect() clips: past the edge adds nothing.
    assert cache.missing_chunks(0, -5 * size, -5 * size, canvas_w + 9 * size, canvas_h + 9 * size) == full - 1
    assert cache.missing_chunks(0, 10, 10, 10, 50) == 0
    assert cache.missing_chunks(0, canvas_w, 0, canvas_w + size, size) == 0
    # Membership only: no LRU touch.
    order = list(cache._cache)
    cache.missing_chunks(0, 0, 0, canvas_w, canvas_h)
    assert list(cache._cache) == order


@pytest.mark.parametrize("style", ["stepped", "sloped", "flat"])
@pytest.mark.parametrize("units", [True, False])
@pytest.mark.parametrize("sprites", [True, False])
def test_level_warmable_agrees_with_level_warm_job(style, units, sprites, _sprite_install) -> None:
    """level_warmable(m) == (level_warm_job(m) is not None) on every level,
    across a fresh level, a built one, a gen bump and (Stepped/Sloped) a
    pending flush: the pin test_margin_warm.py keeps for residency."""
    scn = _scenario([[Unit(6.5, 6.5, CONST)]])
    cache = _cache(style, scn, units=units, sprites=sprites)
    levels = cache.mip_levels()
    assert len(levels) > 1, "a one-level ladder -- vacuous"
    states = []

    def check(state: str) -> None:
        answers = {m: cache.level_warmable(m) for m in levels}
        for m in levels:
            assert answers[m] == (cache.level_warm_job(m) is not None), f"{state}: level {m}"
        states.append(answers)

    check("fresh")
    _warm(cache, levels).run_to_completion()
    cache.get_chunk(levels[0], 0, 0)
    check("built")
    if style != "flat":
        from descape.render import dirty_screen_bbox_iso

        cache._level(0)
        scn.map_manager.get_tile(6, 6).elevation += 1
        changed: set = set()
        bbox = dirty_screen_bbox_iso(
            scn, [6 * MAP_W + 6], cache.elevations, cache.proj, with_units=units, with_sprites=sprites,
            elevation_changed=changed,
        )
        cache.patch(bbox, elevation_changed=changed, rebuild_levels=(0,))
        if units:
            assert any(cache._levels[m].pending_elev for m in levels), "nothing deferred -- vacuous"
        check("pending flush")
    cache.set_unit_filter(UnitFilter(show_gaia=False))
    check("gen bump")
    if units and (sprites or style != "flat"):
        assert any(any(s.values()) for s in states) and any(not all(s.values()) for s in states), "one answer only"


# --- Step 2: D4's ranking on a duck-typed ladder ----------------------------


class _LadderCache:
    """The handful of cache methods MapCanvasItem's ranking reads, over a
    four-level ladder [-2, 1] with chosen residency and cached chunks."""

    chunk_px = 512
    missing_chunks = _ChunkCacheBase.missing_chunks
    chunk_index_range = _ChunkCacheBase.chunk_index_range
    reduced_dims = _ChunkCacheBase.reduced_dims

    def __init__(self, resident=(), cached=()):
        self._resident = set(resident)
        self._cache = dict.fromkeys(cached)

    def mip_levels(self):
        return [-2, -1, 0, 1]

    def mip_scale(self, mip: int = 0) -> float:
        return 2.0 ** -mip

    def canvas_dims(self, mip: int = 0):
        side = int(4096 * 2.0 ** mip)
        return side, side

    def is_level_resident(self, mip: int) -> bool:
        return mip in self._resident


def _all_chunks(mip: int):
    side = int(4096 * 2.0 ** mip) // 512
    return [(mip, cx, cy) for cx in range(side) for cy in range(side)]


def _source(cache, mip: int, rect) -> int | None:
    from descape.viewer_canvas import MapCanvasItem

    return MapCanvasItem(cache)._placeholder_source(mip, rect, 0.0)


@pytest.mark.gui
def test_placeholder_source_ranking() -> None:
    from testkit.qt_window import ensure_qapp

    ensure_qapp()
    from PyQt5.QtCore import QRectF

    small = QRectF(100, 100, 800, 800)  # 4 chunks at level 0, 1 at -1 and -2, 16 at 1
    every = [key for mip in (-2, -1, 0, 1) for key in _all_chunks(mip)]

    # Nearest within PLACEHOLDER_MAX_MISSING, coarser first on a tie.
    assert _source(_LadderCache({-2, -1, 1}, every), 0, small) == -1
    assert _source(_LadderCache({-2, 1}, every), 0, small) == 1
    # A finer level only at distance 1; coarser at any distance.
    assert _source(_LadderCache({1}, every), -1, small) is None
    assert _source(_LadderCache({0}, every), -1, small) == 0
    assert _source(_LadderCache({-2}, every), 0, small) == -2
    # Nearer coarser over the cap, farther one within it: the farther one ranks first.
    assert _source(_LadderCache({-2, -1}, _all_chunks(-2)), 0, QRectF(0, 0, 4096, 4096)) == -2

    # Second rank: every candidate over the cap, the coarser one with the fewest wins, never a finer one.
    big = QRectF(0, 0, 4096, 4096)  # 16 chunks at -1, 4 at -2, 64 at 1
    few_at_m2 = _all_chunks(-2)[:1] + _all_chunks(1)[:60]
    assert _source(_LadderCache({-2, -1, 1}, few_at_m2), 0, big) == -2
    # ...while a finer level at distance 1 within the cap still ranks first.
    assert _source(_LadderCache({-2, -1, 1}, _all_chunks(-2)[:1] + _all_chunks(1)), 0, big) == 1
    few_at_m1 = _all_chunks(-1)[:13]
    assert _source(_LadderCache({-2, -1, 1}, few_at_m1), 0, big) == -1
    assert _source(_LadderCache({1}, ()), 0, big) is None
    # None when only far finer levels are resident.
    assert _source(_LadderCache({0, 1}, every), -2, small) is None
    assert _source(_LadderCache((), every), 0, small) is None


@pytest.mark.gui
def test_a_partial_paint_keeps_the_source_the_full_paint_chose() -> None:
    """Review fold: a hover-sized paint while pending draws the source level
    the paint that set the pending level chose, not a nearer one its small
    rect would rank first, while that source stays resident."""
    from testkit.qt_window import ensure_qapp

    ensure_qapp()
    from PyQt5.QtCore import QRectF

    from descape.viewer_canvas import MapCanvasItem

    cache = _LadderCache({-2, -1}, [*_all_chunks(-2), (-1, 0, 0)])
    cache.level_warmable = lambda mip: True
    item = MapCanvasItem(cache, placeholders=True)
    item._settled_mip = -2
    full, hover = QRectF(0, 0, 4096, 4096), QRectF(100, 100, 100, 100)
    assert item._level_to_draw(0, full, True, 0.0) == -2 and item.pending_mip() == 0
    assert item._placeholder_source(0, hover, 0.0) == -1, "the hover rect ranks the same level -- vacuous"
    assert item._level_to_draw(0, hover, True, 0.0) == -2
    cache._resident.discard(-2)
    assert item._level_to_draw(0, hover, True, 0.0) == -1


def test_an_anchored_window_sits_at_the_anchor_s_fraction_of_the_range() -> None:
    from descape.margin_warm import bounded_chunk_range

    assert bounded_chunk_range(0, 0, 9, 9, 4, 4, anchor=(0.0, 1.0)) == (0, 6, 3, 9)
    assert bounded_chunk_range(0, 0, 9, 9, 4, 4, anchor=(0.5, 0.5)) == (3, 3, 6, 6)
    assert bounded_chunk_range(2, 2, 4, 4, 9, 9, anchor=(0.9, 0.1)) == (2, 2, 4, 4)
    assert bounded_chunk_range(0, 0, 9, 9, 4, 4, anchor=(-3.0, 7.0)) == (0, 6, 3, 9)


# --- real windows ----------------------------------------------------------


def _pump() -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()


def _drain(window, rounds: int = 60, until=None) -> None:
    """Every warm to completion (or until `until()` holds), synchronously,
    with the event loop pumped between warmers so the placeholder's 0 ms
    re-anchor runs."""
    warmers = (window._level_warmer, window._load_warmer, window._margin_warmer)
    for _ in range(rounds):
        _pump()
        if until is not None and until():
            return
        if not any(w.is_active for w in warmers) and not window._placeholder_pending_timer.isActive():
            _pump()
            if not any(w.is_active for w in warmers):
                if until is not None:
                    raise AssertionError("the warms drained without meeting the condition")
                return
        for w in warmers:
            if w.is_active:
                w.run_to_completion()
                if until is not None and until():
                    return
    raise AssertionError("the warms never drained")


def _window(*, placeholders: bool = True, size: int = NARROW, height: int | None = None):
    from PyQt5.QtCore import Qt

    window = conftest.stepped_window(FIXTURE_PATH, size=size, placeholders=placeholders)
    if height is not None:
        window.resize(size, height)
        _pump()
    # Scrollbars appearing on the first zoom would resize the viewport between two grabs.
    for set_policy in (window.map_view.setHorizontalScrollBarPolicy, window.map_view.setVerticalScrollBarPolicy):
        set_policy(Qt.ScrollBarAlwaysOn)
    _drain(window)
    return window


def _set_device_scale(window, scale: float) -> None:
    view = window.map_view
    factor = scale / view.view_device_scale()
    view.scale(factor, factor)


def _grab(window) -> np.ndarray:
    from PyQt5.QtGui import QImage

    from testkit.qt_capture import qimage_rgb888_to_array

    image = window.map_view.viewport().grab().toImage().convertToFormat(QImage.Format_RGB888)
    return qimage_rgb888_to_array(image)


def _level_events(monkeypatch) -> list[tuple[str, int]]:
    """(kind, mip) of every level event from here on, Perf Trace on."""
    from descape import perf_trace

    for name, value in perf_trace._fresh_state().items():
        monkeypatch.setattr(perf_trace, name, value)
    monkeypatch.setattr(perf_trace, "_idle_scheduler", None)
    monkeypatch.setattr(perf_trace, "_enabled", True)
    events: list[tuple[str, int]] = []
    real = perf_trace.level_event

    def spy(kind, mip, ms, tiles=None, split=None):
        events.append((kind, mip))
        real(kind, mip, ms, tiles, split)

    monkeypatch.setattr(perf_trace, "level_event", spy)
    return events


# Level 0's residual 0.66: past -1's crossing, onto a level the fit's warm never builds.
CROSSING_SCALE = 0.66


def _crossed(window, scale: float = CROSSING_SCALE) -> int:
    """Zooms onto level 0 from a drained fit; returns it. The grab is the caller's."""
    cache = window._cache
    assert cache.mip_levels() == [-2, -1, 0, 1], "the fixture's ladder moved"
    assert not cache.is_level_resident(0), "level 0 already built -- vacuous"
    _set_device_scale(window, scale)
    target = window.map_view.viewport_chunk_target()
    assert target is not None and target[0] == 0
    return 0


def _hold_warms(window) -> None:
    """Keeps every warm from starting by itself: the 0 ms re-anchor and the poll."""
    window._placeholder_pending_timer.stop()
    window.map_view.stop_viewport_poll()
    window._cancel_warms()


@pytest.mark.gui
def test_placeholders_off_builds_the_crossed_level_in_the_paint() -> None:
    """T2a: the default (every test window, D8)."""
    window = _window(placeholders=False)
    try:
        level = _crossed(window)
        _grab(window)
        item = window.map_view._canvas_item
        assert window._cache.is_level_resident(level)
        assert item._last_drawn_mip == item._last_mip == level
        assert item.pending_mip() is None and item._settled_mip is None
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_a_crossing_onto_an_unbuilt_level_paints_a_resident_one(monkeypatch) -> None:
    """T2b."""
    window = _window()
    try:
        item = window.map_view._canvas_item
        events = _level_events(monkeypatch)
        calls = []
        real = item.on_placeholder
        item.on_placeholder = lambda mip: (calls.append(mip), real(mip))
        level = _crossed(window)
        _grab(window)
        _grab(window)
        assert not window._cache.is_level_resident(level)
        assert ("build", level) not in events
        assert item._last_mip == level
        assert item._last_drawn_mip != level and window._cache.is_level_resident(item._last_drawn_mip)
        assert item.pending_mip() == level
        assert calls == [level]
        assert (f"placeholder<{item._last_drawn_mip}", level) in events
    finally:
        _hold_warms(window)
        conftest.close_window(window)


def _best_shift(a: np.ndarray, b: np.ndarray, reach: int = 3) -> tuple[int, int]:
    """The (dx, dy) in +/-reach minimising mean |a[y, x] - b[y + dy, x + dx]|."""
    h, w = a.shape[:2]
    core = a[reach : h - reach, reach : w - reach].astype(np.int32)
    scores = {}
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            moved = b[reach + dy : h - reach + dy, reach + dx : w - reach + dx].astype(np.int32)
            scores[(dx, dy)] = float(np.abs(core - moved).mean())
    return min(scores, key=scores.get)


@pytest.mark.gui
def test_a_placeholder_frame_registers_with_the_refined_one(monkeypatch) -> None:
    """T2c: same path as a forced draw of F, byte for byte; and against the
    refined frame the best alignment is (0, 0). Control: a 2 px shift is
    seen as (2, 0)."""
    window = _window()
    try:
        view = window.map_view
        item = view._canvas_item
        for other in view.scene().items():
            if other is not item:
                other.setVisible(False)
        level = _crossed(window)
        placeholder = _grab(window)
        source = item._last_drawn_mip
        assert source != level

        with monkeypatch.context() as m:
            m.setattr(item, "_level_to_draw", lambda *_args: source)
            forced = _grab(window)
        assert np.array_equal(placeholder, forced)

        _drain(window)
        assert item.pending_mip() is None and item._settled_mip == level
        refined = _grab(window)
        assert item._last_drawn_mip == level
        assert not np.array_equal(placeholder, refined), "the placeholder is the real frame -- vacuous"
        assert _best_shift(placeholder, refined) == (0, 0)

        shifted = np.empty_like(refined)
        shifted[:, 2:] = refined[:, :-2]
        shifted[:, :2] = refined[:, :1]
        assert _best_shift(placeholder, shifted) == (2, 0), "the search cannot see a 2 px shift"
    finally:
        _hold_warms(window)
        conftest.close_window(window)


@pytest.mark.gui
def test_a_scene_render_capture_never_takes_a_placeholder() -> None:
    """T2d: scene.render() passes widget=None, so a capture with a pending
    placeholder draws the real level and moves no state."""
    from PyQt5.QtCore import QRectF

    from testkit.qt_capture import scene_rect_to_array

    window = _window()
    fresh = _window(placeholders=False)
    try:
        item = window.map_view._canvas_item
        _crossed(window)
        _crossed(fresh)
        _grab(window)
        state = (item._settled_mip, item.pending_mip())
        assert state[1] == 0, "no placeholder pending -- vacuous"
        rect = QRectF(256, 384, 300, 200)
        captured = scene_rect_to_array(window.map_view.scene(), rect)
        assert (item._settled_mip, item.pending_mip()) == state
        assert item._last_mip == item._last_drawn_mip == 0
        assert np.array_equal(captured, scene_rect_to_array(fresh.map_view.scene(), rect))
    finally:
        _hold_warms(window)
        conftest.close_window(window)
        conftest.close_window(fresh)


@pytest.mark.gui
def test_no_placeholder_where_no_warm_would_resolve_it(monkeypatch) -> None:
    """T2e (D5): units off, and top-down Flat with sprites off, have no level
    job, so the crossed level builds in the paint as today."""
    window = _window()
    try:
        monkeypatch.setattr(window._cache, "with_units", False)
        assert not window._cache.level_warmable(0)
        _crossed(window)
        _grab(window)
        item = window.map_view._canvas_item
        assert item._last_drawn_mip == 0 and item.pending_mip() is None
        assert window._cache.is_level_resident(0)
    finally:
        conftest.close_window(window)

    from descape.render_cache import FlatChunkCache

    window = conftest.stepped_window(FIXTURE_PATH, size=NARROW, placeholders=True)
    try:
        window._sprites_enabled = False
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        _pump()
        _drain(window)
        cache = window._cache
        assert isinstance(cache, FlatChunkCache) and not cache.sprites_enabled
        target = window.map_view.viewport_chunk_target()
        finer = target[0] + 2
        assert finer in cache.mip_levels()
        assert not cache.level_warmable(finer)
        _set_device_scale(window, 0.75 * 2.0 ** finer)
        assert window.map_view.viewport_chunk_target()[0] == finer
        _grab(window)
        item = window.map_view._canvas_item
        assert item._last_drawn_mip == finer and item.pending_mip() is None and item._settled_mip == finer
    finally:
        conftest.close_window(window)


def _paints(monkeypatch, item) -> list[tuple[int, int | None]]:
    """(selected, drawn) per live paint of `item`."""
    seen: list[tuple[int, int | None]] = []
    real = item.paint

    def spy(painter, option, widget=None):
        real(painter, option, widget)
        if widget is not None:
            seen.append((item._last_mip, item._last_drawn_mip))

    monkeypatch.setattr(item, "paint", spy)
    return seen


@pytest.mark.gui
def test_partial_exposes_leave_the_level_pending_until_refine(monkeypatch) -> None:
    """T2g (D2): a partial paint of a pending level, ready or forced to
    build, draws it for that rect and leaves it pending; refine() settles."""
    window = _window()
    try:
        view = window.map_view
        item = view._canvas_item
        cache = window._cache
        level = _crossed(window)
        _grab(window)
        assert item.pending_mip() == level
        _hold_warms(window)
        view.on_viewport_changed = lambda: None
        window._level_warmer.start(cache, [level])
        window._level_warmer.run_to_completion()
        assert cache.is_level_resident(level)
        _pump()  # the zoom's own full repaint, before the partial ones
        seen = _paints(monkeypatch, item)

        # A ready rect: its chunks at the level cached by hand.
        size = cache.chunk_px
        cache.get_chunk(level, 1, 1)
        view.invalidate_region((size + 64, size + 64, size + 128, size + 128))
        _pump()
        assert (level, level) in seen, "the ready rect never painted the level -- vacuous"
        assert item.pending_mip() == level

        # A rect that must build: no placeholder source left.
        seen.clear()
        monkeypatch.setattr(item, "_placeholder_source", lambda *_args: None)
        monkeypatch.setattr(item, "_kept_source", lambda *_args: None)
        assert cache.missing_chunks(level, 64, 3 * size + 64, 128, 3 * size + 128) == 1
        view.invalidate_region((64, 3 * size + 64, 128, 3 * size + 128))
        _pump()
        assert (level, level) in seen, "the build rect never painted the level -- vacuous"
        assert cache.missing_chunks(level, 64, 3 * size + 64, 128, 3 * size + 128) == 0
        assert item.pending_mip() == level

        item.refine(level)
        assert item.pending_mip() is None and item._settled_mip == level
    finally:
        conftest.close_window(window)


# --- Step 3: the viewer's refine, warm and watchdog -------------------------


def _force_settled(text: str) -> bool:
    return "placeholder: force-settled" in text


@pytest.mark.gui
def test_refine_settles_and_the_settled_frame_is_todays() -> None:
    """T3a."""
    from descape import debug_log

    window = _window()
    plain = _window(placeholders=False)
    try:
        debug_log.clear()
        level = _crossed(window)
        _crossed(plain)
        _grab(window)
        item = window.map_view._canvas_item
        assert item.pending_mip() == level
        _drain(window)
        assert item.pending_mip() is None and item._settled_mip == level
        assert not _force_settled(debug_log.get_log_text())
        assert np.array_equal(_grab(window), _grab(plain))
        assert item._last_drawn_mip == level
    finally:
        conftest.close_window(window)
        conftest.close_window(plain)


def _view_chunks(window, mip: int):
    view = window.map_view
    return view._canvas_item.view_chunks(mip, view._current_scene_rect(), view.view_device_scale())


@pytest.mark.gui
def test_the_pending_warm_covers_the_whole_view_just_past_a_crossing() -> None:
    """T3b (D3): at residual ~0.51 the view covers about twice a residual-1.0
    span per axis; the pending warm takes the whole range and refine, not
    the watchdog, settles it."""
    from descape import debug_log

    window = _window(size=WIDE, height=WIDE_HEIGHT)
    try:
        view = window.map_view
        cache = window._cache
        debug_log.clear()
        level = _crossed(window, scale=0.51)
        cx0, cy0, cx1, cy1 = _view_chunks(window, level)
        span_w, _span_h = view.viewport_chunk_span(cache, level)
        assert span_w >= 3, "the viewport is under 3 chunks wide -- vacuous"
        assert cx1 - cx0 + 1 > span_w, "the view fits a residual-1.0 span -- vacuous"
        _grab(window)
        item = view._canvas_item
        assert item.pending_mip() == level
        # Up to the refine; what the neighbour warm after it leaves cached is the next test's.
        _drain(window, until=lambda: item.pending_mip() is None)
        assert item.pending_mip() is None
        assert cache.missing_chunks(level, cx0 * 512, cy0 * 512, (cx1 + 1) * 512, (cy1 + 1) * 512) == 0
        assert not _force_settled(debug_log.get_log_text())
    finally:
        conftest.close_window(window)


def _queued_levels(warmer) -> set[int]:
    mips = {mip for mip, _job, _notify in warmer._queue}
    if warmer._job is not None:
        mips.add(warmer._job_mip)
    return mips


@pytest.mark.gui
def test_the_pending_level_warms_first_and_alone() -> None:
    """T3c (D6)."""
    from descape import margin_warm

    window = _window()
    try:
        cache = window._cache
        level = _crossed(window)
        # A load warm already in flight, on a level of the anchor's warm set, over the view.
        cx0, cy0, cx1, cy1 = _view_chunks(window, -1)
        for key in [(-1, cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)]:
            chunk = cache._cache.pop(key, None)
            if chunk is not None:
                cache._cache_bytes -= chunk.nbytes
        busy = margin_warm.load_warm_chunks(cache, -1, cx0, cy0, cx1, cy1)
        window._load_warmer.start(cache, -1, busy, on_drained=window._pump_load_warm)
        _grab(window)
        window._placeholder_pending_timer.stop()
        window.map_view.stop_viewport_poll()
        window._on_placeholder_pending()
        assert _queued_levels(window._level_warmer) == {level}
        assert window._load_warmer.mip == -1

        window._level_warmer.run_to_completion()
        cx0, cy0, cx1, cy1 = _view_chunks(window, level)
        view_keys = {(cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)}
        assert window._load_warmer.mip == level
        assert set(window._load_warmer._queue) <= view_keys and window._load_warmer._queue
        queued = [mip for mip, _chunks in window._load_warm_queue]
        assert queued[0] == level and not set(window._load_warm_queue[0][1]) & view_keys, "the ring is not next"
        assert -1 in queued[1:], "the in-flight level was not queued again behind"

        window._load_warmer.run_to_completion()
        assert window.map_view._canvas_item.pending_mip() is None, "the view entry drained without a refine"
        assert 1 in _queued_levels(window._level_warmer), "refine did not start the neighbour warm"
        assert window._load_warmer.mip == level, "the settled level's ring is not the first load warm after refine"
    finally:
        _hold_warms(window)
        conftest.close_window(window)


@pytest.mark.gui
def test_a_pan_while_pending_warms_the_new_view() -> None:
    """T3d."""
    window = _window()
    try:
        view = window.map_view
        item = view._canvas_item
        level = _crossed(window)
        _grab(window)
        _pump()
        window._level_warmer.run_to_completion()
        assert window._load_warmer.mip == level, "the pending view entry is not in flight -- vacuous"
        hbar = view.horizontalScrollBar()
        hbar.setValue(hbar.value() + 700)
        window._on_viewport_changed()
        assert item.pending_mip() == level
        _drain(window)
        assert item.pending_mip() is None
        assert item.view_ready(level, view._current_scene_rect(), view.view_device_scale())
    finally:
        conftest.close_window(window)


def _view_centre_tile(window) -> tuple[int, int]:
    view = window.map_view
    rect = view.viewport().rect()
    for fy in (0.5, 0.6, 0.7, 0.8, 0.4):
        for fx in (0.5, 0.6, 0.7, 0.8, 0.4):
            tile = view._pick_tile(view.mapToScene(int(rect.width() * fx), int(rect.height() * fy)))
            if tile is not None:
                return tile
    raise AssertionError("no tile under the view")


def _elevate(window, tile) -> None:
    window._on_tool_selected("elevation")
    window.brush_size_spin.setValue(3)
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(*tile, 0)
    window.on_edit_stroke_end()


@pytest.mark.gui
def test_an_elevation_edit_while_pending_shows_no_stale_pixels(monkeypatch) -> None:
    """T3e: a level an edit makes non-resident drops out of D4, so the grab
    after the edit equals a fresh edited window's at the level it drew."""
    window = _window()
    fresh = _window(placeholders=False)
    try:
        item = window.map_view._canvas_item
        _crossed(window)
        _crossed(fresh)
        _grab(window)
        assert item.pending_mip() == 0
        tile = _view_centre_tile(window)
        before = window.scenario.map_manager.get_tile(*tile).elevation
        _elevate(window, tile)
        _elevate(fresh, tile)
        assert window.scenario.map_manager.get_tile(*tile).elevation != before, "the stroke changed nothing"
        edited = _grab(window)
        drawn = item._last_drawn_mip
        monkeypatch.setattr(fresh.map_view._canvas_item, "_level_to_draw", lambda *_args: drawn)
        assert np.array_equal(edited, _grab(fresh))
    finally:
        _hold_warms(window)
        conftest.close_window(window)
        conftest.close_window(fresh)


def _wait(ms: float) -> None:
    deadline = time.perf_counter() + ms / 1000
    while time.perf_counter() < deadline:
        _pump()
        time.sleep(0.005)


@pytest.mark.gui
def test_the_watchdog_settles_a_placeholder_no_warm_will_resolve(monkeypatch) -> None:
    """T3f."""
    from descape import debug_log, viewer

    monkeypatch.setattr(viewer, "PLACEHOLDER_STUCK_MS", 40)
    window = _window()
    try:
        item = window.map_view._canvas_item
        level = _crossed(window)
        _grab(window)
        _pump()
        assert item.pending_mip() == level and window._placeholder_watchdog.isActive()

        # A warm active: it re-arms.
        assert window._level_warmer.is_active
        window._placeholder_watchdog.stop()
        window._on_placeholder_watchdog()
        assert item.pending_mip() == level and window._placeholder_watchdog.isActive()

        # Every warm dropped: it settles and logs.
        debug_log.clear()
        window.map_view.stop_viewport_poll()
        window._cancel_warms()
        _wait(150)
        assert item.pending_mip() is None and item._settled_mip == level
        assert f"placeholder: force-settled level {level}" in debug_log.get_log_text()
        _grab(window)
        assert item._last_drawn_mip == level and window._cache.is_level_resident(level)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_a_sprites_toggle_while_pending_rearms_the_warm(monkeypatch) -> None:
    """T3f's toggle half: _on_sprites_toggled() cancels the warms and
    _rearm_pending_warm() restarts them, so refine settles without the watchdog."""
    from descape import debug_log

    window = _window()
    try:
        item = window.map_view._canvas_item
        level = _crossed(window)
        _grab(window)
        _pump()
        assert item.pending_mip() == level
        debug_log.clear()
        window._on_sprites_toggled(not window._sprites_enabled)
        assert window._level_warmer.is_active or window._load_warmer.is_active, "nothing re-armed"
        _drain(window)
        assert item.pending_mip() is None
        assert not _force_settled(debug_log.get_log_text())
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_preload_off_settles_and_close_leaves_nothing_running(monkeypatch) -> None:
    """T3g."""
    from descape import viewer

    monkeypatch.setattr(viewer, "PLACEHOLDER_STUCK_MS", 40)
    window = _window()
    try:
        item = window.map_view._canvas_item
        _crossed(window)
        _grab(window)
        assert item.pending_mip() == 0
        dialog = viewer.SettingsDialog(window)
        dialog._on_preload_zoom_levels_toggled(False)
        dialog.close()
        assert item.pending_mip() is None and item._settled_mip == 0
        assert not item._placeholders
        _grab(window)
        assert item._last_drawn_mip == 0
    finally:
        conftest.close_window(window)

    monkeypatch.setattr(viewer.settings, "_preload_zoom_levels", True)
    window = _window()
    item = window.map_view._canvas_item
    _crossed(window)
    _grab(window)
    _pump()
    assert item.pending_mip() == 0 and window._placeholder_watchdog.isActive()
    window._placeholder_pending_timer.start()
    conftest.close_window(window)
    # closeEvent() itself stops both, before any interval elapses.
    assert not window._placeholder_watchdog.isActive() and not window._placeholder_pending_timer.isActive()
    _wait(150)
    assert not any(w.is_active for w in (window._level_warmer, window._load_warmer, window._margin_warmer))
    assert not window._placeholder_watchdog.isActive() and not window._placeholder_pending_timer.isActive()


@pytest.mark.gui
def test_a_notch_onto_a_warmed_neighbour_lands_sharp() -> None:
    """T3i (D3b): with the warms run to completion, a zoom-in notch across a
    level (residual ~0.98 to ~0.54) and a zoom-out notch (~0.51 to ~0.99)
    each land sharp: no placeholder, no chunk composited in the paint."""
    window = _window(size=WIDE, height=WIDE_HEIGHT)
    try:
        view = window.map_view
        item = view._canvas_item
        cache = window._cache
        assert view.viewport_chunk_span(cache, 0)[0] >= 3, "the viewport is under 3 chunks wide -- vacuous"

        def settle_at(scale: float) -> None:
            _set_device_scale(window, scale)
            _grab(window)
            window._on_viewport_changed()
            _drain(window)
            assert item.pending_mip() is None

        def notch(scale: float, level: int) -> None:
            _set_device_scale(window, scale)
            assert view.viewport_chunk_target()[0] == level
            keys = set(cache._cache)
            _grab(window)
            assert item._last_mip == item._last_drawn_mip == level, f"a placeholder at level {level}"
            assert item.pending_mip() is None and item._settled_mip == level
            assert set(cache._cache) == keys, f"the notch onto level {level} composited in its paint"

        settle_at(0.49)
        assert view.viewport_chunk_target()[0] == -1
        notch(0.54, 0)
        settle_at(0.51)
        notch(0.495, -1)
    finally:
        conftest.close_window(window)


def _wheel_at(view, pos, up: bool = True) -> None:
    """One real 120-unit wheel notch at viewport point `pos`, a fresh gesture."""
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent

    point = QPointF(pos)
    event = QWheelEvent(point, point, QPoint(0, 0), QPoint(0, 120 if up else -120), Qt.NoButton, Qt.NoModifier,
                        Qt.NoScrollPhase, False)
    view._end_wheel_gesture()
    view.wheelEvent(event)


@pytest.mark.gui
@pytest.mark.parametrize("on_cursor", [False, True], ids=["centre", "edge-cursor"])
def test_a_wheel_notch_across_a_level_lands_sharp_wherever_the_view_sits(on_cursor) -> None:
    """T3i, review fold: a roll's shape in real wheel notches (x1.25). One
    notch crosses from level -2 onto -1 (residual ~0.56) and the neighbour
    warm runs there, at that level's entry; two more notches stay on -1; the
    fourth crosses onto 0 (residual ~0.55). The view sits at four sub-chunk
    offsets, on a viewport whose height mod 512 is over 256, small enough
    that the widened patch fits this fixture's chunk budget. Anchored at the
    centre, and at a cursor by the bottom-right edge, where the view the
    crossing lands on sits in the corner of the one the warm saw. Each lands
    sharp: no placeholder and no chunk composited in the paint."""
    from PyQt5.QtCore import QPoint

    window = _window(size=1360, height=1030)
    try:
        view = window.map_view
        item = view._canvas_item
        cache = window._cache
        vp = view.viewport().rect()
        assert vp.height() % cache.chunk_px > cache.chunk_px // 2, "the viewport height fits its chunks -- vacuous"
        view.set_zoom_anchor_mode(on_cursor)
        anchor = QPoint(vp.width() - 8, vp.height() - 8) if on_cursor else vp.center()
        vbar, hbar = view.verticalScrollBar(), view.horizontalScrollBar()
        for offset in (0, 30, 60, 90):
            _set_device_scale(window, 0.224)
            hbar.setValue((hbar.minimum() + hbar.maximum()) // 2)
            vbar.setValue((vbar.minimum() + vbar.maximum()) // 2 + offset)
            _grab(window)
            _drain(window)
            _wheel_at(view, anchor)  # 0.224 -> 0.28: onto level -1, the roll's cursor recorded
            assert view.viewport_chunk_target()[0] == -1
            _grab(window)
            window._cancel_warms()
            window._on_viewport_changed()
            _drain(window)
            assert item.pending_mip() is None
            for _ in range(2):  # 0.35, 0.4375: still level -1
                _wheel_at(view, anchor)
                _grab(window)
            assert view.viewport_chunk_target()[0] == -1
            keys = set(cache._cache)
            _wheel_at(view, anchor)  # 0.547: level 0
            assert view.viewport_chunk_target()[0] == 0
            _grab(window)
            assert item._last_mip == item._last_drawn_mip == 0, f"a placeholder at offset {offset}"
            assert set(cache._cache) == keys, f"the notch at offset {offset} composited in its paint"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_the_widened_finer_warm_never_evicts_the_settled_view() -> None:
    """Review fold: on a window where the finer neighbour's widened patch
    does not fit the chunk LRU beside the settled view (this fixture's byte
    budget is one level-0 canvas, ~119 chunks), it falls back to the plain
    span, so after refine and every warm drained the settled view is still
    fully cached and its next repaint composites nothing."""
    window = _window(size=WIDE, height=WIDE_HEIGHT)
    try:
        view = window.map_view
        item = view._canvas_item
        cache = window._cache
        level = _crossed(window, scale=0.51)
        target = view.viewport_chunk_target()
        span_w, span_h = view.viewport_chunk_span(cache, level + 1)
        raw = view.viewport_chunk_target_at(level + 1)
        wide = (raw[0], raw[1], raw[0] + 2 * span_w, raw[1] + 2 * span_h)
        assert not window._fits_chunk_budget(target, level + 1, wide), "the wide patch fits -- vacuous"
        _grab(window)
        _drain(window, until=lambda: item.pending_mip() is None)
        _grab(window)  # the refine's repaint
        _drain(window)
        cx0, cy0, cx1, cy1 = _view_chunks(window, level)
        size = cache.chunk_px
        assert cache.missing_chunks(level, cx0 * size, cy0 * size, (cx1 + 1) * size, (cy1 + 1) * size) == 0
        keys = set(cache._cache)
        _grab(window)
        assert set(cache._cache) == keys
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_perf_trace_carries_the_placeholder_and_refine_events(monkeypatch) -> None:
    """T3h."""
    from descape import debug_log, perf_trace

    window = _window()
    try:
        _level_events(monkeypatch)
        debug_log.clear()
        _crossed(window)
        _grab(window)
        _drain(window)
        perf_trace.flush_idle()
        text = debug_log.get_log_text()
        assert "perf view" in text
        view_lines = [line for line in text.splitlines() if "perf view" in line]
        assert any("placeholder<" in line for line in view_lines), view_lines
        assert any(" refine " in line for line in view_lines), view_lines
    finally:
        conftest.close_window(window)


# --- Step 4: only the app opts in -------------------------------------------


@pytest.mark.gui
def test_only_the_apps_window_allows_placeholders() -> None:
    from testkit.qt_window import ensure_qapp

    ensure_qapp()
    from descape import viewer

    assert viewer.LIVE_PLACEHOLDERS is False
    assert "_app_window" in viewer.main.__code__.co_names
    plain = viewer.ViewerWindow()
    app = viewer._app_window()
    try:
        assert not plain._placeholders_allowed
        assert app._placeholders_allowed and app._placeholders_on()
    finally:
        conftest.close_window(plain)
        conftest.close_window(app)
