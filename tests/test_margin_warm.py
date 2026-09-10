"""Idle-time viewport-margin chunk warm -- the ring builder, the driver, and
the viewer hooks that feed them (maintainer plan 2026-09-07's Steps A2/A3).

Structured the way tests/test_level_warm.py is: Qt-free checks against duck-
typed fakes first (ring geometry/ordering, the driver's tick/cancel/retarget
contract, render_cache's two new predicates), then gui-marked checks that
need a real QGraphicsView paint cycle to prove the paint-free
viewport_chunk_target() reading actually agrees with what a real paint
selects.

_FakeCache below is deliberately NOT a real IsoChunkCache/FlatChunkCache: it
implements only the handful of methods margin_warm.py and
viewer_canvas.level_rect_for() actually call, so the ring/driver tests stay
pure-Python and fast. The predicate-parity and gui checks further down use
the real caches (via test_level_warm.py's own fixtures) precisely where a
fake would hide a real divergence.
"""

from __future__ import annotations

import math

import pytest

import conftest
from descape import level_warm, margin_warm
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from test_level_warm import Unit, _flat_cache, _iso_cache, _scenario, _warm
from test_unit_sprites import CONST


class _FakeCache:
    """Duck-typed cache exposing only what margin_warm.py and
    viewer_canvas.level_rect_for() call -- no real compositing, so ring/
    driver geometry is testable with no scenario, no PyQt5, no QApplication.
    """

    def __init__(self, canvas_dims, chunk_px: int = 512, mip_scales=None, resident_mips=()):
        self._canvas_dims = canvas_dims  # {mip: (w, h)}
        self.chunk_px = chunk_px
        self._mip_scales = mip_scales or {}
        self._resident_mips = set(resident_mips)
        self._cache: set[tuple[int, int, int]] = set()
        self.get_chunk_calls: list[tuple[int, int, int]] = []
        self.fail_on: set[tuple[int, int, int]] = set()

    def canvas_dims(self, mip: int = 0):
        return self._canvas_dims[mip]

    def mip_scale(self, mip: int = 0):
        return self._mip_scales[mip]

    def has_chunk(self, mip, cx, cy) -> bool:
        return (mip, cx, cy) in self._cache

    def is_level_resident(self, mip: int) -> bool:
        return mip in self._resident_mips

    def get_chunk(self, mip, cx, cy):
        key = (mip, cx, cy)
        if key in self.fail_on:
            raise RuntimeError("boom")
        self.get_chunk_calls.append(key)
        self._cache.add(key)
        return key


class _Rect:
    """A plain duck-typed stand-in for QRectF -- level_rect_for() only ever
    calls .left()/.top()/.right()/.bottom(), so a real QRectF (and the
    PyQt5 import that would come with it) isn't needed for this file's
    Qt-free checks."""

    def __init__(self, x0, y0, x1, y1):
        self._x0, self._y0, self._x1, self._y1 = x0, y0, x1, y1

    def left(self):
        return self._x0

    def top(self):
        return self._y0

    def right(self):
        return self._x1

    def bottom(self):
        return self._y1


# --- ring_chunks: geometry and ordering -------------------------------------


def test_ring_depth1_size_and_membership() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    ring = margin_warm.ring_chunks(cache, 0, 2, 5, 4, 6)

    assert len(ring) == 14, "depth-1 ring around a 3x2 viewport should be 5x4 - 3x2 = 14 chunks"
    assert (1, 5) in ring  # left edge
    assert (5, 5) in ring  # right edge
    assert (2, 4) in ring  # top edge
    assert (2, 7) in ring  # bottom edge
    assert (1, 4) in ring  # corner
    assert (3, 5) not in ring, "a chunk inside the viewport itself must never be a margin chunk"


def test_ring_clips_at_grid_corner() -> None:
    cache = _FakeCache({0: (2 * 512, 2 * 512)}, resident_mips={0})  # 2x2 grid, indices 0/1
    ring = margin_warm.ring_chunks(cache, 0, 0, 0, 0, 0)
    assert set(ring) == {(1, 0), (0, 1), (1, 1)}


def test_ring_drops_already_resident_chunks() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    cache._cache.add((0, 1, 5))
    ring = margin_warm.ring_chunks(cache, 0, 2, 5, 4, 6)
    assert (1, 5) not in ring
    assert len(ring) == 13


def test_ring_empty_when_viewport_covers_the_whole_grid() -> None:
    """The fit-to-view case: nothing left to expand into, so the opening
    level's margin is empty by geometry, with no explicit "skip the fit
    mip" branch needed anywhere."""
    cache = _FakeCache({0: (5 * 512, 3 * 512)}, resident_mips={0})  # grid 5x3
    ring = margin_warm.ring_chunks(cache, 0, 0, 0, 4, 2)
    assert ring == []


def test_ring_orders_leading_edge_first_then_trailing_then_corners_last() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    ring = margin_warm.ring_chunks(cache, 0, 2, 5, 4, 6, lead=(1, 0))

    leading = ring.index((5, 5))  # right edge -- the side lead=(1, 0) heads toward
    trailing = ring.index((1, 5))  # left edge -- the opposite side
    corner = ring.index((5, 4))  # top-right corner
    assert leading < trailing < corner


def test_ring_with_no_lead_has_no_favoured_side() -> None:
    """lead=(0, 0) (a zoom, or the very first fire) degenerates to plain
    edge-then-corner order -- correct behaviour for that case, not a
    missing feature."""
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    ring = margin_warm.ring_chunks(cache, 0, 2, 5, 4, 6)
    corner_index = ring.index((1, 4))
    edge_indices = [ring.index(c) for c in [(1, 5), (5, 5), (2, 4), (2, 7)]]
    assert all(i < corner_index for i in edge_indices), "every edge chunk must sort before every corner"


# --- MarginWarmer: the driver contract --------------------------------------


def test_margin_warmer_one_get_chunk_per_tick() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(1, 1), (2, 2), (3, 3)])
    assert warmer.is_active

    assert warmer.tick() is True
    assert cache.get_chunk_calls == [(0, 1, 1)]
    assert warmer.tick() is True
    assert cache.get_chunk_calls == [(0, 1, 1), (0, 2, 2)]
    assert warmer.tick() is False, "the last chunk must drain the queue and report False"
    assert cache.get_chunk_calls == [(0, 1, 1), (0, 2, 2), (0, 3, 3)]
    assert not warmer.is_active


def test_margin_warmer_cancel_drops_the_rest() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(1, 1), (2, 2)])
    warmer.tick()

    warmer.cancel()
    warmer.run_to_completion()

    assert not warmer.is_active
    assert cache.get_chunk_calls == [(0, 1, 1)]


def test_margin_warmer_start_replaces_an_in_flight_ring() -> None:
    """The retarget check -- A4's whole mechanism is start() replacing
    whatever was queued outright, with no separate retarget API."""
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(1, 1), (2, 2)])
    warmer.tick()

    warmer.start(cache, 0, [(9, 9)])
    warmer.run_to_completion()

    assert cache.get_chunk_calls == [(0, 1, 1), (0, 9, 9)]


def test_margin_warmer_refuses_to_start_when_the_level_is_not_resident() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips=set())
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(1, 1)])
    assert not warmer.is_active
    assert cache.get_chunk_calls == []


def test_margin_warmer_empty_chunk_list_is_a_no_op() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [])
    assert not warmer.is_active


def test_margin_warmer_logs_and_continues_past_a_bad_chunk(monkeypatch) -> None:
    from descape import debug_log

    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    cache.fail_on.add((0, 2, 2))
    logged = []
    monkeypatch.setattr(debug_log, "log", lambda msg: logged.append(msg))

    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(1, 1), (2, 2), (3, 3)])
    warmer.run_to_completion()

    assert logged, "the failing chunk was never logged -- vacuous"
    assert cache.get_chunk_calls == [(0, 1, 1), (0, 3, 3)]


def test_margin_warmer_on_drained_fires_once_the_queue_empties() -> None:
    """2026-09-07 plan's load-time margin warm, Step 3: on_drained is the
    hook _pump_load_warm() chains the next neighbour mip's chunks off of."""
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    drained = []
    warmer.start(cache, 0, [(1, 1), (2, 2)], on_drained=lambda: drained.append(True))
    warmer.tick()
    assert drained == [], "must not fire before the queue actually empties"
    warmer.tick()
    assert drained == [True]


def test_margin_warmer_on_drained_does_not_fire_on_a_refused_or_empty_start() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips=set())
    warmer = margin_warm.MarginWarmer()
    drained = []
    warmer.start(cache, 0, [(1, 1)], on_drained=lambda: drained.append(True))
    assert drained == [], "a refused start has nothing to wait for"

    resident_cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer.start(resident_cache, 0, [], on_drained=lambda: drained.append(True))
    assert drained == [], "an empty chunk list has nothing to wait for either"


def test_margin_warmer_on_drained_does_not_fire_on_cancel() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    drained = []
    warmer.start(cache, 0, [(1, 1), (2, 2)], on_drained=lambda: drained.append(True))
    warmer.tick()
    warmer.cancel()
    assert drained == [], "a cancelled queue never drained -- the callback must not fire"


# --- load_warm_chunks: centre range unioned with one ring -------------------


def test_load_warm_chunks_is_centre_range_then_ring() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    cx0, cy0, cx1, cy1 = 2, 5, 4, 6
    chunks = margin_warm.load_warm_chunks(cache, 0, cx0, cy0, cx1, cy1)

    expected_centre = [(cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)]
    expected_ring = margin_warm.ring_chunks(cache, 0, cx0, cy0, cx1, cy1, lead=(0, 0))
    assert chunks == expected_centre + expected_ring
    assert len(expected_centre) == 6, "a 3x2 viewport should have 6 centre chunks -- vacuous otherwise"


def test_load_warm_chunks_drops_already_resident_centre_chunks() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    cache._cache.add((0, 2, 5))
    chunks = margin_warm.load_warm_chunks(cache, 0, 2, 5, 4, 6)
    assert (2, 5) not in chunks


# --- bounded_chunk_range: the fit-to-view guard -----------------------------


def test_bounded_chunk_range_shrinks_a_fit_to_view_range_to_the_span() -> None:
    """The regression guard for the bug the load-time warm's raw projection
    hits at every real call site: load_scenario() always fires
    _start_level_warm() right after a fit-to-view render, so
    viewport_chunk_target_at()'s raw range covers the WHOLE neighbour-mip
    grid, not a viewport's worth. bounded_chunk_range() must cut that down
    to span_w x span_h, centred, not just clamp the edges."""
    cx0, cy0, cx1, cy1 = margin_warm.bounded_chunk_range(0, 0, 19, 19, span_w=3, span_h=2)
    assert (cx1 - cx0 + 1, cy1 - cy0 + 1) == (3, 2)
    # Centred on the input range's own midpoint (9, 9).
    assert cx0 <= 9 <= cx1
    assert cy0 <= 9 <= cy1


def test_bounded_chunk_range_slides_to_fit_near_an_edge_without_shrinking() -> None:
    """A span centred near the input range's own edge must slide to fit,
    not shrink -- an even-width input range's midpoint floors DOWN (Python
    integer division), which without the final clamp-to-input-range step
    would push the span outside the input range entirely; see the
    function's own docstring for the [5, 6]/span=2 example this guards."""
    cx0, cy0, cx1, cy1 = margin_warm.bounded_chunk_range(2, 5, 4, 6, span_w=3, span_h=2)
    assert (cx1 - cx0 + 1, cy1 - cy0 + 1) == (3, 2), "must not shrink just because it's near an edge"
    assert 2 <= cx0 and cx1 <= 4
    assert 5 <= cy0 and cy1 <= 6


def test_bounded_chunk_range_never_exceeds_the_input_range() -> None:
    """A span_w/span_h bigger than what viewport_chunk_target_at() actually
    projected has nothing extra to clamp to -- must not request chunks
    outside the range it was handed."""
    cx0, cy0, cx1, cy1 = margin_warm.bounded_chunk_range(2, 5, 4, 6, span_w=99, span_h=99)
    assert (cx0, cy0, cx1, cy1) == (2, 5, 4, 6)


# --- level_rect_for: hand-computed sanity check -----------------------------


def test_level_rect_for_hand_computed() -> None:
    from descape.viewer_canvas import level_rect_for

    cache = _FakeCache({0: (1000, 800)}, mip_scales={0: 2.0})

    assert level_rect_for(cache, 0, _Rect(50, 30, 210, 170)) == (25, 15, 105, 85)
    # High edge clamps to canvas_dims(mip).
    assert level_rect_for(cache, 0, _Rect(1900, 1500, 2100, 1700)) == (950, 750, 1000, 800)
    # Fully outside the canvas -> degenerate -> None.
    assert level_rect_for(cache, 0, _Rect(3000, 3000, 3100, 3100)) is None


# --- is_level_resident/has_chunk: predicate parity --------------------------


@pytest.fixture
def _sprite_install(tmp_path, monkeypatch):
    """test_level_warm.py's own sprite_install fixture, duplicated rather
    than imported: importing a pytest fixture under any name and then
    using that same name as a test parameter is flagged by ruff's F811 (a
    parameter necessarily "redefines" whatever name it shadows) -- there is
    no way to reuse a fixture across test modules that both satisfies
    pytest's own by-name fixture lookup and avoids that lint, so the dozen
    lines of setup are kept here instead."""
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


def test_is_level_resident_agrees_with_level_warm_job_after_a_completed_warm(_sprite_install) -> None:
    """Pins render_cache's two predicates together: is_level_resident()
    answers "safe to get_chunk() right now", level_warm_job() answers "is
    there anything worth WARMING" -- distinct questions that must still
    agree once a level warm has actually completed, or a future change to
    one could silently diverge from the other with no test noticing."""
    scenario = _scenario([[Unit(6.5, 6.5, CONST)]])
    iso = _iso_cache(scenario)
    assert not iso.is_level_resident(1), "level 1 already resident -- vacuous"
    _warm(iso, [1]).run_to_completion()
    assert iso.is_level_resident(1)
    assert iso.level_warm_job(1) is None

    flat = _flat_cache(scenario)
    assert not flat.is_level_resident(1), "level 1 already resident -- vacuous"
    _warm(flat, [1]).run_to_completion()
    assert flat.is_level_resident(1)
    assert flat.level_warm_job(1) is None


# --- gui: the viewer hooks ---------------------------------------------------


def _zoomed_window():
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(FIXTURE_PATH)
    window.resize(300, 300)
    QApplication.processEvents()
    window.map_view.scale(6.0, 6.0)
    QApplication.processEvents()
    return window


@pytest.mark.gui
def test_viewport_chunk_target_matches_a_real_paint() -> None:
    """The drift guard: after a real paint cycle, viewport_chunk_target()'s
    mip must equal what MapCanvasItem._last_mip actually painted, and its
    chunk range must contain every chunk any real paint's own exposed rect
    resolved to at that mip."""
    from PyQt5.QtWidgets import QApplication

    from descape.viewer_canvas import level_rect_for

    window = _zoomed_window()
    try:
        item = window.map_view._canvas_item
        assert item is not None, "Stepped mode produced no MapCanvasItem"
        cache = item._cache

        exposed_ranges = []
        original_paint = item.paint

        def spy_paint(painter, option, widget=None):
            mip = item._select_mip(painter)
            rect = option.exposedRect.intersected(item._bounding_rect)
            if not rect.isEmpty():
                level_rect = level_rect_for(cache, mip, rect)
                if level_rect is not None:
                    x0, y0, x1, y1 = level_rect
                    exposed_ranges.append((mip, *cache.chunk_index_range(mip, x0, y0, x1, y1)))
            original_paint(painter, option, widget)

        hbar, vbar = window.map_view.horizontalScrollBar(), window.map_view.verticalScrollBar()
        # Settle every step but the last WITHOUT the spy installed: Qt's
        # scroll optimization only repaints the newly-revealed sliver on
        # each step, so an EARLIER step's sliver sits somewhere the viewport
        # has since scrolled past -- comparing it against the FINAL settled
        # target would fail for a reason that has nothing to do with
        # viewport_chunk_target() being wrong. Only the LAST settle's own
        # paint(s) describe the same viewport state viewport_chunk_target()
        # is about to read.
        for _ in range(3):
            hbar.setValue(hbar.value() + 150)
            vbar.setValue(vbar.value() + 75)
            QApplication.processEvents()

        item.paint = spy_paint
        hbar.setValue(hbar.value() + 150)
        vbar.setValue(vbar.value() + 75)
        QApplication.processEvents()

        assert exposed_ranges, "the final settle never painted anything -- vacuous"

        target = window.map_view.viewport_chunk_target()
        assert target is not None
        mip, cx0, cy0, cx1, cy1 = target
        assert mip == item._last_mip

        canvas_w, canvas_h = cache.canvas_dims(mip)
        total = math.ceil(canvas_w / cache.chunk_px) * math.ceil(canvas_h / cache.chunk_px)
        assert (cx1 - cx0 + 1) * (cy1 - cy0 + 1) < total, "target range covers the whole grid -- vacuous"

        for m, ex0, ey0, ex1, ey1 in exposed_ranges:
            if m != mip:
                continue
            assert cx0 <= ex0 and ex1 <= cx1 and cy0 <= ey0 and ey1 <= cy1, (
                f"a real paint's own exposed rect resolved to chunks {(ex0, ey0, ex1, ey1)} at mip {m}, "
                f"outside viewport_chunk_target()'s range {(cx0, cy0, cx1, cy1)}"
            )
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_scrollbar_move_wheel_and_resize_each_start_the_poll() -> None:
    from test_zoom_status import _wheel

    window = _zoomed_window()
    try:
        timer = window.map_view._viewport_poll_timer

        timer.stop()
        assert not timer.isActive()
        window.map_view.horizontalScrollBar().setValue(window.map_view.horizontalScrollBar().value() + 10)
        assert timer.isActive(), "scrollContentsBy must start the poll"

        timer.stop()
        assert not timer.isActive()
        _wheel(window.map_view, up=True)
        assert timer.isActive(), "wheelEvent must start the poll"

        timer.stop()
        assert not timer.isActive()
        # resizeEvent's own body is one line (super() then
        # _capture_zoom_baseline()) -- calling the latter directly is what's
        # actually being checked here, and sidesteps the offscreen
        # platform's own unrelated flakiness around whether/when a real
        # top-level resize() is delivered to a deeply-nested child widget.
        window.map_view._capture_zoom_baseline()
        assert timer.isActive(), "_capture_zoom_baseline (which resizeEvent calls) must start the poll"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_poll_is_not_starved_by_continuous_motion_but_stops_when_still() -> None:
    """The regression guard for A2.4's central trap: a REPEATING poll that
    keeps firing (and notifying) across several moves, then stops itself
    the first time it finds nothing changed -- never a restarted single-
    shot that would go silent for the whole duration of a continuous drag.
    Drives _on_viewport_poll_tick() directly (no event loop needed), the
    same way LevelWarmer's own tests drive tick() directly."""
    window = _zoomed_window()
    try:
        map_view = window.map_view
        calls = []
        map_view.on_viewport_changed = lambda: calls.append(None)
        map_view._last_viewport_target = None

        map_view._on_viewport_poll_tick()  # baseline fire: None -> real target
        assert len(calls) == 1

        hbar = map_view.horizontalScrollBar()
        hbar.setValue(hbar.value() + 400)
        map_view._on_viewport_poll_tick()  # moved since the last fire
        assert len(calls) == 2, "continuous motion must not starve the hook"

        map_view._viewport_poll_timer.start()
        map_view._on_viewport_poll_tick()  # nothing moved since the previous fire
        assert len(calls) == 2, "an unchanged target must not re-notify"
        assert not map_view._viewport_poll_timer.isActive(), "an unchanged target must stop the poll"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_on_viewport_changed_starts_nothing_with_the_setting_off(monkeypatch) -> None:
    from descape import settings

    monkeypatch.setattr(settings, "_preload_zoom_levels", False)
    window = _zoomed_window()
    try:
        window._on_viewport_changed()
        assert not window._margin_warmer.is_active
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_on_viewport_changed_warms_a_ring_that_bounds_residency() -> None:
    """Step A7's residency bound: after a real pan and a drained margin
    warm, chunks in the first ring out are cached; chunks two rings out are
    not -- bounding the memory claim empirically rather than by argument
    alone."""
    from descape import margin_warm as margin_warm_module

    window = _zoomed_window()
    try:
        window._on_viewport_changed()
        assert window._margin_warmer.is_active, "no ring was queued -- vacuous"
        window._margin_warmer.run_to_completion()

        cache = window._cache
        target = window.map_view.viewport_chunk_target()
        assert target is not None
        mip, cx0, cy0, cx1, cy1 = target

        first_ring = margin_warm_module.ring_chunks(cache, mip, cx0, cy0, cx1, cy1)
        assert first_ring == [], "every first-ring chunk should already be warmed and dropped -- vacuous otherwise"

        second_ring = margin_warm_module.ring_chunks(cache, mip, cx0 - 1, cy0 - 1, cx1 + 1, cy1 + 1)
        assert second_ring, "no second ring exists on this canvas -- vacuous"
        for cx, cy in second_ring:
            assert not cache.has_chunk(mip, cx, cy), (
                f"chunk {(cx, cy)} two rings out from the viewport was warmed -- the ring isn't bounded"
            )
    finally:
        conftest.close_window(window)


# --- viewport_chunk_target_at / the load-time warm --------------------------


@pytest.mark.gui
def test_viewport_chunk_target_at_matches_viewport_chunk_target_at_the_live_mip() -> None:
    """viewport_chunk_target_at(mip) reuses viewport_chunk_target()'s own
    scene-rect projection (2026-09-07 plan's load-time margin warm, Step
    1) -- calling it with the mip a real paint already selected must agree
    with viewport_chunk_target() exactly, not just approximately."""
    window = _zoomed_window()
    try:
        target = window.map_view.viewport_chunk_target()
        assert target is not None
        mip, cx0, cy0, cx1, cy1 = target

        at_live_mip = window.map_view.viewport_chunk_target_at(mip)
        assert at_live_mip == (cx0, cy0, cx1, cy1)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_viewport_chunk_target_at_none_with_no_canvas_item() -> None:
    from descape.map_view import MapView

    window = conftest.shown_window()
    try:
        view = window.map_view
        assert isinstance(view, MapView)
        assert view._canvas_item is None, "a freshly-shown window with no document -- vacuous otherwise"
        assert view.viewport_chunk_target_at(0) is None
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_load_scenario_warms_both_neighbour_mips_chunks_with_no_pan_or_zoom() -> None:
    """The actual claim this feature makes: opening a file alone -- no
    simulated pan/zoom at all -- ends with a neighbour mip's own
    viewport-sized centre-plus-ring already resident, once the level and
    load warmers are both drained."""
    window = conftest.stepped_window(FIXTURE_PATH)
    try:
        assert window._cache is not None
        fit = window.map_view._fit_baseline_scale()
        mips = level_warm.neighbour_mips(window._cache, fit * window.map_view.devicePixelRatioF())
        assert mips, "the opening level has no neighbours on this fixture -- vacuous"

        window._level_warmer.run_to_completion()
        for _ in range(len(mips) + 1):
            if not window._load_warmer.is_active and not window._load_warm_queue:
                break
            window._load_warmer.run_to_completion()

        assert not window._load_warmer.is_active
        assert not window._load_warm_queue

        for mip in mips:
            assert window._cache.is_level_resident(mip), f"mip {mip}'s sprite layer never warmed -- vacuous"
            target = window.map_view.viewport_chunk_target_at(mip)
            assert target is not None
            span_w, span_h = window.map_view.viewport_chunk_span(window._cache, mip)
            cx0, cy0, cx1, cy1 = margin_warm.bounded_chunk_range(*target, span_w, span_h)

            gw, gh = window._cache.canvas_dims(mip)
            total = math.ceil(gw / window._cache.chunk_px) * math.ceil(gh / window._cache.chunk_px)
            queued = (cx1 - cx0 + 1) * (cy1 - cy0 + 1)
            assert queued < total, (
                f"mip {mip}'s queued range covers the whole grid ({queued}/{total}) -- the load-time warm "
                "must never frontload a whole finer level, only a viewport-sized patch"
            )

            for cx, cy in margin_warm.load_warm_chunks(window._cache, mip, cx0, cy0, cx1, cy1):
                assert window._cache.has_chunk(mip, cx, cy), (
                    f"chunk {(cx, cy)} at mip {mip} was never warmed by the load-time warm"
                )
    finally:
        conftest.close_window(window)
