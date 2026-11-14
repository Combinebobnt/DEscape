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
from test_level_warm import MAP_W, Unit, _flat_cache, _iso_cache, _scenario, _warm

from descape import level_warm, margin_warm
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH

import conftest
from test_unit_sprites import CONST


class _FakeCache:
    """Duck-typed cache exposing only what margin_warm.py and
    viewer_canvas.level_rect_for() call -- no real compositing, so ring/
    driver geometry is testable with no scenario, no PyQt5, no QApplication.
    """

    def __init__(self, canvas_dims, chunk_px: int = 512, mip_scales=None, resident_mips=(), clock=None, primed=None):
        self._canvas_dims = canvas_dims  # {mip: (w, h)}
        self.chunk_px = chunk_px
        self._mip_scales = mip_scales or {}
        self._resident_mips = set(resident_mips)
        # level_warmed()'s primed half; None: every resident mip counts as primed.
        self.primed = set(resident_mips) if primed is None else set(primed)
        self._cache: set[tuple[int, int, int]] = set()
        self.get_chunk_calls: list[tuple[int, int, int]] = []
        self.fail_on: set[tuple[int, int, int]] = set()
        # With a _Clock, each get_chunk advances it by cost_ms[(cx, cy)] (default 1ms).
        self._clock = clock
        self.cost_ms: dict[tuple[int, int], float] = {}

    def canvas_dims(self, mip: int = 0):
        return self._canvas_dims[mip]

    def mip_scale(self, mip: int = 0):
        return self._mip_scales[mip]

    def has_chunk(self, mip, cx, cy) -> bool:
        return (mip, cx, cy) in self._cache

    def is_level_resident(self, mip: int) -> bool:
        return mip in self._resident_mips

    def level_warmed(self, mip: int) -> bool:
        """render_cache's rule: primed, or any chunk composited at the mip."""
        return mip in self.primed or any(key[0] == mip for key in self._cache)

    def get_chunk(self, mip, cx, cy):
        key = (mip, cx, cy)
        if key in self.fail_on:
            raise RuntimeError("boom")
        if self._clock is not None:
            self._clock.t += self.cost_ms.get((cx, cy), 1.0) / 1000.0
        self.get_chunk_calls.append(key)
        self._cache.add(key)
        return key


class _Clock:
    """margin_warm._now's stand-in: time moves only when a fake chunk runs."""

    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(margin_warm, "_now", fake)
    return fake


@pytest.fixture
def one_chunk_ticks(monkeypatch) -> None:
    """A zero budget degenerates to exactly one chunk per tick, the pre-T1
    contract, which the queue/cancel/drain tests below are written against."""
    monkeypatch.setattr(margin_warm, "BUDGET_MS", 0)


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


@pytest.mark.usefixtures("one_chunk_ticks")
def test_margin_warmer_zero_budget_warms_exactly_one_chunk_per_tick() -> None:
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


@pytest.mark.usefixtures("one_chunk_ticks")
def test_margin_warmer_cancel_drops_the_rest() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(1, 1), (2, 2)])
    warmer.tick()

    warmer.cancel()
    warmer.run_to_completion()

    assert not warmer.is_active
    assert cache.get_chunk_calls == [(0, 1, 1)]


@pytest.mark.usefixtures("one_chunk_ticks")
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
    monkeypatch.setattr(debug_log, "log", lambda message, **kwargs: logged.append(message))

    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(1, 1), (2, 2), (3, 3)])
    warmer.run_to_completion()

    assert logged, "the failing chunk was never logged -- vacuous"
    assert cache.get_chunk_calls == [(0, 1, 1), (0, 3, 3)]


@pytest.mark.usefixtures("one_chunk_ticks")
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


def test_margin_warmer_pauses_while_a_mouse_button_is_held(monkeypatch) -> None:
    """Item 23: a held mouse button means discrete move events with idle
    gaps between them, so the 0ms timer would otherwise land a chunk
    mid-drag as a hitch. tick() must warm nothing while held. No
    QApplication here, so there's no real QTimer to check the backed-off
    interval on -- that half is checked separately, gui-marked, below."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "mouseButtons", staticmethod(lambda: Qt.LeftButton))

    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(1, 1), (2, 2)])

    assert warmer.tick() is True
    assert cache.get_chunk_calls == [], "held means no chunk warmed"
    assert warmer.tick() is True
    assert cache.get_chunk_calls == [], "still held, still nothing warmed"
    assert warmer.is_active


@pytest.mark.usefixtures("one_chunk_ticks")
def test_margin_warmer_resumes_and_drains_on_release(monkeypatch) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    held = {"value": True}
    monkeypatch.setattr(
        QApplication, "mouseButtons", staticmethod(lambda: Qt.LeftButton if held["value"] else Qt.NoButton)
    )

    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(1, 1), (2, 2)])

    warmer.tick()
    assert cache.get_chunk_calls == [], "still held -- vacuous otherwise"

    held["value"] = False
    assert warmer.tick() is True
    assert cache.get_chunk_calls == [(0, 1, 1)]
    assert warmer.tick() is False
    assert cache.get_chunk_calls == [(0, 1, 1), (0, 2, 2)]
    assert not warmer.is_active


@pytest.mark.gui
def test_margin_warmer_backs_off_the_timer_interval_while_held_and_restores_it(monkeypatch) -> None:
    """The other half of item 23: returning early with the 0ms repeating
    timer still armed would spin a core for the whole drag, so tick() must
    retarget the real QTimer's interval, not just skip the warm. Needs a
    real QApplication for a real QTimer to exist on the driver at all."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    conftest.ensure_qapp()
    held = {"value": True}
    monkeypatch.setattr(
        QApplication, "mouseButtons", staticmethod(lambda: Qt.LeftButton if held["value"] else Qt.NoButton)
    )

    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(1, 1), (2, 2)])

    warmer.tick()
    assert warmer._timer.interval() == margin_warm.MarginWarmer.HELD_INTERVAL_MS

    held["value"] = False
    warmer.tick()
    assert warmer._timer.interval() == 0
    warmer.cancel()


def test_wheel_driven_retarget_is_unaffected_by_the_held_gate() -> None:
    """A wheel/keyboard pan holds no mouse button, so a retarget through
    that path must warm normally -- the existing no-monkeypatch tests
    already prove this implicitly (no QApplication means mouseButtons()
    reads NoButton), but this pins it explicitly against item 23's own
    claim rather than relying on that as an accident of test setup."""
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(1, 1)])
    assert warmer.tick() is False
    assert cache.get_chunk_calls == [(0, 1, 1)]


@pytest.mark.usefixtures("one_chunk_ticks")
def test_margin_warmer_on_drained_does_not_fire_on_cancel() -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    drained = []
    warmer.start(cache, 0, [(1, 1), (2, 2)], on_drained=lambda: drained.append(True))
    warmer.tick()
    warmer.cancel()
    assert drained == [], "a cancelled queue never drained -- the callback must not fire"


# --- MarginWarmer: the budgeted tick (Batch F T1) ----------------------------


def test_budgeted_tick_warms_several_chunks_while_the_next_still_fits(clock) -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0}, clock=clock)
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(i, 0) for i in range(8)])

    # 2.5ms chunks, 8ms budget: after two (5ms) a third fits, after three (7.5ms) a fourth doesn't.
    for i in range(8):
        cache.cost_ms[(i, 0)] = 2.5
    assert warmer.tick() is True
    assert cache.get_chunk_calls == [(0, 0, 0), (0, 1, 0), (0, 2, 0)]
    assert warmer.tick() is True
    assert len(cache.get_chunk_calls) == 6


def test_budgeted_tick_stops_when_the_slowest_chunk_so_far_would_overrun(clock) -> None:
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0}, clock=clock)
    cache.cost_ms[(0, 0)] = 5.0
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0)])

    assert warmer.tick() is True
    assert cache.get_chunk_calls == [(0, 0, 0)], "5ms + a predicted 5ms passes 8ms, so the tick must stop"


def test_budgeted_tick_always_runs_one_chunk_even_past_the_budget(clock) -> None:
    """Otherwise a chunk slower than the budget would never be warmed."""
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0}, clock=clock)
    cache.cost_ms[(0, 0)] = 20.0
    cache.cost_ms[(1, 0)] = 20.0
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0)])

    assert warmer.tick() is True
    assert cache.get_chunk_calls == [(0, 0, 0)]
    assert warmer.tick() is False
    assert cache.get_chunk_calls == [(0, 0, 0), (0, 1, 0)]


def test_budgeted_tick_logs_a_bad_chunk_and_continues_in_the_same_tick(clock, monkeypatch) -> None:
    from descape import debug_log

    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0}, clock=clock)
    cache.fail_on.add((0, 1, 0))
    logged = []
    monkeypatch.setattr(debug_log, "log", lambda message, **kwargs: logged.append(message))
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0), (2, 0)])

    assert warmer.tick() is False
    assert logged, "the failing chunk was never logged -- vacuous"
    assert cache.get_chunk_calls == [(0, 0, 0), (0, 2, 0)]


def test_budgeted_tick_ends_at_the_drain_and_leaves_a_chained_queue_for_the_next_tick(clock) -> None:
    """on_drained may start() a fresh queue on this same warmer (the load warm
    chains the next mip that way); that queue must not be eaten by the budget
    left in the tick that drained the first one."""
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0}, clock=clock)
    warmer = margin_warm.MarginWarmer()
    drained = []

    def chain() -> None:
        drained.append(True)
        warmer.start(cache, 0, [(9, 9)])

    warmer.start(cache, 0, [(0, 0), (1, 0)], on_drained=chain)
    assert warmer.tick() is False
    assert drained == [True]
    assert cache.get_chunk_calls == [(0, 0, 0), (0, 1, 0)]
    assert warmer.is_active, "the chained queue was dropped or consumed"

    assert warmer.tick() is False
    assert cache.get_chunk_calls[-1] == (0, 9, 9)
    assert drained == [True], "on_drained is per start(), and the chained start() passed none"


def test_budgeted_tick_reports_tick_and_slowest_chunk_to_perf_trace(clock, monkeypatch) -> None:
    from descape import perf_trace

    monkeypatch.setattr(perf_trace, "_enabled", True)
    for name, value in perf_trace._fresh_state().items():
        monkeypatch.setattr(perf_trace, name, value)
    monkeypatch.setattr(perf_trace, "_idle_scheduler", None)
    cache = _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0}, clock=clock)
    cache.cost_ms[(0, 0)] = 1.0
    cache.cost_ms[(1, 0)] = 4.0
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0), (2, 0)])

    warmer.tick()
    [(tick_ms, chunks, slowest_ms)] = perf_trace._warm_ticks
    assert chunks == 2, "1ms then 4ms; 5ms + a predicted 4ms passes 8ms"
    assert tick_ms == pytest.approx(5.0)
    assert slowest_ms == pytest.approx(4.0)


# --- MarginWarmer: the cold-first defer (TASK-031.82 D3) ---------------------


@pytest.fixture
def input_clock(monkeypatch) -> _Clock:
    """level_warm's input clock, shared with margin_warm's; no stamp yet."""
    fake = _Clock()
    monkeypatch.setattr(margin_warm, "_now", fake)
    monkeypatch.setattr(level_warm, "_now", fake)
    monkeypatch.setattr(level_warm, "_last_input", None)
    return fake


def _cold_cache(clock, **kwargs) -> _FakeCache:
    return _FakeCache({0: (20 * 512, 20 * 512)}, resident_mips={0}, clock=clock, primed=(), **kwargs)


def test_a_cold_level_defers_while_input_is_live_and_runs_once_it_is_quiet(input_clock) -> None:
    cache = _cold_cache(input_clock)
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0)])
    level_warm.note_input()
    before = margin_warm.deferrals

    assert warmer.tick() is True
    assert cache.get_chunk_calls == [], "a cold first chunk ran inside the input window"
    assert warmer._queue == [(0, 0), (1, 0)], "the deferral dropped queued chunks"
    assert margin_warm.deferrals == before + 1

    input_clock.t += level_warm.INPUT_QUIET_MS / 1000.0
    assert warmer.tick() is False
    assert cache.get_chunk_calls == [(0, 0, 0), (0, 1, 0)]


@pytest.mark.parametrize("warmed_by", ("primed", "composited"))
def test_a_warmed_level_runs_inside_the_input_window(input_clock, warmed_by) -> None:
    cache = _cold_cache(input_clock)
    if warmed_by == "primed":
        cache.primed.add(0)
    else:
        cache._cache.add((0, 5, 5))
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0)])
    level_warm.note_input()
    assert warmer.tick() is False
    assert cache.get_chunk_calls == [(0, 0, 0)]


def test_a_cold_level_runs_with_no_input_stamp_or_with_the_input_budget_pinned(input_clock, monkeypatch) -> None:
    """The bench's --no-input-budget sets INPUT_BUDGET_MS = BUDGET_MS, so
    tick_budget_ms() always reads the input budget: the defer must read
    input_live(), or an old stamp would defer a never-warmed level forever."""
    cache = _cold_cache(input_clock)
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0)])
    assert warmer.tick() is False, "no stamp: nothing to defer for"

    monkeypatch.setattr(level_warm, "INPUT_BUDGET_MS", level_warm.BUDGET_MS)
    level_warm.note_input()
    input_clock.t += 1.0
    assert level_warm.tick_budget_ms() == level_warm.INPUT_BUDGET_MS, "vacuous: the pin is off"
    warmer.start(cache, 0, [(1, 0)])
    cache._cache.clear()
    assert warmer.tick() is False
    assert cache.get_chunk_calls == [(0, 0, 0), (0, 1, 0)]


@pytest.mark.parametrize("opt_out", ("switch", "run_to_completion"))
def test_the_defer_never_applies_when_opted_out(input_clock, monkeypatch, opt_out) -> None:
    cache = _cold_cache(input_clock)
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0)])
    if opt_out == "switch":
        monkeypatch.setattr(margin_warm, "DEFER_COLD_FIRST", False)
    level_warm.note_input()
    if opt_out == "run_to_completion":
        warmer.run_to_completion()
    else:
        warmer.tick()
    assert cache.get_chunk_calls[:1] == [(0, 0, 0)]


def test_an_empty_queue_drains_and_fires_on_drained_inside_the_input_window(input_clock) -> None:
    cache = _cold_cache(input_clock)
    warmer = margin_warm.MarginWarmer()
    drained = []
    warmer.start(cache, 0, [(0, 0)], on_drained=lambda: drained.append(True))
    warmer._queue = []
    level_warm.note_input()
    assert warmer.tick() is False
    assert drained == [True]


def test_a_deferral_keeps_in_flight_jobs_and_their_results_install(inline_pool, input_clock) -> None:
    cache = _JobCache({0: (20 * 512, 20 * 512)}, resident_mips={0}, clock=input_clock, primed=())
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0), (2, 0)])
    assert warmer.tick() is True
    assert warmer._in_flight, "vacuous: nothing in flight"
    queued = list(warmer._queue)
    cache._cache.clear()  # the level reads cold again, as on a new level after a crossing
    level_warm.note_input()
    assert warmer.tick() is True
    assert warmer._queue == queued
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    assert cache.installed, "an in-flight result was dropped by the deferral"
    warmer.cancel()


@pytest.mark.gui
def test_a_deferral_backs_the_timer_off_and_the_next_quiet_tick_restores_it(input_clock) -> None:
    conftest.ensure_qapp()
    cache = _cold_cache(input_clock)
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0)])
    level_warm.note_input()
    warmer.tick()
    assert warmer._timer.interval() == margin_warm.MarginWarmer.HELD_INTERVAL_MS
    input_clock.t += 1.0
    warmer.tick()
    assert warmer._timer.interval() == 0
    warmer.cancel()


def test_a_deferral_reaches_the_perf_view_line(input_clock, monkeypatch) -> None:
    from descape import perf_trace

    monkeypatch.setattr(perf_trace, "_enabled", True)
    for name, value in perf_trace._fresh_state().items():
        monkeypatch.setattr(perf_trace, name, value)
    monkeypatch.setattr(perf_trace, "_idle_scheduler", None)
    cache = _cold_cache(input_clock)
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0)])
    level_warm.note_input()
    warmer.tick()
    warmer.tick()
    assert perf_trace._warm_deferred == {0: 2}
    assert "deferred 2 ticks (cold levels 0 x2)" in perf_trace._warm_summary()


# --- MarginWarmer: worker threads (Batch F T2) -------------------------------


class _FakeJob:
    def __init__(self, key, epoch) -> None:
        self.key = key
        self.epoch = epoch

    def run(self) -> bool:
        return True


class _JobCache(_FakeCache):
    """_FakeCache plus render_cache's worker hooks, with install_chunk()'s
    epoch rule; prepare costs cost_ms on the clock, the job itself nothing."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epoch = 0
        self.prepared: list[tuple[int, int, int]] = []
        self.installed: list[tuple[int, int, int]] = []

    def prepare_chunk_job(self, mip, cx, cy):
        key = (mip, cx, cy)
        if self._clock is not None:
            self._clock.t += self.cost_ms.get((cx, cy), 1.0) / 1000.0
        self.prepared.append(key)
        return _FakeJob(key, self.epoch)

    def install_chunk(self, job, painted) -> bool:
        if not painted or job.epoch != self.epoch or job.key in self._cache:
            return False
        self._cache.add(job.key)
        self.installed.append(job.key)
        return True


class _InlineExecutor:
    """Runs a job at submit(); its result still arrives by queued signal."""

    def submit(self, fn, *args):
        fn(*args)


@pytest.fixture
def inline_pool(monkeypatch):
    conftest.ensure_qapp()
    monkeypatch.setattr(margin_warm, "_executor", _InlineExecutor())
    monkeypatch.setattr(margin_warm, "WORKERS", 2)


def _pump(warmer, limit: int = 200) -> None:
    from PyQt5.QtWidgets import QApplication

    for _ in range(limit):
        if not warmer.is_active:
            return
        QApplication.processEvents()
    pytest.fail("the warm never drained")


def test_pooled_chunks_install_only_when_the_event_loop_delivers_them(inline_pool, clock) -> None:
    cache = _JobCache({0: (20 * 512, 20 * 512)}, resident_mips={0}, clock=clock)
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0)])

    assert warmer.tick() is True
    assert cache.prepared == [(0, 0, 0), (0, 1, 0)]
    assert cache.installed == [], "a result installed inside the tick, not via the queued signal"
    assert cache.get_chunk_calls == [], "a job-capable chunk ran synchronously"
    assert warmer.is_active, "chunks in flight must keep the warm active"

    _pump(warmer)
    assert cache.installed == [(0, 0, 0), (0, 1, 0)]


def test_pooled_tick_stops_at_workers_in_flight_and_resumes_on_a_result(inline_pool, clock) -> None:
    cache = _JobCache({0: (20 * 512, 20 * 512)}, resident_mips={0}, clock=clock)
    warmer = margin_warm.MarginWarmer()
    chunks = [(i, 0) for i in range(5)]
    warmer.start(cache, 0, chunks)

    assert warmer.tick() is True
    assert len(cache.prepared) == 2, "1ms preps fit the budget, so only the in-flight cap can stop at 2"
    assert not warmer._timer.isActive(), "a full pool must stop the idle timer, not spin it"

    _pump(warmer)
    assert cache.installed == [(0, cx, cy) for cx, cy in chunks]


def test_on_drained_waits_for_the_last_in_flight_result(inline_pool) -> None:
    cache = _JobCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    drained = []
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0)], on_drained=lambda: drained.append(len(cache.installed)))

    assert warmer.tick() is True, "the queue is empty but a chunk is still in flight"
    assert drained == []
    _pump(warmer)
    assert drained == [1], "on_drained fired before the in-flight chunk landed, or more than once"


def test_cancel_drops_in_flight_results_and_never_fires_on_drained(inline_pool) -> None:
    cache = _JobCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    drained = []
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0)], on_drained=lambda: drained.append(True))
    warmer.tick()
    warmer.cancel()

    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    assert cache.prepared, "nothing was dispatched -- vacuous"
    assert cache.installed == []
    assert drained == []
    assert not warmer.is_active


def test_a_retarget_drops_the_previous_rings_results(inline_pool) -> None:
    cache = _JobCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0)])
    warmer.tick()
    warmer.start(cache, 0, [(5, 5)])

    _pump(warmer)
    assert cache.installed == [(0, 5, 5)]


def test_a_mutation_before_delivery_rejects_the_result(inline_pool) -> None:
    """The epoch rule seen from the driver: the chunk is dropped, not
    installed stale, and the warm still drains."""
    cache = _JobCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0)])
    warmer.tick()
    cache.epoch += 1

    _pump(warmer)
    assert cache.prepared == [(0, 0, 0)]
    assert cache.installed == []
    assert not cache.has_chunk(0, 0, 0)


def test_a_worker_exception_is_logged_and_the_warm_continues(inline_pool, monkeypatch) -> None:
    from descape import debug_log

    logged = []
    monkeypatch.setattr(debug_log, "log", lambda message, **kwargs: logged.append(message))
    monkeypatch.setattr(_FakeJob, "run", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    cache = _JobCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0)])

    _pump(warmer)
    assert len(logged) == 2 and "boom" in logged[0]
    assert cache.installed == []


def test_run_to_completion_requeues_chunks_already_in_flight(inline_pool) -> None:
    cache = _JobCache({0: (20 * 512, 20 * 512)}, resident_mips={0})
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0), (2, 0)])
    warmer.tick()
    assert warmer._in_flight, "nothing in flight -- vacuous"

    warmer.run_to_completion()
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    assert sorted(cache.get_chunk_calls) == [(0, 0, 0), (0, 1, 0), (0, 2, 0)]
    assert cache.installed == [], "an abandoned result installed after the synchronous drain"
    assert not warmer.is_active


def test_no_workers_keeps_every_chunk_synchronous(inline_pool, monkeypatch, clock) -> None:
    monkeypatch.setattr(margin_warm, "WORKERS", 0)
    cache = _JobCache({0: (20 * 512, 20 * 512)}, resident_mips={0}, clock=clock)
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0)])

    assert warmer.tick() is False
    assert cache.prepared == []
    assert cache.get_chunk_calls == [(0, 0, 0), (0, 1, 0)]


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
    assert cx0 >= 2 and cx1 <= 4
    assert cy0 >= 5 and cy1 <= 6


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

    # A current level with deferred elevation work (2026-09-29 warm-tick plan):
    # neither resident nor warm-free until its flush job has run.
    from descape.render_bbox import dirty_screen_bbox_iso

    iso._level(0)
    scenario.map_manager.get_tile(6, 6).elevation += 1
    changed: set = set()
    bbox = dirty_screen_bbox_iso(
        scenario, [6 * MAP_W + 6], iso.elevations, iso.proj, with_units=True, with_sprites=True,
        elevation_changed=changed,
    )
    iso.patch(bbox, elevation_changed=changed, rebuild_levels=(0,))
    assert iso._levels[1].gen == iso._source_gen and iso._levels[1].pending_elev, "nothing deferred -- vacuous"
    assert not iso.is_level_resident(1)
    assert iso.level_warm_job(1) is not None
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


@pytest.mark.gui
def test_a_real_windows_margin_warm_reaches_the_perf_view_line(monkeypatch) -> None:
    """Batch F T1: the warm: bucket is fed by the viewer's own warmer, not
    only by a hand-driven one."""
    from descape import debug_log, perf_trace

    for name, value in perf_trace._fresh_state().items():
        monkeypatch.setattr(perf_trace, name, value)
    window = _zoomed_window()
    try:
        window._on_viewport_changed()
        assert window._margin_warmer.is_active, "no ring was queued -- vacuous"
        monkeypatch.setattr(perf_trace, "_enabled", True)
        debug_log.clear()
        window._margin_warmer.run_to_completion()
        perf_trace.flush_idle()
        assert "warm: " in debug_log.get_log_text()
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


@pytest.mark.gui
def test_top_down_flat_sprites_off_warms_both_neighbour_mips_chunks() -> None:
    """TASK-031.72: the twin above in top-down Flat with Show sprites off.
    No level job runs there, but every level is resident, so each
    neighbour's load warm still fills its chunk range (before, the level
    never read as resident and nothing was warmed)."""
    from PyQt5.QtWidgets import QApplication

    from descape.render_cache import FlatChunkCache

    window = conftest.stepped_window(FIXTURE_PATH)
    try:
        window._sprites_enabled = False
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        cache = window._cache
        assert isinstance(cache, FlatChunkCache) and not cache.sprites_enabled
        levels = cache.mip_levels()
        assert len(levels) >= 3, f"ladder {levels} has no level with two neighbours"
        mid = levels[len(levels) // 2]
        view_scale = 0.75 * 2.0 ** mid
        factor = view_scale / window.map_view.view_device_scale()
        window.map_view.scale(factor, factor)
        QApplication.processEvents()
        window._cancel_warms()
        window._start_level_warm()
        view = window.map_view.viewport_chunk_target()
        assert view is not None and view[0] == mid
        mips = level_warm.neighbour_mips_of(cache, view[0])
        assert len(mips) == 2, f"the view's level has neighbours {mips} -- not both sides"
        assert not window._level_warmer.is_active, "a sprites-off Flat level queued a job"
        for _ in range(len(mips) + 1):
            if not window._load_warmer.is_active and not window._load_warm_queue:
                break
            window._load_warmer.run_to_completion()
        assert not window._load_warmer.is_active and not window._load_warm_queue

        for mip in mips:
            target = window.map_view.viewport_chunk_target_at(mip)
            assert target is not None
            cx0, cy0, cx1, cy1 = window._neighbour_warm_range(mip, target, view)
            ring = [(cx, cy) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1)]
            missing = [chunk for chunk in ring if not cache.has_chunk(mip, *chunk)]
            assert not missing, f"mip {mip}: {len(missing)} of {len(ring)} chunks never warmed"
            assert cache.is_level_resident(mip)
    finally:
        conftest.close_window(window)


def _window_with_a_load_warm_in_flight(monkeypatch):
    """A Stepped window whose level warm has drained, so the load-time warm
    is ticking. monkeypatch pins the setting so a slot that writes it can't
    leak past teardown."""
    from descape import settings

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = conftest.stepped_window(FIXTURE_PATH)
    window._level_warmer.run_to_completion()
    assert window._load_warmer.is_active, "no load warm started -- vacuous"
    return window


@pytest.mark.gui
def test_turning_preload_off_drops_the_load_warm_and_its_queue(monkeypatch) -> None:
    """Batch F T0: the toggle used to cancel only the level and margin
    warms, so the load warm kept ticking with the setting off."""
    from descape.settings_dialog import SettingsDialog

    window = _window_with_a_load_warm_in_flight(monkeypatch)
    try:
        dialog = SettingsDialog(window)
        dialog._on_preload_zoom_levels_toggled(False)

        assert not window._load_warmer.is_active
        assert not window._load_warm_queue
        assert not window._level_warmer.is_active
        assert not window._margin_warmer.is_active
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_a_unit_filter_change_drops_every_in_flight_warm(monkeypatch) -> None:
    """Batch F T0: _on_filter_changed() used to call set_unit_filter() with
    warms still in flight, so a load-warm tick could rebuild a whole stale
    level. Goes through the real menu signal."""
    window = _window_with_a_load_warm_in_flight(monkeypatch)
    try:
        window._on_viewport_changed()
        window.show_trees_action.setChecked(False)

        assert window._cache.unit_filter.show_trees is False, "the filter never changed -- vacuous"
        assert not window._load_warmer.is_active
        assert not window._load_warm_queue
        assert not window._margin_warmer.is_active
        assert not window._level_warmer.is_active
    finally:
        conftest.close_window(window)


# --- the zoom-out reach (TASK-031.11) ----------------------------------------


def _reach_cache() -> _FakeCache:
    return _FakeCache({0: (60, 60)}, chunk_px=10)


def test_reach_chunks_order_by_the_factor_each_enters_the_view_about_a_centred_anchor() -> None:
    """D4 by hand: view x in [22, 28], anchor 25, so a chunk enters at
    (anchor - near edge) / 3 on its side; y overlaps everywhere."""
    chunks = margin_warm.reach_chunks(_reach_cache(), 0, (0, 0, 5, 0), (22, 0, 28, 10), (25, 5))
    # 2 overlaps (1); 1 and 3 enter at 5/3; 0 and 4 at 5; 5 at 25/3. Ties keep range order.
    assert chunks == [(2, 0), (1, 0), (3, 0), (0, 0), (4, 0), (5, 0)]


def test_reach_chunks_grow_the_far_side_of_an_edge_anchor_first() -> None:
    """Anchor 27 sits 1 from the right edge: the left edge moves five times
    as fast. Left reach 5: 1 at 7/5, 0 at 17/5. Right reach 1: 3 at 3, 4 at 13."""
    chunks = margin_warm.reach_chunks(_reach_cache(), 0, (0, 0, 5, 0), (22, 0, 28, 10), (27, 5))
    assert chunks == [(2, 0), (1, 0), (3, 0), (0, 0), (4, 0), (5, 0)]
    far = margin_warm.reach_chunks(_reach_cache(), 0, (0, 0, 5, 0), (22, 0, 28, 10), (23, 5))
    assert far == [(2, 0), (3, 0), (1, 0), (4, 0), (5, 0), (0, 0)]


def test_reach_chunks_past_a_view_edge_on_the_anchor_never_enter() -> None:
    """The right edge is the anchor: it never moves, so chunks past it sort
    last, in range order, behind every chunk the left side reaches."""
    chunks = margin_warm.reach_chunks(_reach_cache(), 0, (0, 0, 5, 0), (22, 0, 28, 10), (28, 5))
    assert chunks == [(2, 0), (1, 0), (0, 0), (3, 0), (4, 0), (5, 0)]


def test_reach_chunks_take_the_larger_axis_and_drop_cached_and_excluded_chunks() -> None:
    """3x3 around a centre view [12, 18]^2, anchor (13, 15): x reaches 1 left
    and 5 right, y 3 each way. A chunk enters once both axes do."""
    cache = _reach_cache()
    rng, view, anchor = (0, 0, 2, 2), (12, 12, 18, 18), (13, 15)
    assert margin_warm.reach_chunks(cache, 0, rng, view, anchor) == [
        (1, 1), (2, 1), (1, 0), (1, 2), (2, 0), (2, 2), (0, 0), (0, 1), (0, 2),
    ]
    cache._cache.add((0, 2, 1))
    assert margin_warm.reach_chunks(cache, 0, rng, view, anchor, exclude={(1, 1), (0, 0)}) == [
        (1, 0), (1, 2), (2, 0), (2, 2), (0, 1), (0, 2),
    ]


def _reach_window(style: str):
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(FIXTURE_PATH, size=1360)
    window.resize(1360, 1030)
    QApplication.processEvents()
    if style == "Flat":
        window.iso_action.setChecked(True)
        window.terrain_style_combo.setCurrentText("Flat")
        QApplication.processEvents()
    for set_policy in (window.map_view.setHorizontalScrollBarPolicy, window.map_view.setVerticalScrollBarPolicy):
        set_policy(Qt.ScrollBarAlwaysOn)
    QApplication.processEvents()
    return window


@pytest.mark.gui
@pytest.mark.parametrize("style", ["Stepped", "Flat"])
@pytest.mark.parametrize(("edge", "level"), [(False, -1), (True, -1), (False, 2)], ids=["centred", "edge", "native"])
def test_the_reach_holds_the_widest_view_a_zoom_out_shows_at_its_level(style, edge, level) -> None:
    """D1 against a real view: at residual 0.98, the reach is the chunks a
    paint reads once a zoom about the anchor reaches the widest view still
    on the level. Level -1 centred on the map's middle with the anchor at
    the viewport centre, and past the map's bottom-right corner with the
    anchor at the top-left, where the scrollbars clamp the zoom-out (level
    -1, since at level 0 this fixture's overscroll margin outgrows the
    widest view). Level 2 sits above the native tier (TASK-599), so its
    widest scale is 1.5, not a power of two. Iso Flat is an IsoChunkCache
    on zero elevations."""
    from PyQt5.QtCore import QPoint, QPointF

    from descape import zoom_preframe
    from descape.render_cache import IsoChunkCache

    window = _reach_window(style)
    try:
        view = window.map_view
        item = view._canvas_item
        cache = window._cache
        assert isinstance(cache, IsoChunkCache), "not an iso cache -- the Flat case needs Isometric View"
        levels = cache.mip_levels()
        assert levels == [-2, -1, 0, 1, 2, 3], "the fixture's ladder moved"
        if level == 2:
            assert 1.0 / cache.mip_scale(level - 1) != 2.0 ** (level - 1), "no native tier -- vacuous"
        scale = 0.98 / cache.mip_scale(level)
        factor = scale / view.view_device_scale()
        view.scale(factor, factor)
        bounds = item.boundingRect()
        vp = view.viewport().rect()
        w, h = vp.width() / scale, vp.height() / scale
        view.centerOn(bounds.bottomRight() + QPointF(0.3 * w, 0.3 * h) if edge else bounds.center())
        anchor = QPoint(8, 8) if edge else vp.center()
        view.set_zoom_anchor_mode(True)
        view._last_zoom_anchor = anchor
        start = view.viewport_chunk_target()
        assert start[0] == level
        reach = view.viewport_reach_target(level)
        assert reach is not None
        assert view.viewport_reach_target(levels[0]) is None, "the coarsest level has a reach"

        widest = 1.0 / cache.mip_scale(level - 1)
        zoom_preframe.zoom_about(view, widest * 1.001 / view.view_device_scale(), anchor)
        assert view.viewport_chunk_target()[0] == level, "the zoom left the level"
        if edge:
            hbar, vbar = view.horizontalScrollBar(), view.verticalScrollBar()
            assert hbar.value() == hbar.maximum() and vbar.value() == vbar.maximum(), "no clamp -- vacuous"
        shown = item.view_chunks(level, view._current_scene_rect(), view.view_device_scale())
        x0, y0, x1, y1 = reach
        sx0, sy0, sx1, sy1 = shown
        assert x0 <= sx0 and y0 <= sy0 and sx1 <= x1 and sy1 <= y1, f"view {shown} outside the reach {reach}"
        assert sx0 - x0 <= 1 and sy0 - y0 <= 1 and x1 - sx1 <= 1 and y1 - sy1 <= 1, f"reach {reach} loose on {shown}"
        grown = (x1 - x0 + 1) * (y1 - y0 + 1)
        assert grown > (start[3] - start[1] + 1) * (start[4] - start[2] + 1), "the reach is no wider than the view"
    finally:
        conftest.close_window(window)
