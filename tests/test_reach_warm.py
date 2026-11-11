"""The zoom-out reach warm's wiring in the viewer (TASK-031.11, TASK-031.10).

A level's reach is the chunks the widest view at it reads, zoomed out about
the wheel anchor (D1). Its two entries, the settled level's and the coarser
neighbour's (D2), run on the load warmer below every load entry and only once
every other warm is idle (D3), sized to the chunk LRU (D5), re-armed only when
their ranges change (D6) and never recomputed on a drain (D7). Every window
here opts in (D9); the rest of the suite keeps today's warms. The roll that
shows the payoff, no chunk composited in a zoom-out notch's paint, is in
tests/test_placeholder_paint.py beside T3i.
"""

from __future__ import annotations

import pytest
from test_native_composite import native_backend, native_kernel  # noqa: F401 -- fixtures
from test_placeholder_paint import _drain, _grab, _pump, _set_device_scale, _wheel_at, _window

from descape import composite_backend, level_warm, margin_warm

import conftest

SETTLE_SCALE = 0.99


def _settled(*, anchor: str = "top-left", at=None, reach: bool = True, placeholders: bool = False,
             scale: float = SETTLE_SCALE):
    """A Stepped window at level 0, residual 0.99 (or `scale`), the reach armed by a
    re-anchor and nothing drained since. `anchor` is the wheel's viewport
    point (top-left, bottom-right or centre); `at` the map fraction the view
    centres on, else the map's right edge just inside it, so the reach has
    work the view's ring does not cover. Placeholders off by default, so the
    paint builds level 0 and no pending level defers the arm; on, the arm
    waits for refine."""
    from PyQt5.QtCore import QPoint, QPointF

    window = _window(placeholders=placeholders, size=1360, height=1030)
    window.set_reach_warm_allowed(reach)
    view = window.map_view
    view.set_zoom_anchor_mode(True)
    vp = view.viewport().rect()
    points = {"top-left": QPoint(8, 8), "bottom-right": QPoint(vp.width() - 8, vp.height() - 8), "centre": vp.center()}
    _wheel_at(view, points[anchor])
    _set_device_scale(window, scale)
    b = view._canvas_item.boundingRect()
    w = vp.width() / scale
    if at is None:
        view.centerOn(QPointF(b.right() + 0.3 * w, b.center().y()))
    else:
        view.centerOn(QPointF(b.left() + at[0] * b.width(), b.top() + at[1] * b.height()))
    _grab(window)
    view.stop_viewport_poll()
    window._cancel_warms()
    window._on_viewport_changed()
    level = window._cache.mip_for_scale(scale)
    assert view.viewport_chunk_target()[0] == level
    assert (window._pending_level() == level) == placeholders
    return window


def _running(window):
    """Drains until a reach entry runs; returns it."""
    _drain(window, until=lambda: window._reach_running is not None)
    return window._reach_running


def _evict(cache, mip: int, chunks) -> None:
    """Drops `chunks` (cx, cy) at `mip` from the LRU, as _evict_rects() does."""
    for cx, cy in chunks:
        chunk = cache._cache.pop((mip, cx, cy), None)
        if chunk is not None:
            cache._cache_bytes -= chunk.nbytes


def _range(rng):
    cx0, cy0, cx1, cy1 = rng
    return [(cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)]


def _near(window) -> set:
    """The view's chunks and their depth-1 ring at its level: the paint's and the margin warmer's."""
    _mip, cx0, cy0, cx1, cy1 = window.map_view.viewport_chunk_target()
    return {(cx, cy) for cx in range(cx0 - 1, cx1 + 2) for cy in range(cy0 - 1, cy1 + 2)}


def _spy_reach_starts(window, monkeypatch) -> list:
    """(entry, level active, margin active, load entries queued, pending) per reach start."""
    starts = []
    real = window._load_warmer.start

    def spy(cache, mip, chunks, *, on_drained=None, **kwargs):
        if on_drained == window._on_reach_drained:
            starts.append((
                window._reach_running, window._level_warmer.is_active, window._margin_warmer.is_active,
                len(window._load_warm_queue), window._pending_level(),
            ))
        return real(cache, mip, chunks, on_drained=on_drained, **kwargs)

    monkeypatch.setattr(window._load_warmer, "start", spy)
    return starts


def _spy_reach_chunks(window, monkeypatch) -> list:
    calls = []
    real = window._reach_chunks

    def spy(mip, rng):
        calls.append((mip, rng))
        return real(mip, rng)

    monkeypatch.setattr(window, "_reach_chunks", spy)
    return calls


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_the_reach_starts_only_once_every_other_warm_is_idle(monkeypatch) -> None:
    """D3, along a real settle: each reach start finds no load entry queued
    or running, no level job, no margin ring and no pending level. Then each
    blocker set by hand holds a re-armed reach off until it clears, the level
    warmer's through its on_drained (D8). A drained entry retries the refused
    zoom frames (D10)."""
    window = _settled()
    try:
        starts = _spy_reach_starts(window, monkeypatch)
        retries = []
        real_retry = window.map_view.retry_refused_preframes
        monkeypatch.setattr(window.map_view, "retry_refused_preframes", lambda: (retries.append(1), real_retry()))
        assert window._reach_queue, "the re-anchor armed no reach -- vacuous"
        # D8: a last level job that notifies nobody (a pack derive) still pumps the chain.
        assert window._level_warmer.is_active, "the re-anchor queued no level job -- vacuous"
        assert window._level_warmer._on_drained == window._on_level_drained
        _drain(window)
        assert starts, "no reach entry ever started"
        assert all(not (level or margin or queued or pending is not None) for _e, level, margin, queued, pending in starts)
        assert not window._reach_queue and window._reach_running is None

        cache, view = window._cache, window.map_view
        settled = window._reach_keys[0]
        near = _near(window)
        work = [c for c in _range(settled[1]) if c not in near]
        assert work, "the settled reach reaches no further than the ring -- vacuous"
        other = (0, 0)
        assert other not in work and other not in near
        target = view.viewport_chunk_target()

        # A margin ring holds it off; its drain starts it.
        _evict(cache, 0, work + [c for c in near if not (target[1] <= c[0] <= target[3] and target[2] <= c[1] <= target[4])])
        ring = margin_warm.ring_chunks(cache, 0, *target[1:])
        assert ring, "the ring is cached -- vacuous"
        window._margin_warmer.start(cache, 0, ring, on_drained=window._on_margin_drained)
        window._queue_reach()
        assert window._reach_running is None and window._reach_queue
        window._margin_warmer.run_to_completion()
        assert window._reach_running == settled
        _drain(window)

        # A load entry queued holds it off; its drain starts it.
        _evict(cache, 0, [*work, other])
        window._load_warm_queue.append((0, [other]))
        window._queue_reach()
        assert window._reach_running is None and window._load_warmer.mip == 0
        window._load_warmer.run_to_completion()
        assert window._reach_running == settled
        _drain(window)

        # A level job holds it off; the level warmer's on_drained starts it.
        _evict(cache, 0, work)
        busy = {"on": True}
        real_active = level_warm.LevelWarmer.is_active
        monkeypatch.setattr(level_warm.LevelWarmer, "is_active", property(lambda w: busy["on"] or real_active.fget(w)))
        window._queue_reach()
        assert window._reach_running is None
        busy["on"] = False
        window._on_level_drained()
        assert window._reach_running == settled
        before = len(retries)
        _drain(window)
        assert len(retries) > before, "a drained reach entry did not retry the refused zoom frames (D10)"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_a_load_entry_queued_mid_reach_preempts_it_and_the_reach_resumes_recomputed(monkeypatch) -> None:
    """D3: a neighbour's load entry queued while a reach entry runs cancels
    it and starts at once; the reach entry goes back to the head of
    _reach_queue and, once the load chain is empty, starts again from a
    recomputed queue (its old one is not trusted)."""
    window = _settled()
    try:
        entry = _running(window)
        calls = _spy_reach_chunks(window, monkeypatch)
        cache, view = window._cache, window.map_view
        target = view.viewport_chunk_target()
        coarser = target[0] - 1
        rng = window._neighbour_warm_range(coarser, view.viewport_chunk_target_at(coarser), target)
        _evict(cache, coarser, _range(rng))
        patch = margin_warm.load_warm_chunks(cache, coarser, *rng)
        assert patch
        window._queue_load_warm(coarser)
        assert window._reach_running is None, "the load entry did not preempt the running reach entry"
        assert window._load_warmer.is_active and window._load_warmer.mip == coarser
        assert window._reach_queue[0] == entry
        assert not calls
        assert _running(window) == entry
        assert calls == [entry], "the resumed entry was not recomputed"
        _drain(window)
        assert all(cache.has_chunk(coarser, *c) for c in patch)
        assert window._reach_chunks(*entry) == []
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_a_re_anchor_cancels_a_running_reach_and_cancel_warms_clears_it(monkeypatch) -> None:
    """D3: _start_level_warm() cancels a running reach entry and re-arms the
    reach from scratch; a re-anchor onto another level re-arms it there.
    _cancel_warms() drops all of it."""
    window = _settled()
    try:
        entry = _running(window)
        cancels = []
        real_cancel = window._load_warmer.cancel
        monkeypatch.setattr(window._load_warmer, "cancel", lambda: (cancels.append(window._reach_running), real_cancel()))
        calls = _spy_reach_chunks(window, monkeypatch)
        keys = window._reach_keys
        window._start_level_warm()
        assert entry in cancels, "the re-anchor left the reach entry running"
        assert window._reach_keys == keys and calls, "the re-anchor did not re-arm the reach"

        view = window.map_view
        centre = view.mapToScene(view.viewport().rect().center())
        _set_device_scale(window, 0.45)
        view.centerOn(centre)
        view.stop_viewport_poll()
        window._on_viewport_changed()
        assert window.map_view.viewport_chunk_target()[0] == -1
        assert window._reach_running is None or window._reach_running[0] == -1
        assert all(mip == -1 for mip, _rng in window._reach_keys + tuple(window._reach_queue))

        window._cancel_warms()
        assert window._reach_queue == [] and window._reach_running is None and window._reach_keys == ()
        assert not window._load_warmer.is_active
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_an_unchanged_fire_keeps_the_reach_and_a_pan_rearms_it_once_the_view_stops() -> None:
    """D6: a poll fire that recomputes the same (mip, range) keys restarts
    nothing. A pan changes them: the reach is re-armed, but its new ring
    holds it off on every fire while the pan goes on (D3), and it starts
    once the ring drains."""
    window = _settled()
    try:
        view = window.map_view
        entry = _running(window)
        token = window._load_warmer._token
        keys = window._reach_keys
        window._on_viewport_changed()
        assert window._reach_running == entry and window._load_warmer._token == token, "an unchanged fire restarted it"
        assert window._reach_keys == keys

        hbar = view.horizontalScrollBar()
        step = 3 * window._cache.chunk_px
        for _ in range(2):
            hbar.setValue(hbar.value() - step)
            view.stop_viewport_poll()
            window._on_viewport_changed()
            assert window._margin_warmer.is_active, "the pan's ring is cached -- vacuous"
            assert window._reach_running is None, "the reach ticked beside a moving pan's ring"
        assert window._reach_keys != keys, "the pan left the reach's keys unchanged"
        moved = window._reach_keys
        assert _running(window) in moved
        _drain(window)
        assert window._reach_keys == moved
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_a_declined_chunk_does_not_loop_the_reach(monkeypatch) -> None:
    """D7: a chunk the warm drops stays uncached, so a drained entry is not
    recomputed until the next arm, or the chain would restart it forever."""
    window = _settled()
    try:
        cache = window._cache
        settled = window._reach_queue[0]
        planned = window._reach_chunks(*settled)
        assert planned, "the settled reach has nothing to warm -- vacuous"
        declined = planned[-1]
        real_get, real_prepare = cache.get_chunk, cache.prepare_chunk_job

        def get_chunk(mip, cx, cy):
            if (mip, cx, cy) == (settled[0], *declined):
                raise RuntimeError("declined")
            return real_get(mip, cx, cy)

        def prepare_chunk_job(mip, cx, cy, heights=None):
            if (mip, cx, cy) == (settled[0], *declined):
                return None
            return real_prepare(mip, cx, cy, heights)

        monkeypatch.setattr(cache, "get_chunk", get_chunk)
        monkeypatch.setattr(cache, "prepare_chunk_job", prepare_chunk_job)
        starts = _spy_reach_starts(window, monkeypatch)
        _drain(window)
        assert not cache.has_chunk(settled[0], *declined)
        assert [s[0] for s in starts].count(settled) == 1, f"the settled entry started {starts}"
        assert not window._reach_queue and window._reach_running is None
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_the_budget_drops_the_coarser_reach_first_then_the_settled_one(monkeypatch) -> None:
    """D5: each entry is taken whole while the total fits the LRU: the
    settled view or its reach, the finer neighbour's chosen range, the
    coarser neighbour's patch or its reach, and the coarsest level's
    resident chunks."""
    window = _settled(anchor="centre", at=(0.5, 0.5))
    try:
        view, cache = window.map_view, window._cache
        target = view.viewport_chunk_target()
        mip = target[0]
        levels = cache.mip_levels()
        settled_rng, coarser_rng = view.viewport_reach_target(mip), view.viewport_reach_target(mip - 1)
        finer = window._patch_footprint(
            mip + 1, *window._neighbour_warm_range(mip + 1, view.viewport_chunk_target_at(mip + 1), target)
        )
        coarsest = cache.level_chunk_count(levels[0])
        assert coarsest > 0, "no coarsest-level chunk is resident -- vacuous"
        settled_fp = window._patch_footprint(mip, *settled_rng)
        need_settled = settled_fp + finer + window._coarser_patch_footprint(mip) + coarsest
        need_both = settled_fp + finer + window._patch_footprint(mip - 1, *coarser_rng) + coarsest
        assert need_both > need_settled, "the coarser reach is no bigger than its patch -- vacuous"
        both = [(mip, settled_rng), (mip - 1, coarser_rng)]
        for budget, want in ((need_both, both), (need_both - 1, both[:1]), (need_settled - 1, [])):
            monkeypatch.setattr(window, "_chunk_budget", lambda b=budget: b)
            assert window._reach_entries() == want, f"budget {budget}"
        monkeypatch.setattr(window, "_chunk_budget", lambda: need_both)
        real_count = cache.level_chunk_count
        monkeypatch.setattr(cache, "level_chunk_count", lambda m, outside=None: real_count(m, outside) + (m == levels[0]))
        assert window._reach_entries() == both[:1], "the coarsest level's resident chunks were not counted"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_a_re_anchor_that_queues_nothing_else_still_starts_the_reach() -> None:
    """With every level and neighbour warm current, a re-anchor queues no
    level job and no load entry, so nothing would pump: _queue_reach() does."""
    window = _settled()
    try:
        _drain(window)
        mip, rng = window._reach_keys[0]
        near = _near(window)
        _evict(window._cache, mip, [c for c in _range(rng) if c not in near])
        window._start_level_warm()
        assert not window._level_warmer.is_active and not window._load_warm_queue, "the re-anchor queued work -- vacuous"
        assert window._reach_running == (mip, rng), "nothing started the reach"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_the_coarser_reach_caches_what_its_crossing_patch_leaves_cold() -> None:
    """D2: after a settle, every chunk of the coarser neighbour's reach is
    cached; with the reach off, its crossing patch alone leaves some cold
    (the anchor at the view's bottom-right, which on this small map is
    where the coarser reach outgrows the patch)."""
    cached = {}
    for reach in (True, False):
        window = _settled(anchor="bottom-right", at=(0.5, 0.5), reach=reach)
        try:
            _drain(window)
            view, cache = window.map_view, window._cache
            mip = view.viewport_chunk_target()[0] - 1
            rng = view.viewport_reach_target(mip)
            cached[reach] = [cache.has_chunk(mip, *c) for c in _range(rng)]
        finally:
            conftest.close_window(window)
    assert all(cached[True]), "the coarser reach left chunks cold"
    assert not all(cached[False]), "the crossing patch already covers the coarser reach -- vacuous"


@pytest.mark.gui
def test_no_reach_without_a_worker_path() -> None:
    """D1: top-down Flat and the numpy backend composite every chunk on the
    GUI thread, so they queue no reach, opted in or not."""
    from PyQt5.QtWidgets import QApplication

    window = _window(size=1360, height=1030)
    try:
        window.set_reach_warm_allowed(True)
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        QApplication.processEvents()
        _set_device_scale(window, SETTLE_SCALE)
        window.map_view.stop_viewport_poll()
        assert not window._cache.pools_chunks()
        window._cancel_warms()
        window._on_viewport_changed()
        assert window._reach_keys == () and not window._reach_queue
    finally:
        conftest.close_window(window)
    with composite_backend.use_backend("numpy"):
        window = _settled()
        try:
            assert not window._cache.pools_chunks()
            assert window._reach_keys == () and not window._reach_queue
            _drain(window)
            assert window._reach_keys == ()
        finally:
            conftest.close_window(window)


def _view_range(window):
    _mip, cx0, cy0, cx1, cy1 = window.map_view.viewport_chunk_target()
    return cx0, cy0, cx1, cy1


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_a_reach_preempted_by_a_ring_resumes_on_a_fire_whose_ring_is_cached() -> None:
    """Review fold: one fire's ring has uncached chunks and preempts the
    reach; the next (the view nudged back) finds its ring cached, so the
    margin warmer starts nothing and the old ring's drain callback never
    comes, and the same D6 keys re-arm nothing. That fire resumes the reach."""
    window = _settled()
    try:
        cache = window._cache
        entry = _running(window)
        cx0, cy0, cx1, cy1 = _view_range(window)
        ring = [c for c in _near(window) if not (cx0 <= c[0] <= cx1 and cy0 <= c[1] <= cy1) and cache.has_chunk(0, *c)]
        assert ring, "the view has no cached ring -- vacuous"
        _evict(cache, 0, ring)
        keys = window._reach_keys
        window._on_viewport_changed()
        assert window._margin_warmer.is_active and window._reach_running is None and window._reach_queue[0] == entry
        for c in ring:
            cache.get_chunk(0, *c)
        window._on_viewport_changed()
        assert not window._margin_warmer.is_active and window._reach_keys == keys
        assert window._reach_running == entry, "the reach stayed parked with every warmer idle"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_the_reach_never_evicts_the_coarsest_levels_chunks() -> None:
    """Review fold (D5): after a roll in, the coarsest level's chunks are the
    LRU's oldest, and the fit a roll back out reads. With the LRU full of
    another level's newer chunks, a reach entry's inserts must not evict
    them: it refreshes their recency before it starts."""
    window = _settled()
    try:
        _drain(window)
        cache = window._cache
        coarsest = cache.mip_levels()[0]
        resident = [key for key in cache._cache if key[0] == coarsest]
        assert resident, "no coarsest-level chunk is resident -- vacuous"
        mip, rng = window._reach_keys[0]
        near = _near(window)
        work = [c for c in _range(rng) if c not in near and cache.has_chunk(mip, *c)]
        assert work, "the settled reach has nothing past the ring -- vacuous"
        _evict(cache, mip, work)
        full = cache.chunk_px * cache.chunk_px * 3
        assert cache.max_bytes is not None
        w, h = cache.canvas_dims(1)
        for cy in range(-(-h // cache.chunk_px)):
            for cx in range(-(-w // cache.chunk_px)):
                if cache._cache_bytes + full <= cache.max_bytes and not cache.has_chunk(1, cx, cy):
                    cache.get_chunk(1, cx, cy)
        assert cache._cache_bytes + full > cache.max_bytes, "level 1 never filled the LRU -- vacuous"
        assert all(key in cache._cache for key in resident), "the fill itself evicted the coarsest level"
        window._queue_reach()
        _drain(window)
        assert all(cache.has_chunk(mip, *c) for c in work), "the reach never ran"
        lost = [key for key in resident if key not in cache._cache]
        assert not lost, f"the reach evicted {len(lost)} of the coarsest level's {len(resident)} chunks"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_a_crossing_onto_a_cold_level_drops_the_reach_until_refine(monkeypatch) -> None:
    """Review fold (R4's guard), placeholders on: a zoom onto level 1 at a
    view none of its chunks cover draws a placeholder (level 2, never built,
    would build in the paint on this unit-less fixture); the re-anchor on it
    drops the running reach, the pending level's view entry is the load
    warmer's first start, no reach entry starts while it is pending, and
    after refine the reach is armed on the new level and runs."""
    from PyQt5.QtCore import QPointF

    window = _settled(placeholders=True)
    try:
        view, cache = window.map_view, window._cache
        item = view._canvas_item
        _running(window)
        assert item.pending_mip() is None
        assert cache.is_level_resident(1)
        starts = []
        real = window._load_warmer.start

        def spy(c, mip, chunks, *, on_drained=None, **kwargs):
            starts.append((mip, on_drained == window._on_reach_drained, window._pending_level()))
            return real(c, mip, chunks, on_drained=on_drained, **kwargs)

        monkeypatch.setattr(window._load_warmer, "start", spy)
        _set_device_scale(window, 1.48)  # level 1 near its top, so its reach grows past the ring
        b = item.boundingRect()
        view.centerOn(QPointF(b.left() + 0.25 * b.width(), b.center().y()))
        view.stop_viewport_poll()
        _grab(window)
        assert item.pending_mip() == 1, "the crossing drew no placeholder -- vacuous"
        _pump()  # the 0 ms re-anchor on the pending level
        assert window._reach_running is None and not window._reach_queue, "the reach outlived the re-anchor"
        _drain(window, until=lambda: item.pending_mip() is None)
        assert starts and starts[0][:2] == (1, False), f"the pending view entry did not start first: {starts}"
        assert not any(reach for _m, reach, _p in starts), "a reach entry started while level 1 was pending"
        # The coarser entry may be budget-dropped here (D5); the settled one is level 1's.
        assert window._reach_keys[:1] and window._reach_keys[0][0] == 1, "the refine did not re-arm the reach on level 1"
        assert _running(window) in window._reach_keys, "the reach did not resume after refine"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_a_pending_view_entry_preempts_the_reach_and_the_pending_level_holds_it_off(monkeypatch) -> None:
    """Review fold: the pending path (_queue_pending_load_warm()) preempts a
    running reach entry back to the head of _reach_queue rather than
    cancelling the load warmer under it, which would leave _reach_running
    set with nothing running and the chain shut. While a level is pending no
    reach starts (D3's fifth gate); once it settles the reach resumes."""
    window = _settled()
    try:
        cache = window._cache
        entry = _running(window)
        assert entry[0] == 0
        pending = {"on": True}
        monkeypatch.setattr(window, "_pending_level", lambda: 0 if pending["on"] else None)
        _evict(cache, 0, _range(_view_range(window)))
        window._queue_load_warm(0)
        assert window._reach_running is None, "the pending view entry left the reach entry marked running"
        assert window._reach_queue[0] == entry
        assert window._load_warmer.is_active and window._load_warmer._on_drained == window._pump_load_warm
        _drain(window)
        assert window._reach_running is None and window._reach_queue[:1] == [entry], "a reach started while pending"
        pending["on"] = False
        window._pump_load_warm()
        assert window._reach_running == entry
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_the_pending_levels_load_entry_defers_a_cold_chunk_during_input(monkeypatch) -> None:
    """TASK-031.82 D3 (user, 2026-11-10): the pending level's load entry defers
    like any other while input is live, so a roll click never waits behind
    its cold first chunk; it runs once input goes quiet."""
    from descape import level_warm

    window = _settled(reach=False)
    try:
        assert isinstance(window._load_warmer, margin_warm.MarginWarmer)
        monkeypatch.setattr(window, "_pending_level", lambda: 0)
        monkeypatch.setattr(window, "_maybe_refine", lambda: None)
        monkeypatch.setattr(window._cache, "level_warmed", lambda mip: False)
        window._load_warmer.cancel()
        window._load_warm_queue = [(0, [(0, 0), (1, 0)])]
        window._pump_load_warm()
        warmer = window._load_warmer
        assert warmer.is_active and warmer.mip == 0, "vacuous: the pending entry did not start"
        queued = list(warmer._queue)
        before = margin_warm.deferrals
        monkeypatch.setattr(level_warm, "_last_input", level_warm._now())
        assert warmer.tick() is True
        assert (margin_warm.deferrals, warmer._queue, warmer._in_flight) == (before + 1, queued, [])
        monkeypatch.setattr(level_warm, "_last_input", None)
        warmer.tick()
        assert margin_warm.deferrals == before + 1
        assert warmer._queue != queued or warmer._in_flight, "the quiet tick warmed nothing"
        warmer.cancel()
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.usefixtures("native_backend")
def test_a_coarser_patch_at_the_coarsest_level_is_counted_once(monkeypatch) -> None:
    """Review fold (D5): for a view one level above the coarsest, the coarser
    neighbour's crossing patch is coarsest-level chunks too; those already
    resident are not counted again among the coarsest level's."""
    window = _settled(anchor="centre", at=(0.5, 0.5), scale=0.495)
    try:
        _drain(window)
        view, cache = window.map_view, window._cache
        mip = view.viewport_chunk_target()[0]
        levels = cache.mip_levels()
        assert mip - 1 == levels[0]
        patch = window._footprint_rect(mip - 1, *window._coarser_patch_range(mip))
        outside = cache.level_chunk_count(levels[0], outside=patch)
        assert outside < cache.level_chunk_count(levels[0]), "no resident chunk under the patch -- vacuous"
        target = view.viewport_chunk_target()
        rng = view.viewport_reach_target(mip)
        finer = window._patch_footprint(
            mip + 1, *window._neighbour_warm_range(mip + 1, view.viewport_chunk_target_at(mip + 1), target)
        )
        need = window._patch_footprint(mip, *rng) + finer + window._coarser_patch_footprint(mip) + outside
        monkeypatch.setattr(window, "_chunk_budget", lambda: need)
        assert window._reach_entries() == [(mip, rng)]
        monkeypatch.setattr(window, "_chunk_budget", lambda: need - 1)
        assert window._reach_entries() == []
    finally:
        conftest.close_window(window)
