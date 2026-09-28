"""Idle-time warm of a bounded chunk ring just outside the current viewport,
continuously re-targeted as the user pans/zooms (maintainer plan
2026-09-07's Phase A, Step A3). Sibling to level_warm.py -- both share the
0ms-QTimer pacing mechanism (level_warm._IdleTimerDriver) -- rather than an
extension of it: the unit of work differs (an atomic get_chunk() call here,
a sliced generator step there), and so does the budget rule below.

**A budgeted multi-chunk tick, with a predictive stop rather than
LevelWarmer's check-after overrun.** Through v0.8 a tick warmed exactly one
chunk with no budget, because a chunk cost 39.5-117ms (Stepped) and no
budget could honestly bound that. The compiled whole-chunk composite brought
a chunk down to ~2-5ms p90 on the dev machine, so a tick now runs chunks
until BUDGET_MS is spent. get_chunk() is still atomic, so the stop rule is
predictive: the first chunk always runs, and each further one runs only if
the elapsed time plus the slowest chunk seen this tick still fits. With even
chunk costs a tick stays inside max(BUDGET_MS, one chunk): several chunks on
a fast machine, exactly one where a chunk alone fills the budget. That is a
common-case bound, not a guarantee: a chunk slower than every earlier one in
the same tick overruns by the difference (cold Stepped mip 0 on June: tick
max 10.3ms against a 9.3ms chunk max). A single chunk slower than
the budget is the case worker threads (maintainer plan Batch F T2) would
remove; Perf Trace's `warm:` bucket reports it as `max chunk`.

**Worker threads (maintainer plan Batch F T2).** Where the cache offers a
ChunkJob (render_cache.prepare_chunk_job: native backend, Stepped or Sloped),
a tick does only the Python half of each chunk on the GUI thread and hands
the nogil kernel to a small pool. The budget above then bounds that Python
half, and "max chunk" in Perf Trace is the GUI-side cost; worker kernel time
is reported separately. Results come back as a queued signal, so they install
on the GUI thread, possibly inside one of viewer.py's nested event loops:
install_chunk() rejects any job a mutation has overtaken, and cancel() drops
in-flight jobs without waiting for them. Everything else (numpy, Flat, a
chunk the kernel can't take, no QApplication, run_to_completion) still runs
get_chunk() synchronously, exactly as before.

render_cache._ChunkCacheBase.is_level_resident()/has_chunk() are this
module's other half of the correctness argument -- see MarginWarmer.start()
and ring_chunks() below for how each is used."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

from descape import debug_log, perf_trace
from descape.level_warm import _IdleTimerDriver

# Wall-clock budget per tick: the 8ms warm-tick budget Batch F's gates use.
# Separate from level_warm.BUDGET_MS, whose overrun rule differs.
BUDGET_MS = 8

# The tick's clock, a module attribute so tests can drive it deterministically.
_now = time.perf_counter

# Kernel worker threads, shared by every MarginWarmer; 0 keeps every chunk on
# the GUI thread. Also each warmer's cap on jobs in flight, since a prepared
# job is GUI time a retarget throws away.
WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))

# The pool, created on first use; tests swap in an inline executor.
_executor = None
_bridge = None


def _pool():
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="margin-warm")
    return _executor


def _result_bridge():
    """One process-wide QObject on the GUI thread whose queued signal carries
    worker results back. Per-process rather than per-warmer, so a closed
    window's warmer can't leave a worker emitting on a deleted QObject."""
    global _bridge
    if _bridge is None:
        from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot

        class _Bridge(QObject):
            done = pyqtSignal(object)

            def __init__(self) -> None:
                super().__init__()
                # Queued even for an emit on this thread (the tests' inline executor).
                self.done.connect(self._deliver, Qt.QueuedConnection)

            @pyqtSlot(object)
            def _deliver(self, payload) -> None:
                payload[0](*payload[1:])

        _bridge = _Bridge()
    return _bridge


def _run_job(deliver, token: int, job, bridge) -> None:
    """Worker-thread side: the kernel only, then post the result back."""
    start = time.perf_counter()
    try:
        outcome = job.run()
    except Exception as exc:  # noqa: BLE001
        outcome = exc
    bridge.done.emit((deliver, token, job, outcome, (time.perf_counter() - start) * 1000))


def _grid_max_chunk_index(cache, mip: int) -> tuple[int, int]:
    """(max_cx, max_cy) -- the last valid chunk index in each axis of
    `mip`'s own grid. Shared by ring_chunks() and bounded_chunk_range()
    rather than each computing it separately."""
    grid_w, grid_h = cache.canvas_dims(mip)
    chunk_px = cache.chunk_px
    return -(-grid_w // chunk_px) - 1, -(-grid_h // chunk_px) - 1


def bounded_chunk_range(
    cx0: int,
    cy0: int,
    cx1: int,
    cy1: int,
    span_w: int,
    span_h: int,
) -> tuple[int, int, int, int]:
    """Centers a `span_w` x `span_h` window inside [cx0, cx1] x [cy0, cy1]
    -- the bound the load-time margin warm applies to viewport_chunk_
    target_at()'s raw projection before it ever reaches load_warm_chunks()
    (2026-09-07 plan's load-time margin warm).

    **Why the raw projection needs bounding at all, when viewport_chunk_
    target() itself never does.** viewport_chunk_target_at() reuses the
    CURRENT on-screen scene rect verbatim (by design -- see its own
    docstring); load_scenario() always calls _start_level_warm() right
    after a fit-to-view render, so that scene rect is the WHOLE map, not a
    normal viewport's worth. Projected onto a neighbour mip finer than the
    opening one, the raw range covers that mip's entire grid -- exactly
    the "frontload a whole finer level's chunk grid" cost the parent
    frontload plan measured and rejected (5.9-97.3s), not the ~20-chunk
    patch the Design section's own sizing arithmetic assumes. `span_w`/
    `span_h` (MapView.viewport_chunk_span()) are sized from the viewport's
    own on-screen dimensions, independent of whatever scale happens to be
    selected right now, which is what keeps this patch viewport-sized
    regardless of whether the caller is at a real zoom or fit-to-view.

    Never wider/taller than the input range itself, and never outside it
    either -- clamped to [cx0, cx1]/[cy0, cy1] directly rather than to the
    level's own grid bounds: the input range is already grid-valid (it
    came from viewport_chunk_target_at(), itself clamped to canvas_dims()),
    so anchoring the clamp there is both sufficient and exact -- a
    midpoint-based centre can otherwise round outside an EVEN-width input
    range (e.g. [5, 6]'s midpoint floors to 5, and a span of 2 centred
    there would reach down to 4, one chunk below cy0) if it weren't
    clamped back against the input range's own edges afterward."""
    span_w = min(span_w, cx1 - cx0 + 1)
    span_h = min(span_h, cy1 - cy0 + 1)
    mid_x, mid_y = (cx0 + cx1) // 2, (cy0 + cy1) // 2

    nx0 = max(cx0, min(mid_x - span_w // 2, cx1 - span_w + 1))
    ny0 = max(cy0, min(mid_y - span_h // 2, cy1 - span_h + 1))
    return nx0, ny0, nx0 + span_w - 1, ny0 + span_h - 1


def ring_chunks(
    cache,
    mip: int,
    cx0: int,
    cy0: int,
    cx1: int,
    cy1: int,
    *,
    lead: tuple[int, int] = (0, 0),
    depth: int = 1,
) -> list[tuple[int, int]]:
    """The depth-`depth` ring of chunk indices just outside the inclusive
    viewport chunk range [cx0, cx1] x [cy0, cy1] at `mip`, clipped to that
    level's own grid dimensions and with already-resident chunks dropped
    (cache.has_chunk()).

    **Depth 1 makes "nearest-to-viewport-edge first" (the parent plan's
    Step A5) automatic rather than something this function has to compute
    separately**: every depth-1 ring chunk is edge-adjacent to the viewport
    by construction, so the only ordering question left is DIRECTION, not
    distance -- which is why the ordering below is a sort key, not a
    distance walk. A depth-2+ ring would need that distance walk; this one
    doesn't, and Step A5 fixed depth at 1 for unrelated (ring-cost) reasons
    anyway.

    Ordering: chunks on the LEADING edge(s) -- the side(s) the pan is
    heading toward, from the sign of `lead` -- sort first, then the
    remaining (trailing) edges, then the four corners last (a corner is
    diagonally adjacent to the viewport, reachable only after either
    bordering edge chunk is, so it is the least urgent of the three tiers).
    `lead=(0, 0)` (a zoom, or the very first fire) degenerates to plain
    edge-then-corner order with no side favoured -- correct behaviour for
    that case, not a special case bolted on.

    Empty for a fit-to-view viewport (covers the whole canvas already, so
    every ring index clips off-grid) with no explicit "skip the fit mip"
    branch needed: the parent plan's premise that the opening level has
    nothing left to preload falls out of this geometry rather than being
    asserted separately."""
    max_cx, max_cy = _grid_max_chunk_index(cache, mip)
    lead_x, lead_y = lead

    ranked: list[tuple[int, int, int]] = []
    for cx in range(cx0 - depth, cx1 + depth + 1):
        for cy in range(cy0 - depth, cy1 + depth + 1):
            if cx0 <= cx <= cx1 and cy0 <= cy <= cy1:
                continue  # inside the viewport itself, not a margin chunk
            if not (0 <= cx <= max_cx and 0 <= cy <= max_cy):
                continue  # off this level's grid
            on_x_edge = cx < cx0 or cx > cx1
            on_y_edge = cy < cy0 or cy > cy1
            if on_x_edge and on_y_edge:
                priority = 2  # corner
            else:
                leading = (
                    (cx < cx0 and lead_x < 0)
                    or (cx > cx1 and lead_x > 0)
                    or (cy < cy0 and lead_y < 0)
                    or (cy > cy1 and lead_y > 0)
                )
                priority = 0 if leading else 1
            ranked.append((priority, cx, cy))

    ranked.sort(key=lambda entry: entry[0])
    return [(cx, cy) for _priority, cx, cy in ranked if not cache.has_chunk(mip, cx, cy)]


def load_warm_chunks(
    cache,
    mip: int,
    cx0: int,
    cy0: int,
    cx1: int,
    cy1: int,
) -> list[tuple[int, int]]:
    """The queue for the load-time margin warm (2026-09-07 plan's load-time
    margin warm, Design section): the viewport's own chunk range at `mip`
    -- centre chunks, not just its margin -- unioned with one depth-1
    ring_chunks() ring around it.

    Unlike a real pan, nothing has painted `mip` yet, so its centre chunks
    are exactly as cold as its margin (ring_chunks() itself excludes the
    viewport's own chunks precisely because a real pan's paint already
    built those). `lead=(0, 0)` for the ring half -- no direction to favour
    for a level nobody has navigated to yet, the same case ring_chunks'
    own docstring says that argument already covers correctly.

    Centre chunks first (already-resident ones dropped, same as
    ring_chunks), then the ring -- both cheap wins over "no queue at
    all," ordered by their claim to be the more useful of the two rather
    than by any measured cost difference."""
    centre = [
        (cx, cy)
        for cx in range(cx0, cx1 + 1)
        for cy in range(cy0, cy1 + 1)
        if not cache.has_chunk(mip, cx, cy)
    ]
    ring = ring_chunks(cache, mip, cx0, cy0, cx1, cy1, lead=(0, 0))
    return centre + ring


class MarginWarmer(_IdleTimerDriver):
    """Drives one cache's margin-ring warm, a BUDGET_MS worth of get_chunk()
    calls per idle tick -- see this module's own docstring for the
    predictive stop rule that keeps a tick inside max(BUDGET_MS, one chunk).

    One instance per window, reused across documents and retargeted on
    every viewport-changed poll fire (ViewerWindow._on_viewport_changed) --
    start() replaces whatever was queued outright, which IS the parent
    plan's A4 retarget mechanism; a second mechanism on top of that would be
    new machinery this doesn't need.

    **Paused while any mouse button is held (item 23, decided 2026-09-10).**
    A mouse pan's move events are discrete, so the 0ms idle timer still
    fires in the gaps between them -- landing a tick mid-drag as a hitch
    (22-40ms per chunk when this was decided, up to BUDGET_MS a tick now). tick() gates on QApplication.mouseButtons() and backs the timer
    off to HELD_INTERVAL_MS while held, restoring 0ms on release; warming
    DURING a drag was considered and rejected (a fast drag already outruns
    a finite margin per the parent plan's own Step A5 analysis, and middle
    drag has no speed cap). Wheel and keyboard pans hold no button and are
    unaffected. Applies equally to both MarginWarmer instances (the
    navigation ring and the load-time _load_warmer) since it lives here
    rather than in each call site."""

    HELD_INTERVAL_MS = 50

    def __init__(self) -> None:
        super().__init__()
        self._cache = None
        self._mip: int | None = None
        self._queue: list[tuple[int, int]] = []
        self._on_drained = None
        # Chunks on the pool; a result carrying an older token is dropped.
        self._in_flight: list[tuple[int, int]] = []
        self._token = 0
        self._sync = False

    @property
    def is_active(self) -> bool:
        return bool(self._queue or self._in_flight)

    def run_to_completion(self) -> None:
        """Test-only drain, every chunk synchronous: nothing pumps the event
        loop here, so a pooled result could never arrive. Chunks already on
        the pool are abandoned and queued again."""
        self._token += 1
        self._queue[:0] = self._in_flight
        self._in_flight = []
        self._sync = True
        try:
            super().run_to_completion()
        finally:
            self._sync = False

    def start(self, cache, mip: int, chunks: list[tuple[int, int]], *, on_drained=None) -> None:
        """Queues `chunks` (mip-level chunk indices) on `cache`, replacing
        any margin warm already in flight -- LevelWarmer.start()'s shape,
        including "an empty/refused start leaves is_active False and
        schedules no timer at all".

        Refuses outright, queuing nothing, if the level isn't resident
        (cache.is_level_resident(mip) is False): see that method's own
        docstring for the freeze this precondition exists to prevent --
        get_chunk() on a not-yet-visited level would build the WHOLE level
        synchronously inside a tick, with no user action to blame it on.

        `on_drained`, if given, is called with no arguments once this
        queue empties (2026-09-07 plan's load-time margin warm, Step 3) --
        NOT on a refused/empty start (there's nothing to wait for then,
        and firing it would let a caller's chain-the-next-mip logic run
        one instance ahead of where it queued anything), and not on
        cancel() either, which drops it unfired for the same reason
        LevelWarmer.cancel() does."""
        self.cancel()
        if not chunks or not cache.is_level_resident(mip):
            return
        self._cache = cache
        self._mip = mip
        self._queue = list(chunks)
        self._on_drained = on_drained
        self._schedule()

    def cancel(self) -> None:
        """Drops the queued ring -- called alongside LevelWarmer.cancel()
        from ViewerWindow._cancel_warms(), for the same cancel-before-mutate
        race argument level_warm.py's own module docstring makes: a chunk
        warm and a scenario/elevation mutation must never interleave. Jobs
        already on the pool finish into their own scratch and are dropped
        on arrival; nothing waits for them."""
        self._cache = None
        self._mip = None
        self._queue = []
        self._on_drained = None
        self._token += 1
        self._in_flight = []
        self._stop_timer()

    def tick(self) -> bool:
        """Warms chunks until the next one would not fit BUDGET_MS (at least
        one; see the module docstring), then returns. A pooled chunk counts
        only its GUI-thread half, and a tick also stops at WORKERS jobs in
        flight, with the timer stopped until a result frees a slot. True
        while chunks remain queued or in flight afterward; False once THIS
        queue drains -- on_drained,
        if given, runs before returning, and may itself start a fresh queue
        (2026-09-07 plan's load-time margin warm chains the next neighbour
        mip this way), so a False return means "this call's own queue is
        empty", not "the timer is idle" -- check is_active for that. The
        tick ends at the drain even with budget left: that fresh queue has
        replaced self._queue and starts on the next tick, not inside this
        one.

        No StopIteration/install split the way LevelWarmer.tick() has:
        get_chunk() populates the cache itself, so there is nothing built-
        but-not-yet-installed to hand off -- which removes that module's
        single most delicate ordering hazard rather than reproducing it
        here.

        While any mouse button is held, warms nothing and backs the timer
        off instead (see class docstring) -- the queue is left untouched,
        so this always returns True here: the timer only ever reaches this
        branch with a non-empty queue, since a drained queue stops the
        timer before it can fire again."""
        from PyQt5.QtWidgets import QApplication

        if QApplication.mouseButtons():
            self._set_interval(self.HELD_INTERVAL_MS)
            return True
        self._set_interval(0)

        if not self._queue:
            if self._in_flight:
                self._stop_timer()
                return True
            return self._drained()
        pooled = self._pooled()
        start = _now()
        deadline = start + BUDGET_MS / 1000.0
        slowest = 0.0
        chunks = 0
        while self._queue:
            if pooled and len(self._in_flight) >= WORKERS:
                break
            before = _now()
            if chunks and before + slowest >= deadline:
                break
            cx, cy = self._queue.pop(0)
            try:
                job = self._cache.prepare_chunk_job(self._mip, cx, cy) if pooled else None
                if job is None:
                    self._cache.get_chunk(self._mip, cx, cy)
                else:
                    self._in_flight.append((cx, cy))
                    _pool().submit(_run_job, self._on_result, self._token, job, _result_bridge())
            except Exception as exc:  # noqa: BLE001
                # Same reasoning as LevelWarmer.tick()'s blanket handler: a
                # margin warm is an optimization, not worth propagating an
                # exception into the Qt event loop over.
                debug_log.log(f"margin warm: dropped a chunk ({exc!r})")
            slowest = max(slowest, _now() - before)
            chunks += 1
        if chunks:
            perf_trace.warm_tick((_now() - start) * 1000, chunks, slowest * 1000)
        if self._queue or self._in_flight:
            if pooled and (len(self._in_flight) >= WORKERS or not self._queue):
                self._stop_timer()
            return True
        return self._drained()

    def _pooled(self) -> bool:
        if WORKERS <= 0 or self._sync or not hasattr(self._cache, "prepare_chunk_job"):
            return False
        from PyQt5.QtWidgets import QApplication

        return QApplication.instance() is not None

    def _on_result(self, token: int, job, outcome, worker_ms: float) -> None:
        """GUI-thread side of a pooled chunk, via the bridge's queued signal."""
        if token != self._token:
            return
        self._in_flight.remove(job.key[1:])
        try:
            if isinstance(outcome, Exception):
                raise outcome
            self._cache.install_chunk(job, outcome)
        except Exception as exc:  # noqa: BLE001
            debug_log.log(f"margin warm: dropped a chunk ({exc!r})")
        perf_trace.warm_worker(worker_ms)
        if self._queue:
            self._schedule()
        elif not self._in_flight:
            self._drained()

    def _drained(self) -> bool:
        self._stop_timer()
        callback, self._on_drained = self._on_drained, None
        if callback is not None:
            callback()
        return False
