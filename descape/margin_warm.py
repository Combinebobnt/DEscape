"""Idle-time warm of a bounded chunk ring just outside the current viewport,
continuously re-targeted as the user pans/zooms (maintainer plan
2026-09-07's Phase A, Step A3). Sibling to level_warm.py -- both share the
0ms-QTimer pacing mechanism (level_warm._IdleTimerDriver) -- rather than an
extension of it: LevelWarmer's whole contract is a wall-clock BUDGET with a
bounded overrun, and a margin chunk cannot honour that.

**One chunk per tick, no wall-clock budget at all -- the central difference
from LevelWarmer, stated here so it doesn't get "fixed" back into a budget
later.** LevelWarmer's BUDGET_MS bound holds because its unit of work (one
sliced generator step) is a few microseconds; a margin chunk's unit of work
is one get_chunk() call, measured at 39.5-117ms for Stepped
and under 25ms for Flat -- both far past
what a "bounded overrun" framing could honestly claim, and get_chunk() is
atomic (it cannot be resumed mid-composite the way a sprite walk's `yield`
can). Admitting one into a budget-checked tick() would make LevelWarmer's
own module docstring's central claim false for whoever reads it expecting
one pacing story to cover both. So the honest per-tick bound here is stated
in those terms -- "one chunk, 39.5-117ms measured (Stepped), under 25ms
(Flat)" -- rather than a millisecond number, and the two drivers stay
separate classes rather than sharing one that would have to average two
very different costs.

render_cache._ChunkCacheBase.is_level_resident()/has_chunk() are this
module's other half of the correctness argument -- see MarginWarmer.start()
and ring_chunks() below for how each is used."""

from __future__ import annotations

from descape import debug_log
from descape.level_warm import _IdleTimerDriver


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
    """Drives one cache's margin-ring warm, one get_chunk() per idle tick --
    see this module's own docstring for why that's a single chunk with no
    wall-clock budget, unlike LevelWarmer's sliced-generator ticks.

    One instance per window, reused across documents and retargeted on
    every viewport-changed poll fire (ViewerWindow._on_viewport_changed) --
    start() replaces whatever was queued outright, which IS the parent
    plan's A4 retarget mechanism; a second mechanism on top of that would be
    new machinery this doesn't need."""

    def __init__(self) -> None:
        super().__init__()
        self._cache = None
        self._mip: int | None = None
        self._queue: list[tuple[int, int]] = []
        self._on_drained = None

    @property
    def is_active(self) -> bool:
        return bool(self._queue)

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
        warm and a scenario/elevation mutation must never interleave."""
        self._cache = None
        self._mip = None
        self._queue = []
        self._on_drained = None
        self._stop_timer()

    def tick(self) -> bool:
        """Warms exactly one chunk, then returns. True while chunks remain
        queued afterward; False once THIS queue drains -- on_drained, if
        given, runs before returning, and may itself start a fresh queue
        (2026-09-07 plan's load-time margin warm chains the next neighbour
        mip this way), so a False return means "this call's own queue is
        empty", not "the timer is idle" -- check is_active for that.

        No StopIteration/install split the way LevelWarmer.tick() has:
        get_chunk() populates the cache itself, so there is nothing built-
        but-not-yet-installed to hand off -- which removes that module's
        single most delicate ordering hazard rather than reproducing it
        here."""
        if not self._queue:
            return self._drained()
        cx, cy = self._queue.pop(0)
        try:
            self._cache.get_chunk(self._mip, cx, cy)
        except Exception as exc:  # noqa: BLE001
            # Same reasoning as LevelWarmer.tick()'s blanket handler: a
            # margin warm is an optimization, not worth propagating an
            # exception into the Qt event loop over.
            debug_log.log(f"margin warm: dropped a chunk ({exc!r})")
        if self._queue:
            return True
        return self._drained()

    def _drained(self) -> bool:
        self._stop_timer()
        callback, self._on_drained = self._on_drained, None
        if callback is not None:
            callback()
        return False
