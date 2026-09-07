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
    grid_w, grid_h = cache.canvas_dims(mip)
    chunk_px = cache.chunk_px
    max_cx = -(-grid_w // chunk_px) - 1
    max_cy = -(-grid_h // chunk_px) - 1
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

    @property
    def is_active(self) -> bool:
        return bool(self._queue)

    def start(self, cache, mip: int, chunks: list[tuple[int, int]]) -> None:
        """Queues `chunks` (mip-level chunk indices) on `cache`, replacing
        any margin warm already in flight -- LevelWarmer.start()'s shape,
        including "an empty/refused start leaves is_active False and
        schedules no timer at all".

        Refuses outright, queuing nothing, if the level isn't resident
        (cache.is_level_resident(mip) is False): see that method's own
        docstring for the freeze this precondition exists to prevent --
        get_chunk() on a not-yet-visited level would build the WHOLE level
        synchronously inside a tick, with no user action to blame it on."""
        self.cancel()
        if not chunks or not cache.is_level_resident(mip):
            return
        self._cache = cache
        self._mip = mip
        self._queue = list(chunks)
        self._schedule()

    def cancel(self) -> None:
        """Drops the queued ring -- called alongside LevelWarmer.cancel()
        from ViewerWindow._cancel_warms(), for the same cancel-before-mutate
        race argument level_warm.py's own module docstring makes: a chunk
        warm and a scenario/elevation mutation must never interleave."""
        self._cache = None
        self._mip = None
        self._queue = []
        self._stop_timer()

    def tick(self) -> bool:
        """Warms exactly one chunk, then returns. True while chunks remain,
        False (with the timer stopped) once the queue is drained.

        No StopIteration/install split the way LevelWarmer.tick() has:
        get_chunk() populates the cache itself, so there is nothing built-
        but-not-yet-installed to hand off -- which removes that module's
        single most delicate ordering hazard rather than reproducing it
        here."""
        if not self._queue:
            self._stop_timer()
            return False
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
        self._stop_timer()
        return False
