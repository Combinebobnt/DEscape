"""Incremental warm of a mip level's sprite layer, sliced across QTimer ticks
on the main thread (maintainer plan 2026-09-04).

**No threads, deliberately, and that is the design rather than a shortcut.**
The first zoom to a not-yet-visited mip level pays a 0.6-4.3s cold sprite
resolve-and-decode inside a Qt paint, which reads as a freeze. The obvious fix
is a background thread; it was measured and rejected. That work only partly
releases the GIL, so a worker leaves an uncontrolled 221ms worst-case stall on
the main thread anyway (measured 2026-09-04) while costing an LRU lock, an
elevations snapshot, a per-unit value snapshot and a cross-thread generation
token -- the live scenario's units and the cache's elevations array are both
mutated in place, so a throwaway cache instance would not have been enough.

Slicing the same walk across event-loop turns instead gives a BOUNDED
worst-case stall (BUDGET_MS, overrun by at most one ~10ms cold sprite decode),
needs none of that machinery, and is race-free by construction: a tick only
ever resumes between event-loop iterations, and every mutating path cancels
the warm before it mutates, so a warm slice and a mutation cannot interleave.

The cache owns what a level is and how one is installed (render_cache.
LevelWarmJob); this module owns only pacing, so render_cache.py stays Qt-free
and every check below is drivable with no QApplication at all via
run_to_completion().
"""

from __future__ import annotations

import time

from descape import debug_log

# Wall-clock budget per tick. PROVISIONAL -- to be tuned against the in-app
# pass, not derived. The measured per-unit slice floor is 10.3ms (one cold
# .sld decode), which is the lower bound this cannot go under.
#
# The honest worst tick is NOT 12ms, and not the 22ms the budget-plus-one-
# decode arithmetic suggests either: a level's final step runs
# _building_bboxes_iso whole (unsliced, correctly -- it is ~4ms on a small
# map), and render_cache's own measurements put it at 15-20ms on a real one.
# So a tick that spends its whole budget and then finishes a level lands
# nearer 30ms. Still bounded, which is the entire point next to the threaded
# design's uncontrolled 221ms, but the number to tune against is that one.
BUDGET_MS = 12


def neighbour_mips(cache, scale: float) -> list[int]:
    """The levels worth warming for a view sitting at `scale`: the opening
    level's NEIGHBOURS, clamped to the enumerated ladder -- never the opening
    level itself.

    That exclusion is the part that looks like an omission. Warming the
    opening level would duplicate work that always loses the race: a level is
    ~700ms of compute, i.e. ~60 ticks, and the canvas's first Qt paint is
    dispatched by the same event loop long before 60 idle ticks elapse, so the
    paint builds that level synchronously regardless. What this feature
    removes is the FIRST-ZOOM stutter, which is the neighbours. (Putting the
    opening level back in the set would still be safe -- the install
    predicates in render_cache's level_warm_job() drop a result the paint
    already built -- just wasteful.)

    Empty for a single-level ladder, which is Sloped and any small map whose
    tile_px has no exact power-of-two neighbour."""
    levels = cache.mip_levels()
    opening = cache.mip_for_scale(scale)
    return [mip for mip in (opening - 1, opening + 1) if levels[0] <= mip <= levels[-1]]


class _IdleTimerDriver:
    """Shared 0ms-QTimer plumbing between LevelWarmer and
    margin_warm.MarginWarmer (2026-09-07 plan's A3.1) -- the pacing
    mechanism (a 0ms QTimer, ticked on Qt event-loop idle) is identical
    between the two; only what one tick actually DOES differs, and that
    stays on each subclass. Extracted rather than duplicated because a
    second warmer with its own copy of this ~35 lines is exactly the kind
    of thing that silently drifts -- a fix made to one timer's lazy-import/
    headless-no-op/restart handling that never gets ported to the other.

    A subclass owns self._queue/self._job-equivalent state and must
    implement tick() -> bool (True while work remains, False once drained),
    calling self._schedule() from its own start() and self._stop_timer()
    from its own cancel() and from tick() itself once drained -- exactly
    what LevelWarmer below does."""

    def __init__(self) -> None:
        self._timer = None

    def tick(self) -> bool:
        raise NotImplementedError

    def run_to_completion(self) -> None:
        """Drains the whole queue synchronously -- see LevelWarmer's own
        docstring for why this exists (test-only; nothing in the app calls
        it, since doing so would reintroduce the multi-second freeze this
        whole idle-timer design exists to avoid)."""
        while self.tick():
            pass

    def _schedule(self) -> None:
        """Starts a 0ms QTimer, which Qt fires on event-loop idle -- so a
        warm makes no progress at all while the user is panning
        continuously, by design: the point is to use the idle time after a
        file opens (or, for MarginWarmer, between poll fires), not to
        compete with input.

        Created here rather than in __init__ so a driver can be constructed
        and driven (run_to_completion) with no QApplication at all. With no
        QApplication there is no event loop to tick on either, so this is a
        no-op rather than a failure."""
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication

        if QApplication.instance() is None:
            return
        if self._timer is None:
            self._timer = QTimer()
            self._timer.setInterval(0)
            self._timer.timeout.connect(self.tick)
        self._timer.start()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()


class LevelWarmer(_IdleTimerDriver):
    """Drives one cache's queued level warms, a budget's worth per event-loop
    turn.

    One instance per window, reused across documents: start() replaces
    whatever was queued, so a re-open cannot leave the previous document's
    cache being warmed."""

    def __init__(self) -> None:
        super().__init__()
        self._cache = None
        self._queue: list = []
        self._job = None
        self._job_mip: int | None = None
        self._on_job_done = None

    @property
    def is_active(self) -> bool:
        return self._job is not None or bool(self._queue)

    def start(self, cache, mips, *, on_job_done=None) -> None:
        """Queues `mips` on `cache`, replacing any warm already in flight.

        A level with nothing worth warming is not queued at all -- the
        cache's own level_warm_job() decides that (sprites off, units off, or
        the level already current), and it is asked HERE rather than at tick
        time, so an all-no-op start leaves is_active False and schedules no
        timer at all. That also fixes each job's revalidation baseline at
        warm start, which is the window the plan's predicates are stated
        over; building the generator itself runs none of the walk.

        `on_job_done`, if given, is called with a mip once THAT mip's job
        finishes (2026-09-07 plan's load-time margin warm, Step 2) --
        whether it installed cleanly or was dropped mid-walk (see tick()),
        since either way there is nothing left to wait for on that mip. It
        does NOT fire for a mip that never got a job queued at all (nothing
        worth warming, or already current): a caller that also needs those
        covered checks cache.is_level_resident(mip) itself right after
        start() returns, rather than this method inferring which is which."""
        self.cancel()
        jobs = [(mip, job) for mip, job in ((mip, cache.level_warm_job(mip)) for mip in mips) if job is not None]
        if not jobs:
            return
        self._cache = cache
        self._queue = jobs
        self._on_job_done = on_job_done
        self._schedule()

    def cancel(self) -> None:
        """Drops the in-flight job and the queue. Called by every path that
        mutates the scenario, mutates the elevations array or replaces the
        cache -- BEFORE it mutates, which is what makes a slice and a
        mutation unable to interleave.

        Dropping a half-built layer costs only the ticks already spent: it is
        never installed, and never was installable (the walk's payload rides
        StopIteration, so there is no partial value to observe). Drops
        on_job_done too -- a cancelled job's mip never finishes, so the
        callback must never fire for it."""
        self._job = None
        self._job_mip = None
        self._queue = []
        self._cache = None
        self._on_job_done = None
        self._stop_timer()

    def tick(self) -> bool:
        """Advances the current job for up to BUDGET_MS. Returns True while
        work remains, False once the queue is drained (at which point the
        timer is stopped).

        The budget is checked AFTER at least one step, so a tick always makes
        progress -- a zero-length budget would otherwise spin forever."""
        deadline = time.perf_counter() + BUDGET_MS / 1000.0
        while True:
            if self._job is None and not self._start_next_job():
                self._stop_timer()
                return False
            job = self._job
            try:
                next(job.gen)
            except StopIteration as done:
                # MUST be caught before the blanket handler below --
                # StopIteration is an Exception subclass, so the other order
                # silently swallows every normal completion and never installs
                # anything, with every cancel test still green.
                self._job = None
                self._install(job, done.value)
                self._notify_job_done()
            except Exception as exc:  # noqa: BLE001
                # Belt and braces for an un-enumerated future mutation path:
                # drop this level rather than let an exception reach the Qt
                # event loop. A warm is an optimization; failing one is not
                # worth a dialog.
                self._job = None
                debug_log.log(f"level warm: dropped a level mid-walk ({exc!r})")
                self._notify_job_done()
            if time.perf_counter() >= deadline:
                return True

    def _notify_job_done(self) -> None:
        if self._on_job_done is not None:
            self._on_job_done(self._job_mip)

    def _start_next_job(self) -> bool:
        if not self._queue:
            return False
        self._job_mip, self._job = self._queue.pop(0)
        return True

    def _install(self, job, payload) -> None:
        try:
            job.install(payload)
        except Exception as exc:  # noqa: BLE001
            # Same reasoning as tick()'s handler, and NOT covered by it: this
            # runs inside that method's `except StopIteration` block, so an
            # exception here would chain past the sibling handler and reach
            # the event loop. FlatChunkCache's row-count assert is the real
            # candidate.
            debug_log.log(f"level warm: install failed ({exc!r})")
