"""Incremental warm of a mip level's sprite layer, and then of its native unit
pack's pending tiles (maintainer plan 2026-09-27), sliced across QTimer ticks
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
worst-case stall (BUDGET_MS, overrun by at most one step: one unit's resolve
or one assembly slice, see BUDGET_MS),
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

from descape import debug_log, gc_hold, perf_trace

# Wall-clock budget per tick. PROVISIONAL -- to be tuned against the in-app
# pass, not derived. The measured per-unit slice floor is 10.3ms (one cold
# .sld decode), which is the lower bound this cannot go under.
#
# The honest worst tick is the budget plus one step. Since the 2026-09-29
# warm-tick plan a level's assembly is sliced too (render.ASSEMBLY_SLICE),
# its install only commits, and a job's unsliced first step (walk setup,
# UnitPack(), a flush) only starts a tick, so the step past the budget is one
# unit's resolve or an assembly slice. A resolve is a cold .sld decode at the
# coarse levels, and at the fine ones an editor marker's first build or a
# many-piece composite: old-allies' level 1 warm peaked at 10.8-11.0ms steps
# (a 1x3 blocker's marker), 17-19ms ticks (2026-10-07, after the separable
# outline; Perf Trace's `max step` names the unit).
BUDGET_MS = 12

# The tick's clock, a module attribute so tests can drive it deterministically.
_now = time.perf_counter


def neighbour_mips(cache, scale: float) -> list[int]:
    """neighbour_mips_of() the level a view sitting at `scale` paints."""
    return neighbour_mips_of(cache, cache.mip_for_scale(scale))


def neighbour_mips_of(cache, mip: int) -> list[int]:
    """The levels worth warming around `mip`: the anchor level's NEIGHBOURS,
    clamped to the enumerated ladder -- never the anchor level itself.

    That exclusion is the part that looks like an omission. Warming the
    anchor level would duplicate work that always loses the race: a level is
    ~700ms of compute, i.e. ~60 ticks, and the canvas's first Qt paint is
    dispatched by the same event loop long before 60 idle ticks elapse, so the
    paint builds that level synchronously regardless. What this feature
    removes is the FIRST-ZOOM stutter, which is the neighbours. (Putting the
    anchor level back in the set would still be safe -- the install
    predicates in render_cache's level_warm_job() drop a result the paint
    already built -- just wasteful.)

    Empty for a single-level ladder, which is Sloped and any small map whose
    tile_px has no exact power-of-two neighbour."""
    levels = cache.mip_levels()
    return [m for m in (mip - 1, mip + 1) if levels[0] <= m <= levels[-1]]


def _step_name(label: str, cache, unit) -> str:
    """Perf Trace's `max step` name: the step label, plus `p<player> #<index>
    const <c>` when the step resolved a walk's unit. Called only for a tick
    that becomes the line's worst (perf_trace.level_warm_tick)."""
    if unit is None:
        return label
    from descape import render

    scenario = getattr(cache, "scenario", None)
    if scenario is not None:
        for player_id, i, u in render.unit_own_tile_index(scenario).get((int(unit.x), int(unit.y)), ()):
            if u is unit:
                return f"{label} p{player_id} #{i} const {unit.unit_const}"
    return f"{label} const {unit.unit_const}"


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
    what LevelWarmer below does.

    **Both warmers pause while any mouse button is held** (_held_backoff;
    MarginWarmer's item 23, shared since the 2026-09-29 warm-tick plan). A
    mouse pan or stroke delivers discrete move events, so the 0ms timer would
    still fire in the gaps between them and land a tick mid-drag as a hitch;
    for LevelWarmer that tick could also flush a level mid-stroke, which the
    per-step re-arm would repeat every gap. Wheel and keyboard pans hold no
    button and are unaffected."""

    HELD_INTERVAL_MS = 50

    def __init__(self) -> None:
        self._timer = None
        # The deferred gen-2 collect waits until every registered driver is idle.
        gc_hold.register(self)

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
        warm makes no progress at all while the user is panning with the
        keyboard or wheel, by design: the point is to use the idle time
        after a file opens (or, for MarginWarmer, between poll fires), not
        to compete with input. A mouse-held pan is different: move events
        are discrete, so the 0ms timer still fires in the gaps between them
        unless the tick backs off (both subclasses do, via _held_backoff()).

        Created here rather than in __init__ so a driver can be constructed
        and driven (run_to_completion) with no QApplication at all. With no
        QApplication there is no event loop to tick on either, so this is a
        no-op rather than a failure."""
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication

        if QApplication.instance() is None:
            return
        # Full collections wait until warming stops (gc_hold); headless never holds.
        gc_hold.engage()
        if self._timer is None:
            self._timer = QTimer()
            self._timer.setInterval(0)
            self._timer.timeout.connect(self.tick)
        self._timer.start()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()

    def _set_interval(self, ms: int) -> None:
        """Retargets the already-started timer's fire interval without
        stopping it -- the mouse-held backoff (_held_backoff) uses this rather
        than a fresh _schedule() call, which would also (re)start a timer
        that may not be running yet at __init__ time."""
        if self._timer is not None:
            self._timer.setInterval(ms)

    def _held_backoff(self) -> bool:
        """True while a mouse button is held, with the timer backed off to
        HELD_INTERVAL_MS so the early-returning tick doesn't spin a core; else
        restores 0ms and returns False. Headless, mouseButtons() reads
        NoButton, so run_to_completion() is unaffected."""
        from PyQt5.QtWidgets import QApplication

        if QApplication.mouseButtons():
            self._set_interval(self.HELD_INTERVAL_MS)
            return True
        self._set_interval(0)
        return False

    @staticmethod
    def _overdue_collect() -> bool:
        """True after running a gc hold's overdue collect (gc_hold.HOLD_MAX_S),
        which is then the tick's only work, as with a job's unsliced first step.
        The warm is still running, so the hold resumes with a fresh start."""
        if not gc_hold.overdue():
            return False
        gc_hold.release("overdue")
        gc_hold.engage()
        return True


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
        self._job_notify = False
        self._on_job_done = None
        # Install ms this tick, for Perf Trace; None while it is off.
        self._tick_installs: list[float] | None = None
        # This tick's ms per step label (`walk -1 setup`, `install`...); None while off.
        self._tick_split: dict[str, float] | None = None
        # This tick's longest single step, (ms, label, unit or None); None while off.
        self._tick_max: tuple[float, str, object] | None = None
        # What the current job's previous step yielded: the walk's unit, else None.
        self._job_prev = None
        # The current job has not stepped yet: its first step is unsliced setup.
        self._job_fresh = False

    @property
    def is_active(self) -> bool:
        return self._job is not None or bool(self._queue)

    def start(self, cache, mips, *, pack_only=(), on_job_done=None) -> set[int]:
        """Queues `mips` on `cache`, replacing any warm already in flight.

        Each mip gets its sprite-layer job, then its unit-pack derive
        (cache.pack_warm_job, maintainer plan 2026-09-27) chained after it.
        `pack_only` mips not in `mips` get just the derive, queued last.

        A job with nothing worth doing is not queued at all -- the cache's
        own level_warm_job()/pack_warm_job() decide that (sprites off, units
        off, numpy, or the level already current), and they are asked HERE
        rather than at tick time, so an all-no-op start leaves is_active
        False and schedules no timer at all. That also fixes each job's
        revalidation baseline at warm start, which is the window the plan's
        predicates are stated over; building the generator itself runs none
        of the walk.

        `on_job_done`, if given, is called with a `mips` entry once the LAST
        job queued for it finishes (2026-09-07 plan's load-time margin warm,
        Step 2), so a neighbour's load warm starts after its pack derive --
        whether it installed cleanly or was dropped mid-walk (see tick()),
        since either way there is nothing left to wait for on that mip. It
        never fires for a `pack_only` mip, nor for a mip that got no job at
        all: returns the set of mips it will fire for, and a caller that
        also needs the rest covered checks cache.is_level_resident(mip)
        itself right after start() returns."""
        self.cancel()
        jobs = []
        for mip in mips:
            level_job = cache.level_warm_job(mip)
            pack_job = cache.pack_warm_job(mip, after_level_warm=level_job is not None)
            if level_job is not None:
                jobs.append((mip, level_job, pack_job is None))
            if pack_job is not None:
                jobs.append((mip, pack_job, True))
        for mip in pack_only:
            if mip in mips:
                continue
            pack_job = cache.pack_warm_job(mip)
            if pack_job is not None:
                jobs.append((mip, pack_job, False))
        if not jobs:
            return set()
        self._cache = cache
        self._queue = jobs
        self._on_job_done = on_job_done
        self._schedule()
        return {mip for mip, _job, notify in jobs if notify}

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
        self._job_notify = False
        self._job_prev = None
        self._queue = []
        self._cache = None
        self._on_job_done = None
        self._stop_timer()
        gc_hold.idle_soon()

    def tick(self) -> bool:
        """Advances the current job for up to BUDGET_MS. Returns True while
        work remains, False once the queue is drained (at which point the
        timer is stopped).

        The budget is checked AFTER at least one step, so a tick always makes
        progress -- a zero-length budget would otherwise spin forever. A job
        starts only at the top of a tick and its completion ends the tick
        (2026-09-29 warm-tick plan), so its unsliced first step never lands
        after a spent budget.

        While a mouse button is held, advances nothing and returns True (see
        _IdleTimerDriver._held_backoff), before Perf Trace counts a tick. An
        overdue gc hold's collect is a whole tick too (_overdue_collect)."""
        if self._held_backoff():
            return True
        if self._overdue_collect():
            return True
        if not perf_trace.is_enabled():
            return self._tick()
        start = time.perf_counter()
        gc0 = perf_trace.gc_ms()
        self._tick_installs = []
        self._tick_split = {}
        self._tick_max = (0.0, "", None)
        cache = self._cache
        try:
            with perf_trace.where("level-warm"):
                return self._tick()
        finally:
            gc1 = perf_trace.gc_ms()
            gc_ms = None if gc0 is None or gc1 is None else gc1 - gc0
            tick_ms = (time.perf_counter() - start) * 1000
            ms, label, unit = self._tick_max
            max_step = (ms, lambda: _step_name(label, cache, unit)) if label else None
            perf_trace.level_warm_tick(tick_ms, self._tick_installs, self._tick_split, gc_ms, max_step=max_step)
            self._tick_installs = None
            self._tick_split = None
            self._tick_max = None

    def _tick(self) -> bool:
        deadline = _now() + BUDGET_MS / 1000.0
        # A job's first step only runs at the top of a tick: its setup (the
        # walk's pre-yield work, UnitPack(), a flush) is unsliced.
        if self._job is None and not self._start_next_job():
            return self._more()
        while True:
            job = self._job
            try:
                if self._tick_split is None:
                    self._job_fresh = False
                    next(job.gen)
                else:
                    self._timed_next(job)
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
            if self._job is None:
                # A finished job ends the tick (on_job_done may have run a
                # load-warm queue build); the next job starts on a fresh one.
                return self._more()
            if _now() >= deadline:
                return True

    def _more(self) -> bool:
        """True while jobs remain queued; else stops the timer (drained)."""
        if self._queue:
            return True
        self._stop_timer()
        gc_hold.idle_soon()
        return False

    def _timed_next(self, job) -> None:
        """next(job.gen) with its ms added to the tick's split, labelled by job
        kind and mip; a job's first step is labelled `setup` apart. The
        tick's longest step is kept with the unit the job yielded BEFORE it:
        the walk yields at the top of its per-unit body, so a step's cost is
        resolving the previous step's unit."""
        label = f"{job.kind} {self._job_mip}"
        if self._job_fresh:
            self._job_fresh = False
            label += " setup"
        t0 = _now()
        value = None
        try:
            value = next(job.gen)
        finally:
            ms = (_now() - t0) * 1000
            self._split_add(label, ms)
            if self._tick_max is not None and ms > self._tick_max[0]:
                self._tick_max = (ms, label, self._job_prev)
            self._job_prev = value

    def _split_add(self, label: str, ms: float) -> None:
        if self._tick_split is not None:
            self._tick_split[label] = self._tick_split.get(label, 0.0) + ms

    def _notify_job_done(self) -> None:
        if self._job_notify and self._on_job_done is not None:
            t0 = _now()
            self._on_job_done(self._job_mip)
            self._split_add("done", (_now() - t0) * 1000)

    def _start_next_job(self) -> bool:
        if not self._queue:
            return False
        self._job_mip, self._job, self._job_notify = self._queue.pop(0)
        self._job_fresh = True
        self._job_prev = None
        return True

    def _install(self, job, payload) -> None:
        try:
            with perf_trace.level("install", self._job_mip) as ev:
                job.install(payload)
            if ev is not None and self._tick_installs is not None:
                self._tick_installs.append(ev.ms)
                self._split_add("install", ev.ms)
        except Exception as exc:  # noqa: BLE001
            # Same reasoning as tick()'s handler, and NOT covered by it: this
            # runs inside that method's `except StopIteration` block, so an
            # exception here would chain past the sibling handler and reach
            # the event loop. FlatChunkCache's row-count assert is the real
            # candidate.
            debug_log.log(f"level warm: install failed ({exc!r})")
