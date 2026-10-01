"""Holds automatic full (gen-2) gc collections off while the idle-time warms
run, and does the deferred collection on its own event-loop turn once all
warming has stopped (maintainer plan 2026-09-29, warm gc hold).

A full collection walks every object built since the load-time gc.freeze()
(viewer._freeze_loaded_document), which by then includes the warms' own sprite
layers, grids and packs: 25-55 ms that otherwise lands inside one warm tick,
after the tick budget had already been met. The stroke solves the same problem
with gc.disable() (map_view._pause_gc_for_stroke); this module does it for the
warm drivers, bounded, with two differences:

- Only gen 2 is held, by raising threshold2 (gc.set_threshold). Gen-0/1 keep
  freeing young cycles, so memory stays bounded over a multi-second hold, and
  the stroke's gc.isenabled() bookkeeping is never touched.
- release() is gc.collect() THEN the restore, when a full collection came due
  during the hold. Restoring first would leave count2 over the threshold, so
  the next gen-1 collection would promote to a full one inside whatever
  handler runs next. When none came due it only restores.

A hold ends at the first of the idle collect (QUIET_MS after every registered
driver has gone quiet and no mouse button is held) or an overdue collect, which
each warm tick runs as its only work once a hold is older than HOLD_MAX_S.

Qt-lazy like level_warm._IdleTimerDriver: nothing here needs a QApplication
until a hold is engaged. engage() is called from _schedule() after its
no-QApplication early return, from an active driver's overdue tick, and from
viewer._freeze_loaded_document() only while one of the viewer's warms is active,
which needs a shown window. So headless run_to_completion() never holds.
"""

from __future__ import annotations

import gc
import time
import weakref

from descape import debug_log, perf_trace

# Idle time after the last warm drains before the deferred collect. PROVISIONAL,
# tuned in the in-app pass like level_warm.BUDGET_MS.
QUIET_MS = 300
# A hold older than this is collected by the next warm tick. PROVISIONAL.
HOLD_MAX_S = 5.0
# threshold2 while held: count2 never reaches it (C int range).
HELD_THRESHOLD2 = 1 << 30

# Module attributes so tests can drive them deterministically.
_now = time.perf_counter
_collect = gc.collect
_count = gc.get_count

_saved: tuple[int, int, int] | None = None
_since: float | None = None
_drivers: weakref.WeakSet = weakref.WeakSet()
_timer = None


def register(driver) -> None:
    """Adds a warm driver (anything with is_active) the idle fire waits on."""
    _drivers.add(driver)


def is_engaged() -> bool:
    return _saved is not None


def engage() -> None:
    """Holds gen 2 off. Idempotent: only the first call saves the threshold
    tuple and stamps the start; every call stops a pending idle collect."""
    global _saved, _since
    if _saved is None:
        _saved = gc.get_threshold()
        _since = _now()
        gc.set_threshold(_saved[0], _saved[1], HELD_THRESHOLD2)
    _stop_idle_timer()


def idle_soon() -> None:
    """A driver drained or was cancelled: arms the idle collect for QUIET_MS
    from now, if a hold is engaged."""
    global _timer
    if _saved is None:
        return
    from PyQt5.QtWidgets import QApplication

    if QApplication.instance() is None:
        return
    if _timer is None:
        from PyQt5.QtCore import QTimer

        _timer = QTimer()
        _timer.setSingleShot(True)
        _timer.setInterval(QUIET_MS)
        _timer.timeout.connect(_on_idle)
    _timer.start()


def _on_idle() -> None:
    """The idle fire: re-arms while any driver is active or a mouse button is
    held (a stroke or drag), else collects and restores."""
    from PyQt5.QtWidgets import QApplication

    if _saved is None:
        return
    if QApplication.mouseButtons() or any(driver.is_active for driver in list(_drivers)):
        idle_soon()
        return
    release()


def overdue() -> bool:
    """True once a hold has lasted longer than HOLD_MAX_S."""
    return _saved is not None and _now() - _since > HOLD_MAX_S


def release(reason: str = "idle") -> None:
    """The deferred collection, then the saved threshold back. No-op when not
    engaged. Collects only if a full collection came due during the hold
    (count2 past the saved threshold2): otherwise CPython would not have run
    one, and a forced collect every time warming stops is 30-90 ms of gc it
    never paid (2026-09-30 stress log: 51 in 2 minutes)."""
    global _saved, _since
    if _saved is None:
        return
    held = _now() - _since
    due = _count()[2] > _saved[2]
    t0 = _now()
    if due:
        with perf_trace.phase("gc_release"):
            _collect()
    ms = (_now() - t0) * 1000
    gc.set_threshold(*_saved)
    _saved, _since = None, None
    _stop_idle_timer()
    if due:
        debug_log.log(f"warm gc: {reason} collect {ms:.1f} ms, held {held:.1f} s")
    else:
        debug_log.log(f"warm gc: {reason} release, no full collection due, held {held:.1f} s")


def reset() -> None:
    """Restores the saved threshold without collecting and stops the idle
    timer: document lifetime boundaries (load, close) and tests."""
    global _saved, _since
    if _saved is not None:
        gc.set_threshold(*_saved)
    _saved, _since = None, None
    _stop_idle_timer()


def _stop_idle_timer() -> None:
    if _timer is not None:
        _timer.stop()
