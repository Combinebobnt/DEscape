"""In-app draw-perf trace (draw-perf plan Step 1) -- attributes felt drag
latency to a phase instead of guessing, covering the layer
tools/bench_draw_stroke.py cannot see: Qt's own repaint. Modelled on
debug_log.py -- module-level state, no class -- and, like it, in-process
only. Off by default; every hook is a single bool check when disabled, and
phase() hands back one shared pre-built null context manager rather than
allocating, so the zero-cost claim holds with no per-call branching inside
the hot path itself.

Aggregated per STROKE, not per step: at 60 steps/sec a per-step line fills
debug_log's 1000-line ring buffer in seconds. flush() emits one line per
drag on mouse release.

repaint is tracked separately from the step-nested phases (pick, highlight,
stroke_scan, bbox, patch, invalidate): Qt's paint() fires on a paint event
AFTER invalidate_region() schedules one, not synchronously inside
mouseMoveEvent, so it never nests inside a step() boundary -- reported as
its own call count/total/max instead of being forced into a ms/step
column it cannot honestly belong to."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext

from descape import debug_log

_enabled = os.environ.get("DESCAPE_PERF_TRACE") == "1"
_NULL_PHASE = nullcontext()

_current_step: dict[str, float] = {}
_step_totals: list[float] = []
_phase_sums: dict[str, float] = {}
_phase_order: list[str] = []
_repaint_durations: list[float] = []
_armed_label: str | None = None


def enable(on: bool) -> None:
    global _enabled
    _enabled = on


def is_enabled() -> bool:
    return _enabled


def _record(name: str, elapsed_ms: float) -> None:
    if name == "repaint":
        _repaint_durations.append(elapsed_ms)
        return
    if name not in _current_step and name not in _phase_sums:
        _phase_order.append(name)
    _current_step[name] = _current_step.get(name, 0.0) + elapsed_ms


class _PhaseTimer:
    __slots__ = ("_name", "_t0")

    def __init__(self, name: str) -> None:
        self._name = name

    def __enter__(self) -> "_PhaseTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        _record(self._name, (time.perf_counter() - self._t0) * 1000)


def phase(name: str):
    if not _enabled:
        return _NULL_PHASE
    return _PhaseTimer(name)


def arm(label: str) -> None:
    """Records a pending flush label for the next first_paint_done() call --
    the load path's way of capturing the deferred first-composite cost
    without threading a callback through MapView.set_source(). No-op while
    disabled, matching every other hook here."""
    global _armed_label
    if not _enabled:
        return
    _armed_label = label


def first_paint_done() -> None:
    """Pairs with arm(): if a label is armed, clears it and flushes the
    repaint duration(s) recorded since under that label. A no-op if nothing
    is armed (disabled, or a paint that isn't a load's first)."""
    global _armed_label
    if not _enabled or _armed_label is None:
        return
    label = _armed_label
    _armed_label = None
    flush(label)


def step() -> None:
    """Closes out the current drag step and starts the next. A no-op while
    disabled, and also while the current step recorded no phases -- e.g. the
    first call of a drag, or a mouse-move that only re-touched an
    already-painted tile (map_view._touch_tile's own dedupe)."""
    global _current_step
    if not _enabled or not _current_step:
        return
    _step_totals.append(sum(_current_step.values()))
    for phase_name, elapsed_ms in _current_step.items():
        _phase_sums[phase_name] = _phase_sums.get(phase_name, 0.0) + elapsed_ms
    _current_step = {}


def flush(label: str) -> None:
    """Emits the accumulated drag's summary to debug_log.log() -- deliberately
    NOT _log_status(), so a drag doesn't spam the visible status pane -- then
    resets all state for the next drag. A no-op if no step was ever closed
    (e.g. a click with no drag)."""
    global _current_step, _step_totals, _phase_sums, _phase_order, _repaint_durations
    if not _enabled:
        return
    n = len(_step_totals)
    if n == 0 and not _repaint_durations:
        _current_step = {}
        return
    lines = []
    if n:
        total = sum(_step_totals)
        mean_step = total / n
        max_step = max(_step_totals)
        phase_line = " ".join(
            f"{name} {_phase_sums.get(name, 0.0) / n:.1f}" for name in _phase_order
        )
        lines.append(
            f"perf drag {label}: {n} steps, {total:.0f}ms total, {mean_step:.1f}ms/step (max {max_step:.1f})"
        )
        lines.append(f"  | {phase_line}")
    if _repaint_durations:
        r_n = len(_repaint_durations)
        r_total = sum(_repaint_durations)
        r_max = max(_repaint_durations)
        if n:
            lines.append(f"  | repaint: {r_n} calls, {r_total:.0f}ms total (max {r_max:.1f})")
        else:
            lines.append(f"perf {label}: repaint: {r_n} calls, {r_total:.0f}ms total (max {r_max:.1f})")
    debug_log.log("\n".join(lines))
    _current_step = {}
    _step_totals = []
    _phase_sums = {}
    _phase_order = []
    _repaint_durations = []
