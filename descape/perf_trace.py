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

A step is one mouse event that entered new tiles. Through v0.8 it was one
cursor tile; after it, an event's gap-filled cursor path (Draw, Elevate,
Set Elevation) is one step, so ms/step can read higher than older logs at
the same per-tile cost. Compare per-tile cost across that change with care.
Logs from before Batch F Track S call the stroke bookkeeping phase
stroke_scan (an O(map) scan); it is stroke_dirty now, O(tiles written).

repaint is tracked separately from the step-nested phases (pick, highlight,
stroke_dirty, bbox, patch, invalidate): Qt's paint() fires on a paint event
AFTER invalidate_region() schedules one, not synchronously inside
mouseMoveEvent, so it never nests inside a step() boundary -- reported as
its own call count/total/max instead of being forced into a ms/step
column it cannot honestly belong to. Work inside a paint is split with
repaint_phase() (chunk `composite`, `blit`), printed in brackets after the
repaint total with `other` for the rest outside the level events.

Margin-warm ticks (descape.margin_warm) are idle-time work outside any
step, so they are reported the same way, as a `warm:` bucket on the `perf
view` line. It shows the longest tick and the longest single chunk
separately: the tick is budgeted, so only the chunk figure says whether one
chunk alone is too slow to fit a budget.

A drag is bracketed by MapView's begin_drag()/end_drag(). Hover moves time
pick/highlight too, so begin_drag() drops what hover left in the open step,
and flushes repaints from before the drag (pans, zoom notches) as their own
`perf view` line. Outside a drag, flush_idle() does the same once the
canvas has been quiet for a moment (viewer_canvas's idle timer), so panning
shows up on its own instead of inside the next drag's repaint row.

One-shot actions (Place Unit, Nudge, Undo, Fill, a zoom notch...) are
bracketed by op() and print a `perf op` line: the total, the phases opened
inside it and `untimed`, the total minus the phase sum. Consecutive ops with
the same label merge into one `xN` line, which is written when a different
op starts, a drag begins, the idle flush runs, or Help > Debug Log opens
(flush_pending_op()). An op opened inside a drag
or inside another op is only a phase of it (one with the same label as the
open op is transparent), so it never disturbs the drag's own step.

Level events (level_event()/level()) are cache-side builds: a level's sprite
walk, a deferred elevation flush, a unit pack, a Flat icon layer. Each is
tagged with where it ran (`repaint`, `margin-warm`, `level-warm`,
`op:<label>`, `other`) and printed aggregated on the next line: the op's own
line if an op was open, otherwise the next drag, load or view line.

Cold sprite counters (`sprites +files +frames`) are deltas of unit_sprites'
two cache sizes. They undercount once a cap evicts, and a lookup that missed
still counts as a file. Terrain texture loads (`textures +N files X ms`,
asset_source) are recorded as they happen instead: files decoded, and X the
ms a caller was blocked loading one or waiting on a prefetched one, with the
wait and the prefetch pool's own decodes in parentheses when there were any.
Chunk `composite` still includes the blocked time.
stall() is the event-loop watchdog's report (timer in
viewer_canvas); it names the op, phase, span or level event that covered the
stall, or says it was untimed (during a drag if one was active anywhere in the
stall's window, else since the last op). Then, whatever the cause, it prints
`covered X of Y` and the top 3 entries by ms: X is the union of every timed
entry's interval inside the window (ops and spans left out), so a gc inside a
phase, or a level build inside a repaint, is not counted twice. The entries
are a count-bounded buffer (512), so when a long window outlasts it the
figure prints as `covered >= X`: entries inside the window were dropped.

span() is a wall-time meter for a whole handler (a stroke's press, step and
release). It is not a phase, so it never enters step totals: it reaches the
stall window, and inside a drag its wall time sums into the drag line's `wall
X, untimed Y`, where untimed is the wall minus the phases recorded inside
spans. A phase run between events (a deferred callback) is on the drag line
but in no span, so it is left out of both. It lands in the drag's open step,
so the next step() folds it into that event's step (a re-touch of already
painted tiles never calls step(): map_view._touch_tile returns first), or,
with no later step, it prints on the drag's `end:` line.

Garbage collection is timed through a gc.callbacks hook (set_gc_hook(), driven
by the viewer's Perf Trace toggle). A gen-2 collection prints as `gc gen2`
on the op, drag, load or view line it landed in, and on a stall line that
covered it; it nests inside whatever phase was running, so it is not additive
with the phases. The hook runs on whichever thread collected (a margin-warm
worker can), so it only appends to a deque; routing happens on the GUI thread
in _drain_gc(). Young generations feed gc_ms(), and one of at least
YOUNG_GC_MIN_MS reaches the stall window as `gc gen0`/`gc gen1`, nowhere else."""

from __future__ import annotations

import gc
import os
import sys
import time
from collections import deque
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Self

from descape import composite_backend, debug_log

_enabled = os.environ.get("DESCAPE_PERF_TRACE") == "1"
_NULL_PHASE = nullcontext()
# A gen-0/1 collection this long or longer reaches the stall window.
YOUNG_GC_MIN_MS = 5.0
# Detail prefix of a span's _recent entry: in a stall's cause, never its covered sum.
_SPAN = "span "

_current_step: dict[str, float] = {}
_step_totals: list[float] = []
_phase_sums: dict[str, float] = {}
_phase_order: list[str] = []
_repaint_durations: list[float] = []
# name -> [calls, ms] for repaint_phase()s, and level-event ms inside repaints.
_repaint_phases: dict[str, list] = {}
_repaint_level_ms = 0.0
# One (tick_ms, chunks, slowest_chunk_ms) per margin-warm tick.
_warm_ticks: list[tuple[float, int, float]] = []
# Kernel ms per chunk run on a margin-warm worker thread, timed there and
# recorded on arrival, so this module stays GUI-thread-only.
_warm_worker: list[float] = []
_armed_label: str | None = None
_drag_active = False
# Called on every repaint recorded outside a drag; viewer_canvas installs a
# timer restart here that ends in flush_idle(), keeping this module Qt-free.
_idle_scheduler = None
# Open ops, outermost first; _top_op is the one that owns _current_step.
_op_stack: list[_OpTimer] = []
_top_op: _OpTimer | None = None
# The written-when-something-else-happens `perf op` line, see _PendingOp.
_pending_op: _PendingOp | None = None
# Where-tags for level events, innermost last.
_where_stack: list[str] = []
# (kind, mip, where) -> _LevelAgg, for events outside any top-level op.
_level_events: dict[tuple[str, int, str], _LevelAgg] = {}
# One (tick_ms, installs, slowest_install_ms) per LevelWarmer tick.
_level_warm_ticks: list[tuple[float, int, float]] = []
# The slowest of those ticks: (tick_ms, ms per step label, gc_ms).
_level_warm_worst: tuple[float, dict[str, float], float | None] | None = None
# Mip transitions seen by paint since the last view line, and the last mip.
_mip_changes: list[tuple[int, int]] = []
_view_mip: int | None = None
# Sprite cache sizes at the last sample, and deltas owed to the next view line.
_sprite_base: tuple[int, int] | None = None
_sprite_pending = [0, 0]
# (end_time, ms, op_label, detail) for recent ops, phases, spans and level
# events. A 2-step stroke click adds ~15 plus its repaints, hence the size.
# Count-bounded for memory: a long stall window can outlast it (`covered >=`).
_RECENT_MAXLEN = 512
_recent: deque = deque(maxlen=_RECENT_MAXLEN)
# (label, end_time) of the last top-level op; a nested one is only a phase.
_last_op: tuple[str, float] | None = None
# When the last drag ended, so a stall can tell a drag was active in its window.
_drag_end: float | None = None
# Open span()s, outermost first; only the outermost adds to _drag_wall.
_span_stack: list[_SpanTimer] = []
# This drag's span wall ms, and the phase ms recorded inside those spans.
_drag_wall = 0.0
_drag_span_phase_ms = 0.0
# (end_time, ms, generation) per gen-2 collection and per young one of at
# least YOUNG_GC_MIN_MS, appended by _on_gc on any thread.
_gc_raw: deque = deque()
_gc_t0: float | None = None
# Every collection's ms since the hook went in, all generations (gc_ms()).
_gc_total_ms = 0.0
# (kind, ms) per terrain texture load, wait or prefetch, appended on any thread; see _drain_textures().
_texture_raw: deque = deque()


def _default_sprite_counts() -> tuple[int, int]:
    # Never imports unit_sprites itself: a process that hasn't decoded a
    # sprite yet has nothing to count.
    module = sys.modules.get("descape.unit_sprites")
    return (0, 0) if module is None else module.cache_counts()


_sprite_counter = _default_sprite_counts


def enable(on: bool) -> None:
    global _enabled
    _enabled = on


def is_enabled() -> bool:
    return _enabled


def set_idle_scheduler(scheduler) -> None:
    global _idle_scheduler
    _idle_scheduler = scheduler


def _schedule_idle() -> None:
    if _idle_scheduler is not None and not _drag_active:
        _idle_scheduler()


def _op_label() -> str | None:
    return _op_stack[-1].label if _op_stack else None


def _on_gc(phase: str, info: dict) -> None:
    """The gc.callbacks hook. May run on a worker thread: deque append only."""
    global _gc_t0, _gc_total_ms
    if not _enabled:
        _gc_t0 = None
        return
    now = time.perf_counter()
    if phase == "start":
        _gc_t0 = now
        return
    if _gc_t0 is None:
        return
    ms = (now - _gc_t0) * 1000
    _gc_t0 = None
    _gc_total_ms += ms
    generation = info.get("generation")
    if generation == 2 or ms >= YOUNG_GC_MIN_MS:
        _gc_raw.append((now, ms, generation))


def set_gc_hook(on: bool) -> None:
    """Installs or removes _on_gc; the viewer's Perf Trace toggle drives it."""
    installed = _on_gc in gc.callbacks
    if on and not installed:
        gc.callbacks.append(_on_gc)
    elif not on and installed:
        gc.callbacks.remove(_on_gc)


def gc_ms() -> float | None:
    """Every collection's ms since the hook went in, all generations; None
    while the hook is out (gc not measured, which is not zero)."""
    return _gc_total_ms if _on_gc in gc.callbacks else None


class _GcAgg:
    __slots__ = ("count", "max", "ms")

    def __init__(self) -> None:
        self.count = 0
        self.ms = 0.0
        self.max = 0.0

    def add(self, ms: float) -> None:
        self.count += 1
        self.ms += ms
        self.max = max(self.max, ms)

    def merge(self, other: _GcAgg) -> None:
        self.count += other.count
        self.ms += other.ms
        self.max = max(self.max, other.max)

    def text(self) -> str | None:
        if not self.count:
            return None
        if self.count == 1:
            return f"gc gen2 {self.ms:.1f}ms"
        return f"gc gen2 {self.ms:.1f}ms x{self.count} (max {self.max:.1f})"


# Gen-2 collections outside any top-level op, owed to the next drag/load/view line.
_gc_pending = _GcAgg()


def _drain_gc() -> None:
    """Routes collected events (GUI thread): each to _recent for stall(), and a
    gen-2 one also to the open top-level op, else to the next drag/load/view line."""
    while _gc_raw:
        end, ms, generation = _gc_raw.popleft()
        _recent.append((end, ms, _op_label(), f"gc gen{generation}"))
        if generation == 2:
            (_top_op.gc if _top_op is not None else _gc_pending).add(ms)


class _TexAgg:
    """Texture work in one line's window: `ms` is the time a caller was
    blocked (its own loads plus waits on prefetched ones); prefetch-pool
    decodes count as files, with their worker ms apart."""

    __slots__ = ("files", "ms", "pool_files", "pool_ms", "wait_ms")

    def __init__(self) -> None:
        self.files = 0
        self.ms = 0.0
        self.wait_ms = 0.0
        self.pool_files = 0
        self.pool_ms = 0.0

    def add(self, kind: str, ms: float) -> None:
        if kind == "load":
            self.files += 1
            self.ms += ms
        elif kind == "wait":
            self.ms += ms
            self.wait_ms += ms
        else:
            self.pool_files += 1
            self.pool_ms += ms

    def merge(self, other: _TexAgg) -> None:
        self.files += other.files
        self.ms += other.ms
        self.wait_ms += other.wait_ms
        self.pool_files += other.pool_files
        self.pool_ms += other.pool_ms

    def text(self) -> str | None:
        files = self.files + self.pool_files
        if not files and not self.ms:
            return None
        text = f"textures {files:+d} files {self.ms:.1f}ms"
        parts = [f"wait {self.wait_ms:.1f}"] if self.wait_ms else []
        if self.pool_files:
            parts.append(f"prefetched {self.pool_files} in {self.pool_ms:.1f}ms")
        return f"{text} ({', '.join(parts)})" if parts else text


# Texture work outside any top-level op, owed to the next drag/load/view line.
_textures_pending = _TexAgg()


class _TextureTimer:
    __slots__ = ("_kind", "_t0")

    def __init__(self, kind: str) -> None:
        self._kind = kind

    def __enter__(self) -> Self:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        _texture_raw.append((self._kind, (time.perf_counter() - self._t0) * 1000))


def texture_load():
    """Times one terrain texture file load on the calling thread (asset_source).
    Safe on any thread: it only appends, and the GUI thread routes it in
    _drain_textures(). texture_wait() and texture_prefetch() likewise."""
    if not _enabled:
        return _NULL_PHASE
    return _TextureTimer("load")


def texture_wait():
    """Times a caller blocked on a prefetched texture load still running."""
    if not _enabled:
        return _NULL_PHASE
    return _TextureTimer("wait")


def texture_prefetch():
    """Times one texture load on the prefetch pool: a file, not blocked time."""
    if not _enabled:
        return _NULL_PHASE
    return _TextureTimer("prefetch")


def _drain_textures() -> None:
    """Routes recorded texture work (GUI thread) to the open top-level op,
    else to the next drag/load/view line."""
    while _texture_raw:
        kind, ms = _texture_raw.popleft()
        (_top_op.textures if _top_op is not None else _textures_pending).add(kind, ms)


def _record(name: str, elapsed_ms: float) -> None:
    global _drag_span_phase_ms
    _recent.append((time.perf_counter(), elapsed_ms, _op_label(), name))
    if _span_stack and _drag_active:
        _drag_span_phase_ms += elapsed_ms
    if name == "repaint":
        _repaint_durations.append(elapsed_ms)
        _schedule_idle()
        return
    if name not in _current_step and name not in _phase_sums:
        _phase_order.append(name)
    _current_step[name] = _current_step.get(name, 0.0) + elapsed_ms


class _PhaseTimer:
    __slots__ = ("_name", "_t0")

    def __init__(self, name: str) -> None:
        self._name = name

    def __enter__(self) -> Self:
        if self._name == "repaint":
            _where_stack.append("repaint")
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        elapsed_ms = (time.perf_counter() - self._t0) * 1000
        if self._name == "repaint":
            _where_stack.pop()
        _record(self._name, elapsed_ms)


def phase(name: str):
    if not _enabled:
        return _NULL_PHASE
    return _PhaseTimer(name)


class _SpanTimer:
    """span()'s timer. `_mark` is where its not-yet-credited wall time starts:
    _credit_open_span() moves it when a drag line prints before it exits."""

    __slots__ = ("_mark", "_name", "_t0")

    def __init__(self, name: str) -> None:
        self._name = name

    def __enter__(self) -> Self:
        self._t0 = self._mark = time.perf_counter()
        _span_stack.append(self)
        return self

    def __exit__(self, *exc) -> None:
        global _drag_wall
        end = time.perf_counter()
        if self in _span_stack:
            _span_stack.remove(self)
        if not _span_stack and _drag_active:
            _drag_wall += (end - self._mark) * 1000
        _recent.append((end, (end - self._t0) * 1000, _op_label(), _SPAN + self._name))


def span(name: str):
    """Wall time of a whole handler, for the drag line's `wall`/`untimed` and
    the stall window; never a phase, so wrapping phases double-counts nothing.
    A span nested in another adds only to the stall window."""
    if not _enabled:
        return _NULL_PHASE
    return _SpanTimer(name)


def _credit_open_span() -> None:
    """Adds the outermost open span's wall so far to the drag: the release
    span is still open when its own stroke-end handler prints the drag line."""
    global _drag_wall
    if _span_stack and _drag_active:
        now = time.perf_counter()
        outer = _span_stack[0]
        _drag_wall += (now - outer._mark) * 1000
        outer._mark = now


class _RepaintPhaseTimer:
    __slots__ = ("_name", "_t0")

    def __init__(self, name: str) -> None:
        self._name = name

    def __enter__(self) -> Self:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        agg = _repaint_phases.setdefault(self._name, [0, 0.0])
        agg[0] += 1
        agg[1] += (time.perf_counter() - self._t0) * 1000


def repaint_phase(name: str):
    """A phase of the paint it runs in, printed in the repaint summary's
    brackets; a no-op outside a repaint, so a shared site (a chunk composite
    reached from patch() as well as from paint()) never lands in a drag step."""
    if not _enabled or not _where_stack or _where_stack[-1] != "repaint":
        return _NULL_PHASE
    return _RepaintPhaseTimer(name)


class _Where:
    __slots__ = ("_tag",)

    def __init__(self, tag: str) -> None:
        self._tag = tag

    def __enter__(self) -> None:
        _where_stack.append(self._tag)

    def __exit__(self, *exc) -> None:
        _where_stack.pop()


def where(tag: str):
    """Tags level events recorded inside it (`margin-warm`, `level-warm`)."""
    if not _enabled:
        return _NULL_PHASE
    return _Where(tag)


def _sprite_delta_since(base: tuple[int, int] | None, now: tuple[int, int]) -> tuple[int, int]:
    if base is None:
        return (0, 0)
    return (now[0] - base[0], now[1] - base[1])


def _sprites_text(files: int, frames: int) -> str | None:
    if not files and not frames:
        return None
    return f"sprites {files:+d} files {frames:+d} frames"


def _take_view_sprites() -> str | None:
    """The cold-sprite delta owed to a drag/load/view line, then re-based."""
    global _sprite_base, _sprite_pending
    now = _sprite_counter()
    files, frames = _sprite_delta_since(_sprite_base, now)
    files += _sprite_pending[0]
    frames += _sprite_pending[1]
    _sprite_base = now
    _sprite_pending = [0, 0]
    return _sprites_text(files, frames)


class _LevelAgg:
    __slots__ = ("count", "ms", "split", "tiles")

    def __init__(self) -> None:
        self.count = 0
        self.ms = 0.0
        self.tiles = 0
        self.split: dict[str, float] = {}

    def add(self, ms: float, tiles: int | None, split: dict[str, float] | None) -> None:
        self.count += 1
        self.ms += ms
        if tiles:
            self.tiles += tiles
        for name, part_ms in (split or {}).items():
            self.split[name] = self.split.get(name, 0.0) + part_ms

    def merge(self, other: _LevelAgg) -> None:
        self.count += other.count
        self.ms += other.ms
        self.tiles += other.tiles
        for name, part_ms in other.split.items():
            self.split[name] = self.split.get(name, 0.0) + part_ms


def _levels_text(events: dict[tuple[str, int, str], _LevelAgg]) -> str | None:
    if not events:
        return None
    parts = []
    for (kind, mip, where_tag), agg in events.items():
        text = f"{mip} {kind} {agg.ms:.1f}ms"
        if agg.count > 1:
            text += f" x{agg.count}"
        if agg.tiles:
            text += f" {agg.tiles} tiles"
        if agg.split:
            text += " [" + " ".join(f"{name} {ms:.1f}" for name, ms in agg.split.items()) + "]"
        parts.append(f"{text} ({where_tag})")
    return "levels: " + ", ".join(parts)


def level_event(
    kind: str, mip: int, ms: float, tiles: int | None = None, split: dict[str, float] | None = None
) -> None:
    """Records one cache-side level build. Qt-free, so render_cache and
    level_warm call it directly; level() below times one for them."""
    global _repaint_level_ms
    if not _enabled:
        return
    where_tag = _where_stack[-1] if _where_stack else "other"
    if where_tag == "repaint":
        _repaint_level_ms += ms
    bucket = _top_op.levels if _top_op is not None else _level_events
    key = (kind, mip, where_tag)
    agg = bucket.get(key)
    if agg is None:
        agg = bucket[key] = _LevelAgg()
    agg.add(ms, tiles, split)
    _recent.append((time.perf_counter(), ms, _op_label(), f"level {mip} {kind} ({where_tag})"))


class _LevelTimer:
    """level()'s timer. The site may change `kind` or set `tiles` inside the
    block, and mark() splits it into named parts."""

    __slots__ = ("_mark", "_t0", "kind", "mip", "ms", "split", "tiles")

    def __init__(self, kind: str, mip: int) -> None:
        self.kind = kind
        self.mip = mip
        self.tiles: int | None = None
        self.split: dict[str, float] | None = None
        self.ms = 0.0

    def __enter__(self) -> Self:
        self._t0 = self._mark = time.perf_counter()
        return self

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        if self.split is None:
            self.split = {}
        self.split[name] = (now - self._mark) * 1000
        self._mark = now

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000
        level_event(self.kind, self.mip, self.ms, self.tiles, self.split)


def level(kind: str, mip: int):
    """Times a level build; yields None while disabled, like phase()."""
    if not _enabled:
        return _NULL_PHASE
    return _LevelTimer(kind, mip)


class _PendingOp:
    """One `perf op` line not yet written: a run of same-label ops."""

    __slots__ = ("gc", "label", "levels", "phases", "sprites", "textures", "totals")

    def __init__(self, label: str) -> None:
        self.label = label
        self.totals: list[float] = []
        self.phases: dict[str, float] = {}
        self.levels: dict[tuple[str, int, str], _LevelAgg] = {}
        self.sprites = [0, 0]
        self.gc = _GcAgg()
        self.textures = _TexAgg()

    def add(self, op: _OpTimer, total_ms: float) -> None:
        self.gc.merge(op.gc)
        self.textures.merge(op.textures)
        self.totals.append(total_ms)
        for name, ms in op.phases.items():
            self.phases[name] = self.phases.get(name, 0.0) + ms
        for key, agg in op.levels.items():
            mine = self.levels.get(key)
            if mine is None:
                mine = self.levels[key] = _LevelAgg()
            mine.merge(agg)
        self.sprites[0] += op.sprites[0]
        self.sprites[1] += op.sprites[1]

    def line(self) -> str:
        n = len(self.totals)
        total = sum(self.totals)
        if n == 1:
            head = f"perf op {self.label}: {total:.0f}ms"
        else:
            head = f"perf op {self.label} x{n}: {total:.0f}ms total, {total / n:.1f}ms mean (max {max(self.totals):.1f})"
        untimed = max(0.0, total - sum(self.phases.values()))
        phases = [f"{name} {ms / n:.1f}" for name, ms in self.phases.items()]
        phases.append(f"untimed {untimed / n:.1f}")
        parts = [head, " ".join(phases)]
        extras = (_sprites_text(*self.sprites), self.textures.text(), _levels_text(self.levels), self.gc.text())
        parts.extend(e for e in extras if e)
        return " | ".join(parts)


def _write_pending_op() -> None:
    global _pending_op
    if _pending_op is not None:
        debug_log.log(_pending_op.line())
        _pending_op = None


def flush_pending_op() -> None:
    """Writes the pending `perf op` line now, for a reader of the log (Help >
    Debug Log). Not gated on _enabled: the line was recorded while tracing."""
    _write_pending_op()


class _OpTimer:
    __slots__ = ("_sprites0", "_t0", "gc", "label", "levels", "phases", "sprites", "textures", "top")

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> Self:
        global _top_op, _current_step, _phase_order, _sprite_base
        # Top-level only outside a drag and any other op; otherwise a phase.
        self.top = not _drag_active and _top_op is None
        # Collections and texture loads from before this op belong to whatever was open then.
        _drain_gc()
        _drain_textures()
        if self.top:
            self.gc = _GcAgg()
            self.textures = _TexAgg()
            if _pending_op is None or _pending_op.label != self.label:
                _write_pending_op()
                if _armed_label is None:
                    _flush_view()
            # Hover pick/highlight since the last flush, not part of this op.
            _current_step = {}
            _phase_order = [name for name in _phase_order if name in _phase_sums]
            self.levels = {}
            now = _sprite_counter()
            gap = _sprite_delta_since(_sprite_base, now)
            _sprite_pending[0] += gap[0]
            _sprite_pending[1] += gap[1]
            self._sprites0 = _sprite_base = now
            _top_op = self
        _op_stack.append(self)
        _where_stack.append(f"op:{self.label}")
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        global _top_op, _current_step, _phase_order, _pending_op, _sprite_base, _last_op
        end = time.perf_counter()
        total_ms = (end - self._t0) * 1000
        _drain_gc()
        _drain_textures()
        # Popped unconditionally: an op that raised must not leave its tag behind.
        _where_stack.pop()
        _op_stack.pop()
        _recent.append((end, total_ms, self.label, None))
        if not self.top:
            _record(self.label, total_ms)
            return
        _last_op = (self.label, end)
        _top_op = None
        self.phases = _current_step
        _current_step = {}
        _phase_order = [name for name in _phase_order if name in _phase_sums]
        now = _sprite_counter()
        self.sprites = _sprite_delta_since(self._sprites0, now)
        _sprite_base = now
        if _pending_op is None:
            _pending_op = _PendingOp(self.label)
        _pending_op.add(self, total_ms)
        _schedule_idle()


class _SameOp:
    """A nested op with the open op's own label: adds nothing."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc) -> None:
        return None


_SAME_OP = _SameOp()


def op(label: str):
    """Brackets a one-shot action for a `perf op` line. The label is slugged
    the way stroke labels are (`Place unit` -> `place-unit`)."""
    if not _enabled:
        return _NULL_PHASE
    label = label.lower().replace(" ", "-")
    if _op_stack and _op_stack[-1].label == label:
        return _SAME_OP
    return _OpTimer(label)


def view_mip(mip: int) -> None:
    """Called by each paint with the mip it drew at; a change shows on the
    next view line as `mip -2->-1`."""
    global _view_mip
    if not _enabled:
        return
    if _view_mip is not None and mip != _view_mip:
        _mip_changes.append((_view_mip, mip))
    _view_mip = mip


def level_warm_tick(
    tick_ms: float, install_ms: Sequence[float], split: dict[str, float] | None = None, gc_ms: float | None = None
) -> None:
    """Records one LevelWarmer tick for the `warm:` bucket, keeping the worst
    tick's split (ms per step label) and its gc ms for the line. Same
    idle-flush restart as warm_tick()."""
    global _level_warm_worst
    if not _enabled:
        return
    _level_warm_ticks.append((tick_ms, len(install_ms), max(install_ms, default=0.0)))
    if _level_warm_worst is None or tick_ms > _level_warm_worst[0]:
        _level_warm_worst = (tick_ms, dict(split or {}), gc_ms)
    _recent.append((time.perf_counter(), tick_ms, _op_label(), "level-warm tick"))
    _schedule_idle()


def _covered_text(
    intervals: list[tuple[float, float]], by_name: dict[str, float], late_ms: float, truncated: bool = False
) -> str:
    """`covered X of Y` plus the top 3 names by ms. X is the union of the
    intervals, so nested entries (gc in a phase, a level in a repaint) count once.
    `truncated` (_recent evicted entries inside the window) prints `covered >= X`."""
    covered = 0.0
    run_start = run_end = None
    for start, end in sorted(intervals):
        if run_end is None or start > run_end:
            if run_end is not None:
                covered += run_end - run_start
            run_start, run_end = start, end
        else:
            run_end = max(run_end, end)
    if run_end is not None:
        covered += run_end - run_start
    text = f"covered {'>= ' if truncated else ''}{covered * 1000:.0f} of {late_ms:.0f}"
    # Rounded so float noise from the window clip can't reorder a tie.
    top = sorted(by_name.items(), key=lambda item: -round(item[1], 3))[:3]
    if top:
        text += ": " + ", ".join(f"{name} {ms:.0f}" for name, ms in top)
    return text


def stall(late_ms: float) -> None:
    """The watchdog's report: its timer fired `late_ms` late. Attributed to
    the outermost op and the most specific phase, span or level event that
    ended inside the stall and covered at least half of it; then the window's
    covered time and top entries (_covered_text), whatever the cause."""
    if not _enabled:
        return
    _drain_gc()
    now = time.perf_counter()
    window_start = now - late_ms / 1000
    best_op = None
    best_detail = None
    gc_in_window = _GcAgg()
    intervals: list[tuple[float, float]] = []
    by_name: dict[str, float] = {}
    for end, ms, label, detail in _recent:
        if end < window_start:
            continue
        if detail == "gc gen2":
            gc_in_window.add(ms)
        if detail is not None and not detail.startswith(_SPAN):
            start = max(end - ms / 1000, window_start)
            intervals.append((start, end))
            by_name[detail] = by_name.get(detail, 0.0) + (end - start) * 1000
        if ms < late_ms / 2:
            continue
        if detail is None:
            if best_op is None or ms > best_op[1]:
                best_op = (label, ms)
        elif best_detail is None or ms < best_detail[1]:
            best_detail = (detail, ms, label)
    if best_op is not None:
        if best_detail is not None and best_detail[2] == best_op[0]:
            cause = f"in op {best_op[0]}: {best_detail[0]}"
        else:
            cause = f"in op {best_op[0]}"
    elif best_detail is not None:
        cause = f"in {best_detail[0]}"
    elif _drag_active or (_drag_end is not None and _drag_end >= window_start):
        cause = "untimed, during a drag"
    elif _last_op is not None:
        cause = f"untimed, last op {_last_op[0]} {(now - _last_op[1]) * 1000:.0f}ms ago"
    else:
        cause = "untimed, no op yet"
    gc_text = gc_in_window.text()
    if gc_text and not cause.endswith("gc gen2"):
        cause += f"; {gc_text}"
    # Full and its oldest entry inside the window: older ones in it were evicted.
    truncated = len(_recent) == _recent.maxlen and _recent[0][0] > window_start
    cause += f"; {_covered_text(intervals, by_name, late_ms, truncated)}"
    debug_log.log(f"perf stall {late_ms:.0f}ms ({cause})")


def warm_tick(tick_ms: float, chunks: int, slowest_chunk_ms: float) -> None:
    """Records one margin-warm tick. Like a repaint, it restarts the idle
    flush outside a drag, so the `perf view` line waits for the warm to go
    quiet instead of splitting it."""
    if not _enabled:
        return
    _warm_ticks.append((tick_ms, chunks, slowest_chunk_ms))
    _recent.append((time.perf_counter(), tick_ms, _op_label(), "margin-warm tick"))
    _schedule_idle()


def warm_worker(kernel_ms: float) -> None:
    """Records one worker-thread chunk kernel, on the GUI thread once its
    result arrives. Same idle-flush restart as warm_tick()."""
    if not _enabled:
        return
    _warm_worker.append(kernel_ms)
    _schedule_idle()


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
    disabled, and also while the current step recorded no phases. A
    mouse-move that only re-touched painted tiles never calls this
    (map_view._touch_tile returns first), so a deferred callback's phase
    recorded before it stays open and joins the next event's step."""
    global _current_step
    if not _enabled or not _current_step:
        return
    _step_totals.append(sum(_current_step.values()))
    for phase_name, elapsed_ms in _current_step.items():
        _phase_sums[phase_name] = _phase_sums.get(phase_name, 0.0) + elapsed_ms
    _current_step = {}


def _repaint_summary() -> str:
    """With repaint_phase()s recorded, their split in brackets: ms and calls
    each, then `other`, the repaints' time outside those and the level events."""
    r = _repaint_durations
    text = f"repaint: {len(r)} calls, {sum(r):.0f}ms total (max {max(r):.1f})"
    if not _repaint_phases:
        return text
    parts = [f"{name} {ms:.1f} x{calls}" for name, (calls, ms) in _repaint_phases.items()]
    timed = sum(ms for _calls, ms in _repaint_phases.values()) + _repaint_level_ms
    parts.append(f"other {max(0.0, sum(r) - timed):.1f}")
    return f"{text} [{'; '.join(parts)}]"


def _clear_repaints() -> None:
    global _repaint_durations, _repaint_phases, _repaint_level_ms
    _repaint_durations = []
    _repaint_phases = {}
    _repaint_level_ms = 0.0


def _warm_summary() -> str:
    """`max chunk` is GUI-thread time per chunk; for a worker chunk that is
    its preparation only, with the kernel under `worker`. `level` is
    LevelWarmer's ticks, with the slowest tick's split and its installs'
    count and slowest."""
    w, k, lw = _warm_ticks, _warm_worker, _level_warm_ticks
    parts = []
    if w:
        parts.append(
            f"{len(w)} ticks, {sum(t[1] for t in w)} chunks, {sum(t[0] for t in w):.0f}ms total "
            f"(max tick {max(t[0] for t in w):.1f}, max chunk {max(t[2] for t in w):.1f})"
        )
    if k:
        parts.append(f"worker {len(k)} chunks, {sum(k):.0f}ms total (max {max(k):.1f})")
    if lw:
        parts.append(
            f"level {len(lw)} ticks, max tick {max(t[0] for t in lw):.1f}{_worst_tick_text()}, "
            f"installs {sum(t[1] for t in lw)} (max {max(t[2] for t in lw):.1f})"
        )
    return "warm: " + ", ".join(parts)


def _worst_tick_text() -> str:
    """The slowest level-warm tick's split: its steps, `other` for the rest
    of the tick, and its gc ms, which nests inside the steps."""
    if _level_warm_worst is None:
        return ""
    tick_ms, split, gc_ms = _level_warm_worst
    parts = [f"{name} {ms:.1f}" for name, ms in split.items()]
    parts.append(f"other {max(0.0, tick_ms - sum(split.values())):.1f}")
    if gc_ms is not None:
        parts.append(f"gc {gc_ms:.1f}")
    return " [" + "; ".join(parts) + "]"


def _take_extras() -> list[str]:
    """The level events, cold sprites and mip changes owed to the next drag,
    load or view line, cleared once taken."""
    global _level_events, _mip_changes, _gc_pending, _textures_pending
    _drain_gc()
    _drain_textures()
    extras = []
    levels = _levels_text(_level_events)
    if levels:
        extras.append(levels)
    sprites = _take_view_sprites()
    if sprites:
        extras.append(sprites)
    textures = _textures_pending.text()
    if textures:
        extras.append(textures)
    _textures_pending = _TexAgg()
    if _mip_changes:
        extras.append("mip " + " ".join(f"{a}->{b}" for a, b in _mip_changes))
    gc_text = _gc_pending.text()
    if gc_text:
        extras.append(gc_text)
    _level_events = {}
    _mip_changes = []
    _gc_pending = _GcAgg()
    return extras


def _flush_view() -> None:
    """Logs pending repaints and warm ticks as a `perf view` line and clears
    only those (plus the extras it prints)."""
    global _warm_ticks, _warm_worker, _level_warm_ticks, _level_warm_worst
    _drain_gc()
    parts = []
    if _repaint_durations:
        parts.append(_repaint_summary())
    if _warm_ticks or _warm_worker or _level_warm_ticks:
        parts.append(_warm_summary())
    if parts or _level_events or _gc_pending.count:
        parts.extend(_take_extras())
        debug_log.log(f"perf view: {', '.join(parts)}, composite {composite_backend.active_backend()}")
        _clear_repaints()
        _warm_ticks = []
        _warm_worker = []
        _level_warm_ticks = []
        _level_warm_worst = None


def begin_drag() -> None:
    """Called when a stroke starts, before its first tile is touched. The
    press span is already open: its wall from before this call counts."""
    global _drag_active, _current_step, _phase_order, _drag_wall, _drag_span_phase_ms
    if not _enabled:
        return
    _write_pending_op()
    if _armed_label is None:
        _flush_view()
    # Hover pick/highlight since the last flush, not part of this drag.
    _current_step = {}
    _phase_order = [name for name in _phase_order if name in _phase_sums]
    _drag_wall = 0.0
    _drag_span_phase_ms = 0.0
    _drag_active = True


def end_drag(label: str) -> None:
    """Called after the stroke-end callback. Flushes under `label` whatever
    that callback didn't (Convert and Cliff strokes flush nothing themselves)."""
    global _drag_active, _drag_end
    if not _enabled:
        return
    _credit_open_span()
    _drag_active = False
    _drag_end = time.perf_counter()
    if _step_totals:
        flush(label)


def flush_idle() -> None:
    """The idle timer's target: repaints recorded outside any drag or armed
    load become a `perf view` line. Hover phases are left for begin_drag()."""
    if not _enabled or _drag_active or _armed_label is not None:
        return
    _write_pending_op()
    _flush_view()


def flush(label: str) -> None:
    """Emits the accumulated drag's summary to debug_log.log() -- deliberately
    NOT _log_status(), so a drag doesn't spam the visible status pane -- then
    resets all state for the next drag. A no-op if no step was ever closed
    (e.g. a click with no drag).

    With spans recorded, the header carries `wall X, untimed Y`: the drag's
    span wall time (press, steps, and the release up to this call), and that
    minus the phases recorded inside those spans."""
    global _current_step, _step_totals, _phase_sums, _phase_order, _drag_wall, _drag_span_phase_ms
    if not _enabled:
        return
    _credit_open_span()
    n = len(_step_totals)
    if n == 0 and not _repaint_durations:
        _current_step = {}
        return
    _write_pending_op()
    extras = _take_extras()
    lines = []
    # Last on the header line, so tester traces say which composite path ran.
    backend = f", composite {composite_backend.active_backend()}"
    if n:
        total = sum(_step_totals)
        mean_step = total / n
        max_step = max(_step_totals)
        phase_line = " ".join(
            f"{name} {_phase_sums[name] / n:.1f}" for name in _phase_order if name in _phase_sums
        )
        wall = ""
        if _drag_wall:
            wall = f", wall {_drag_wall:.0f}ms, untimed {max(0.0, _drag_wall - _drag_span_phase_ms):.0f}ms"
        lines.append(
            f"perf drag {label}: {n} steps, {total:.0f}ms total, {mean_step:.1f}ms/step (max {max_step:.1f})"
            f"{wall}{backend}"
        )
        lines.append(f"  | {phase_line}")
        if _current_step:
            # Phases after the last step(): the stroke-end handler's own split.
            lines.append("  | end: " + " ".join(f"{name} {ms:.1f}" for name, ms in _current_step.items()))
    if _repaint_durations:
        if n:
            lines.append(f"  | {_repaint_summary()}")
        else:
            extra = "".join(f", {e}" for e in extras)
            lines.append(f"perf {label}: {_repaint_summary()}{extra}{backend}")
            extras = []
    lines.extend(f"  | {e}" for e in extras)
    debug_log.log("\n".join(lines))
    _current_step = {}
    _step_totals = []
    _phase_sums = {}
    _phase_order = []
    _drag_wall = 0.0
    _drag_span_phase_ms = 0.0
    _clear_repaints()


def _fresh_state() -> dict[str, object]:
    """Every recorded-state global at its empty value (not _enabled or the
    installed hooks). Test fixtures reset from this, so a new global can't
    be missed at one of several hand-kept reset sites."""
    return {
        "_current_step": {},
        "_step_totals": [],
        "_phase_sums": {},
        "_phase_order": [],
        "_repaint_durations": [],
        "_repaint_phases": {},
        "_repaint_level_ms": 0.0,
        "_warm_ticks": [],
        "_warm_worker": [],
        "_armed_label": None,
        "_drag_active": False,
        "_op_stack": [],
        "_top_op": None,
        "_pending_op": None,
        "_where_stack": [],
        "_level_events": {},
        "_level_warm_ticks": [],
        "_level_warm_worst": None,
        "_mip_changes": [],
        "_view_mip": None,
        "_sprite_base": None,
        "_sprite_pending": [0, 0],
        "_recent": deque(maxlen=_RECENT_MAXLEN),
        "_last_op": None,
        "_drag_end": None,
        "_span_stack": [],
        "_drag_wall": 0.0,
        "_drag_span_phase_ms": 0.0,
        "_gc_raw": deque(),
        "_gc_t0": None,
        "_gc_total_ms": 0.0,
        "_gc_pending": _GcAgg(),
        "_texture_raw": deque(),
        "_textures_pending": _TexAgg(),
    }


def reset() -> None:
    """Drops everything recorded, written or not. For tests."""
    globals().update(_fresh_state())
