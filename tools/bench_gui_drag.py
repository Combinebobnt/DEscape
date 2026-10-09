#!/usr/bin/env python3
"""Drag-stroke latency and stroke continuity through REAL X input events.
Informational only, never pass/fail.

Why real events: Qt's xcb backend compresses queued mouse motion, and MapView
now coalesces moves itself too (one handled move per paint, GH #179), so a
slow stroke handler sees cursor positions several tiles apart. QTest and
sendEvent() bypass the X queue entirely, which is why this needs a real X
server (Xvfb is fine) and xdotool rather than an offscreen window.

What it reports, per drag:
  - steps, paints, and steps per paint (1.00 means every paint follows one
    stroke step: the motion queue is being compressed, not piling up).
  - cursor-tile gap: Chebyshev distance between successive cursor tiles
    handed to the viewer. Above 1 means xcb's compression or MapView's
    coalescing skipped tiles.
  - delivered-tile continuity: for each tile the viewer was asked to paint,
    the Chebyshev distance to the nearest tile delivered before it. All 1
    means the stroke is continuous (MapView gap-fills Draw/Elevate/Set
    Elevation/Convert/Cliff). Anything above 1 is a hole in the stroke.
  - units converted and placed by the drag (Convert: only non-zero if the
    drag rows cross units not already --convert-to's; Cliff places at release).
    For Convert and Cliff "steps per paint" means less: Convert repaints on a
    throttle, Cliff only at release.
  - the Perf Trace line for the drag (ms/step, per-phase totals, repaint).

Settings are isolated (pin_install_path() + isolate_settings()), so nothing
here writes the real config.yaml.

One shell command, Xvfb on TCP (works where Unix-socket X access isn't
available), killed by its own PID. Never `pkill -f "Xvfb :N"`: that also matches the
invoking shell's own command line and kills it.

    Xvfb :57 -listen tcp -nolisten unix -screen 0 1920x1200x24 & XPID=$!; sleep 1; \\
    DISPLAY=127.0.0.1:57 .venv/bin/python3 tools/bench_gui_drag.py --style Stepped --brush 1,5,9; \\
    kill $XPID; wait $XPID

The `wait` matters: without it the call can end before Xvfb removes
/tmp/.X57-lock, and that stale lock holds a small PID which exists again in
the next sandboxed call, so Xvfb refuses the display as "already active".
If it does, pick a display with no /tmp/.X<N>-lock.

--profile-patch prints one line per cache patch() in the drag: bbox size,
elevation-changed tiles, what the elevation splice decided (units spliced and
how many were seeds rather than shared-tile co-occupants, or why it fell
back), source gen before/after, resident levels, and
time in _refresh_source_caches, per-level rebuilds (_level() on a stale level)
and per-level compositing. Its `gc` column is time in cyclic-GC collections
during that patch (via gc.callbacks), tagged by generation; it is nested inside
whichever phase the collection landed in, not additive with them. A per-drag
`gc during drag` line counts every collection in the drag window, in a patch
or not, so a collection deferred past the stroke still shows. --no-preload
turns off neighbouring zoom-level preload, for an A/B against the default.

--clicks replaces the per-brush drags with the Set elevation stall repro
(the sequence from a user stress log): at fit zoom, a Units visit so the pick index is
live, back to Terrain, one large Set elevation drag (--clicks-rows rows at
--clicks-brush), Ctrl+Z, then 12 rapid 2-step clicks 0.15-0.3 s apart. It
prints the tiles the drag wrote, every `perf stall` line, every drag header
(`wall`, `untimed`) and then the sequence's whole debug log. --style and
--scenario apply; --tool, --brush and --delays do not. The plan's second
shape is --clicks-no-undo --clicks-scope all; outlines stay off either way,
so the scope only matters where the hidden footprint rebuild still runs.

--undo is the large-elevation undo probe: at --undo-zoom x fit, one stroke
per tool in --undo-tools over a centred square of about --undo-tiles tiles,
driven through the window's stroke handlers (rows of cursor tiles, not X
input, so the tile set is the same at every zoom), then a timed
window.undo(), --undo-reps times. Per undo it prints the op line, the
bbox's phase split (a timed copy of _dirty_screen_bbox, checked against the
real rect), the early-exit over-estimate, and the uncapped elevation splice
of the undo's elevation-changed tiles on the visible level, part by part,
against a full grid and pack build. --undo-plain times the undo alone, for an
A/B against another checkout with --root.

--fit drags the per-brush strokes at fit zoom (the zoom a load opens at)
instead of the 1:1 pin, on rows clipped to the map diamond. --trees leaves
Trees on (Eye candy keeps its default, off) instead of forcing both off.
--window WxH sizes the main window (default 1600x1000).

--flow gh180 is GH #180's tester flow, Trees on: load --scenario straight
into --style (the style is set before the load, as when a file is opened
while Sloped is showing), keep the fit the load opened at, settle, a
--flow-brush Draw drag of --flow-draw-rows rows painting --flow-terrain,
wait for the repaint, Ctrl+Z, a Set elevation drag of --flow-elev-rows rows
two levels off the map's most common one, Ctrl+Z. It prints the whole debug
log (every perf line) and one `gh180 gates:` summary line: G1 Draw ms/step,
G2 release handler plus the next paint, the post-stroke composite and mip,
G3 Set elevation ms/step, G4 each undo plus its next paint (with the side
of STEPPED_FULL_RERENDER_THRESHOLD it took), G5 the load's first-paint
total, G6 the slowest single step. On old-allies-final-v2 the default
--flow-draw-rows remove over UNIT_SPLICE_MAX_UNITS trees (the tester's
wholesale stroke end) and undo below the 20k-tile re-render threshold, and
the default --flow-elev-rows undo above it with margin (slow, coalesced Set
elevation steps cut row corners and write fewer tiles), so one run covers
both sides.

--root DIR imports descape and testkit from another checkout, so one copy of
this script drives a baseline worktree and a lane branch alike.
"""

from __future__ import annotations

import argparse
import bisect
import gc
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SCENARIO = ROOT / "examples" / "0_June_Event_Scenario.aoe2scenario"
# --flow gh180: BEACH_WET_ROCK (the tester's terrain); row counts per the docstring's
# --flow-draw-rows/--flow-elev-rows paragraph.
FLOW_TERRAIN = 109
FLOW_DRAW_ROWS = 4
FLOW_ELEV_ROWS = 9


def _pump(seconds: float) -> None:
    from PyQt5.QtWidgets import QApplication

    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def _chebyshev(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _hist(values) -> str:
    counts = Counter(values)
    return " ".join(f"{k}:{counts[k]}" for k in sorted(counts)) or "-"


def _splice_fallback_reason(render_cache, original, scenario, units_by_tile, unit_filter, tiles, cap=None) -> str:
    """Why _elevation_splices() returned None, via `original` re-run with no cap.
    `cap` is the one the caller passed (Sloped's), else Stepped's module cap."""
    name = "_ELEV_SPLICE_MAX_UNITS" if cap is None else "the caller's cap"
    cap = render_cache._ELEV_SPLICE_MAX_UNITS if cap is None else cap
    out = original(scenario, units_by_tile, unit_filter, tiles, 10**9)
    if out is not None and len(out) > cap:
        return f"over {name} ({cap}): component would be {len(out)}"
    return "unknown (a unit missing from the own-tile index)"


class _PatchProfile:
    """Wraps the chunk caches' patch path to log what one stroke step costs."""

    def __init__(self) -> None:
        from descape import render_cache

        self.rc = render_cache
        self.lines: list[str] = []
        self._current: dict | None = None
        self._saved: list[tuple[object, str, object]] = []
        self._gc_t0: float | None = None
        self._gc_events: list[tuple[int, float, bool]] = []  # (generation, ms, inside a patch)

    def _on_gc(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._gc_t0 = time.perf_counter()
            return
        if self._gc_t0 is None:
            return
        ms = (time.perf_counter() - self._gc_t0) * 1000
        self._gc_t0 = None
        self._gc_events.append((info["generation"], ms, self._current is not None))
        if self._current is not None:
            self._current["gc"].append((info["generation"], ms))

    def _wrap(self, owner, name: str, make) -> None:
        original = owner.__dict__[name] if name in owner.__dict__ else getattr(owner, name)
        self._saved.append((owner, name, original))
        setattr(owner, name, make(original))

    def install(self) -> None:
        rc = self.rc
        prof = self

        def patch(original):
            # *args/**kwargs so it runs on builds with and without patch()'s newer parameters.
            def wrapper(cache, bbox, *args, **kwargs):
                if prof._current is not None:
                    return original(cache, bbox, *args, **kwargs)
                elevation_changed = args[0] if args else kwargs.get("elevation_changed")
                gen0 = getattr(cache, "_source_gen", None)
                resident = cache.resident_levels()
                chunks0 = len(cache._cache)
                prof._current = cur = {"refresh": 0.0, "rebuild": {}, "composite": {}, "splice": "-", "gc": []}
                t0 = time.perf_counter()
                try:
                    return original(cache, bbox, *args, **kwargs)
                finally:
                    total = (time.perf_counter() - t0) * 1000
                    prof._current = None
                    x0, y0, x1, y1 = bbox
                    n = "None" if elevation_changed is None else len(elevation_changed)
                    rebuild = " ".join(f"{m}:{ms:.1f}" for m, ms in sorted(cur["rebuild"].items())) or "-"
                    comp = " ".join(f"{m}:{ms:.1f}" for m, ms in sorted(cur["composite"].items())) or "-"
                    rebuild_levels = kwargs.get("rebuild_levels")
                    gcs = cur["gc"]
                    gc_col = (
                        f"{sum(ms for _, ms in gcs):.1f}[{','.join(f'g{g}' for g, _ in gcs)}]" if gcs else "-"
                    )
                    prof.lines.append(
                        f"    patch {total:6.1f}ms  bbox {x1 - x0}x{y1 - y0}  elev_changed {n}  splice {cur['splice']}"
                        f"  gen {gen0}->{getattr(cache, '_source_gen', None)}  resident {resident}"
                        f"  rebuild_levels {rebuild_levels}  evicted {chunks0 - len(cache._cache)}"
                        f"  | gc {gc_col}  refresh {cur['refresh']:.1f}  rebuild {rebuild}  composite(incl. rebuild) {comp}"
                    )
            return wrapper

        def refresh(original):
            def wrapper(cache, elevation_changed=None, **kwargs):
                t0 = time.perf_counter()
                try:
                    return original(cache, elevation_changed, **kwargs)
                finally:
                    if prof._current is not None:
                        prof._current["refresh"] += (time.perf_counter() - t0) * 1000
            return wrapper

        def splices(original):
            def wrapper(scenario, units_by_tile, unit_filter, tiles, cap=None):
                out = original(scenario, units_by_tile, unit_filter, tiles, cap)
                if prof._current is not None:
                    if out is None:
                        why = _splice_fallback_reason(rc, original, scenario, units_by_tile, unit_filter, tiles, cap)
                        prof._current["splice"] = f"None[{why}]"
                    else:
                        seeds = sum(1 for s in out if s.new_own_tile in tiles)
                        prof._current["splice"] = f"{len(out)} units ({seeds} seeds)"
                return out
            return wrapper

        def level(original):
            def wrapper(cache, mip):
                stale = cache._levels[mip].gen != cache._source_gen
                t0 = time.perf_counter()
                try:
                    return original(cache, mip)
                finally:
                    if stale and prof._current is not None:
                        rebuild = prof._current["rebuild"]
                        rebuild[mip] = rebuild.get(mip, 0.0) + (time.perf_counter() - t0) * 1000
            return wrapper

        def composite(original):
            def wrapper(cache, mip, x0, y0, x1, y1):
                t0 = time.perf_counter()
                try:
                    return original(cache, mip, x0, y0, x1, y1)
                finally:
                    if prof._current is not None:
                        comp = prof._current["composite"]
                        comp[mip] = comp.get(mip, 0.0) + (time.perf_counter() - t0) * 1000
            return wrapper

        self._wrap(rc._ChunkCacheBase, "patch", patch)
        self._wrap(rc, "_elevation_splices", splices)
        for cls in (rc.IsoChunkCache, rc.SlopedChunkCache, rc.FlatChunkCache):
            self._wrap(cls, "_refresh_source_caches", refresh)
            self._wrap(cls, "_composite_rect", composite)
        self._wrap(rc.IsoChunkCache, "_level", level)
        gc.callbacks.append(self._on_gc)

    def take(self) -> list[str]:
        """This drag's patch lines plus its gc summary; also clears both for the next drag."""
        events, self._gc_events = self._gc_events, []
        if events:
            gens = _hist(g for g, _, _ in events)
            in_patch = sum(1 for _, _, inside in events if inside)
            summary = (
                f"    gc during drag: {len(events)} collections (gen {gens}), {in_patch} inside a patch,"
                f" total {sum(ms for _, ms, _ in events):.1f}ms, max {max(ms for _, ms, _ in events):.1f}ms"
            )
        else:
            summary = "    gc during drag: none"
        out, self.lines = [*self.lines, summary], []
        return out


def _unit_owners(window) -> dict[int, int]:
    """id(unit) -> owning player, so a drag's converted and placed units can be counted."""
    return {id(u): p for p, units in enumerate(window.scenario.unit_manager.units) for u in units}


def _drag(
    window, record, paints, *, row: int, px: int, delay_ms: float, button: str, profile=None,
    start: tuple[int, int] | None = None, length: int = 1000,
) -> list[str]:
    """One horizontal drag of `length` px; `start` is its viewport point (default: row `row` of the 1:1 grid)."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    from descape import debug_log

    view = window.map_view
    origin = view.viewport().mapToGlobal(QPoint(0, 0))
    sx, sy = start if start is not None else (100, 150 + row * 150)
    x0, y0 = origin.x() + sx, origin.y() + sy
    cmd = ["xdotool", "mousemove", str(x0), str(y0), "sleep", "0.3", "mousedown", button]
    for i in range(1, length // px + 1):
        # A 3px zig-zag, so the drag is not perfectly axis-aligned.
        cmd += ["sleep", str(delay_ms / 1000), "mousemove", str(x0 + i * px), str(y0 + (i % 2) * 3)]
    cmd += ["sleep", "0.3", "mouseup", button]

    _pump(0.5)
    record.clear()
    paints.clear()
    debug_log.clear()
    if profile is not None:
        profile.take()
    owners0 = _unit_owners(window)
    proc = subprocess.Popen(cmd)
    t0 = time.perf_counter()
    while proc.poll() is None:
        QApplication.processEvents()
    _pump(0.3)
    t1 = time.perf_counter()

    step_times = sorted(t for t, _ in record)
    paint_times = [t for t in paints if t0 <= t <= t1]
    per_paint = []
    prev = t0
    for pt in paint_times:
        n = bisect.bisect_right(step_times, pt) - bisect.bisect_right(step_times, prev)
        if n:
            per_paint.append(n)
        prev = pt

    cursor = [tiles[-1] for _, tiles in record]
    cursor_gaps = [_chebyshev(a, b) for a, b in pairwise(cursor)]
    seen: list[tuple[int, int]] = []
    continuity = []
    for _, tiles in record:
        for tile in tiles:
            if seen:
                continuity.append(min(_chebyshev(tile, s) for s in seen))
            seen.append(tile)

    owners1 = _unit_owners(window)
    converted = sum(1 for k, p in owners1.items() if k in owners0 and owners0[k] != p)
    placed = sum(1 for k in owners1 if k not in owners0)

    mean = sum(per_paint) / len(per_paint) if per_paint else 0.0
    return [
        f"  units converted {converted}, placed {placed}",
        (
            f"  wall {t1 - t0:.2f}s, steps {len(record)}, paints {len(paint_times)}, "
            f"steps per paint mean {mean:.2f} max {max(per_paint, default=0)}"
        ),
        f"  cursor-tile gap hist {_hist(cursor_gaps)} (skipped >1: {sum(g > 1 for g in cursor_gaps)})",
        f"  delivered-tile continuity hist {_hist(continuity)} (holes: {sum(c > 1 for c in continuity)})",
        "  " + debug_log.get_log_text().strip().replace("\n", "\n  "),
        *(profile.take() if profile is not None else ()),
    ]


def _run(args) -> list[str]:
    from PyQt5.QtGui import QTransform

    from descape import composite_backend, perf_trace, settings, viewer_canvas
    from testkit import qt_window

    # Trees/eye candy/ticks off unless --trees: they are not what the default run measures.
    # Module globals, per testkit/qt_window.py. A drag never opens the Large-fill modal.
    settings._paint_trees = args.trees
    settings._paint_eye_candy = False
    settings._distance_ticks = False
    if args.no_preload:
        settings._preload_zoom_levels = False
    perf_trace.enable(True)
    profile = None
    if args.profile_patch:
        profile = _PatchProfile()
        profile.install()

    paints: list[float] = []
    original_paint = viewer_canvas.MapCanvasItem.paint

    def timed_paint(self, *a, **k):
        original_paint(self, *a, **k)
        paints.append(time.perf_counter())

    viewer_canvas.MapCanvasItem.paint = timed_paint

    window = qt_window.stepped_window(Path(args.scenario), show=False)
    output = []
    try:
        # Convert needs the unit pick index, which only Units mode builds.
        window.mode_combo.setCurrentText("Units" if args.tool == "convert" else "Terrain")
        if args.style != "Stepped":
            window.terrain_style_combo.setCurrentText(args.style)
        window._on_tool_selected(args.tool)
        if args.tool == "convert" and not window.units_panel.select_owner(args.convert_to):
            raise SystemExit(f"--convert-to {args.convert_to}: no such owner in the Units panel")
        if args.tool == "cliff":
            _select_cliff_piece(window, args.cliff_piece)
        window.paint_trees_check.setChecked(args.trees)
        window.paint_eye_candy_check.setChecked(False)
        window.move(0, 0)
        window.resize(*args.window)
        window.show()
        view = window.map_view
        _pump(0.2)
        if args.fit:
            _fit(view)
        else:
            view.setTransform(QTransform())  # 1:1, mip 0
            view.centerOn(view.sceneRect().center())
        _pump(args.settle)
        fit_rows = _diamond_rows(view, window.scenario.map_manager, 4) if args.fit else None

        # MapView captured the bound callback at construction, so wrap its copy.
        record: list[tuple[float, list[tuple[int, int]]]] = []
        deliver = view._on_stroke_tiles

        def recording(tiles, modifiers):
            deliver(tiles, modifiers)
            record.append((time.perf_counter(), list(tiles)))

        view._on_stroke_tiles = recording
        vp = view.viewport()
        output.append(f"{args.style}, {args.tool}, viewport {vp.width()}x{vp.height()}, scale {view.transform().m11():g}")
        output.append(composite_backend.describe())
        output.append(
            f"preload zoom levels {'off' if args.no_preload else 'on'}, sprites {window._sprites_enabled},"
            f" trees {window.paint_trees_check.isChecked()}, visible mip {_visible_mip(view)}"
        )
        row = 0
        for brush_size in args.brush:
            window.brush_size_spin.setValue(brush_size)
            for delay in args.delays:
                output.append(f"--- brush {brush_size}, {args.px}px/move, {delay:g}ms between moves")
                geometry = {}
                if fit_rows is not None:
                    rows, span = fit_rows
                    lo, hi = span(rows[row % 4])
                    geometry = {"start": (lo, rows[row % 4]), "length": min(1000, hi - lo)}
                output += _drag(
                    window, record, paints, row=row % 4, px=args.px, delay_ms=delay, button=args.button, profile=profile,
                    **geometry,
                )
                row += 1
    finally:
        window.edit_history.mark_saved()
        window.close()
    return output


def _select_cliff_piece(window, unit_const: int) -> None:
    from descape import cliff_catalog

    families = cliff_catalog.families()
    name = next((n for n, pieces in families.items() if any(p.unit_const == unit_const for p in pieces)), None)
    if name is None:
        raise SystemExit(f"--cliff-piece {unit_const}: not a cliff piece")
    window.cliff_family_combo.setCurrentIndex(window.cliff_family_combo.findData(name))
    window.cliff_piece_combo.setCurrentIndex(window.cliff_piece_combo.findData(unit_const))


def _fit(view) -> None:
    """The fit a load opens at (MapView fits its map rect), applied again."""
    from PyQt5.QtCore import Qt

    view.fitInView(view._map_rect if view._map_rect is not None else view.sceneRect(), Qt.KeepAspectRatio)


def _visible_mip(view):
    target = view.viewport_chunk_target()
    return None if target is None else target[0]


def _diamond_rows(view, mm, n_rows: int, extent: float = 0.6):
    """`n_rows` viewport y's spread over +-extent of the map diamond's half
    height, and span(y) -> (x_lo, x_hi), that row clipped to 90% of the
    diamond. Stepped/Sloped draw the map as a diamond, so a press in the
    fitted rect's corner is off the map and opens no stroke."""
    corners = [
        view.mapFromScene(view._tile_polygon(x, y).boundingRect().center())
        for x, y in ((0, 0), (mm.map_width - 1, 0), (0, mm.map_height - 1), (mm.map_width - 1, mm.map_height - 1))
    ]
    left, right = min(p.x() for p in corners), max(p.x() for p in corners)
    top, bottom = min(p.y() for p in corners), max(p.y() for p in corners)
    cx, cy, hw, hh = (left + right) / 2, (top + bottom) / 2, (right - left) / 2, (bottom - top) / 2

    def span(y: int) -> tuple[int, int]:
        half = 0.9 * hw * max(0.0, 1 - abs(y - cy) / hh)
        return int(cx - half), int(cx + half)

    rows = [int(cy + hh * (-extent + 2 * extent * r / max(1, n_rows - 1))) for r in range(n_rows)]
    return rows, span


def _serpentine_cmd(origin, rows, span, px: int, delay_s: float) -> list[str]:
    """xdotool args for one left-button drag along `rows`, alternating direction, `px` per move."""
    cmd = ["mousemove", str(origin.x() + span(rows[0])[0]), str(origin.y() + rows[0]), "sleep", "0.3", "mousedown", "1"]
    for r, y in enumerate(rows):
        lo, hi = span(y)
        xs = range(lo, hi + 1, px) if r % 2 == 0 else range(hi, lo - 1, -px)
        for x in xs:
            cmd += ["sleep", f"{delay_s:g}", "mousemove", str(origin.x() + x), str(origin.y() + y)]
    return [*cmd, "sleep", "0.3", "mouseup", "1"]


# --clicks' pause after each click, seconds: the stress log's 0.15-0.3 s cadence.
_CLICK_SPACING_S = (0.15, 0.3, 0.2, 0.25) * 3


def _xdotool(cmd: list[str]) -> None:
    """One xdotool sequence, with the GUI thread processing events throughout."""
    from PyQt5.QtWidgets import QApplication

    proc = subprocess.Popen(["xdotool", *cmd])
    while proc.poll() is None:
        QApplication.processEvents()


def _run_clicks(args) -> list[str]:
    """--clicks: the Set elevation stall repro, shaped like the stress log that
    found it. A Units visit (the pick index goes live), back to Terrain, one
    large Set elevation drag at fit zoom, Ctrl+Z, then 12 rapid 2-step clicks.
    Prints every stall line and drag header (wall, untimed), then the whole
    debug log of the sequence for attribution."""
    import re
    from collections import Counter

    from PyQt5.QtCore import QPoint, Qt

    from descape import composite_backend, debug_log, perf_trace, settings
    from testkit import qt_window

    settings._paint_trees = False
    settings._paint_eye_candy = False
    settings._distance_ticks = False
    # Read by MapView's constructor. Outlines stay off (the user's config), so the scope only
    # changes work where the hidden rebuild still runs (the plan's "before" side).
    settings._footprint_outlines = False
    settings._footprint_scope = args.clicks_scope
    perf_trace.enable(True)  # before the window: its constructor starts the stall watchdog and gc hook
    window = qt_window.stepped_window(Path(args.scenario), show=False)
    output = []
    try:
        if args.style != "Stepped":
            window.terrain_style_combo.setCurrentText(args.style)
        window.move(0, 0)
        window.resize(1600, 1000)
        window.show()
        view = window.map_view
        _pump(0.2)
        view.fitInView(view.sceneRect(), Qt.KeepAspectRatio)
        _pump(args.settle)

        window.mode_combo.setCurrentText("Units")
        _pump(1.0)
        if view._unit_index is None:
            raise SystemExit("--clicks: Units mode built no pick index")
        units = len(view._unit_index.entries)
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("set_level")
        window.brush_size_spin.setValue(args.clicks_brush)
        mm = window.scenario.map_manager
        common = Counter(tile.elevation for tile in mm.terrain).most_common(1)[0][0]
        top = window.elevation_level_spin.maximum()
        level = common + 2 if common + 2 <= top else max(0, common - 2)
        window.elevation_level_spin.setValue(level)
        _pump(1.0)

        vp = view.viewport()
        origin = vp.mapToGlobal(QPoint(0, 0))
        w, h = vp.width(), vp.height()
        rows, span = _diamond_rows(view, mm, args.clicks_rows)
        cmd = _serpentine_cmd(origin, rows, span, 40, 0.016)

        debug_log.clear()
        cursor0 = window.edit_history.cursor
        _xdotool(cmd)
        _pump(0.3)
        if window.edit_history.cursor != cursor0 + 1:
            raise SystemExit(
                f"--clicks: the large drag pushed no stroke record (cursor {cursor0}->{window.edit_history.cursor}, "
                f"tool {view._tool}, mode {window.mode}, scale {view.transform().m11():.3f}, viewport {w}x{h})\n"
                + debug_log.get_log_text()
            )
        tiles = len(window.edit_history.records[window.edit_history.cursor - 1].touched_indices())

        # Ctrl+Z as the user pressed it; a direct undo only if the key never reached the window.
        if args.clicks_no_undo:
            # The clicks land on tiles the drag just set, so they set the old level back to write anything.
            window.elevation_level_spin.setValue(common)
            undo_via = f"skipped (--clicks-no-undo; clicks set level {common})"
        else:
            window.activateWindow()
            window.raise_()
            centre = (origin.x() + w // 2, origin.y() + h // 2)
            _xdotool(["mousemove", str(centre[0]), str(centre[1]), "key", "ctrl+z"])
            _pump(0.3)
            undo_via = "ctrl+z"
            if window.edit_history.cursor != cursor0:
                window.undo()
                undo_via = "direct window.undo() (ctrl+z did not reach the window)"

        cmd = []
        for i, pause in enumerate(_CLICK_SPACING_S):
            row = rows[i % len(rows)]
            lo, hi = span(row)
            x = origin.x() + lo + (hi - lo - args.px) * i // (len(_CLICK_SPACING_S) - 1)
            y = origin.y() + row
            cmd += ["mousemove", str(x), str(y), "mousedown", "1", "sleep", "0.03"]
            cmd += ["mousemove", str(x + args.px // 2), str(y), "sleep", "0.03", "mouseup", "1", "sleep", str(pause)]
        _xdotool(cmd)
        _pump(1.0)
        perf_trace.flush_idle()

        text = debug_log.get_log_text()
        lines = [line.split("] ", 1)[-1] for line in text.splitlines()]
        stalls = [line for line in lines if line.startswith("perf stall")]
        drags = [line for line in lines if line.startswith("perf drag")]
        sizes = [re.match(r"perf stall (\d+)ms", line).group(1) for line in stalls]
        untimed = sum(1 for line in stalls if "(untimed" in line)
        output.append(
            f"{args.style}, fit scale {view.transform().m11():.3f}, viewport {w}x{h}, units {units}, "
            f"brush {args.clicks_brush}, level {level} (most common {common}), "
            f"footprint scope {view._footprint_scope} (outlines off)"
        )
        output.append(composite_backend.describe())
        output.append(f"large drag: {tiles} tiles written, {args.clicks_rows} rows; undo via {undo_via}")
        output.append(f"stalls: {len(stalls)} [{' '.join(sizes)} ms], {untimed} with an untimed cause")
        output += [f"  {line}" for line in stalls]
        output.append(f"drags: {len(drags)}")
        output += [f"  {line}" for line in drags]
        output.append("--- full debug log of the sequence")
        output.append(text)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return output


def _pump_until_quiet(paints: list, quiet_s: float = 1.0, cap_s: float = 20.0, min_ms: float = 2.0) -> None:
    """Pumps until no canvas paint of min_ms or more has landed for quiet_s.
    A cursor resting on the viewport keeps a trickle of sub-ms paints going,
    so a plain "no paint" wait would never end."""
    from PyQt5.QtWidgets import QApplication

    start = time.perf_counter()
    seen, last_change = len(paints), start
    while time.perf_counter() - start < cap_s:
        QApplication.processEvents()
        time.sleep(0.005)
        if len(paints) != seen:
            if any((b - a) * 1000 >= min_ms for a, b in paints[seen:]):
                last_change = time.perf_counter()
            seen = len(paints)
        elif time.perf_counter() - last_change > quiet_s:
            return


def _stall_ms(calls: list[tuple[float, float]], paints: list[tuple[float, float]]) -> tuple[float, float, float]:
    """(handler, next paint, total) ms for the last handler call: the handler's
    wall, then the first canvas paint that started after it returned, to that
    paint's end. A user feels the total as one stall. No duration floor: Qt merges
    pending updates into one paint, so a hover repaint can't land ahead of it."""
    if not calls:
        return (0.0, 0.0, 0.0)
    t0, t1 = calls[-1]
    nxt = next(((a, b) for a, b in paints if a >= t1), None)
    if nxt is None:
        return ((t1 - t0) * 1000, 0.0, (t1 - t0) * 1000)
    return ((t1 - t0) * 1000, (nxt[1] - nxt[0]) * 1000, (nxt[1] - t0) * 1000)


def _drag_header(lines: list[str], label: str) -> tuple[int, float, float] | None:
    """(steps, ms/step, max step ms) from the last `perf drag <label>:` line."""
    import re

    for line in reversed(lines):
        m = re.search(rf"perf drag {label}: (\d+) steps, \d+ms total, ([\d.]+)ms/step \(max ([\d.]+)\)", line)
        if m:
            return int(m.group(1)), float(m.group(2)), float(m.group(3))
    return None


GH180_SECTIONS = ("load", "draw", "undo1", "elev", "undo2")


def gh180_gates(sections: dict[str, list[str]], stalls: dict[str, tuple[float, float, float]]) -> str:
    """The `gh180 gates:` summary line, from the flow's debug log split per
    phase (GH180_SECTIONS, timestamps stripped) and _stall_ms() per handler
    (draw, undo1, undo2)."""
    import re

    def section(name: str) -> list[str]:
        return sections.get(name, [])

    parts: list[str] = []
    load = next((ln for ln in section("load") if "First paint composited in" in ln), "")
    m = re.search(r"First paint composited in ([\d.]+)s \(total ([\d.]+)s, mip (-?\d+)", load)
    if m:
        parts.append(f"G5_first_paint={float(m.group(1)) * 1000:.0f} G5_total={float(m.group(2)) * 1000:.0f}")
    draw = section("draw")
    head = _drag_header(draw, "paint-terrain")
    if head:
        parts.append(f"G1_steps={head[0]} G1_ms_step={head[1]:.1f} G1_max={head[2]:.1f}")
    handler, paint, total = stalls["draw"]
    parts.append(f"G2_release={handler:.0f} G2_next_paint={paint:.0f} G2_total={total:.0f}")
    after = draw[next((i for i, ln in enumerate(draw) if "perf drag paint-terrain:" in ln), len(draw)) + 1:]
    comp = next((re.search(r"composite ([\d.]+) x(\d+)", ln) for ln in after if re.search(r"composite [\d.]+ x\d", ln)), None)
    if comp:
        parts.append(f"post_stroke_composite={float(comp.group(1)):.0f}x{comp.group(2)}")
    elev = section("elev")
    head3 = _drag_header(elev, "set-elevation")
    if head3:
        parts.append(f"G3_steps={head3[0]} G3_ms_step={head3[1]:.1f} G3_max={head3[2]:.1f}")
    for key in ("undo1", "undo2"):
        sec = section(key)
        rerender = next((re.search(r"(\d+) tiles dirty, re-rendered", ln) for ln in sec if "re-rendered full map" in ln), None)
        op = next((re.search(r"perf op undo: (\d+)ms", ln) for ln in sec if "perf op undo:" in ln), None)
        handler, paint, total = stalls[key]
        side = f"rerender:{rerender.group(1)}" if rerender else "patch"
        parts.append(
            f"G4_{key}_op={op.group(1) if op else '?'} G4_{key}_next_paint={paint:.0f} G4_{key}_total={total:.0f}"
            f" G4_{key}_path={side}"
        )
    maxes = [h[2] for h in (head, head3) if h]
    if maxes:
        parts.append(f"G6_max_step={max(maxes):.1f}")
    return "gh180 gates: " + " ".join(parts)


def _run_flow_gh180(args) -> list[str]:
    """--flow gh180: GH #180's tester flow. See the module docstring."""
    from collections import Counter

    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    from descape import composite_backend, debug_log, perf_trace, settings, viewer, viewer_canvas
    from testkit.qt_window import ScenarioLoadError

    settings._paint_trees = True
    settings._paint_eye_candy = False
    settings._distance_ticks = False
    perf_trace.enable(True)  # before the window: its constructor starts the stall watchdog and gc hook

    # Class-level, before the window exists: MapView and the Undo action bind their callbacks at construction.
    paints: list[tuple[float, float]] = []
    calls: dict[str, list[tuple[float, float]]] = {"on_edit_stroke_end": [], "undo": []}
    saved = [
        (viewer_canvas.MapCanvasItem, "paint"), (viewer.ViewerWindow, "on_edit_stroke_end"), (viewer.ViewerWindow, "undo"),
    ]
    originals = {(owner, name): owner.__dict__[name] for owner, name in saved}

    def timed_paint(self, *a, **k):
        t0 = time.perf_counter()
        originals[(viewer_canvas.MapCanvasItem, "paint")](self, *a, **k)
        paints.append((t0, time.perf_counter()))

    def timed(name):
        inner = originals[(viewer.ViewerWindow, name)]

        # No *args: a QAction's triggered(bool) would pass `checked` into a varargs slot.
        def wrapper(self):
            t0 = time.perf_counter()
            try:
                return inner(self)
            finally:
                calls[name].append((t0, time.perf_counter()))
        return wrapper

    viewer_canvas.MapCanvasItem.paint = timed_paint
    viewer.ViewerWindow.on_edit_stroke_end = timed("on_edit_stroke_end")
    viewer.ViewerWindow.undo = timed("undo")
    window = viewer.ViewerWindow()
    output: list[str] = []
    # Raw debug log text per GH180_SECTIONS phase, cleared after each so the ring buffer can't drop any.
    raw: dict[str, str] = {}

    def take(name: str) -> None:
        text = debug_log.get_log_text()
        raw[name] = "" if text == "(empty)" else text
        debug_log.clear()

    def settle_and_flush() -> None:
        _pump_until_quiet(paints)
        perf_trace.flush_idle()

    def ctrl_z(origin, w: int, h: int) -> str:
        cursor0 = window.edit_history.cursor
        window.activateWindow()
        window.raise_()
        _xdotool(["mousemove", str(origin.x() + w // 2), str(origin.y() + h // 2), "key", "ctrl+z"])
        settle_and_flush()
        if window.edit_history.cursor == cursor0:
            window.undo()
            settle_and_flush()
            return "direct window.undo() (ctrl+z did not reach the window)"
        return "ctrl+z"

    def drag(origin, rows, span, what: str) -> int:
        cursor0 = window.edit_history.cursor
        _xdotool(_serpentine_cmd(origin, rows, span, args.px, args.delays[0] / 1000))
        settle_and_flush()
        if window.edit_history.cursor != cursor0 + 1:
            raise SystemExit(
                f"--flow gh180: the {what} drag pushed no record (cursor {cursor0}->{window.edit_history.cursor},"
                f" tool {window.map_view._tool}, mode {window.mode}, rows {rows}, span {[span(y) for y in rows]},"
                f" origin {origin.x()},{origin.y()}, modal {QApplication.activeModalWidget()},"
                f" popup {QApplication.activePopupWidget()}, enabled {window.isEnabled()}/{window.map_view.isEnabled()},"
                f" visible tops {[type(t).__name__ for t in QApplication.topLevelWidgets() if t.isVisible()]})\n"
                + debug_log.get_log_text()
            )
        return len(window.edit_history.records[window.edit_history.cursor - 1].touched_indices())

    try:
        if args.style != "Stepped":
            # No document yet: the handler only records the style, so the load renders in it.
            window.terrain_style_combo.setCurrentText(args.style)
        # Terrain mode and Draw before the load too, so the load's fit sees the panels the drags will.
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("draw")
        window.move(0, 0)
        window.resize(*args.window)
        window.show()
        _pump(0.5)
        debug_log.clear()
        window.load_scenario(Path(args.scenario))
        if window.scenario is None:
            raise ScenarioLoadError(f"{args.scenario} failed to load")
        settle_and_flush()
        load_style, load_mode = window._terrain_style, window.mode
        view = window.map_view
        load_vp = view.viewport().size()
        _pump(args.settle)
        perf_trace.flush_idle()
        take("load")
        mm = window.scenario.map_manager

        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("draw")
        window.brush_size_spin.setValue(args.flow_brush)
        window.terrain_panel.set_terrain(args.flow_terrain)
        window.paint_trees_check.setChecked(True)
        _pump(0.5)
        refit = view.viewport().size() != load_vp
        if refit:
            # The terrain panel narrowed the viewport since the load's own fit: refit, as a user would.
            _fit(view)
            settle_and_flush()
            _pump(args.settle)
        perf_trace.flush_idle()
        vp = view.viewport()
        origin = vp.mapToGlobal(QPoint(0, 0))
        w, h = vp.width(), vp.height()
        units = sum(len(u) for u in window.scenario.unit_manager.units)
        output.append(
            f"{args.style} ({load_style}, mode {load_mode} at load; refit after {refit}),"
            f" fit scale {view.transform().m11():.4f},"
            f" visible mip {_visible_mip(view)}, viewport {w}x{h}, screen"
            f" {QApplication.primaryScreen().size().width()}x{QApplication.primaryScreen().size().height()},"
            f" map {mm.map_width}x{mm.map_height}, units {units}"
        )
        output.append(composite_backend.describe())
        output.append(
            f"draw: brush {args.flow_brush}, terrain {args.flow_terrain}, trees {window.paint_trees_check.isChecked()},"
            f" eye candy {window.paint_eye_candy_check.isChecked()}, {args.flow_draw_rows} rows,"
            f" {args.px}px per move, {args.delays[0]:g}ms between moves"
        )
        debug_log.clear()
        units0 = sum(len(u) for u in window.scenario.unit_manager.units)
        rows, span = _diamond_rows(view, mm, args.flow_draw_rows)
        written = drag(origin, rows, span, "Draw")
        mip_after = _visible_mip(view)
        units1 = sum(len(u) for u in window.scenario.unit_manager.units)
        stalls = {"draw": _stall_ms(calls["on_edit_stroke_end"], paints)}
        output.append(f"  draw wrote {written} tiles, units {units0}->{units1}, visible mip after {mip_after}")
        take("draw")

        output.append(f"  undo 1 via {ctrl_z(origin, w, h)}")
        stalls["undo1"] = _stall_ms(calls["undo"], paints)
        take("undo1")

        window._on_tool_selected("set_level")
        window.brush_size_spin.setValue(args.flow_brush)
        common = Counter(tile.elevation for tile in mm.terrain).most_common(1)[0][0]
        top = window.elevation_level_spin.maximum()
        level = common + 2 if common + 2 <= top else max(0, common - 2)
        window.elevation_level_spin.setValue(level)
        _pump(0.5)
        perf_trace.flush_idle()
        debug_log.clear()
        rows, span = _diamond_rows(view, mm, args.flow_elev_rows)
        written = drag(origin, rows, span, "Set elevation")
        output.append(f"set elevation: level {level} (most common {common}), {args.flow_elev_rows} rows, wrote {written} tiles")
        take("elev")

        output.append(f"  undo 2 via {ctrl_z(origin, w, h)}")
        stalls["undo2"] = _stall_ms(calls["undo"], paints)
        take("undo2")
        sections = {k: [ln.split("] ", 1)[-1] for ln in v.splitlines()] for k, v in raw.items()}
        output.append(gh180_gates(sections, stalls))
        for name in GH180_SECTIONS:
            output.append(f"--- debug log: {name}")
            output.append(raw.get(name, ""))
    finally:
        for (owner, name), original in originals.items():
            setattr(owner, name, original)
        if window.scenario is not None:
            window.edit_history.mark_saved()
        window.close()
    return output


def _undo_stroke_tiles(mm, target: int) -> list[list[tuple[int, int]]]:
    """--undo's stroke: rows of cursor tiles 9 apart over a centred square of
    about `target` tiles, one list per row (one MapView event each)."""
    side = min(mm.map_width - 10, mm.map_height - 10, int(target**0.5))
    x0, y0 = (mm.map_width - side) // 2, (mm.map_height - side) // 2
    rows = []
    for r, y in enumerate(range(y0 + 4, y0 + side, 9)):
        xs = range(x0, x0 + side) if r % 2 == 0 else range(x0 + side - 1, x0 - 1, -1)
        rows.append([(x, y) for x in xs])
    return rows


def _bbox_split(scenario, dirty_indices, elevations, proj, canvas, with_sprites: bool):
    """render._dirty_screen_bbox() for Stepped (radii 0), re-run on a COPY of the
    pre-edit elevations with each phase timed. Returns (rect, {phase: ms}).
    The caller checks the rect against the real call's, which keeps this honest."""
    import numpy as np

    from descape import iso_geometry, render

    t = time.perf_counter
    ms: dict[str, float] = {}
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    t0 = t()
    dirty_xy = {(mm.terrain[i].x, mm.terrain[i].y) for i in dirty_indices}
    changed: set = set()
    pre_elevation: dict = {}
    for x, y in dirty_xy:
        pre = int(elevations[y, x])
        if pre != mm.get_tile(x, y).elevation:
            changed.add((x, y))
            pre_elevation[(x, y)] = pre
    for x, y in dirty_xy:
        elevations[y, x] = mm.get_tile(x, y).elevation
    if any(not (proj.min_elev <= int(elevations[y, x]) <= proj.max_elev) for x, y in dirty_xy):
        return None, ms
    ms["tile_loop"] = (t() - t0) * 1000
    t0 = t()
    seed = set(dirty_xy)
    for x, y in list(dirty_xy):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    seed.add((nx, ny))
    ms["dilation"] = (t() - t0) * 1000
    t0 = t()
    footprint_owners: dict = {}
    if changed:
        for units in scenario.unit_manager.units:
            for u in units:
                if (int(u.x), int(u.y)) not in changed:
                    continue
                bounds = render.unit_tile_bounds(u, w, h)
                if bounds is None:
                    continue
                fx0, fx1, fy0, fy1 = bounds
                own = (int(u.x), int(u.y))
                for fx in range(fx0, fx1):
                    for fy in range(fy0, fy1):
                        seed.add((fx, fy))
                        footprint_owners.setdefault((fx, fy), set()).add(own)
    ms["units_scan"] = (t() - t0) * 1000
    t0 = t()
    seed_tiles = list(seed)
    sxs = np.fromiter((x for x, _ in seed_tiles), dtype=np.int64, count=len(seed_tiles))
    sys_ = np.fromiter((y for _, y in seed_tiles), dtype=np.int64, count=len(seed_tiles))
    ms["fromiter"] = (t() - t0) * 1000
    t0 = t()
    lo, hi = render._observed_elevation_range(elevations, pre_elevation, sxs, sys_)
    if footprint_owners:
        owners = list({o for owns in footprint_owners.values() for o in owns})
        olo, ohi = render._observed_elevation_range(
            elevations, pre_elevation,
            np.array([x for x, _ in owners], dtype=np.int64), np.array([y for _, y in owners], dtype=np.int64),
        )
        owner_range = {o: (int(olo[i]), int(ohi[i])) for i, o in enumerate(owners)}
        seed_index = {tt: i for i, tt in enumerate(seed_tiles)}
        for tile, owns in footprint_owners.items():
            i = seed_index[tile]
            for o in owns:
                lo[i] = min(lo[i], owner_range[o][0])
                hi[i] = max(hi[i], owner_range[o][1])
    tx0s, ty0s, tx1s, ty1s = iso_geometry.tile_screen_bounds_over(sxs, sys_, lo, hi, proj)
    x0, y0, x1, y1 = int(tx0s.min()), int(ty0s.min()), int(tx1s.max()), int(ty1s.max())
    ms["range_merge"] = (t() - t0) * 1000
    t0 = t()
    if with_sprites:
        pad_l, pad_u, pad_r, pad_d = render._sprite_reach_px(proj)
        band = [(x, y) for x, y in changed & render.anchor_tiles(scenario) if 0 <= x < w and 0 <= y < h]
        if band:
            bxs = np.array([x for x, _ in band], dtype=np.int64)
            bys = np.array([y for _, y in band], dtype=np.int64)
            blo, bhi = render._observed_elevation_range(elevations, pre_elevation, bxs, bys)
            tx0, ty_hi = iso_geometry.tile_screen_origin(bxs, bys, bhi, proj)
            ty_lo = ty_hi + (bhi - blo) * proj.elev_step
            x0 = min(x0, int(tx0.min()) - pad_l)
            x1 = max(x1, int(tx0.max()) + 2 * proj.half_w + pad_r)
            y0 = min(y0, int(ty_hi.min()) - proj.half_h - pad_u)
            y1 = max(y1, int(ty_lo.max()) + 2 * proj.half_h + pad_d)
    ms["sprite_band"] = (t() - t0) * 1000
    cw, ch = canvas
    return (max(0, x0), max(0, y0), min(cw, x1), min(ch, y1)), ms


def _early_exit_estimate(scenario, dirty_xy, elevations, proj, canvas, with_sprites: bool):
    """The plan's cheap over-estimate: the dirty index bounds dilated by 1 plus
    the map's largest unit footprint span, swept over the map's whole elevation
    range, plus the sprite reach. (rect, covers_canvas)."""
    import numpy as np

    from descape import iso_geometry, render

    h, w = elevations.shape
    span = 1
    for units in scenario.unit_manager.units:
        for u in units:
            b = render.unit_tile_bounds(u, w, h)
            if b is not None:
                span = max(span, b[1] - b[0], b[3] - b[2])
    r = 1 + span
    xs = [x for x, _ in dirty_xy]
    ys = [y for _, y in dirty_xy]
    bx0, bx1 = max(0, min(xs) - r), min(w - 1, max(xs) + r)
    by0, by1 = max(0, min(ys) - r), min(h - 1, max(ys) + r)
    cx = np.array([bx0, bx1, bx0, bx1], dtype=np.int64)
    cy = np.array([by0, by0, by1, by1], dtype=np.int64)
    lo = np.full(4, int(elevations.min()), dtype=np.int64)
    hi = np.full(4, int(elevations.max()), dtype=np.int64)
    tx0, ty0, tx1, ty1 = iso_geometry.tile_screen_bounds_over(cx, cy, lo, hi, proj)
    x0, y0, x1, y1 = int(tx0.min()), int(ty0.min()), int(tx1.max()), int(ty1.max())
    if with_sprites:
        pad_l, pad_u, pad_r, pad_d = render._sprite_reach_px(proj)
        x0, y0, x1, y1 = x0 - pad_l, y0 - pad_u - proj.half_h, x1 + pad_r, y1 + pad_d + 2 * proj.half_h
    cw, ch = canvas
    return (x0, y0, x1, y1), x0 <= 0 and y0 <= 0 and x1 >= cw and y1 >= ch


class _SpliceTimer:
    """Times the in-op elevation splice's parts (reanchor, grid patch, pack
    refresh) while --undo runs an undo with the cap lifted."""

    def __init__(self) -> None:
        from descape import render, render_cache

        self.ms: dict[str, float] = {}
        self._saved = []
        for owner, name, key in (
            (render_cache, "_reanchor_units", "reanchor"),
            (render, "patch_bystander_grid", "grid_patch"),
            (render_cache, "_refreshed_pack", "pack_refresh"),
        ):
            original = getattr(owner, name)
            self._saved.append((owner, name, original))
            setattr(owner, name, self._timed(original, key))

    def _timed(self, original, key):
        def wrapper(*a, **k):
            t0 = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                self.ms[key] = self.ms.get(key, 0.0) + (time.perf_counter() - t0) * 1000
        return wrapper

    def remove(self) -> None:
        for owner, name, original in self._saved:
            setattr(owner, name, original)


def _splice_costs(cache, mip: int, changed: set) -> list[str]:
    """The undo's would-be elevation component on level `mip` (seeds and
    members, uncapped) and the full grid and pack build times there."""
    from descape import native_composite, render, render_cache
    from descape.composite_backend import native

    t = time.perf_counter
    scen = cache.scenario
    mm = scen.map_manager
    w, h = mm.map_width, mm.map_height
    t0 = t()
    own_index = render.unit_own_tile_index(scen)
    seeds = []
    for own in changed:
        for player_id, i, unit in own_index.get(own, ()):
            if not cache.unit_filter.matches(player_id, unit):
                continue
            occupied = render.unit_occupied_tiles(unit, w, h)
            if occupied is not None:
                seeds.append((player_id, i, unit, own, tuple(occupied)))
    members = render_cache._shared_tile_component(
        scen, cache.units_by_tile, cache.unit_filter, own_index, seeds, 10**9
    )
    t_component = (t() - t0) * 1000
    if members is None:
        return [f"      component: None (a unit missing from the own-tile index), {len(seeds)} seeds"]
    lvl = cache._level(mip)
    t0 = t()
    render.build_bystander_grid(lvl.building_bboxes, cache.chunk_px)
    t_grid_full = (t() - t0) * 1000
    t_pack_full = 0.0
    if native is not None:
        t0 = t()
        native_composite.UnitPack(
            False, w, h, lvl.proj, cache.units_by_tile, lvl.sprites, cache._units_version, cache.elevations,
        )
        t_pack_full = (t() - t0) * 1000
    return [
        (
            f"      component: {len(seeds)} seeds -> {len(members)} members (cap {render_cache._ELEV_SPLICE_MAX_UNITS}),"
            f" walk {t_component:.1f}ms; full grid build {t_grid_full:.1f}ms, full pack build {t_pack_full:.1f}ms"
        ),
    ]


def _run_undo(args) -> list[str]:
    """--undo: a large Set elevation or Elevate stroke at one zoom, then a timed
    window.undo(), per tool and rep. See the module docstring."""
    from PyQt5.QtCore import Qt

    from descape import composite_backend, debug_log, perf_trace, render, render_cache, settings, viewer
    from testkit import qt_window

    settings._paint_trees = False
    settings._paint_eye_candy = False
    settings._distance_ticks = False
    perf_trace.enable(True)
    window = qt_window.stepped_window(Path(args.scenario), show=False)
    output = []
    try:
        window.move(0, 0)
        window.resize(1600, 1000)
        window.show()
        view = window.map_view
        _pump(0.2)
        view.fitInView(view.sceneRect(), Qt.KeepAspectRatio)
        if args.undo_zoom > 1:
            view.scale(args.undo_zoom, args.undo_zoom)
            view.centerOn(view.sceneRect().center())
        _pump(args.settle)
        target = view.viewport_chunk_target()
        mip = target[0]
        cache = window._cache
        mm = window.scenario.map_manager
        canvas = render._canvas_pixel_dims(window._iso_proj)
        output.append(
            f"{args.style}, zoom fit x{args.undo_zoom:g} (scale {view.transform().m11():.3f}, visible mip {mip}),"
            f" map {mm.map_width}x{mm.map_height}, units {sum(len(u) for u in window.scenario.unit_manager.units)},"
            f" canvas {canvas[0]}x{canvas[1]}"
        )
        output.append(composite_backend.describe())

        captured: dict = {}
        real_bbox = viewer.dirty_screen_bbox_iso

        def bbox_probe(scenario, dirty_indices, elevations, proj, **kw):
            pre = elevations.copy()

            def split():
                captured["split"] = _bbox_split(
                    scenario, dirty_indices, pre.copy(), proj, render._canvas_pixel_dims(proj),
                    kw.get("with_sprites", False),
                )

            if args.undo_split_first:
                split()
            t0 = time.perf_counter()
            rect = real_bbox(scenario, dirty_indices, elevations, proj, **kw)
            captured["bbox_ms"] = (time.perf_counter() - t0) * 1000
            captured["rect"] = rect
            captured["changed"] = set(kw.get("elevation_changed") or ())
            captured["dirty"] = len(dirty_indices)
            captured["dirty_xy"] = {(mm.terrain[i].x, mm.terrain[i].y) for i in dirty_indices}
            if not args.undo_split_first:
                split()
            # The real function again on a copy: tells a first-call effect from the split's own coverage.
            t0 = time.perf_counter()
            real_bbox(scenario, dirty_indices, pre.copy(), proj, **{**kw, "elevation_changed": set()})
            captured["bbox_ms_2"] = (time.perf_counter() - t0) * 1000
            captured["pre"] = pre
            return rect

        for tool in args.undo_tools:
            window.mode_combo.setCurrentText("Terrain")
            window._on_tool_selected(tool)
            window.brush_size_spin.setValue(args.undo_brush)
            if tool == "set_level":
                common = Counter(tile.elevation for tile in mm.terrain).most_common(1)[0][0]
                top = window.elevation_level_spin.maximum()
                window.elevation_level_spin.setValue(common + 2 if common + 2 <= top else max(0, common - 2))
            passes = (True,) if args.undo_plain else (True, False)
            for rep, capped in [(r, c) for r in range(args.undo_reps) for c in passes]:
                rows = _undo_stroke_tiles(mm, args.undo_tiles)
                window.on_edit_stroke_start()
                for row in rows:
                    window.on_edit_stroke_tiles(row, Qt.NoModifier)
                    perf_trace.step()
                window.on_edit_stroke_end()
                _pump(args.settle)
                written = len(window.edit_history.records[window.edit_history.cursor - 1].touched_indices())
                debug_log.clear()
                captured.clear()
                cap = render_cache._ELEV_SPLICE_MAX_UNITS
                sloped_cap = render_cache._SLOPED_ELEV_SPLICE_MAX_UNITS
                timer = None
                if args.undo_plain:
                    pass
                elif capped:
                    viewer.dirty_screen_bbox_iso = bbox_probe
                else:
                    render_cache._ELEV_SPLICE_MAX_UNITS = 10**9
                    render_cache._SLOPED_ELEV_SPLICE_MAX_UNITS = 10**9
                    timer = _SpliceTimer()
                try:
                    t0 = time.perf_counter()
                    window.undo()
                    wall = (time.perf_counter() - t0) * 1000
                finally:
                    viewer.dirty_screen_bbox_iso = real_bbox
                    render_cache._ELEV_SPLICE_MAX_UNITS = cap
                    render_cache._SLOPED_ELEV_SPLICE_MAX_UNITS = sloped_cap
                    if timer is not None:
                        timer.remove()
                perf_trace.flush_idle()
                ops = [ln.split("] ", 1)[-1] for ln in debug_log.get_log_text().splitlines() if "perf op undo" in ln]
                label = "capped (today)" if capped else "cap lifted (in-op splice)"
                output.append(
                    f"--- {tool} b{args.undo_brush} rep {rep + 1}, {label}: {written} tiles written; loadavg {_loadavg()}"
                )
                output += [f"    {ln}" for ln in ops]
                if args.undo_plain:
                    output.append(f"    undo wall {wall:.1f}ms")
                    _pump(args.settle)
                    continue
                if not capped:
                    parts = timer.ms
                    total = sum(parts.values())
                    output.append(
                        f"    undo wall {wall:.1f}ms; splice parts: "
                        + "  ".join(f"{k} {v:.1f}" for k, v in parts.items()) + f"  = {total:.1f}ms"
                    )
                    _pump(args.settle)
                    continue
                output.append(
                    f"    undo wall {wall:.1f}ms (instrumented bbox incl.; bbox real {captured.get('bbox_ms', 0):.1f},"
                    f" again on a copy {captured.get('bbox_ms_2', 0):.1f})"
                )
                if "rect" not in captured:
                    output.append("    (no bbox call: the undo took another path)")
                    _pump(args.settle)
                    continue
                rect = captured["rect"]
                split_rect, split = captured["split"]
                if split_rect is None:
                    output.append("    (bbox split: out of the cached elevation range)")
                    _pump(args.settle)
                    continue
                est, covers = _early_exit_estimate(
                    window.scenario, captured["dirty_xy"], captured["pre"], window._iso_proj, canvas, cache.sprites_enabled
                )
                is_canvas = rect is not None and rect == (0, 0, canvas[0], canvas[1])
                output.append(
                    f"    dirty {captured['dirty']}, elevation-changed {len(captured['changed'])};"
                    f" bbox rect {rect} {'== canvas' if is_canvas else '< canvas'}"
                    f" ({'' if split_rect == rect else 'SPLIT RECT DIFFERS ' + str(split_rect)})"
                )
                total = sum(split.values())
                output.append(
                    "    bbox split: " + "  ".join(f"{k} {v:.1f}" for k, v in split.items())
                    + f"  = {total:.1f}ms; tile_loop+dilation {100 * (split['tile_loop'] + split['dilation']) / total:.0f}%"
                )
                output.append(f"    early-exit estimate {est} covers canvas: {covers}")
                output += _splice_costs(cache, mip, captured["changed"])
                _pump(args.settle)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return output


def _csv(kind):
    return lambda text: [kind(v) for v in text.split(",")]


def _size(text: str) -> tuple[int, int]:
    w, sep, h = text.lower().partition("x")
    if not sep or not w.isdigit() or not h.isdigit():
        raise argparse.ArgumentTypeError(f"expected WxH, got {text!r}")
    return int(w), int(h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    parser.add_argument("--style", default="Stepped", choices=["Flat", "Stepped", "Sloped"])
    parser.add_argument("--tool", default="draw", choices=["draw", "elevation", "set_level", "convert", "cliff"])
    parser.add_argument("--convert-to", type=int, default=2, help="Convert's destination owner (--tool convert)")
    parser.add_argument("--cliff-piece", type=int, default=264, help="cliff unit const (--tool cliff; no brush)")
    parser.add_argument("--brush", type=_csv(int), default=[1, 5, 9], help="comma-separated brush sizes")
    parser.add_argument("--delays", type=_csv(float), default=[16.0], help="comma-separated ms between moves")
    parser.add_argument("--px", type=int, default=40, help="pixels per mouse move")
    parser.add_argument("--button", default="1", choices=["1", "3"], help="xdotool button: 1 left, 3 right")
    parser.add_argument("--settle", type=float, default=5.0, help="seconds to let the viewport warm before dragging")
    parser.add_argument("--no-preload", action="store_true", help="turn off neighbouring zoom-level preload")
    parser.add_argument("--profile-patch", action="store_true", help="one line per cache patch() (see above)")
    parser.add_argument("--root", type=Path, help="DEscape checkout to import descape and testkit from")
    parser.add_argument("--fit", action="store_true", help="per-brush drags at fit zoom, not 1:1 (see above)")
    parser.add_argument("--trees", action="store_true", help="leave Trees on (Eye candy stays off)")
    parser.add_argument("--window", type=_size, default=(1600, 1000), help="main window size WxH (default 1600x1000)")
    parser.add_argument("--flow", choices=["gh180"], help="a scripted tester flow (see above)")
    parser.add_argument("--flow-brush", type=int, default=9, help="--flow brush size (default 9)")
    parser.add_argument(
        "--flow-terrain", type=int, default=FLOW_TERRAIN, help=f"--flow Draw terrain id (default {FLOW_TERRAIN})"
    )
    parser.add_argument(
        "--flow-draw-rows", type=int, default=FLOW_DRAW_ROWS, help=f"--flow Draw drag rows (default {FLOW_DRAW_ROWS})"
    )
    parser.add_argument(
        "--flow-elev-rows", type=int, default=FLOW_ELEV_ROWS,
        help=f"--flow Set elevation drag rows (default {FLOW_ELEV_ROWS})",
    )
    parser.add_argument("--clicks", action="store_true", help="the Set elevation stall repro (see above)")
    parser.add_argument("--clicks-brush", type=int, default=9, help="--clicks brush size, drag and clicks (default 9)")
    parser.add_argument("--clicks-rows", type=int, default=6, help="rows in --clicks' large drag (default 6)")
    parser.add_argument("--clicks-no-undo", action="store_true", help="--clicks without the Ctrl+Z after the drag")
    parser.add_argument(
        "--clicks-scope", default="multitile", choices=["multitile", "buildings", "all"],
        help="--clicks footprint scope, outlines off (default multitile)",
    )
    parser.add_argument("--undo", action="store_true", help="the large-elevation undo probe (see above)")
    parser.add_argument("--undo-zoom", type=float, default=1.0, help="--undo zoom as a multiple of fit (default 1)")
    parser.add_argument(
        "--undo-tools", type=_csv(str), default=["set_level", "elevation"], help="--undo tools (default both)"
    )
    parser.add_argument("--undo-brush", type=int, default=9, help="--undo brush size (default 9)")
    parser.add_argument("--undo-tiles", type=int, default=16000, help="--undo stroke square, tiles (default 16000)")
    parser.add_argument("--undo-reps", type=int, default=3, help="--undo reps per tool (default 3)")
    parser.add_argument(
        "--undo-split-first", action="store_true", help="--undo: time the bbox split before the real call, not after"
    )
    parser.add_argument(
        "--undo-plain", action="store_true", help="--undo: the timed undo alone, no probes (for an A/B with --root)"
    )
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        sys.exit("needs a real X display (see the Xvfb recipe in --help)")
    root = (args.root or ROOT).resolve()
    sys.path.insert(0, str(root))
    from testkit import qt_window, settings_isolation

    # xcb, not offscreen: motion compression is the thing being measured.
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    qt_window.ensure_qapp()
    settings_isolation.pin_install_path()
    with tempfile.TemporaryDirectory(prefix="bench_gui_drag_") as tmp:
        settings_isolation.isolate_settings(Path(tmp))
        if args.flow == "gh180":
            output = _run_flow_gh180(args)
        else:
            output = _run_clicks(args) if args.clicks else _run_undo(args) if args.undo else _run(args)
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    print(f"descape from {root} ({sha.stdout.strip() or '?'}); loadavg {_loadavg()}")
    print("\n".join(output))


def _loadavg() -> str:
    try:
        return Path("/proc/loadavg").read_text().strip()
    except OSError:
        return "?"


if __name__ == "__main__":
    main()
