#!/usr/bin/env python3
"""Drag-stroke latency and stroke continuity through REAL X input events.
Informational only, never pass/fail.

Why real events: Qt's xcb backend compresses queued mouse motion, so a slow
stroke handler sees cursor positions several tiles apart. QTest and
sendEvent() bypass that queue entirely, which is why this needs a real X
server (Xvfb is fine) and xdotool rather than an offscreen window.

What it reports, per drag:
  - steps, paints, and steps per paint (1.00 means every paint follows one
    stroke step: the motion queue is being compressed, not piling up).
  - cursor-tile gap: Chebyshev distance between successive cursor tiles
    handed to the viewer. Above 1 means compression skipped tiles.
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


def _splice_fallback_reason(render_cache, original, scenario, units_by_tile, unit_filter, tiles) -> str:
    """Why _elevation_splices() returned None, via `original` re-run with no cap."""
    cap = render_cache._ELEV_SPLICE_MAX_UNITS
    render_cache._ELEV_SPLICE_MAX_UNITS = 10**9
    try:
        out = original(scenario, units_by_tile, unit_filter, tiles)
    finally:
        render_cache._ELEV_SPLICE_MAX_UNITS = cap
    if out is not None and len(out) > cap:
        return f"over _ELEV_SPLICE_MAX_UNITS ({cap}): component would be {len(out)}"
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
            def wrapper(scenario, units_by_tile, unit_filter, tiles):
                out = original(scenario, units_by_tile, unit_filter, tiles)
                if prof._current is not None:
                    if out is None:
                        why = _splice_fallback_reason(rc, original, scenario, units_by_tile, unit_filter, tiles)
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


def _drag(window, record, paints, *, row: int, px: int, delay_ms: float, button: str, profile=None) -> list[str]:
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    from descape import debug_log

    view = window.map_view
    origin = view.viewport().mapToGlobal(QPoint(0, 0))
    x0, y0 = origin.x() + 100, origin.y() + 150 + row * 150
    cmd = ["xdotool", "mousemove", str(x0), str(y0), "sleep", "0.3", "mousedown", button]
    for i in range(1, 1000 // px + 1):
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

    # Trees/eye candy/ticks off: a Large-fill modal would hang the run, and
    # they are not what this measures. Module globals, per testkit/qt_window.py.
    settings._paint_trees = False
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
        window.paint_trees_check.setChecked(False)
        window.paint_eye_candy_check.setChecked(False)
        window.move(0, 0)
        window.resize(1600, 1000)
        window.show()
        view = window.map_view
        _pump(0.2)
        view.setTransform(QTransform())  # 1:1, mip 0
        view.centerOn(view.sceneRect().center())
        _pump(args.settle)

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
        output.append(f"preload zoom levels {'off' if args.no_preload else 'on'}, sprites {window._sprites_enabled}")
        row = 0
        for brush_size in args.brush:
            window.brush_size_spin.setValue(brush_size)
            for delay in args.delays:
                output.append(f"--- brush {brush_size}, {args.px}px/move, {delay:g}ms between moves")
                output += _drag(
                    window, record, paints, row=row % 4, px=args.px, delay_ms=delay, button=args.button, profile=profile,
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
        # Stepped/Sloped draw the map as a diamond, so a press in the fitted rect's corner is off
        # the map and opens no stroke: rows are clipped to the diamond from the corner tiles.
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

        rows = [int(cy + hh * (-0.6 + 1.2 * r / max(1, args.clicks_rows - 1))) for r in range(args.clicks_rows)]
        x_lo = span(rows[0])[0]
        cmd = ["mousemove", str(origin.x() + x_lo), str(origin.y() + rows[0]), "sleep", "0.3", "mousedown", "1"]
        for r, y in enumerate(rows):
            lo, hi = span(y)
            xs = range(lo, hi + 1, 40) if r % 2 == 0 else range(hi, lo - 1, -40)
            for x in xs:
                cmd += ["sleep", "0.016", "mousemove", str(origin.x() + x), str(origin.y() + y)]
        cmd += ["sleep", "0.3", "mouseup", "1"]

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


def _csv(kind):
    return lambda text: [kind(v) for v in text.split(",")]


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
    parser.add_argument("--clicks", action="store_true", help="the Set elevation stall repro (see above)")
    parser.add_argument("--clicks-brush", type=int, default=9, help="--clicks brush size, drag and clicks (default 9)")
    parser.add_argument("--clicks-rows", type=int, default=6, help="rows in --clicks' large drag (default 6)")
    parser.add_argument("--clicks-no-undo", action="store_true", help="--clicks without the Ctrl+Z after the drag")
    parser.add_argument(
        "--clicks-scope", default="multitile", choices=["multitile", "buildings", "all"],
        help="--clicks footprint scope, outlines off (default multitile)",
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
        output = _run_clicks(args) if args.clicks else _run(args)
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
