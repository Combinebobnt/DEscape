#!/usr/bin/env python3
"""Copy Region and Paste Region cost on a real map: one copy of --region,
then one paste per --anchors entry, each with Terrain, Elevation and Units
checked (Elevation unchecked under --no-elevation). With --undo, each paste
is followed by an undo and a redo. Informational only, never pass/fail,
like bench_stroke_end.py.

Drives a real offscreen ViewerWindow (testkit.qt_window.stepped_window) with
the install pinned and settings isolated to a throwaway dir, Perf Trace on.
Per action (copy, paste, undo, redo) it prints:

  call       the handler's own wall time (copy_region() / paste_region() / undo() / redo())
  first_pe   the first QApplication.processEvents() after it: the repaint
  paint      every MapCanvasItem.paint until the canvas goes quiet
  counts     level builds and installs per mip, in the op line and in the view
             line: `-2 build` is a level build, `-1 install` counts warm jobs
             installed, not builds
  then the Perf Trace lines the action produced: its `perf op` line (phases,
  untimed and the level events inside it, e.g. `-2 build (op:paste)`) and
  the `perf view` line after the pump (the repaint's own level builds and
  the level-warm installs that followed).

A summary at the end totals the counts per action kind.

Before each paste the view is centred on the paste's own block and allowed
to settle, so the repaint is a real visible one. Counts matter more than
milliseconds here: how many `-2 build`s and `-1 install`s one paste, undo
or redo causes.

Presets: `small` is a 68x50 region pasted at five anchors; `big` is a
123x78 region (~2860 units on old-allies) at three anchors that keep it on
the 240x240 map. --region/--anchors override the preset's.

`--root DIR` imports descape (and testkit) from another checkout, so one copy
of this script times a baseline worktree and a lane branch alike.

Usage (from a DEscape checkout root):
  .venv/bin/python tools/bench_copy_paste.py [scenario] [--root DIR] [--style Stepped]
      [--scale 0.25] [--preset small|big] [--region 80,10,148,60]
      [--anchors 160,10;160,70;...] [--sprites on|off] [--undo] [--no-elevation]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DEFAULT_SCENARIO = TOOL_ROOT / "examples" / "old-allies-final-v2.aoe2scenario"
PRESETS = {
    "small": ("80,10,148,60", "160,10;160,70;10,100;90,100;160,130"),
    "big": ("39,162,162,240", "117,0;0,40;110,100"),
}
WARM_PUMP_S = 2.0
# `-2 build 62.1ms [walk ...] (op:paste)` or `-1 install 2.1ms x2 (level-warm)`.
_LEVEL_EVENT = re.compile(r"(-?\d+) (build|install) [\d.]+ms(?: \[[^\]]*\])?(?: x(\d+))?")


def _pump(seconds: float) -> None:
    from PyQt5.QtWidgets import QApplication

    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        QApplication.processEvents()
        time.sleep(0.005)


def _pump_until_quiet(paints: list, quiet_s: float = 0.4, cap_s: float = 8.0) -> None:
    from PyQt5.QtWidgets import QApplication

    start = time.perf_counter()
    last_n, last_change = len(paints), start
    while time.perf_counter() - start < cap_s:
        QApplication.processEvents()
        time.sleep(0.005)
        if len(paints) != last_n:
            last_n, last_change = len(paints), time.perf_counter()
        elif time.perf_counter() - last_change > quiet_s:
            return


def _perf_lines() -> list[str]:
    """Writes the pending op and view lines, then returns every perf line logged."""
    from descape import debug_log, perf_trace

    perf_trace.flush_pending_op()
    perf_trace._flush_view()
    return [line for line in debug_log.get_log_text().splitlines() if "perf " in line]


def _level_counts(line: str) -> Counter:
    """(mip, kind) -> count of the level events on one perf line."""
    counts: Counter = Counter()
    levels = line.split("levels: ", 1)
    if len(levels) < 2:
        return counts
    for mip, kind, times in _LEVEL_EVENT.findall(levels[1]):
        counts[(int(mip), kind)] += int(times or 1)
    return counts


def _format_counts(counts: Counter) -> str:
    return " ".join(f"{mip} {kind} {n}" for (mip, kind), n in sorted(counts.items())) or "none"


def _timed(label: str, kind: str, paints: list, action, totals: dict) -> None:
    from PyQt5.QtWidgets import QApplication

    from descape import debug_log

    _pump(0.3)
    _perf_lines()
    debug_log.clear()
    del paints[:]
    t0 = time.perf_counter()
    action()
    call_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    QApplication.processEvents()
    first_pe_ms = (time.perf_counter() - t1) * 1000
    _pump_until_quiet(paints)
    _pump(WARM_PUMP_S)
    print(f"  {label}: call {call_ms:.1f}ms  first_pe {first_pe_ms:.1f}ms  paint {sum(paints):.1f}ms ({len(paints)})")
    lines = _perf_lines()
    op_counts: Counter = Counter()
    view_counts: Counter = Counter()
    for line in lines:
        if "perf op " in line:
            op_counts += _level_counts(line)
        elif "perf view" in line:
            view_counts += _level_counts(line)
    print(f"    counts op: {_format_counts(op_counts)} | view: {_format_counts(view_counts)}")
    for line in lines:
        print(f"    {line}")
    slot = totals.setdefault(kind, {"n": 0, "op": Counter(), "view": Counter()})
    slot["n"] += 1
    slot["op"] += op_counts
    slot["view"] += view_counts
    sys.stdout.flush()


def _parse_box(text: str) -> tuple[int, ...]:
    return tuple(int(v) for v in text.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", type=Path, nargs="?", default=DEFAULT_SCENARIO)
    parser.add_argument("--root", type=Path, help="DEscape checkout to import descape from")
    parser.add_argument("--style", default="Stepped", choices=("Stepped", "Sloped", "Flat"))
    parser.add_argument("--scale", type=float, default=0.25, help="view scale, 0.25 lands on mip -2")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="small")
    parser.add_argument("--region", help="x0,y0,x1,y1 in tiles (half-open); overrides the preset's")
    parser.add_argument("--anchors", help="x,y;x,y;... paste top-left tiles; overrides the preset's")
    parser.add_argument("--sprites", choices=("on", "off"), default="on")
    parser.add_argument("--undo", action="store_true", help="undo, then redo, after each paste")
    parser.add_argument("--no-elevation", action="store_true", help="paste terrain and units only")
    args = parser.parse_args()
    preset_region, preset_anchors = PRESETS[args.preset]
    region = _parse_box(args.region or preset_region)
    anchors = [_parse_box(a) for a in (args.anchors or preset_anchors).split(";") if a.strip()]

    root = (args.root or TOOL_ROOT).resolve()
    sys.path.insert(0, str(root))
    from testkit import qt_window, settings_isolation

    qt_window.ensure_qapp()
    settings_isolation.pin_install_path()
    from PyQt5.QtGui import QTransform

    import descape
    from descape import composite_backend, perf_trace, settings, viewer_canvas

    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    print(f"descape from {root} ({sha.stdout.strip() or '?'}), imported {Path(descape.__file__).parent}", flush=True)
    print(composite_backend.describe(), flush=True)
    totals: dict = {}
    with tempfile.TemporaryDirectory(prefix="bench_copy_paste_") as tmp:
        settings_isolation.isolate_settings(Path(tmp))
        settings._distance_ticks = False
        paints: list[float] = []
        orig_paint = viewer_canvas.MapCanvasItem.paint

        def timed_paint(self, *a, **k):
            t0 = time.perf_counter()
            try:
                orig_paint(self, *a, **k)
            finally:
                paints.append((time.perf_counter() - t0) * 1000)

        viewer_canvas.MapCanvasItem.paint = timed_paint
        window = qt_window.stepped_window(args.scenario, show=False)
        try:
            if args.style != window.terrain_style_combo.currentText():
                window.terrain_style_combo.setCurrentText(args.style)
            window.move(0, 0)
            window.resize(1600, 1000)
            window.show()
            _pump(0.5)
            if window.show_sprites_action.isChecked() != (args.sprites == "on"):
                window.show_sprites_action.setChecked(args.sprites == "on")
            view = window.map_view
            view.setTransform(QTransform.fromScale(args.scale, args.scale))
            x0, y0, x1, y1 = region
            view.center_on_tile((x0 + x1) // 2, (y0 + y1) // 2)
            _pump(4.0)
            units = sum(len(u) for u in window.scenario.unit_manager.units)
            target = view.viewport_chunk_target()
            print(
                f"{args.scenario.name} style={args.style} scale={args.scale} sprites={args.sprites} "
                f"preset={args.preset} elevation={'off' if args.no_elevation else 'on'} undo={args.undo} "
                f"units {units} visible mip {target and target[0]} region {region}",
                flush=True,
            )
            window.paste_terrain_check.setChecked(True)
            window.paste_elevation_check.setChecked(not args.no_elevation)
            window.paste_units_check.setChecked(True)
            perf_trace.enable(True)
            window.on_region_selected(region)
            _timed("copy", "copy", paints, window.copy_region, totals)
            w, h = x1 - x0, y1 - y0
            for ax, ay in anchors:
                view.center_on_tile(ax + w // 2, ay + h // 2)
                _pump(2.0)
                window._hover_tile = (ax, ay)
                _timed(f"paste at ({ax}, {ay})", "paste", paints, window.paste_region, totals)
                if args.undo:
                    _timed(f"undo at ({ax}, {ay})", "undo", paints, window.undo, totals)
                    _timed(f"redo at ({ax}, {ay})", "redo", paints, window.redo, totals)
        finally:
            viewer_canvas.MapCanvasItem.paint = orig_paint
            perf_trace.enable(False)
            window.edit_history.mark_saved()
            window.close()
    print("summary (level events per action, summed):")
    for kind, slot in totals.items():
        print(f"  {kind} x{slot['n']}: op {_format_counts(slot['op'])} | view {_format_counts(slot['view'])}")


if __name__ == "__main__":
    main()
