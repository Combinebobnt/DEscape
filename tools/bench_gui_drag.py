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
    Elevation). Anything above 1 is a hole in the stroke.
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
"""

from __future__ import annotations

import argparse
import bisect
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from testkit import qt_window, settings_isolation

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


def _drag(window, record, paints, *, row: int, px: int, delay_ms: float, button: str) -> list[str]:
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

    mean = sum(per_paint) / len(per_paint) if per_paint else 0.0
    return [
        (
            f"  wall {t1 - t0:.2f}s, steps {len(record)}, paints {len(paint_times)}, "
            f"steps per paint mean {mean:.2f} max {max(per_paint, default=0)}"
        ),
        f"  cursor-tile gap hist {_hist(cursor_gaps)} (skipped >1: {sum(g > 1 for g in cursor_gaps)})",
        f"  delivered-tile continuity hist {_hist(continuity)} (holes: {sum(c > 1 for c in continuity)})",
        "  " + debug_log.get_log_text().strip().replace("\n", "\n  "),
    ]


def _run(args) -> list[str]:
    from PyQt5.QtGui import QTransform

    from descape import perf_trace, settings, viewer_canvas

    # Trees/eye candy/ticks off: a Large-fill modal would hang the run, and
    # they are not what this measures. Module globals, per testkit/qt_window.py.
    settings._paint_trees = False
    settings._paint_eye_candy = False
    settings._distance_ticks = False
    perf_trace.enable(True)

    paints: list[float] = []
    original_paint = viewer_canvas.MapCanvasItem.paint

    def timed_paint(self, *a, **k):
        original_paint(self, *a, **k)
        paints.append(time.perf_counter())

    viewer_canvas.MapCanvasItem.paint = timed_paint

    window = qt_window.stepped_window(Path(args.scenario), show=False)
    output = []
    try:
        window.mode_combo.setCurrentText("Terrain")
        if args.style != "Stepped":
            window.terrain_style_combo.setCurrentText(args.style)
        window._on_tool_selected(args.tool)
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
        row = 0
        for brush_size in args.brush:
            window.brush_size_spin.setValue(brush_size)
            for delay in args.delays:
                output.append(f"--- brush {brush_size}, {args.px}px/move, {delay:g}ms between moves")
                output += _drag(window, record, paints, row=row % 4, px=args.px, delay_ms=delay, button=args.button)
                row += 1
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
    parser.add_argument("--tool", default="draw", choices=["draw", "elevation", "set_level"])
    parser.add_argument("--brush", type=_csv(int), default=[1, 5, 9], help="comma-separated brush sizes")
    parser.add_argument("--delays", type=_csv(float), default=[16.0], help="comma-separated ms between moves")
    parser.add_argument("--px", type=int, default=40, help="pixels per mouse move")
    parser.add_argument("--button", default="1", choices=["1", "3"], help="xdotool button: 1 left, 3 right")
    parser.add_argument("--settle", type=float, default=5.0, help="seconds to let the viewport warm before dragging")
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        sys.exit("needs a real X display (see the Xvfb recipe in --help)")

    # xcb, not offscreen: motion compression is the thing being measured.
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    qt_window.ensure_qapp()
    settings_isolation.pin_install_path()
    with tempfile.TemporaryDirectory(prefix="bench_gui_drag_") as tmp:
        settings_isolation.isolate_settings(Path(tmp))
        output = _run(args)
    print("\n".join(output))


if __name__ == "__main__":
    main()
