#!/usr/bin/env python3
"""Trigger overlay rings (TASK-355): what a large trigger union costs to build
and to paint, per terrain style.

On one scenario, the union selection is the largest prefix of triggers whose
drawable shapes stay within `viewer_trigger_overlay.MAX_OVERLAY_SHAPES` (walls are never drawn,
so they do not count), as Select All up to the cap would draw. Printed once:

- the selection's shapes, rects (areas plus unit footprints on the map),
  distinct rects, and border tiles (`_region_edge_walk()` entries over every
  rect, and over the distinct ones);
- **derive**: `parse_triggers()` plus `shapes_for_trigger()` over the
  selection, no MapView: the part of a refresh that runs with no ring at all.

Then per style, medians of `--runs`:

- **ring**: one whole-map `MapView._rect_ring_path()`;
- **refresh**: the union's `ViewerWindow._refresh_trigger_overlay()`, which is
  also what a unit edit with the union held runs;
- **refresh cold**: the same refresh, with the ring memo cleared before timing;
- **refresh nofp**: warm refresh with border fingerprint validation bypassed
  (only when the imported tree exposes the validation seam);
- **elev step**: the trigger rebuild alone after a real one-tile elevation
  stroke on a union rect's border, undone between calls (Stepped and Sloped);
- **paint**: `MapView.render()` of the 900x700 window into a QImage, union held.

Refresh rows also show refresh minus derive, and its x4 cost in brackets.
Every timing line names the load average; the header names the Qt platform.
Read medians, and compare two checkouts on one machine at one time.

`--root DIR` imports descape (and testkit) from another checkout, so one copy
of this script times a baseline worktree and a lane branch alike. Settings are
isolated (pin_install_path() + isolate_settings()), so nothing here writes the
real config.yaml.

Informational only, matching the other tools/bench_*.py: always runs, never
pass/fail.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STYLES = ("Flat", "Stepped", "Sloped")


def _loadavg() -> str:
    try:
        return Path("/proc/loadavg").read_text().strip()
    except OSError:
        return "?"


def _median_ms(fn, runs: int) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> None:
    root_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario", nargs="?", type=Path, default=root_default / "examples" / "old-allies-final-v2.aoe2scenario"
    )
    parser.add_argument("--root", type=Path, help="DEscape checkout to import descape from")
    parser.add_argument("--runs", type=int, default=7, help="Timed calls per median (default 7)")
    parser.add_argument("--styles", default=",".join(STYLES), help="Comma-separated subset of Flat,Stepped,Sloped")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    if not styles or any(s not in STYLES for s in styles):
        parser.error("--styles takes Flat, Stepped and/or Sloped")

    root = (args.root or root_default).resolve()
    sys.path.insert(0, str(root))
    from testkit import qt_window, settings_isolation

    settings_isolation.pin_install_path()
    settings_dir = tempfile.TemporaryDirectory(prefix="bench-trigger-overlay-")  # removed at exit (TASK-606)
    settings_isolation.isolate_settings(Path(settings_dir.name))
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage, QPainter
    from PyQt5.QtWidgets import QApplication

    from descape import library_compat, trigger_geometry, viewer_trigger_overlay
    from descape.scenario_io import parse_triggers

    qt_window.ensure_qapp()
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    print(
        f"descape from {root} ({sha.stdout.strip() or '?'}); {args.scenario.name}; "
        f"platform {QApplication.platformName()}; median of {args.runs}"
    )

    window = qt_window.stepped_window(args.scenario, show=False)
    try:
        window.resize(900, 700)
        qt_window.show_and_settle(window)
        window.mode_combo.setCurrentText("Triggers")
        QApplication.processEvents()
        view = window.map_view
        loaded = window.scenario
        vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
        references = window._unit_reference_index()
        manager = parse_triggers(loaded)

        def drawable(index):
            return [
                s
                for s in trigger_geometry.shapes_for_trigger(
                    manager.triggers[index], vocabulary, trigger_index=index, references=references
                )
                if s.shape != trigger_geometry.SHAPE_WALL
            ]

        selection: list[int] = []
        shapes = []
        for index in range(len(manager.triggers)):
            more = drawable(index)
            if len(shapes) + len(more) > viewer_trigger_overlay.MAX_OVERLAY_SHAPES:
                break
            shapes.extend(more)
            selection.append(index)
        rects = [
            r
            for r in (
                view._trigger_area_rect(s.coords)
                for s in shapes
                if s.shape in (trigger_geometry.SHAPE_AREA, trigger_geometry.SHAPE_UNIT)
            )
            if r is not None
        ]

        def border(rs):
            return sum(2 * (r[2] - r[0]) + 2 * (r[3] - r[1]) for r in rs)

        print(
            f"union: {len(selection)} triggers, {len(shapes)} shapes, {len(rects)} rects "
            f"({len(set(rects))} distinct), {border(rects)} border tiles ({border(set(rects))} distinct)"
        )

        def derive():
            fresh = parse_triggers(loaded)
            for index in selection:
                list(
                    trigger_geometry.shapes_for_trigger(
                        fresh.triggers[index], vocabulary, trigger_index=index, references=references
                    )
                )

        derived = _median_ms(derive, args.runs)
        print(f"derive {derived:.1f} ms | loadavg {_loadavg()}")

        whole = (0, 0, loaded.map_manager.map_width, loaded.map_manager.map_height)
        image = QImage(900, 700, QImage.Format_ARGB32_Premultiplied)

        def paint():
            image.fill(0)
            painter = QPainter(image)
            view.render(painter)
            painter.end()

        for style in styles:
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            ring = _median_ms(lambda: view._rect_ring_path(whole), args.runs)
            print(f"{style:8s} ring         {ring:6.1f} ms | loadavg {_loadavg()}")
            window._trigger_overlay_ref = (tuple(selection), frozenset())
            window._refresh_trigger_overlay()

            def report(name, elapsed, *, subtract_derive=True, style=style):
                net = elapsed - derived if subtract_derive else elapsed
                detail = f"  minus derive {net:6.1f} [{net * 4:6.1f} x4]" if subtract_derive else f" [{net * 4:6.1f} x4]"
                print(f"{style:8s} {name:12s} {elapsed:6.1f} ms{detail} | loadavg {_loadavg()}")

            refresh = _median_ms(window._refresh_trigger_overlay, args.runs)
            report("refresh", refresh)
            assert view.trigger_overlay_items(), f"{style}: the union drew nothing"

            times = []
            for _ in range(args.runs):
                memo = getattr(view, "_trigger_ring_memo", None)
                if memo is not None:
                    memo.clear()
                t0 = time.perf_counter()
                window._refresh_trigger_overlay()
                times.append((time.perf_counter() - t0) * 1000)
            report("refresh cold", statistics.median(times))

            validate = getattr(view, "_ring_fingerprint_matches", None)
            if validate is not None:
                view._ring_fingerprint_matches = set
                try:
                    report("refresh nofp", _median_ms(window._refresh_trigger_overlay, args.runs))
                finally:
                    view._ring_fingerprint_matches = validate

            if style != "Flat" and rects:
                x, y, _x1, _y1 = rects[0]
                window.mode_combo.setCurrentText("Terrain")
                window._on_tool_selected("elevation")
                window.brush_size_spin.setValue(1)
                QApplication.processEvents()
                times = []
                for _ in range(args.runs):
                    before = loaded.map_manager.get_tile(x, y).elevation
                    rebuild = view._rebuild_trigger_overlay
                    # The stroke patches heights, but leaves its trigger rebuild for the timer.
                    view._rebuild_trigger_overlay = lambda: None
                    try:
                        window.on_edit_stroke_start()
                        window.on_edit_stroke_tile(x, y, Qt.NoModifier)
                        window.on_edit_stroke_end()
                    finally:
                        view._rebuild_trigger_overlay = rebuild
                    assert loaded.map_manager.get_tile(x, y).elevation == before + 1, "border tile did not rise one step"
                    t0 = time.perf_counter()
                    view._rebuild_trigger_overlay()
                    times.append((time.perf_counter() - t0) * 1000)
                    window._move_history(window.edit_history.peek_undo(), window.edit_history.undo, "Undo")
                    assert loaded.map_manager.get_tile(x, y).elevation == before, "undo did not restore the border tile"
                    QApplication.processEvents()
                report("elev step", statistics.median(times), subtract_derive=False)
                window.mode_combo.setCurrentText("Triggers")
                # Entering Triggers repopulates the panel and replaces the bench's selection.
                window._trigger_overlay_ref = (tuple(selection), frozenset())
                window._refresh_trigger_overlay()
                QApplication.processEvents()

            painted = _median_ms(paint, args.runs)
            print(f"{style:8s} paint        {painted:6.1f} ms | loadavg {_loadavg()}")
            window._trigger_overlay_ref = None
            window._refresh_trigger_overlay()
            QApplication.processEvents()
    finally:
        window.edit_history.mark_saved()
        window.close()


if __name__ == "__main__":
    main()
