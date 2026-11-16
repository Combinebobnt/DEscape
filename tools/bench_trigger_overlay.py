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

The paint section (TASK-508; `--sections paint`) repeats per style, Flat in
both Isometric View states, at `--zooms` factors of the fit view. Columns,
medians in ms, `rings` meaning the ring items alone (fill, outline and unit
outline, the other overlay items hidden):

- **full** and **64x64**: a full viewport repaint() and a centred 64x64 one,
  overlay on, off, and rings;
- **pan**: rings, a full repaint after each scroll of 0.4 of the viewport's
  longer side between two positions, so a ring memo misses every time (`-`
  without the scroll range, `!hit` if a paint reused the memo anyway);
- **level**: rings, the first full repaint at each zoom in ascending order
  after a refresh painted once at the smallest zoom, so each zoom builds
  whatever level it needs first;
- **lod px**: pixels of a whole-overlay grab that differ from the same grab
  with each ring item swapped for a plain QGraphicsPathItem of its path;
- **cold** rows at x1 and x16: the first full repaint after
  `_refresh_trigger_overlay()`, rings and the whole overlay.
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


RING_ROLES = ("fill_", "outline_", "units_")


def _paint_section(window, selection, styles, zooms, runs) -> None:
    """The TASK-508 paint rows (module docstring). Runs on any checkout: ring
    items are found by their data(0) role, not their class."""
    import numpy as np
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QApplication, QGraphicsPathItem

    view = window.map_view
    viewport = view.viewport()
    small = QRect(viewport.width() // 2 - 32, viewport.height() // 2 - 32, 64, 64)
    variants = []
    for style in styles:
        if style == "Flat":
            variants += [("Flat iso", "Flat", True), ("Flat top", "Flat", False)]
        else:
            variants.append((style, style, None))

    def ring_items():
        return [i for i in view.trigger_overlay_items() if str(i.data(0)).startswith(RING_ROLES)]

    def rings_only():
        for item in view.trigger_overlay_items():
            item.setVisible(str(item.data(0)).startswith(RING_ROLES))

    def refresh():
        window._trigger_overlay_ref = (tuple(selection), frozenset())
        window._refresh_trigger_overlay()

    def grab():
        image = viewport.grab().toImage().convertToFormat(QImage.Format_ARGB32)
        bits = image.constBits()
        bits.setsize(image.byteCount())
        rows = np.frombuffer(bits, np.uint8).reshape(image.height(), image.bytesPerLine())
        return rows[:, : image.width() * 4].reshape(image.height(), image.width(), 4).copy()

    def lod_pixels() -> int:
        """Ring items against plain items of the same exact path, whole overlay on."""
        repaint()
        ours = grab()
        plain = []
        for item in ring_items():
            twin = QGraphicsPathItem(item.path())
            twin.setPen(item.pen())
            twin.setBrush(item.brush())
            twin.setZValue(item.zValue())
            twin.setOpacity(item.opacity())
            view.scene().addItem(twin)
            item.setVisible(False)
            plain.append((item, twin))
        repaint()
        exact = grab()
        for item, twin in plain:
            view.scene().removeItem(twin)
            item.setVisible(True)
        return int(np.any(ours != exact, axis=2).sum())

    def repaint(rect=None):
        viewport.update()
        if rect is None:
            viewport.repaint()
        else:
            viewport.repaint(rect)

    def timed(fn):
        fn()  # warm
        return _median_ms(fn, runs)

    for label, style, isometric in variants:
        window.terrain_style_combo.setCurrentText(style)
        QApplication.processEvents()
        if isometric is not None:
            window.iso_action.setChecked(isometric)
            QApplication.processEvents()
        view.set_isometric(view._isometric)
        QApplication.processEvents()
        fit = view.transform()
        centre = view._viewport_center_scene()

        def set_zoom(factor, settle=True, fit=fit, centre=centre):
            view.setTransform(fit)
            view.scale(factor, factor)
            view._center_on_exact(centre)
            if settle:
                QApplication.processEvents()

        refresh()
        QApplication.processEvents()
        print(f"{label:8s} paint section: fit m11 {fit.m11():.4f}, {len(ring_items())} ring items | loadavg {_loadavg()}")
        for factor in zooms:
            set_zoom(factor)
            view.set_trigger_overlay_enabled(True)
            on = (timed(repaint), timed(lambda: repaint(small)))
            view.set_trigger_overlay_enabled(False)
            off = (timed(repaint), timed(lambda: repaint(small)))
            view.set_trigger_overlay_enabled(True)
            rings_only()
            ring = (timed(repaint), timed(lambda: repaint(small)))
            h = view.horizontalScrollBar()
            # The memo cell can reach 0.354 of the longer side, so a shorter scroll may keep its key.
            step = int(max(viewport.width(), viewport.height()) * 0.4)
            if h.maximum() - h.minimum() > step:
                home = h.value()
                away = home + step if home + step <= h.maximum() else home - step
                spots = [away, home]
                sets = {id(s): s for s in (getattr(i, "ring_set", lambda: None)() for i in ring_items()) if s is not None}
                pans = [0]

                def pan(h=h, spots=spots, pans=pans):
                    h.setValue(spots[0])
                    spots.reverse()
                    pans[0] += 1
                    repaint()

                missed = sum(s.memo_misses for s in sets.values())
                panned = f"{timed(pan):6.1f}"
                missed = sum(s.memo_misses for s in sets.values()) - missed
                if sets and missed < len(sets) * pans[0]:
                    panned += "!hit"  # some pan paint reused the memo, so the row is not all misses
                h.setValue(home)
            else:
                panned = f"{'-':>6s}"
            view.set_trigger_overlay_enabled(True)
            pixels = lod_pixels()
            print(
                f"{label:8s} x{factor:<4g} full on/off/rings {on[0]:6.1f} {off[0]:5.1f} {ring[0]:6.1f}"
                f" | 64x64 {on[1]:6.1f} {off[1]:5.1f} {ring[1]:6.1f} | pan rings {panned}"
                f" | lod px {pixels:6d} | loadavg {_loadavg()}"
            )
        # Level row: per run, a fresh refresh painted at the smallest zoom, then each zoom's first paint.
        firsts: dict[float, list[float]] = {f: [] for f in zooms[1:]}
        for _ in range(runs):
            set_zoom(zooms[0])
            refresh()
            rings_only()
            repaint()
            for factor in zooms[1:]:
                set_zoom(factor, settle=False)
                t0 = time.perf_counter()
                repaint()
                firsts[factor].append((time.perf_counter() - t0) * 1000)
                QApplication.processEvents()
        cells = " ".join(f"x{f:g} {statistics.median(t):6.1f}" for f, t in firsts.items())
        print(f"{label:8s} level rings  {cells} | loadavg {_loadavg()}")
        for factor in (1, 16):
            set_zoom(factor)
            for name, hide in (("rings", True), ("overlay", False)):
                times = []
                for _ in range(runs):
                    refresh()
                    if hide:
                        rings_only()
                    t0 = time.perf_counter()
                    repaint()
                    times.append((time.perf_counter() - t0) * 1000)
                    QApplication.processEvents()
                print(f"{label:8s} cold x{factor:<3g} {name:8s} {statistics.median(times):6.1f} ms | loadavg {_loadavg()}")
        window._trigger_overlay_ref = None
        window._refresh_trigger_overlay()
        QApplication.processEvents()
    if "Flat" in styles:
        window.iso_action.setChecked(True)


def main() -> None:
    root_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario", nargs="?", type=Path, default=root_default / "examples" / "old-allies-final-v2.aoe2scenario"
    )
    parser.add_argument("--root", type=Path, help="DEscape checkout to import descape from")
    parser.add_argument("--runs", type=int, default=7, help="Timed calls per median (default 7)")
    parser.add_argument("--styles", default=",".join(STYLES), help="Comma-separated subset of Flat,Stepped,Sloped")
    parser.add_argument("--sections", default="build,paint", help="Comma-separated subset of build,paint")
    parser.add_argument("--zooms", default="0.5,1,4,16,64", help="Paint section zooms, factors of the fit view")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    if not styles or any(s not in STYLES for s in styles):
        parser.error("--styles takes Flat, Stepped and/or Sloped")
    sections = {s.strip() for s in args.sections.split(",") if s.strip()}
    if not sections or sections - {"build", "paint"}:
        parser.error("--sections takes build and/or paint")
    zooms = sorted(float(z) for z in args.zooms.split(",") if z.strip())

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

        for style in styles if "build" in sections else ():
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
        if "paint" in sections:
            _paint_section(window, selection, styles, zooms, args.runs)
    finally:
        window.edit_history.mark_saved()
        window.close()


if __name__ == "__main__":
    main()
