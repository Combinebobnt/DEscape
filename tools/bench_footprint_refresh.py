#!/usr/bin/env python3
"""Times View > Footprint Outlines' refresh_footprint_overlay() per edit, offscreen,
through a real ViewerWindow, for Stepped and Sloped at scope Multi-tile and All:

  (a) a full rebuild (the chunk planner reset first, where the build has one);
  (b) one Move of a mid-map building by one tile (_move_units), then its undo;
  (c) one Set Elevation drag step at brush 9 (on_edit_stroke_tile() plus
      processEvents()), only the deferred flush's footprint refresh;
  (d) a refresh with nothing changed.

Every refresh_footprint_overlay() call an action makes is summed; the ring
layer and the selection refresh that share the flush are not timed. Raw and x4
medians, /proc/loadavg per case. `--chunk N` overrides
outline_chunks.CHUNK_TILES. DESCAPE_BENCH_ROOT points the import at another
checkout's descape/ and testkit/, so one script drives base and head; run them
interleaved. Informational, never pass/fail.

  .venv/bin/python3 tools/bench_footprint_refresh.py [--scenario PATH] [--chunk N]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = Path(os.environ.get("DESCAPE_BENCH_ROOT") or ROOT).resolve()
sys.path.insert(0, str(CODE_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from testkit import qt_window, settings_isolation

DEFAULT_SCENARIO = ROOT / "examples" / "old-allies-final-v2.aoe2scenario"
STYLES = ("Stepped", "Sloped")
BRUSH = 9


def _load() -> str:
    return Path("/proc/loadavg").read_text().split(" ")[0]


def _row(label: str, samples: list[float]) -> str:
    median = statistics.median(samples)
    return f"    {label:<26} median {median:8.2f} ms (x4 {4 * median:8.2f})  min {min(samples):8.2f}  n={len(samples)}"


class _Timer:
    """Wraps the view's refresh_footprint_overlay() and sums every call between take()s."""

    def __init__(self, view) -> None:
        self._real = view.refresh_footprint_overlay
        self._ms = 0.0
        view.refresh_footprint_overlay = self._call

    def _call(self) -> None:
        started = time.perf_counter()
        self._real()
        self._ms += (time.perf_counter() - started) * 1000

    def take(self) -> float:
        ms, self._ms = self._ms, 0.0
        return ms


def _mid_building(view):
    """The multi-tile non-GAIA unit nearest the map centre."""
    from descape import unit_pick

    cx, cy = view._map_width / 2, view._map_height / 2
    entries = [
        e for e in unit_pick.footprint_entries(view._unit_index, unit_pick.FOOTPRINT_SCOPE_MULTITILE) if e.player_id
    ]
    entry = min(entries, key=lambda e: (e.own_x - cx) ** 2 + (e.own_y - cy) ** 2)
    return unit_pick.unit_key(entry.player_id, entry.unit)


def _reset_planner(view) -> None:
    chunks = getattr(view, "_footprint_chunks", None)
    if chunks is not None:
        chunks.reset()


def _case(window, timer: _Timer, repeats: int) -> list[str]:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    view = window.map_view
    full, move, undo, step, idle = [], [], [], [], []
    for _ in range(repeats):
        _reset_planner(view)
        timer.take()
        view.refresh_footprint_overlay()
        full.append(timer.take())
        view.refresh_footprint_overlay()
        idle.append(timer.take())

    key = _mid_building(view)
    window.mode_combo.setCurrentText("Units")
    QApplication.processEvents()
    for _ in range(repeats):
        entry = view._unit_index.entry_for_key(key)
        timer.take()
        window._move_units(window._ensure_unit_edits(), [entry], 1.0, 0.0, "Move unit")
        QApplication.processEvents()
        move.append(timer.take())
        window.undo()
        QApplication.processEvents()
        undo.append(timer.take())

    entry = view._unit_index.entry_for_key(key)
    x0, y0 = entry.own_x - 4, entry.own_y
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected("elevation")
    window.brush_size_spin.setValue(BRUSH)
    QApplication.processEvents()
    timer.take()
    window.on_edit_stroke_start()
    for k in range(repeats * 2):
        window.on_edit_stroke_tile(x0 + k, y0, Qt.NoModifier)
        QApplication.processEvents()
        step.append(timer.take())
    window.on_edit_stroke_end()
    QApplication.processEvents()
    window.undo()
    QApplication.processEvents()
    timer.take()
    window.mode_combo.setCurrentText("Units")
    QApplication.processEvents()
    return [
        _row("(a) full rebuild", full),
        _row("(b) one-unit Move", move),
        _row("(b') its undo", undo),
        _row(f"(c) brush-{BRUSH} elevation step", step),
        _row("(d) unchanged refresh", idle),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--chunk", type=int, default=None, help="override outline_chunks.CHUNK_TILES")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--styles", default=",".join(STYLES))
    args = parser.parse_args()

    from PyQt5.QtWidgets import QApplication

    from descape import unit_pick

    if args.chunk is not None:
        from descape import outline_chunks

        outline_chunks.CHUNK_TILES = args.chunk
    try:
        from descape import outline_chunks as chunks_module

        chunk_tiles = chunks_module.CHUNK_TILES
    except ImportError:
        chunk_tiles = None
    qt_window.ensure_qapp()
    print(f"# {CODE_ROOT}, {args.scenario.name}, CHUNK_TILES {chunk_tiles}, repeats {args.repeats}, load {_load()}")
    with tempfile.TemporaryDirectory() as tmp:
        settings_isolation.pin_install_path()
        settings_isolation.isolate_settings(Path(tmp))
        window = qt_window.stepped_window(args.scenario, size=1200)
        try:
            window.footprint_action.setChecked(True)
            QApplication.processEvents()
            view = window.map_view
            timer = _Timer(view)
            for style in args.styles.split(","):
                if view._terrain_style != style.lower():
                    window.terrain_style_combo.setCurrentText(style)
                    window.refresh_map()
                    QApplication.processEvents()
                for scope in (unit_pick.FOOTPRINT_SCOPE_MULTITILE, unit_pick.FOOTPRINT_SCOPE_ALL):
                    view.set_footprint_scope(scope)
                    count = len(unit_pick.footprint_entries(view._unit_index, scope))
                    print(f"  {style} {scope}: {count} entries, load {_load()}")
                    for line in _case(window, timer, args.repeats):
                        print(line, flush=True)
                view.set_footprint_scope(unit_pick.FOOTPRINT_SCOPE_MULTITILE)
        finally:
            window.edit_history.mark_saved()
            window.close()


if __name__ == "__main__":
    main()
