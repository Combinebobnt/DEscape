#!/usr/bin/env python3
"""Paint Can (flood fill) latency at full-map scale. Informational only,
matching bench_incremental_latency.py's own convention -- always runs, never
pass/fail, perf_counter, best-of-N.

descape/fill_tools.py's flood_fill_terrain() is O(tiles touched), but the
surrounding machinery -- EditHistory.apply()'s snapshot/diff and
ViewerWindow._apply_dirty()'s repaint path -- was sized for per-touch brush
edits, not a single click that can dirty every tile on the map. The Paint
Can implementation plan requires this measurement before any of
_apply_dirty()'s documented fallbacks (coalesced rects, a full-map-fallback
threshold, a busy-cursor wrap) ship -- no speculative optimization without a
real number from this script.

Drives a real offscreen ViewerWindow (QT_QPA_PLATFORM=offscreen) rather than
calling descape.render's lower-level functions directly the way
bench_incremental_latency.py does: _apply_dirty()'s Flat-mode path calls
real QGraphicsView.invalidate_region() once per dirty tile, and that Qt call
overhead is exactly what bucket 3 below needs to measure, not something a
bypass could approximate.

Cases: the shipped blank_120x120 template (14,400 tiles -- every fill's real
worst case on the smallest supported map) and an in-memory 480x480 blank map
(230,400 tiles, descape.scenario_new.load_blank_scenario) -- each in both
terrain styles. Four buckets per case, isolated with a fresh EditHistory/
window state so each is timed independently rather than accumulating:
  1. flood_fill_terrain() alone.
  2. EditHistory.apply()'s snapshot+diff overhead on top of (1).
  3. ViewerWindow._apply_dirty()'s repaint cost.
  4. End-to-end on_fill(), and undo() of that same record -- undo/redo push
     the identical dirty-index set through _apply_dirty() too, so a slow
     fill implies a slow undo.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from descape.edit_history import EditHistory
from descape.fill_tools import flood_fill_terrain
from descape.scenario_io import BLANK_TEMPLATE_PATH
from descape.scenario_new import load_blank_scenario

_FILL_TERRAIN = 15  # GRASS_1 -- any fixed non-zero TerrainId, same choice as this feature's gui tests

BEST_OF = 3


def _revert_to_blank(mm) -> None:
    """Undoes the previous bucket's fill by direct field writes -- no
    EditHistory/undo() involved, so it doesn't pollute the next bucket's
    timing or record count."""
    for tile in mm.terrain:
        tile.terrain_id = 0
        tile.layer = -1


def _best(samples: list[float]) -> float:
    return min(samples)


def _bench_case(window, label: str) -> str:
    from PyQt5.QtWidgets import QApplication

    mm = window.scenario.map_manager
    tiles = mm.map_width * mm.map_height
    lines = [f"  {label} ({tiles:,} tiles)"]

    for style in ("Flat", "Stepped"):
        window.terrain_style_combo.setCurrentText(style)
        QApplication.processEvents()

        fill_samples, apply_overhead_samples, apply_dirty_samples = [], [], []
        for _ in range(BEST_OF):
            _revert_to_blank(mm)

            t0 = time.perf_counter()
            filled = flood_fill_terrain(mm, 0, 0, _FILL_TERRAIN)
            fill_ms = (time.perf_counter() - t0) * 1000
            fill_samples.append(fill_ms)
            assert len(filled) == tiles, f"expected a full-map fill, got {len(filled)}/{tiles}"
            _revert_to_blank(mm)

            hist = EditHistory()
            t0 = time.perf_counter()
            dirty = hist.apply("bench", mm.terrain, lambda: flood_fill_terrain(mm, 0, 0, _FILL_TERRAIN))
            apply_total_ms = (time.perf_counter() - t0) * 1000
            apply_overhead_samples.append(apply_total_ms - fill_ms)

            t0 = time.perf_counter()
            window._apply_dirty(dirty)
            apply_dirty_samples.append((time.perf_counter() - t0) * 1000)

            _revert_to_blank(mm)

        # End-to-end, through the real on_fill()/undo() path -- best-of-BEST_OF,
        # each iteration starting from a clean revert (on_fill's own no-op
        # early-out would otherwise make every iteration after the first free).
        e2e_samples, undo_samples = [], []
        for _ in range(BEST_OF):
            _revert_to_blank(mm)
            window.edit_history.reset()

            t0 = time.perf_counter()
            window.on_fill(0, 0, 0)
            e2e_samples.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            window.undo()
            undo_samples.append((time.perf_counter() - t0) * 1000)

        lines.append(
            f"    {style:8s} flood_fill_terrain: {_best(fill_samples):8.2f}ms  "
            f"apply() overhead: {_best(apply_overhead_samples):8.2f}ms  "
            f"_apply_dirty: {_best(apply_dirty_samples):8.2f}ms  "
            f"on_fill e2e: {_best(e2e_samples):8.2f}ms  "
            f"undo: {_best(undo_samples):8.2f}ms"
        )
    _revert_to_blank(mm)
    window.edit_history.reset()
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["bench_fill_latency"])
    from descape.viewer import ViewerWindow

    output = []
    for label, path in [(f"blank_{BLANK_TEMPLATE_PATH.stem.split('_')[1]}", BLANK_TEMPLATE_PATH)]:
        window = ViewerWindow()
        window.load_scenario(path)
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("fill")
        window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(_FILL_TERRAIN))
        output.append(_bench_case(window, "blank_120x120"))
        window.edit_history.mark_saved()
        window.close()

    scenario = load_blank_scenario(480)
    window = ViewerWindow()
    window.scenario = scenario
    window._render_current()
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected("fill")
    window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(_FILL_TERRAIN))
    output.append(_bench_case(window, "blank_480x480"))
    window.edit_history.mark_saved()
    window.close()

    print("\n".join(output))
    del app


if __name__ == "__main__":
    main()
