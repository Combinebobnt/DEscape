#!/usr/bin/env python3
"""Verifies Phase 3's MapView<->render.py coupling in descape/viewer.py --
specifically Risk #6 from the parent isometric-Z-height plan: the
elevation snapshot (MapView._iso_elevations) and
IsoProjection (MapView._iso_proj) a Stepped-mode render carries must never
drift from what actually produced the pixels currently on screen, including
right after a live elevation edit made in Flat mode. tools/verify_iso_
geometry.py and tools/verify_iso_render.py check the projection math and
compositor in isolation; this script drives the real ViewerWindow headlessly
(QT_QPA_PLATFORM=offscreen) to exercise the actual wiring those two scripts
can't reach -- toolbar-combo-driven mode switches, live edits through
on_edit_stroke_*, and the edit-tool scope cut (decision #6).

Checks, per real example file that supports elevation editing
(terrain_write_supported and map_is_square -- the same gating
ViewerWindow._update_tool_enabled() already applies to the Elevate/Set
Elevation toolbar actions; files that don't support it are reported as
skipped, not failed):
  1. Elevation snapshot freshness -- switching to Stepped right after
     loading, and again after a live elevation edit made in Flat mode, both
     times MapView._iso_elevations exactly matches
     scenario.map_manager.terrain's current elevations.
  2. Pick round-trip on every edited tile -- after switching to Stepped
     post-edit, iso_geometry.screen_to_tile() at the exact center screen
     pixel of each edited tile's diamond returns that tile's own (x, y).
     Only the wiring is under test here (does it hand screen_to_tile a
     fresh elevation array) -- the projection math itself is already
     covered independently by verify_iso_geometry.py/verify_iso_render.py.
  3. Edit tools enabled in both styles -- Phase 3's decision #6 scope cut
     (Terrain/Elevate/Set Elevation disabled whenever Elevation View =
     Stepped) was lifted in Phase 4 once Stepped got its own bounded
     incremental redraw (descape.render.refresh_region_iso); this checks
     the toolbar actions reflect that in both styles now, not that Stepped
     is still gated off.
  4. A real live edit made *while already in Stepped mode* -- paints one
     tile and elevates another through the actual on_edit_stroke_* path
     with Elevation View already set to Stepped, then confirms: the
     displayed scene's actual pixels (captured via QGraphicsScene.render()
     over a fixed rect -- see _scene_rect_to_array(), item-agnostic so this
     keeps working across Phase B-C's MapCanvasItem, not just the older
     QGraphicsPixmapItem) changed at the edited region, not just the
     backing state, MapView._iso_elevations reflects the edit (same
     freshness property check 1 already covers for the Flat-then-switch
     case, exercised here for the edit-happens-in-Stepped case instead),
     and undoing restores the pre-edit pixels exactly. This is the one
     thing tools/verify_iso_incremental.py's Qt-free checks can't reach: it
     calls descape.render.refresh_region_iso()/composite_rect_iso() directly,
     never touching ViewerWindow._apply_dirty's Stepped branch or the real
     Qt paint path -- both only get exercised by driving the actual widget,
     which is what this script is for.

No code changes for the v2.6 contact-shadow feature -- painting it rides
the same real Qt paint path check 4 already drives (a shadow is composited
inside composite_rect_iso(), which _apply_dirty's Stepped branch already
calls), and its own bbox widening rides tools/verify_iso_incremental.py's
byte-identity bar (see that script's own note), not this one's pixel-
snapshot comparison. Nothing here is shadow-specific enough to need its
own check.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from descape import iso_geometry
from descape.viewer import ViewerWindow
from testkit import qt_capture


def _current_elevations(scenario) -> np.ndarray:
    mm = scenario.map_manager
    elevations = np.zeros((mm.map_height, mm.map_width), dtype=np.int64)
    for tile in mm.terrain:
        elevations[tile.y, tile.x] = tile.elevation
    return elevations


# Used by check 4, which needs to confirm an edit made real on-screen pixel
# changes, not just that ViewerWindow's backing state changed (a bug there
# could leave the backing state correct while the displayed pixels silently
# went stale). Item-agnostic: works the same whether the pixels come from a
# QGraphicsPixmapItem (Flat mode) or a custom-painted QGraphicsItem like
# Phase B-C's MapCanvasItem (Stepped mode) -- unlike reading
# mv._pixmap_item.pixmap() directly, which stopped existing for Stepped mode
# once Phase B-C replaced it.
_scene_rect_to_array = qt_capture.scene_rect_to_array


def check_file(path: Path) -> tuple[bool, str]:
    window = ViewerWindow()
    try:
        window.load_scenario(path)
        scenario = window.scenario
        if scenario is None:
            return False, "failed to load"
        if not (scenario.terrain_write_supported and scenario.map_is_square):
            return True, "skipped (file doesn't support elevation editing)"

        problems = []
        mv = window.map_view

        # 1a. Fresh load, switch to Stepped: snapshot must match load-time state.
        window.terrain_style_combo.setCurrentText("Stepped")
        expected = _current_elevations(scenario)
        if mv._iso_elevations is None or not np.array_equal(mv._iso_elevations, expected):
            problems.append("post-load Stepped snapshot doesn't match mm.terrain")

        # Back to Flat to make a live edit through the real edit-tool path
        # (on_edit_stroke_start/tile/end -- the same calls MapView's mouse
        # handlers make, just without synthesizing real QMouseEvents).
        window.terrain_style_combo.setCurrentText("Flat")
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        if not window.elevation_action.isEnabled():
            problems.append("elevation tool unexpectedly disabled in Flat + Terrain mode")

        mm = scenario.map_manager
        edited_tiles = [(0, 0), (min(3, mm.map_width - 1), 0), (0, min(2, mm.map_height - 1))]
        window.on_edit_stroke_start()
        for x, y in edited_tiles:
            window.on_edit_stroke_tile(x, y, 0)
        window.on_edit_stroke_end()

        # 1b. Switch to Stepped again: snapshot must reflect the just-applied edit.
        window.terrain_style_combo.setCurrentText("Stepped")
        expected = _current_elevations(scenario)
        if mv._iso_elevations is None or not np.array_equal(mv._iso_elevations, expected):
            problems.append("post-edit Stepped snapshot doesn't match mm.terrain")

        # 2. Pick round-trip: center pixel of each edited tile's own diamond.
        if mv._iso_proj is not None and mv._iso_elevations is not None:
            for x, y in edited_tiles:
                elevation = int(mv._iso_elevations[y, x])
                ox, oy = iso_geometry.tile_screen_origin(x, y, elevation, mv._iso_proj)
                sx, sy = ox + mv._iso_proj.half_w, oy + mv._iso_proj.half_h
                got = iso_geometry.screen_to_tile(sx, sy, mv._iso_elevations, mv._iso_proj)
                if got != (x, y):
                    problems.append(f"tile ({x},{y}): screen_to_tile at its own center pixel returned {got}")
        else:
            problems.append("Stepped mode has no _iso_proj/_iso_elevations after switching")

        # 3. Phase 4 lifted decision #6's scope cut: edit tools must be
        # enabled in Stepped too now, not disabled. (window is currently
        # Stepped, from step 1b above.)
        if not window.elevation_action.isEnabled():
            problems.append("elevation tool unexpectedly disabled while Terrain Style = Stepped (Phase 4)")
        window.terrain_style_combo.setCurrentText("Flat")
        if not window.elevation_action.isEnabled():
            problems.append("elevation tool didn't stay enabled after switching back to Flat")
        window.terrain_style_combo.setCurrentText("Stepped")

        # 4. A real live edit made while ALREADY in Stepped mode -- the one
        # path check 1's Flat-then-switch sequence and tools/verify_iso_
        # incremental.py's Qt-free checks both miss (see this script's own
        # module docstring). Two more, fresh tiles so this doesn't overlap
        # the edits already made in step 1 above.
        stepped_edit_tiles = [
            (min(5, mm.map_width - 1), min(1, mm.map_height - 1)),
            (min(1, mm.map_width - 1), min(5, mm.map_height - 1)),
        ]
        capture_rect = mv._map_rect  # fixed scene-space rect, valid before/after/undo -- map dims never change
        capture_before = _scene_rect_to_array(mv.scene(), capture_rect)
        window._on_tool_selected("elevation")
        if not window.elevation_action.isEnabled():
            problems.append("elevation tool unexpectedly disabled going into the Stepped live-edit check")
        window.on_edit_stroke_start()
        for x, y in stepped_edit_tiles:
            window.on_edit_stroke_tile(x, y, 0)
        window.on_edit_stroke_end()

        capture_after = _scene_rect_to_array(mv.scene(), capture_rect)
        if capture_after.shape != capture_before.shape:
            problems.append(f"Stepped live edit changed the captured shape: {capture_before.shape} -> {capture_after.shape}")
        elif np.array_equal(capture_after, capture_before):
            problems.append("Stepped live edit: displayed pixels didn't change at all")

        expected = _current_elevations(scenario)
        if mv._iso_elevations is None or not np.array_equal(mv._iso_elevations, expected):
            problems.append("Stepped live edit: MapView._iso_elevations doesn't match mm.terrain afterward")

        window.undo()
        capture_undone = _scene_rect_to_array(mv.scene(), capture_rect)
        if capture_undone.shape == capture_before.shape and not np.array_equal(capture_undone, capture_before):
            diff = int(np.count_nonzero(np.any(capture_undone != capture_before, axis=2)))
            problems.append(f"Stepped live edit: undo didn't restore the pre-edit pixels ({diff} pixels differ)")

        if problems:
            return False, "; ".join(problems)
        return True, (
            f"OK ({len(edited_tiles)} Flat-mode tiles round-tripped; both styles keep edit tools "
            f"enabled; {len(stepped_edit_tiles)} Stepped-mode tiles live-edited, displayed pixels "
            f"updated, undo restored)"
        )
    finally:
        # window.close() would run closeEvent()'s _confirm_discard_changes()
        # -- a modal QMessageBox that blocks forever with nothing to click it
        # in an offscreen script, since the elevation edits above always
        # leave edit_history dirty. Mark saved first so that check short-
        # circuits to True with no dialog; the window is discarded right
        # after anyway; no on-disk file is ever touched.
        window.edit_history.mark_saved()
        window.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario_dir", type=Path, nargs="?", default=ROOT / "examples", help="Directory of .aoe2scenario files"
    )
    args = parser.parse_args()

    files = sorted(args.scenario_dir.glob("*.aoe2scenario"))
    if not files:
        print(f"No .aoe2scenario files found in {args.scenario_dir}")
        sys.exit(1)

    # Held in a variable deliberately, even though never read again below --
    # an unassigned QApplication(...) call was seen to crash the very first
    # ViewerWindow() with "Must construct a QApplication before a QWidget"
    # while developing this script (PyQt5 doesn't keep it alive via its own
    # singleton registration alone). Do not "clean up" this assignment.
    app = QApplication.instance() or QApplication(sys.argv[:1])

    failures = 0
    for path in files:
        try:
            ok, detail = check_file(path)
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"{status}  {path.name:38s} {detail}")

    print(f"\n{len(files) - failures}/{len(files)} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
