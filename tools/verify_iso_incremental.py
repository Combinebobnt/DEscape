#!/usr/bin/env python3
"""Verifies descape/render.py's refresh_region_iso() -- Phase 4 of the real
isometric Z-height terrain rendering plan.
Phase 2's tools/verify_iso_render.py checks the full-map compositor;
Phase 1's tools/verify_iso_geometry.py checks the projection primitives in
isolation. This script checks the *incremental* path Phase 4 adds on top of
both: does a bounded-region patch, after a real edit, reproduce exactly what
a full re-composite would show, everywhere -- not just inside the tiles the
edit directly touched.

Byte-identity against a full re-composite, not visual similarity, is the
acceptance bar throughout, per the parent plan's own Phase 4 section.

Checks, per real example file that supports elevation editing
(terrain_write_supported and map_is_square -- the same gate ViewerWindow._
update_tool_enabled() applies to the Elevate/Set Elevation toolbar actions):
  1. Incremental-vs-full: a scripted sequence of edits (terrain paint,
     elevation raise, elevation lower, a large Set-Elevation jump), each
     applied via descape.edit_history.EditHistory the same way viewer.py's
     stroke handling does, each immediately checked byte-identical to an
     independent fresh render_terrain_iso() call after that one edit.
  2. Undo/redo: undoing every edit above via refresh_region_iso, one at a
     time, restores byte-identity to the pre-edit render; redoing restores
     byte-identity to the fully-edited state.
  3. Out-of-range fallback: an elevation outside [proj.min_elev, proj.
     max_elev] (never reachable through the real UI, which clamps -- see
     viewer.py's on_edit_stroke_tile -- but not a precondition this
     function should trust blindly) makes refresh_region_iso() return None
     without touching img at all, rather than computing a corrupting
     out-of-canvas write.

Per-touched-tile incremental latency across every real example file used to
be reported here (informational, not pass/fail) -- moved to
tools/bench_incremental_latency.py (the pytest migration plan's Ordering
step 4), since it was never a check_ prefixed pass/fail check to begin with.

No code changes for the v2.6 contact-shadow feature (descape.iso_geometry.
shadow_quad_indices) -- it rides the SAME byte-identity bar this script
already enforces, for free: dirty_screen_bbox_iso()'s seed-tile sweep goes
through iso_geometry.tile_screen_bounds_swept(), widened for the shadow's
own up-screen reach (see that function's docstring), so an edit's dirty
bbox already covers any shadow it could cast or newly expose. If that
widening were ever wrong, THIS script's byte-identity check is what would
catch it -- an incremental patch differing from a fresh full render exactly
where a shadow was missed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from descape import iso_geometry
from descape.edit_history import EditHistory
from descape.elevation_tools import set_tile_elevation
from descape.render import (
    refresh_region_iso,
    render_terrain_iso,
    render_terrain_iso_with_proj,
    tile_pixels_for_map,
)
from descape.scenario_io import load_map_and_units


def _pick_different_terrain(current: int) -> int:
    return 15 if current == 2 else 2  # GRASS_1 / BEACH, both valid in every DE structure version


def _editable(scenario) -> bool:
    return scenario.terrain_write_supported and scenario.map_is_square


def _scripted_ops(mm):
    """(label, apply_fn) pairs -- apply_fn(tiles) mutates mm's tiles exactly
    the way viewer.py's on_edit_stroke_tile does for each tool, in a fixed,
    arbitrary-but-real sequence covering terrain paint, a small elevation
    raise (with whatever neighbor propagation MapManager._elevation_tile_
    recursion does), a small lower, and a large absolute jump (bigger skirt
    drops, more likely to touch the lateral-expansion path). A center-ish
    tile, not a corner, so propagation isn't artificially clipped by the
    map edge on every file."""
    w, h = mm.map_width, mm.map_height
    cx, cy = w // 2, h // 2
    paint_a = (min(2, w - 1), 0)
    paint_b = (0, min(2, h - 1))

    def op_paint():
        for x, y in (paint_a, paint_b):
            tile = mm.get_tile(x, y)
            tile.terrain_id = _pick_different_terrain(tile.terrain_id)
            tile.layer = -1

    def op_raise():
        tile = mm.get_tile(cx, cy)
        set_tile_elevation(mm, cx, cy, min(iso_geometry.MAX_ELEVATION, tile.elevation + 1))

    def op_lower():
        tile = mm.get_tile(cx, cy)
        set_tile_elevation(mm, cx, cy, max(iso_geometry.MIN_ELEVATION, tile.elevation - 1))

    def op_big_jump():
        tx, ty = min(cx + 3, w - 1), min(cy + 3, h - 1)
        tile = mm.get_tile(tx, ty)
        target = iso_geometry.MAX_ELEVATION if tile.elevation < iso_geometry.MAX_ELEVATION else iso_geometry.MIN_ELEVATION
        set_tile_elevation(mm, tx, ty, target)

    return [("terrain paint", op_paint), ("elevation raise", op_raise), ("elevation lower", op_lower), ("big jump", op_big_jump)]


def check_incremental_and_undo_redo(path: Path) -> tuple[bool, str]:
    scenario = load_map_and_units(path)
    if not _editable(scenario):
        return True, "skipped (file doesn't support elevation editing)"
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)

    img, elevations, proj = render_terrain_iso_with_proj(scenario)
    pre_edit_baseline = img.copy()

    hist = EditHistory()
    ops = _scripted_ops(mm)
    post_edit_snapshots = []  # img.copy() after each op, for the redo check
    problems = []

    for label, apply_fn in ops:
        hist.begin_stroke(mm.terrain)
        apply_fn()
        dirty = hist.commit_stroke(label, mm.terrain)
        if not dirty:
            problems.append(f"[{label}] produced no dirty tiles -- expected a real change")
            continue
        bbox = refresh_region_iso(img, scenario, dirty, elevations, proj, tile_px)
        if bbox is None:
            problems.append(f"[{label}] refresh_region_iso declined unexpectedly (out-of-range guard?)")
            continue
        full = render_terrain_iso(scenario)
        if img.shape != full.shape:
            problems.append(f"[{label}] shape drifted: incremental {img.shape} vs full {full.shape}")
        elif not np.array_equal(img, full):
            diff = int(np.count_nonzero(np.any(img != full, axis=2)))
            problems.append(f"[{label}] incremental result differs from a full re-composite at {diff} pixels")
        post_edit_snapshots.append(img.copy())

    # Undo every op, one at a time, via the same incremental path -- must
    # land back on the pre-edit render exactly.
    for _ in ops:
        dirty = hist.undo(mm.terrain)
        if not dirty:
            continue
        bbox = refresh_region_iso(img, scenario, dirty, elevations, proj, tile_px)
        if bbox is None:
            problems.append("undo step: refresh_region_iso declined unexpectedly")
    if not np.array_equal(img, pre_edit_baseline):
        diff = int(np.count_nonzero(np.any(img != pre_edit_baseline, axis=2)))
        problems.append(f"undo-to-start doesn't match the pre-edit render at {diff} pixels")

    # Redo every op back -- must land on the same post-edit state recorded
    # above (the final entry, i.e. after every op was applied).
    for _ in ops:
        dirty = hist.redo(mm.terrain)
        if not dirty:
            continue
        bbox = refresh_region_iso(img, scenario, dirty, elevations, proj, tile_px)
        if bbox is None:
            problems.append("redo step: refresh_region_iso declined unexpectedly")
    if post_edit_snapshots and not np.array_equal(img, post_edit_snapshots[-1]):
        diff = int(np.count_nonzero(np.any(img != post_edit_snapshots[-1], axis=2)))
        problems.append(f"redo-to-end doesn't match the fully-edited state at {diff} pixels")

    if problems:
        return False, "; ".join(problems)
    return True, f"OK ({len(ops)} ops, each byte-identical to a full re-composite; undo/redo both exact)"


def check_out_of_range_fallback(path: Path) -> tuple[bool, str]:
    scenario = load_map_and_units(path)
    if not _editable(scenario):
        return True, "skipped (file doesn't support elevation editing)"
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    img, elevations, proj = render_terrain_iso_with_proj(scenario)
    before = img.copy()

    # Never reachable through the real UI (viewer.py's Elevate/Set Elevation
    # both clamp to [0, MAX_ELEVATION]) -- deliberately bypassing that clamp
    # here, the same way a hand-edited or third-party file could produce an
    # out-of-range value this function must still degrade safely against.
    tile = mm.get_tile(0, 0)
    tile.elevation = proj.max_elev + 5
    dirty_indices = [0]

    result = refresh_region_iso(img, scenario, dirty_indices, elevations, proj, tile_px)
    problems = []
    if result is not None:
        problems.append(f"expected None (out-of-range guard), got a bbox {result}")
    if not np.array_equal(img, before):
        problems.append("img was mutated despite the out-of-range guard rejecting the edit")

    if problems:
        return False, "; ".join(problems)
    return True, "OK (out-of-range edit correctly declined, img untouched)"


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

    checks = [
        ("incremental vs full + undo/redo", check_incremental_and_undo_redo),
        ("out-of-range fallback", check_out_of_range_fallback),
    ]

    failures = 0
    for path in files:
        for label, check in checks:
            try:
                ok, detail = check(path)
            except Exception as e:
                ok, detail = False, f"{type(e).__name__}: {e}"
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"{status}  {path.name:38s} [{label:30s}] {detail}")

    print(f"\n{len(files) * len(checks) - failures}/{len(files) * len(checks)} checks passed")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
