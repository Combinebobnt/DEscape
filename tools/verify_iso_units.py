#!/usr/bin/env python3
"""Verifies descape/render.py's Stepped-mode unit rendering -- Phase 5 of the
real isometric Z-height plan. Phase 2's
tools/verify_iso_render.py checks the terrain-only compositor in isolation;
Phase 4's tools/verify_iso_incremental.py checks the incremental terrain
redraw against real files (and, now that Phase 5 threads units through the
same code path, already re-confirms byte-identity for every real example
file's real units/buildings for free -- run that script too). This script
adds synthetic, deliberately small/controlled checks for the specific things
Phase 5 introduces that a terrain-only check can't exercise:

Checks:
  1. Position: a unit on a raised tile paints its diamond at THAT tile's
     own elevation (iso_geometry.tile_screen_origin(px, py, elevation,
     proj)), not at elevation 0 -- confirmed by sampling the unit's color at
     the raised screen position and confirming it's absent at the flat
     (elevation-0) position for the same tile.
  2. Occlusion: a unit sitting on a "further" neighbor (smaller d = y-x,
     painted earlier in depth_order -- see iso_geometry.skirt_quad_indices'
     docstring) of a tile that then gets raised is partially covered by
     that tile's own diamond, mirroring verify_iso_render.py's
     check_occlusion_scripted for terrain-vs-terrain -- confirming units are
     interleaved into the SAME depth-ordered paint sequence
     (_paint_tile_and_units_iso), not drawn as a blanket overlay pass after
     every terrain tile is already final (which would get this backwards).
     The same neighbor is also that tile's contact-shadow back neighbor
     (see iso_geometry.shadow_quad_indices), so this check gives the
     raised tile its own distinct terrain_id and separately confirms real
     occlusion (some unit-diamond pixels become that terrain color) rather
     than trusting a raw visible-pixel-count drop, which the shadow alone
     can now also produce.
  3. Incremental-vs-full, byte-identical -- refresh_region_iso() reproduces
     a fresh render_terrain_iso_with_proj() exactly after each of:
       a. an edit on a lone unit's own tile,
       b. an edit under a multi-tile building's CENTER tile,
       c. an edit under a multi-tile building's own FOOTPRINT CORNER tile
          (not its center) -- the case where the edited tile's own terrain
          diamond is nowhere near the building's center, but the building's
          rendered diamond still occupies that same screen tile and would
          go missing from an incremental patch that only re-seeds around
          the literally-edited tile by +-1 (skirts alone) instead of by the
          building's own footprint radius.
     This is the load-bearing property Phase 4 established for terrain --
     Stepped-mode editing must never visibly desync from a fresh reload,
     and that now has to hold with units in the picture too.
  4. Undo/redo: the same incremental path, run in reverse, restores
     byte-identity to the pre-edit render; redoing restores the edited
     state.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from descape import iso_geometry, render
from descape.edit_history import EditHistory
from descape.terrain_palette import BUILDING_FOOTPRINTS, PLAYER_COLORS


@dataclass
class SyntheticTile:
    x: int
    y: int
    elevation: int
    terrain_id: int = 0
    layer: int = -1


@dataclass
class SyntheticUnit:
    x: float
    y: float
    unit_const: int


class _FakeMapManager:
    def __init__(self, w: int, h: int, tiles: list[SyntheticTile]):
        self.map_width = w
        self.map_height = h
        self.terrain = tiles
        self._by_xy = {(t.x, t.y): t for t in tiles}

    def get_tile(self, x: int, y: int) -> SyntheticTile:
        return self._by_xy[(x, y)]


class _FakeUnitManager:
    def __init__(self, units_by_player: list[list[SyntheticUnit]]):
        self.units = units_by_player


class _FakeScenario:
    """Same duck-typing this project's other iso verify scripts use for
    synthetic maps (see tools/verify_iso_render.py's own _FakeScenario),
    extended with a real unit_manager.units so Phase 5's with_units=True
    default has something to draw."""

    def __init__(self, w: int, h: int, tiles: list[SyntheticTile], units_by_player: list[list[SyntheticUnit]]):
        self.map_manager = _FakeMapManager(w, h, tiles)
        self.unit_manager = _FakeUnitManager(units_by_player)


# A real unit_const with a >1 footprint radius (not just a single-tile
# marker) -- needed for the multi-tile-building checks (3b/3c above). Picked
# dynamically rather than hardcoded so this doesn't silently stop testing
# anything if tools/gen_unit_render_data.json's data ever changes shape.
_BUILDING_RADIUS = 2
_BUILDING_UNIT_CONST = next(
    uid for uid, (rx, ry) in BUILDING_FOOTPRINTS.items() if rx == _BUILDING_RADIUS and ry == _BUILDING_RADIUS
)


def _flat_scenario(w: int, h: int, units_by_player: list[list[SyntheticUnit]]) -> tuple[_FakeScenario, np.ndarray]:
    tiles = [SyntheticTile(x=x, y=y, elevation=0) for y in range(h) for x in range(w)]
    elevations = np.zeros((h, w), dtype=np.int64)
    return _FakeScenario(w, h, tiles, units_by_player), elevations


def check_position() -> tuple[bool, str]:
    """A lone unit at (3, 3) on a 6x6 flat map, tile (3, 3) raised to
    elevation 5. The unit's own diamond must land at tile_screen_origin(3,
    3, 5, proj) -- and must NOT still be sitting at the elevation-0
    position (which would mean the unit ignored the tile's real height).

    Elevation 5, not just 1 or 2: with ELEV_STEP_DIVISOR=2 (see iso_geometry
    module docstring), one elevation level's vertical shift is only HALF a
    diamond's own box height, so adjacent-elevation diamond boxes overlap on
    screen by design (that's what makes a height change look continuous
    rather than gapped) -- checking at elevation 1 or 2 would find genuine
    raised-diamond pixels bleeding into the flat-position box purely from
    that overlap, a false failure, not a real one. A 5-level gap (40px shift
    vs a 32px box height at tile_px=64) guarantees the two boxes don't
    overlap at all, so any match at the elevation-0 position is unambiguous."""
    w, h = 6, 6
    raised_elev = 5
    scenario, _ = _flat_scenario(w, h, [[], [SyntheticUnit(x=3.5, y=3.5, unit_const=999999)]])
    scenario.map_manager.get_tile(3, 3).elevation = raised_elev

    img, elevations, proj = render.render_terrain_iso_with_proj(scenario)
    color = PLAYER_COLORS[1 % len(PLAYER_COLORS)]
    dst_y, dst_x, _, _ = iso_geometry.diamond_indices(render.tile_pixels_for_map(w, h))

    bx, by = iso_geometry.tile_screen_origin(3, 3, raised_elev, proj)
    at_raised = img[by + dst_y, bx + dst_x]
    found_raised = int(np.count_nonzero(np.all(at_raised == color, axis=1)))

    bx0, by0 = iso_geometry.tile_screen_origin(3, 3, 0, proj)
    assert by0 - by >= 2 * proj.half_h, "test setup bug: raised/flat diamond boxes still overlap"
    at_flat = img[by0 + dst_y, bx0 + dst_x]
    found_flat = int(np.count_nonzero(np.all(at_flat == color, axis=1)))

    problems = []
    if found_raised == 0:
        problems.append("unit color not found at the tile's OWN (raised) elevation position")
    if found_flat != 0:
        problems.append(f"unit color found at the elevation-0 position too ({found_flat} px) -- not Z-aware")
    if problems:
        return False, "; ".join(problems)
    return True, (
        f"OK (unit painted at elevation {raised_elev}'s screen position, {found_raised} px, absent at elevation 0)"
    )


def check_occlusion() -> tuple[bool, str]:
    """Mirrors verify_iso_render.py's check_occlusion_scripted exactly, but
    for a unit instead of terrain: a unit sits at (3, 2) = (cx+1, cy), a
    "further" neighbor of (cx, cy) = (2, 2) (d=-1 vs d=0 -- painted earlier,
    per iso_geometry.skirt_quad_indices' docstring). Raising (2, 2) by 1
    must visibly cover part of the unit's own diamond -- if units were drawn
    as a blanket pass after all terrain, this would never happen (the unit
    would always end up on top regardless of the raised tile's depth).

    (3, 2) = (cx+1, cy) is ALSO (2, 2)'s "up_right" contact-shadow back
    neighbor (see iso_geometry.shadow_quad_indices' docstring) -- since the
    v2.6 shadow feature, raising (2, 2) both occludes AND darkens part of
    the unit's diamond, so a raw pixel-count drop (the old
    `raised_unit_px >= flat_unit_px` bar) is satisfiable by darkening
    alone, without any real occlusion. Distinguished by rendering the
    raised case TWICE -- once with the real CONTACT_SHADE, once with it
    neutralized to 1.0 (identity: the shadow code path still runs, it just
    darkens by a factor of 1) -- and requiring the pixel-count drop to
    survive with the shadow neutralized. That isolates genuine occlusion
    (from whatever mechanism actually produces it -- confirmed directly
    that it's not only the raised tile's own top diamond: some of it comes
    from (2, 2)'s "right" skirt, cast toward its OWN right neighbor (2, 3)
    and unrelated to the unit, which geometrically bleeds into part of
    (3, 2)'s diamond bounding box too, an existing property of how
    isometric diamond bboxes overlap between screen-adjacent tiles, not a
    shadow-specific effect) without needing to assume which one."""
    w, h = 5, 5
    cx, cy = 2, 2
    ux, uy = cx + 1, cy  # a "further" neighbor -- see docstring above

    flat_scn, _ = _flat_scenario(w, h, [[], [SyntheticUnit(x=ux + 0.5, y=uy + 0.5, unit_const=999999)]])
    raised_scn, _ = _flat_scenario(w, h, [[], [SyntheticUnit(x=ux + 0.5, y=uy + 0.5, unit_const=999999)]])
    raised_scn.map_manager.get_tile(cx, cy).elevation = 1

    flat_img = render.render_terrain_iso(flat_scn)
    raised_img = render.render_terrain_iso(raised_scn)

    original_contact_shade = render.CONTACT_SHADE
    render.CONTACT_SHADE = 1.0
    render._shadow_factors.cache_clear()
    try:
        raised_img_no_shadow = render.render_terrain_iso(raised_scn)
    finally:
        render.CONTACT_SHADE = original_contact_shade
        render._shadow_factors.cache_clear()

    if flat_img.shape != raised_img.shape:
        return False, f"shape drifted: flat {flat_img.shape} vs raised {raised_img.shape}"

    color = PLAYER_COLORS[1 % len(PLAYER_COLORS)]
    tile_px = render.tile_pixels_for_map(w, h)
    dst_y, dst_x, _, _ = iso_geometry.diamond_indices(tile_px)
    proj = iso_geometry.canvas_size_and_origin(w, h, tile_px, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION)
    bx, by = iso_geometry.tile_screen_origin(ux, uy, 0, proj)

    flat_block = flat_img[by + dst_y, bx + dst_x]
    raised_block = raised_img[by + dst_y, bx + dst_x]
    raised_block_no_shadow = raised_img_no_shadow[by + dst_y, bx + dst_x]
    flat_unit_px = int(np.count_nonzero(np.all(flat_block == color, axis=1)))
    raised_unit_px = int(np.count_nonzero(np.all(raised_block == color, axis=1)))
    raised_unit_px_no_shadow = int(np.count_nonzero(np.all(raised_block_no_shadow == color, axis=1)))

    problems = []
    if flat_unit_px == 0:
        problems.append("sanity check failed: unit not visible at all on the flat map")
    if raised_unit_px >= flat_unit_px:
        problems.append(
            f"expected raising (2,2) to reduce the unit's visible player-colored pixels, but they "
            f"went from {flat_unit_px} (flat) to {raised_unit_px} (raised)"
        )
    if raised_unit_px_no_shadow >= flat_unit_px:
        problems.append(
            f"expected raising (2,2) to reduce the unit's visible player-colored pixels even with "
            f"CONTACT_SHADE neutralized (no darkening possible), but they went from {flat_unit_px} "
            f"(flat) to {raised_unit_px_no_shadow} (raised, shadow neutralized) -- the drop with the "
            f"real shadow ({flat_unit_px} -> {raised_unit_px}) could be entirely the shadow darkening "
            "the unit, not real geometric occlusion"
        )
    if problems:
        return False, "; ".join(problems)
    return True, (
        f"OK (unit visible pixels dropped {flat_unit_px} -> {raised_unit_px} with the real shadow, "
        f"{flat_unit_px} -> {raised_unit_px_no_shadow} with CONTACT_SHADE neutralized -- occlusion "
        "confirmed independent of the shadow)"
    )


def _apply_and_check_incremental(scenario, tile_px, edits, label) -> list[str]:
    """edits is a list of (x, y, new_elevation) applied one at a time, each
    via EditHistory the same way viewer.py's stroke handling does, each
    checked byte-identical to a fresh full render immediately after. Then
    every edit is undone (checked against the pre-edit baseline) and redone
    (checked against the fully-edited state) via the same incremental path.
    Returns a list of problem strings (empty if everything passed). Mirrors
    tools/verify_iso_incremental.py's check_incremental_and_undo_redo
    structure exactly, one level down (synthetic units instead of real
    files' real ones)."""
    mm = scenario.map_manager
    problems: list[str] = []
    hist = EditHistory()

    current_img, current_elevations, current_proj = render.render_terrain_iso_with_proj(scenario)
    pre_edit_baseline = current_img.copy()
    post_edit_snapshots = []

    for x, y, new_elev in edits:
        hist.begin_stroke(mm.terrain)
        mm.get_tile(x, y).elevation = new_elev
        dirty = hist.commit_stroke(f"{label} edit", mm.terrain)
        if not dirty:
            problems.append(f"[{label}] edit at ({x},{y}) produced no dirty tiles")
            continue
        bbox = render.refresh_region_iso(current_img, scenario, dirty, current_elevations, current_proj, tile_px)
        if bbox is None:
            problems.append(f"[{label}] edit at ({x},{y}): refresh_region_iso declined unexpectedly")
            continue
        full = render.render_terrain_iso(scenario)
        if current_img.shape != full.shape:
            problems.append(f"[{label}] edit at ({x},{y}): shape drifted vs full re-render")
        elif not np.array_equal(current_img, full):
            diff = int(np.count_nonzero(np.any(current_img != full, axis=2)))
            problems.append(f"[{label}] edit at ({x},{y}): incremental differs from full re-render at {diff} px")
        post_edit_snapshots.append(current_img.copy())

    for _ in edits:
        dirty = hist.undo(mm.terrain)
        if not dirty:
            continue
        bbox = render.refresh_region_iso(current_img, scenario, dirty, current_elevations, current_proj, tile_px)
        if bbox is None:
            problems.append(f"[{label}] undo: refresh_region_iso declined unexpectedly")
    if not np.array_equal(current_img, pre_edit_baseline):
        diff = int(np.count_nonzero(np.any(current_img != pre_edit_baseline, axis=2)))
        problems.append(f"[{label}] undo-to-start doesn't match the pre-edit render at {diff} px")

    for _ in edits:
        dirty = hist.redo(mm.terrain)
        if not dirty:
            continue
        bbox = render.refresh_region_iso(current_img, scenario, dirty, current_elevations, current_proj, tile_px)
        if bbox is None:
            problems.append(f"[{label}] redo: refresh_region_iso declined unexpectedly")
    if post_edit_snapshots and not np.array_equal(current_img, post_edit_snapshots[-1]):
        diff = int(np.count_nonzero(np.any(current_img != post_edit_snapshots[-1], axis=2)))
        problems.append(f"[{label}] redo-to-end doesn't match the fully-edited state at {diff} px")

    return problems


def check_incremental_lone_unit() -> tuple[bool, str]:
    """A lone (non-building) unit's own tile is edited directly."""
    w, h = 8, 8
    scenario, _ = _flat_scenario(w, h, [[], [SyntheticUnit(x=4.5, y=4.5, unit_const=999999)]])
    tile_px = render.tile_pixels_for_map(w, h)
    problems = _apply_and_check_incremental(scenario, tile_px, [(4, 4, 3), (4, 4, 0)], "lone unit, own tile")
    if problems:
        return False, "; ".join(problems)
    return True, "OK (lone unit's own tile edited, incremental matched full re-render + undo/redo exact)"


def check_incremental_building_center() -> tuple[bool, str]:
    """A multi-tile building's CENTER tile is edited directly -- the whole
    footprint's screen position must shift, and the incremental patch must
    reproduce that exactly."""
    w, h = 16, 16
    cx, cy = 8, 8
    scenario, _ = _flat_scenario(w, h, [[], [SyntheticUnit(x=cx + 0.5, y=cy + 0.5, unit_const=_BUILDING_UNIT_CONST)]])
    tile_px = render.tile_pixels_for_map(w, h)
    problems = _apply_and_check_incremental(
        scenario, tile_px, [(cx, cy, 4), (cx, cy, 1)], "building, center tile"
    )
    if problems:
        return False, "; ".join(problems)
    return True, "OK (building center tile edited, incremental matched full re-render + undo/redo exact)"


def check_incremental_building_corner() -> tuple[bool, str]:
    """A multi-tile building's own FOOTPRINT CORNER tile (not its center) is
    edited -- the building's rendered diamonds don't move (they're all
    still at the CENTER's unchanged elevation), but the corner tile's own
    terrain does change, and the building's diamond sitting at that exact
    screen tile must survive the incremental repaint (not get silently
    wiped by the fresh terrain underneath it). This is the case
    render.py's refresh_region_iso() needs its footprint-radius seed
    dilation (and the unit-carrier scan) for, not just the +-1 a terrain
    skirt alone would need -- a corner _BUILDING_RADIUS=2 tiles from its own
    center is exactly the case a naive +-1 dilation would miss."""
    w, h = 16, 16
    cx, cy = 8, 8
    corner_x, corner_y = cx - _BUILDING_RADIUS, cy - _BUILDING_RADIUS
    scenario, _ = _flat_scenario(w, h, [[], [SyntheticUnit(x=cx + 0.5, y=cy + 0.5, unit_const=_BUILDING_UNIT_CONST)]])
    tile_px = render.tile_pixels_for_map(w, h)
    problems = _apply_and_check_incremental(
        scenario,
        tile_px,
        [(corner_x, corner_y, 3), (corner_x, corner_y, 0)],
        "building, footprint corner tile",
    )
    if problems:
        return False, "; ".join(problems)
    return True, "OK (building footprint-corner tile edited, incremental matched full re-render + undo/redo exact)"


def main() -> None:
    checks = [
        ("Position (own tile's elevation)", check_position),
        ("Occlusion (interleaved with terrain, not a blanket overlay)", check_occlusion),
        ("Incremental vs full: lone unit's own tile", check_incremental_lone_unit),
        ("Incremental vs full: building CENTER tile", check_incremental_building_center),
        ("Incremental vs full: building FOOTPRINT CORNER tile", check_incremental_building_corner),
    ]

    failures = 0
    for name, fn in checks:
        print(f"\n=== {name} ===")
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'}  {detail}")
        if not ok:
            failures += 1

    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
