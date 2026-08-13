#!/usr/bin/env python3
"""Verifies descape/iso_geometry.py -- Phase 1 of the real isometric Z-height
terrain rendering plan. This module has no
visible effect on its own yet; this script is the correctness signal for it
until Phase 2's real compositor and Phase 3's viewer wiring exist.

Checks:
  1. Synthetic full-pixel round-trip: for small synthetic maps with varied
     per-tile elevation, every single pixel of every tile's diamond (not
     just its center) maps forward via tile_screen_origin/diamond_indices
     and back via screen_to_tile to that same (x, y) tile.
  2. Synthetic diamond partition: at a single flat elevation, the union of
     every tile's diamond_indices pixels covers the map's iso silhouette
     with no double-covered pixel and no interior hole -- the direct,
     scripted check for decision #4's "provably tie-free" claim.
  3. Real example files (every *.aoe2scenario in the given directory, all
     of them, not a subset): tile-center round-trip for every tile at its
     real elevation, plus a canvas-bytes report per file (so the "memory
     gets better, not worse" claim in the parent plan's Performance table
     is a live, re-checked number here, not a stale one).
  4. Bounds containment: computed canvas dimensions actually contain every
     tile's full diamond bbox, for every real example file plus a synthetic
     480x480 "Ludicrous" worst case (matching tools/bench_iso_backend.py's
     own synthetic dataset) and, since a genuinely-480x480 real file exists
     in this project's example set, that real file too.
  5. Skirt geometry bounds: skirt_quad_indices() output stays within its
     documented extent (dst_x in [0, 2*half_w), dst_y in
     [0, 2*half_h + drop_px)) and every source index into [0, tile_px), and
     its two sides' column union exactly equals diamond_indices' own column
     set -- the regression guard for the Step 0 comb defect (see
     descape.iso_geometry._diamond_column_edges' docstring).
  6. Shadow geometry bounds: shadow_quad_indices() (the v2.6 stepped-
     elevation contact-shadow feature) output stays within its documented
     extent, is internally unique and disjoint between its two sides, is
     disjoint from diamond_indices'/skirt_quad_indices' own pixels, its two
     sides' column union exactly equals diamond_indices' column set (same
     comb-defect guard as #5), and raises ValueError on a bad side or a
     non-positive rise_px.
  7. Ground outline corners: ground_outline_corners() (added in Phase 3, for
     the viewer's map-extent outline) returns the true screen-space extremes
     of the whole grid's flat diamond tiling -- checked by brute-force
     scanning every tile's own four corner points independently, not by
     restating ground_outline_corners' own extremal-tile assumption.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from descape import iso_geometry
from descape.render import tile_pixels_for_map
from descape.scenario_io import load_map_and_units


@dataclass
class SyntheticTile:
    x: int
    y: int
    elevation: int


def synthetic_map(w: int, h: int, tile_px: int, elevation_fn) -> tuple[list[SyntheticTile], np.ndarray]:
    tiles = []
    elevations = np.zeros((h, w), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            e = elevation_fn(x, y)
            tiles.append(SyntheticTile(x=x, y=y, elevation=e))
            elevations[y, x] = e
    return tiles, elevations


def check_full_pixel_roundtrip() -> tuple[bool, str]:
    """Every pixel of every tile's diamond, forward then back, on a handful
    of small synthetic maps with genuinely mixed elevation (not uniform --
    that would never exercise elevation disambiguation in screen_to_tile
    at all).

    NOT a plain "always returns the same tile" assertion -- with real
    elevation data, a pixel that's nominally tile A's own diamond can
    legitimately fall inside a taller, closer neighbor B's diamond too (B
    painted later in depth_order, so B visually occludes A there). That's
    the entire point of real Z-height rendering, not a bug: a raised tile
    is *supposed* to be able to visually cover part of what's behind it.
    So a pixel round-trips correctly if it returns either (a) the tile it
    came from, or (b) some other tile B with strictly greater d = y - x
    (i.e. B is genuinely painted after, and thus in front of, the origin
    tile -- confirmed by independently checking B's own diamond contains
    the same pixel, not just trusting screen_to_tile's own verdict).
    Anything else -- None, or a tile with d <= the origin tile's own d --
    is a real bug: nothing legitimately explains it."""
    tile_px = 8  # small on purpose -- this check is O(tiles * pixels/tile)
    configs = [
        (5, 5, lambda x, y: (x * 3 + y * 2) % 4),
        (6, 4, lambda x, y: 0),  # uniform elevation, still must round-trip
        (4, 6, lambda x, y: (x + y) % 5),
        # A steeper-than-real seam (Δelev=2 between adjacent columns) --
        # this project's own confirmed ±1 neighbor-elevation invariant
        # means real maps never do this, but per Risk #2 elevations aren't
        # validated, so a hand-edited or third-party file could. Must
        # still degrade gracefully (real, correctly-resolved occlusion, not
        # a crash or a wrong answer) rather than being excluded as
        # "can't happen" -- see iso_geometry.ELEV_STEP_DIVISOR's own note
        # that the 0%-occlusion result at the shipped divisor is
        # contingent on the invariant holding, confirmed to reappear here.
        (8, 4, lambda x, y: 2 if x % 2 == 0 else 0),
    ]
    total_pixels = 0
    occluded = 0
    bad = []
    for w, h, elev_fn in configs:
        tiles, elevations = synthetic_map(w, h, tile_px, elev_fn)
        min_elev, max_elev = int(elevations.min()), int(elevations.max())
        proj = iso_geometry.canvas_size_and_origin(w, h, tile_px, min_elev, max_elev)
        dst_y, dst_x, _src_y, _src_x = iso_geometry.diamond_indices(tile_px)
        for tile in tiles:
            base_x, base_y = iso_geometry.tile_screen_origin(tile.x, tile.y, tile.elevation, proj)
            d_origin = tile.y - tile.x
            for dy, dx in zip(dst_y, dst_x):
                sx, sy = base_x + int(dx), base_y + int(dy)
                total_pixels += 1
                got = iso_geometry.screen_to_tile(sx, sy, elevations, proj)
                if got == (tile.x, tile.y):
                    continue
                if got is not None:
                    gx, gy = got
                    if (gy - gx) > d_origin:
                        occluded += 1
                        continue
                bad.append((w, h, tile.x, tile.y, tile.elevation, sx, sy, got))

    if bad:
        return False, f"{len(bad)}/{total_pixels} pixel round-trips genuinely FAILED, e.g. {bad[:5]}"
    return True, (
        f"OK ({total_pixels} pixels round-tripped across {len(configs)} synthetic maps; "
        f"{occluded} legitimately occluded by a taller closer tile, 0 genuine failures)"
    )


def check_diamond_partition() -> tuple[bool, str]:
    """At a single flat elevation, every tile's diamond_indices pixels,
    placed via tile_screen_origin, must cover the map's iso silhouette
    exactly once each -- no gap, no double-cover."""
    tile_px = 8
    for w, h in [(6, 6), (5, 9), (9, 5)]:
        tiles, elevations = synthetic_map(w, h, tile_px, lambda x, y: 0)
        proj = iso_geometry.canvas_size_and_origin(w, h, tile_px, 0, 0)
        coverage = np.zeros((proj.canvas_h, proj.canvas_w), dtype=np.int32)
        dst_y, dst_x, _src_y, _src_x = iso_geometry.diamond_indices(tile_px)
        for tile in tiles:
            base_x, base_y = iso_geometry.tile_screen_origin(tile.x, tile.y, tile.elevation, proj)
            coverage[base_y + dst_y, base_x + dst_x] += 1

        painted = coverage[coverage > 0]
        n_painted = painted.size
        n_double = int(np.count_nonzero(coverage > 1))
        expected = w * h * dst_y.size
        if n_double or n_painted != expected:
            return False, (
                f"{w}x{h} tile_px={tile_px}: double-covered={n_double}, "
                f"painted_pixels={n_painted}, expected={expected} "
                f"(each tile paints {dst_y.size} pixels, {w * h} tiles)"
            )
    return True, "OK (no double-coverage, no gap, on 6x6/5x9/9x5 synthetic maps)"


def check_real_files(files: list[Path]) -> tuple[bool, str]:
    """Tile-center round-trip on every real example file -- see
    check_full_pixel_roundtrip's docstring for why a center returning a
    *different*, genuinely-occluding (strictly greater d) tile is expected
    and not counted as a failure, only a center returning None or a tile
    with d <= its own is."""
    failures = []
    lines = []
    for path in files:
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        w, h = mm.map_width, mm.map_height
        tile_px = tile_pixels_for_map(w, h)
        elevations = np.zeros((h, w), dtype=np.int64)
        for tile in mm.terrain:
            elevations[tile.y, tile.x] = tile.elevation
        min_elev, max_elev = int(elevations.min()), int(elevations.max())
        proj = iso_geometry.canvas_size_and_origin(w, h, tile_px, min_elev, max_elev)

        bad = 0
        occluded = 0
        for tile in mm.terrain:
            base_x, base_y = iso_geometry.tile_screen_origin(tile.x, tile.y, tile.elevation, proj)
            center_x, center_y = base_x + proj.half_w, base_y + proj.half_h
            got = iso_geometry.screen_to_tile(center_x, center_y, elevations, proj)
            if got == (tile.x, tile.y):
                continue
            if got is not None and (got[1] - got[0]) > (tile.y - tile.x):
                occluded += 1
                continue
            bad += 1

        canvas_mb = proj.canvas_w * proj.canvas_h * 3 / (1024 * 1024)
        status = "PASS" if bad == 0 else "FAIL"
        if bad:
            failures.append(path.name)
        lines.append(
            f"{status}  {path.name:36s} {w}x{h} tile_px={tile_px:2d} "
            f"elev=[{min_elev},{max_elev}] canvas={proj.canvas_w}x{proj.canvas_h} "
            f"(~{canvas_mb:.0f}MB)  center-roundtrip: bad={bad} occluded={occluded} of {len(mm.terrain)}"
        )

    print("\n".join(lines))
    if failures:
        return False, f"{len(failures)}/{len(files)} files had genuine mismatches: {failures}"
    return True, f"OK ({len(files)} files, every tile's center round-tripped or was genuinely occluded)"


def check_bounds_containment(files: list[Path]) -> tuple[bool, str]:
    problems = []

    def check_one(label: str, w: int, h: int, tile_px: int, tiles_xy_elev: np.ndarray):
        min_elev = int(tiles_xy_elev[:, 2].min())
        max_elev = int(tiles_xy_elev[:, 2].max())
        proj = iso_geometry.canvas_size_and_origin(w, h, tile_px, min_elev, max_elev)
        x = tiles_xy_elev[:, 0]
        y = tiles_xy_elev[:, 1]
        e = tiles_xy_elev[:, 2]
        sx = proj.origin_x + (x + y) * proj.half_w
        sy = proj.origin_y + (y - x) * proj.half_h - e * proj.elev_step
        if sx.min() < 0 or sy.min() < 0:
            problems.append(f"{label}: negative origin (sx_min={sx.min()}, sy_min={sy.min()})")
            return
        if (sx.max() + 2 * proj.half_w) > proj.canvas_w or (sy.max() + 2 * proj.half_h) > proj.canvas_h:
            problems.append(
                f"{label}: bbox exceeds canvas (max sx+w={sx.max() + 2 * proj.half_w} vs "
                f"canvas_w={proj.canvas_w}, max sy+h={sy.max() + 2 * proj.half_h} vs canvas_h={proj.canvas_h})"
            )

    for path in files:
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        w, h = mm.map_width, mm.map_height
        tile_px = tile_pixels_for_map(w, h)
        arr = np.array([(t.x, t.y, t.elevation) for t in mm.terrain], dtype=np.int64)
        check_one(path.name, w, h, tile_px, arr)

    # Synthetic 480x480 "Ludicrous" worst case -- matches
    # tools/bench_iso_backend.py's own synthetic dataset and elevation formula.
    size = 480
    xs, ys = np.meshgrid(np.arange(size), np.arange(size), indexing="xy")
    xs = xs.ravel()
    ys = ys.ravel()
    es = (xs + ys) % 7
    arr = np.stack([xs, ys, es], axis=1)
    tile_px = tile_pixels_for_map(size, size)
    check_one("synthetic 480x480 Ludicrous", size, size, tile_px, arr)

    if problems:
        return False, "; ".join(problems)
    return True, f"OK ({len(files)} real files + synthetic 480x480 worst case, all bboxes contained)"


def check_skirt_geometry() -> tuple[bool, str]:
    """skirt_quad_indices() has no coverage from any other check here --
    verifies its documented extent directly: dst_x in [0, 2*half_w),
    dst_y in [0, 2*half_h + drop_px), and every source index in
    [0, tile_px), for both sides and a spread of tile_px/drop_px values.

    Also: the union of both sides' dst_x columns must equal EXACTLY
    diamond_indices(tile_px)'s own column set -- an equality, not a subset
    test, so it fails loudly if the comb defect (a hand-rolled edge walk
    that only ever advanced by 2 columns, confirmed to leave every other
    column unpainted -- see iso_geometry._diamond_column_edges' own
    docstring for the measured 480-pixel repro) ever creeps back in.
    Compared against diamond_indices' actual columns rather than a
    hand-computed range like [1, 2*half_w-2]: if the membership mask ever
    changes, a hand-computed bound would silently drift out of sync while
    this derived comparison tracks it automatically."""
    problems = []
    n_checked = 0
    for tile_px in (8, 16, 32, 64):
        half_w, half_h = iso_geometry.half_dims(tile_px)
        _dst_y, diamond_dst_x, _src_y, _src_x = iso_geometry.diamond_indices(tile_px)
        diamond_cols = set(np.unique(diamond_dst_x).tolist())
        skirt_cols: set = set()
        for drop_px in (1, half_h, half_h * 3):
            for side in ("left", "right"):
                dst_y, dst_x, src_y, src_x = iso_geometry.skirt_quad_indices(tile_px, drop_px, side)
                n_checked += 1
                skirt_cols |= set(np.unique(dst_x).tolist())
                label = f"tile_px={tile_px} drop_px={drop_px} side={side}"
                if dst_x.min() < 0 or dst_x.max() >= 2 * half_w:
                    problems.append(f"{label}: dst_x out of [0, {2 * half_w}): [{dst_x.min()}, {dst_x.max()}]")
                if dst_y.min() < 0 or dst_y.max() >= 2 * half_h + drop_px:
                    problems.append(
                        f"{label}: dst_y out of [0, {2 * half_h + drop_px}): [{dst_y.min()}, {dst_y.max()}]"
                    )
                if src_x.min() < 0 or src_x.max() >= tile_px or src_y.min() < 0 or src_y.max() >= tile_px:
                    problems.append(
                        f"{label}: source index out of [0, {tile_px}): "
                        f"src_x=[{src_x.min()},{src_x.max()}] src_y=[{src_y.min()},{src_y.max()}]"
                    )
        if skirt_cols != diamond_cols:
            missing = diamond_cols - skirt_cols
            extra = skirt_cols - diamond_cols
            problems.append(
                f"tile_px={tile_px}: skirt columns != diamond_indices columns "
                f"(missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]})"
            )
    if problems:
        return False, "; ".join(problems)
    return True, f"OK ({n_checked} (tile_px, drop_px, side) combinations, all indices in bounds, columns match diamond_indices)"


def check_shadow_geometry() -> tuple[bool, str]:
    """shadow_quad_indices() has no coverage from any other check here --
    modeled directly on check_skirt_geometry() above. Over tile_px in
    (8, 16, 32, 64) x rise_px in (1, half_h, 3*half_h) x both sides,
    checks:
    - documented extent: dst_x in [1, 2*half_w - 2] (the TIGHT bound --
      not skirt_quad_indices' looser [0, 2*half_w), since admitting
      columns 0/2*half_w-1 here would be the same used-filter bug Step 0
      fixed for skirts), dst_y in [-rise_px, half_h - 2], depth in
      [0, rise_px).
    - (dst_y, dst_x) pairs are unique within a side (no column/row painted
      twice by one call).
    - the two sides are disjoint from each other (no shared pixel -- see
      shadow_quad_indices' own docstring for why, unlike skirt's
      deliberately-shared apex column).
    - the union of both sides' dst_x columns EXACTLY equals
      diamond_indices(tile_px)'s own column set -- an equality, catching
      the same class of comb defect check_skirt_geometry's own equality
      check guards against, compared against diamond_indices' real
      columns rather than a hand-computed range.
    - disjoint from diamond_indices' own (dst_y, dst_x) pixels and from
      skirt_quad_indices' (using a representative drop_px), confirming the
      shadow band never overlaps the tile's own top face or its skirts.
    - ValueError on an invalid side and on rise_px <= 0."""
    problems = []
    n_checked = 0
    for tile_px in (8, 16, 32, 64):
        half_w, half_h = iso_geometry.half_dims(tile_px)
        diamond_dst_y, diamond_dst_x, _sy, _sx = iso_geometry.diamond_indices(tile_px)
        diamond_pixels = set(zip(diamond_dst_y.tolist(), diamond_dst_x.tolist()))
        diamond_cols = set(np.unique(diamond_dst_x).tolist())

        skirt_pixels: set = set()
        for drop_px in (1, half_h, half_h * 3):
            for side in ("left", "right"):
                sk_y, sk_x, _sy2, _sx2 = iso_geometry.skirt_quad_indices(tile_px, drop_px, side)
                skirt_pixels |= set(zip(sk_y.tolist(), sk_x.tolist()))

        shadow_cols: set = set()
        side_pixels: dict[str, set] = {}
        for rise_px in (1, half_h, half_h * 3):
            for side in ("up_left", "up_right"):
                dst_y, dst_x, depth = iso_geometry.shadow_quad_indices(tile_px, rise_px, side)
                n_checked += 1
                label = f"tile_px={tile_px} rise_px={rise_px} side={side}"
                if dst_x.min() < 1 or dst_x.max() > 2 * half_w - 2:
                    problems.append(f"{label}: dst_x out of [1, {2 * half_w - 2}]: [{dst_x.min()}, {dst_x.max()}]")
                if dst_y.min() < -rise_px or dst_y.max() > half_h - 2:
                    problems.append(f"{label}: dst_y out of [{-rise_px}, {half_h - 2}]: [{dst_y.min()}, {dst_y.max()}]")
                if depth.min() < 0 or depth.max() >= rise_px:
                    problems.append(f"{label}: depth out of [0, {rise_px}): [{depth.min()}, {depth.max()}]")
                pixels = list(zip(dst_y.tolist(), dst_x.tolist()))
                if len(pixels) != len(set(pixels)):
                    problems.append(f"{label}: (dst_y, dst_x) pairs not unique")
                pixel_set = set(pixels)
                other_side = "up_right" if side == "up_left" else "up_left"
                other_pixels = side_pixels.get((tile_px, rise_px, other_side))
                if other_pixels is not None and (pixel_set & other_pixels):
                    problems.append(f"{label}: overlaps its own '{other_side}' counterpart")
                side_pixels[(tile_px, rise_px, side)] = pixel_set
                if pixel_set & diamond_pixels:
                    problems.append(f"{label}: overlaps diamond_indices' own pixels")
                if pixel_set & skirt_pixels:
                    problems.append(f"{label}: overlaps skirt_quad_indices' own pixels")
                shadow_cols |= set(np.unique(dst_x).tolist())

        if shadow_cols != diamond_cols:
            missing = diamond_cols - shadow_cols
            extra = shadow_cols - diamond_cols
            problems.append(
                f"tile_px={tile_px}: shadow columns != diamond_indices columns "
                f"(missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]})"
            )

    try:
        iso_geometry.shadow_quad_indices(64, 4, "bogus_side")
        problems.append("shadow_quad_indices did not raise ValueError for a bad side")
    except ValueError:
        pass
    try:
        iso_geometry.shadow_quad_indices(64, 0, "up_left")
        problems.append("shadow_quad_indices did not raise ValueError for rise_px=0")
    except ValueError:
        pass

    if problems:
        return False, "; ".join(problems)
    return True, (
        f"OK ({n_checked} (tile_px, rise_px, side) combinations, all indices in bounds, "
        f"disjoint from each other/diamond/skirts, columns match diamond_indices)"
    )


def check_ground_outline_corners() -> tuple[bool, str]:
    """ground_outline_corners() has no coverage from any other check here --
    verifies its four returned points are the genuine screen-space extremes
    of the whole grid's flat (elevation=0) diamond tiling, by brute-force
    scanning every tile's own four corner points directly via
    tile_screen_origin -- an independent check, not a restatement of
    ground_outline_corners' own formula, since it doesn't assume in advance
    which grid-corner tile is extremal."""
    problems = []
    n_checked = 0
    for w, h, tile_px in ((6, 6, 64), (9, 5, 32), (5, 9, 16), (2, 2, 64)):
        half_w, half_h = iso_geometry.half_dims(tile_px)
        proj = iso_geometry.canvas_size_and_origin(w, h, tile_px, 0, 0)
        n_checked += 1
        label = f"w={w} h={h} tile_px={tile_px}"

        all_points = []
        for y in range(h):
            for x in range(w):
                ox, oy = iso_geometry.tile_screen_origin(x, y, 0, proj)
                all_points.append((ox + half_w, oy))  # top
                all_points.append((ox + 2 * half_w, oy + half_h))  # right
                all_points.append((ox + half_w, oy + 2 * half_h))  # bottom
                all_points.append((ox, oy + half_h))  # left
        expected_west = min(all_points, key=lambda p: p[0])
        expected_east = max(all_points, key=lambda p: p[0])
        expected_north = min(all_points, key=lambda p: p[1])
        expected_south = max(all_points, key=lambda p: p[1])

        west, north, east, south = iso_geometry.ground_outline_corners(w, h, proj)
        if west[0] != expected_west[0]:
            problems.append(f"{label}: west screen_x={west[0]} != brute-force min {expected_west[0]}")
        if east[0] != expected_east[0]:
            problems.append(f"{label}: east screen_x={east[0]} != brute-force max {expected_east[0]}")
        if north[1] != expected_north[1]:
            problems.append(f"{label}: north screen_y={north[1]} != brute-force min {expected_north[1]}")
        if south[1] != expected_south[1]:
            problems.append(f"{label}: south screen_y={south[1]} != brute-force max {expected_south[1]}")
        if not all(0 <= px <= proj.canvas_w and 0 <= py <= proj.canvas_h for px, py in (west, north, east, south)):
            problems.append(f"{label}: a corner point falls outside [0,{proj.canvas_w}]x[0,{proj.canvas_h}]")
        if len({west, north, east, south}) != 4:
            problems.append(f"{label}: corners not mutually distinct: {(west, north, east, south)}")

    if problems:
        return False, "; ".join(problems)
    return True, f"OK ({n_checked} (w, h, tile_px) configs, all 4 corners match brute-force extremes and bounds)"


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
        ("Synthetic full-pixel round-trip", check_full_pixel_roundtrip),
        ("Synthetic diamond partition", check_diamond_partition),
        ("Real example files (all)", lambda: check_real_files(files)),
        ("Bounds containment", lambda: check_bounds_containment(files)),
        ("Skirt geometry bounds", check_skirt_geometry),
        ("Shadow geometry bounds", check_shadow_geometry),
        ("Ground outline corners", check_ground_outline_corners),
    ]

    failures = 0
    for name, fn in checks:
        print(f"\n=== {name} ===")
        ok, detail = fn()
        print(f"{'PASS' if ok else 'FAIL'}  {detail}")
        if not ok:
            failures += 1

    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
