#!/usr/bin/env python3
"""Verifies descape/render.py's render_terrain_iso() -- Phase 2 of the real
isometric Z-height terrain rendering plan.
Phase 1's tools/verify_iso_geometry.py checks the projection primitives in
isolation; this script checks their *integration* into a real compositor --
right neighbors, right paint order, right edge handling, real pixel output.

Checks:
  1. Determinism: rendering the same scenario twice produces byte-identical
     output -- the baseline every later phase's incremental-vs-full-render
     check (Phase 4) will diff against.
  2. Full coverage: on small synthetic maps (flat and elevated), every
     pixel an independent oracle says should be painted (replaying
     depth_order/diamond_indices/skirt_quad_indices directly, not calling
     render.py's internals) is non-background in the real render, and
     every pixel outside that footprint is exactly background -- no gaps,
     no leaks past the expected silhouette.
  3. Occlusion, scripted: a single raised tile on flat ground partially
     overwrites its "further" neighbors' own diamonds (the ones with
     smaller d = y-x, painted earlier) and leaves its "closer" neighbors'
     diamonds (larger d, painted later) completely untouched -- the direct
     empirical check for decision #4's depth order, confirmed against the
     real renderer's output.
  4. Pick-oracle: builds a full-resolution pick map inside this script only
     (small synthetic map, never shipped) and reports screen_to_tile's
     disagreement rate against it, split by top-face vs. skirt-face pixels
     -- skirt disagreement is expected and accepted (screen_to_tile knows
     nothing about skirts, per Risk #5), top-face disagreement is not.
  5. Shadow clipping: the v2.6 contact shadow (descape.iso_geometry.
     shadow_quad_indices) can reach a genuinely negative ABSOLUTE canvas
     row near the map's own worst-case corner -- confirms
     render._clipped_darken() clips that away instead of crashing or
     wrapping via numpy fancy-index underflow into the wrong rows.

PNG output for a manual eyeball pass (2_Joan_coop_2, C2_ElCid_coop_4,
C2_ElCid_coop_1) used to live here as a 6th, always-pass entry in this
script's own checks list -- moved to tools/gen_iso_reference_pngs.py (the
pytest migration plan's Ordering step 4), since it was never a check_
prefixed pass/fail check to begin with.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from descape import iso_geometry, render
from descape.scenario_io import load_map_and_units


@dataclass
class SyntheticTile:
    x: int
    y: int
    elevation: int
    terrain_id: int = 0


class _FakeMapManager:
    def __init__(self, w: int, h: int, tiles: list[SyntheticTile]):
        self.map_width = w
        self.map_height = h
        self.terrain = tiles


class _FakeUnitManager:
    """Empty per-player unit lists -- just enough for render_terrain_iso()'s
    default with_units=True (Phase 5) to iterate scenario.unit_manager.units
    without an AttributeError. These synthetic maps test terrain compositing
    in isolation, not unit placement (see tools/verify_iso_units.py for
    that), so "no units, on purpose" rather than a defensive getattr in
    render.py itself -- silently skipping units in production code would be
    the worse failure mode."""

    def __init__(self):
        self.units: list[list] = [[]]


class _FakeScenario:
    """Duck-types just enough of LoadedScenario (map_manager.map_width/
    map_height/terrain, unit_manager.units) for render_terrain_iso() -- same
    pattern this project's Phase 0 benchmark (tools/bench_iso_backend.py)
    and Phase 1 verify script already use for synthetic maps."""

    def __init__(self, w: int, h: int, tiles: list[SyntheticTile]):
        self.map_manager = _FakeMapManager(w, h, tiles)
        self.unit_manager = _FakeUnitManager()


def synthetic_scenario(w: int, h: int, elevation_fn) -> tuple[_FakeScenario, np.ndarray]:
    tiles = []
    elevations = np.zeros((h, w), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            e = elevation_fn(x, y)
            tiles.append(SyntheticTile(x=x, y=y, elevation=e))
            elevations[y, x] = e
    return _FakeScenario(w, h, tiles), elevations


def _tile_grid(scenario) -> list:
    mm = scenario.map_manager
    grid: list = [[None] * mm.map_width for _ in range(mm.map_height)]
    for tile in mm.terrain:
        grid[tile.y][tile.x] = tile
    return grid


def _replay_expected_paint(scenario, elevations: np.ndarray):
    """Independent oracle: replays the same paint order render_terrain_iso()
    uses (skirts then top diamond, per tile, in depth_order) via
    iso_geometry's own primitives directly -- NOT by calling
    render._render_tile_iso() -- to check render.py's *integration* of
    those primitives (right neighbor per side, right order, right edge
    handling), independent of render.py's own code. Returns
    (owner_x, owner_y, owner_skirt, proj, tile_px): owner_x/owner_y are
    int32 canvas-shaped arrays (-1 where nothing paints), owner_skirt is a
    bool canvas-shaped array (True where the last paint at that pixel was
    a skirt, not a top diamond)."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    tile_px = render.tile_pixels_for_map(w, h)
    # Fixed legal range, matching render_terrain_iso_with_proj() -- not this
    # synthetic scenario's own observed min/max. Must track render.py's real
    # sizing decision (see iso_geometry.MIN_ELEVATION/MAX_ELEVATION's own
    # comment on why, added for Phase 4) or this independent oracle computes
    # a differently-shaped canvas than the real renderer and every check
    # below fails on a shape mismatch, not a real bug.
    min_elev, max_elev = iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION
    proj = iso_geometry.canvas_size_and_origin(w, h, tile_px, min_elev, max_elev)
    skirt_headroom = (max_elev - min_elev) * proj.elev_step
    canvas_h = proj.canvas_h + skirt_headroom

    owner_x = np.full((canvas_h, proj.canvas_w), -1, dtype=np.int32)
    owner_y = np.full((canvas_h, proj.canvas_w), -1, dtype=np.int32)
    owner_skirt = np.zeros((canvas_h, proj.canvas_w), dtype=bool)

    for x, y in iso_geometry.depth_order(w, h):
        x, y = int(x), int(y)
        e = int(elevations[y, x])
        base_x, base_y = iso_geometry.tile_screen_origin(x, y, e, proj)
        for side, nx, ny in (("left", x - 1, y), ("right", x, y + 1)):
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            delta = e - int(elevations[ny, nx])
            if delta <= 0:
                continue
            drop_px = delta * proj.elev_step
            dst_y, dst_x, _, _ = iso_geometry.skirt_quad_indices(tile_px, drop_px, side)
            owner_x[base_y + dst_y, base_x + dst_x] = x
            owner_y[base_y + dst_y, base_x + dst_x] = y
            owner_skirt[base_y + dst_y, base_x + dst_x] = True
        dst_y, dst_x, _, _ = iso_geometry.diamond_indices(tile_px)
        owner_x[base_y + dst_y, base_x + dst_x] = x
        owner_y[base_y + dst_y, base_x + dst_x] = y
        owner_skirt[base_y + dst_y, base_x + dst_x] = False

    # Deliberately no contact-shadow term (v2.6's stepped-elevation shadow
    # feature, see descape.iso_geometry.shadow_quad_indices): this oracle
    # tracks pixel OWNERSHIP -- which tile last painted a color there --
    # and a shadow never paints, it only darkens whatever's already
    # marked. Adding an ownership entry for it would be wrong twice over:
    # it would falsely reassign ownership of a pixel the shadow merely
    # darkened to the shadow's caster instead of whoever actually painted
    # it, and shadow_quad_indices' own dst_y goes negative by design (see
    # its docstring) -- this oracle indexes owner_x/owner_y directly with
    # no _clipped_paint-style bounds check, so a negative index would wrap
    # via numpy fancy indexing instead of raising. check_full_coverage()
    # below doesn't need one either -- see its own comment for why
    # multiplicative darkening keeps its black<=>unpainted stand-in sound
    # without any oracle changes at all.
    return owner_x, owner_y, owner_skirt, proj, tile_px


def check_determinism(files: list[Path]) -> tuple[bool, str]:
    path = min(files, key=lambda p: p.stat().st_size)
    scenario = load_map_and_units(path)
    img1 = render.render_terrain_iso(scenario)
    img2 = render.render_terrain_iso(scenario)
    if not np.array_equal(img1, img2):
        return False, f"{path.name}: two renders of the same scenario differ"
    return True, f"OK ({path.name}, {img1.shape}, byte-identical across two renders)"


def check_full_coverage() -> tuple[bool, str]:
    # All configs use terrain_id=0 (the SyntheticTile default) -- its
    # fallback flat color (asset_source not configured in this headless
    # test) is color_for_terrain_id(0) == (129, 145, 63), confirmed
    # non-black with real margin -- render_tile()/_render_tile_iso() paint
    # that color as-is now (no elevation-based brightness multiplier since
    # the Phase 3 follow-up removed it, see render_tile()'s docstring), so a
    # painted pixel can never land on (0,0,0) -- making "is this pixel
    # exactly black" a sound stand-in for "is this pixel unpainted" below.
    # Would need revisiting if a config ever used a terrain_id whose
    # fallback color is itself near-black.
    #
    # Still sound with the v2.6 contact shadow in the mix, for two
    # independent reasons (see render._clipped_darken's own docstring):
    # multiplicative darkening never turns a real zero (background,
    # "leaks" half of this check) into anything else (0 * f == 0 exactly),
    # and CONTACT_SHADE floors every darkening factor at 0.65 (never 0),
    # so a genuinely painted pixel can never round down to pure black
    # either (the "unpainted_gaps" half). No oracle change needed for
    # either direction -- see _replay_expected_paint's own comment.
    configs = [
        ("flat 6x6", 6, 6, lambda x, y: 0),
        ("gentle staircase 6x6 (Δ<=1)", 6, 6, lambda x, y: min(x, 3)),
        ("mixed elevation 8x5", 8, 5, lambda x, y: (x + y) % 3),
    ]
    problems = []
    for label, w, h, elev_fn in configs:
        scenario, elevations = synthetic_scenario(w, h, elev_fn)
        owner_x, _owner_y, _owner_skirt, _proj, _tile_px = _replay_expected_paint(scenario, elevations)
        img = render.render_terrain_iso(scenario)
        if img.shape[:2] != owner_x.shape:
            problems.append(f"{label}: render shape {img.shape[:2]} != oracle shape {owner_x.shape}")
            continue
        painted = owner_x >= 0
        is_background = ~np.any(img.astype(bool), axis=2)
        unpainted_gaps = int(np.count_nonzero(painted & is_background))
        leaks = int(np.count_nonzero((~painted) & (~is_background)))
        if unpainted_gaps or leaks:
            problems.append(f"{label}: {unpainted_gaps} unpainted gap pixels, {leaks} leaked-past-footprint pixels")
    if problems:
        return False, "; ".join(problems)
    return True, f"OK ({len(configs)} synthetic maps, zero gaps, zero leaks past the expected footprint)"


def check_occlusion_scripted() -> tuple[bool, str]:
    """Confirmed empirically before writing this assertion (not guessed):
    raising tile (2,2) by 1 on an otherwise-flat 5x5 map partially
    overwrites the diamonds of its two "further" neighbors (x+1,y) and
    (x,y-1) -- smaller d = y-x, painted earlier, i.e. behind -- and leaves
    its two "closer" neighbors (x-1,y) and (x,y+1) -- larger d, painted
    later, i.e. in front -- completely untouched. See iso_geometry's
    skirt_quad_indices docstring for why those specific two neighbors are
    the ones with a visible edge at all."""
    w, h = 5, 5
    cx, cy = 2, 2
    flat_scn, _ = synthetic_scenario(w, h, lambda x, y: 0)
    raised_scn, _ = synthetic_scenario(w, h, lambda x, y: 1 if (x, y) == (cx, cy) else 0)
    flat_img = render.render_terrain_iso(flat_scn)
    raised_img = render.render_terrain_iso(raised_scn)

    tile_px = render.tile_pixels_for_map(w, h)
    # Fixed legal range, matching render_terrain_iso_with_proj() (Phase 4) --
    # both scenarios render onto the identically-shaped/positioned canvas
    # the real renderer now always uses, not one sized from each synthetic
    # scenario's own (here: 0-0 and 0-1) observed range.
    proj_flat = iso_geometry.canvas_size_and_origin(
        w, h, tile_px, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION
    )
    proj_raised = proj_flat
    dst_y, dst_x, _, _ = iso_geometry.diamond_indices(tile_px)

    def region_diff_count(nx: int, ny: int) -> int:
        fx, fy = iso_geometry.tile_screen_origin(nx, ny, 0, proj_flat)
        rx, ry = iso_geometry.tile_screen_origin(nx, ny, 0, proj_raised)
        block_flat = flat_img[fy + dst_y, fx + dst_x]
        block_raised = raised_img[ry + dst_y, rx + dst_x]
        return int(np.count_nonzero(np.any(block_flat != block_raised, axis=1)))

    further = {"(x+1,y)": (cx + 1, cy), "(x,y-1)": (cx, cy - 1)}
    closer = {"(x-1,y)": (cx - 1, cy), "(x,y+1)": (cx, cy + 1)}

    problems = []
    for label, (nx, ny) in further.items():
        n = region_diff_count(nx, ny)
        if n == 0:
            problems.append(f"{label}: expected partial occlusion, got 0 differing pixels")
    for label, (nx, ny) in closer.items():
        n = region_diff_count(nx, ny)
        if n != 0:
            problems.append(f"{label}: expected zero occlusion (painted after the raised tile), got {n}")

    if problems:
        return False, "; ".join(problems)
    return True, (
        "OK (raising (2,2) by 1 partially occludes both further neighbors' own diamonds, "
        "leaves both closer neighbors' diamonds untouched)"
    )


def check_pick_oracle() -> tuple[bool, str]:
    """Every canvas pixel of a small synthetic scenario, cross-checked
    against screen_to_tile(). Uses a gentle staircase (Δelev<=1 between
    any adjacent tiles, respecting this project's own confirmed real-map
    invariant) so top-face disagreement has no known-occlusion excuse --
    any top-face disagreement here is a real bug, not expected behavior."""
    w, h = 6, 6
    scenario, elevations = synthetic_scenario(w, h, lambda x, y: min(x, 3))
    owner_x, owner_y, owner_skirt, proj, _tile_px = _replay_expected_paint(scenario, elevations)

    canvas_h, canvas_w = owner_x.shape
    top_total = top_disagree = skirt_total = skirt_disagree = 0
    for sy in range(canvas_h):
        row_owner_x = owner_x[sy]
        row_owner_y = owner_y[sy]
        row_skirt = owner_skirt[sy]
        for sx in range(canvas_w):
            ox = int(row_owner_x[sx])
            if ox < 0:
                continue
            oy = int(row_owner_y[sx])
            got = iso_geometry.screen_to_tile(sx, sy, elevations, proj)
            if row_skirt[sx]:
                skirt_total += 1
                if got != (ox, oy):
                    skirt_disagree += 1
            else:
                top_total += 1
                if got != (ox, oy):
                    top_disagree += 1

    detail = (
        f"top-face: {top_disagree}/{top_total} disagree; "
        f"skirt-face: {skirt_disagree}/{skirt_total} disagree (expected/accepted, see Risk #5)"
    )
    if top_disagree:
        return False, detail
    return True, f"OK ({detail})"


def check_shadow_clipping() -> tuple[bool, str]:
    """Two halves, both about the contact-shadow band and
    render._clipped_darken().

    Name kept deliberately (it is a tests/migration_manifest.py key, so a
    rename is a manifest change), but the BODY was retargeted when the
    band became a wedge. Its old premise -- a full-canvas shadow pixel at
    a genuinely negative ABSOLUTE canvas row -- is now structurally
    unreachable, not merely hard to hit: the band is confined to the back
    neighbor's own diamond by construction (see shadow_quad_indices'
    docstring), and every real tile's diamond is in bounds on the full
    canvas by construction. No re-tuning of the old parameters can bring
    it back, so this asserts the POSITIVE invariant instead.

    1. Full canvas: every band pixel of every shadow-casting tile lands in
       bounds. Uses a delta small enough to leave a NON-empty band --
       MAX_ELEVATION would NOT work here: on this 6x6 map elev_step is 8,
       so rise_px = 15*8 = 120 against an emptiness threshold of
       2*half_h - 2 = 30, making every band empty and the assertion
       vacuous over zero-size arrays.
    2. Scratch canvas: the `offset` call site, where clipping is still
       genuinely reachable (a rect deliberately straddling the canvas's
       own top-left corner), confirming _clipped_darken still drops
       out-of-bounds pixels rather than crashing or wrapping.

    The anti-wraparound row-0/last-row oracle is kept from the original:
    a wrapped negative index would silently darken the OTHER end of the
    canvas instead of being dropped."""
    w, h = 6, 6
    cx, cy = w - 2, 0
    tile_px = render.tile_pixels_for_map(w, h)
    proj = iso_geometry.canvas_size_and_origin(
        w, h, tile_px, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION
    )
    _half_w, half_h = iso_geometry.half_dims(tile_px)

    # Largest delta that still leaves something exposed to shade.
    seam_delta = max(1, (2 * half_h - 3) // proj.elev_step)
    if seam_delta * proj.elev_step >= 2 * half_h - 2:
        return False, (
            f"test setup picked seam_delta={seam_delta}, whose rise_px "
            f"{seam_delta * proj.elev_step} is already an empty band (threshold {2 * half_h - 2})"
        )

    def elev(x, y):
        return seam_delta if (x, y) == (cx, cy) else 0

    scenario, elevations = synthetic_scenario(w, h, elev)
    rise_px = seam_delta * proj.elev_step
    dst_y, dst_x, _depth, _span = iso_geometry.shadow_quad_indices(tile_px, rise_px, "up_right")
    if dst_y.size == 0:
        return False, "test setup produced an empty band -- doesn't exercise anything"

    # Half 1: the positive invariant, over every real casting (tile, side).
    for y in range(h):
        for x in range(w):
            e = int(elevations[y, x])
            bx, by = iso_geometry.tile_screen_origin(x, y, e, proj)
            for side, nx, ny in (("up_left", x, y - 1), ("up_right", x + 1, y)):
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                delta = e - int(elevations[ny, nx])
                if delta <= 0:
                    continue
                sy, sx, _d, _s = iso_geometry.shadow_quad_indices(tile_px, delta * proj.elev_step, side)
                if sy.size == 0:
                    continue
                ay, ax = by + sy, bx + sx
                if ay.min() < 0 or ay.max() >= proj.canvas_h or ax.min() < 0 or ax.max() >= proj.canvas_w:
                    return False, (
                        f"tile ({x},{y}) side={side}: band left the full canvas -- "
                        f"rows [{ay.min()},{ay.max()}] cols [{ax.min()},{ax.max()}] "
                        f"vs canvas {proj.canvas_h}x{proj.canvas_w}"
                    )

    try:
        img = render.render_terrain_iso(scenario, with_units=False)
    except Exception as e:  # noqa: BLE001 -- exactly what "no crash" means here
        return False, f"render_terrain_iso raised {type(e).__name__}: {e}"

    # Half 2: _render_tile_iso's `offset` (scratch-canvas) call site, where
    # the clip is still genuinely reachable -- a tiny scratch canvas with an
    # offset that puts the caster mostly outside it, so its band straddles
    # every edge at once. Row 0 / column 0 of that scratch canvas are the
    # numpy-negative-index wrap targets, so this also re-checks wraparound
    # at the call site that can actually produce it.
    grid = _tile_grid(scenario)
    caster = grid[cy][cx]
    scratch = np.zeros((2 * tile_px, 2 * tile_px, 3), dtype=np.uint8)
    cbx, cby = iso_geometry.tile_screen_origin(cx, cy, seam_delta, proj)
    for off in ((cbx - 2, cby - 2), (cbx + tile_px, cby + tile_px), (cbx - 3 * tile_px, cby - 3 * tile_px)):
        before = scratch.copy()
        try:
            render._render_tile_iso(scratch, caster, tile_px, proj, elevations, w, h, offset=off)
        except Exception as e:  # noqa: BLE001
            return False, f"_render_tile_iso raised {type(e).__name__}: {e} at scratch offset {off}"
        if off == (cbx - 3 * tile_px, cby - 3 * tile_px) and not np.array_equal(scratch, before):
            return False, (
                f"offset {off} puts the tile entirely off the scratch canvas, yet pixels changed "
                f"-- looks like index wraparound rather than a clip"
            )

    # No wraparound into row 0 (or any row): a wrapped negative index would
    # silently darken pixels at the OTHER end of the canvas instead of
    # being dropped -- compare row 0 (and the canvas's last few rows, the
    # numpy-negative-index wrap targets) against a flat render's own,
    # which this seam should never touch at all (it's confined to the
    # caster's own column range near x=cx).
    flat_scenario, _ = synthetic_scenario(w, h, lambda x, y: 0)
    flat_img = render.render_terrain_iso(flat_scenario, with_units=False)
    if img.shape != flat_img.shape:
        return False, f"shape drifted: raised {img.shape} vs flat {flat_img.shape}"
    if not np.array_equal(img[0], flat_img[0]) or not np.array_equal(img[-1], flat_img[-1]):
        return False, "row 0 or the last row differs from a flat render -- looks like index wraparound"

    return True, (
        f"OK (seam delta={seam_delta} rise_px={rise_px}, {dst_y.size}-px band; every full-canvas band "
        f"pixel in bounds, scratch-canvas clip exercised, no wraparound)"
    )


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
        ("Determinism", lambda: check_determinism(files)),
        ("Full coverage", check_full_coverage),
        ("Occlusion, scripted", check_occlusion_scripted),
        ("Pick-oracle", check_pick_oracle),
        ("Shadow clipping (no crash/wraparound)", check_shadow_clipping),
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
