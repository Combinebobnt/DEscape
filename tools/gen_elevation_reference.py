#!/usr/bin/env python3
"""
Track A1/A2: generates the elevation reference scenarios that settle
Phase 6's sloped-ramp design questions from real in-game screenshots
instead of guessing. Per Track A4 this also closed the "Elevation edits:
in-game verification" item: loading these files at all, with their
elevation edits intact, was what closed it, regardless of what the
screenshots went on to show about ramp shape. What they showed, preliminary
as of 2026-08-09: DE renders continuous interpolated slopes rather than a
small set of discrete pre-authored facets, every differing-elevation seam
ramps rather than staying vertical, and an illegal delta>=2 seam renders as
genuinely malformed geometry rather than being silently repaired back to
delta=1.

Loads examples/blank_map.aoe2scenario (Track A0 -- a blank, square, unit-free
base the user created in-game; a real example file can't be reused here:
this repo's write path only ever touches the terrain struct array, so an
example file's
existing trees can't be cleared first). Writes 3 scenario files into
build/elevation_reference/, each holding 3 widely-separated "regions" (one
per elevation pattern, see PATTERNS below) so the slow in-game editor only
has to open 3 files, not 9. For every written file also renders this tool's
own Stepped-mode compositor and crops one reference PNG per region (Track
A2) -- the point of the exercise is comparing the game's rendering against
this tool's, not producing screenshots in isolation.

Terrain choice: the plan called for checkerboarding two terrains "with
matching blend_priority" so DE's own terrain blending doesn't confound the
elevation readout. Checked directly against the DE dat file (genieutils'
DatFile.terrain_block, via the sibling DEscape .venv that
already has genieutils-py): blend_priority is a UNIQUE value for all 200
terrain entries in the current DE dat -- no two ever match, so "matching
blend_priority" as literally specified is impossible. blend_type (a small
0-7 categorical field on the same struct) was tried as a substitute
grouping value instead (e.g. every plain grass/dirt/forest terrain is
blend_type 0), first as GRASS_2/DIRT_2, then GRASS_2/DESERT_SAND for more
raw texture contrast (116.5 RGB-distance via descape.asset_source.
get_terrain_average_color vs. the original pair's 77.6) after
reference_a's 2026-08-06 screenshot showed lone_bump/lone_pit hard to
tell apart by eye.

Both same-blend_type pairs failed in-game the same way (confirmed
2026-08-09): at strict 1-tile alternation every tile is a blend boundary
on all 4 sides, and DE's terrain blending -- which exists specifically to
smooth transitions *within* a blend_type group -- dilutes the checker
proportional to how many opposite-terrain neighbors each tile touches (4
of 4, the worst case, for a 1-tile checker). A hand-painted contiguous
DESERT_SAND patch (no alternation, so mostly 0-of-4 boundary neighbors)
rendered as vivid, clearly-visible tan right next to the same washed-out
in-frame checker in the same screenshot, confirming it's dilution by
boundary-neighbor count, not a hard on/off absorption. The minimap always
showed the correct checkerboard regardless (proving the per-tile
assignment itself was always fine).

Widening the checker to multi-tile blocks would fix contrast but was
rejected: staircase's elevation rings are ~1 tile wide per level, so a
coarser checker would blur the exact ring-boundary precision that region
exists to read (moot now -- staircase is no longer being pursued -- but
the reasoning still applies to any future ring-precision need). Instead:
pick a pair from *different*
blend_type groups, which get
a masked, crisp transition instead of a smooth one even at 1-tile
alternation -- visible for free in every screenshot of these regions,
where the grey ROAD locator frame (blend_type 5, vs. GRASS_2's blend_type
0) renders sharp against the grass right next to a same-group checker that
doesn't. SNOW (blend_type 7) was tried on that basis and confirmed the
cross-group mechanism works, but hit a second problem in-game
(2026-08-09): SNOW's blend_priority (153) is far above GRASS_2's (119), so
SNOW dominated/overpainted almost the entire checker, leaving GRASS_2 as
scattered decorative tufts rather than a real 50/50 pattern. The real
requirement is a *different* blend_type **and** a close blend_priority --
no exact ties exist (blend_priority is unique across all 200 terrain
entries) -- so BEACH_WHITE (blend_type 2, priority 129, a gap of only 10
vs. GRASS_2's 119, and 121.5 RGB-distance contrast) replaced SNOW.
Confirmed in-game (2026-08-09): still dominated, just in the other
direction (BEACH_WHITE over GRASS_2 this time) -- a 10-point gap produced
domination just as strong as SNOW's 34-point gap did, so this isn't a
tunable "get the priorities closer" problem: whichever terrain has *any*
higher priority appears to win most of a checker tile's area via blend-
mask overlap from all 4 higher-priority neighbors, leaving only a small
unmasked center showing the loser through as a faint dot. A true 50/50
checkerboard at 1-tile alternation is likely not achievable in DE this
way, and chasing a 5th pair isn't worth it -- **every screenshot taken
during this investigation already has DE's own tile-grid overlay etched
into the terrain, independent of color**, which covers the tile-counting
need a working checker would have served. GRASS_2/BEACH_WHITE is being
kept as the final state: a real (if partial) improvement over the
original pair's full invisibility, not a solved problem. BLACK_WALKABLE/
DESERT_SAND was tried even earlier and rejected for an unrelated reason:
BLACK_WALKABLE is a special map-hole terrain with its own soft dark
overlay shader, not a normal flat tile, independent of the blend-
mechanism findings above.

Elevation edits go through descape.elevation_tools.set_tile_elevation (the
real single-tile edit path, propagation and all) for every pattern except
steep_seam, which -- per the plan -- must assign TerrainTile.elevation
directly to produce an illegal (Delta>=2) seam without the propagation
recursion silently repairing it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AoE2ScenarioParser.datasets.terrains import TerrainId

from descape import render
from descape.batch_api import set_terrain
from descape.elevation_tools import set_tile_elevation
from descape.iso_geometry import tile_screen_origin
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario

DEFAULT_BASE = Path(__file__).resolve().parent.parent / "examples" / "blank_map.aoe2scenario"
OUT_DIR = Path(__file__).resolve().parent.parent / "build" / "elevation_reference"

# See the module docstring for why a cross-blend_type pair with a close
# blend_priority replaced same-blend_type and priority-mismatched pairs.
CHECKER_A = TerrainId.GRASS_2.value  # blend_type 0, priority 119
CHECKER_B = TerrainId.BEACH_WHITE.value  # blend_type 2, priority 129
CONTROL_TERRAIN = TerrainId.GRASS_2.value  # plateau_control's single-terrain fill
LOCATOR_TERRAIN = TerrainId.ROAD.value  # grey border frame -- findable on the minimap

BORDER = 1  # locator-frame thickness, tiles
MARGIN = 2  # flat checkerboard margin between the frame and the pattern shape, tiles
GAP = 8  # flat gap between regions in the same file, tiles
BASE_X, BASE_Y = 10, 45  # top-left of the first region in every file

DOC_PATH = Path(__file__).resolve().parent.parent / "maintainer" / "docs" / "ELEVATION_REFERENCE.md"

# This machine's live AoE2:DE "My Scenarios" folder for the Steam/Proton
# install under this user account -- confirmed as the right target because
# this is exactly where the Joan coop trigger-upgrade re-saves actually
# landed after being saved from the in-game editor.
# User/install-specific, not derived from anything in this repo -- if this
# ever runs under a different account or a native (non-Proton) DE install,
# find the right folder the same way: save any scenario in-game and see
# where it lands.
DE_SCENARIO_FOLDER = (
    "~/games/steam/steamapps/compatdata/813780/pfx/drive_c/users/steamuser/"
    "Games/Age of Empires 2 DE/76561198040444738/resources/_common/scenario/"
)

# Per-pattern "what to look for" hints for the generated doc -- the
# questions in PATTERNS say what each pattern settles; these say what to
# actually look at on screen to settle it.
DOC_HINTS = {
    "lone_bump": "Look at the single raised tile from directly above (top-down) if possible: does it form a symmetric cone/pyramid, or something else?",
    "lone_pit": "Same as lone_bump, inverted: does the single low tile form a symmetric notch?",
    "ridge": "Sight along the raised band's length: is the seam's slope constant along the whole run, with no visible faceting?",
    "staircase": "A single point raised straight to elevation 7 -- MapManager's own propagation built the surrounding rings (see the generator's docstring). Compare ring-by-ring against this tool's render: same ring radii, same px-per-level spacing?",
    "plateau": "Check a convex (outer) corner of the 8x8 raised square: does DE's ramp there read as the corner's max height, min height, or an average?",
    "plateau_control": "Same corner as plateau, no checkerboard confound -- compare the two screenshots directly against each other.",
    "l_shape": "Check the concave (inner) corner specifically -- this tool's own render may not show a clear cue there; does DE's differ?",
    "diagonal": "Compare the diagonal edge's ramp shape against plateau's axis-aligned edges -- does orientation change anything?",
    "steep_seam": "An intentionally illegal Delta=2 seam (this tool never lets its own edit tools create one) -- does DE render it as a steep double-ramp, clamp it, or silently repair it back to Delta=1?",
}


# ---------------------------------------------------------------------------
# Pattern generators. Each returns (w, h, cells): the pattern's own bounding
# box size, and a {(local_x, local_y): elevation} dict for every tile whose
# target elevation isn't 0 (the blank base's own baseline, so an omitted
# cell needs no edit at all).
# ---------------------------------------------------------------------------


def _lone(sign: int) -> tuple[int, int, dict]:
    """lone_bump (sign=+1): a single tile raised in an otherwise-flat field.
    lone_pit (sign=-1): elevation can't go negative from a 0 baseline, so
    the equivalent one-level notch is built the other way around -- the
    whole field raised to 1 except the center tile, left at 0."""
    size = 7
    cells: dict = {}
    if sign > 0:
        cells[(3, 3)] = 1
    else:
        for x in range(size):
            for y in range(size):
                if (x, y) != (3, 3):
                    cells[(x, y)] = 1
    return size, size, cells


def _ridge() -> tuple[int, int, dict]:
    w, h = 24, 7
    cells = {(x, 3): 1 for x in range(2, 22)}
    return w, h, cells


def _staircase() -> tuple[int, int, dict]:
    """A single point raised straight to the max legal elevation (7),
    letting MapManager's own propagation build the surrounding rings --
    not hand-built bands. Two hand-built shapes were tried first and both
    leaked: a one-way 0->7 ramp butts its level-7 end directly against the
    flat margin (Delta=7); mirroring it back down to 0 fixed that end but
    every band *also* borders the flat margin above and below along its
    whole length (any level>=2 band vs. the y=0/y=h-1 margin rows is
    already Delta>=2). set_tile_elevation always runs
    MapManager._elevation_tile_recursion regardless of the delta actually
    being edited (see elevation_tools.py); that recursion only ACTS when a
    neighbor is more than 1 off, but both hand-built shapes hit that
    trigger somewhere along their own boundary -- confirmed directly, not
    hypothetical: both leaked stray skirt lines through the checkerboard
    margin and the locator border into the surrounding background in the
    rendered PNG.

    A single center point sidesteps the whole problem: the SAME recursion
    that made the hand-built shapes leak is also exactly the mechanism
    MapManager.set_elevation's own docstring describes ("Sets elevation
    like the in-game elevation mechanics... all tiles around it are
    adjusted accordingly") -- confirmed directly (see the exploration that
    landed on this design): raising one tile to 7 from a flat 0 base
    radiates clean concentric rings (Chebyshev-distance shaped, each ring
    exactly 1 below the next, terminating at 0 with zero manual shape math)
    with no illegal delta anywhere, because every ring's own construction
    is the propagation invariant itself. This is also a more authentic
    test than a hand-built ramp: it's the same one-click behavior DE's own
    elevation brush should produce at a single point, so the comparison
    screenshot is checking the tool's real single-point path, not a shape
    this script invented."""
    radius = 7  # deliberate fixed value, no longer tracking MAX_ELEVATION (now 15) --
    # changing it would regenerate the probe map and invalidate the in-game
    # comparison screenshots already taken against this exact ring shape.
    size = 2 * radius + 3  # +3 margin beyond the ring's own reach on every side
    center = size // 2
    return size, size, {(center, center): radius}


def _plateau_square(n: int = 8) -> tuple[int, int, dict]:
    cells = {(x, y): 1 for x in range(n) for y in range(n)}
    return n, n, cells


def _l_shape(n: int = 12, cut: int = 5) -> tuple[int, int, dict]:
    """n x n square, minus a cut x cut corner -- the concave (inner) corner
    plateau's own convex corners can't exercise."""
    cells = {}
    for x in range(n):
        for y in range(n):
            if x >= n - cut and y < cut:
                continue
            cells[(x, y)] = 1
    return n, n, cells


def _diagonal(n: int = 10) -> tuple[int, int, dict]:
    cells = {(x, y): 1 for x in range(n) for y in range(n) if x + y <= n}
    return n, n, cells


def _steep_seam() -> tuple[int, int, dict]:
    """4 bands at elevation 0/2/4/6 -- Delta=2 between each pair, an
    intentionally illegal seam. Flagged skip_helper by the caller so this
    goes through a raw TerrainTile.elevation assignment, never
    set_tile_elevation's propagation (which would repair it on the spot)."""
    band_w = 6
    n_bands = 4
    w = band_w * n_bands
    h = 10
    cells = {}
    for b in range(1, n_bands):  # band 0 stays at the 0 baseline
        elev = b * 2
        for i in range(band_w):
            x = b * band_w + i
            for y in range(h):
                cells[(x, y)] = elev
    return w, h, cells


# name -> (question, generator, checker: True=checkerboard/False=single CONTROL_TERRAIN,
#          skip_helper: True=raw elevation assignment, bypassing set_tile_elevation)
PATTERNS = {
    "lone_bump": (
        "does an isolated tile ramp on all four sides, or take a discrete shape?",
        lambda: _lone(+1),
        True,
        False,
    ),
    "lone_pit": (
        "same question as lone_bump, for a one-level notch instead of a bump",
        lambda: _lone(-1),
        True,
        # skip_helper=True: NOT for a delta reason (every edge here is legal
        # Delta<=1) -- confirmed directly (dumped the region's elevation
        # grid) that going through set_tile_elevation for the surrounding
        # ring erases the notch regardless of application order.
        # MapManager._elevation_tile_recursion has a "fill a one-tile gap
        # between two equal-height neighbors" rule (the `behind` branch in
        # elevation_tools.py's own quoted excerpt) that's specifically
        # meant for smoothing normal edits, but it also fires on this
        # pattern's own intentional untouched center once enough of the
        # ring is in place, silently flattening the pit into a plateau.
        # Raw assignment sidesteps that rule entirely -- same carve-out as
        # steep_seam, different reason (gap-filling here, not illegal-delta
        # repair there).
        True,
    ),
    "ridge": (
        "the basic seam, axis-aligned -- how does DE render one straight Delta=1 edge?",
        _ridge,
        True,
        False,
    ),
    "staircase": (
        "px per elevation level; is every seam treated alike across all 7 steps?",
        _staircase,
        True,
        False,
    ),
    "plateau": (
        "the corner-height discriminator -- max vs min vs average is readable off a convex corner",
        lambda: _plateau_square(8),
        True,
        False,
    ),
    "plateau_control": (
        "plateau's corner-height question, with terrain blending removed as a confound",
        lambda: _plateau_square(8),
        False,
        False,
    ),
    "l_shape": (
        "the concave (inner) corner, complementing plateau's convex one",
        _l_shape,
        True,
        False,
    ),
    "diagonal": (
        "does ramp shape depend on edge orientation, not just axis-aligned/convex/concave?",
        _diagonal,
        True,
        False,
    ),
    "steep_seam": (
        "does DE render, clamp, or silently repair an illegal Delta>=2 seam?",
        _steep_seam,
        True,
        True,
    ),
}

FILE_GROUPS = [
    ("reference_a", ["lone_bump", "lone_pit", "ridge"]),
    ("reference_b", ["staircase", "plateau", "plateau_control"]),
    ("reference_c", ["l_shape", "diagonal", "steep_seam"]),
]


def _paint_region(mm, pattern_name: str, ox: int, oy: int) -> dict:
    """Paints one pattern's border/checkerboard/elevation at map offset
    (ox, oy) (the frame's own top-left, not the shape's). Returns metadata
    (tile bbox including the frame, shape bbox, max elevation used) for the
    render-cropping and doc-generation passes."""
    question, generator, checker, skip_helper = PATTERNS[pattern_name]
    shape_w, shape_h, cells = generator()

    checker_x0, checker_y0 = ox + BORDER, oy + BORDER
    shape_x0, shape_y0 = checker_x0 + MARGIN, checker_y0 + MARGIN
    checker_x1, checker_y1 = shape_x0 + shape_w + MARGIN, shape_y0 + shape_h + MARGIN
    frame_x0, frame_y0 = checker_x0 - BORDER, checker_y0 - BORDER
    frame_x1, frame_y1 = checker_x1 + BORDER, checker_y1 + BORDER

    for x in range(checker_x0, checker_x1):
        for y in range(checker_y0, checker_y1):
            tile = mm.get_tile(x, y)
            if checker:
                terrain_id = CHECKER_A if (x + y) % 2 == 0 else CHECKER_B
            else:
                terrain_id = CONTROL_TERRAIN
            set_terrain(tile, terrain_id)

    for x in range(frame_x0, frame_x1):
        for y in range(frame_y0, frame_y1):
            on_frame = x in (frame_x0, frame_x1 - 1) or y in (frame_y0, frame_y1 - 1)
            if on_frame:
                set_terrain(mm.get_tile(x, y), LOCATOR_TERRAIN)

    max_elev = 0
    for (lx, ly), elev in cells.items():
        x, y = shape_x0 + lx, shape_y0 + ly
        if skip_helper:
            mm.get_tile(x, y).elevation = elev
        else:
            set_tile_elevation(mm, x, y, elev)
        max_elev = max(max_elev, elev)

    return {
        "name": pattern_name,
        "question": question,
        "frame_bbox": (frame_x0, frame_y0, frame_x1, frame_y1),
        "shape_bbox": (shape_x0, shape_y0, shape_x0 + shape_w, shape_y0 + shape_h),
        "max_elev": max_elev,
    }


def _crop_bbox_px(frame_bbox, max_elev: int, proj) -> tuple[int, int, int, int]:
    """Screen-pixel crop bounds for one region's frame_bbox (tile-space,
    x1/y1 exclusive), covering every corner at both elevation 0 and the
    region's own max_elev, plus skirt headroom below for the tallest drop
    this region can actually show -- deliberately using the region's own
    observed max_elev rather than iso_geometry's fixed [0, MAX_ELEVATION]
    canvas range, which would pad every crop with several times more empty
    canvas than any of these patterns actually needs."""
    x0, y0, x1, y1 = frame_bbox
    half_w, half_h, elev_step = proj.half_w, proj.half_h, proj.elev_step
    corners = [(x0, y0), (x1 - 1, y0), (x0, y1 - 1), (x1 - 1, y1 - 1)]
    xs, ys = [], []
    for cx, cy in corners:
        for e in (0, max_elev):
            sx, sy = tile_screen_origin(cx, cy, e, proj)
            xs += [sx, sx + 2 * half_w]
            ys += [sy, sy + 2 * half_h]
    ys.append(max(ys) + max_elev * elev_step)  # skirt headroom for the tallest drop here
    pad = 4
    return max(0, min(xs) - pad), max(0, min(ys) - pad), max(xs) + pad, max(ys) + pad


def generate(base_path: Path, out_dir: Path, write_renders: bool) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    regions: list[dict] = []

    for file_stem, pattern_names in FILE_GROUPS:
        scenario = load_map_and_units(base_path)
        if not scenario.map_is_square or not scenario.terrain_write_supported:
            raise SystemExit(
                f"{base_path} isn't usable as a Track A0 base "
                f"(square={scenario.map_is_square}, terrain_write_supported="
                f"{scenario.terrain_write_supported})"
            )
        mm = scenario.map_manager
        cursor_x = BASE_X
        file_regions = []
        for pattern_name in pattern_names:
            meta = _paint_region(mm, pattern_name, cursor_x, BASE_Y)
            meta["file"] = file_stem
            file_regions.append(meta)
            frame_x1 = meta["frame_bbox"][2]
            cursor_x = frame_x1 + GAP
        if cursor_x > mm.map_width:
            raise SystemExit(
                f"{file_stem}'s regions need width {cursor_x}, base map is only "
                f"{mm.map_width} wide"
            )

        out_path = out_dir / f"{file_stem}.aoe2scenario"
        write_scenario(scenario, out_path)
        print(f"wrote {out_path} ({len(pattern_names)} regions)")

        if write_renders:
            reloaded = load_map_and_units(out_path)
            img, _elevations, proj = render.render_terrain_iso_with_proj(reloaded, with_units=False)
            # Sloped's own proj is geometrically identical to Stepped's for
            # the same map/tile_px (corner_headroom_px doesn't affect canvas
            # sizing -- see IsoProjection's own comment), so the same
            # _crop_bbox_px() call below is valid for both without separate
            # crop math. Phase 6 (Sloped)'s reference render, for Track C6's
            # side-by-side against the in-game screenshots this script's own
            # docstring hands off (docs/ELEVATION_REFERENCE.md).
            sloped_img, _sloped_elevations, _sloped_corner_rise, _sloped_proj = (
                render.render_terrain_sloped_with_proj(reloaded, with_units=False)
            )
            renders_dir = out_dir / "renders"
            renders_dir.mkdir(parents=True, exist_ok=True)
            for meta in file_regions:
                px0, py0, px1, py1 = _crop_bbox_px(meta["frame_bbox"], meta["max_elev"], proj)
                px1, py1 = min(px1, img.shape[1]), min(py1, img.shape[0])
                crop = img[py0:py1, px0:px1]
                render_path = renders_dir / f"{meta['name']}.png"
                _save_png(crop, render_path)
                meta["render_path"] = render_path
                print(f"  {meta['name']}: cropped {crop.shape[1]}x{crop.shape[0]} -> {render_path}")

                sloped_px1, sloped_py1 = min(px1, sloped_img.shape[1]), min(py1, sloped_img.shape[0])
                sloped_crop = sloped_img[py0:sloped_py1, px0:sloped_px1]
                sloped_render_path = renders_dir / f"{meta['name']}_sloped.png"
                _save_png(sloped_crop, sloped_render_path)
                meta["sloped_render_path"] = sloped_render_path

        regions.extend(file_regions)

    return regions


def _save_png(img, out_path: Path) -> None:
    from PIL import Image

    Image.fromarray(img, mode="RGB").save(out_path)


def _write_doc(regions: list[dict], out_dir: Path) -> None:
    """Writes docs/ELEVATION_REFERENCE.md (Track A3): what to screenshot per
    region and which question it settles, plus where the generated files go
    -- generated, not hand-maintained, matching this repo's convention for
    docs derived from something else that can drift (see docs/ACTORS.md/
    CVARS.md's own generated-index precedent in the sibling orc_slayer
    project's CLAUDE.md, and this repo's own build/color_browser.html)."""
    lines = [
        "# Elevation reference screenshots",
        "",
        "Generated by `tools/gen_elevation_reference.py` -- do not hand-edit, "
        "regenerate instead (maintainer-only: needs the gitignored "
        "examples/ corpus, a hardcoded Proton path, and a live AoE2:DE "
        "install). Settles Phase 6's sloped-ramp design questions "
        "(the v2.6 rendering plan's Track A and Track C) from real in-game "
        "screenshots. Per Track A4 this also closed the project backlog's "
        "now-removed \"Elevation edits: in-game verification\" item -- "
        "loading these files at all, with their elevation edits intact, "
        "was what closed it, regardless of what the screenshots went on "
        "to show about ramp shape.",
        "",
        "## Where the files go",
        "",
        "Copy every `.aoe2scenario` file in `build/elevation_reference/` "
        "(not the `renders/` subfolder -- those are PNGs, not scenarios) "
        "into this install's scenario folder:",
        "",
        f"```\n{DE_SCENARIO_FOLDER}\n```",
        "",
        "Then open each from the in-game scenario editor's own file list "
        "(not Explorer/Finder) and switch to Terrain view to see the raised "
        "regions -- each is flagged with a grey ROAD-terrain frame so it's "
        "findable on the minimap. Only 3 files to open in total, each "
        "holding 3 regions, so this is a 3-load pass, not 9.",
        "",
        "## What to screenshot",
        "",
        "For each region: zoom in close -- every frame is small (well under "
        "40 tiles across) -- so individual tile boundaries are readable, and "
        "get the whole frame in frame. A top-down or near-top-down camera "
        "angle reads corner height most clearly where a hint below calls "
        "for it.",
        "",
    ]

    by_file: dict[str, list[dict]] = {}
    for meta in regions:
        by_file.setdefault(meta["file"], []).append(meta)

    for file_stem, file_regions in by_file.items():
        lines.append(f"### `{file_stem}.aoe2scenario`")
        lines.append("")
        for meta in file_regions:
            fx0, fy0, fx1, fy1 = meta["frame_bbox"]
            render_rel = Path("..") / "build" / "elevation_reference" / "renders" / f"{meta['name']}.png"
            lines.append(f"#### {meta['name']}")
            lines.append("")
            lines.append(f"- **Settles:** {meta['question']}")
            lines.append(f"- **Look for:** {DOC_HINTS[meta['name']]}")
            lines.append(
                f"- **Map location:** tiles ({fx0}, {fy0}) to ({fx1}, {fy1}), "
                f"max elevation {meta['max_elev']}"
            )
            lines.append(f"- **This tool's own render, for comparison:** `{render_rel.as_posix()}`")
            lines.append("")

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {DOC_PATH}")


# pattern_name -> [(local_x, local_y, expected_elevation), ...], relative to
# that region's own shape_bbox top-left -- regression coverage for the two
# real bugs found while building this: a hand-built multi-level ramp
# leaking elevation through the margin into the background (any pattern,
# caught generically by check()'s leak scan below instead), and
# set_tile_elevation's "fill a one-tile gap" rule silently erasing
# lone_pit's own untouched center once the surrounding ring closed (caught
# only by asserting the exact cell value -- a leak scan can't see it, since
# the erased cell is inside the frame, not outside it).
SPOT_CHECKS = {
    "lone_bump": [(3, 3, 1)],
    "lone_pit": [(3, 3, 0)],
    "steep_seam": [(0, 0, 0), (6, 0, 2), (12, 0, 4), (18, 0, 6)],
}


def check(base_path: Path) -> None:
    """--check: writes to a temp dir (not build/), then verifies the actual
    per-tile data a screenshot comparison depends on -- not just structure.
    Earlier versions of this check only asserted region count, non-
    degenerate bboxes, and terrain_write_supported, which is exactly why
    two real bugs (a propagation leak past the frame, and a silently
    erased lone_pit center) only surfaced from eyeballing rendered PNGs,
    twice. Now checks, per file, after a real reload:
    1. No elevation leaked outside every region's own frame_bbox (the leak
       bug's exact signature -- background must stay 0).
    2. SPOT_CHECKS' known cells hold their expected value (the erased-
       center bug's signature -- inside the frame, so (1) can't see it).
    3. Terrain painted correctly: checkerboard/control cells and the
       locator frame."""
    import tempfile

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        regions = generate(base_path, tmp_dir, write_renders=False)
        if len(regions) != len(PATTERNS):
            print(f"FAIL: generated {len(regions)} regions, expected {len(PATTERNS)}")
            failures += 1
        for meta in regions:
            fx0, fy0, fx1, fy1 = meta["frame_bbox"]
            if fx1 <= fx0 or fy1 <= fy0:
                print(f"FAIL: {meta['name']} has a degenerate frame_bbox {meta['frame_bbox']}")
                failures += 1

        by_file: dict[str, list[dict]] = {}
        for meta in regions:
            by_file.setdefault(meta["file"], []).append(meta)

        for file_stem, file_regions in by_file.items():
            out_path = tmp_dir / f"{file_stem}.aoe2scenario"
            reloaded = load_map_and_units(out_path)
            if not reloaded.terrain_write_supported:
                print(f"FAIL: {file_stem} reloaded with terrain_write_supported=False")
                failures += 1
                continue
            mm = reloaded.map_manager

            inside: set[tuple[int, int]] = set()
            for meta in file_regions:
                fx0, fy0, fx1, fy1 = meta["frame_bbox"]
                inside.update((x, y) for x in range(fx0, fx1) for y in range(fy0, fy1))
            leaked = [
                (t.x, t.y, t.elevation)
                for t in mm.terrain
                if t.elevation != 0 and (t.x, t.y) not in inside
            ]
            if leaked:
                sample = ", ".join(f"({x},{y})={e}" for x, y, e in leaked[:5])
                print(f"FAIL: {file_stem} leaked elevation outside every region's frame: {sample} ...")
                failures += 1

            for meta in file_regions:
                sx0, sy0, _, _ = meta["shape_bbox"]
                for lx, ly, expected in SPOT_CHECKS.get(meta["name"], []):
                    got = mm.get_tile(sx0 + lx, sy0 + ly).elevation
                    if got != expected:
                        print(
                            f"FAIL: {meta['name']} cell ({lx},{ly}) expected elevation "
                            f"{expected}, got {got}"
                        )
                        failures += 1

                cx0, cy0, _cx1, _cy1 = (
                    meta["frame_bbox"][0] + BORDER,
                    meta["frame_bbox"][1] + BORDER,
                    meta["frame_bbox"][2] - BORDER,
                    meta["frame_bbox"][3] - BORDER,
                )
                _, _, checker, _ = PATTERNS[meta["name"]]
                sample_tile = mm.get_tile(cx0, cy0)
                expected_terrain = (CHECKER_A, CHECKER_B) if checker else (CONTROL_TERRAIN,)
                if sample_tile.terrain_id not in expected_terrain:
                    print(
                        f"FAIL: {meta['name']} checker corner has terrain_id "
                        f"{sample_tile.terrain_id}, expected one of {expected_terrain}"
                    )
                    failures += 1
                frame_tile = mm.get_tile(meta["frame_bbox"][0], meta["frame_bbox"][1])
                if frame_tile.terrain_id != LOCATOR_TERRAIN:
                    print(
                        f"FAIL: {meta['name']} frame corner has terrain_id "
                        f"{frame_tile.terrain_id}, expected LOCATOR_TERRAIN={LOCATOR_TERRAIN}"
                    )
                    failures += 1

    if failures:
        raise SystemExit(f"{failures} check(s) failed")
    print(f"OK: {len(regions)} regions across {len(FILE_GROUPS)} files, all reload clean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Track A0 blank base scenario")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument("--check", action="store_true", help="Validate without writing any files")
    args = parser.parse_args()

    if not args.base.is_file():
        raise SystemExit(f"Base scenario not found: {args.base}")

    if args.check:
        check(args.base)
        return

    regions = generate(args.base, args.out_dir, write_renders=True)
    _write_doc(regions, args.out_dir)


if __name__ == "__main__":
    main()
