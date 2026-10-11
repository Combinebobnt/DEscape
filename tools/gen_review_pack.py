#!/usr/bin/env python3
"""Renders a REVIEW PACK -- a set of framed captures plus a checklist written
before the images exist -- for a blind agent to judge. Always writes, never
pass/fail, no golden images anywhere. Modelled on tools/gen_seam_eyeball.py
and reusing its fixture, crop and render helpers directly.

The problem this exists for is not "a human has to look". It is that the
session which generated a capture already knows what it was supposed to
prove, and passes it. This repo has that recorded twice: a sprite floated
half a tile above ground for days behind a visual confirmation that passed,
and "is the seam line continuous" was read as answering "does the shadow read
as connected". So the deliverable is a checklist fixed in code before any
image exists, handed to a reviewer that is told nothing about what changed.

Run:
    tools/gen_review_pack.py --pack shadow_band
    tools/gen_review_pack.py --pack shadow_band --inject no-apex-wedge
    tools/gen_review_pack.py --pack ground_texture --inject old-basis
    tools/gen_review_pack.py --check --pack ground_texture

then hand build/review_pack/<pack>/REVIEW.md to a fresh agent, in its own
call, with nothing else. See tools/REVIEW_PACK.md for the protocol and for
what a verdict may not close. The ground_texture pack (TASK-201's iso
texture continuity) renders a real examples/ map with the real install's
textures, so it needs both; shadow_band needs neither. So does sloped_mip
(TASK-031.60's Sloped mip ladder), which stitches its frames from
SlopedChunkCache.render_rect() at the coarser levels, sprites included. So does
ground_borders (TASK-143's terrain blending), over two examples/ maps.

Off-engine throughout (descape.render.composite_rect_iso, no PyQt5), unlike
gen_seam_eyeball.py's main sweep: composite_rect_iso() is byte-identically
the function IsoChunkCache.render_rect() calls per mip chunk, every frame
here is a crop rather than a window grab, and staying Qt-free keeps the tool
clear of the offscreen ViewerWindow's config write-through and its modal
close prompt. Textures are the real .dds set, because that is what the user
sees -- the flat terrain_palette colours a pytest run gets would remove the
texture noise these checks have to survive.

Writes build/review_pack/<pack>/, gitignored. No test reads the PNGs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_seam_eyeball as seam_eyeball
import numpy as np

from descape import asset_source, native_composite, render, terrain_blend
from descape import iso_geometry as ig
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.view_layers import LayerState
from testkit import review_pack as rp

OUT_DIR = ROOT / "build" / "review_pack"

TILE_PX = render.SMALL_MAP_TILE_PIXELS

# Same reasoning as gen_seam_eyeball._ANCHOR: MapView.set_source() draws a
# map-extent outline near the map's own corner that render_terrain_iso()
# never draws.
_ANCHOR = seam_eyeball._ANCHOR

# Where the straight terrace edge sits in _straight_run_scenario, and the
# span of it each run frame crops.
_RUN_EDGE_Y = 60
_RUN_X0, _RUN_X1 = 56, 65

# The inner corner: A and B one level up, both casting onto N = (A.x+1, A.y).
_CORNER_A = _ANCHOR


def _clear_caches() -> None:
    render._shadow_factors.cache_clear()
    render._seam_factors.cache_clear()


# ---------------------------------------------------------------- fixtures


def _flat_scenario():
    """Elevation 0 everywhere: the specificity control. No step, so no band,
    no seam and no wedge are geometrically possible -- only the real terrain
    texture's own noise. A reviewer that reports marks here has noise-level
    detection, and that is worth learning in the same run rather than a run
    later."""
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = 0
    return scenario


def _straight_run_scenario():
    """A single half-plane one level up, so every caster along y == _RUN_EDGE_Y
    has a lower back neighbour on ONE side only (up_left) and a level one on
    the other. That one-sidedness is the whole point: it is the configuration
    where the apex wedge's wrong-side flank draws a perpendicular tick, and
    there is no corner anywhere in the frame to confound the question."""
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = 1 if tile.y >= _RUN_EDGE_Y else 0
    return scenario


def _corner_scenario():
    """A = (x, y) and B = (x+1, y+1) one level up, so both cast onto
    N = (x+1, y): the smallest inner corner, where the two bands meet at the
    casters' unshaded diamond tip columns."""
    ax, ay = _CORNER_A
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = 1 if (tile.x, tile.y) in ((ax, ay), (ax + 1, ay + 1)) else 0
    return scenario


def _tiles_bbox_px(proj, tiles, max_elev: int, pad_px: int = 0):
    """Screen-pixel bounds covering every named tile at both elevation 0 and
    max_elev, plus skirt and contact-shadow headroom below -- the same
    technique as gen_seam_eyeball._crop_bbox_px and gen_elevation_reference's
    own version."""
    xs, ys = [], []
    for tx, ty in tiles:
        for elev in (0, max_elev):
            sx, sy = ig.tile_screen_origin(tx, ty, elev, proj)
            xs += [sx, sx + 2 * proj.half_w]
            ys += [sy, sy + 2 * proj.half_h]
    ys.append(max(ys) + max_elev * proj.elev_step)
    return (
        max(0, min(xs) - pad_px),
        max(0, min(ys) - pad_px),
        max(xs) + pad_px,
        max(ys) + pad_px,
    )


def _projection(scenario, pct: int):
    mm = scenario.map_manager
    return ig.canvas_size_and_origin(
        mm.map_width, mm.map_height, TILE_PX, ig.MIN_ELEVATION, ig.MAX_ELEVATION, elev_step_pct=pct
    )


def _render_rect(scenario, proj, box) -> np.ndarray:
    """Off-engine render of one screen-pixel rect, not the whole canvas.

    composite_rect_iso() is exactly the function IsoChunkCache.render_rect()
    calls per mip chunk, and tools/verify_iso_chunks.py asserts a rect render
    is byte-identical to the same region of a full-canvas one -- so cropping
    at render time rather than after costs nothing in fidelity and takes the
    pack from a 7680x4320 canvas per frame to a few hundred pixels square.
    That is the difference between a --check the default tier can afford and
    one it cannot.
    """
    mm = scenario.map_manager
    elevations = np.zeros((mm.map_height, mm.map_width), dtype=np.int64)
    for tile in mm.terrain:
        elevations[tile.y, tile.x] = tile.elevation
    x0, y0, x1, y1 = box
    return render.composite_rect_iso(
        scenario,
        x0,
        y0,
        x1,
        y1,
        elevations,
        proj,
        TILE_PX,
        units_by_tile={},
        building_bboxes={},
        with_units=False,
    )


# ----------------------------------------------------------------- injects


class _GeometryPatch:
    """Swaps one descape.iso_geometry entry point for the run, and clears both
    factor caches on the way in and out. render.py reaches these functions
    through the module attribute, and _shadow_factors() reaches them the same
    way, so indices and factors stay the same length under a patch."""

    def __init__(self, inject) -> None:
        self.inject = inject

    def __enter__(self):
        if self.inject is not None:
            self.saved = getattr(ig, self.inject.attr)
            setattr(ig, self.inject.attr, self.inject.fn)
        _clear_caches()
        return self

    def __exit__(self, *exc):
        if self.inject is not None:
            setattr(ig, self.inject.attr, self.saved)
        _clear_caches()


_EMPTY = np.zeros(0, dtype=np.int64)


def _inject_no_apex_wedge(_tile_px, _rise_px):
    """COARSE. Gates the apex-wedge pass off entirely, reopening the hole at
    every junction along a terrace so a run of adjacent casters reads as
    separate blocks instead of one band. Validates only that the reviewer is
    looking at the right frame and can see the band at all."""
    return (_EMPTY, _EMPTY, _EMPTY, _EMPTY)


@dataclass(frozen=True)
class _Inject:
    """One known defect: which iso_geometry entry point it replaces, the
    replacement, and the bound --check holds it to. `bound=None` marks the
    pack's coarse inject, which is asserted on component count instead."""

    attr: str
    fn: object
    bound: object


# The localized injects (notch-inject, no-tip) toggled the side-split apex
# wedge and the inner-corner tip pass, which were backed out (GH #15); their
# defects are the clean render now. A localized inject for the line-style
# render is still owed, so the size-differs check below has nothing to compare.
INJECTS = {
    "no-apex-wedge": _Inject("shadow_apex_indices", _inject_no_apex_wedge, None),
}

# The coarse inject is the yardstick any localized one is localized AGAINST,
# so "they differ in size" is asserted on the frames they share rather than
# only against a constant.
COARSE_INJECT = "no-apex-wedge"

# no-apex-wedge must genuinely break the contour, not merely change pixels.
# Clean reads 1 connected component on the pyramid band at pct 100 (the wedge
# bridges into the next terrace ring there; 3 at pct 50); with the wedge gone
# it reads 18.
COARSE_MIN_COMPONENT_RATIO = 3.0

# Topology is countable, so it is asserted here rather than asked of a
# reviewer. Band at pct 100: one piece. Corner diff mask: the two casters'
# lines stop either side of a 2-column bare slot at the vertex, so two pieces.
CLEAN_BAND_COMPONENTS = 1
CLEAN_CORNER_COMPONENTS = 2


# ----------------------------------------------------------------- rendering


def _render_pair(scenario, pct: int, box_tiles, max_elev: int, pad_px: int):
    """(shipped crop, contact-neutralised crop). The control neutralises
    CONTACT_SHADE ONLY, leaving the seam line in place, so the amplified diff
    isolates the contact band by itself. Neutralising both would fold the
    continuous seam line into the same mask and hide exactly the breaks
    band_connectivity asks about."""
    proj = _projection(scenario, pct)
    box = _tiles_bbox_px(proj, box_tiles, max_elev, pad_px=pad_px)
    _clear_caches()
    shipped = _render_rect(scenario, proj, box)
    saved = render.CONTACT_SHADE
    try:
        render.CONTACT_SHADE = 1.0
        _clear_caches()
        control = _render_rect(scenario, proj, box)
    finally:
        render.CONTACT_SHADE = saved
        _clear_caches()
    return shipped, control


_BAND_TILES = [
    (tx, ty)
    for tx in range(_ANCHOR[0] - seam_eyeball.PYRAMID_RADIUS, _ANCHOR[0] + seam_eyeball.PYRAMID_RADIUS + 1)
    for ty in range(_ANCHOR[1] - seam_eyeball.PYRAMID_RADIUS, _ANCHOR[1] + seam_eyeball.PYRAMID_RADIUS + 1)
]
_RUN_TILES = [(tx, ty) for tx in range(_RUN_X0, _RUN_X1) for ty in (_RUN_EDGE_Y - 1, _RUN_EDGE_Y)]
_CORNER_TILES = [
    (_CORNER_A[0] + dx, _CORNER_A[1] + dy) for dx in (-1, 0, 1, 2) for dy in (-1, 0, 1, 2)
]

# (fixture, tiles the crop must cover, that fixture's peak elevation).
_SCENES = {
    "band": (seam_eyeball._pyramid_scenario, _BAND_TILES, seam_eyeball.PYRAMID_MAX_ELEV),
    "run": (_straight_run_scenario, _RUN_TILES, 1),
    "corner": (_corner_scenario, _CORNER_TILES, 1),
    # Same fixture geometry and framing as the run frames, so the specificity
    # control differs from them only in having no step to cast anything.
    "flat": (_flat_scenario, _RUN_TILES, 1),
}

_SCENARIO_CACHE: dict[str, object] = {}

# Context kept around the feature once the crop tightens onto it. Enough to
# see the edge sitting in real ground, not so much that the ground wins the
# frame.
_FEATURE_PAD_PX = 12


def _scenario(scene: str):
    if scene not in _SCENARIO_CACHE:
        _SCENARIO_CACHE[scene] = _SCENES[scene][0]()
    return _SCENARIO_CACHE[scene]


def _scene_pair(scene: str, pct: int, box=None):
    """(shipped crop, control crop), tightened onto where the contact shading
    actually landed rather than onto the tiles it was cast from.

    `box` is passed in for an injected run so every pack frames the identical
    view: a defect that widened the feature would otherwise widen the crop
    with it, and the reviewer could read which pack it was holding off the
    framing alone.
    """
    _factory, tiles, max_elev = _SCENES[scene]
    raw, control = _render_pair(_scenario(scene), pct, tiles, max_elev, 0)
    if box is None:
        box = rp.feature_bbox(rp.changed_pixels(raw, control), _FEATURE_PAD_PX, raw.shape[:2])
    return rp.crop(raw, box), rp.crop(control, box)


def _clean_boxes() -> dict[tuple[str, int], tuple[int, int, int, int]]:
    """Feature crop boxes, measured once off the un-injected render."""
    boxes = {}
    with _GeometryPatch(None):
        for scene, pcts in (("band", (50, 100)), ("run", (10, 50, 100)), ("corner", (50, 100))):
            for pct in pcts:
                _factory, tiles, max_elev = _SCENES[scene]
                raw, control = _render_pair(_scenario(scene), pct, tiles, max_elev, 0)
                boxes[(scene, pct)] = rp.feature_bbox(
                    rp.changed_pixels(raw, control), _FEATURE_PAD_PX, raw.shape[:2]
                )
    return boxes


def _flat_frame(size: tuple[int, int]) -> np.ndarray:
    """The specificity control, cropped to the same pixel size as the run
    frames it sits beside. There is no feature to crop to here, by
    construction -- that is the whole point of the frame."""
    factory, tiles, max_elev = _SCENES["flat"]
    if "flat" not in _SCENARIO_CACHE:
        _SCENARIO_CACHE["flat"] = factory()
    scenario = _SCENARIO_CACHE["flat"]
    proj = _projection(scenario, 100)
    x0, y0, _x1, _y1 = _tiles_bbox_px(proj, tiles, max_elev)
    height, width = size
    _clear_caches()
    return _render_rect(scenario, proj, (x0, y0, x0 + width, y0 + height))


# --------------------------------------------------------------- pack specs


def _shadow_band_spec() -> rp.PackSpec:
    """The pilot pack, re-posed in round 3. Each question sits on frames that
    can answer it: a pyramid for whether the band breaks, a pure one-sided
    straight run for perpendicular artifacts, a bare inner corner for whether
    its vertex mass is split, and a flat field where none of them is possible.
    Every closed question's EXCLUDE clause is written against a mask measured
    off the injects themselves, not against assumed geometry."""
    frames: list[rp.Frame] = []
    for pct in (50, 100):
        frames += [
            rp.Frame(f"band_pct{pct}_raw", f"band_pct{pct}_raw.png", "raw", caption=f"Terraced hill, vertical scale {pct}%."),
            rp.Frame(
                f"band_pct{pct}_control",
                f"band_pct{pct}_control.png",
                "control",
                caption=f"The same hill with one shading pass switched off, vertical scale {pct}%.",
            ),
            rp.Frame(
                f"band_pct{pct}_diff",
                f"band_pct{pct}_diff.png",
                "diff",
                caption=(
                    f"Amplified difference of the two frames above, vertical scale {pct}%. "
                    "Only the pixels that pass touched, on black."
                ),
            ),
        ]
    for pct in (10, 50, 100):
        frames += [
            rp.Frame(
                f"run_pct{pct}_raw",
                f"run_pct{pct}_raw.png",
                "raw",
                caption=f"A single straight terrace edge, vertical scale {pct}%.",
            ),
            rp.Frame(
                f"run_pct{pct}_diff",
                f"run_pct{pct}_diff.png",
                "diff",
                caption=f"Amplified difference against the same edge with one shading pass switched off, vertical scale {pct}%.",
            ),
        ]
    for pct in (50, 100):
        frames += [
            rp.Frame(
                f"corner_pct{pct}_raw",
                f"corner_pct{pct}_raw.png",
                "raw",
                caption=f"Two raised tiles meeting at a corner over one lower tile, vertical scale {pct}%.",
            ),
            rp.Frame(
                f"corner_pct{pct}_control",
                f"corner_pct{pct}_control.png",
                "control",
                caption=f"The same corner with one shading pass switched off, vertical scale {pct}%.",
            ),
            rp.Frame(
                f"corner_pct{pct}_diff",
                f"corner_pct{pct}_diff.png",
                "diff",
                caption=(
                    f"Amplified difference of the two corner frames above, vertical scale {pct}%. "
                    "Only the pixels that pass touched, on black."
                ),
            ),
        ]
    frames.append(
        rp.Frame(
            "flat_raw",
            "flat_raw.png",
            "raw",
            caption="Ground at a single uniform height, same terrain and same scale as the straight-edge frames.",
        )
    )

    checks = (
        rp.Check(
            id="band_connectivity",
            question=(
                "First, in your own words, describe the shading the two amplified difference "
                "frames show along each terrace ring of the hill: where it is thick, where it "
                "thins, and whether it ever stops.\n\n"
                "Then the closed question, which is about whether that shading is broken, not "
                "about how even or how thick it is. Judge it on the difference frames, where only "
                "the pixels one shading pass touched appear, on black. The raw and switched-off "
                "frames are context only: both carry a separate thin dark line along every "
                "terrace edge, drawn by a different pass, and a line that also appears in the "
                "switched-off frame is NOT what this question asks about, however continuous or "
                "broken it looks.\n"
                "- A BREAK is a place where a ring's shading stops entirely: plain black, with no "
                "shading at all, between two separate pieces of it. At this enlargement such a "
                "stretch is at least about 40 pixels long.\n"
                "- NOT a break: a stretch where the shading thins to a single line only a few "
                "pixels thick at this enlargement (one pixel in the original) but carries on to "
                "the next piece. That counts as connected, however thin it is beside the thicker "
                "pieces it joins, and however much the ring looks like a row of blocks strung on "
                "a line.\n\n"
                "Answer BREAKS-PRESENT if at least one ring has a break, NO-BREAKS if every "
                "ring's shading is continuous, even where only a thin line carries it."
            ),
            frames=(
                "band_pct50_raw",
                "band_pct50_control",
                "band_pct50_diff",
                "band_pct100_raw",
                "band_pct100_control",
                "band_pct100_diff",
            ),
            answers=("BREAKS-PRESENT", "NO-BREAKS", "CANNOT-TELL"),
        ),
        rp.Check(
            id="straight_run_edge",
            question=(
                "First, in your own words, describe every dark mark you can see along this straight "
                "terrace edge, and for each one say whether it touches the main dark contour or is "
                "separated from it by a strip of ordinary un-darkened ground.\n\n"
                "Then the closed question, which is about ONE specific mark and not about anything "
                "else you described. Both of the following sit on the same side of the contour, on "
                "the lower ground, so which side a mark is on does not distinguish them. What "
                "distinguishes them is whether the mark is attached:\n"
                "- A solid wedge TOUCHING the contour, thickest where it meets the line and "
                "tapering smoothly to nothing along it, its darkness strongest at the line and "
                "fading out. One per tile. That is this edge's expected shading. It is NOT what "
                "this question asks about, however prominent or spur-like its thick end looks.\n"
                "- A thin DETACHED line, roughly one pixel thick in the original and so a few "
                "pixels thick at this enlargement, of near-uniform darkness end to end, standing "
                "clear of the contour with un-darkened ground between it and the line, and running "
                "away from it at an angle. One per tile.\n\n"
                "Does the second kind appear -- a thin detached line standing off the contour with "
                "clear ground between? Judge this on the raw frames. The amplified difference "
                "frames may corroborate what you already see there, but cannot settle it on their "
                "own: they show every pixel the pass touched, expected shading included, so a mark "
                "that only appears in a difference frame does not count. "
                "Answer PASS if no such detached line is present, FAIL if one is."
            ),
            frames=(
                "run_pct10_raw",
                "run_pct10_diff",
                "run_pct50_raw",
                "run_pct50_diff",
                "run_pct100_raw",
                "run_pct100_diff",
            ),
        ),
        # Raw before diff: once a reviewer has seen the amplified mass, it would
        # be judging the raw frames from memory of it.
        rp.Check(
            id="corner_raw_shape",
            question=(
                "Look only at the two raw corner frames. First, in your own words, describe the "
                "dark mass in the middle of the frame, where the shading from the two raised tiles "
                "meets over the lower tile between them, and anything that runs into it.\n\n"
                "Then the closed question. Is that mass split through by a narrow vertical column "
                "of un-darkened ground, as bright as the open ground around it, running its full "
                "depth from top to bottom? Two things are NOT such a split: the stepped outline "
                "of the mass, where its depth changes in small steps every couple of columns; and "
                "the terrain texture's own light and dark mottling. "
                "Answer SPLIT if such a column is present, WHOLE if the mass is one continuous "
                "piece across its width."
            ),
            frames=("corner_pct50_raw", "corner_pct100_raw"),
            answers=("SPLIT", "WHOLE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="corner_shape",
            question=(
                "Now the amplified difference frames of the same corner, where only the pixels "
                "one shading pass touched appear, on black; the switched-off frames are context "
                "only. First, in your own words, describe the bright shape in the middle of the "
                "frame, where the two tiles' shading meets, and anything that joins it.\n\n"
                "Then the closed question, which is about that middle mass only. Is it split "
                "through by a narrow vertical column with no shading at all, plain black, running "
                "its full depth from top to bottom? Two things are NOT such a split:\n"
                "- The stepped outline of the mass, where its depth changes in small steps every "
                "couple of columns. That is its expected edge.\n"
                "- Any break in the thin lines leading away from the mass to either side. That is "
                "about the lines, not the mass, and does not count however clear it is.\n\n"
                "Answer SPLIT if such a column is present, WHOLE if the mass is one continuous "
                "piece across its width."
            ),
            frames=("corner_pct50_control", "corner_pct50_diff", "corner_pct100_control", "corner_pct100_diff"),
            answers=("SPLIT", "WHOLE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="flat_ground_marks",
            question=(
                "Are there any short dark marks, ticks or lines on this ground, other than the "
                "terrain texture's own mottling? "
                "MARKS-PRESENT if yes, NO-MARKS if the ground carries only its texture."
            ),
            frames=("flat_raw",),
            answers=("MARKS-PRESENT", "NO-MARKS", "CANNOT-TELL"),
        ),
    )

    return rp.PackSpec(
        id="shadow_band",
        title="Terrain shading review",
        frames=tuple(frames),
        checks=checks,
        preamble=(
            "These are offscreen renders from an isometric map editor. Each frame is a crop, "
            "enlarged by a whole-number factor with no smoothing, so a one-pixel feature in the "
            "original appears that many pixels wide here. Judge only what you can actually see."
        ),
        injects=tuple(INJECTS),
    )


# ------------------------------------------------------ ground_texture pack


GROUND_MAP = ROOT / "examples" / "8tp3w9j.aoe2scenario"
GRASS, FOREST = 0, 10  # GRASS_1, FOREST_OAK

# frame id -> (style, tile_px, centre tile, crop (w, h), terrain ids it may show).
# Regions measured off GROUND_MAP: 18x18 level grass at (122, 92) and an
# unrelated one at (158, 113), a 14x14 grass hillside (one level per tile at
# most) at (212, 145), a level forest edge at (15, 189), a sloping one at
# (39, 136). --check asserts every crop shows only its own terrain, and the
# level ones no elevation feature, so a map edit cannot quietly change a frame.
_GROUND_FRAMES = {
    "grass_angled": ("stepped", 64, (131, 101), (512, 256), (GRASS,)),
    "grass_angled_near": ("stepped", 128, (131, 101), (720, 360), (GRASS,)),
    "grass_hillside": ("sloped", 64, (219, 152), (384, 192), (GRASS,)),
    "forest_edge_angled": ("stepped", 64, (21, 195), (384, 192), (GRASS, FOREST)),
    "forest_edge_hillside": ("sloped", 64, (45, 142), (384, 192), (GRASS, FOREST)),
    "grass_overhead": ("flat", 64, (167, 122), (512, 256), (GRASS,)),
}
# Flat blits whole contiguous crops, so no per-cell join can break there.
_GROUND_SPECIFICITY = "grass_overhead"

# one-tile-offset's tiles, one inside every angled frame and off-centre;
# --check asserts each angled frame shows exactly one of them.
_OFFSET_TILES = frozenset({(129, 102), (217, 153), (23, 197), (47, 141)})

_GROUND_SCENE: dict[str, object] = {}


def _ground_scene():
    """(scenario, elevations), loaded once."""
    if not _GROUND_SCENE:
        scenario = load_map_and_units(str(GROUND_MAP))
        mm = scenario.map_manager
        elevations = np.zeros((mm.map_height, mm.map_width), dtype=np.int64)
        for tile in mm.terrain:
            elevations[tile.y, tile.x] = tile.elevation
        _GROUND_SCENE.update(scenario=scenario, elevations=elevations)
    return _GROUND_SCENE["scenario"], _GROUND_SCENE["elevations"]


def _ground_render(frame_id: str, scenario=None, elevations=None) -> np.ndarray:
    """One frame at 1:1, off-engine and settings-free: tile_px and the
    default elev_step are explicit, so neither a user's graphics quality nor
    their vertical scale moves a crop."""
    style, tile_px, (cx, cy), (cw, ch), _terrains = _GROUND_FRAMES[frame_id]
    if scenario is None:
        scenario, elevations = _ground_scene()
    if style == "flat":
        x0, y0 = (2 * cx + 1) * tile_px // 2 - cw // 2, (2 * cy + 1) * tile_px // 2 - ch // 2
        return render.composite_rect_flat(scenario, x0, y0, x0 + cw, y0 + ch, tile_px, with_units=False)
    mm = scenario.map_manager
    proj = ig.canvas_size_and_origin(
        mm.map_width, mm.map_height, tile_px, ig.MIN_ELEVATION, ig.MAX_ELEVATION,
        elev_step_pct=ig.ELEV_STEP_DEFAULT_PCT, corner_headroom_steps=1 if style == "sloped" else 0,
    )
    sx, sy = ig.tile_screen_origin(cx, cy, int(elevations[cy, cx]), proj)
    x0, y0 = sx + proj.half_w - cw // 2, sy + proj.half_h - ch // 2
    if style == "sloped":
        rise = ig.corner_rise_px(elevations, proj, rule=render.SLOPE_CORNER_RULE)
        return render.composite_rect_sloped(
            scenario, x0, y0, x0 + cw, y0 + ch, rise, proj, tile_px, {}, {}, with_units=False
        )
    return render.composite_rect_iso(
        scenario, x0, y0, x0 + cw, y0 + ch, elevations, proj, tile_px, {}, {}, with_units=False
    )


def _legacy_inverse_sample(dst_x, dst_y, half_w: int, half_h: int, tile_px: int):
    """The pre-TASK-201 basis, verbatim: the centre quarter of each crop,
    its axes turned 45 degrees to _crop_offset's progression."""
    u = (dst_x + 0.5 - half_w) / half_w
    v = (dst_y + 0.5 - half_h) / half_h
    a = (u + v) / 2.0
    b = (v - u) / 2.0
    src_x = np.clip(np.floor((a + 1.0) / 2.0 * tile_px), 0, tile_px - 1).astype(np.int64)
    src_y = np.clip(np.floor((b + 1.0) / 2.0 * tile_px), 0, tile_px - 1).astype(np.int64)
    return src_y, src_x


_REAL_CROP_OFFSET = render._crop_offset
_REAL_TEXTURE = asset_source.get_terrain_texture_array


def _offset_crop(x: int, y: int, texture_size: int, tile_px: int) -> tuple[int, int]:
    if (x, y) in _OFFSET_TILES:
        return _REAL_CROP_OFFSET(x + 1, y, texture_size, tile_px)
    return _REAL_CROP_OFFSET(x, y, texture_size, tile_px)


_TEXTURE_VARIANTS: dict[tuple, np.ndarray] = {}


def _texture_variant(kind: str, tile_px: int):
    """A get_terrain_texture_array stand-in serving one derived texture."""

    def get(terrain_id: int):
        path = asset_source.get_terrain_texture_path(terrain_id)
        if path is None:
            return None
        key = (kind, path, tile_px)
        if key not in _TEXTURE_VARIANTS:
            _TEXTURE_VARIANTS[key] = _derive_texture(kind, path, terrain_id, tile_px)
        return _TEXTURE_VARIANTS[key]

    return get


def _derive_texture(kind: str, path: Path, terrain_id: int, tile_px: int) -> np.ndarray:
    from PIL import Image

    if kind == "unfiltered":
        # The same size the loader makes, point-sampled with no prefilter.
        size = asset_source.LOADED_TEXTURE_SIZE
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB").resize((size, size), Image.Resampling.NEAREST))
    else:
        # Every crop block whose (bx + by) is odd, turned 90 degrees in place:
        # a checkerboard of turned cells at this tile_px.
        arr = np.array(_REAL_TEXTURE(terrain_id))
        n = arr.shape[0] // tile_px
        for by in range(n):
            for bx in range(n):
                if (bx + by) % 2:
                    ys, xs = slice(by * tile_px, (by + 1) * tile_px), slice(bx * tile_px, (bx + 1) * tile_px)
                    arr[ys, xs] = np.rot90(arr[ys, xs])
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True)
class _GroundInject:
    """`patches(style, tile_px)` lists (object, attribute, replacement) for
    one frame's render; `bound` is set only on the localized inject."""

    patches: object
    bound: object = None


def _patch_old_basis(_style, _tile_px):
    """COARSE. The pre-TASK-201 render: every cell's crop read in the old
    basis, so no cell's pattern continues into its neighbour's."""
    return [(ig, "_inverse_sample", _legacy_inverse_sample)]


def _patch_one_tile_offset(_style, _tile_px):
    """LOCALIZED. One cell per angled frame takes its right-hand
    neighbour's crop: the smallest possible break, one cell's worth."""
    return [(render, "_crop_offset", _offset_crop)]


def _patch_unfiltered(style, tile_px):
    """COARSE, angled frames only. Point-sampled source with no prefilter,
    on top of the angled views' ~2.8:1 vertical minification."""
    return [] if style == "flat" else [(asset_source, "get_terrain_texture_array", _texture_variant("unfiltered", tile_px))]


def _patch_turned(style, tile_px):
    """COARSE, angled frames only. Every other cell's crop turned 90
    degrees, in a checkerboard."""
    return [] if style == "flat" else [(asset_source, "get_terrain_texture_array", _texture_variant("turned", tile_px))]


GROUND_INJECTS = {
    "old-basis": _GroundInject(_patch_old_basis),
    "one-tile-offset": _GroundInject(_patch_one_tile_offset, bound="one-cell"),
    "unfiltered": _GroundInject(_patch_unfiltered),
    "turned-cells": _GroundInject(_patch_turned),
}
GROUND_COARSE_INJECT = "old-basis"


def _clear_texture_caches() -> None:
    """Every memo holding index tables derived from _inverse_sample."""
    for fn in (
        ig.diamond_indices,
        ig.skirt_quad_indices,
        ig.sloped_quad_indices,
        native_composite.stepped_tables,
        native_composite.sloped_shapes,
    ):
        fn.cache_clear()
    _clear_caches()


def _ground_frame_images(inject: str | None) -> dict[str, np.ndarray]:
    images = {}
    for frame_id, (style, tile_px, *_rest) in _GROUND_FRAMES.items():
        patches = GROUND_INJECTS[inject].patches(style, tile_px) if inject else []
        saved = [(obj, name, getattr(obj, name)) for obj, name, _new in patches]
        for obj, name, new in patches:
            setattr(obj, name, new)
        _clear_texture_caches()
        try:
            images[frame_id] = _ground_render(frame_id)
        finally:
            for obj, name, old in saved:
                setattr(obj, name, old)
            _clear_texture_caches()
    return images


def _ground_texture_spec() -> rp.PackSpec:
    """TASK-201's iso texture continuity, on a real map with real textures.
    Each check's options are written against the injects' measured masks:
    a break is one cell's border (one-tile-offset is exactly one component
    per angled frame), a turned cell is a 90-degree turn of the crop, and
    speckle is the isolated-pixel rate --check measures, ~0.5% on clean
    angled grass against ~10% unfiltered."""
    angled = ("grass_angled", "grass_angled_near", "grass_hillside", "forest_edge_angled", "forest_edge_hillside")
    captions = {
        "grass_angled": "Grassland on level ground, angled view, normal zoom.",
        "grass_angled_near": "The same grassland, angled view, zoomed in.",
        "grass_hillside": "Grassland on a hillside, angled view; the ground slopes across the frame.",
        "forest_edge_angled": "Where grassland meets forest floor, level ground, angled view.",
        "forest_edge_hillside": "Where grassland meets forest floor on a hillside, angled view.",
        "grass_overhead": (
            "Grassland seen straight from above, same ground texture and same zoom as the first angled "
            "grassland frame. Here each map cell is a square."
        ),
    }
    frames = tuple(rp.Frame(fid, f"{fid}.png", "raw", caption=captions[fid]) for fid in _GROUND_FRAMES)
    breaks_rule = (
        "- A BREAK is a straight edge, at least one cell long, where the pattern on one side does not "
        "continue on the other: the blades, specks and blotches stop at the line and an unrelated stretch "
        "of the same kind of ground starts. One such edge anywhere counts, even if every other cell joins "
        "up, and a single cell whose pattern does not match any of its neighbours counts.\n"
    )
    checks = (
        rp.Check(
            id="pattern_joins",
            question=(
                "In the angled frames the ground is a grid of diamond-shaped map cells seen from an angle, each "
                "painted with a patch of a ground texture. No grid lines are drawn. First, in your own words, "
                "describe how the ground's pattern behaves across each frame: does it read as one continuous "
                "surface, or can you make out individual cells, and if so, by what (lines, mismatched pattern, "
                "brightness)?\n\n"
                "Then the closed question, which is about whether the PATTERN breaks at cell borders within one "
                "kind of ground (grass, or forest floor), not about brightness.\n"
                + breaks_rule
                + "- NOT a break: (1) the border between two different kinds of ground, where grassland meets "
                "forest floor; that line is expected however sharp it is. (2) On the hillside frames, a change "
                "in brightness, or a stretch or squash of the pattern, where the slope changes, while the "
                "pattern itself carries on across the line. (3) The texture's own blotches and streaks.\n\n"
                "Answer BREAKS-PRESENT if at least one break is visible, NO-BREAKS if the pattern carries on "
                "across every cell border within each kind of ground."
            ),
            frames=angled,
            answers=("BREAKS-PRESENT", "NO-BREAKS", "CANNOT-TELL"),
        ),
        rp.Check(
            id="pattern_direction",
            question=(
                "Same angled frames. First, in your own words, describe any direction the ground pattern has: "
                "elongated blades, streaks or blotches, which way they lean, and whether that lean differs "
                "from one cell to the next.\n\n"
                "Then the closed question, which is about DIRECTION only. Do neighbouring diamond-shaped cells "
                "show the same kind of ground with its features turned to visibly different angles from each "
                "other, so the ground reads as a patchwork of separately turned pieces? Three things are NOT "
                "that: (1) a cell whose pattern merely fails to continue into its neighbour while its features "
                "lean the same way; (2) on the hillside frames, a stretch or squash of the pattern where the "
                "slope changes; (3) the border between grassland and forest floor.\n\n"
                "Answer TURNED if neighbouring cells show the pattern turned to different angles, ALIGNED if "
                "within each kind of ground the pattern leans the same way all across the frame."
            ),
            frames=angled,
            answers=("TURNED", "ALIGNED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="fine_grain",
            question=(
                "Compare the three angled grassland frames with the overhead grassland frame, which shows the "
                "same grass texture at the same zoom seen straight from above. First, in your own words, "
                "describe the fine-scale look of the angled grass against the overhead: grain size, sharpness, "
                "and any isolated bright or dark single pixels, glitter, jagged one-pixel streaks or rippling "
                "bands.\n\n"
                "Then the closed question. Does the angled grass carry pixel-level noise the overhead does not: "
                "isolated single bright and dark pixels or one-pixel jagged streaks sprinkled densely all over "
                "the grass, so it reads as glittery or salt-and-pepper rather than as a grass texture? An "
                "occasional stray pixel here and there does NOT count, and neither does a finer or more "
                "compressed grain, which is expected because the angled view squeezes the ground vertically.\n\n"
                "Answer SPECKLED if such dense pixel noise is present, SMOOTH if the angled grass reads as "
                "cleanly as the overhead apart from that expected squeeze."
            ),
            frames=("grass_angled", "grass_angled_near", "grass_hillside", "grass_overhead"),
            answers=("SPECKLED", "SMOOTH", "CANNOT-TELL"),
        ),
        rp.Check(
            id="overhead_joins",
            question=(
                "Look only at the overhead grassland frame, where each map cell is a square. No grid lines are "
                "drawn. First, in your own words, describe how the ground's pattern behaves across the frame.\n\n"
                "Then the closed question.\n"
                + breaks_rule
                + "- NOT a break: the texture's own blotches and streaks.\n\n"
                "Answer BREAKS-PRESENT if at least one break is visible, NO-BREAKS if the pattern carries on "
                "across the whole frame."
            ),
            frames=(_GROUND_SPECIFICITY,),
            answers=("BREAKS-PRESENT", "NO-BREAKS", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id="ground_texture",
        title="Ground texture review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders of bare ground (no units or objects) from a map editor for a strategy "
            "game. Each frame is a crop, enlarged by a whole-number factor with no smoothing, so a one-pixel "
            "feature in the original appears that many pixels wide here. Judge only what you can actually see."
        ),
        injects=tuple(GROUND_INJECTS),
    )


# Isolated-pixel speckle, as fine_grain's options are written: a pixel more
# than SPECKLE_DELTA luma from its 3x3 median. Clean angled grass ~0.5%,
# overhead ~0.06%, the unfiltered inject ~8-11% (measured 2026-10-07).
SPECKLE_DELTA = 30
CLEAN_SPECKLE_MAX = 0.015
UNFILTERED_SPECKLE_MIN = 0.05


def _speckle_rate(img: np.ndarray) -> float:
    from numpy.lib.stride_tricks import sliding_window_view

    luma = img.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    med = np.median(sliding_window_view(luma, (3, 3)), axis=(-2, -1))
    return float((np.abs(luma[1:-1, 1:-1] - med) > SPECKLE_DELTA).mean())


def _ground_prerequisites() -> str | None:
    """Why the ground_texture pack cannot render here, or None."""
    if not GROUND_MAP.is_file():
        return f"needs {GROUND_MAP.relative_to(ROOT)} (the gitignored examples/ corpus)"
    if asset_source.get_install_path() is None:
        return "needs a configured AoE2:DE install for the terrain textures"
    return None


def _ground_purity_failures() -> list[str]:
    """Each crop shows only its own terrain, and the level ones no elevation
    feature: re-render with every other terrain swapped (and, for Stepped,
    the whole map flattened to the frame's level) and require no change."""
    failures = []
    for frame_id, (style, _tile_px, (cx, cy), _size, terrains) in _GROUND_FRAMES.items():
        base_scn, base_elev = _ground_scene()
        clean = _ground_render(frame_id)
        scn = load_map_and_units(str(GROUND_MAP))
        for tile in scn.map_manager.terrain:
            if tile.terrain_id not in terrains:
                tile.terrain_id = 6 if tile.terrain_id != 6 else 14
        if not np.array_equal(_ground_render(frame_id, scn, base_elev), clean):
            failures.append(f"frame {frame_id}: a terrain other than {terrains} reaches the crop")
        if style == "stepped":
            level = np.full_like(base_elev, base_elev[cy, cx])
            if not np.array_equal(_ground_render(frame_id, base_scn, level), clean):
                failures.append(f"frame {frame_id}: an elevation feature reaches the crop")
    return failures


def _check_ground_texture() -> None:
    """--check for ground_texture: spec coupling, frame band, crop purity,
    each inject's reach (printed), the specificity frame untouched by every
    inject, one-tile-offset exactly one bounded cell per angled frame and
    under old-basis' largest delta, and the speckle rates fine_grain's
    options rest on."""
    reason = _ground_prerequisites()
    if reason:
        raise SystemExit(f"ground_texture: {reason}")
    failures: list[str] = []
    spec = _ground_texture_spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    failures += _ground_purity_failures()

    clean = _ground_frame_images(None)
    for frame in spec.frames:
        try:
            scaled, _factor = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))

    angled = [fid for fid in _GROUND_FRAMES if fid != _GROUND_SPECIFICITY]
    largest_cell = max(ig.diamond_indices(tp)[0].size for _s, tp, *_r in _GROUND_FRAMES.values())
    # A sloped cell stretches past its diamond (measured 1.25x); 1.5x of the largest is still one cell.
    one_cell = rp.LocalizedBound(max_changed_fraction=0.02, max_component_px=largest_cell * 3 // 2)
    reach: dict[str, dict[str, tuple[int, int]]] = {}
    injected_by_name = {}
    for name, inject in GROUND_INJECTS.items():
        injected = injected_by_name[name] = _ground_frame_images(name)
        touched = {}
        for frame_id in _GROUND_FRAMES:
            mask = rp.changed_pixels(clean[frame_id], injected[frame_id])
            changed = int(np.count_nonzero(mask))
            if changed:
                touched[frame_id] = (changed, rp.largest_component_px(mask))
            if inject.bound and frame_id in angled:
                failures += one_cell.failures(mask, f"inject {name} on {frame_id}")
                if changed and rp.components(mask) != 1:
                    failures.append(f"inject {name} on {frame_id}: {rp.components(mask)} pieces, not one cell")
        if _GROUND_SPECIFICITY in touched:
            failures.append(f"inject {name}: reached the specificity frame {_GROUND_SPECIFICITY}")
        missed = [fid for fid in angled if fid not in touched]
        if missed:
            failures.append(f"inject {name}: changed nothing in {', '.join(missed)}")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")

    coarse = reach[GROUND_COARSE_INJECT]
    for name, inject in GROUND_INJECTS.items():
        if inject.bound is None:
            continue
        for frame_id, (_px, comp) in sorted(reach[name].items()):
            if frame_id in coarse and comp >= coarse[frame_id][1]:
                failures.append(
                    f"inject {name} on {frame_id}: largest delta {comp} px is not under the coarse "
                    f"inject's {coarse[frame_id][1]} px there, so the two no longer differ in size"
                )

    rates = []
    for frame_id in _check_frames(spec, "fine_grain"):
        clean_rate = _speckle_rate(clean[frame_id])
        noisy_rate = _speckle_rate(injected_by_name["unfiltered"][frame_id])
        rates.append(f"{frame_id} {clean_rate:.2%}/{noisy_rate:.2%}")
        if frame_id == _GROUND_SPECIFICITY:
            continue
        if clean_rate > CLEAN_SPECKLE_MAX:
            failures.append(f"{frame_id}: clean speckle {clean_rate:.2%} over {CLEAN_SPECKLE_MAX:.1%}")
        if noisy_rate < UNFILTERED_SPECKLE_MIN:
            failures.append(f"{frame_id}: unfiltered speckle {noisy_rate:.2%} under {UNFILTERED_SPECKLE_MIN:.0%}")
    print(f"speckle (clean/unfiltered): {', '.join(rates)}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(
        f"OK: ground_texture frames couple to its checks, land in {rp.MIN_LONG_EDGE}-{rp.MAX_LONG_EDGE} px and "
        f"show only their own terrain; all {len(GROUND_INJECTS)} injects bite every angled frame and none "
        f"reaches {_GROUND_SPECIFICITY}; one-tile-offset is one bounded cell per angled frame"
    )


def _check_frames(spec: rp.PackSpec, check_id: str) -> tuple[str, ...]:
    return next(c.frames for c in spec.checks if c.id == check_id)


# --------------------------------------------------------- sloped_mip pack


SLOPED_MAPS = {"ground": GROUND_MAP, "farms": ROOT / "examples" / "old-allies-final-v2.aoe2scenario"}
# Small chunks so every frame crosses several chunk edges; --check asserts the
# clean frames are byte-identical to the app's DEFAULT_CHUNK_PX.
SLOPED_CHUNK_PX = 64
# One ground footprint per frame: this many level-0 pixels, so -1 and -2 frame the same tiles.
_SLOPED_CROP_L0 = (768, 384)
# frame id -> (map, mip, centre tile, units and sprites on). hill and forest are
# ground_texture's two hillsides; farms is a plateau one level up with farms on
# its edge, buildings a plateau with a player's buildings on it, level_ground
# flat elevation 0 with a farm, a mill and buildings (the specificity frame).
_SLOPED_FRAMES = {
    "hill_a": ("ground", -1, (219, 152), False),
    "hill_b": ("ground", -2, (219, 152), False),
    "forest_a": ("ground", -1, (45, 142), True),
    "forest_b": ("ground", -2, (45, 142), True),
    "farms_a": ("farms", -1, (89, 215), True),
    "farms_b": ("farms", -2, (89, 215), True),
    "buildings_a": ("ground", -1, (22, 214), True),
    "buildings_b": ("ground", -2, (22, 214), True),
    "level_ground": ("farms", -2, (43, 133), True),
}
_SLOPED_SPECIFICITY = "level_ground"
# The localized inject shifts one chunk in each of these frames.
_SLOPED_ROW_SHIFT_FRAMES = ("hill_a", "hill_b", "forest_a", "forest_b")

_SLOPED_SCENE: dict[tuple[str, bool], object] = {}


class _PinnedSettings:
    """Pins the two settings a full Sloped render depends on (vertical scale,
    graphics quality) through settings' in-process cache, never set_*()."""

    def __enter__(self):
        from descape import settings

        self.saved = (settings._elev_step_pct, settings._graphics_quality)
        settings._elev_step_pct = ig.ELEV_STEP_DEFAULT_PCT
        settings._graphics_quality = 3
        return self

    def __exit__(self, *exc):
        from descape import settings

        settings._elev_step_pct, settings._graphics_quality = self.saved


def _sloped_cache(map_key: str, units: bool, chunk_px: int = SLOPED_CHUNK_PX):
    """A fresh SlopedChunkCache over its own scenario load, so an inject's
    patched builds never leak into another run. Scenario loads are memoized."""
    from descape.render_cache import SlopedChunkCache

    key = (map_key, units)
    if key not in _SLOPED_SCENE:
        _SLOPED_SCENE[key] = load_map_and_units(str(SLOPED_MAPS[map_key]))
    scenario = _SLOPED_SCENE[key]
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scenario)
    return SlopedChunkCache(
        scenario, elevations, corner_rise, proj, proj.tile_px, chunk_px=chunk_px, with_units=units, sprites=units
    )


def _sloped_box(cache, mip: int, centre) -> tuple[int, int, int, int]:
    """The level-`mip` pixel crop centred on a tile's own surface."""
    lvl = cache._levels[mip]
    cx, cy = centre
    sx, sy = ig.tile_screen_origin(cx, cy, int(cache.elevations[cy, cx]), lvl.proj)
    cw, ch = (n >> -mip for n in _SLOPED_CROP_L0)
    x0, y0 = sx + lvl.proj.half_w - cw // 2, sy + lvl.proj.half_h - ch // 2
    return x0, y0, x0 + cw, y0 + ch


def _shift_chunk(cache, mip: int, box) -> tuple[int, int]:
    """The level-`mip` chunk origin chunk-row-shift moves in this crop: of the
    chunks whose left and right edges both fall inside the crop, the one where
    the shift changes the most pixels inside it (the first in row order on a
    tie). Measured on the clean cache through its own _composite_rect, which
    the shift leaves untouched."""
    x0, y0, x1, y1 = box
    c = cache.chunk_px
    best = None
    for cy in range(y0 // c, (y1 - 1) // c + 1):
        for cx in range(-(-x0 // c), (x1 - 1) // c):
            if not (x0 < cx * c and (cx + 1) * c < x1):
                continue
            rect = cache._chunk_rect(mip, cx, cy)
            clean = cache._composite_rect(mip, *rect)
            probe = _SlopedInject("chunk-row-shift")
            probe.shift_chunks.add((id(cache.scenario), cache._levels[mip].tile_px, rect[0], rect[1]))
            with probe:
                shifted = cache._composite_rect(mip, *rect)
            ix0, iy0 = max(x0, rect[0]) - rect[0], max(y0, rect[1]) - rect[1]
            ix1, iy1 = min(x1, rect[2]) - rect[0], min(y1, rect[3]) - rect[1]
            changed = int(np.count_nonzero(rp.changed_pixels(clean[iy0:iy1, ix0:ix1], shifted[iy0:iy1, ix0:ix1])))
            if best is None or changed > best[0]:
                best = (changed, rect[0], rect[1])
    return best[1], best[2]


class _SlopedInject:
    """Patches for one known ladder defect, applied around the cache builds and
    renders of a whole run. `bound` is set only on the localized inject."""

    def __init__(self, name: str, bound=None) -> None:
        self.name = name
        self.bound = bound
        self.shift_chunks: set[tuple[int, int, int, int]] = set()  # (scenario id, tile_px, x0, y0)

    def __enter__(self):
        from descape.render_cache import SlopedChunkCache

        self.saved = [(SlopedChunkCache, "_level_rise"), (render, "composite_rect_sloped")]
        self.saved = [(obj, name, getattr(obj, name)) for obj, name in self.saved]
        real_rise, real_composite = self.saved[0][2], self.saved[1][2]
        if self.name == "level0-rise":
            true_rise: dict[tuple, np.ndarray] = {}

            def level_rise(cache, lvl):
                lvl = real_rise(cache, lvl)
                if lvl.tile_px * 4 == cache.tile_px:
                    key = (id(cache.scenario), lvl.tile_px)
                    if key not in true_rise:
                        true_rise[key] = ig.corner_rise_px(cache.elevations, lvl.proj, rule=render.SLOPE_CORNER_RULE)
                    # The coarsest level's unit layers read level 0's corner_rise as is, never recomputed.
                    wrong = cache.corner_rise
                    lvl.headroom = render._unit_rise_headroom_px(wrong, cache.elevations, lvl.proj)
                    lvl.corner_rise = wrong
                return lvl

            def composite(scenario, x0, y0, x1, y1, corner_rise, proj, *args, **kwargs):
                # Its terrain keeps the level's own corner_rise, so only the objects move.
                corner_rise = true_rise.get((id(scenario), proj.tile_px), corner_rise)
                return real_composite(scenario, x0, y0, x1, y1, corner_rise, proj, *args, **kwargs)

            SlopedChunkCache._level_rise = level_rise
            render.composite_rect_sloped = composite
        elif self.name == "chunk-row-shift":

            def composite(scenario, x0, y0, x1, y1, corner_rise, proj, *args, **kwargs):
                if (id(scenario), proj.tile_px, x0, y0) in self.shift_chunks:
                    # One chunk's terrain reads every corner's height from the row above.
                    corner_rise = np.roll(corner_rise, 1, axis=0)
                return real_composite(scenario, x0, y0, x1, y1, corner_rise, proj, *args, **kwargs)

            render.composite_rect_sloped = composite
        return self

    def __exit__(self, *exc):
        for obj, name, old in self.saved:
            setattr(obj, name, old)


SLOPED_INJECTS = {"level0-rise": None, "chunk-row-shift": "one-chunk"}
SLOPED_COARSE_INJECT = "level0-rise"


def _sloped_frame_images(inject: str | None, chunk_px: int = SLOPED_CHUNK_PX) -> dict[str, np.ndarray]:
    """Every sloped_mip frame at 1:1, each stitched from a fresh cache's chunks
    through SlopedChunkCache.render_rect(mip=...), never a direct render."""
    images: dict[str, np.ndarray] = {}
    with _PinnedSettings():
        caches = {}
        for _map, _mip, _centre, units in _SLOPED_FRAMES.values():
            if (_map, units) not in caches:
                caches[(_map, units)] = _sloped_cache(_map, units, chunk_px)
        patch = _SlopedInject(inject, SLOPED_INJECTS[inject]) if inject else None
        if inject == "chunk-row-shift":
            # Measured on the clean caches; level0-rise must find its levels unbuilt instead.
            for frame_id in _SLOPED_ROW_SHIFT_FRAMES:
                map_key, mip, centre, units = _SLOPED_FRAMES[frame_id]
                cache = caches[(map_key, units)]
                x0, y0 = _shift_chunk(cache, mip, _sloped_box(cache, mip, centre))
                patch.shift_chunks.add((id(cache.scenario), cache._levels[mip].tile_px, x0, y0))
        if patch is not None:
            patch.__enter__()
        try:
            for frame_id, (map_key, mip, centre, units) in _SLOPED_FRAMES.items():
                cache = caches[(map_key, units)]
                images[frame_id] = cache.render_rect(*_sloped_box(cache, mip, centre), mip=mip)
        finally:
            if patch is not None:
                patch.__exit__(None, None, None)
    return images


def _sloped_mip_spec() -> rp.PackSpec:
    """TASK-031.60's Sloped mip ladder: the coarser levels' stitched chunks on
    hillsides, under trees, buildings and farm fields on slopes, and on level
    ground. The `_a` frames are one level coarser than full detail, the `_b`
    frames two; the reviewer is told only that both are zoomed out."""
    captions = {
        "hill_a": "Grassland on a hillside, zoomed out.",
        "hill_b": "The same hillside, zoomed further out.",
        "forest_a": "Forest and grassland on a hillside, with trees standing on it, zoomed out.",
        "forest_b": "The same forest hillside, zoomed further out.",
        "farms_a": "Farm fields and buildings at the edge of a raised plateau, zoomed out.",
        "farms_b": "The same farm fields, zoomed further out.",
        "buildings_a": "A group of buildings on and around a raised plateau, zoomed out.",
        "buildings_b": "The same buildings, zoomed further out.",
        "level_ground": "Buildings, a farm field and other objects on perfectly level ground, zoomed far out.",
    }
    frames = tuple(rp.Frame(fid, f"{fid}.png", "raw", caption=captions[fid]) for fid in _SLOPED_FRAMES)
    checks = (
        rp.Check(
            id="ground_continuity",
            question=(
                "These four frames show bare and forested hillsides. The ground is a grid of diamond-shaped "
                "map cells at different heights, drawn as one continuous sloping surface. First, in your own "
                "words, describe how the ground runs across each frame: its slopes, its texture, and any place "
                "where it looks interrupted.\n\n"
                "Then the closed question, which is about the GROUND only. Is there a straight horizontal or "
                "vertical line, running across the ground, where the ground on one side does not continue on "
                "the other: its texture, shading or slope jumps, is offset up or down, or is cut off, or a "
                "strip of black shows through? NOT such a line: (1) the border between grassland and forest "
                "floor, which follows the diagonal cell grid and wanders; (2) a change of shading where the "
                "slope bends, which also runs along the diagonal grid; (3) trees and anything else standing on "
                "the ground, however they are placed; (4) the frame's own edges.\n\n"
                "Answer BREAK if at least one such straight line is visible, CONTINUOUS if the ground carries "
                "on across every frame."
            ),
            frames=("hill_a", "hill_b", "forest_a", "forest_b"),
            answers=("BREAK", "CONTINUOUS", "CANNOT-TELL"),
        ),
        rp.Check(
            id="objects_on_ground",
            question=(
                "These frames show buildings, trees and other objects standing on sloping ground. First, in "
                "your own words, describe how each kind of object meets the ground under it: where its base "
                "is, and what ground shows around that base.\n\n"
                "Then the closed question. Does any object look detached from the ground it stands on: "
                "floating clear above it with ground or a gap showing under its base, or sitting well away "
                "from the patch of ground it plainly belongs to, so it reads as pasted onto the wrong place? "
                "NOT detached: (1) a tall object reaching up over the ground and objects behind it, which every "
                "tall object does; (2) dark shadows beside objects; (3) objects too small to tell where their "
                "base is.\n\n"
                "Answer DETACHED if at least one object looks detached, SEATED if every judgeable object sits on "
                "its ground."
            ),
            frames=("farms_a", "farms_b", "buildings_a", "buildings_b"),
            answers=("DETACHED", "SEATED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="fields_on_slope",
            question=(
                "Look at the farm fields: flat, ploughed or planted rectangular-looking patches laid on the "
                "ground, some of them on the slope at the plateau's edge. First, in your own words, describe "
                "each field: its outline, how it bends with the ground, and what its edges meet.\n\n"
                "Then the closed question. Does every field lie on its own patch of ground, following the "
                "ground's slope under it, with its edges meeting the surrounding ground? Or does any field sit "
                "off its ground: raised above it or sunk into it, shifted across neighbouring cells, or bent "
                "in a way the ground around it is not? NOT off the ground: buildings or other objects standing "
                "on or in front of a field and hiding part of it.\n\n"
                "Answer ON-GROUND if every field lies on its ground, OFF-GROUND if any field does not."
            ),
            frames=("farms_a", "farms_b"),
            answers=("ON-GROUND", "OFF-GROUND", "CANNOT-TELL"),
        ),
        rp.Check(
            id="level_ground",
            question=(
                "This frame shows perfectly level ground with buildings, a farm field and other objects on it. "
                "Is anything wrong with how it is drawn: any object or field detached from the ground under "
                "it, or any straight horizontal or vertical line where the ground on one side is offset, cut "
                "off or does not continue on the other? The border between two kinds of ground, which follows "
                "the diagonal cell grid, does not count.\n\n"
                "Answer FLAWS-PRESENT if you see any of these, NO-FLAWS if the frame shows none."
            ),
            frames=(_SLOPED_SPECIFICITY,),
            answers=("FLAWS-PRESENT", "NO-FLAWS", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id="sloped_mip",
        title="Zoomed-out terrain review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from a map editor for an isometric strategy game, zoomed out, so a map "
            "cell is only a few pixels across before enlargement. Each frame is a crop, enlarged by a "
            "whole-number factor with no smoothing, so a one-pixel feature in the original appears that many "
            "pixels wide here. Judge only what you can actually see."
        ),
        injects=tuple(SLOPED_INJECTS),
    )


def _sloped_prerequisites() -> str | None:
    """Why the sloped_mip pack cannot render here, or None."""
    for path in SLOPED_MAPS.values():
        if not path.is_file():
            return f"needs {path.relative_to(ROOT)} (the gitignored examples/ corpus)"
    if asset_source.get_install_path() is None:
        return "needs a configured AoE2:DE install for the terrain textures and sprites"
    return None


def _check_sloped_mip() -> None:
    """--check for sloped_mip: spec coupling, frame band, the clean frames
    byte-identical at the app's chunk size, each inject's reach (printed), the
    specificity frame untouched by both, level0-rise reaching only the
    coarsest level's object frames, and chunk-row-shift one chunk-bounded
    piece in each of its frames, under level0-rise's largest delta."""
    from descape.render_cache import DEFAULT_CHUNK_PX

    reason = _sloped_prerequisites()
    if reason:
        raise SystemExit(f"sloped_mip: {reason}")
    failures: list[str] = []
    spec = _sloped_mip_spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")

    clean = _sloped_frame_images(None)
    for frame in spec.frames:
        try:
            scaled, _factor = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))
    app_chunks = _sloped_frame_images(None, DEFAULT_CHUNK_PX)
    for frame_id, img in clean.items():
        if not np.array_equal(img, app_chunks[frame_id]):
            failures.append(f"frame {frame_id}: differs from the same crop stitched at chunk_px {DEFAULT_CHUNK_PX}")
    # Real art, not marks: every object frame's level holds sprites, the farm frames farm drapes too.
    with _PinnedSettings():
        for frame_id, (map_key, mip, _centre, units) in _SLOPED_FRAMES.items():
            if not units:
                continue
            lvl = _sloped_cache(map_key, units)._level(mip)
            if not lvl.sprites or not lvl.sprites.by_anchor:
                failures.append(f"frame {frame_id}: level {mip} built no sprites")
            elif frame_id.startswith("farms") and not lvl.sprites.farm_by_tile:
                failures.append(f"frame {frame_id}: level {mip} built no farm drape")

    one_chunk = rp.LocalizedBound(max_changed_fraction=0.5, max_component_px=SLOPED_CHUNK_PX**2)
    reach: dict[str, dict[str, tuple[int, int]]] = {}
    for name, bound in SLOPED_INJECTS.items():
        injected = _sloped_frame_images(name)
        touched = {}
        for frame_id in _SLOPED_FRAMES:
            mask = rp.changed_pixels(clean[frame_id], injected[frame_id])
            changed = int(np.count_nonzero(mask))
            if changed:
                touched[frame_id] = (changed, rp.largest_component_px(mask))
            if bound and frame_id in _SLOPED_ROW_SHIFT_FRAMES:
                failures += one_chunk.failures(mask, f"inject {name} on {frame_id}")
        if _SLOPED_SPECIFICITY in touched:
            failures.append(f"inject {name}: reached the specificity frame {_SLOPED_SPECIFICITY}")
        if name == SLOPED_COARSE_INJECT:
            want = {fid for fid, (_m, mip, _c, units) in _SLOPED_FRAMES.items() if mip == -2 and units}
            want.discard(_SLOPED_SPECIFICITY)
        else:
            want = set(_SLOPED_ROW_SHIFT_FRAMES)
        if missed := sorted(want - set(touched)):
            failures.append(f"inject {name}: changed nothing in {', '.join(missed)}")
        if stray := sorted(set(touched) - want):
            failures.append(f"inject {name}: reached {', '.join(stray)}, outside its frames")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")

    coarse = reach[SLOPED_COARSE_INJECT]
    for name, bound in SLOPED_INJECTS.items():
        if bound is None:
            continue
        for frame_id, (_px, comp) in sorted(reach[name].items()):
            if frame_id in coarse and comp >= coarse[frame_id][1]:
                failures.append(
                    f"inject {name} on {frame_id}: largest delta {comp} px is not under the coarse "
                    f"inject's {coarse[frame_id][1]} px there, so the two no longer differ in size"
                )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(
        f"OK: sloped_mip frames couple to its checks, land in {rp.MIN_LONG_EDGE}-{rp.MAX_LONG_EDGE} px and match "
        f"chunk_px {DEFAULT_CHUNK_PX}; {SLOPED_COARSE_INJECT} bites every coarsest-level object frame, "
        f"chunk-row-shift one bounded chunk per hillside frame, and neither reaches {_SLOPED_SPECIFICITY}"
    )


# ------------------------------------------------------ ground_borders pack


BORDER_MAPS = {
    "river": ROOT / "examples" / "F7_2_Dos Pilas (648).aoe2scenario",
    "island": ROOT / "examples" / "C2_ElCid_coop_6_v0_13.aoe2scenario",
}
BORDER_TILE_PX = 64
# frame id -> (map, style, centre tile, crop px, blending on). river: grass, sand, water, shallows and
# broken road meeting at elevation 0 with no layered tile. island: C2_ElCid_coop_6's island shore,
# elevation 1 throughout, 53% of its tiles layered. forest_floor: oak forest floor no blend reaches.
_BORDER_FRAMES = {
    "river": ("river", "stepped", (54, 192), (640, 400), True),
    "river_plain": ("river", "stepped", (54, 192), (640, 400), False),
    "island_angled": ("island", "stepped", (125, 91), (720, 400), True),
    "island_overhead": ("island", "flat", (125, 91), (704, 448), True),
    "island_shore": ("island", "stepped", (125, 91), (720, 400), True),
    "island_shore_overhead": ("island", "flat", (125, 91), (704, 448), True),
    "forest_floor": ("river", "stepped", (405, 123), (384, 240), True),
}
# Rendered with every tile's layer cleared, so their edges are neighbour transitions alone.
_BORDER_NO_LAYERS = ("island_shore", "island_shore_overhead")
_BORDER_SPECIFICITY = "forest_floor"
_BORDER_BLENDED = ("river", "island_angled", "island_overhead", *_BORDER_NO_LAYERS)
# The spread checks' expected answers rest on these orders (terrain ids: water 1, sand 2, grass 0,
# broken road 25, and the dirt 6, 11 and 3 that ring the road in the river frame).
_BORDER_PRIORITY_ORDERS = ((1, 2), (2, 0), (25, 6), (25, 11), (25, 3))

_BORDER_SCENE: dict[str, tuple] = {}


def _border_scene(map_key: str):
    """(scenario, elevations), loaded once per map."""
    if map_key not in _BORDER_SCENE:
        scenario = load_map_and_units(str(BORDER_MAPS[map_key]))
        mm = scenario.map_manager
        elevations = np.zeros((mm.map_height, mm.map_width), dtype=np.int64)
        for tile in mm.terrain:
            elevations[tile.y, tile.x] = tile.elevation
        _BORDER_SCENE[map_key] = (scenario, elevations)
    return _BORDER_SCENE[map_key]


def _border_origin(frame_id: str, style: str) -> tuple[int, int]:
    """The crop's top-left canvas pixel, centred on the frame's centre tile."""
    map_key, _style, (cx, cy), (cw, ch), _on = _BORDER_FRAMES[frame_id]
    tp = BORDER_TILE_PX
    if style == "flat":
        return (2 * cx + 1) * tp // 2 - cw // 2, (2 * cy + 1) * tp // 2 - ch // 2
    scenario, elevations = _border_scene(map_key)
    proj = _border_proj(scenario)
    sx, sy = ig.tile_screen_origin(cx, cy, int(elevations[cy, cx]), proj)
    return sx + proj.half_w - cw // 2, sy + proj.half_h - ch // 2


def _border_proj(scenario):
    mm = scenario.map_manager
    return ig.canvas_size_and_origin(
        mm.map_width, mm.map_height, BORDER_TILE_PX, ig.MIN_ELEVATION, ig.MAX_ELEVATION,
        elev_step_pct=ig.ELEV_STEP_DEFAULT_PCT, corner_headroom_steps=0,
    )


def _border_render(frame_id: str, blend: bool, style: str | None = None) -> np.ndarray:
    """One frame at 1:1 through the app's own rect compositors, units off, tile_px and elev_step
    explicit so no user setting moves a crop. `style` renders the frame's tiles in another style."""
    map_key, frame_style, _centre, (cw, ch), _on = _BORDER_FRAMES[frame_id]
    style = style or frame_style
    scenario, elevations = _border_scene(map_key)
    layers = LayerState(terrain_blend=blend)
    x0, y0 = _border_origin(frame_id, style)
    with _layers_cleared(scenario, frame_id in _BORDER_NO_LAYERS):
        if style == "flat":
            return render.composite_rect_flat(
                scenario, x0, y0, x0 + cw, y0 + ch, BORDER_TILE_PX, with_units=False, layers=layers
            )
        return render.composite_rect_iso(
            scenario, x0, y0, x0 + cw, y0 + ch, elevations, _border_proj(scenario), BORDER_TILE_PX, {}, {},
            with_units=False, layers=layers,
        )


@contextmanager
def _layers_cleared(scenario, on: bool):
    """Every tile's layer set to -1 for the duration (when `on`), then restored."""
    tiles = list(scenario.map_manager.terrain) if on else []
    saved = [tile.layer for tile in tiles]
    for tile in tiles:
        tile.layer = -1
    try:
        yield
    finally:
        for tile, layer in zip(tiles, saved, strict=True):
            tile.layer = layer


def _one_tile_pick(map_key: str) -> tuple[int, int]:
    """The tile one-tile-plain draws plainly on this map: of the tiles within 2 of its frames' shared
    centre that a higher neighbour bleeds into and that carry no layer, the one whose blended block
    differs from its plain crop in the most pixels (the first in row order on a tie)."""
    scenario, _elevations = _border_scene(map_key)
    mm = scenario.map_manager
    cx, cy = next(c for fid, (m, _s, c, _z, _on) in _BORDER_FRAMES.items() if m == map_key and fid in _BORDER_BLENDED)
    ys, xs = np.mgrid[cy - 2 : cy + 3, cx - 2 : cx + 3]
    xs, ys = xs.ravel(), ys.ravel()
    best = None
    for x, y, key in zip(xs.tolist(), ys.tolist(), render._blend_keys(scenario, xs, ys), strict=True):
        if key is None or key[1] >= 0 or not key[2]:
            continue
        tile = mm.terrain[y * mm.map_width + x]
        blended = render._top_block(tile, BORDER_TILE_PX, blend=key)
        changed = int(np.count_nonzero(rp.changed_pixels(blended, render._top_block(tile, BORDER_TILE_PX))))
        if best is None or changed > best[0]:
            best = (changed, x, y)
    return best[1], best[2]


def _flipped_blend_mask(real):
    """A get_blend_mask stand-in serving 128 - alpha: the neighbour's weight read as the tile's own."""
    flipped: dict[tuple, np.ndarray] = {}

    def get(mode: int, tile_index: int, tile_px: int):
        key = (mode, tile_index, tile_px)
        if key not in flipped:
            alpha = real(mode, tile_index, tile_px)
            if alpha is None:
                return None
            out = (128 - alpha.astype(np.int16)).astype(alpha.dtype)
            out.flags.writeable = False
            flipped[key] = out
        return flipped[key]

    return get


class _BorderInject:
    """Patches for one known blending defect, around a whole run's renders. blend-off needs none:
    the run renders every frame with the View > Layers row off."""

    def __init__(self, name: str | None, plain_tiles: dict | None = None) -> None:
        self.name = name
        self.plain_tiles = plain_tiles or {}  # id(scenario) -> (x, y)
        self.saved: list[tuple] = []

    def _patch(self, obj, attr: str, new) -> None:
        self.saved.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, new)

    def __enter__(self):
        asset_source.get_blend_mask.cache_clear()  # before polarity-flip swaps in an uncached stand-in
        if self.name == "polarity-flip":
            self._patch(asset_source, "get_blend_mask", _flipped_blend_mask(asset_source.get_blend_mask))
        elif self.name == "priority-reversed":
            # Every order between two terrains turned round; 1000 keeps them all non-negative.
            table = {tid: e._replace(priority=1000 - e.priority) for tid, e in terrain_blend.BLEND_TABLE.items()}
            self._patch(terrain_blend, "BLEND_TABLE", table)
        elif self.name == "one-tile-plain":
            real = render._blend_keys
            plain = self.plain_tiles

            def keys(scenario, xs, ys, terrain_ids=None, layer_ids=None):
                out = real(scenario, xs, ys, terrain_ids, layer_ids)
                if (drop := plain.get(id(scenario))) is not None:
                    for i in np.flatnonzero((np.asarray(xs) == drop[0]) & (np.asarray(ys) == drop[1])).tolist():
                        out[i] = None
                return out

            self._patch(render, "_blend_keys", keys)
        elif self.name == "edge-residue":
            # TASK-283's defect: blendomatic's own 4-24 of 128 kept along every bleeding edge.
            self._patch(terrain_blend, "fade_bleeding_edges", lambda alpha, _tile: alpha)
        terrain_blend.clear_caches()
        return self

    def __exit__(self, *exc):
        for obj, attr, old in reversed(self.saved):
            setattr(obj, attr, old)
        asset_source.get_blend_mask.cache_clear()
        terrain_blend.clear_caches()


BORDER_INJECTS = {
    "blend-off": None, "polarity-flip": None, "priority-reversed": None, "edge-residue": None, "one-tile-plain": "one-tile",
}
BORDER_COARSE_INJECT = "blend-off"


def _border_frame_images(inject: str | None) -> dict[str, np.ndarray]:
    plain_tiles = {}
    if inject == "one-tile-plain":
        plain_tiles = {id(_border_scene(m)[0]): _one_tile_pick(m) for m in BORDER_MAPS}
    with _BorderInject(inject, plain_tiles):
        return {
            fid: _border_render(fid, blend=on and inject != "blend-off")
            for fid, (*_rest, on) in _BORDER_FRAMES.items()
        }


def _ground_borders_spec() -> rp.PackSpec:
    """TASK-143's terrain blending (GH #13): neighbour transitions on an unlayered river junction,
    which ground spreads against a plain-cells control, and C2_ElCid_coop_6's heavily layered island
    shore angled and overhead. Round 2 (TASK-283) splits the island: island_shore asks about
    neighbour transitions on frames with every layer cleared, island_patches about the layered
    frames' patches, whose oracle stays the in-game capture (round 1's island_edges could not tell
    the two apart, since layer patches sit on the shoreline)."""
    captions = {
        "river": (
            "Angled view: a river with sandy banks running through grassland, shallow water with plants at the "
            "top, and a stretch of broken paved road on the right."
        ),
        "river_plain": (
            "The same ground as the river frame, but with every map cell drawn plainly and wholly in its own "
            "single kind of ground."
        ),
        "island_angled": "Angled view: a sandy island shore and the water around it, including shallows with plants.",
        "island_overhead": "The same island shore seen straight from above; here each map cell is a square.",
        "island_shore": "Angled view: an island shore and the water around it.",
        "island_shore_overhead": "The shore frame's ground seen straight from above; here each map cell is a square.",
        "forest_floor": "Angled view of a stretch of forest floor; the trees themselves are not drawn.",
    }
    frames = tuple(rp.Frame(fid, f"{fid}.png", "raw" if on else "control", caption=captions[fid])
                   for fid, (*_rest, on) in _BORDER_FRAMES.items())
    texture_features = (
        "features inside one ground's own texture, such as the individual paving stones of the road and the bare "
        "earth between them, ripples and darker or lighter areas in the water, or the plants in the shallows"
    )
    island_features = (
        "features inside one ground's own texture, such as ripples and darker or lighter areas in the water, the "
        "plants in the shallows, or rocks"
    )

    def spread(check_id: str, a: str, b: str, where: str, a_word: str, b_word: str) -> rp.Check:
        return rp.Check(
            id=check_id,
            question=(
                f"Compare the river frame with the plain-cells frame. In the plain-cells frame every map cell is "
                f"drawn wholly in its own ground, so each border there runs exactly along the cell grid. Look at "
                f"{where}. First, in your own words, describe how that border's position and shape differ between "
                f"the two frames.\n\n"
                f"Then the closed question. Relative to where that border runs in the plain-cells frame, which "
                f"ground reaches across it in the river frame: does the {a} extend out over cells that are plain "
                f"{b} in the plain-cells frame, or does the {b} extend over cells that are plain {a} there?\n\n"
                f"Answer {a_word}-SPREADS or {b_word}-SPREADS, or NEITHER if the border sits where the plain-cells "
                f"frame has it."
            ),
            frames=("river", "river_plain"),
            answers=(f"{a_word}-SPREADS", f"{b_word}-SPREADS", "NEITHER", "CANNOT-TELL"),
        )

    checks = (
        rp.Check(
            id="river_borders",
            question=(
                "The river frame shows several kinds of ground meeting: water, sand, grassland, shallow water with "
                "plants, and a broken paved road. The ground is a grid of diamond-shaped map cells; no grid lines "
                "are drawn. First, in your own words, describe how each pair of grounds meets: what the meeting "
                "line looks like, how wide any transition between them is, and whether its shape follows the cell "
                "grid.\n\n"
                "Then the closed question. Is any place where two different grounds meet drawn as a hard, sharp "
                "edge: one ground stopping dead against the other along straight cell edges, so the grid's zigzag "
                "or staircase shows along that line, even for a single cell? NOT such an edge: (1) the frame's own "
                f"edges; (2) {texture_features}; (3) a sharp edge that wanders irregularly rather than following "
                "the straight cell edges.\n\n"
                "Answer HARD if at least one such edge is visible, SOFT if every meeting line is a gradual "
                "transition."
            ),
            frames=("river",),
            answers=("HARD", "SOFT", "CANNOT-TELL"),
        ),
        rp.Check(
            id="river_transitions",
            question=(
                "Same river frame. Follow each transition between two grounds along its length, cell after cell. "
                "First, in your own words, describe how one of them (say, the water's edge) changes as you follow "
                "it.\n\n"
                "Then the closed question. Is any transition broken: does it stop, or jump sideways, at a straight "
                "cell edge; change width abruptly from one cell to the next; or does a band or blob of one ground "
                "sit detached inside the other, separated from the place where the two grounds meet by a strip of "
                "the other ground? NOT broken: (1) a transition whose width or course varies gradually; "
                f"(2) {texture_features}; (3) the frame's own edges.\n\n"
                "Answer BROKEN if at least one transition is broken anywhere, CONTINUOUS if each runs on unbroken."
            ),
            frames=("river",),
            answers=("BROKEN", "CONTINUOUS", "CANNOT-TELL"),
        ),
        spread("spread_shore", "water", "sand", "where the river's water meets its sand banks (not the shallows "
               "with plants at the top)", "WATER", "SAND"),
        spread("spread_banks", "sand", "grassland", "where the sand banks meet the grassland on either side",
               "SAND", "GRASS"),
        spread("spread_road", "road", "earth", "where the broken paved road on the right meets the cells of plain "
               "orange-brown bare earth that surround it. Those earth cells are a ground of their own, distinct from "
               "the bare earth between the road's paving stones; the plain-cells frame shows which cells are which",
               "ROAD", "EARTH"),
        rp.Check(
            id="island_shore",
            question=(
                "The shore frame and the shore overhead frame show one stretch of ground: land meeting water, and "
                "kinds of water meeting each other. First, in your own words, describe how each pair of grounds "
                "meets: what the meeting line looks like, how wide any transition between them is, and whether "
                "its shape follows the cell grid.\n\n"
                "Then the closed question. Is any place where two different grounds meet drawn as a hard, sharp "
                "edge that follows the straight cell edges (in the angled frame a zigzag or staircase of "
                "diamond-cell edges, in the overhead frame straight horizontal and vertical square-cell edges), "
                f"even for a single cell? NOT such an edge: (1) the frames' own edges; (2) {island_features}; (3) a "
                "sharp edge that wanders irregularly rather than following the straight cell edges.\n\n"
                "Answer HARD if at least one such edge is visible, SOFT if every meeting line is a gradual "
                "transition."
            ),
            frames=("island_shore", "island_shore_overhead"),
            answers=("HARD", "SOFT", "CANNOT-TELL"),
        ),
        rp.Check(
            id="island_patches",
            question=(
                "The two island frames (angled and overhead) show one shore. First, in your own words, describe "
                "any shapes or patches of a different-looking ground you can see lying within the water or within "
                "the land, away from the main line where land meets water, and what their outlines look like.\n\n"
                "Then the closed question. Do such patches have hard, sharp outlines that follow the straight cell "
                "edges (in the angled frame a zigzag or staircase of diamond-cell edges, in the overhead frame "
                f"straight horizontal and vertical square-cell edges)? NOT such outlines: {island_features}, and "
                "outlines that fade gradually or wander irregularly.\n\n"
                "Answer GRID-OUTLINES if at least one patch has such an outline, SOFT-OUTLINES if patches exist "
                "but none does, or NO-PATCHES if there are no such patches."
            ),
            frames=("island_angled", "island_overhead"),
            answers=("GRID-OUTLINES", "SOFT-OUTLINES", "NO-PATCHES", "CANNOT-TELL"),
        ),
        rp.Check(
            id="forest_floor",
            question=(
                "The forest-floor frame shows one stretch of ground. Is there anywhere in it a place where one kind "
                "of ground meets another, a patch of a different-looking ground, or a straight edge, line or seam "
                "following a grid? The texture's own blotches, specks and darker or lighter areas do not count.\n\n"
                "Answer PRESENT if you see any of these, NONE if the frame shows none."
            ),
            frames=(_BORDER_SPECIFICITY,),
            answers=("PRESENT", "NONE", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id="ground_borders",
        title="Ground borders review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders of bare ground (no units or objects) from a map editor for an isometric "
            "strategy game, at normal zoom. Each frame is a crop, enlarged by a whole-number factor with no "
            "smoothing, so a one-pixel feature in the original appears that many pixels wide here. Judge only what "
            "you can actually see."
        ),
        injects=tuple(BORDER_INJECTS),
    )


def _border_prerequisites() -> str | None:
    """Why the ground_borders pack cannot render here, or None."""
    for path in BORDER_MAPS.values():
        if not path.is_file():
            return f"needs {path.relative_to(ROOT)} (the gitignored examples/ corpus)"
    if asset_source.get_install_path() is None:
        return "needs a configured AoE2:DE install for the terrain textures"
    if not render.blend_effective():
        return "needs the install's blendomatic_x1.dat, or blending never engages"
    return None


def _border_tile_blocks(frame_id: str, style: str) -> dict[tuple[int, int], tuple[slice, slice]]:
    """Flat only: each tile wholly inside the crop -> its block's slices in the frame."""
    _m, _s, _c, (cw, ch), _on = _BORDER_FRAMES[frame_id]
    x0, y0 = _border_origin(frame_id, style)
    tp = BORDER_TILE_PX
    out = {}
    for ty in range(-(-y0 // tp), (y0 + ch) // tp):
        for tx in range(-(-x0 // tp), (x0 + cw) // tp):
            out[(tx, ty)] = (slice(ty * tp - y0, (ty + 1) * tp - y0), slice(tx * tp - x0, (tx + 1) * tp - x0))
    return out


def _border_interior_failures() -> tuple[list[str], str]:
    """Overhead, where a tile's pixels are its own square: every tile with no blend key is
    byte-identical with blending off, and some keyed tiles change. Both maps (the river's
    tiles rendered overhead for this, not a frame)."""
    failures, shown = [], []
    for frame_id in ("river", "island_overhead"):
        on, off = _border_render(frame_id, True, "flat"), _border_render(frame_id, False, "flat")
        scenario = _border_scene(_BORDER_FRAMES[frame_id][0])[0]
        blocks = _border_tile_blocks(frame_id, "flat")
        xs, ys = (np.array(v) for v in zip(*blocks, strict=True))
        plain = keyed = changed = 0
        for (x, y), key in zip(blocks, render._blend_keys(scenario, xs, ys), strict=True):
            same = np.array_equal(on[blocks[(x, y)]], off[blocks[(x, y)]])
            if key is None:
                plain += 1
                if not same:
                    failures.append(f"{frame_id}: tile ({x}, {y}) has no blend key but changes with blending on")
            else:
                # A keyed tile may still match: beach 2 and 79 (desert 13 and 14) share one texture.
                keyed += 1
                changed += not same
        if not plain or not changed:
            failures.append(f"{frame_id}: {plain} plain and {changed} changed keyed tiles, need both")
        shown.append(f"{frame_id} {plain} plain/{changed} of {keyed} keyed changed")
    return failures, ", ".join(shown)


def _border_seam_residues(frame_id: str, blend: bool = True) -> list[tuple[float, tuple[int, int], tuple[int, int]]]:
    """Overhead: (residue, tile, neighbour) for every edge-adjacent pair wholly in the crop where the
    neighbour's terrain is the only one bleeding into the tile and neither is layered. The residue is
    the mean summed-RGB difference between the tile's pixel line along that edge and the neighbour's
    own texture on the same line (TASK-283). Measured against the texture, not the neighbour's pixels,
    since a pair straddling the texture's wrap point steps 20-30 with no blending involved."""
    img = _border_render(frame_id, blend, "flat").astype(np.int64)
    scenario = _border_scene(_BORDER_FRAMES[frame_id][0])[0]
    blocks = _border_tile_blocks(frame_id, "flat")
    xs, ys = (np.array(v) for v in zip(*blocks, strict=True))
    with _layers_cleared(scenario, frame_id in _BORDER_NO_LAYERS):
        tids = render.tile_field_mirror(scenario, "terrain_id").reshape(-1)
        layer_ids = render.tile_field_mirror(scenario, "layer").reshape(-1)
        keys = dict(zip(blocks, render._blend_keys(scenario, xs, ys), strict=True))
    width = scenario.map_manager.map_width
    tp = BORDER_TILE_PX
    edge_px = {(1, 0): np.s_[:, -1], (-1, 0): np.s_[:, 0], (0, 1): np.s_[-1, :], (0, -1): np.s_[0, :]}

    out = []
    for (x, y), key in keys.items():
        if key is None or key[1] >= 0 or len({e[0] for e in key[2]}) != 1:
            continue
        n_tid = key[2][0][0]
        for (dx, dy), line in edge_px.items():
            n = (x + dx, y + dy)
            n_layer = int(layer_ids[n[1] * width + n[0]])
            if n not in blocks or int(tids[n[1] * width + n[0]]) != n_tid or n_layer not in (-1, n_tid):
                continue
            texture = asset_source.get_terrain_texture_array(n_tid)
            ox, oy = render._crop_offset(x, y, texture.shape[0], tp)
            own = img[blocks[(x, y)]][line].sum(axis=-1)
            pure = texture[oy : oy + tp, ox : ox + tp].astype(np.int64)[line].sum(axis=-1)
            out.append((float(np.abs(own - pure).mean()), (x, y), n))
    return out


# Alpha 0 composes to the neighbour's exact pixel. Unfaded blendomatic edges measured up to 28.8
# on these maps, blend-off's plain cell edges 160-351.
BORDER_SEAM_RESIDUE_MAX = 0.0


def _check_ground_borders() -> None:
    """--check for ground_borders: spec coupling, frame band, blending engaged, the spread checks'
    priority orders, interior tiles unchanged (overhead, per tile), every single-neighbour bleeding
    edge wholly the neighbour's (overhead), each inject's reach (printed), neither the control nor
    the specificity frame reached by any inject, every coarse-class inject reaching every blended
    frame, and one-tile-plain bounded to one tile's area per style under blend-off's delta."""
    reason = _border_prerequisites()
    if reason:
        raise SystemExit(f"ground_borders: {reason}")
    failures: list[str] = []
    spec = _ground_borders_spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    prio = {tid: e.priority for tid, e in terrain_blend.BLEND_TABLE.items()}
    for high, low in _BORDER_PRIORITY_ORDERS:
        if not prio[high] > prio[low]:
            failures.append(f"terrain {high} no longer outranks {low}, so a spread check's expected answer moved")

    clean = _border_frame_images(None)
    for frame in spec.frames:
        try:
            scaled, _factor = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))
    interior, interior_shown = _border_interior_failures()
    failures += interior
    print(f"interior (overhead tiles): {interior_shown}")
    seams_shown = []
    for frame_id in ("river", "island_overhead", "island_shore_overhead"):
        residues = _border_seam_residues(frame_id)
        worst = max(residues, default=None)
        if not residues:
            failures.append(f"{frame_id}: no single-neighbour bleeding edge to measure")
        elif worst[0] > BORDER_SEAM_RESIDUE_MAX:
            failures.append(
                f"{frame_id}: tile {worst[1]}'s edge toward {worst[2]} keeps {worst[0]:.1f} summed RGB of its own "
                f"ground, over {BORDER_SEAM_RESIDUE_MAX}"
            )
        seams_shown.append(f"{frame_id} {len(residues)} edges, max {worst[0] if worst else 0:.1f}")
    print(f"bleeding-edge residue (overhead): {', '.join(seams_shown)}")

    # One tile's own pixels per style: a 64 px square overhead, a 1024 px diamond angled.
    tile_area = {"flat": BORDER_TILE_PX**2, "stepped": len(ig.diamond_indices(BORDER_TILE_PX)[0])}
    one_tile = {
        fid: rp.LocalizedBound(max_changed_fraction=0.02, max_component_px=tile_area[style])
        for fid, (_m, style, *_rest) in _BORDER_FRAMES.items()
    }
    reach: dict[str, dict[str, tuple[int, int]]] = {}
    for name, bound in BORDER_INJECTS.items():
        injected = _border_frame_images(name)
        touched = {}
        for frame_id in _BORDER_FRAMES:
            mask = rp.changed_pixels(clean[frame_id], injected[frame_id])
            changed = int(np.count_nonzero(mask))
            if changed:
                touched[frame_id] = (changed, rp.largest_component_px(mask))
            if bound and frame_id in _BORDER_BLENDED:
                failures += one_tile[frame_id].failures(mask, f"inject {name} on {frame_id}")
        if missed := [fid for fid in _BORDER_BLENDED if fid not in touched]:
            failures.append(f"inject {name}: changed nothing in {', '.join(missed)}")
        if stray := sorted(set(touched) - set(_BORDER_BLENDED)):
            failures.append(f"inject {name}: reached {', '.join(stray)}, outside the blended frames")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    print(f"one-tile-plain tiles: {', '.join(f'{m} {_one_tile_pick(m)}' for m in BORDER_MAPS)}")

    coarse = reach[BORDER_COARSE_INJECT]
    for name, bound in BORDER_INJECTS.items():
        if bound is None:
            continue
        for frame_id, (_px, comp) in sorted(reach[name].items()):
            if frame_id in coarse and comp >= coarse[frame_id][1]:
                failures.append(
                    f"inject {name} on {frame_id}: largest delta {comp} px is not under the coarse "
                    f"inject's {coarse[frame_id][1]} px there, so the two no longer differ in size"
                )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(
        f"OK: ground_borders frames couple to its checks and land in {rp.MIN_LONG_EDGE}-{rp.MAX_LONG_EDGE} px; "
        f"unkeyed overhead tiles are unchanged by blending; every single-neighbour bleeding edge is wholly the "
        f"neighbour's; all {len(BORDER_INJECTS)} injects bite every blended frame and none reaches "
        f"{_BORDER_SPECIFICITY} or the plain-cells control; one-tile-plain stays within one tile's area"
    )


PACKS = {
    "shadow_band": _shadow_band_spec,
    "ground_texture": _ground_texture_spec,
    "sloped_mip": _sloped_mip_spec,
    "ground_borders": _ground_borders_spec,
}


# ---------------------------------------------------------------- rendering


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    """Every frame in the shadow_band pack, at 1:1, before upscaling."""
    boxes = _clean_boxes()
    images: dict[str, np.ndarray] = {}
    with _GeometryPatch(INJECTS.get(inject) if inject else None):
        for pct in (50, 100):
            raw, control = _scene_pair("band", pct, boxes[("band", pct)])
            images[f"band_pct{pct}_raw"] = raw
            images[f"band_pct{pct}_control"] = control
            images[f"band_pct{pct}_diff"] = seam_eyeball._amplify_diff(raw, control)
        for pct in (10, 50, 100):
            raw, control = _scene_pair("run", pct, boxes[("run", pct)])
            images[f"run_pct{pct}_raw"] = raw
            images[f"run_pct{pct}_diff"] = seam_eyeball._amplify_diff(raw, control)
        for pct in (50, 100):
            raw, control = _scene_pair("corner", pct, boxes[("corner", pct)])
            images[f"corner_pct{pct}_raw"] = raw
            images[f"corner_pct{pct}_control"] = control
            images[f"corner_pct{pct}_diff"] = seam_eyeball._amplify_diff(raw, control)
        images["flat_raw"] = _flat_frame(images["run_pct100_raw"].shape[:2])
    return images


def _review_markdown(spec: rp.PackSpec, rendered: dict[str, tuple[int, int, int]], pack_dir: Path) -> str:
    lines = [
        f"# {spec.title}",
        "",
        spec.preamble,
        "",
        "## How to answer",
        "",
        "- Answer every check below, one line each, in the exact format at the bottom.",
        "- Open each image file listed for a check before answering it. The paths are absolute.",
        "- **Open only those image files.** Do not read any other file, do not run any command,",
        "  and do not explore the surrounding directories or repository. The code that produced",
        "  these frames would tell you what to expect, and an answer informed by it is worthless:",
        "  the whole reason you were asked is that you do not know what changed.",
        "- Answer only from what is visible in the frames a check names. Do not reason from what",
        "  the code might do, and do not use a frame a check does not list.",
        "- `CANNOT-TELL` is a real answer, not a cop-out: use it when the framing does not let you",
        "  see the thing being asked about. Guessing instead makes the answer worthless.",
        "- Add one short sentence per check saying what in the image drove the answer.",
        "",
        "## Frames",
        "",
        "| file | enlarged | caption |",
        "|---|---|---|",
    ]
    for frame in spec.frames:
        width, height, factor = rendered[frame.id]
        lines.append(f"| `{pack_dir / frame.filename}` | {factor}x, {width}x{height} | {frame.caption} |")

    lines += ["", "## Checks", ""]
    for check in spec.checks:
        lines += [
            f"### {check.id}",
            "",
            check.question,
            "",
            "Allowed answers: " + " / ".join(f"`{a}`" for a in check.answers),
            "",
            "Frames for this check:",
            "",
        ]
        lines += [f"- `{pack_dir / spec.frame(fid).filename}`" for fid in check.frames]
        lines.append("")

    lines += ["## Response format", "", "```"]
    lines += [f"{check.id}: <{'|'.join(check.answers)}> -- <what you saw>" for check in spec.checks]
    lines += ["```", ""]
    return "\n".join(lines)


def _run_token(pack_id: str, inject: str | None, replica: int = 0) -> str:
    """An opaque directory name per run, INCLUDING the clean one.

    Every frame path appears verbatim in REVIEW.md, so a directory called
    `shadow_band__notch-inject` hands the reviewer the answer key in the one
    place it is guaranteed to read. Naming only the injects opaquely would
    not help either: a run that was not in the opaque form would be the clean
    one by elimination. Deterministic, so re-running overwrites rather than
    accumulating, and the mapping is printed to the operator's terminal --
    which the reviewer never sees.

    `replica` gives the same run a second and third directory, for the k=3
    clean replication REVIEW_PACK.md requires. Replica 0 hashes exactly as
    before, so earlier rounds' directory names still hold.
    """
    key = f"{pack_id}|{inject or 'clean'}" + (f"|{replica}" if replica else "")
    return hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:10]


def _pack_injects(pack_id: str) -> dict:
    return {"ground_texture": GROUND_INJECTS, "sloped_mip": SLOPED_INJECTS, "ground_borders": BORDER_INJECTS}.get(
        pack_id, INJECTS
    )


def _pack_images(pack_id: str, inject: str | None) -> dict[str, np.ndarray]:
    if pack_id == "ground_texture":
        return _ground_frame_images(inject)
    if pack_id == "sloped_mip":
        return _sloped_frame_images(inject)
    if pack_id == "ground_borders":
        return _border_frame_images(inject)
    return _frame_images(inject)


def _frame_tile_px(pack_id: str, frame_id: str) -> int:
    if pack_id == "ground_texture":
        return _GROUND_FRAMES[frame_id][1]
    if pack_id == "sloped_mip":
        # Graphics quality 3 puts both 240x240 maps at 64 px per tile at level 0.
        return 64 >> -_SLOPED_FRAMES[frame_id][1]
    if pack_id == "ground_borders":
        return BORDER_TILE_PX
    return TILE_PX


def generate(out_dir: Path, pack_id: str, inject: str | None, replica: int = 0) -> list[Path]:
    spec = PACKS[pack_id]()
    rp.validate_spec(spec)
    injects = _pack_injects(pack_id)
    if inject is not None and inject not in injects:
        raise SystemExit(f"unknown inject {inject!r} for {pack_id}; choose from {', '.join(injects)}")
    if pack_id == "ground_texture" and (reason := _ground_prerequisites()):
        raise SystemExit(f"ground_texture: {reason}")
    if pack_id == "sloped_mip" and (reason := _sloped_prerequisites()):
        raise SystemExit(f"sloped_mip: {reason}")
    if pack_id == "ground_borders" and (reason := _border_prerequisites()):
        raise SystemExit(f"ground_borders: {reason}")

    pack_dir = out_dir / pack_id / _run_token(pack_id, inject, replica)
    pack_dir.mkdir(parents=True, exist_ok=True)

    images = _pack_images(pack_id, inject)
    written: list[Path] = []
    rendered: dict[str, tuple[int, int, int]] = {}
    for frame in spec.frames:
        scaled, factor = rp.frame_to_band(images[frame.id])
        path = pack_dir / frame.filename
        seam_eyeball._save_png(scaled, path)
        written.append(path)
        rendered[frame.id] = (scaled.shape[1], scaled.shape[0], factor)

    review_path = pack_dir / "REVIEW.md"
    review_path.write_text(_review_markdown(spec, rendered, pack_dir))
    written.append(review_path)

    # No `inject` key, deliberately: the manifest sits beside REVIEW.md in a
    # directory the reviewer is told to read, so recording which run this is
    # here would defeat the opaque directory name above. The operator gets it
    # on stdout instead.
    manifest = {
        "pack": spec.id,
        "frames": [
            {
                "id": f.id,
                "file": f.filename,
                "kind": f.kind,
                "tile_px": _frame_tile_px(pack_id, f.id),
                "width": rendered[f.id][0],
                "height": rendered[f.id][1],
                "upscale": rendered[f.id][2],
            }
            for f in spec.frames
        ],
        "checks": [{"id": c.id, "frames": list(c.frames), "answers": list(c.answers)} for c in spec.checks],
    }
    manifest_path = pack_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    written.append(manifest_path)
    return written


# -------------------------------------------------------------------- check


def _component_count(scene: str, inject: str | None, pct: int = 100) -> int:
    box = _clean_boxes()[(scene, pct)]
    with _GeometryPatch(INJECTS.get(inject) if inject else None):
        raw, control = _scene_pair(scene, pct, box)
    return rp.components(rp.changed_pixels(raw, control))


def check(pack_id: str = "shadow_band") -> None:
    """--check for one pack; see each pack's own check for what it asserts."""
    if pack_id == "ground_texture":
        _check_ground_texture()
    elif pack_id == "sloped_mip":
        _check_sloped_mip()
    elif pack_id == "ground_borders":
        _check_ground_borders()
    else:
        _check_shadow_band()


def _check_shadow_band() -> None:
    """--check: deterministic, Qt-free, writes nothing.

    Asserts the four things a review pack can be wrong about without anyone
    noticing: the checklist and the frames are coupled both ways, every frame
    lands in the judgeable band, each inject actually changes pixels, and the
    localized injects really do differ in SIZE from the coarse one. That last
    one is the whole negative control: a reviewer that catches "the band is
    gone" has proved nothing about whether it resolves a 2px tick.

    It also prints, per inject, which frames it touched and by how much.
    Pre-registering a verdict matrix means predicting the off-diagonal cells,
    and "this inject cannot reach that frame" is a thing to measure rather
    than argue from geometry.
    """
    failures: list[str] = []
    spec = PACKS["shadow_band"]()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")

    clean = _frame_images(None)
    for frame in spec.frames:
        img = clean.get(frame.id)
        if img is None:
            failures.append(f"frame {frame.id}: nothing rendered")
            continue
        try:
            scaled, _factor = rp.frame_to_band(img)
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))

    reach: dict[str, dict[str, tuple[int, int]]] = {}
    for name, inject in INJECTS.items():
        injected = _frame_images(name)
        touched: dict[str, tuple[int, int]] = {}
        for frame in spec.frames:
            mask = rp.changed_pixels(clean[frame.id], injected[frame.id])
            changed = int(np.count_nonzero(mask))
            if changed:
                touched[frame.id] = (changed, rp.largest_component_px(mask))
                if inject.bound is not None:
                    failures += inject.bound.failures(mask, f"inject {name} on {frame.id}")
        if not touched:
            failures.append(f"inject {name}: changed no pixels anywhere in the pack")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")

    coarse = reach[COARSE_INJECT]
    for name in INJECTS:
        if name == COARSE_INJECT:
            continue
        for frame_id, (_px, comp) in sorted(reach[name].items()):
            if frame_id in coarse and comp >= coarse[frame_id][1]:
                failures.append(
                    f"inject {name} on {frame_id}: largest delta {comp} px is not under the coarse "
                    f"inject's {coarse[frame_id][1]} px there, so the two no longer differ in size"
                )

    clean_components = _component_count("band", None)
    coarse_components = _component_count("band", "no-apex-wedge")
    if clean_components != CLEAN_BAND_COMPONENTS:
        failures.append(
            f"clean band is {clean_components} connected components at pct 100, not {CLEAN_BAND_COMPONENTS}"
        )
    for pct in (50, 100):
        got = _component_count("corner", None, pct)
        if got != CLEAN_CORNER_COMPONENTS:
            failures.append(
                f"corner diff mask at pct {pct} is {got} components, not {CLEAN_CORNER_COMPONENTS}, "
                f"so corner_shape's SPLIT/WHOLE no longer maps onto the render"
            )
    if coarse_components < clean_components * COARSE_MIN_COMPONENT_RATIO:
        failures.append(
            f"inject no-apex-wedge: band goes from {clean_components} to {coarse_components} "
            f"connected components, under the {COARSE_MIN_COMPONENT_RATIO}x the coarse control needs"
        )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(
        f"OK: shadow_band frames couple to its checks, every frame lands in "
        f"{rp.MIN_LONG_EDGE}-{rp.MAX_LONG_EDGE} px, all {len(INJECTS)} injects bite and stay inside their bounds, and the band goes "
        f"{clean_components} -> {coarse_components} components under the coarse one, and the corner "
        f"diff mask is {CLEAN_CORNER_COMPONENTS} pieces"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument("--pack", default="shadow_band", choices=sorted(PACKS), help="Which pack to render")
    parser.add_argument(
        "--inject",
        choices=sorted({*INJECTS, *GROUND_INJECTS, *SLOPED_INJECTS, *BORDER_INJECTS}),
        help="Render the pack with one known defect applied",
    )
    parser.add_argument(
        "--replica", type=int, default=0, help="Replica number, for a second or third opaque directory of the same run"
    )
    parser.add_argument("--check", action="store_true", help="Validate without writing any files")
    args = parser.parse_args()

    if args.check:
        check(args.pack)
        return

    written = generate(args.out_dir, args.pack, args.inject, args.replica)
    for path in written:
        # relative_to(ROOT) raises for an --out-dir outside the repo -- every
        # file is already written by this point, so fall back to the absolute
        # path rather than crash on reporting after the real work succeeded.
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")
    print(
        f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory "
        f"{_run_token(args.pack, args.inject, args.replica)} -- record that pairing OUTSIDE the pack, "
        f"and hand the reviewer only that directory's REVIEW.md"
    )


if __name__ == "__main__":
    main()
