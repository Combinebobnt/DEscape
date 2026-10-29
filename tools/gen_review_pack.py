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

from descape import asset_source, native_composite, render, terrain_blend, unit_kind
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
            scenario, x0, y0, x0 + cw, y0 + ch, rise, proj, tile_px, {}, {}, with_units=False,
            fold=ig.tile_folds(elevations),
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
    """A get_terrain_texture_array stand-in serving one derived texture. AoE2
    only below 64 px tiles: Return of Rome crops a shrunk copy there (GH #67)."""

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


# ------------------------------------------------------ sprite_shrink pack
#
# GH #254 (TASK-440): Graphics Quality Default and above shrinks sprites with a premultiplied box
# filter instead of the nearest gather. Real install sprites only, on a flat grey ground; units
# and the building are composited piece by piece through render._clipped_paint_rgba, as the map does.


# (const, rotation): an archer, a spearman and a farmer with a hoe, and a market, each at a facing
# where the nearest gather breaks the most thin parts (measured: ink components per figure). The
# units get frames of their own, or the market's width would set their enlargement.
_SHRINK_GROUPS = {"figures": ((4, 0.0), (93, 3.142), (259, 1.571)), "building": ((84, 0.0),)}
_SHRINK_TEAM = 1
# frame id -> (group, half_w): 0.67x and 0.33x (the 64 and 32 px/tile levels at Default), and 1x,
# where neither filter changes a pixel, so neither symptom can occur.
_SHRINK_FRAMES = {
    "figures_1": ("figures", 32), "building_1": ("building", 32),
    "figures_2": ("figures", 48), "building_2": ("building", 48),
    "figures_3": ("figures", 16), "building_3": ("building", 16),
}
_SHRINK_SPECIFICITY = ("figures_2", "building_2")
_SHRINK_SHRUNK = ("figures_1", "building_1", "figures_3", "building_3")
_SHRINK_GROUND = 128
_SHRINK_PAD = 8
SHRINK_INJECTS = {"nearest": None, "straight-average": "edge-px"}
SHRINK_COARSE_INJECT = "nearest"
# The fringe measure (_shrink_fringe()): edge px whose own colour, in the clean render, is at
# least SHRINK_LIGHT above the ground, rendered at least SHRINK_FRINGE_DELTA below it. A correct
# blend of a light colour and the ground never lands below the ground, so clean's is 0 by
# construction. Straight-average measured 0.73 and 0.89 on the building frames (234 and 73 light
# edge px, 2026-10-22), 0.35 and 0.83 on the figures (17 and 6 px, too few to gate on), so a
# frame with SHRINK_LIGHT_MIN_PX light edge px must reach SHRINK_STRAIGHT_FRINGE_MIN.
SHRINK_LIGHT = 16
SHRINK_FRINGE_DELTA = 8
SHRINK_LIGHT_MIN_PX = 50
SHRINK_STRAIGHT_FRINGE_MIN = 0.5
SHRINK_STRAY_DELTA_MAX = 1


def _straight_box(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    """The straight-average inject: _box_resize_rgba()'s taps without premultiplying, so a
    transparent px's colour (black) is averaged into its opaque neighbours'."""
    from descape import unit_sprites

    src_h, src_w = rgba.shape[:2]
    if width > src_w or height > src_h:
        return unit_sprites._resize_rgba(rgba, width, height)
    ys, wy = unit_sprites._area_taps(src_h, height)
    xs, wx = unit_sprites._area_taps(src_w, width)
    src = rgba.astype(np.float32)
    rows = sum(wy[:, k, None, None] * src[ys[:, k]] for k in range(wy.shape[1]))
    out = sum(wx[None, :, k, None] * rows[:, xs[:, k]] for k in range(wx.shape[1]))
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


@contextmanager
def _shrink_inject(name: str | None):
    """Smooth shrinking on (the clean run and straight-average) or off (nearest), with the scaled
    cache dropped on both sides so no run reads another's sprites."""
    from descape import unit_sprites

    was, real = unit_sprites.smooth_shrink(), unit_sprites._box_resize_rgba
    unit_sprites.set_smooth_shrink(name != "nearest")
    if name == "straight-average":
        unit_sprites._box_resize_rgba = _straight_box
    unit_sprites._scaled_cache.clear()
    unit_sprites.clear_slice_cache()
    try:
        yield
    finally:
        unit_sprites._box_resize_rgba = real
        unit_sprites.set_smooth_shrink(was)
        unit_sprites._scaled_cache.clear()
        unit_sprites.clear_slice_cache()


def _shrink_lineup(frame_id: str) -> tuple[np.ndarray, list[tuple[np.ndarray, int, int]]]:
    """(frame, figures): the frame's group left to right, hotspots on one baseline, each piece
    painted onto the grey ground; figures are (rgba, x0, y0), each figure's own pieces flattened
    by source-over at its place in the frame, for the --check measurements."""
    from descape import unit_sprites

    group, half_w = _SHRINK_FRAMES[frame_id]
    placed = []
    for const, rotation in _SHRINK_GROUPS[group]:
        pieces = unit_sprites.sprite_pieces_for(const, rotation, _SHRINK_TEAM, half_w)
        if not pieces:
            raise SystemExit(f"sprite_shrink: const {const} draws no sprite on this install")
        boxes = [
            (p.dx - p.draw.hotspot_x, p.dy - p.draw.hotspot_y, p.draw.rgba.shape[1], p.draw.rgba.shape[0]) for p in pieces
        ]
        left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
        right, bottom = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
        placed.append((pieces, boxes, left, top, right, bottom))
    above = max(-p[3] for p in placed)
    below = max(p[5] for p in placed)
    width = sum(p[4] - p[2] for p in placed) + _SHRINK_PAD * (len(placed) + 1)
    frame = np.full((above + below + 2 * _SHRINK_PAD, width, 3), _SHRINK_GROUND, dtype=np.uint8)
    figures = []
    x = _SHRINK_PAD
    for pieces, boxes, left, top, right, bottom in placed:
        ax, ay = x - left, _SHRINK_PAD + above  # the hotspot
        flat = np.zeros((bottom - top, right - left, 4), dtype=np.uint8)
        for piece, (bx, by, _w, _h) in zip(pieces, boxes, strict=True):
            render._clipped_paint_rgba(frame, ay + by, ax + bx, piece.draw.rgba)
            unit_sprites._source_over(flat, by - top, bx - left, piece.draw.rgba)
        figures.append((flat, ax + left, ay + top))
        x += right - left + _SHRINK_PAD
    return frame, figures


def _shrink_frame_images(inject: str | None) -> dict[str, np.ndarray]:
    with _shrink_inject(inject):
        return {fid: _shrink_lineup(fid)[0] for fid in _SHRINK_FRAMES}


def _sprite_shrink_spec() -> rp.PackSpec:
    """GH #254's box-filtered sprite shrink: thin parts and outlines at the two shrinking levels,
    and the 1x frames where neither filter can change a pixel (specificity)."""
    captions = {
        "figures": "Three figures standing on a plain grey background, left to right: an archer, a spearman and a "
        "farmer holding a hoe.",
        "building": "A market building with stalls, a bell tower and a rope, on a plain grey background.",
    }
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=captions[group]) for fid, (group, _hw) in _SHRINK_FRAMES.items()
    )
    small, large = ", ".join(_SHRINK_SHRUNK), " and ".join(_SHRINK_SPECIFICITY)
    sizes = (
        f"Frames {small} show three figures and a market building at two smaller sizes; frames {large} show the same "
        "figures and building drawn larger, for comparison."
    )
    not_broken = (
        "(1) a part that stays continuous but changes in thickness, colour or brightness along its length, faint, "
        "greyish or see-through stretches included; (2) a part passing behind another part; (3) a gap that the larger "
        "frames show too, such as between two separate stalls, crates or barrels"
    )
    checks = (
        rp.Check(
            id="thin_parts",
            question=(
                f"{sizes} First, in your own words, describe the thinnest straight parts in the smaller frames, such "
                "as the bow and its string, the spear's shaft, the hoe's handle, and the market's rope, poles, "
                "railings and beams, and how each one looks along its length.\n\n"
                "Then the closed question. In the smaller frames, is any such thin straight part drawn broken: "
                "interrupted by gaps of grey background so that it shows as a row of separate dashes or dots, or with "
                "a stretch missing so that its pieces float apart, where the larger frames show one continuous piece? "
                f"NOT broken: {not_broken}.\n\n"
                "Answer BROKEN if at least one thin straight part is broken, WHOLE if every one runs unbroken."
            ),
            frames=(*_SHRINK_SHRUNK, *_SHRINK_SPECIFICITY),
            answers=("BROKEN", "WHOLE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="edge_tone",
            question=(
                f"{sizes} First, in your own words, describe the outer edges of the figures and the building in the "
                "smaller frames, where they meet the grey background, and the colour of their thin parts (the spear "
                "shaft, the hoe handle, the bow, the market's rope and poles), each compared with how the same edge "
                "or part looks in the larger frames.\n\n"
                "Then the closed question. In the smaller frames, is there a dark fringe that the larger frames do not "
                "have: pixels along the edge of a light-coloured part, or along a light-coloured thin part, that are "
                "darker than both that part's own colour in the larger frames and the grey background? NOT a dark "
                "fringe: (1) a dark outline or dark colour that the same part also has in the larger frames; (2) an "
                "edge or thin part that looks paler, greyer or fainter than in the larger frames, blending toward the "
                "grey; (3) a part that is missing or broken in places.\n\n"
                "Answer DARK-FRINGE if you see such a fringe, CLEAN if you do not."
            ),
            frames=(*_SHRINK_SHRUNK, *_SHRINK_SPECIFICITY),
            answers=("DARK-FRINGE", "CLEAN", "CANNOT-TELL"),
        ),
        rp.Check(
            id="large_figures",
            question=(
                f"Frames {large} show three figures and a market building. Is any thin straight part in them, such as "
                "the bow, the spear's shaft, the hoe's handle, or the market's rope, poles and railings, drawn broken "
                "into a row of separate dashes or dots by gaps of grey background, where the object itself would be "
                "one continuous piece? Not counted: a part that changes in thickness or colour along its length, a "
                "part passing behind another part, and gaps between separate objects such as stalls, crates or "
                "barrels.\n\n"
                "Answer PRESENT if you see such a broken part, NONE if you do not."
            ),
            frames=_SHRINK_SPECIFICITY,
            answers=("PRESENT", "NONE", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id="sprite_shrink",
        title="Figure detail review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from a map editor for an isometric strategy game: game figures drawn over a "
            "flat grey ground. Each frame is enlarged by a whole-number factor with no smoothing, so one original "
            "pixel appears as a square block of pixels. Judge only what you can actually see."
        ),
        injects=tuple(SHRINK_INJECTS),
    )


def _shrink_prerequisites() -> str | None:
    if asset_source.get_install_path() is None:
        return "needs a configured AoE2:DE install for the unit and building art"
    return None


def _shrink_luma(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def _shrink_edges() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per shrunk frame, two masks from the clean render: its edge px (figure px with 0 < alpha
    < 255, the only px premultiplying changes) and the light ones among them, whose own colour
    is at least SHRINK_LIGHT above the ground."""
    masks = {}
    with _shrink_inject(None):
        for fid in _SHRINK_SHRUNK:
            frame, figures = _shrink_lineup(fid)
            edge = np.zeros(frame.shape[:2], dtype=bool)
            light = np.zeros(frame.shape[:2], dtype=bool)
            for rgba, x0, y0 in figures:
                at = (slice(y0, y0 + rgba.shape[0]), slice(x0, x0 + rgba.shape[1]))
                partial = (rgba[..., 3] > 0) & (rgba[..., 3] < 255)
                edge[at] |= partial
                light[at] |= partial & (_shrink_luma(rgba[..., :3]) >= _SHRINK_GROUND + SHRINK_LIGHT)
            masks[fid] = (edge, light)
    return masks


def _shrink_components(inject: str | None) -> dict[str, int]:
    """Per shrunk frame, ink (alpha > 0) components summed over its figures."""
    with _shrink_inject(inject):
        return {
            fid: sum(rp.components(rgba[..., 3] > 0) for rgba, _x0, _y0 in _shrink_lineup(fid)[1])
            for fid in _SHRINK_SHRUNK
        }


def _shrink_fringe(frame: np.ndarray, light: np.ndarray) -> float:
    """The fraction of the light edge px rendered at least SHRINK_FRINGE_DELTA below the ground."""
    if not light.any():
        return 0.0
    return float(np.mean(_shrink_luma(frame)[light] <= _SHRINK_GROUND - SHRINK_FRINGE_DELTA))


def _check_sprite_shrink() -> None:
    """--check for sprite_shrink: spec coupling, frame band, both 1x frames untouched by every
    inject, each inject reaching every shrunk frame, straight-average moving nothing off the edge
    px beyond float rounding and staying under nearest's largest delta; then the symptom each
    check rests on, measured per shrunk frame. thin_parts: nearest breaks the ink into more
    components than the clean render, straight-average keeps the clean count (its alpha is the
    same). edge_tone: the clean render puts no light edge px below the ground (_shrink_fringe(),
    0 by construction), nearest none either, straight-average at least SHRINK_STRAIGHT_FRINGE_MIN."""
    reason = _shrink_prerequisites()
    if reason:
        raise SystemExit(f"sprite_shrink: {reason}")
    failures: list[str] = []
    spec = _sprite_shrink_spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    clean = _shrink_frame_images(None)
    for frame in spec.frames:
        try:
            scaled, _factor = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))

    edges = _shrink_edges()
    images = {None: clean}
    reach: dict[str, dict[str, tuple[int, int]]] = {}
    for name, bound in SHRINK_INJECTS.items():
        injected = images[name] = _shrink_frame_images(name)
        touched = {}
        for fid in _SHRINK_FRAMES:
            mask = rp.changed_pixels(clean[fid], injected[fid])
            if mask.any():
                touched[fid] = (int(np.count_nonzero(mask)), rp.largest_component_px(mask))
            if bound is not None and fid in edges:
                # Off the edge px, only float rounding may move a byte.
                delta = np.abs(clean[fid].astype(int) - injected[fid].astype(int))[~edges[fid][0]]
                if delta.size and delta.max() > SHRINK_STRAY_DELTA_MAX:
                    failures.append(f"inject {name} on {fid}: moves a px off the edge px by {delta.max()}")
        if stray := [fid for fid in _SHRINK_SPECIFICITY if fid in touched]:
            failures.append(f"inject {name} changed the 1x frames {', '.join(stray)}, where no filter should")
        if missed := [fid for fid in _SHRINK_SHRUNK if fid not in touched]:
            failures.append(f"inject {name}: changed nothing in {', '.join(missed)}")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    coarse = reach[SHRINK_COARSE_INJECT]
    for name, bound in SHRINK_INJECTS.items():
        if bound is None:
            continue
        for fid, (_px, comp) in sorted(reach[name].items()):
            if fid in coarse and comp >= coarse[fid][1]:
                failures.append(
                    f"inject {name} on {fid}: largest delta {comp} px is not under the coarse inject's "
                    f"{coarse[fid][1]} px there, so the two no longer differ in size"
                )

    components = {name: _shrink_components(name) for name in (None, *SHRINK_INJECTS)}
    gated: set[int] = set()
    for fid in _SHRINK_SHRUNK:
        c_count, n_count, s_count = (components[name][fid] for name in (None, "nearest", "straight-average"))
        light = edges[fid][1]
        c_fringe, n_fringe, s_fringe = (
            _shrink_fringe(images[name][fid], light) for name in (None, "nearest", "straight-average")
        )
        print(
            f"{fid}: ink components clean {c_count} / nearest {n_count} / straight-average {s_count}; light edge px "
            f"{int(light.sum())}, below the ground: clean {c_fringe:.2f} / nearest {n_fringe:.2f} / "
            f"straight-average {s_fringe:.2f}"
        )
        if n_count <= c_count:
            failures.append(f"{fid}: nearest leaves {n_count} ink components, not more than the clean {c_count}")
        if s_count != c_count:
            failures.append(f"{fid}: straight-average moved the ink components {c_count} -> {s_count}; its alpha should not")
        if c_fringe:
            failures.append(f"{fid}: the clean render puts {c_fringe:.2f} of the light edge px below the ground")
        if light.sum() >= SHRINK_LIGHT_MIN_PX:
            gated.add(_SHRINK_FRAMES[fid][1])
            if s_fringe < SHRINK_STRAIGHT_FRINGE_MIN:
                failures.append(
                    f"{fid}: straight-average puts {s_fringe:.2f} of the light edge px below the ground, "
                    f"under {SHRINK_STRAIGHT_FRINGE_MIN}"
                )
    if missing := {_SHRINK_FRAMES[fid][1] for fid in _SHRINK_SHRUNK} - gated:
        failures.append(f"half_w {sorted(missing)}: no frame has {SHRINK_LIGHT_MIN_PX} light edge px to gate edge_tone on")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(
        f"OK: sprite_shrink frames couple to its checks and land in {rp.MIN_LONG_EDGE}-{rp.MAX_LONG_EDGE} px; no inject "
        f"touches the 1x frames; nearest breaks the ink into more components on every shrunk frame; straight-average "
        f"keeps the components, puts light edge px below the ground where the clean render puts none, moves nothing "
        f"else, and stays under nearest's largest delta"
    )


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
# flat elevation 0 with fields, walls and buildings (the specificity frame), off round 2's footprint ring.
_SLOPED_FRAMES = {
    "hill_a": ("ground", -1, (219, 152), False),
    "hill_b": ("ground", -2, (219, 152), False),
    "forest_a": ("ground", -1, (45, 142), True),
    "forest_b": ("ground", -2, (45, 142), True),
    "farms_a": ("farms", -1, (89, 215), True),
    "farms_b": ("farms", -2, (89, 215), True),
    "buildings_a": ("ground", -1, (22, 214), True),
    "buildings_b": ("ground", -2, (22, 214), True),
    "level_ground": ("farms", -2, (140, 44), True),
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


def _level_px(n: int, mip: int) -> int:
    """`n` level-0 pixels at level `mip`: doubled per finer level, halved per coarser one."""
    return n << mip if mip >= 0 else n >> -mip


def _sloped_box(cache, mip: int, centre, crop_l0=_SLOPED_CROP_L0) -> tuple[int, int, int, int]:
    """The level-`mip` pixel crop of `crop_l0` level-0 pixels centred on a tile's own surface."""
    lvl = cache._levels[mip]
    cx, cy = centre
    sx, sy = ig.tile_screen_origin(cx, cy, int(cache.elevations[cy, cx]), lvl.proj)
    cw, ch = (_level_px(n, mip) for n in crop_l0)
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

        self.saved = [
            (SlopedChunkCache, "_level_rise"),
            (render, "composite_rect_sloped"),
            (render, "_resolve_unit_sprite"),
            (render, "_paint_terrain_sloped"),
        ]
        self.saved = [(obj, name, getattr(obj, name)) for obj, name in self.saved]
        real_rise, real_composite, real_resolve, real_terrain = (old for _o, _n, old in self.saved)
        # (scenario id, tile_px) -> level 0's corner_rise, for the coarsest level only.
        coarsest: dict[tuple, np.ndarray] = {}

        def note_coarsest(cache, lvl):
            lvl = real_rise(cache, lvl)
            if lvl.tile_px * 4 == cache.tile_px:
                coarsest[(id(cache.scenario), lvl.tile_px)] = cache.corner_rise
            return lvl

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
        elif self.name == "drape-rise":
            drape_rise: list[np.ndarray] = []  # the wrong rise while a coarsest-level composite runs

            def terrain(img, tile, tile_px, proj, corner_rise, map_w, map_h, *args, **kwargs):
                sprites = kwargs.get("sprites")
                if not drape_rise or sprites is None or (tile.x, tile.y) not in sprites.farm_by_tile:
                    return real_terrain(img, tile, tile_px, proj, corner_rise, map_w, map_h, *args, **kwargs)
                # Bare ground where the field belongs, then the field against level 0's rise.
                real_terrain(img, tile, tile_px, proj, corner_rise, map_w, map_h, *args, **{**kwargs, "sprites": None})
                return real_terrain(img, tile, tile_px, proj, drape_rise[0], map_w, map_h, *args, **kwargs)

            def composite(scenario, x0, y0, x1, y1, corner_rise, proj, *args, **kwargs):
                wrong = coarsest.get((id(scenario), proj.tile_px))
                if wrong is None:
                    return real_composite(scenario, x0, y0, x1, y1, corner_rise, proj, *args, **kwargs)
                # Taller by the largest lift, so a field drawn up into this rect from below it is a candidate.
                lift = max(0, int((wrong - corner_rise).max()))
                drape_rise.append(wrong)
                try:
                    out = real_composite(
                        scenario, x0, y0, x1, y1 + lift, corner_rise, proj, *args, **{**kwargs, "unit_pack": None}
                    )
                finally:
                    drape_rise.clear()
                return out[: y1 - y0]

            SlopedChunkCache._level_rise = note_coarsest
            render.composite_rect_sloped = composite
            render._paint_terrain_sloped = terrain
        elif self.name == "building-step":
            stepped: dict[int, np.ndarray] = {}  # id(true rise) -> stepped rise; the level keeps the key alive

            def resolve(scenario, proj, elevations, unit_filter, corner_rise, *args, **kwargs):
                unit = args[4]
                if (
                    corner_rise is not None
                    and (id(scenario), proj.tile_px) in coarsest
                    and unit.unit_const in unit_kind.building_consts()
                ):
                    # Buildings on raised ground stand one elev_step high; terrain and fields keep their rise.
                    if id(corner_rise) not in stepped:
                        lift = proj.elev_step * _SLOPED_BUILDING_STEPS
                        stepped[id(corner_rise)] = np.where(corner_rise > 0, corner_rise + lift, corner_rise)
                    corner_rise = stepped[id(corner_rise)]
                return real_resolve(scenario, proj, elevations, unit_filter, corner_rise, *args, **kwargs)

            SlopedChunkCache._level_rise = note_coarsest
            render._resolve_unit_sprite = resolve
        return self

    def __exit__(self, *exc):
        for obj, name, old in self.saved:
            setattr(obj, name, old)


SLOPED_INJECTS = {"level0-rise": None, "chunk-row-shift": "one-chunk", "drape-rise": None, "building-step": "one-step"}
SLOPED_COARSE_INJECT = "level0-rise"
# building-step's lift in elev_steps; its bound in --check stays one elev_step.
_SLOPED_BUILDING_STEPS = 1
# The frames each inject must change, and no others. level0-rise moves every coarsest-level
# object, drape-rise the coarsest level's farm drapes, building-step its buildings on raised ground.
_SLOPED_REACH = {
    "level0-rise": ("forest_b", "farms_b", "buildings_b"),
    "chunk-row-shift": _SLOPED_ROW_SHIFT_FRAMES,
    "drape-rise": ("farms_b",),
    "building-step": ("farms_b", "buildings_b"),
}


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
        "level_ground": "Buildings, farm fields and other objects on perfectly level ground, zoomed far out.",
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
                "Then the closed question, which is about the GROUND only. Is there an exactly vertical straight "
                "edge in the ground where the ground on one side does not continue on the other: the ground on "
                "one side is clearly lighter or darker than on the other along the edge's whole length, a "
                "lighter or darker band or patch stops dead against it instead of carrying on, the ground is "
                "offset up or down across it, or a strip of black shows through? Any such edge counts, however "
                "short: one no longer than a single map cell counts as much as one crossing the whole frame. "
                "NOT such an edge: (1) any exactly horizontal edge: at this zoom the outline of a lighter or "
                "darker band running along the slope is drawn in short horizontal steps, and those steps are "
                "expected; (2) a vertical speck or streak in the ground's own grain, where the ground just "
                "beside it is equally bright on both sides; (3) the border between grassland and forest floor, "
                "which follows the diagonal cell grid and wanders; (4) trees and anything else standing on the "
                "ground, however they are placed; (5) the frame's own edges.\n\n"
                "Answer BREAK if at least one such vertical edge is visible, CONTINUOUS if the ground carries "
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
                "This frame shows perfectly level ground with buildings, farm fields and other objects on it. "
                "Is anything wrong with how it is drawn: any object or field detached from the ground under "
                "it, or any straight horizontal or vertical line where the ground on one side is offset, cut "
                "off or does not continue on the other? The border between two kinds of ground, which follows "
                "the diagonal cell grid, does not count, and neither do the thin coloured lines drawn on purpose "
                "along the edges of farm fields.\n\n"
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


def _sloped_masks(frame_id: str, frames=_SLOPED_FRAMES, crop_l0=_SLOPED_CROP_L0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(object ink, farm field, ink band) for one clean frame at 1:1: every
    pixel a sprite draws with nonzero alpha, every pixel whose topmost
    terrain tile is a farm drape's, and the ink plus the elev_step above it
    (ink just below the frame included, which a lift carries into it).
    `frames` and `crop_l0` are the pack's (sloped_mip's by default)."""
    map_key, mip, centre, units = frames[frame_id]
    with _PinnedSettings():
        cache = _sloped_cache(map_key, units)
        x0, y0, x1, y1 = box = _sloped_box(cache, mip, centre, crop_l0)
        lvl = cache._level(mip)
        step = lvl.proj.elev_step
        ink = np.zeros((y1 - y0 + step, x1 - x0), dtype=bool)
        field = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        if not lvl.sprites:
            return ink[:-step], field, ink[:-step]
        for draws in lvl.sprites.by_anchor.values():
            for draw, ax, ay, _rise in draws:
                alpha = draw.rgba[..., 3] > 0
                top, left = ay - draw.hotspot_y - y0, ax - draw.hotspot_x - x0
                sy0, sx0 = max(0, -top), max(0, -left)
                sy1, sx1 = min(alpha.shape[0], ink.shape[0] - top), min(alpha.shape[1], ink.shape[1] - left)
                if sy1 > sy0 and sx1 > sx0:
                    ink[top + sy0 : top + sy1, left + sx0 : left + sx1] |= alpha[sy0:sy1, sx0:sx1]
        if lvl.sprites.farm_by_tile:
            map_w = cache.scenario.map_manager.map_width
            ids = render.composite_ids_rect_sloped(
                cache.scenario, *box, cache._level_corner_rise(lvl), lvl.proj, lvl.tile_px
            )
            field = np.isin(ids, [ty * map_w + tx for tx, ty in lvl.sprites.farm_by_tile])
    band = ink[:-step] | ink[step:]
    return ink[:-step], field, band


def _check_sloped_mip() -> None:
    """--check for sloped_mip: spec coupling, frame band, the clean frames
    byte-identical at the app's chunk size, each inject's reach (printed) and
    exactly its own frames, none reaching the specificity frame,
    chunk-row-shift one chunk-bounded piece per hillside frame, drape-rise
    moving fields but no object pixel, and building-step inside one
    elev_step of object ink. Both localized injects stay under level0-rise's
    largest delta on every frame they share."""
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
    masks = {fid: _sloped_masks(fid) for fid in sorted({*_SLOPED_REACH["drape-rise"], *_SLOPED_REACH["building-step"]})}
    reach: dict[str, dict[str, tuple[int, int]]] = {}
    for name, bound in SLOPED_INJECTS.items():
        injected = _sloped_frame_images(name)
        touched, shown = {}, []
        for frame_id in _SLOPED_FRAMES:
            mask = rp.changed_pixels(clean[frame_id], injected[frame_id])
            changed = int(np.count_nonzero(mask))
            where, note = f"inject {name} on {frame_id}", ""
            if bound == "one-chunk" and frame_id in _SLOPED_ROW_SHIFT_FRAMES:
                failures += one_chunk.failures(mask, where)
            if name == "drape-rise" and frame_id in _SLOPED_REACH[name]:
                # Fields leave their ground; no object pixel moves with them.
                ink, field, _band = masks[frame_id]
                field = field & ~ink
                moved, on_ink = int(np.count_nonzero(mask & field)), int(np.count_nonzero(mask & ink))
                note = f", fields {moved}/{int(field.sum())}px, object ink {on_ink}/{int(ink.sum())}px"
                if moved < 0.25 * np.count_nonzero(field):
                    failures.append(f"{where}: moved {moved} of {int(field.sum())} field px, under a quarter")
                if on_ink > 0.01 * np.count_nonzero(ink):
                    failures.append(f"{where}: changed {on_ink} object px, over 1% of {int(ink.sum())}")
            elif name == "building-step" and frame_id in _SLOPED_REACH[name]:
                # A building one elev_step high changes only its own ink and the step above it.
                band = masks[frame_id][2]
                outside = int(np.count_nonzero(mask & ~band))
                note = f", outside one elev_step of ink {outside}px"
                if outside:
                    failures.append(f"{where}: {outside} px changed outside one elev_step of object ink")
            if changed:
                touched[frame_id] = (changed, rp.largest_component_px(mask))
                delta = np.abs(clean[frame_id].astype(np.int16) - injected[frame_id]).sum(axis=-1).max()
                shown.append(
                    f"{frame_id} {changed}px bbox {rp.delta_bbox(mask)} largest {touched[frame_id][1]}px "
                    f"max delta {delta}{note}"
                )
        if _SLOPED_SPECIFICITY in touched:
            failures.append(f"inject {name}: reached the specificity frame {_SLOPED_SPECIFICITY}")
        want = set(_SLOPED_REACH[name])
        if missed := sorted(want - set(touched)):
            failures.append(f"inject {name}: changed nothing in {', '.join(missed)}")
        if stray := sorted(set(touched) - want):
            failures.append(f"inject {name}: reached {', '.join(stray)}, outside its frames")
        reach[name] = touched
        print(f"reach: {name} -> {'; '.join(shown) or 'nothing'}")

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
        f"chunk_px {DEFAULT_CHUNK_PX}; all {len(SLOPED_INJECTS)} injects bite exactly their own frames, "
        f"the two localized ones inside their bounds and under {SLOPED_COARSE_INJECT}, drape-rise moves "
        f"fields and no object, and none reaches {_SLOPED_SPECIFICITY}"
    )


# --------------------------------------------------------- sloped_fine pack


# TASK-146's finer level (+1, 128 px a tile). One footprint of this many level-0 pixels per scene,
# so a +1 frame is 1500x750 (inside MAX_LONG_EDGE, never downscaled) and its level-0 twin 750x375.
_FINE_CROP_L0 = (750, 375)
# scene -> (map, centre tile, units and sprites on): sloped_mip's scenes, the level ground moved
# off the footprint ring its round 2 flagged (elevation 0 for 12 tiles each way: fields, walls, buildings).
_FINE_SCENES = {
    "hill": ("ground", (219, 152), False),
    "forest": ("ground", (45, 142), True),
    "farms": ("farms", (89, 215), True),
    "buildings": ("ground", (22, 214), True),
    "level_ground": ("farms", (140, 44), True),
}
# frame id -> (map, mip, centre tile, units): each scene at level 0 (`_a`) and at +1 (`_b`).
_FINE_FRAMES = {
    f"{scene}_{tag}": (map_key, mip, centre, units)
    for scene, (map_key, centre, units) in _FINE_SCENES.items()
    for tag, mip in (("a", 0), ("b", 1))
}
_FINE_SPECIFICITY = ("level_ground_a", "level_ground_b")
# block-shift lifts one _FINE_BLOCK_PX square of +1 chunks in each of these frames.
_FINE_BLOCK_FRAMES = ("hill_b", "forest_b")
_FINE_BLOCK_PX = 512
# Every inject's lift, in +1 elev_steps: 160 px on screen at 1x, past the 126-168 px round 1 resolved.
_FINE_LIFT_STEPS = 10


class _FineInject:
    """Patches for one known +1 defect, applied around the cache builds and
    renders of a whole run. Each reaches only level +1 of a cache."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.block_chunks: set[tuple[int, int, int, int]] = set()  # (scenario id, tile_px, x0, y0)

    def __enter__(self):
        from descape.render_cache import SlopedChunkCache

        self.saved = [(SlopedChunkCache, "_level_rise"), (render, "composite_rect_sloped"), (render, "_paint_terrain_sloped")]
        self.saved = [(obj, name, getattr(obj, name)) for obj, name in self.saved]
        real_rise, real_composite, real_terrain = (old for _o, _n, old in self.saved)
        # (scenario id, +1 tile_px) -> (+1's true corner_rise, the same lifted wherever the ground is raised).
        fine: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

        def lifted(cache, lvl) -> tuple[np.ndarray, np.ndarray]:
            key = (id(cache.scenario), lvl.tile_px)
            if key not in fine:
                true = ig.corner_rise_px(cache.elevations, lvl.proj, rule=render.SLOPE_CORNER_RULE)
                fine[key] = true, np.where(true > 0, true + lvl.proj.elev_step * _FINE_LIFT_STEPS, true)
            return fine[key]

        if self.name == "fine-rise":

            def level_rise(cache, lvl):
                lvl = real_rise(cache, lvl)
                if lvl.tile_px == 2 * cache.tile_px:
                    # +1's unit layers read a rise lifted on raised ground; its terrain keeps the true one.
                    wrong = lifted(cache, lvl)[1]
                    lvl.headroom = render._unit_rise_headroom_px(wrong, cache.elevations, lvl.proj)
                    lvl.corner_rise = wrong
                return lvl

            def composite(scenario, x0, y0, x1, y1, corner_rise, proj, *args, **kwargs):
                if (key := (id(scenario), proj.tile_px)) in fine:
                    corner_rise = fine[key][0]
                return real_composite(scenario, x0, y0, x1, y1, corner_rise, proj, *args, **kwargs)

            SlopedChunkCache._level_rise = level_rise
            render.composite_rect_sloped = composite
        elif self.name == "block-shift":

            def composite(scenario, x0, y0, x1, y1, corner_rise, proj, *args, **kwargs):
                if (id(scenario), proj.tile_px, x0, y0) in self.block_chunks:
                    # A block of +1 ground drawn ten elev_steps above where it lies.
                    corner_rise = corner_rise + proj.elev_step * _FINE_LIFT_STEPS
                return real_composite(scenario, x0, y0, x1, y1, corner_rise, proj, *args, **kwargs)

            render.composite_rect_sloped = composite
        elif self.name == "fine-drape":
            drape_rise: list[np.ndarray] = []  # the lifted rise while a +1 composite runs

            def note_fine(cache, lvl):
                lvl = real_rise(cache, lvl)
                if lvl.tile_px == 2 * cache.tile_px:
                    lifted(cache, lvl)
                return lvl

            def terrain(img, tile, tile_px, proj, corner_rise, map_w, map_h, *args, **kwargs):
                sprites = kwargs.get("sprites")
                if not drape_rise or sprites is None or (tile.x, tile.y) not in sprites.farm_by_tile:
                    return real_terrain(img, tile, tile_px, proj, corner_rise, map_w, map_h, *args, **kwargs)
                # Bare ground where the field belongs, then the field against the lifted rise.
                real_terrain(img, tile, tile_px, proj, corner_rise, map_w, map_h, *args, **{**kwargs, "sprites": None})
                return real_terrain(img, tile, tile_px, proj, drape_rise[0], map_w, map_h, *args, **kwargs)

            def composite(scenario, x0, y0, x1, y1, corner_rise, proj, *args, **kwargs):
                entry = fine.get((id(scenario), proj.tile_px))
                if entry is None:
                    return real_composite(scenario, x0, y0, x1, y1, corner_rise, proj, *args, **kwargs)
                wrong = entry[1]
                # Taller by the lift, so a field drawn up into this rect from below it is a candidate.
                lift = max(0, int((wrong - corner_rise).max()))
                drape_rise.append(wrong)
                try:
                    out = real_composite(
                        scenario, x0, y0, x1, y1 + lift, corner_rise, proj, *args, **{**kwargs, "unit_pack": None}
                    )
                finally:
                    drape_rise.clear()
                return out[: y1 - y0]

            SlopedChunkCache._level_rise = note_fine
            render.composite_rect_sloped = composite
            render._paint_terrain_sloped = terrain
        return self

    def __exit__(self, *exc):
        for obj, name, old in self.saved:
            setattr(obj, name, old)


FINE_INJECTS = {"fine-rise": None, "block-shift": "one-block", "fine-drape": None}
FINE_COARSE_INJECT = "fine-rise"
# The frames each inject must change, and no others: fine-rise lifts every +1 object on raised
# ground, block-shift one block of ground per hillside frame, fine-drape the +1 farm fields.
_FINE_REACH = {
    "fine-rise": ("forest_b", "farms_b", "buildings_b"),
    "block-shift": _FINE_BLOCK_FRAMES,
    "fine-drape": ("farms_b",),
}


def _fine_block(cache, box) -> set[tuple[int, int, int, int]]:
    """The +1 chunks block-shift lifts in this crop: every chunk whose rect lies
    inside the _FINE_BLOCK_PX square centred on it, keyed as the inject reads them."""
    x0, y0, x1, y1 = box
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    c, half = cache.chunk_px, _FINE_BLOCK_PX // 2
    tile_px = cache._levels[1].tile_px
    return {
        (id(cache.scenario), tile_px, cx * c, cy * c)
        for cy in range(-(-(my - half) // c), (my + half) // c)
        for cx in range(-(-(mx - half) // c), (mx + half) // c)
    }


def _fine_frame_images(inject: str | None, chunk_px: int = SLOPED_CHUNK_PX) -> dict[str, np.ndarray]:
    """Every sloped_fine frame at 1:1, each stitched from a fresh cache's chunks
    through SlopedChunkCache.render_rect(mip=...), never a direct render."""
    images: dict[str, np.ndarray] = {}
    with _PinnedSettings():
        caches = {}
        for map_key, _mip, _centre, units in _FINE_FRAMES.values():
            if (map_key, units) not in caches:
                caches[(map_key, units)] = _sloped_cache(map_key, units, chunk_px)
        patch = _FineInject(inject) if inject else None
        if inject == "block-shift":
            for frame_id in _FINE_BLOCK_FRAMES:
                map_key, mip, centre, units = _FINE_FRAMES[frame_id]
                cache = caches[(map_key, units)]
                patch.block_chunks |= _fine_block(cache, _sloped_box(cache, mip, centre, _FINE_CROP_L0))
        if patch is not None:
            patch.__enter__()
        try:
            for frame_id, (map_key, mip, centre, units) in _FINE_FRAMES.items():
                cache = caches[(map_key, units)]
                images[frame_id] = cache.render_rect(*_sloped_box(cache, mip, centre, _FINE_CROP_L0), mip=mip)
        finally:
            if patch is not None:
                patch.__exit__(None, None, None)
    return images


def _sloped_fine_spec() -> rp.PackSpec:
    """TASK-146's finer Sloped level: each scene at level +1 (`_b`) beside its
    level-0 twin (`_a`) of the same tiles, at the same on-screen size. The
    reviewer is told both are zoomed in and that each scene appears twice."""
    places = {
        "hill": "Grassland on a hillside",
        "forest": "Forest and grassland on a hillside, with trees standing on it",
        "farms": "Farm fields and buildings at the edge of a raised plateau",
        "buildings": "A group of buildings on and around a raised plateau",
        "level_ground": "Buildings, farm fields and other objects on perfectly level ground",
    }
    captions = {}
    for scene, place in places.items():
        captions[f"{scene}_a"] = f"{place}, zoomed in."
        captions[f"{scene}_b"] = "The same ground again, at the same size."
    frames = tuple(rp.Frame(fid, f"{fid}.png", "raw", caption=captions[fid]) for fid in _FINE_FRAMES)
    checks = (
        rp.Check(
            id="ground_continuity",
            question=(
                "These four frames show bare and forested hillsides. The ground is a grid of diamond-shaped "
                "map cells at different heights, drawn as one continuous sloping surface. First, in your own "
                "words, describe how the ground runs across each frame: its slopes, its texture, and any place "
                "where it looks interrupted.\n\n"
                "Then the closed question, which is about the GROUND only. Is there a straight edge in the "
                "ground, exactly vertical or exactly horizontal, where the ground on one side does not continue "
                "on the other: its texture, shading or slope jumps, is offset up or down, or is cut off, or a "
                "strip of black shows through? Any such edge counts, however short: one no longer than a single "
                "map cell counts as much as one crossing the whole frame. NOT such an edge: (1) the border "
                "between grassland and forest floor, which follows the diagonal cell grid and wanders; (2) a "
                "change of shading where the slope bends, which runs along the diagonal grid and never forms an "
                "exactly vertical or horizontal edge; (3) a lighter or darker patch inside the ground's own "
                "texture, with the same height and slope on both sides of it; (4) trees and anything else "
                "standing on the ground, however they are placed; (5) the frame's own edges.\n\n"
                "Answer BREAK if at least one such straight edge is visible, CONTINUOUS if the ground carries "
                "on across every frame."
            ),
            frames=("hill_a", "hill_b", "forest_a", "forest_b"),
            answers=("BREAK", "CONTINUOUS", "CANNOT-TELL"),
        ),
        rp.Check(
            id="objects_on_ground",
            question=(
                "These frames show buildings, trees and other objects standing on sloping ground; each scene "
                "appears twice, as a frame ending _a and its twin ending _b. First, in your own words, describe "
                "how each kind of object meets the ground under it: where its base is, and what ground shows "
                "around that base.\n\n"
                "Then the closed question. Does every object sit on its own footprint, the patch of ground "
                "directly under its base? Or does any object look detached from it: floating clear above it "
                "with ground or a gap showing under its base, sunk into it, or standing away from the patch "
                "it plainly belongs to, so it reads as pasted onto the wrong place? NOT detached: (1) a tall "
                "object reaching up over the ground and objects behind it, which every tall object does; "
                "(2) dark shadows beside objects; (3) objects too small to tell where their base is.\n\n"
                "Answer DETACHED if at least one object looks detached, SEATED if every judgeable object sits on "
                "its own footprint."
            ),
            frames=("forest_a", "forest_b", "farms_a", "farms_b", "buildings_a", "buildings_b"),
            answers=("DETACHED", "SEATED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="fields_on_slope",
            question=(
                "Look at the farm fields: flat, ploughed or planted patches laid on the ground, some of them "
                "on the slope at the plateau's edge. First, in your own words, describe each field: its "
                "outline, how it bends with the ground, and what its edges meet.\n\n"
                "Then the closed question. Does every field follow the slope of the ground under it, lying on "
                "its own patch of ground with its edges meeting the surrounding ground? Or does any field sit "
                "off its ground: raised above it or sunk into it, shifted across neighbouring cells, or bent "
                "in a way the ground around it is not? NOT off the ground: buildings or other objects standing "
                "on or in front of a field and hiding part of it.\n\n"
                "Answer ON-GROUND if every field follows its ground, OFF-GROUND if any field does not."
            ),
            frames=("farms_a", "farms_b"),
            answers=("ON-GROUND", "OFF-GROUND", "CANNOT-TELL"),
        ),
        rp.Check(
            id="level_ground",
            question=(
                "These two frames show the same perfectly level ground with buildings, farm fields and other "
                "objects on it. Is anything wrong with how either is drawn: any object or field detached from "
                "the ground under it, or any straight horizontal or vertical line where the ground on one side "
                "is offset, cut off or does not continue on the other? NOT such a flaw: (1) the border between "
                "two kinds of ground, which follows the diagonal cell grid; (2) the thin coloured lines drawn on "
                "purpose along the edges of farm fields; (3) the straight sides of walls, towers, buildings and "
                "other objects standing on the ground, which are part of the object, not the ground.\n\n"
                "Answer FLAWS-PRESENT if you see any of these, NO-FLAWS if neither frame shows any."
            ),
            frames=_FINE_SPECIFICITY,
            answers=("FLAWS-PRESENT", "NO-FLAWS", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id="sloped_fine",
        title="Zoomed-in terrain review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from a map editor for an isometric strategy game, zoomed in, so a map "
            "cell is over a hundred pixels across. Each scene appears twice, as a frame ending _a and one ending "
            "_b: the same patch of ground shown at the same size, one of them possibly sharper or blockier than "
            "the other, which alone is not a flaw. Each frame is a crop, enlarged by a whole-number factor with "
            "no smoothing where it is enlarged at all. Judge only what you can actually see."
        ),
        injects=tuple(FINE_INJECTS),
    )


def _check_sloped_fine() -> None:
    """--check for sloped_fine: spec coupling, frame band, the clean frames
    byte-identical at the app's chunk size, real sprites and farm drapes at
    +1, each inject's reach (printed) and exactly its own frames, none
    reaching the specificity frames or any level-0 twin, block-shift inside
    one block per hillside frame, fine-drape moving fields but no object
    pixel, and the coarse inject larger than block-shift wherever both reach."""
    from descape.render_cache import DEFAULT_CHUNK_PX

    reason = _sloped_prerequisites()
    if reason:
        raise SystemExit(f"sloped_fine: {reason}")
    failures: list[str] = []
    spec = _sloped_fine_spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")

    clean = _fine_frame_images(None)
    for frame in spec.frames:
        try:
            scaled, _factor = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))
    for scene in _FINE_SCENES:
        a, b = clean[f"{scene}_a"], clean[f"{scene}_b"]
        if (b.shape[0], b.shape[1]) != (2 * a.shape[0], 2 * a.shape[1]):
            failures.append(f"scene {scene}: +1 frame {b.shape[:2]} is not twice its level-0 twin {a.shape[:2]}")
    app_chunks = _fine_frame_images(None, DEFAULT_CHUNK_PX)
    for frame_id, img in clean.items():
        if not np.array_equal(img, app_chunks[frame_id]):
            failures.append(f"frame {frame_id}: differs from the same crop stitched at chunk_px {DEFAULT_CHUNK_PX}")
    with _PinnedSettings():
        for frame_id, (map_key, mip, _centre, units) in _FINE_FRAMES.items():
            if not units:
                continue
            lvl = _sloped_cache(map_key, units)._level(mip)
            if not lvl.sprites or not lvl.sprites.by_anchor:
                failures.append(f"frame {frame_id}: level {mip} built no sprites")
            elif frame_id.startswith("farms") and not lvl.sprites.farm_by_tile:
                failures.append(f"frame {frame_id}: level {mip} built no farm drape")

    one_block = rp.LocalizedBound(max_changed_fraction=0.5, max_component_px=_FINE_BLOCK_PX**2)
    masks = {fid: _sloped_masks(fid, _FINE_FRAMES, _FINE_CROP_L0) for fid in _FINE_REACH["fine-drape"]}
    reach: dict[str, dict[str, tuple[int, int]]] = {}
    for name, bound in FINE_INJECTS.items():
        injected = _fine_frame_images(name)
        touched, shown = {}, []
        for frame_id in _FINE_FRAMES:
            mask = rp.changed_pixels(clean[frame_id], injected[frame_id])
            changed = int(np.count_nonzero(mask))
            where, note = f"inject {name} on {frame_id}", ""
            if bound == "one-block" and frame_id in _FINE_BLOCK_FRAMES:
                failures += one_block.failures(mask, where)
            if name == "fine-drape" and frame_id in _FINE_REACH[name]:
                # Fields leave their ground; no object pixel moves with them.
                ink, field, _band = masks[frame_id]
                field = field & ~ink
                moved, on_ink = int(np.count_nonzero(mask & field)), int(np.count_nonzero(mask & ink))
                note = f", fields {moved}/{int(field.sum())}px, object ink {on_ink}/{int(ink.sum())}px"
                if moved < 0.25 * np.count_nonzero(field):
                    failures.append(f"{where}: moved {moved} of {int(field.sum())} field px, under a quarter")
                if on_ink > 0.01 * np.count_nonzero(ink):
                    failures.append(f"{where}: changed {on_ink} object px, over 1% of {int(ink.sum())}")
            if changed:
                touched[frame_id] = (changed, rp.largest_component_px(mask))
                delta = np.abs(clean[frame_id].astype(np.int16) - injected[frame_id]).sum(axis=-1).max()
                shown.append(
                    f"{frame_id} {changed}px bbox {rp.delta_bbox(mask)} largest {touched[frame_id][1]}px "
                    f"max delta {delta}{note}"
                )
        if hit := sorted(set(touched) & set(_FINE_SPECIFICITY)):
            failures.append(f"inject {name}: reached the specificity frames {', '.join(hit)}")
        want = set(_FINE_REACH[name])
        if missed := sorted(want - set(touched)):
            failures.append(f"inject {name}: changed nothing in {', '.join(missed)}")
        if stray := sorted(set(touched) - want):
            failures.append(f"inject {name}: reached {', '.join(stray)}, outside its frames")
        reach[name] = touched
        print(f"reach: {name} -> {'; '.join(shown) or 'nothing'}")

    coarse = reach[FINE_COARSE_INJECT]
    for frame_id, (_px, comp) in sorted(reach["block-shift"].items()):
        if frame_id in coarse and comp >= coarse[frame_id][1]:
            failures.append(
                f"inject block-shift on {frame_id}: largest delta {comp} px is not under the coarse "
                f"inject's {coarse[frame_id][1]} px there, so the two no longer differ in size"
            )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    with _PinnedSettings():
        lift = _FINE_LIFT_STEPS * _sloped_cache("farms", True)._levels[1].proj.elev_step
    print(
        f"OK: sloped_fine frames couple to its checks, land in {rp.MIN_LONG_EDGE}-{rp.MAX_LONG_EDGE} px and match "
        f"chunk_px {DEFAULT_CHUNK_PX}; all {len(FINE_INJECTS)} injects bite exactly their own +1 frames "
        f"(a {lift} px lift, at 1x on screen), block-shift inside one {_FINE_BLOCK_PX} px block and under "
        f"{FINE_COARSE_INJECT}, fine-drape moves fields and no object, and none reaches a level-0 twin or "
        f"{' / '.join(_FINE_SPECIFICITY)}"
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
        elif self.name == "seam-conform-off":
            # TASK-284's defect: each blendomatic tile's own non-bleeding edges, so same-terrain neighbours step.
            self._patch(terrain_blend, "conform_edges", lambda alpha, _tile: alpha)
        terrain_blend.clear_caches()
        return self

    def __exit__(self, *exc):
        for obj, attr, old in reversed(self.saved):
            setattr(obj, attr, old)
        asset_source.get_blend_mask.cache_clear()
        terrain_blend.clear_caches()


BORDER_INJECTS = {
    "blend-off": None, "polarity-flip": None, "priority-reversed": None, "edge-residue": None, "one-tile-plain": "one-tile",
    "seam-conform-off": None,
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
    the two apart, since layer patches sit on the shoreline). Round 3 (TASK-284): island_shore's
    clean answer is SOFT once same-terrain seams conform, with blend-off, polarity-flip,
    one-tile-plain and seam-conform-off (its localized control) as its matching injects."""
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
                f"(2) {texture_features}; (3) the frame's own edges; (4) a separate patch of a third ground (such "
                "as the cells of orange-brown earth or dirt) that sits wholly inside another ground in the "
                "plain-cells frame too. Only a band or blob that is detached in the river frame but joined to its "
                "own ground in the plain-cells frame counts as detached.\n\n"
                "Answer BROKEN if at least one transition is broken anywhere, CONTINUOUS if each runs on unbroken."
            ),
            frames=("river", "river_plain"),
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
                "following a grid? The texture's own blotches, specks and darker or lighter areas do not count, nor "
                "does the texture's own repeat: the same specks or marks recurring on a regular lattice.\n\n"
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
            texture = asset_source.terrain_texture_at(n_tid, asset_source.terrain_texture_size(tp))
            ox, oy = render._crop_offset(x, y, texture.shape[0], tp)
            own = img[blocks[(x, y)]][line].sum(axis=-1)
            pure = texture[oy : oy + tp, ox : ox + tp].astype(np.int64)[line].sum(axis=-1)
            out.append((float(np.abs(own - pure).mean()), (x, y), n))
    return out


# Alpha 0 composes to the neighbour's exact pixel. Unfaded blendomatic edges measured up to 28.8
# on these maps, blend-off's plain cell edges 160-351.
BORDER_SEAM_RESIDUE_MAX = 0.0


def _edge_line_composed(own_tid: int, key, line, offset: tuple[int, int], texel_line) -> np.ndarray:
    """One edge line's summed RGB as terrain_blend._compose's influence steps make it, with `key`'s
    masks read on `line` but every texture read on `texel_line` of the crop at `offset`."""
    size = asset_source.terrain_texture_size(BORDER_TILE_PX)
    ox, oy = offset
    tp = BORDER_TILE_PX

    def texels(tid: int) -> np.ndarray:
        return asset_source.terrain_texture_at(tid, size)[oy : oy + tp, ox : ox + tp][texel_line].astype(np.int64)

    out = texels(own_tid)
    for n_tid, mode, tile in key[2] if key else ():
        alpha = asset_source.get_blend_mask(mode, tile, tp)[line].astype(np.int64)[:, None]
        out = (out * alpha + texels(n_tid) * (128 - alpha) + 64) >> 7
    return out.sum(axis=-1)


def _same_terrain_seam_steps(frame_id: str) -> list[tuple[float, int, tuple[int, int], tuple[int, int]]]:
    """Overhead: (step, allowance, tile, neighbour) for every edge-adjacent pair wholly in the crop of
    one terrain, neither layered, at least one blended (TASK-284). The step is the mean summed-RGB
    difference between the neighbour's shared-edge line composed with its own masks and composed with
    the tile's masks on the tile's side of that edge, both on the neighbour's texels, so the
    textures' own contrast cancels and only the masks' disagreement is left. The allowance is the
    compose rounding, 1 per channel per influence entry of the side with more."""
    scenario = _border_scene(_BORDER_FRAMES[frame_id][0])[0]
    blocks = _border_tile_blocks(frame_id, "flat")
    xs, ys = (np.array(v) for v in zip(*blocks, strict=True))
    with _layers_cleared(scenario, frame_id in _BORDER_NO_LAYERS):
        tids = render.tile_field_mirror(scenario, "terrain_id").reshape(-1)
        layer_ids = render.tile_field_mirror(scenario, "layer").reshape(-1)
        keys = dict(zip(blocks, render._blend_keys(scenario, xs, ys), strict=True))
    width = scenario.map_manager.map_width
    size = asset_source.terrain_texture_size(BORDER_TILE_PX)

    def plain_layer(t: tuple[int, int], tid: int) -> bool:
        return int(layer_ids[t[1] * width + t[0]]) in (-1, tid)

    out = []
    for (x, y), key in keys.items():
        tid = int(tids[y * width + x])
        if not plain_layer((x, y), tid):
            continue
        for dx, dy in ((1, 0), (0, 1)):
            n = (x + dx, y + dy)
            if n not in blocks or int(tids[n[1] * width + n[0]]) != tid or not plain_layer(n, tid):
                continue
            if key is None and keys[n] is None:
                continue
            line, n_line = (np.s_[:, -1], np.s_[:, 0]) if dx else (np.s_[-1, :], np.s_[0, :])
            offset = render._crop_offset(*n, size, BORDER_TILE_PX)
            own = _edge_line_composed(tid, keys[n], n_line, offset, n_line)
            theirs = _edge_line_composed(tid, key, line, offset, n_line)
            allowance = 3 * max(len(k[2]) if k else 0 for k in (key, keys[n]))
            out.append((float(np.abs(own - theirs).mean()), allowance, (x, y), n))
    return out


def _over_allowance(steps) -> tuple | None:
    """The pair furthest over its rounding allowance, or None when every pair is within it."""
    worst = max(steps, key=lambda s: s[0] - s[1], default=None)
    return worst if worst is not None and worst[0] > worst[1] else None


_BORDER_SEAM_FRAMES = ("river", "island_overhead", "island_shore_overhead")


def _check_ground_borders() -> None:
    """--check for ground_borders: spec coupling, frame band, blending engaged, the spread checks'
    priority orders, interior tiles unchanged (overhead, per tile), every single-neighbour bleeding
    edge wholly the neighbour's (overhead), same-terrain neighbours' masks agreeing along their shared
    edge within rounding (overhead), each inject's reach (printed), neither the control nor the
    specificity frame reached by any inject, every coarse-class inject reaching every blended frame,
    one-tile-plain bounded to one tile's area per style under blend-off's delta, seam-conform-off
    breaking the same-terrain bound and edge-residue still breaking the residue bound."""
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
    for frame_id in _BORDER_SEAM_FRAMES:
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
    same_shown = []
    for frame_id in _BORDER_SEAM_FRAMES:
        steps = _same_terrain_seam_steps(frame_id)
        if not steps:
            failures.append(f"{frame_id}: no blended same-terrain pair to measure")
        elif over := _over_allowance(steps):
            step, allowance, tile, n = over
            failures.append(
                f"{frame_id}: same-terrain tiles {tile} and {n} disagree by {step:.1f} summed RGB along their shared "
                f"edge, over the {allowance} rounding allowance"
            )
        same_shown.append(f"{frame_id} {len(steps)} pairs, max {max((s[0] for s in steps), default=0):.1f}")
    print(f"same-terrain seams, mask-isolated (overhead): {', '.join(same_shown)}")

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

    # Each seam bound's own inject must break it, and conform_edges must not hide edge-residue.
    with _BorderInject("seam-conform-off"):
        broken = {fid: _over_allowance(_same_terrain_seam_steps(fid)) for fid in _BORDER_SEAM_FRAMES}
    if not any(broken.values()):
        failures.append("inject seam-conform-off: every same-terrain seam stays within its rounding allowance")
    shown = ", ".join(f"{fid} {over[0]:.1f}" if over else f"{fid} within" for fid, over in broken.items())
    print(f"seam-conform-off, worst same-terrain seam over its allowance: {shown}")
    with _BorderInject("edge-residue"):
        residue = {fid: max(_border_seam_residues(fid), default=(0.0,))[0] for fid in _BORDER_SEAM_FRAMES}
    if not any(r > BORDER_SEAM_RESIDUE_MAX for r in residue.values()):
        failures.append(f"inject edge-residue: no bleeding edge keeps more than {BORDER_SEAM_RESIDUE_MAX}")
    print(f"edge-residue, worst bleeding-edge residue: {', '.join(f'{f} {r:.1f}' for f, r in residue.items())}")

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
        f"neighbour's; same-terrain neighbours agree along their shared edge within rounding; all "
        f"{len(BORDER_INJECTS)} injects bite every blended frame and none reaches {_BORDER_SPECIFICITY} or the "
        f"plain-cells control; one-tile-plain stays within one tile's area; seam-conform-off and edge-residue "
        f"each break their own seam bound"
    )


# ------------------------------------------------------------ garrison_flags
#
# TASK-264 (GH #203): a garrisoned host draws the game's garrison flag. Every
# frame is one pair of twins from tools/gen_garrison_flags_scenario.py's scene,
# the same build() that writes build/garrison_flags/ for the in-game check:
# the garrisoned twin on the left, its empty twin on the right, on flat grass
# or water, Stepped, Show Garrisoned off (the app's default), through a fresh
# IsoChunkCache at the scene's own 64 px tiles.

# frame id -> (player, host const). P1 Franks, P2 Japanese, P3 Persians, Castle Age.
_GARRISON_FRAMES = {
    "pair_1": (1, 109), "pair_2": (2, 109), "pair_3": (3, 109), "pair_4": (1, 82), "pair_5": (2, 82),
    "pair_6": (3, 79), "pair_7": (1, 45), "pair_8": (2, 655),
}
_GARRISON_SPECIFICITY = "pair_8"  # Mosque: no garrison graphic, so no flag can appear
_GARRISON_LOCALIZED_HOST = 79  # the Watch Tower, in pair_6 only
_GARRISON_PAD_PX = 12
GARRISON_TILE_PX = 64
_GARRISON_SCENE: dict[str, object] = {}


def _garrison_scene() -> dict[str, object]:
    """{loaded, placed}: the scenario tool's own build(), once per process."""
    if not _GARRISON_SCENE:
        import gen_garrison_flags_scenario as scene

        loaded, _options, _units, placed = scene.build()
        _GARRISON_SCENE.update(loaded=loaded, placed=placed)
    return _GARRISON_SCENE


def _garrison_pair(frame_id: str) -> list[tuple[object, object | None]]:
    """[(host, occupant or None)] of one frame, the garrisoned twin first."""
    player, const = _GARRISON_FRAMES[frame_id]
    placed = _garrison_scene()["placed"]
    return [(host, occ) for twin, host, occ in placed if (twin.player, twin.const) == (player, const)]


@contextmanager
def _garrison_kept(frame_id: str | None):
    """The scene with only `frame_id`'s pair and its occupants left in it (None keeps all)."""
    loaded = _garrison_scene()["loaded"]
    lists = loaded.unit_manager.units
    saved = [list(units) for units in lists]
    if frame_id is not None:
        keep = {id(u) for pair in _garrison_pair(frame_id) for u in pair if u is not None}
        for units in lists:
            units[:] = [u for u in units if id(u) in keep]
    loaded.unit_gen += 1
    try:
        yield loaded
    finally:
        for units, before in zip(lists, saved, strict=True):
            units[:] = before
        loaded.unit_gen += 1


@contextmanager
def _garrison_flags_patched(wrap):
    """unit_sprites.garrison_flag_pieces_for() wrapped for the run (None: as shipped)."""
    from descape import unit_sprites

    real = unit_sprites.garrison_flag_pieces_for
    if wrap is not None:
        unit_sprites.garrison_flag_pieces_for = wrap(real)
    try:
        yield
    finally:
        unit_sprites.garrison_flag_pieces_for = real


def _zero_offsets(real):
    """COARSE. Every flag at its host's anchor, (0, 0), as if the .dat offsets were dropped."""
    from dataclasses import replace

    def pieces(unit_const, team_index, half_w, art=None):
        return [replace(p, dx=0, dy=0) for p in real(unit_const, team_index, half_w, art)]
    return pieces


def _one_untinted(real):
    """LOCALIZED. The Watch Tower's flag alone drawn in no team colour (GAIA's untinted art)."""
    def pieces(unit_const, team_index, half_w, art=None):
        return real(unit_const, 0 if unit_const == _GARRISON_LOCALIZED_HOST else team_index, half_w, art)
    return pieces


def _no_flags(_real):
    """Not a reviewer run: --check's flag-free reference, to measure where flags paint."""
    return lambda *_args, **_kwargs: []


GARRISON_INJECTS = {"zero-offsets": _zero_offsets, "one-untinted": _one_untinted}
GARRISON_COARSE_INJECT = "zero-offsets"
# Measured 2026-10-22: one-untinted changes 171 px of pair_6's 230x189 (0.39%), one component;
# zero-offsets changes 728 px there, 364 in its largest.
GARRISON_LOCALIZED_BOUND = rp.LocalizedBound(max_changed_fraction=0.008, max_component_px=250)
# Summed |dRGB| per changed px one-untinted must reach, so its flag reads as off-colour.
GARRISON_MIN_COLOUR_SHIFT = 60


def _garrison_cache(loaded):
    """A fresh Stepped cache over the scene, Show Garrisoned off as the app ships."""
    from descape.render_cache import IsoChunkCache
    from descape.unit_filter import UnitFilter

    elevations, proj = render.elevations_and_proj(loaded)
    return IsoChunkCache(loaded, elevations, proj, GARRISON_TILE_PX, unit_filter=UnitFilter(show_garrisoned=False), sprites=True)


def _garrison_ink_box(cache, units) -> tuple[int, int, int, int]:
    """Level-0 bbox of every sprite piece `units` paint, flags included."""
    loaded = cache.scenario
    boxes = []
    for unit in units:
        player = next(p for p, us in enumerate(loaded.unit_manager.units) if any(u is unit for u in us))
        for draw, px, py in render.unit_sprite_draws_at(loaded, cache.proj, cache.elevations, None, player, unit):
            h, w = draw.rgba.shape[:2]
            boxes.append((px - draw.hotspot_x, py - draw.hotspot_y, px - draw.hotspot_x + w, py - draw.hotspot_y + h))
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def _garrison_boxes() -> dict[str, tuple[int, int, int, int]]:
    """Each frame's crop: its pair's clean ink plus a pad, measured once."""
    if "boxes" not in _GARRISON_SCENE:
        with _garrison_kept(None) as loaded:
            cache = _garrison_cache(loaded)
            boxes = {}
            for frame_id in _GARRISON_FRAMES:
                x0, y0, x1, y1 = _garrison_ink_box(cache, [host for host, _occ in _garrison_pair(frame_id)])
                pad = _GARRISON_PAD_PX
                boxes[frame_id] = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        _GARRISON_SCENE["boxes"] = boxes
    return _GARRISON_SCENE["boxes"]


def _garrison_frame_images(inject: str | None, wrap=None, frame_id: str | None = None) -> dict[str, np.ndarray]:
    """Every garrison_flags frame at 1:1 under `inject` (or a raw `wrap`); with
    `frame_id`, that frame alone, rendered from a scene holding only its pair."""
    from descape import unit_sprites

    boxes = _garrison_boxes()
    wrap = GARRISON_INJECTS[inject] if inject else wrap
    with _garrison_flags_patched(wrap), _garrison_kept(frame_id) as loaded:
        unit_sprites._scaled_cache.clear()
        unit_sprites.clear_slice_cache()
        cache = _garrison_cache(loaded)
        frames = [frame_id] if frame_id else list(_GARRISON_FRAMES)
        return {fid: cache.render_rect(*boxes[fid], mip=0).copy() for fid in frames}


def _garrison_foot_fractions(frame_id: str, wrap=None) -> list[float]:
    """Per flag on the frame's garrisoned twin, where its pole's foot sits down
    the host's own body ink: 0 its top row, 1 its bottom row. The place checks'
    UPPER / LOWER is this against 0.5; the lowest foot answers for a host with two."""
    from descape import unit_sprites

    loaded = _garrison_scene()["loaded"]
    (host, _occupant), _empty = _garrison_pair(frame_id)
    player = next(p for p, us in enumerate(loaded.unit_manager.units) if any(u is host for u in us))
    half_w = GARRISON_TILE_PX // 2
    art = render.player_art_for(loaded, player)
    team = loaded.team_indices[player]
    body = unit_sprites.sprite_pieces_for(host.unit_const, render.stored_rotation(player, host), team, half_w, art=art)
    tops, bottoms = [], []
    for p in body:
        ys = np.nonzero((p.draw.rgba[..., 3] > 0).any(axis=1))[0]
        tops.append(p.dy - p.draw.hotspot_y + int(ys[0]))
        bottoms.append(p.dy - p.draw.hotspot_y + int(ys[-1]) + 1)
    top, bottom = min(tops), max(bottoms)
    with _garrison_flags_patched(wrap):
        flags = unit_sprites.garrison_flag_pieces_for(host.unit_const, team, half_w, art)
    return [(f.dy - top) / (bottom - top) for f in flags]


def _garrison_flags_spec() -> rp.PackSpec:
    """TASK-264's garrison flags (GH #203): eight pairs of a building and its
    twin, the left one holding a unit. Seven hosts draw a flag; the Mosque
    pair can draw none. zero-offsets is the place checks' matching inject,
    one-untinted (the Watch Tower's flag in no team colour) the tint check's."""
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption="Two copies of one building, on open ground or water.")
        for fid in _GARRISON_FRAMES
    )

    def place(n: int) -> rp.Check:
        return rp.Check(
            id=f"place_{n}",
            question=(
                f"Frame pair_{n} shows two copies of the same building, a left one and a right one. First, in your "
                "own words, describe every difference you can see between the two copies.\n\n"
                "Then the closed question. If one copy carries a small pennant on a thin pole that the other copy "
                "lacks, look at where that pole's foot (its lowest point) touches the building. Measure it against "
                "that building's own height, from the highest to the lowest pixel of the building itself, not "
                "counting the pennant: is the foot in the upper half of that height (UPPER) or the lower half "
                "(LOWER)? If the copy carries two such pennants, answer for the lower foot. Answer NONE if neither "
                "copy carries a pennant the other lacks."
            ),
            frames=(f"pair_{n}",),
            answers=("UPPER", "LOWER", "NONE", "CANNOT-TELL"),
        )

    tint = rp.Check(
        id="tint",
        question=(
            "Look at every frame. In each, the buildings carry patches of one team colour (blue, red or green). "
            "First, for each frame where one copy carries a small pennant that the other lacks, describe the "
            "pennant's colours.\n\n"
            "Then the closed question. Is every such pennant's coloured cloth in the same team colour as the "
            "patches on its own building (ALL-SAME), or is at least one pennant clearly not in its building's team "
            "colour, for example plain white, grey or another team's colour (SOME-DIFFERENT)? Answer NONE if no "
            "frame shows such a pennant."
        ),
        frames=tuple(_GARRISON_FRAMES),
        answers=("ALL-SAME", "SOME-DIFFERENT", "NONE", "CANNOT-TELL"),
    )
    return rp.PackSpec(
        id="garrison_flags",
        title="Building pairs review",
        frames=frames,
        checks=(*(place(n) for n in range(1, len(_GARRISON_FRAMES) + 1)), tint),
        preamble=(
            "These are offscreen renders from an isometric map editor for a strategy game. Each frame is a crop, "
            "enlarged by a whole-number factor with no smoothing, of two copies of one building standing side by "
            "side. Judge only what you can actually see."
        ),
        injects=tuple(GARRISON_INJECTS),
    )


def _garrison_prerequisites() -> str | None:
    """Why the garrison_flags pack cannot render here, or None."""
    if asset_source.get_install_path() is None:
        return "no AoE2:DE install visible (set AOE2DE_INSTALL_PATH)"
    if not (ROOT / "examples" / "blank_map.aoe2scenario").is_file():
        return "examples/blank_map.aoe2scenario is missing"
    return None


def _check_garrison_flags() -> None:
    """--check for garrison_flags: spec coupling and frame band; each crop holds
    its own pair and nothing else; flags paint on every frame but the
    specificity one; every clean foot is UPPER and every zero-offsets foot
    LOWER (the place checks' discriminator, measured); each inject's reach,
    with one-untinted inside its bound, on pair_6 alone, under the coarse
    inject there, and a visible colour shift."""
    if reason := _garrison_prerequisites():
        raise SystemExit(f"garrison_flags: {reason}")
    failures: list[str] = []
    spec = PACKS["garrison_flags"]()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")

    clean = _garrison_frame_images(None)
    bare = _garrison_frame_images(None, wrap=_no_flags)
    for frame_id, img in clean.items():
        try:
            scaled, _factor = rp.frame_to_band(img)
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame_id)
        except rp.PackError as exc:
            failures.append(str(exc))
        alone = _garrison_frame_images(None, frame_id=frame_id)[frame_id]
        if not np.array_equal(img, alone):
            failures.append(f"{frame_id}: {int(np.count_nonzero(rp.changed_pixels(img, alone)))} px come from another pair")
        flag_px = int(np.count_nonzero(rp.changed_pixels(img, bare[frame_id])))
        if (flag_px > 0) == (frame_id == _GARRISON_SPECIFICITY):
            failures.append(f"{frame_id}: flags paint {flag_px} px")
        print(f"flag ink: {frame_id} {flag_px}px")

    for frame_id in _GARRISON_FRAMES:
        if frame_id == _GARRISON_SPECIFICITY:
            continue
        for label, wrap, upper in (("clean", None, True), (GARRISON_COARSE_INJECT, _zero_offsets, False)):
            fractions = _garrison_foot_fractions(frame_id, wrap)
            lowest = max(fractions)
            print(f"feet: {frame_id} {label} {', '.join(f'{f:.2f}' for f in fractions)}")
            if (lowest < 0.5) != upper:
                failures.append(f"{frame_id} {label}: lowest foot at {lowest:.2f}, not {'UPPER' if upper else 'LOWER'}")

    reach: dict[str, dict[str, tuple[int, int]]] = {}
    for name in GARRISON_INJECTS:
        injected = _garrison_frame_images(name)
        touched: dict[str, tuple[int, int]] = {}
        for frame_id in _GARRISON_FRAMES:
            mask = rp.changed_pixels(clean[frame_id], injected[frame_id])
            if np.any(mask):
                touched[frame_id] = (int(np.count_nonzero(mask)), rp.largest_component_px(mask))
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
        if _GARRISON_SPECIFICITY in touched:
            failures.append(f"inject {name} reaches the specificity frame {_GARRISON_SPECIFICITY}")
    flagged = set(_GARRISON_FRAMES) - {_GARRISON_SPECIFICITY}
    if set(reach[GARRISON_COARSE_INJECT]) != flagged:
        failures.append(f"zero-offsets touches {sorted(reach[GARRISON_COARSE_INJECT])}, not every flagged frame")
    if set(reach["one-untinted"]) != {"pair_6"}:
        failures.append(f"one-untinted touches {sorted(reach['one-untinted'])}, not pair_6 alone")
    else:
        injected = _garrison_frame_images("one-untinted")["pair_6"]
        mask = rp.changed_pixels(clean["pair_6"], injected)
        failures += GARRISON_LOCALIZED_BOUND.failures(mask, "inject one-untinted on pair_6")
        if reach["one-untinted"]["pair_6"][1] >= reach[GARRISON_COARSE_INJECT]["pair_6"][1]:
            failures.append("one-untinted's largest delta on pair_6 is not under zero-offsets' there")
        shift = float(np.abs(clean["pair_6"][mask].astype(int) - injected[mask].astype(int)).sum(axis=1).mean())
        print(f"one-untinted colour shift: mean |dRGB| {shift:.0f} over {int(np.count_nonzero(mask))} px")
        if shift < GARRISON_MIN_COLOUR_SHIFT:
            failures.append(f"one-untinted shifts the tower flag's colour by only {shift:.0f}, under {GARRISON_MIN_COLOUR_SHIFT}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(
        f"OK: garrison_flags frames couple to its checks and land in {rp.MIN_LONG_EDGE}-{rp.MAX_LONG_EDGE} px, each "
        f"crop holds only its pair, flags paint on every frame but {_GARRISON_SPECIFICITY}, every clean foot is UPPER "
        f"and every zero-offsets foot LOWER, and both injects reach only their own frames"
    )


PACKS = {
    "sprite_shrink": _sprite_shrink_spec,
    "shadow_band": _shadow_band_spec,
    "ground_texture": _ground_texture_spec,
    "sloped_mip": _sloped_mip_spec,
    "sloped_fine": _sloped_fine_spec,
    "ground_borders": _ground_borders_spec,
    "garrison_flags": _garrison_flags_spec,
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
    return {
        "sprite_shrink": SHRINK_INJECTS,
        "ground_texture": GROUND_INJECTS, "sloped_mip": SLOPED_INJECTS, "sloped_fine": FINE_INJECTS,
        "ground_borders": BORDER_INJECTS,
        "garrison_flags": GARRISON_INJECTS,
    }.get(pack_id, INJECTS)


def _pack_images(pack_id: str, inject: str | None) -> dict[str, np.ndarray]:
    if pack_id == "sprite_shrink":
        return _shrink_frame_images(inject)
    if pack_id == "ground_texture":
        return _ground_frame_images(inject)
    if pack_id == "sloped_mip":
        return _sloped_frame_images(inject)
    if pack_id == "sloped_fine":
        return _fine_frame_images(inject)
    if pack_id == "ground_borders":
        return _border_frame_images(inject)
    if pack_id == "garrison_flags":
        return _garrison_frame_images(inject)
    return _frame_images(inject)


def _frame_tile_px(pack_id: str, frame_id: str) -> int:
    if pack_id == "sprite_shrink":
        return 2 * _SHRINK_FRAMES[frame_id]
    if pack_id == "ground_texture":
        return _GROUND_FRAMES[frame_id][1]
    if pack_id in ("sloped_mip", "sloped_fine"):
        # Graphics quality 3 puts both 240x240 maps at 64 px per tile at level 0.
        return _level_px(64, (_SLOPED_FRAMES if pack_id == "sloped_mip" else _FINE_FRAMES)[frame_id][1])
    if pack_id == "ground_borders":
        return BORDER_TILE_PX
    if pack_id == "garrison_flags":
        return GARRISON_TILE_PX
    return TILE_PX


def generate(out_dir: Path, pack_id: str, inject: str | None, replica: int = 0) -> list[Path]:
    spec = PACKS[pack_id]()
    rp.validate_spec(spec)
    injects = _pack_injects(pack_id)
    if inject is not None and inject not in injects:
        raise SystemExit(f"unknown inject {inject!r} for {pack_id}; choose from {', '.join(injects)}")
    if pack_id == "sprite_shrink" and (reason := _shrink_prerequisites()):
        raise SystemExit(f"sprite_shrink: {reason}")
    if pack_id == "ground_texture" and (reason := _ground_prerequisites()):
        raise SystemExit(f"ground_texture: {reason}")
    if pack_id in ("sloped_mip", "sloped_fine") and (reason := _sloped_prerequisites()):
        raise SystemExit(f"{pack_id}: {reason}")
    if pack_id == "ground_borders" and (reason := _border_prerequisites()):
        raise SystemExit(f"ground_borders: {reason}")
    if pack_id == "garrison_flags" and (reason := _garrison_prerequisites()):
        raise SystemExit(f"garrison_flags: {reason}")

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
    if pack_id == "sprite_shrink":
        _check_sprite_shrink()
        return
    if pack_id == "ground_texture":
        _check_ground_texture()
    elif pack_id == "sloped_mip":
        _check_sloped_mip()
    elif pack_id == "sloped_fine":
        _check_sloped_fine()
    elif pack_id == "ground_borders":
        _check_ground_borders()
    elif pack_id == "garrison_flags":
        _check_garrison_flags()
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
        choices=sorted({
            *INJECTS, *GROUND_INJECTS, *SLOPED_INJECTS, *FINE_INJECTS, *BORDER_INJECTS, *SHRINK_INJECTS,
            *GARRISON_INJECTS,
        }),
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
