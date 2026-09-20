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
    tools/gen_review_pack.py --pack shadow_band --inject notch-inject

then hand build/review_pack/<pack>/REVIEW.md to a fresh agent, in its own
call, with nothing else. See tools/REVIEW_PACK.md for the protocol and for
what a verdict may not close.

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
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_seam_eyeball as seam_eyeball
import numpy as np

from descape import iso_geometry as ig
from descape import render
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
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
    where the apex wedge's wrong-side flank used to draw a perpendicular tick,
    and there is no corner anywhere in the frame to confound the question."""
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = 1 if tile.y >= _RUN_EDGE_Y else 0
    return scenario


def _corner_scenario():
    """A = (x, y) and B = (x+1, y+1) one level up, so both cast onto
    N = (x+1, y): the smallest inner corner, and the only place the tip pass
    draws. Same fixture as tools/gen_contact_shadow_eyeball.py's."""
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

# Captured at import, before any patch can replace the module attribute.
_APEX_INDICES = ig.shadow_apex_indices


def _inject_no_apex_wedge(_tile_px, _rise_px, _sides="both"):
    """COARSE. Gates the apex-wedge pass off entirely, reopening the hole at
    every junction along a terrace so a run of adjacent casters reads as
    separate blocks instead of one band. Validates only that the reviewer is
    looking at the right frame and can see the band at all."""
    return (_EMPTY, _EMPTY, _EMPTY, _EMPTY)


def _inject_notch(tile_px, rise_px, _sides="both"):
    """LOCALIZED. Widens the wedge back to both flanks regardless of which
    side actually casts, which is the historical defect exactly: a 1px spur
    at the opposite slope to the contour, at every apex on a one-sided run.
    The real thing at the real size, rather than a synthetic perturbation
    whose size would be this tool's own choice."""
    return _APEX_INDICES(tile_px, rise_px, "both")


def _inject_no_tip(_tile_px, _rise_px, _side="up_right"):
    """LOCALIZED. Gates the inner-corner tip pass off, which is literally the
    pre-4bd5f07 render: the two casters' bands converge into a 2-column bare
    stripe at the vertex and leave it un-shaded. corner_symptom's matching
    control, and the only inject in the pack that reopens a GAP rather than
    adding something."""
    return (_EMPTY, _EMPTY, _EMPTY, _EMPTY)


@dataclass(frozen=True)
class _Inject:
    """One known defect: which iso_geometry entry point it replaces, the
    replacement, and the bound --check holds it to. `bound=None` marks the
    pack's coarse inject, which is asserted on component count instead."""

    attr: str
    fn: object
    bound: object


# Ceilings measured over every frame each inject actually touches, not just
# the frames its own check names -- an inject that leaks into a neighbouring
# frame is how an off-diagonal pre-registration goes wrong. Worst case today:
# notch-inject 0.540% / 16 px (band_pct100), no-tip 0.455% / 28 px
# (corner_pct100). The fraction ceilings differ because the crops do: the
# corner frame is 6150 px against the run frames' 66176, so one constant
# across both would mean two different things.
NOTCH_BOUND = rp.LocalizedBound(max_changed_fraction=0.0062, max_component_px=24)
TIP_BOUND = rp.LocalizedBound(max_changed_fraction=0.0053, max_component_px=32)

INJECTS = {
    "no-apex-wedge": _Inject("shadow_apex_indices", _inject_no_apex_wedge, None),
    "notch-inject": _Inject("shadow_apex_indices", _inject_notch, NOTCH_BOUND),
    "no-tip": _Inject("shadow_tip_indices", _inject_no_tip, TIP_BOUND),
}

# The coarse inject is the yardstick the localized ones are localized AGAINST,
# so "they differ in size" is asserted on the frames they share rather than
# only against a constant. Measured margins are thin on the run frames (16 vs
# 18 px at pct 100), which is the point: the constant above would not catch a
# localized inject drifting up to meet the coarse one there.
COARSE_INJECT = "no-apex-wedge"

# no-apex-wedge must genuinely break the contour, not merely change pixels.
# Clean reads 3 connected components on the pyramid band (one per terrace
# ring); with the wedge gone it reads 18.
COARSE_MIN_COMPONENT_RATIO = 3.0


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
    """The pilot pack. Three questions, each on frames that can answer it:
    a pyramid for contour continuity, a pure one-sided straight run for
    perpendicular artifacts, a bare inner corner for which corner symptom is
    present, and a flat field where none of the three is possible."""
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
        frames.append(
            rp.Frame(
                f"corner_pct{pct}_raw",
                f"corner_pct{pct}_raw.png",
                "raw",
                caption=f"Two raised tiles meeting at a corner over one lower tile, vertical scale {pct}%.",
            )
        )
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
                "Follow the dark shading that runs along each terrace edge of the hill. "
                "Does it read as ONE unbroken contour per terrace ring, or does it read as a "
                "row of separate blocks with clear breaks between them? "
                "Answer PASS if unbroken, FAIL if broken into separate pieces."
            ),
            frames=(
                "band_pct50_raw",
                "band_pct50_control",
                "band_pct50_diff",
                "band_pct100_raw",
                "band_pct100_control",
                "band_pct100_diff",
            ),
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
        rp.Check(
            id="corner_symptom",
            question=(
                "Look at the vertex where the shading from the two raised tiles meets over the "
                "lower tile between them. Which is true there? "
                "GAP: the dark contour breaks, leaving un-shaded ground at the vertex. "
                "EXTRA: there is additional darkening at the vertex beyond the two contours. "
                "NEITHER: the two contours meet cleanly with nothing missing and nothing added."
            ),
            frames=("corner_pct50_raw", "corner_pct100_raw"),
            answers=("GAP", "EXTRA", "NEITHER", "CANNOT-TELL"),
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


PACKS = {"shadow_band": _shadow_band_spec}


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
            images[f"corner_pct{pct}_raw"] = _scene_pair("corner", pct, boxes[("corner", pct)])[0]
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


def _run_token(pack_id: str, inject: str | None) -> str:
    """An opaque directory name per run, INCLUDING the clean one.

    Every frame path appears verbatim in REVIEW.md, so a directory called
    `shadow_band__notch-inject` hands the reviewer the answer key in the one
    place it is guaranteed to read. Naming only the injects opaquely would
    not help either: a run that was not in the opaque form would be the clean
    one by elimination. Deterministic, so re-running overwrites rather than
    accumulating, and the mapping is printed to the operator's terminal --
    which the reviewer never sees.
    """
    return hashlib.sha1(f"{pack_id}|{inject or 'clean'}".encode(), usedforsecurity=False).hexdigest()[:10]


def generate(out_dir: Path, pack_id: str, inject: str | None) -> list[Path]:
    spec = PACKS[pack_id]()
    rp.validate_spec(spec)
    if inject is not None and inject not in INJECTS:
        raise SystemExit(f"unknown inject {inject!r}; choose from {', '.join(INJECTS)}")

    pack_dir = out_dir / pack_id / _run_token(pack_id, inject)
    pack_dir.mkdir(parents=True, exist_ok=True)

    images = _frame_images(inject)
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
        "tile_px": TILE_PX,
        "frames": [
            {
                "id": f.id,
                "file": f.filename,
                "kind": f.kind,
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


def _band_component_count(inject: str | None, pct: int = 100) -> int:
    box = _clean_boxes()[("band", pct)]
    with _GeometryPatch(INJECTS.get(inject) if inject else None):
        raw, control = _scene_pair("band", pct, box)
    return rp.components(rp.changed_pixels(raw, control))


def check() -> None:
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

    clean_components = _band_component_count(None)
    coarse_components = _band_component_count("no-apex-wedge")
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
        f"{clean_components} -> {coarse_components} components under the coarse one"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument("--pack", default="shadow_band", choices=sorted(PACKS), help="Which pack to render")
    parser.add_argument("--inject", choices=sorted(INJECTS), help="Render the pack with one known defect applied")
    parser.add_argument("--check", action="store_true", help="Validate without writing any files")
    args = parser.parse_args()

    if args.check:
        check()
        return

    written = generate(args.out_dir, args.pack, args.inject)
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
        f"\nrun `{args.inject or 'clean'}` is directory "
        f"{_run_token(args.pack, args.inject)} -- record that pairing OUTSIDE the pack, "
        f"and hand the reviewer only that directory's REVIEW.md"
    )


if __name__ == "__main__":
    main()
