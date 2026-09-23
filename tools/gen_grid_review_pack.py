#!/usr/bin/env python3
"""Review pack for View > Grid baked into the terrain (GH #76): a unit is not
crossed by a grid line, nearer raised ground hides farther lines, a building
is not crossed either, and how the grid reads over an elevation step.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script because that tool's frames and injects are shadow-band specific.
Needs a real AoE2:DE install (sprites); renders off-engine through the app's
own chunk caches, so composite_rect_flat / _iso / _sloped with the grid baked.

Run:
    tools/gen_grid_review_pack.py
    tools/gen_grid_review_pack.py --replica 1
    tools/gen_grid_review_pack.py --inject no-occlusion
    tools/gen_grid_review_pack.py --check      # writes nothing

Writes build/review_pack/grid/<opaque token>/, gitignored.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_review_pack as grp
import gen_seam_eyeball as seam_eyeball
import numpy as np

from descape import asset_source, grid_overlay, render, render_cache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit import review_pack as rp

PACK_ID = "grid"
OUT_DIR = grp.OUT_DIR
ARCHER = 4
CASTLE = 82
PLAYER = 1
# DESERT_SAND: even enough that a thin line reads against it.
GROUND_TERRAIN = 14
PLATEAU_ELEV = 8
# (blend, thickness) per frame. The unit frames take the strongest valid setting: a line over a
# sprite has to read against busy sprite art, not flat sand.
DARK = (-80, 2)
LIGHT = (80, 2)
BODY = (100, 4)

# Raised 2x2 block; the lower ground behind it is up-screen (larger x, smaller y).
_PLATEAU = {(x, y) for x in (61, 62) for y in (61, 62)}

# scene -> (units as (const, x, y), raised tiles)
_SCENES = {
    # Every unit stands on MAJOR lines (index % 4 == 0): a minor line over a sprite measured invisible
    # (round 1). Tile (63, 60) owns y_low 60 and x_high 64. Position swept in 1/8 tiles for body contrast.
    "figure": ([(ARCHER, 63.375, 60.5)], set()),
    # Flat's icon fills its own tile, so off-centre is what puts lines x=64 and y=60 under it.
    "figure_flat": ([(ARCHER, 64.25, 59.75)], set()),
    # Footprint 62..65, so majors x=64 and y=64 run through the middle of the castle.
    "keep": ([(CASTLE, 64.0, 64.0)], set()),
    # A figure on the block's front tile: the coarse inject crosses it, the localized one is cut by it, so
    # the coarse one's largest delta outgrows the localized one's on every frame they share (measured).
    "rise": ([(ARCHER, 61.5, 62.5)], _PLATEAU),
}

_FIGURE = [(63, 60, 0)]
_FIGURE_FLAT = [(64, 59, 0)]
_KEEP_TILES = [(x, y, 0) for x in (61, 66) for y in (61, 66)]
_RISE_TILES = [(x, y, 0) for x in (59, 64) for y in (59, 64)] + [(x, y, PLATEAU_ELEV) for x, y in _PLATEAU]
# The block's top and the ground behind it: a raised tile's back edges carry the seam contour.
_BACK_TILES = [(x, y, PLATEAU_ELEV) for x, y in _PLATEAU] + [(63, 59, 0), (64, 60, 0)]

# frame id -> (scene, style, crop tiles as (x, y, elevation), pad as (side, top, bottom), grid as (blend,
# thickness) or None for off). Pad is in tiles for Flat, half-tile widths and heights for the iso styles.
_FIGURE_PAD = (1, 4, 1)
_FRAMES = {
    "figure_a": ("figure_flat", "flat", _FIGURE_FLAT, (1, 1, 1), BODY),
    "figure_b": ("figure", "stepped", _FIGURE, _FIGURE_PAD, BODY),
    "figure_c": ("figure", "sloped", _FIGURE, _FIGURE_PAD, BODY),
    "figure_d": ("figure", "stepped", _FIGURE, _FIGURE_PAD, None),
    "keep": ("keep", "stepped", _KEEP_TILES, (0, 10, 0), BODY),
    "rise_a": ("rise", "stepped", _RISE_TILES, (0, 6, 0), DARK),
    "rise_b": ("rise", "sloped", _RISE_TILES, (0, 6, 0), DARK),
    "step": ("rise", "stepped", _BACK_TILES, (1, 1, 0), LIGHT),
}
SPECIFICITY_FRAME = "figure_d"
# Frames whose question the coarse inject must visibly answer. Round 1 changed 5392 castle px and no
# reviewer saw it, so the floor is on contrast, not on pixel count.
BODY_FRAMES = ("figure_a", "figure_b", "figure_c", "keep")
MIN_BODY_DELTA = 40  # mean per-channel |delta| over the changed body pixels
STRONG_DELTA = 150  # summed-channel |delta| a pixel needs to count as a visible mark
MIN_STRONG_COMPONENT = 40


def _scenario(scene: str, with_units: bool):
    placed, raised = _SCENES[scene]
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation = PLATEAU_ELEV if (tile.x, tile.y) in raised else 0
        tile.terrain_id = GROUND_TERRAIN
    units = [[] for _ in base.unit_manager.units]
    for i, (const, x, y) in enumerate(placed if with_units else ()):
        units[PLAYER].append(SimpleNamespace(
            unit_const=const, x=x, y=y, z=0.0, rotation=0.0,
            reference_id=8000 + i, status=2, garrisoned_in_id=-1,
        ))
    return SimpleNamespace(
        map_manager=base.map_manager,
        unit_manager=SimpleNamespace(units=units),
        team_indices=base.team_indices,
        player_colors=base.player_colors,
        # The library's tiles look their scenario up by uuid, so it must stay alive.
        _base=base,
    )


def _cache(scn, style: str):
    """The app's own chunk cache for this style, sprites on, as the viewer builds it."""
    mm = scn.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "flat":
        return render_cache.FlatChunkCache(scn, tile_px, sprites=True)
    if style == "stepped":
        elevations, proj = render.elevations_and_proj(scn)
        return render_cache.IsoChunkCache(scn, elevations, proj, tile_px, sprites=True)
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
    return render_cache.SlopedChunkCache(scn, elevations, corner_rise, proj, tile_px, sprites=True)


def _box(cache, style: str, tiles, pad):
    """Bounds of every named tile's diamond at its own elevation, padded."""
    side, top, bottom = pad
    if style == "flat":
        px = cache._mip_tile_px[0]
        xs = [x for x, _y, _e in tiles]
        ys = [y for _x, y, _e in tiles]
        return (min(xs) - side) * px, (min(ys) - top) * px, (max(xs) + 1 + side) * px, (max(ys) + 1 + bottom) * px
    proj = cache.proj if style == "sloped" else cache._levels[0].proj
    xs, ys = [], []
    for tx, ty, elev in tiles:
        sx, sy = grp.ig.tile_screen_origin(tx, ty, elev, proj)
        xs += [sx, sx + 2 * proj.half_w]
        ys += [sy, sy + 2 * proj.half_h]
    return (
        min(xs) - side * proj.half_w, max(0, min(ys) - top * proj.half_h),
        max(xs) + side * proj.half_w, max(ys) + bottom * proj.half_h,
    )


def _composite(frame_id: str, with_units: bool, grid_on: bool) -> np.ndarray:
    scene, style, tiles, pad, grid = _FRAMES[frame_id]
    cache = _cache(_scenario(scene, with_units), style)
    if grid_on and grid is not None:
        cache.set_grid(grid_overlay.grid_bake(True, *grid))
    return cache._composite_rect(0, *_box(cache, style, tiles, pad))


# ----------------------------------------------------------------- injects


@contextmanager
def _recorded_grid():
    """Swap the per-tile grid painters for recorders: each call's arguments
    minus the image, replayable onto any composite of the same rect."""
    calls = []
    real = {"_paint_grid_iso": render._paint_grid_iso, "_paint_grid_flat": render._paint_grid_flat}

    def recorder(fn):
        return lambda _img, *args, **kwargs: calls.append((fn, args, kwargs))

    for name, fn in real.items():
        setattr(render, name, recorder(fn))
    try:
        yield calls
    finally:
        for name, fn in real.items():
            setattr(render, name, fn)


def _replay(img: np.ndarray, calls) -> np.ndarray:
    out = img.copy()
    for fn, args, kwargs in calls:
        fn(out, *args, **kwargs)
    return out


def _grid_over_sprites(frame_id: str, clean: np.ndarray) -> np.ndarray:
    """The pre-fix render: the whole lattice lerped over the finished
    composite, sprites and nearer raised ground included."""
    with _recorded_grid() as calls:
        _composite(frame_id, with_units=False, grid_on=True)
    return _replay(_composite(frame_id, with_units=True, grid_on=False), calls)


def _no_occlusion(frame_id: str, clean: np.ndarray) -> np.ndarray:
    """Every tile's grid painted after the whole terrain pass, before units:
    wherever the unoccluded lattice differs from the baked one, take the
    lattice, except on a unit's own pixels, which still paint last."""
    with _recorded_grid() as calls:
        _composite(frame_id, with_units=False, grid_on=True)
    lattice = _replay(_composite(frame_id, with_units=False, grid_on=False), calls)
    baked = _composite(frame_id, with_units=False, grid_on=True)
    out = clean.copy()
    mask = rp.changed_pixels(lattice, baked) & ~_body_mask(frame_id)
    out[mask] = lattice[mask]
    return out


INJECTS = {
    "grid-over-sprites": _grid_over_sprites,
    "no-occlusion": _no_occlusion,
}
COARSE_INJECT = "grid-over-sprites"
LOCALIZED_INJECT = "no-occlusion"


# --------------------------------------------------------------- the spec

_BODY_QUESTION = (
    "First describe the small standing figure and the ground right around it, including any straight "
    "lines on that ground. Then: is any straight line drawn ON TOP of the figure's own body (head, "
    "torso, legs, or what it holds), running across it? Lines on the ground that stop at the figure's "
    "outline do not count, and neither do lines seen through its soft shadow on the ground. CROSSED if "
    "a line is drawn over the body, CLEAR if not."
)


def _spec() -> rp.PackSpec:
    figure_caption = "A small standing figure on flat ground, seen from above{}."
    frames = (
        rp.Frame("figure_a", "figure_a.png", "raw", caption=figure_caption.format(" straight down")),
        rp.Frame("figure_b", "figure_b.png", "raw", caption=figure_caption.format(" at an angle")),
        rp.Frame("figure_c", "figure_c.png", "raw", caption=figure_caption.format(" at an angle")),
        rp.Frame("figure_d", "figure_d.png", "raw", caption=figure_caption.format(" at an angle")),
        rp.Frame("keep", "keep.png", "raw", caption="A large building on flat ground, seen from above at an angle."),
        rp.Frame("rise_a", "rise_a.png", "raw", caption="A raised block of ground and the lower ground around it."),
        rp.Frame("rise_b", "rise_b.png", "raw", caption="A raised mound of ground and the lower ground around it."),
        rp.Frame("step", "step.png", "raw", caption="Close-up of where raised ground meets lower ground."),
    )
    checks = (
        rp.Check(id="figure_a_body", question=_BODY_QUESTION, frames=("figure_a",),
                 answers=("CROSSED", "CLEAR", "CANNOT-TELL")),
        rp.Check(id="figure_b_body", question=_BODY_QUESTION, frames=("figure_b",),
                 answers=("CROSSED", "CLEAR", "CANNOT-TELL")),
        rp.Check(id="figure_c_body", question=_BODY_QUESTION, frames=("figure_c",),
                 answers=("CROSSED", "CLEAR", "CANNOT-TELL")),
        rp.Check(id="figure_d_body", question=_BODY_QUESTION, frames=("figure_d",),
                 answers=("CROSSED", "CLEAR", "CANNOT-TELL")),
        rp.Check(
            id="keep_body",
            question=(
                "First describe the large building and the ground around and between its parts. Then: is "
                "any straight line drawn ON TOP of the building itself (its walls, towers or roofs), running "
                "across it? Lines on the ground that stop at the building's outline do not count, including "
                "ground visible between its parts. CROSSED if a line is drawn over the building, CLEAR if not."
            ),
            frames=("keep",),
            answers=("CROSSED", "CLEAR", "CANNOT-TELL"),
        ),
        rp.Check(
            id="rise_lines",
            question=(
                "The raised ground is nearer to you than the lower ground behind it (farther up the frame). "
                "First describe the lines on the lower ground behind it, and the lines on the raised ground "
                "itself. Then: does any line from the lower ground behind carry on across the raised ground, "
                "either running diagonally across its sloping or vertical sides, or crossing its top at "
                "places that are not the outlines of the top's own tiles? The lines outlining the top's own "
                "tiles and the raised ground's own edges do not count, and ignore the small figure. THROUGH "
                "if a farther line carries on across the raised ground, HIDDEN if the farther lines stop "
                "where it begins."
            ),
            frames=("rise_a", "rise_b"),
            answers=("THROUGH", "HIDDEN", "CANNOT-TELL"),
        ),
        rp.Check(
            id="step_lines",
            question=(
                "First describe the light lines in this frame, on the raised ground and on the lower ground "
                "behind it. Then: do all the light lines read as the same colour and weight, or does some "
                "stretch of them read clearly darker or greyer than the rest? UNIFORM if they all read "
                "alike, UNEVEN if some stretch reads darker or greyer. If UNEVEN, say where, and whether it "
                "reads as a flaw or as a natural outline of the raised ground."
            ),
            frames=("step",),
            answers=("UNIFORM", "UNEVEN", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map lines review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from a map editor for a strategy game. Each frame is a crop, "
            "enlarged by a whole-number factor with no smoothing. Judge only what you can actually see."
        ),
        injects=tuple(INJECTS),
    )


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    images = {}
    for frame_id in _FRAMES:
        clean = _composite(frame_id, with_units=True, grid_on=True)
        images[frame_id] = clean if inject is None else INJECTS[inject](frame_id, clean)
    return images


def generate(out_dir: Path, inject: str | None, replica: int = 0) -> list[Path]:
    spec = _spec()
    rp.validate_spec(spec)
    pack_dir = out_dir / PACK_ID / grp._run_token(PACK_ID, inject, replica)
    pack_dir.mkdir(parents=True, exist_ok=True)
    images = _frame_images(inject)
    written, rendered = [], {}
    for frame in spec.frames:
        scaled, factor = rp.frame_to_band(images[frame.id])
        path = pack_dir / frame.filename
        seam_eyeball._save_png(scaled, path)
        written.append(path)
        rendered[frame.id] = (scaled.shape[1], scaled.shape[0], factor)
    review = pack_dir / "REVIEW.md"
    review.write_text(grp._review_markdown(spec, rendered, pack_dir))
    written.append(review)
    return written


def _body_mask(frame_id: str) -> np.ndarray:
    """Pixels a unit's sprite or mark owns: its composite with and without units, grid off."""
    return rp.changed_pixels(
        _composite(frame_id, with_units=True, grid_on=False), _composite(frame_id, with_units=False, grid_on=False)
    )


def _body_contrast(clean, injected, bodies) -> list[str]:
    """The coarse inject's mark on each unit must be strong enough to see, not just nonzero."""
    failures = []
    for fid in BODY_FRAMES:
        delta = np.abs(clean[fid].astype(np.int16) - injected[fid].astype(np.int16))
        changed = delta.any(axis=-1) & bodies[fid]
        mean = float(delta[changed].mean()) if changed.any() else 0.0
        strong = rp.largest_component_px((delta.sum(axis=-1) > STRONG_DELTA) & bodies[fid])
        print(f"body contrast: {fid} mean {mean:.1f}/channel, strongest mark {strong}px")
        if mean < MIN_BODY_DELTA or strong < MIN_STRONG_COMPONENT:
            failures.append(f"{COARSE_INJECT} is too faint on the unit in {fid}")
    return failures


def check() -> None:
    """Spec coupling, frame band, determinism, and each inject's reach per frame,
    split into what lands on a unit's own pixels and what lands on ground."""
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    clean = _frame_images(None)
    again = _frame_images(None)
    for frame in spec.frames:
        if rp.changed_pixels(clean[frame.id], again[frame.id]).any():
            failures.append(f"frame {frame.id}: two clean captures differ")
        try:
            scaled, _f = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))
    bodies = {frame.id: _body_mask(frame.id) for frame in spec.frames}
    reach = {}
    for name in INJECTS:
        injected = _frame_images(name)
        touched = {}
        for frame in spec.frames:
            mask = rp.changed_pixels(clean[frame.id], injected[frame.id])
            n = int(np.count_nonzero(mask))
            if n:
                on_body = int(np.count_nonzero(mask & bodies[frame.id]))
                touched[frame.id] = (n, rp.largest_component_px(mask), on_body)
        if not touched:
            failures.append(f"inject {name}: changed no pixels")
        if SPECIFICITY_FRAME in touched:
            failures.append(f"inject {name}: reached the {SPECIFICITY_FRAME} frame")
        reach[name] = touched
        if name == COARSE_INJECT:
            failures.extend(_body_contrast(clean, injected, bodies))
        shown = ", ".join(
            f"{fid} {px}px/{comp}max/{body}body" for fid, (px, comp, body) in sorted(touched.items())
        )
        print(f"reach: {name} -> {shown or 'nothing'}")
    coarse, local = reach[COARSE_INJECT], reach[LOCALIZED_INJECT]
    shared = [fid for fid in local if fid in coarse]
    if not shared:
        failures.append(f"{LOCALIZED_INJECT} shares no frame with {COARSE_INJECT}")
    failures.extend(
        f"{LOCALIZED_INJECT} is not smaller than {COARSE_INJECT} on {fid}"
        for fid in shared
        if local[fid][1] >= coarse[fid][1]
    )
    failures.extend(
        f"{LOCALIZED_INJECT} lands on a unit in {fid}" for fid, (_px, _c, body) in local.items() if body
    )
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(f"OK: {PACK_ID} frames couple to checks, land in band, all {len(INJECTS)} injects bite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--inject", choices=sorted(INJECTS))
    parser.add_argument("--replica", type=int, default=0, help="a second or third opaque directory of one run")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if asset_source.get_install_path() is None:
        raise SystemExit("needs a configured AoE2:DE install: the pack judges sprites against the grid")
    if args.check:
        check()
        return
    for path in generate(args.out_dir, args.inject, args.replica):
        print(f"wrote {path}")
    print(f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory "
          f"{grp._run_token(PACK_ID, args.inject, args.replica)}")


if __name__ == "__main__":
    main()
