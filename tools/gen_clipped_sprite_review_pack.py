#!/usr/bin/env python3
"""Review pack for GH #107: big objects (mountains, ruins) whose art reaches
past their own tile, composited terrain first and objects second, so nearer
ground no longer cuts them and only higher nearer ground still covers them.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_hero_glow_review_pack.py. Renders off-engine
through the app's own Stepped and Sloped chunk caches with units and sprites
on, the default View > Layers state and the native kernel live, so a clean
frame is the kernel's own two passes. Needs a configured AoE2:DE install (the
unit sprites and terrain textures) and the built kernel.

Sloped covers by a tile's lowest drawn corner: a tile hides an object only
where its whole surface is above the object's base. The `slope` frame sits
on that rule, with the object at the top of a ramp that falls away in front.

Run:
    tools/gen_clipped_sprite_review_pack.py --out-dir /tmp/review
    tools/gen_clipped_sprite_review_pack.py --out-dir /tmp/review --inject interleaved
    tools/gen_clipped_sprite_review_pack.py --out-dir /tmp/review --replica 1
    tools/gen_clipped_sprite_review_pack.py --check      # writes nothing

Writes <out-dir>/tall_objects/<opaque token>/ (default build/review_pack,
gitignored). Pass --out-dir whenever the checkout's own path names the work:
every frame path is printed in REVIEW.md.

The `inverted` and `one-tile` injects change the cover step inside the numpy
path, the only place it can be changed without a rebuilt kernel. The two
backends are held byte-identical by tests/test_native_composite.py, and
--check re-asserts it on every clean frame, so those runs differ from a clean
one by the injected defect alone. `interleaved` and `no-cover` run in the
kernel.
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

from descape import (
    asset_source,
    composite_backend,
    iso_geometry,
    native_composite,
    render,
    render_cache,
    unit_sprites,
)
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit import review_pack as rp

PACK_ID = "tall_objects"
OUT_DIR = grp.OUT_DIR
MOUNTAIN = 310  # MNTN1: a one-tile footprint under art six tiles wide
RUINS = 1837  # CastleRuins
GRASS, SAND = 0, 14
# The object's tile, mid-map. Depth grows with y - x: larger is nearer the viewer.
OX = OY = 60
_DEPTH = OY - OX


def _depth(x: int, y: int) -> int:
    return (y - x) - _DEPTH


def _flat(_x: int, _y: int) -> tuple[int, int]:
    return 0, GRASS


def _terrace(x: int, y: int) -> tuple[int, int]:
    """Two levels of sand, three depth rows in front of the object and beyond."""
    return (2, SAND) if _depth(x, y) >= 3 else (0, GRASS)


def _plateau(x: int, y: int) -> tuple[int, int]:
    """The object stands at the front edge of a two-level sand plateau."""
    return (2, SAND) if _depth(x, y) <= 0 else (0, GRASS)


def _slope(x: int, y: int) -> tuple[int, int]:
    """High ground behind the object, falling one level per depth row in front of it."""
    return max(0, min(3, 3 - _depth(x, y))), GRASS


# frame id -> (object const or None, ground function, style)
_FRAMES = {
    "open": (MOUNTAIN, _flat, "stepped"),
    "terrace": (MOUNTAIN, _terrace, "stepped"),
    "plateau": (MOUNTAIN, _plateau, "stepped"),
    "slope": (RUINS, _slope, "sloped"),
    "bare": (None, _terrace, "stepped"),
}


def _scenario(const: int | None, ground):
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation, tile.terrain_id = ground(tile.x, tile.y)
    units = [[] for _ in base.unit_manager.units]
    if const is not None:
        units[0].append(SimpleNamespace(
            unit_const=const, x=OX + 0.5, y=OY + 0.5, z=0.0, rotation=0.0,
            reference_id=8000, status=2, garrisoned_in_id=-1,
        ))
    return SimpleNamespace(
        map_manager=base.map_manager,
        unit_manager=SimpleNamespace(units=units),
        team_indices=base.team_indices,
        player_colors=base.player_colors,
        _base=base,
    )


def _cache(scn, style: str):
    """The app's own chunk cache for this style, sprites on, default layers."""
    mm = scn.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = render.elevations_and_proj(scn)
        return render_cache.IsoChunkCache(scn, elevations, proj, tile_px, sprites=True)
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
    return render_cache.SlopedChunkCache(scn, elevations, corner_rise, proj, tile_px, sprites=True)


# Past both objects' reach round the object's tile, so every frame shares one crop size.
_REACH_UP, _REACH_DOWN, _REACH_SIDE = 290, 150, 250


def _box(cache) -> tuple[int, int, int, int]:
    proj = cache._levels[0].proj
    x0, y0, x1, y1 = grp._tiles_bbox_px(proj, [(OX, OY)], 0)
    return max(0, x0 - _REACH_SIDE), max(0, y0 - _REACH_UP), x1 + _REACH_SIDE, y1 + _REACH_DOWN


def _render(frame_id: str) -> np.ndarray:
    const, ground, style = _FRAMES[frame_id]
    cache = _cache(_scenario(const, ground), style)
    return cache._composite_rect(0, *_box(cache))


# ----------------------------------------------------------------- injects


def _interleaved(per_tile):
    def paint(img, tiles, keys, units_by_tile, tile_px, proj, heights, map_w, map_h, **kwargs):
        for i, tile in enumerate(tiles):
            blend = None if keys is None else keys[i]
            per_tile(img, tile, units_by_tile, tile_px, proj, heights, map_w, map_h, blend=blend, **kwargs)

    return paint


def _inject_interleaved():
    """COARSE: the single-pass order restored, terrain and objects interleaved
    per tile, in the kernel and in the per-tile path alike."""
    return [
        (native_composite.UnitPack, "two_pass", property(lambda self: False)),
        (render, "_paint_candidates_iso", _interleaved(render._paint_tile_and_units_iso)),
        (render, "_paint_candidates_sloped", _interleaved(render._paint_tile_and_units_sloped)),
    ]


_NEVER_BELOW = 1 << 20


def _inject_no_cover():
    """The cover step never fires: every sprite's base is reported far above
    any ground, so no tile stands higher than it."""
    real = render._resolve_unit_sprite

    def lifted(*args, **kwargs):
        contribution = real(*args, **kwargs)
        if contribution is not None:
            for key, entries in contribution.by_anchor.items():
                contribution.by_anchor[key] = [(draw, px, py, _NEVER_BELOW) for draw, px, py, _rise in entries]
        return contribution

    return render, "_resolve_unit_sprite", lifted


def _inject_inverted():
    """The cover rule swapped: ground at or below an object's base covers it,
    ground above it does not. Numpy path only (see the module docstring)."""
    real_floor = render._ink_floor

    def restore(img, terrain, ink, base_y, base_x, dst_y, dst_x, height, *, extent=None):
        ay, ax = base_y + dst_y, base_x + dst_x
        ok = (ay >= 0) & (ay < ink.shape[0]) & (ax >= 0) & (ax < ink.shape[1])
        ay, ax = ay[ok], ax[ok]
        hit = (ink[ay, ax] != render.INK_NONE) & (ink[ay, ax] >= height)
        ay, ax = ay[hit], ax[hit]
        img[ay, ax] = terrain[ay, ax]
        ink[ay, ax] = render.INK_NONE

    # A floor under every height, so each tile takes the cover step.
    return [
        (render, "_restore_under", restore),
        (render, "_ink_floor", lambda *args: min(real_floor(*args), -1)),
        (composite_backend, "native", None),
    ]


# Two depth rows in front of the object, under its art in every frame that has one.
_BITE_TILE = (OX - 1, OY + 1)


def _inject_one_tile():
    """LOCALIZED: one tile of level or lower ground in front of the object
    covers it, the smallest piece of the old staircase. Numpy path only."""
    real_floor, real_iso, real_sloped = render._ink_floor, render._occlude_iso, render._occlude_sloped
    # The real floor still gates every other tile; the patched one below only makes the pass run.
    real_floor_value = [render.INK_NONE]

    def bite(img, terrain, ink, base_y, base_x, dst_y, dst_x):
        render._restore_under(img, terrain, ink, base_y, base_x, dst_y, dst_x, render.INK_NONE)

    def occlude_iso(img, terrain, ink, tile, tile_px, proj, elevations, map_h, offset, floor):
        if (tile.x, tile.y) != _BITE_TILE:
            real_iso(img, terrain, ink, tile, tile_px, proj, elevations, map_h, offset, real_floor_value[0])
            return
        base_x, base_y = iso_geometry.tile_screen_origin(tile.x, tile.y, int(elevations[tile.y, tile.x]), proj)
        dst_y, dst_x, _sy, _sx = iso_geometry.diamond_indices(tile_px)
        bite(img, terrain, ink, base_y - offset[1], base_x - offset[0], dst_y, dst_x)

    def occlude_sloped(img, terrain, ink, tile, tile_px, proj, corner_rise, offset, floor):
        if (tile.x, tile.y) != _BITE_TILE:
            real_sloped(img, terrain, ink, tile, tile_px, proj, corner_rise, offset, real_floor_value[0])
            return
        corners = [int(corner_rise[tile.y + dy, tile.x + dx]) for dy in (0, 1) for dx in (0, 1)]
        base_x, base_y, dst_y, dst_x, _sy, _sx, _uv = render._sloped_tile_quad(tile.x, tile.y, tile_px, proj, *corners, offset)
        bite(img, terrain, ink, base_y, base_x, dst_y, dst_x)

    def floor(*args):
        real_floor_value[0] = real_floor(*args)
        return min(real_floor_value[0], -1)

    return [
        (render, "_occlude_iso", occlude_iso),
        (render, "_occlude_sloped", occlude_sloped),
        (render, "_ink_floor", floor),
        (composite_backend, "native", None),
    ]


# name -> builder returning one or a list of (module, attribute, replacement)
INJECTS = {
    "interleaved": _inject_interleaved,
    "one-tile": _inject_one_tile,
    "no-cover": _inject_no_cover,
    "inverted": _inject_inverted,
}
COARSE_INJECT = "interleaved"
# Held under the coarse one's size wherever they share a frame.
LOCALIZED_INJECT = "one-tile"
SPECIFICITY_FRAME = "bare"


@contextmanager
def _patched(inject: str | None):
    patches = []
    if inject is not None:
        built = INJECTS[inject]()
        patches = built if isinstance(built, list) else [built]
    saved = [(mod, name, getattr(mod, name)) for mod, name, _new in patches]
    for mod, name, new in patches:
        setattr(mod, name, new)
    unit_sprites.clear_caches()
    try:
        yield
    finally:
        for mod, name, old in saved:
            setattr(mod, name, old)
        unit_sprites.clear_caches()


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=caption)
        for fid, caption in (
            ("open", "A large object standing on level grass."),
            ("terrace", "A large object standing on grass, with an area of raised sandy ground in front of it."),
            ("plateau", "A large object standing at the edge of raised sandy ground, with lower grass in front of it."),
            ("slope", "A large object standing on ground that falls away towards the bottom of the frame."),
            ("bare", "Grass with an area of raised sandy ground."),
        )
    )
    outline = (
        "Look at the large object in the middle of the frame and follow its outline along its lower half, from its "
        "left end round the bottom to its right end. First describe that outline: is it irregular like the rest of "
        "the object, or does any stretch of it run dead straight? Then: is the lower part of the object complete, or "
        "is part of it sliced away along one or more long, perfectly straight diagonal lines, with plain ground shown "
        "where the object would otherwise continue? Do not count the object's own shadow, and do not count the "
        "straight edges of the ground itself where no object reaches them. WHOLE if the lower outline is the "
        "object's own irregular shape throughout, SLICED if any stretch of it is cut along a straight diagonal line."
    )
    checks = (
        rp.Check(id="lower_outline", question=outline, frames=("open",), answers=("WHOLE", "SLICED", "CANNOT-TELL")),
        rp.Check(
            id="lower_outline_edge", question=outline, frames=("plateau",), answers=("WHOLE", "SLICED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="lower_outline_fall", question=outline, frames=("slope",), answers=("WHOLE", "SLICED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="raised_overlap",
            question=(
                "The lower part of the frame is raised ground: sandy, with a greenish fringe along its far edge, "
                "which is the dark zigzag line running across the frame. The large object stands on the lower "
                "grass beyond that line. First describe what you see where the object meets the zigzag line. "
                "Then: does any part of the object continue past the line, drawn on top of the raised ground, or "
                "does the object stop at the line, with the raised ground unbroken in front of it? ON-RAISED if "
                "any of the object lies over the raised ground, BEHIND if none does."
            ),
            frames=("terrace",),
            answers=("ON-RAISED", "BEHIND", "CANNOT-TELL"),
        ),
        rp.Check(
            id="ground_only",
            question=(
                "First list everything you can see in this frame. Then: is there anything standing on the ground "
                "(an object, a structure, or a fragment of one), or is the frame ground only? OBJECT or GROUND-ONLY."
            ),
            frames=("bare",),
            answers=("OBJECT", "GROUND-ONLY", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Large object review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from a map editor for a strategy game, seen from above at an angle: "
            "the ground is a grid of diamond-shaped tiles, and the bottom of a frame is nearest the viewer. Each "
            "frame is a crop, enlarged by a whole-number factor with no smoothing. Judge only what you can "
            "actually see."
        ),
        injects=tuple(INJECTS),
    )


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    with _patched(inject):
        return {frame_id: _render(frame_id) for frame_id in _FRAMES}


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


def check() -> None:
    """Spec coupling, frame band, each inject's reach printed per frame, and
    the kernel against the numpy path on every clean frame."""
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    clean = _frame_images(None)
    with composite_backend.use_backend("numpy"):
        reference = _frame_images(None)
    for frame in spec.frames:
        if not np.array_equal(clean[frame.id], reference[frame.id]):
            failures.append(f"{frame.id}: the kernel and the numpy path disagree")
        try:
            scaled, _f = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))
    reach = {}
    for name in INJECTS:
        injected = _frame_images(name)
        touched = {}
        for frame in spec.frames:
            mask = rp.changed_pixels(clean[frame.id], injected[frame.id])
            n = int(np.count_nonzero(mask))
            if n:
                touched[frame.id] = (n, rp.largest_component_px(mask))
        if not touched:
            failures.append(f"inject {name}: changed no pixels")
        if SPECIFICITY_FRAME in touched:
            failures.append(f"inject {name}: reached the specificity frame")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    # Each check's own inject must reach its frame, or a clean verdict there is untrusted.
    for name, frame_ids in (
        ("interleaved", ("open", "plateau", "slope")),
        ("one-tile", ("open", "plateau", "slope")),
        ("inverted", ("open", "plateau", "slope", "terrace")),
        ("no-cover", ("terrace",)),
    ):
        failures += [f"inject {name}: did not reach {fid}" for fid in frame_ids if fid not in reach[name]]
    coarse = reach[COARSE_INJECT]
    failures += [
        f"inject {LOCALIZED_INJECT}: {fid} component {comp} not under the coarse one's"
        for fid, (_px, comp) in reach[LOCALIZED_INJECT].items()
        if fid in coarse and comp >= coarse[fid][1]
    ]
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(f"OK: {PACK_ID} frames couple to checks, land in band, all {len(INJECTS)} injects bite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--inject", choices=sorted(INJECTS))
    parser.add_argument("--replica", type=int, default=0, help="a second or third opaque directory of the same run")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if asset_source.get_install_path() is None:
        raise SystemExit("needs a configured AoE2:DE install: unit sprites and terrain textures")
    if composite_backend.native is None:
        raise SystemExit(f"needs the native kernel live: {composite_backend.describe()}")
    if args.check:
        check()
        return
    for path in generate(args.out_dir, args.inject, args.replica):
        print(f"wrote {path}")
    print(
        f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory "
        f"{grp._run_token(PACK_ID, args.inject, args.replica)} -- record that pairing OUTSIDE the pack"
    )


if __name__ == "__main__":
    main()
