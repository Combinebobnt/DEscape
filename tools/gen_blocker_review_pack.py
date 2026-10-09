#!/usr/bin/env python3
"""Review pack for GH #121: a 1x3 / 3x1 blocker's editor marker over its whole
footprint, read as one barrier rather than one tile or a chain of three.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_marker_review_pack.py, whose scene helpers it
reuses. Renders off-engine in Stepped, Sloped and Flat (top-down) with units
and sprites on. Flat's frame came with its rectangle badge (wave 9): before it,
a multi-tile marker's Flat icon was the one-tile diamond contain-fitted into
the footprint rect, which the `half-tile` inject restores. Needs a configured
AoE2:DE install for the terrain textures.

Run:
    tools/gen_blocker_review_pack.py
    tools/gen_blocker_review_pack.py --replica 1
    tools/gen_blocker_review_pack.py --inject one-tile
    tools/gen_blocker_review_pack.py --check      # writes nothing

Writes build/review_pack/blockers/<opaque token>/, gitignored.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_marker_review_pack as mrp
import gen_review_pack as grp
import gen_seam_eyeball as seam_eyeball
import numpy as np

from descape import asset_source, iso_geometry, terrain_palette, unit_kind, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit import review_pack as rp

PACK_ID = "blockers"
OUT_DIR = grp.OUT_DIR
BLOCKER_1X3 = 2423
BLOCKER_3X1 = 2424
BLOCKER_1X1 = 1776

# (const, player, centre tile x, centre tile y); every placement at x.5/y.5, like the corpus.
_PLACED = [
    (BLOCKER_1X3, 1, 58, 60),
    (BLOCKER_3X1, 2, 62, 58),
    (BLOCKER_1X1, 1, 61, 62),
]
_TILES = [(58, 59), (58, 60), (58, 61), (61, 58), (62, 58), (63, 58), (61, 62)]

# frame id -> (scene, style, mip)
_FRAMES = {
    "stepped_m": ("badges", "stepped", 0),
    "stepped_l": ("badges", "stepped", 1),
    "sloped_m": ("badges", "sloped", 0),
    "flat_m": ("badges", "flat", 0),
    "ground": ("blank", "stepped", 0),
}


def _scenario(scene: str, style: str):
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation = mrp._elevation(style, tile.x, tile.y)
        tile.terrain_id = mrp.CHECKER[(tile.x + tile.y) % 2]
    units = [[] for _ in base.unit_manager.units]
    for i, (const, player, x, y) in enumerate(_PLACED if scene == "badges" else []):
        units[player].append(SimpleNamespace(
            unit_const=const, x=x + 0.5, y=y + 0.5, z=0.0, rotation=0.0,
            reference_id=7100 + i, status=2, garrisoned_in_id=-1,
        ))
    return SimpleNamespace(
        map_manager=base.map_manager,
        unit_manager=SimpleNamespace(units=units),
        team_indices=base.team_indices,
        player_colors=base.player_colors,
        _base=base,
    )


def _render(frame_id: str) -> np.ndarray:
    scene, style, mip = _FRAMES[frame_id]
    cache = mrp._cache(_scenario(scene, style), style)
    x0, y0, x1, y1 = mrp._box(cache, style, mip, _TILES)
    return cache._composite_rect(mip, x0, y0, x1, y1)


# ----------------------------------------------------------------- injects


def _inject_no_badges():
    """Coarse: every badge gone (the objects fall back to nothing drawable)."""
    return [(unit_kind, "invisible_category", lambda const: None),
            (sys.modules[__name__], "_PLACED", [])]


def _inject_one_tile():
    """The pre-GH #121 render: no blocker span, so a one-tile badge on the centre tile."""
    return terrain_palette, "BLOCKER_TILE_SPANS", {}


def _inject_half_tile():
    """Flat's pre-wave-9 icon: the one-tile diamond contain-fitted into the footprint rect."""
    real = unit_sprites._marker_icon

    def one_tile(category, team_index, footprint_w, footprint_h, _span):
        return real(category, team_index, footprint_w, footprint_h, (1, 1))

    return unit_sprites, "_marker_icon", one_tile


def _inject_chain():
    """One one-tile badge per footprint tile: covers the span, but as three pieces."""
    return [_chain_iso(), _chain_flat()]


def _chain_flat():
    """Flat's twin: one one-tile square badge per footprint cell, laid end to end."""
    real = unit_sprites.rect_marker_for

    @cache  # unit_sprites.clear_caches() calls cache_clear() on whatever sits here
    def chained(category, team_index, w, h, half_w):
        side = min(w, h)
        cell = real(category, team_index, side, side, half_w).rgba
        rgba = np.concatenate([cell] * (h // side), axis=0) if h > w else np.concatenate([cell] * (w // side), axis=1)
        return unit_sprites.SpriteDraw(rgba=rgba, hotspot_x=0, hotspot_y=0)

    return unit_sprites, "rect_marker_for", chained


def _chain_iso():
    real = unit_sprites.sprite_pieces_for

    def chained(unit_const, rotation, team_index, half_w, *args, **kwargs):
        span = terrain_palette.BLOCKER_TILE_SPANS.get(unit_const)
        if span is None:
            return real(unit_const, rotation, team_index, half_w, *args, **kwargs)
        draw = unit_sprites.marker_for(unit_kind.invisible_category(unit_const), team_index, half_w)
        half_h = iso_geometry.half_dims(2 * half_w)[1]
        pieces = []
        for i in range(span[0]):
            for j in range(span[1]):
                ox, oy = i - (span[0] - 1) // 2, j - (span[1] - 1) // 2
                pieces.append(unit_sprites.SpritePiece(draw=draw, dx=(ox + oy) * half_w, dy=(oy - ox) * half_h))
        return pieces

    return unit_sprites, "sprite_pieces_for", chained


# name -> builder returning one or a list of (module, attribute, replacement)
INJECTS = {
    "no-badges": _inject_no_badges,
    "one-tile": _inject_one_tile,
    "chain": _inject_chain,
    "half-tile": _inject_half_tile,
}
COARSE_INJECT = "no-badges"


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
            ("stepped_m", "A few badges on a checkered ground, at normal zoom."),
            ("stepped_l", "The same badges, zoomed in."),
            ("sloped_m", "The same badges on checkered, sloping ground, at normal zoom."),
            ("flat_m", "The same badges seen straight from above, on a checkered ground of square cells."),
            ("ground", "Checkered ground, same framing as the normal-zoom badges."),
        )
    )
    flat_checker = (
        "In this frame the map is seen straight from above: the ground is a checkerboard of two alternating "
        "ground textures, one SQUARE cell per map tile. Each badge's fill is semi-transparent, so the checker "
        "pattern shows faintly through it; that faint pattern is the ground, not part of the badge. "
    )
    checker = (
        "The ground is a checkerboard of two alternating ground textures, one diamond-shaped cell per map "
        "tile. Each badge's fill is semi-transparent, so the checker pattern shows faintly through it; that "
        "faint pattern is the ground, not part of the badge. "
    )
    checks = (
        rp.Check(
            id="badge_cells",
            question=(
                checker + "First, for each badge, describe its outline and list the checker cells it covers "
                "(count them, and say in which direction a longer badge runs). Then: how many cells does the "
                "LARGEST badge in each frame cover? THREE if the largest badge covers a straight run of three "
                "cells, ONE if every badge covers about one cell, NONE if there are no badges."
            ),
            frames=("stepped_m", "sloped_m"),
            answers=("THREE", "ONE", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="badge_pieces",
            question=(
                checker + "Ignoring that faint ground pattern, look at how each badge itself is drawn: its "
                "solid outline ring and its white symbols. First describe, for every badge, its outline and how "
                "many symbols it carries. Then: is any badge made of several smaller diamond-shaped pieces in a "
                "row, each with its own outline ring or its own symbol? SEGMENTED if any badge is, SINGLE if "
                "every badge is one shape with one outline and at most one symbol, NONE if there are no badges."
            ),
            frames=("stepped_l",),
            answers=("SEGMENTED", "SINGLE", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="flat_cover",
            question=(
                flat_checker + "First, for each badge, describe its outline shape, list the checker cells it "
                "touches, and say roughly how much of each of those cells its fill covers. Then: does any badge's "
                "fill cover a straight run of three cells, reaching the edges of all three? FILLED if one does, "
                "PARTIAL if every badge leaves clearly visible uncovered ground inside some of the cells it "
                "touches, NONE if there are no badges."
            ),
            frames=("flat_m",),
            answers=("FILLED", "PARTIAL", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="flat_pieces",
            question=(
                flat_checker + "Ignoring that faint ground pattern, look at how each badge itself is drawn: its "
                "solid outline ring and its white symbols. First describe, for every badge, its outline and how "
                "many symbols it carries. Then: is any badge made of several smaller pieces in a row, each with "
                "its own outline ring or its own symbol? SEGMENTED if any badge is, SINGLE if every badge is one "
                "shape with one outline and at most one symbol, NONE if there are no badges."
            ),
            frames=("flat_m",),
            answers=("SEGMENTED", "SINGLE", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="ground_objects",
            question=(
                "Are there any objects in this frame -- badges, symbols, coloured blocks -- or only the checkered "
                "ground? OBJECTS-PRESENT or GROUND-ONLY."
            ),
            frames=("ground",),
            answers=("OBJECTS-PRESENT", "GROUND-ONLY", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map badge footprint review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from a map editor for a strategy game. Each frame is a crop, "
            "enlarged by a whole-number factor with no smoothing. Judge only what you can actually see."
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
    """Spec coupling, frame band, and each inject's reach, printed per frame."""
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    clean = _frame_images(None)
    for frame in spec.frames:
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
        if "ground" in touched:
            failures.append(f"inject {name}: reached the bare-ground frame")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    coarse = reach[COARSE_INJECT]
    for name, touched in reach.items():
        if name == COARSE_INJECT:
            continue
        for fid, (_px, comp) in touched.items():
            if comp >= coarse.get(fid, (0, 0))[1]:
                failures.append(f"inject {name}: {fid} component {comp}px is not under the coarse inject's")
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
        raise SystemExit("needs a configured AoE2:DE install for the terrain textures")
    if args.check:
        check()
        return
    for path in generate(args.out_dir, args.inject, args.replica):
        print(f"wrote {path}")
    print(f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory "
          f"{grp._run_token(PACK_ID, args.inject, args.replica)}")


if __name__ == "__main__":
    main()
