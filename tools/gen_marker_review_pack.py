#!/usr/bin/env python3
"""Review pack for GH #53 Part B's editor-only markers: the badge DEscape draws
for Invisible Objects, Map Revealers and Blockers, which have no art in-game.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_pasture_review_pack.py, whose helpers it shares.
Renders off-engine in Stepped, Sloped and Flat at three zoom levels, with
units and sprites on. Needs a configured AoE2:DE install (terrain textures,
and the game's visibility icon for revealers).

Run:
    tools/gen_marker_review_pack.py
    tools/gen_marker_review_pack.py --inject off-centre
    tools/gen_marker_review_pack.py --check      # writes nothing

Writes build/review_pack/markers/<opaque token>/, gitignored.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_review_pack as grp
import gen_seam_eyeball as seam_eyeball
import numpy as np

from descape import asset_source, editor_markers, render, render_cache, unit_kind, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit import review_pack as rp

PACK_ID = "markers"
OUT_DIR = grp.OUT_DIR
INVISIBLE_A = 1291
REVEALER = 837
BLOCKER = 1776
# A per-tile checkerboard, so each badge's own tile is visible (GOTCHAS: a uniform ground proves nothing about tiles).
CHECKER = (14, 0)  # DESERT_SAND, GRASS_1

# Along one screen-horizontal line (x + y constant), two tile steps apart.
_ROW = [
    (INVISIBLE_A, 1, 57, 63),
    (INVISIBLE_A, 2, 59, 61),
    (REVEALER, 1, 61, 59),
    (BLOCKER, 1, 63, 57),
]
_ROW_TILES = [(x, y) for _c, _p, x, y in _ROW]
_REVEALER_TILES = [(61, 59)]
# Where the revealer's game art is looked up when it must fail, as a missing install would.
_MISSING_ICON = "no/such/visibility.png"

# Mip levels of a 120x120 map (base tile_px 64): far-out 16, normal 64, zoomed-in 128.
_ZOOMS = {"s": -2, "m": 0, "l": 1}

# frame id -> (scene, style, mip, crop tiles)
_FRAMES = {
    **{f"stepped_{z}": ("row", "stepped", mip, _ROW_TILES) for z, mip in _ZOOMS.items()},
    # SlopedChunkCache renders only level 0 (the view scales it), so Sloped has one frame.
    "sloped_m": ("row", "sloped", 0, _ROW_TILES),
    **{f"flat_{z}": ("row", "flat", mip, _ROW_TILES) for z, mip in _ZOOMS.items()},
    "eye_a": ("revealer", "stepped", 1, _REVEALER_TILES),
    "eye_b": ("revealer_glyph", "stepped", 1, _REVEALER_TILES),
    "ground": ("blank", "stepped", 0, _ROW_TILES),
}
_MAX_ELEV = 3


def _elevation(style: str, x: int, y: int) -> int:
    # Sloped gets a ramp across the row, so badges sit on slopes; the others stay flat.
    return 0 if style != "sloped" else max(0, min(_MAX_ELEV, (x - 56) // 2))


def _scenario(scene: str, style: str):
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation = _elevation(style, tile.x, tile.y)
        tile.terrain_id = CHECKER[(tile.x + tile.y) % 2]
    units = [[] for _ in base.unit_manager.units]
    placed = {"row": _ROW, "revealer": [_ROW[2]], "revealer_glyph": [_ROW[2]], "blank": []}[scene]
    for i, (const, player, x, y) in enumerate(placed):
        units[player].append(SimpleNamespace(
            unit_const=const, x=x + 0.5, y=y + 0.5, z=0.0, rotation=0.0,
            reference_id=7000 + i, status=2, garrisoned_in_id=-1,
        ))
    return SimpleNamespace(
        map_manager=base.map_manager,
        unit_manager=SimpleNamespace(units=units),
        team_indices=base.team_indices,
        player_colors=base.player_colors,
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


def _box(cache, style: str, mip: int, tiles):
    if style == "flat":
        px = cache._mip_tile_px[mip]
        xs = [x for x, _y in tiles]
        ys = [y for _x, y in tiles]
        return (min(xs) - 1) * px, (min(ys) - 1) * px, (max(xs) + 2) * px, (max(ys) + 2) * px
    proj = cache.proj if style == "sloped" else cache._levels[mip].proj
    x0, y0, x1, y1 = grp._tiles_bbox_px(proj, tiles, _MAX_ELEV if style == "sloped" else 0)
    pad = proj.half_w
    return max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad // 2


def _render(frame_id: str) -> np.ndarray:
    scene, style, mip, tiles = _FRAMES[frame_id]
    scn = _scenario(scene, style)
    with _glyph_only(scene == "revealer_glyph"):
        cache = _cache(scn, style)
        x0, y0, x1, y1 = _box(cache, style, mip, tiles)
        return cache._composite_rect(mip, x0, y0, x1, y1)


@contextmanager
def _glyph_only(on: bool):
    """The revealer without its game art: the path a missing install takes."""
    real = editor_markers.VISIBILITY_ICON_SUBPATH
    if on:
        editor_markers.VISIBILITY_ICON_SUBPATH = _MISSING_ICON
        unit_sprites.clear_caches()
    try:
        yield
    finally:
        if on:
            editor_markers.VISIBILITY_ICON_SUBPATH = real
            unit_sprites.clear_caches()


# ----------------------------------------------------------------- injects


def _inject_no_markers():
    return unit_kind, "invisible_category", lambda const: None


def _inject_off_centre():
    real = unit_sprites.marker_for.__wrapped__

    @cache  # clear_caches() calls marker_for.cache_clear()
    def shifted(category, team_index, half_w):
        draw = real(category, team_index, half_w)
        # Half a tile down-right: the badge's centre lands on its tile's lower-right edge.
        return unit_sprites.SpriteDraw(draw.rgba, draw.hotspot_x - half_w // 2, draw.hotspot_y - half_w // 4)

    return unit_sprites, "marker_for", shifted


def _inject_no_tint():
    real = unit_sprites.marker_for.__wrapped__
    return unit_sprites, "marker_for", cache(lambda category, _team, half_w: real(category, 1, half_w))


def _inject_no_skip():
    """The coloured mark painted under each marker, as if it had not resolved to a sprite."""
    real = render.sprite_draws_by_anchor

    def unskipped(*args, **kwargs):
        return dataclasses.replace(real(*args, **kwargs), skip_ids=frozenset())

    return render, "sprite_draws_by_anchor", unskipped


def _inject_tiny_symbol():
    return editor_markers, "MIN_SYMBOL_PX", 1


def _inject_no_art():
    return editor_markers, "VISIBILITY_ICON_SUBPATH", _MISSING_ICON


# name -> builder returning one or a list of (module, attribute, replacement)
INJECTS = {
    "no-markers": _inject_no_markers,
    "off-centre": _inject_off_centre,
    "no-tint": _inject_no_tint,
    "no-skip": _inject_no_skip,
    "tiny-symbol": _inject_tiny_symbol,
    "no-art": _inject_no_art,
}
COARSE_INJECT = "no-markers"


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
            ("stepped_s", "A line of small badges on a checkered ground, zoomed far out."),
            ("stepped_m", "The same line of badges, at normal zoom."),
            ("stepped_l", "The same line of badges, zoomed in."),
            ("sloped_m", "The same line of badges on checkered, sloping ground, at normal zoom."),
            ("flat_s", "The same badges on a checkered ground seen straight from above, zoomed far out."),
            ("flat_m", "The same top-down view, at normal zoom."),
            ("flat_l", "The same top-down view, zoomed in."),
            ("eye_a", "Close-up of one badge."),
            ("eye_b", "Close-up of one badge, a second render."),
            ("ground", "Checkered ground, same framing as the normal-zoom line of badges."),
        )
    )
    checker = (
        "The ground is a checkerboard of two alternating ground textures, one cell per map tile "
        "(diamond-shaped cells in the angled views, squares in the top-down ones). "
    )
    checks = (
        rp.Check(
            id="badge_cell",
            question=(
                checker + "First describe where each badge sits relative to the checker cells around it. Then, "
                "judging by each badge's middle point (a badge on sloping ground may overhang its cell's edges "
                "a little): is every badge's middle in the middle of a single cell? ONE-CELL if so, STRADDLING "
                "if any badge's middle sits on or near a boundary or corner between cells."
            ),
            frames=("stepped_m", "sloped_m", "flat_m"),
            answers=("ONE-CELL", "STRADDLING", "CANNOT-TELL"),
        ),
        rp.Check(
            id="twin_colour",
            question=(
                "Two of the badges in each frame carry the same symbol as each other (an eye with a line "
                "through it). First describe the fill colour of each of those two badges. Then: are their fills "
                "clearly different colours from each other? DIFFERENT or SAME."
            ),
            frames=("stepped_m", "flat_l"),
            answers=("DIFFERENT", "SAME", "CANNOT-TELL"),
        ),
        rp.Check(
            id="fill_ground",
            question=(
                checker + "Look closely at the inside of each badge, around its symbol. First describe what you "
                "see there. Then: can the ground's texture be seen faintly through the badge's fill, or is the "
                "fill one flat opaque colour that hides the ground completely? SEE-THROUGH if the ground shows "
                "through every badge, OPAQUE if any badge's fill is a solid block of colour with no ground "
                "visible through it."
            ),
            frames=("stepped_l", "sloped_m"),
            answers=("SEE-THROUGH", "OPAQUE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="far_badges",
            question=(
                "In these far-out views, first describe what each badge looks like. Then: does any badge carry a "
                "white or light symbol or marking inside it, or is each one a plain filled shape with at most a "
                "darker rim? SYMBOL if any badge carries a light marking, PLAIN if none does."
            ),
            frames=("stepped_s", "flat_s"),
            answers=("SYMBOL", "PLAIN", "CANNOT-TELL"),
        ),
        rp.Check(
            id="eye_pair",
            question=(
                "Two close-ups each show one badge. First describe the symbol in each, in detail. Then: are the "
                "two symbols the same drawing, or two clearly different drawings (different shapes, colours or "
                "style, not just a slight shift)? SAME or DIFFERENT."
            ),
            frames=("eye_a", "eye_b"),
            answers=("SAME", "DIFFERENT", "CANNOT-TELL"),
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
        title="Map badge review",
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


def generate(out_dir: Path, inject: str | None) -> list[Path]:
    spec = _spec()
    rp.validate_spec(spec)
    pack_dir = out_dir / PACK_ID / grp._run_token(PACK_ID, inject)
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
    if not any(
        comp < coarse.get(fid, (0, 0))[1]
        for name, t in reach.items() if name != COARSE_INJECT
        for fid, (_px, comp) in t.items()
    ):
        failures.append("no inject is smaller than the coarse one on a shared frame")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(f"OK: {PACK_ID} frames couple to checks, land in band, all {len(INJECTS)} injects bite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--inject", choices=sorted(INJECTS))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if asset_source.get_install_path() is None:
        raise SystemExit("needs a configured AoE2:DE install: terrain textures and the revealer's game art")
    if args.check:
        check()
        return
    for path in generate(args.out_dir, args.inject):
        print(f"wrote {path}")
    print(f"\nrun `{args.inject or 'clean'}` is directory {grp._run_token(PACK_ID, args.inject)}")


if __name__ == "__main__":
    main()
