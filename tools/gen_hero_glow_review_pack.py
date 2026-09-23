#!/usr/bin/env python3
"""Review pack for GH #39's hero glow: the gold ring DEscape bakes round a
hero's sprite (the game draws its own procedurally, with no art to load).

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_marker_review_pack.py. Renders off-engine
through the app's own Stepped and Sloped chunk caches with units and sprites
on and the default View > Layers state. Needs a configured AoE2:DE install
(the unit sprites and terrain textures).

Run:
    tools/gen_hero_glow_review_pack.py
    tools/gen_hero_glow_review_pack.py --inject offset
    tools/gen_hero_glow_review_pack.py --check      # writes nothing

Writes build/review_pack/hero_glow/<opaque token>/, gitignored.
"""

from __future__ import annotations

import argparse
import dataclasses
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

from descape import asset_source, render, render_cache, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit import review_pack as rp

PACK_ID = "hero_glow"
OUT_DIR = grp.OUT_DIR
HERO = 165  # HCHARL, infantry
NON_HERO = 4  # ARCHR: carries a HeroGlow graphic but no hero bits, so no ring
HOUSE = 70
GRASS = 0

# A screen-horizontal pair (x - y constant): the hero is the left figure.
_PAIR = [(HERO, 1, 59, 59), (NON_HERO, 1, 60, 60)]
_PAIR_TILES = [(59, 59), (60, 60)]
# The house's footprint starts one row later in paint order, so it is drawn over the hero.
# The hero stands near his tile's front edge, so the roof covers his lower half.
_BEHIND = [(HERO, 1, 60, 60.45), (HOUSE, 2, 59, 61)]
_BEHIND_TILES = [(60, 60), (59, 61), (60, 62)]
_ALONE = [(NON_HERO, 1, 59, 59)]
_ALONE_TILES = [(59, 59)]

# frame id -> (placements, style, mip, crop tiles)
_FRAMES = {
    "pair": (_PAIR, "stepped", 1, _PAIR_TILES),
    "sloped_pair": (_PAIR, "sloped", 0, _PAIR_TILES),
    "behind": (_BEHIND, "stepped", 1, _BEHIND_TILES),
    "alone": (_ALONE, "stepped", 1, _ALONE_TILES),
}


def _scenario(placements):
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = GRASS
    units = [[] for _ in base.unit_manager.units]
    for i, (const, player, x, y) in enumerate(placements):
        span = render.tile_span(const, render.NON_BUILDING_SPAN)
        units[player].append(SimpleNamespace(
            unit_const=const, x=x + span[0] / 2, y=y + span[1] / 2, z=0.0, rotation=0.0,
            reference_id=8000 + i, status=2, garrisoned_in_id=-1,
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


def _box(cache, style: str, mip: int, tiles):
    proj = cache.proj if style == "sloped" else cache._levels[mip].proj
    x0, y0, x1, y1 = grp._tiles_bbox_px(proj, tiles, 0)
    # Sprites stand up to about two tiles above their anchor.
    return max(0, x0 - proj.half_w // 2), max(0, y0 - 4 * proj.half_h), x1 + proj.half_w // 2, y1 + proj.half_h // 2


def _render(frame_id: str) -> np.ndarray:
    placements, style, mip, tiles = _FRAMES[frame_id]
    cache = _cache(_scenario(placements), style)
    x0, y0, x1, y1 = _box(cache, style, mip, tiles)
    return cache._composite_rect(mip, x0, y0, x1, y1)


# ----------------------------------------------------------------- injects

_OFFSET_PX = 4


def _ring_only(glowed: unit_sprites.SpriteDraw, plain_alpha: np.ndarray) -> np.ndarray:
    """The ring's own px of a _with_glow() result, the sprite's px cleared."""
    r = unit_sprites.HERO_GLOW_RADIUS
    ring = glowed.rgba.copy()
    h, w = plain_alpha.shape
    ring[r:r + h, r:r + w][plain_alpha > 0] = 0
    return ring


def _inject_offset():
    """The ring shifted 4 px down-right of the sprite, the sprite still drawn on top."""
    real = unit_sprites._with_glow

    def shifted(draw):
        glowed = real(draw)
        ring = _ring_only(glowed, draw.rgba[..., 3])
        r, k = unit_sprites.HERO_GLOW_RADIUS, _OFFSET_PX
        h, w = draw.rgba.shape[:2]
        pad = r + k
        out = np.zeros((h + 2 * pad, w + 2 * pad, 4), dtype=np.uint8)
        out[pad - r + k:pad - r + k + ring.shape[0], pad - r + k:pad - r + k + ring.shape[1]] = ring
        body = out[pad:pad + h, pad:pad + w]
        mask = draw.rgba[..., 3] > 0
        body[mask] = draw.rgba[mask]
        return unit_sprites.SpriteDraw(out, draw.hotspot_x + pad, draw.hotspot_y + pad)

    return unit_sprites, "_with_glow", shifted


def _inject_unoccluded():
    """Each ring painted again after whatever sprite is in front of it, so no
    later sprite covers it."""
    real = render.sprite_draws_by_anchor

    def over(*args, **kwargs):
        layer = real(*args, **kwargs)
        order = sorted(layer.by_anchor, key=lambda k: (k[1], k[0]))
        by_anchor = {k: list(v) for k, v in layer.by_anchor.items()}
        bboxes = dict(layer.bboxes)
        for key in order:
            for draw, px, py in layer.by_anchor[key]:
                # The opaque inner ring only: re-painting the half-alpha one would darken it everywhere.
                gold = np.all(draw.rgba[..., :3] == unit_sprites.HERO_GLOW_GOLD, axis=2) & (draw.rgba[..., 3] == 255)
                if not gold.any():
                    continue
                ring = draw.rgba.copy()
                ring[~gold] = 0
                last = order[-1]
                by_anchor[last].append((unit_sprites.SpriteDraw(ring, draw.hotspot_x, draw.hotspot_y), px, py))
                bx = bboxes[key]
                lb = bboxes[last]
                bboxes[last] = (min(lb[0], bx[0]), min(lb[1], bx[1]), max(lb[2], bx[2]), max(lb[3], bx[3]))
        return dataclasses.replace(layer, by_anchor=by_anchor, bboxes=bboxes)

    return render, "sprite_draws_by_anchor", over


def _inject_no_glow():
    return unit_sprites, "_with_glow", lambda draw: draw


def _inject_halo():
    """COARSE: a solid gold blob 12 px round every hero, the sprite on top."""
    def halo(draw):
        k = 12
        h, w = draw.rgba.shape[:2]
        out = np.zeros((h + 2 * k, w + 2 * k, 4), dtype=np.uint8)
        mask = np.zeros(out.shape[:2], dtype=bool)
        mask[k:k + h, k:k + w] = draw.rgba[..., 3] > 0
        ys, xs = np.nonzero(mask)
        grown = np.zeros_like(mask)
        grown[max(0, ys.min() - k):ys.max() + k + 1, max(0, xs.min() - k):xs.max() + k + 1] = True
        out[grown] = (*unit_sprites.HERO_GLOW_GOLD, 255)
        body = out[k:k + h, k:k + w]
        inner = draw.rgba[..., 3] > 0
        body[inner] = draw.rgba[inner]
        return unit_sprites.SpriteDraw(out, draw.hotspot_x + k, draw.hotspot_y + k)

    return unit_sprites, "_with_glow", halo


# name -> builder returning one or a list of (module, attribute, replacement)
INJECTS = {
    "halo": _inject_halo,
    "offset": _inject_offset,
    "unoccluded": _inject_unoccluded,
    "no-glow": _inject_no_glow,
}
COARSE_INJECT = "halo"
SPECIFICITY_FRAME = "alone"


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
            ("pair", "Two figures standing side by side on grass."),
            ("sloped_pair", "The same two figures, in a second render mode."),
            ("behind", "A figure standing just behind a house."),
            ("alone", "One figure standing on grass."),
        )
    )
    checks = (
        rp.Check(
            id="left_edge",
            question=(
                "Look at the LEFT figure. First describe its outermost edge all the way round: what colour the "
                "pixels just outside its own drawing are, and whether that changes from side to side. Then: is "
                "the figure outlined by a thin band of one bright colour that touches the figure's own pixels "
                "everywhere, with no grass showing between band and figure on any side? HUGGING if so. GAPPED if "
                "there is such a band but grass shows between it and the figure somewhere, or the band is missing "
                "along one side. NONE if there is no such band at all."
            ),
            frames=("pair", "sloped_pair"),
            answers=("HUGGING", "GAPPED", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="band_width",
            question=(
                "Look at the LEFT figure again. First describe any band or patch of colour around it and how wide "
                "it is compared with the figure itself. Then: is it a narrow line following the figure's edge, or "
                "a broad patch or block that fills much of the space around the figure? THIN for a narrow line, "
                "WIDE for a broad patch, NONE if there is neither."
            ),
            frames=("pair",),
            answers=("THIN", "WIDE", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="house_front",
            question=(
                "A figure stands just behind a house, and the house's roof covers part of the figure. First "
                "describe what you see where the roof overlaps the figure. Then: is any bright band or outline "
                "from the figure's edge drawn on top of the roof, or does the roof cover everything behind it? "
                "ON-ROOF if any of it is drawn over the roof, COVERED if none is."
            ),
            frames=("behind",),
            answers=("ON-ROOF", "COVERED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="lone_edge",
            question=(
                "First describe the figure's outermost edge. Then: is the figure outlined by a band of one bright "
                "colour that is not part of the figure's own clothing or equipment? BAND or NO-BAND."
            ),
            frames=("alone",),
            answers=("BAND", "NO-BAND", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Unit figure review",
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
        if SPECIFICITY_FRAME in touched:
            failures.append(f"inject {name}: reached the specificity frame")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    coarse = reach[COARSE_INJECT]
    for name, touched in reach.items():
        if name == COARSE_INJECT:
            continue
        for fid, (_px, comp) in touched.items():
            if fid in coarse and comp >= coarse[fid][1]:
                failures.append(f"inject {name}: {fid} component {comp} not under the coarse one's")
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
        raise SystemExit("needs a configured AoE2:DE install: unit sprites and terrain textures")
    if args.check:
        check()
        return
    for path in generate(args.out_dir, args.inject):
        print(f"wrote {path}")
    print(f"\nrun `{args.inject or 'clean'}` is directory {grp._run_token(PACK_ID, args.inject)}")


if __name__ == "__main__":
    main()
