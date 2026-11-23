#!/usr/bin/env python3
"""Review pack for a placed Pasture's annex-tree art (GH #66): hut, corner
posts, edge fences, seeded variants, the pasture drape and depth order.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script because that tool's frames and injects are shadow-band specific.
Needs a real AoE2:DE install (sprites); renders off-engine via
render_iso.composite_rect_iso, Stepped, with units and sprites on.

Run:
    tools/gen_pasture_review_pack.py
    tools/gen_pasture_review_pack.py --inject swap-fences
    tools/gen_pasture_review_pack.py --check      # writes nothing

Writes build/review_pack/pasture/<opaque token>/, gitignored.
"""

from __future__ import annotations

import argparse
import copy
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

from descape import asset_source, game_dataset, render_bbox, render_iso, render_sprites, render_units, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit import review_pack as rp
from testkit.render_family import set_render_global

PACK_ID = "pasture"
OUT_DIR = grp.OUT_DIR
TILE_PX = grp.TILE_PX
PASTURE = 1897
VILLAGER = 83
SHEEP = 594
WAR_ELEPHANT = 239
# DESERT_SAND: contrasts with the green pasture drape (117).
GROUND_TERRAIN = 14
GAIA = 0
PLAYER = 1

# Scene -> list of (unit_const, x, y, reference_id, owner), each owner's list in this
# order (GAIA's walks first). Every pasture sits on tiles 58..61.
_SCENES = {
    # Seeds picked so the two front edges show all 5 distinct fence images (fencesA
    # has 3, fencesB 2) with 8 of 10 pieces differing from the unseeded frame.
    "single": [(PASTURE, 60.0, 60.0, 9353, PLAYER)],
    "single_b": [(PASTURE, 60.0, 60.0, 9033, PLAYER)],
    # A villager just in front of the hut, on tile (59, 61), later in depth than the
    # hut's slot but listed first, so a missing slot paints the hut over his head.
    "inside": [(VILLAGER, 59.7, 61.0, 5001, PLAYER), (PASTURE, 60.0, 60.0, 9133, PLAYER)],
    # TASK-016: GAIA sheep (walked first) on the back corner (62, 58)'s slot tile. No one sheep
    # overlaps both its post and a fence (measured), so one takes the post, one the fence at (61.3, 58).
    "back_corner": [
        (SHEEP, 61.9, 58.1, 5002, GAIA), (SHEEP, 61.3, 58.15, 5004, GAIA), (PASTURE, 60.0, 60.0, 9133, PLAYER),
    ],
    # An elephant inside the front corner (58, 62), listed after the pasture: it
    # overlaps the post and the fence at (58.7, 62).
    "front_corner": [(PASTURE, 60.0, 60.0, 9133, PLAYER), (WAR_ELEPHANT, 58.3, 61.9, 5003, PLAYER)],
    "blank": [],
}

_PLOT_TILES = [(tx, ty) for tx in (57, 62) for ty in (57, 62)]
# Where the two front (screen-bottom) edges meet: column x = 58 and row y = 61.
_SIDE_TILES = [(58, 59), (58, 60), (58, 61), (59, 61), (60, 61)]
_FIGURE_TILES = [(tx, ty) for tx in (59, 60) for ty in (59, 61)]
# The four tiles around each corner point.
_BACK_TILES = [(tx, ty) for tx in (61, 62) for ty in (57, 58)]
_FRONT_TILES = [(tx, ty) for tx in (57, 58) for ty in (61, 62)]

# Fences are 2-3 px at the app's own tile size, too small to read a direction off,
# so the side close-ups render at twice it (sprite scale 4/3, still the app's resize).
_SIDE_TILE_PX = 2 * TILE_PX

# frame id -> (scene, crop tiles, tile heights of headroom above, tile_px)
_FRAMES = {
    "plot_raw": ("single", _PLOT_TILES, 4, TILE_PX),
    "sides_a_raw": ("single", _SIDE_TILES, 1, _SIDE_TILE_PX),
    "sides_b_raw": ("single_b", _SIDE_TILES, 1, _SIDE_TILE_PX),
    "figure_raw": ("inside", _FIGURE_TILES, 3, TILE_PX),
    "back_raw": ("back_corner", _BACK_TILES, 3, _SIDE_TILE_PX),
    "front_raw": ("front_corner", _FRONT_TILES, 3, _SIDE_TILE_PX),
    "ground_raw": ("blank", _PLOT_TILES, 4, TILE_PX),
}


def _scenario(scene: str):
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = GROUND_TERRAIN
    units = [[] for _ in base.unit_manager.units]
    for const, x, y, ref, owner in _SCENES[scene]:
        units[owner].append(SimpleNamespace(
            unit_const=const, x=x, y=y, z=0.0, rotation=7.0 if const == PASTURE else 0.0,
            reference_id=ref, status=2, garrisoned_in_id=-1,
        ))
    return SimpleNamespace(
        map_manager=base.map_manager,
        unit_manager=SimpleNamespace(units=units),
        team_indices=base.team_indices,
        player_colors=base.player_colors,
        # The library's tiles look their scenario up by uuid, so it must stay alive.
        _base=base,
    )


def _projection(scenario, tile_px: int):
    mm = scenario.map_manager
    return grp.ig.canvas_size_and_origin(
        mm.map_width, mm.map_height, tile_px, grp.ig.MIN_ELEVATION, grp.ig.MAX_ELEVATION, elev_step_pct=50
    )


def _box(proj, tiles, headroom: int):
    x0, y0, x1, y1 = grp._tiles_bbox_px(proj, tiles, 0)
    return x0, max(0, y0 - headroom * proj.half_h), x1, y1


def _render(frame_id: str) -> np.ndarray:
    scene, tiles, headroom, tile_px = _FRAMES[frame_id]
    scn = _scenario(scene)
    proj = _projection(scn, tile_px)
    mm = scn.map_manager
    elevations = np.zeros((mm.map_height, mm.map_width), dtype=np.int64)
    sprites = render_sprites.sprite_draws_by_anchor(scn, proj, elevations)
    units_by_tile = render_units._units_by_tile(scn)
    bboxes = render_sprites.merge_sprite_bboxes(
        render_bbox._building_bboxes_iso(scn, mm.map_width, mm.map_height, proj, elevations), sprites
    )
    x0, y0, x1, y1 = _box(proj, tiles, headroom)
    return render_iso.composite_rect_iso(
        scn, x0, y0, x1, y1, elevations, proj, tile_px, units_by_tile, bboxes,
        with_units=True, sprites=sprites,
    )


# ----------------------------------------------------------------- injects


def _is_fence(piece) -> bool:
    return "fences" in piece["file_name"]


def _no_fences(pieces):
    return [p for p in pieces if not _is_fence(p)]


def _swap_fences(pieces):
    swap = {"fencesA": "fencesB", "fencesB": "fencesA"}
    out = []
    for piece in pieces:
        name = piece["file_name"]
        hit = next((old for old in swap if old in name), None)
        out.append(piece if hit is None else {**piece, "file_name": name.replace(hit, swap[hit])})
    return out


def _unseeded(pieces):
    return [{k: v for k, v in p.items() if k != "seeded"} for p in pieces]


def _no_slots(pieces):
    # Pre-slot behaviour: every piece paints at the unit's single anchor tile, the footprint's last.
    return [{k: v for k, v in p.items() if k != "slot"} for p in pieces]


def _posts_centred(pieces):
    return [{**p, "dx": 0, "dy": 0} if "corner_posts" in p["file_name"] else p for p in pieces]


# name -> (piece transform or None, drape off?, list order?). list-order is the
# pre-TASK-016 merge: sprites sharing a tile paint in unit-list order.
INJECTS = {
    "no-fences": (_no_fences, False, False),
    "swap-fences": (_swap_fences, False, False),
    "unseeded": (_unseeded, False, False),
    "no-slots": (_no_slots, False, False),
    "posts-centred": (_posts_centred, False, False),
    "no-drape": (None, True, False),
    "list-order": (None, False, True),
}
COARSE_INJECT = "no-fences"


@contextmanager
def _patched(inject: str | None):
    import pytest

    real_map = unit_sprites.graphic_map
    game = game_dataset.active()
    saved_draped = render_sprites._DRAPED_BY_GAME[game]
    unit_sprites._scaled_cache.clear()
    unit_sprites.clear_slice_cache()
    mp = pytest.MonkeyPatch()
    if inject is not None:
        transform, drape_off, list_order = INJECTS[inject]
        if transform is not None:
            table = copy.deepcopy(real_map())
            for const in (1893, 1897):
                table[const]["pieces"] = transform(table[const]["pieces"])
            unit_sprites.graphic_map = lambda: table
        if drape_off:
            render_sprites._DRAPED_BY_GAME[game] = frozenset()
        if list_order:
            set_render_global(mp, "SUBTILE_SPRITE_ORDER", False)
    try:
        yield
    finally:
        mp.undo()
        unit_sprites.graphic_map = real_map
        render_sprites._DRAPED_BY_GAME[game] = saved_draped
        unit_sprites._scaled_cache.clear()
        unit_sprites.clear_slice_cache()


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = (
        rp.Frame("plot_raw", "plot_raw.png", "raw", caption="One building on its plot, seen from above at an angle."),
        rp.Frame("sides_a_raw", "sides_a_raw.png", "raw", caption="Close-up of the two near sides of one plot."),
        rp.Frame("sides_b_raw", "sides_b_raw.png", "raw", caption="The same close-up of a second, separately placed plot."),
        rp.Frame("figure_raw", "figure_raw.png", "raw", caption="Close-up of the building with a small standing figure beside it."),
        rp.Frame("back_raw", "back_raw.png", "raw", caption="Close-up of a plot's top corner with two animals standing near it."),
        rp.Frame("front_raw", "front_raw.png", "raw", caption="Close-up of a plot's bottom corner with a large animal standing at it."),
        rp.Frame("ground_raw", "ground_raw.png", "raw", caption="Bare ground, same terrain and same framing as the single-plot frame."),
    )
    checks = (
        rp.Check(
            id="plot_ground",
            question=(
                "First describe the ground under and around the building. Then: is there a roughly "
                "diamond-shaped patch of ground around the building whose texture clearly differs from the "
                "ground further out? DIFFERENT if so, SAME if the ground is one texture throughout."
            ),
            frames=("plot_raw",),
            answers=("DIFFERENT", "SAME", "CANNOT-TELL"),
        ),
        rp.Check(
            id="corner_objects",
            question=(
                "The plot is a diamond with four corners. First describe what stands at each corner, if "
                "anything. Then: do all four corners each carry a tall thin upright post standing on the "
                "corner point itself? Ignore the building in the middle and any low broken fence pieces "
                "along the sides. ALL-FOUR, SOME (one to three), or NONE."
            ),
            frames=("plot_raw",),
            answers=("ALL-FOUR", "SOME", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="side_pieces",
            question=(
                "Each close-up shows two sides of a plot, marked by a thin coloured outline, meeting at "
                "the bottom corner. First describe any low, short fence pieces you can see along those "
                "sides (not the tall posts at the corners). Then the closed question: take each such "
                "piece's long direction. Does every piece lie ALONG the outline of the side it sits on, or "
                "do the pieces on one side run across that side's outline at a steep angle, parallel to "
                "the OTHER side instead? ALONG if all follow their own side, CROSSING if some cross it, "
                "NO-PIECES if there are no such pieces at all."
            ),
            frames=("sides_a_raw", "sides_b_raw"),
            answers=("ALONG", "CROSSING", "NO-PIECES", "CANNOT-TELL"),
        ),
        rp.Check(
            id="piece_variety",
            question=(
                "Look only at the low fence pieces along the sides in the two close-ups, not the tall corner "
                "posts. First describe how the pieces compare with each other, within one close-up and "
                "between the two. Then: are they visibly different shapes from one another (longer or "
                "shorter, broken differently, more or fewer rails), and do the two plots differ from each "
                "other? VARIED if pieces differ within a side and the two plots are not copies of each "
                "other, IDENTICAL if every piece along a side is the same shape or the two close-ups are "
                "copies of each other, NO-PIECES if there are none."
            ),
            frames=("sides_a_raw", "sides_b_raw"),
            answers=("VARIED", "IDENTICAL", "NO-PIECES", "CANNOT-TELL"),
        ),
        rp.Check(
            id="figure_order",
            question=(
                "Find the small standing figure next to the round building. First describe where the figure "
                "stands relative to the building. Then: where they overlap on screen, which is drawn on top? "
                "BUILDING-OVER-FIGURE if the building hides "
                "part of the figure, FIGURE-OVER-BUILDING if the figure is drawn over the building, "
                "NO-OVERLAP if they do not overlap at all."
            ),
            frames=("figure_raw",),
            answers=("BUILDING-OVER-FIGURE", "FIGURE-OVER-BUILDING", "NO-OVERLAP", "CANNOT-TELL"),
        ),
        rp.Check(
            id="top_post",
            question=(
                "Find the plot's top corner point and the animal standing closest to it. First describe what "
                "stands on that corner point and where that animal stands relative to it. Then, about the tall "
                "thin upright post: ANIMAL-OVER-POST if a post stands on the corner point and that animal hides "
                "part of it, POST-OVER-ANIMAL if a post there hides part of that animal, NO-POST if no tall "
                "post stands on the corner point itself."
            ),
            frames=("back_raw",),
            answers=("ANIMAL-OVER-POST", "POST-OVER-ANIMAL", "NO-POST", "CANNOT-TELL"),
        ),
        rp.Check(
            id="top_fence",
            question=(
                "Look at the low fence pieces running from the top corner along the plot's two upper sides, "
                "not the tall corner post, and at both animals. First describe any fence piece that lies near "
                "an animal. Then: FENCE-OVER-ANIMAL if any such fence piece hides part of either animal, CLEAR "
                "if none does, including when no fence piece is there at all."
            ),
            frames=("back_raw",),
            answers=("FENCE-OVER-ANIMAL", "CLEAR", "CANNOT-TELL"),
        ),
        rp.Check(
            id="bottom_post",
            question=(
                "Find the large animal and the plot's bottom corner point. First describe what stands on that "
                "corner point and where the animal stands relative to it. Then, about the tall thin upright "
                "post: POST-OVER-ANIMAL if a post stands on the corner point and hides part of the animal, "
                "ANIMAL-OVER-POST if the animal hides part of a post there, NO-POST if no tall post stands on "
                "the corner point itself."
            ),
            frames=("front_raw",),
            answers=("POST-OVER-ANIMAL", "ANIMAL-OVER-POST", "NO-POST", "CANNOT-TELL"),
        ),
        rp.Check(
            id="bottom_fence",
            question=(
                "Look at the low fence pieces running from the bottom corner along the plot's two lower sides, "
                "not the tall corner post. First describe any that lie near the large animal. Then: "
                "ANIMAL-OVER-FENCE if the animal hides part of any such fence piece, CLEAR if it hides none, "
                "including when no fence piece is there at all."
            ),
            frames=("front_raw",),
            answers=("ANIMAL-OVER-FENCE", "CLEAR", "CANNOT-TELL"),
        ),
        rp.Check(
            id="ground_objects",
            question=(
                "Are there any objects in this frame -- posts, fence pieces, buildings, outlines -- or only "
                "bare ground texture? OBJECTS-PRESENT or GROUND-ONLY."
            ),
            frames=("ground_raw",),
            answers=("OBJECTS-PRESENT", "GROUND-ONLY", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Building plot review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from an isometric map editor for a strategy game. Each frame is a "
            "crop, enlarged by a whole-number factor with no smoothing. Judge only what you can actually see."
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
        if "ground_raw" in touched:
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
        raise SystemExit("needs a configured AoE2:DE install: the pack judges sprite art")
    if args.check:
        check()
        return
    for path in generate(args.out_dir, args.inject):
        print(f"wrote {path}")
    print(f"\nrun `{args.inject or 'clean'}` is directory {grp._run_token(PACK_ID, args.inject)}")


if __name__ == "__main__":
    main()
