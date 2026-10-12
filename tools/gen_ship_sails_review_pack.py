#!/usr/bin/env python3
"""Review pack for GH #20: sailing ships drawn with their sails and owner
colour (unit_graphic_map.json's ship `pieces`), and Butterflies 1-3 drawn from
their particle atlas instead of as a coloured mark.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_pasture_review_pack.py, whose off-engine render
it shares: render.composite_rect_iso, Stepped, units and sprites on, no Qt.
Needs a configured AoE2:DE install (sprites, particles and terrain).

The PACK_ID and every frame/check name are neutral on purpose: frame paths
appear in REVIEW.md, and nothing a reviewer reads may name the subject.

Frames a-e are Galleon, War Galley, Galley, Transport Ship and Fire Ship, four
facings each, one owner colour per frame. Frame f is the control, the Fishing
Ship, a single graphic no inject may touch. Frame g is the three butterflies
on neighbouring sand tiles. Frame h is three consecutive frames of one
butterfly loop read through descape.particle_art, the middle one stored
rotated in the atlas: the app itself draws only a loop's first frame, which is
upright in all three effects, so this is the only frame the rotation inject
can reach. Frame i is open water in frame a's framing.

Injects:
    old-table  coarse: the graphic map as it was (no ship pieces, no butterfly
               entries), i.e. bare hulls and coloured marks.
    sails-low  every sail's dy sign flipped, so it hangs below its mast.
    turned     a rotated atlas frame turned the other way (a half turn out).

Run:
    tools/gen_ship_sails_review_pack.py --out-dir <neutral dir>
    tools/gen_ship_sails_review_pack.py --out-dir <neutral dir> --replica 1
    tools/gen_ship_sails_review_pack.py --out-dir <neutral dir> --inject sails-low
    tools/gen_ship_sails_review_pack.py --check      # writes nothing

Writes <out-dir>/water_objects/<opaque token>/ (build/review_pack/ by default, gitignored).
"""

from __future__ import annotations

import argparse
import copy
import math
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

from descape import asset_source, particle_art, render, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit import review_pack as rp

PACK_ID = "water_objects"
OUT_DIR = grp.OUT_DIR
# Twice the app's own tile size (sprite scale 4/3), a level its mip ladder has.
TILE_PX = 2 * grp.TILE_PX

GALLEON, WAR_GALLEY, GALLEY, TRANSPORT, FIRE_SHIP, FISHING_SHIP = 442, 21, 539, 545, 529, 13
BUTTERFLIES = (1608, 1609, 1610)
WATER, SAND = 22, 14

# A 2x2 block of boats, 256 px apart across and down at TILE_PX.
_BOAT_TILES = [(58, 58), (61, 61), (54, 62), (57, 65)]
_FACINGS = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
# frame id -> (ship const, owning player, that player's colour as the caption states it).
# No blue or cyan owner: the water is blue.
_BOATS = {
    "a": (GALLEON, 2, "red"),
    "b": (WAR_GALLEY, 4, "yellow"),
    "c": (GALLEY, 3, "green"),
    "d": (TRANSPORT, 8, "orange"),
    "e": (FIRE_SHIP, 6, "purple"),
    "f": (FISHING_SHIP, 2, "red"),
}
SAIL_FRAMES = ("a", "b", "c", "d", "e")
CONTROL_FRAME = "f"
# Side by side, 128 px apart: a butterfly is a few px, so the crop stays tight.
_INSECT_TILES = [(58, 58), (59, 59), (60, 60)]

# Frame h: three consecutive loop frames of one effect, the middle one `rotated`.
LOOP_EFFECT = "butterflies_2_low"
LOOP_INDICES = (0, 1, 2)
_LOOP_SOURCE = 400
_LOOP_GROUND = (86, 122, 58)
_LOOP_GAP = 6


def _scenario(units: list[tuple[int, int, float, float, float]], ground: int):
    """A flat map of one terrain; `units` is (player, const, x, y, rotation)."""
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = ground
    placed = [[] for _ in base.unit_manager.units]
    for ref, (player, const, x, y, rotation) in enumerate(units):
        placed[player].append(SimpleNamespace(
            unit_const=const, x=x, y=y, z=0.0, rotation=rotation,
            reference_id=1000 + ref, status=2, garrisoned_in_id=-1,
        ))
    return SimpleNamespace(
        map_manager=base.map_manager,
        unit_manager=SimpleNamespace(units=placed),
        team_indices=base.team_indices,
        player_colors=base.player_colors,
        # The library's tiles look their scenario up by uuid, so it must stay alive.
        _base=base,
    )


def _boat_scene(frame_id: str):
    const, player, _colour = _BOATS[frame_id]
    return _scenario(
        [(player, const, tx + 0.5, ty + 0.5, facing)
         for (tx, ty), facing in zip(_BOAT_TILES, _FACINGS, strict=True)],
        WATER,
    )


def _insect_scene():
    return _scenario(
        [(0, const, tx + 0.5, ty + 0.5, 0.0)
         for const, (tx, ty) in zip(BUTTERFLIES, _INSECT_TILES, strict=True)],
        SAND,
    )


def _render(scn, tiles, pad_px: int, head: int, foot: int) -> np.ndarray:
    """One Stepped crop round `tiles`, `head`/`foot` half-tile heights taller."""
    mm = scn.map_manager
    proj = grp.ig.canvas_size_and_origin(
        mm.map_width, mm.map_height, TILE_PX, grp.ig.MIN_ELEVATION, grp.ig.MAX_ELEVATION, elev_step_pct=50
    )
    elevations = np.zeros((mm.map_height, mm.map_width), dtype=np.int64)
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)
    units_by_tile = render._units_by_tile(scn)
    bboxes = render.merge_sprite_bboxes(
        render._building_bboxes_iso(scn, mm.map_width, mm.map_height, proj, elevations), sprites
    )
    x0, y0, x1, y1 = grp._tiles_bbox_px(proj, tiles, 0)
    return render.composite_rect_iso(
        scn, max(0, x0 - pad_px), max(0, y0 - head * proj.half_h), x1 + pad_px, y1 + foot * proj.half_h,
        elevations, proj, TILE_PX, units_by_tile, bboxes, with_units=True, sprites=sprites,
    )


def _boat_frame(scn) -> np.ndarray:
    # A Galleon reaches 104 px to each side of its anchor and 184 px above it here.
    return _render(scn, _BOAT_TILES, 56, 5, 1)


def _loop_planes() -> list[np.ndarray]:
    """Each of LOOP_INDICES on its own 400 px source canvas, placed by its hotspot."""
    planes = []
    for index in LOOP_INDICES:
        plane = np.zeros((_LOOP_SOURCE, _LOOP_SOURCE, 4), dtype=np.uint8)
        frame = particle_art.native_frame(LOOP_EFFECT, index)
        if frame is not None:
            rgba, _playercolor, hx, hy = frame
            x0, y0 = _LOOP_SOURCE // 2 - hx, _LOOP_SOURCE // 2 - hy
            plane[y0 : y0 + rgba.shape[0], x0 : x0 + rgba.shape[1]] = rgba
        planes.append(plane)
    return planes


# Fixed, not measured per run: every run must crop the same source px.
_LOOP_BOX = slice(80, 280), slice(120, 330)


def _loop_ink_outside_box() -> int:
    total = 0
    for plane in _loop_planes():
        ink = plane[..., 3] > 0
        total += int(ink.sum()) - int(ink[_LOOP_BOX].sum())
    return total


def _loop_frame() -> np.ndarray:
    """LOOP_INDICES side by side on flat ground colour, all cropped to one box."""
    ground = np.array(_LOOP_GROUND, dtype=np.float32)
    panels = []
    for plane in _loop_planes():
        cut = plane[_LOOP_BOX]
        alpha = cut[..., 3:4].astype(np.float32) / 255.0
        panels.append((ground * (1 - alpha) + cut[..., :3] * alpha).astype(np.uint8))
        panels.append(np.full((cut.shape[0], _LOOP_GAP, 3), 235, dtype=np.uint8))
    return np.concatenate(panels[:-1], axis=1)


def _rgb(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[..., :3])


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    with _patched(inject):
        images = {fid: _rgb(_boat_frame(_boat_scene(fid))) for fid in _BOATS}
        images["g"] = _rgb(_render(_insect_scene(), _INSECT_TILES, 40, 4, 1))
        images["h"] = _loop_frame()
        images["i"] = _rgb(_boat_frame(_scenario([], WATER)))
    return images


# ----------------------------------------------------------------- injects


def _old_table(table: dict) -> dict:
    """The table before GH #20: bare hulls, and no butterfly entries at all."""
    out = {}
    for const, entry in table.items():
        if entry.get("particle"):
            continue
        is_ship = entry["file_name"].startswith("u_shp_")
        out[const] = {k: v for k, v in entry.items() if k != "pieces"} if is_ship else entry
    return out


def _sails_low(table: dict) -> dict:
    """Every sail's dy sign flipped: it lands as far below its seat as it should be above."""
    out = copy.deepcopy(table)
    for entry in out.values():
        if entry["file_name"].startswith("u_shp_") and "pieces" in entry:
            for piece in entry["pieces"]:
                if not piece.get("parent"):
                    piece["dy"] = -piece["dy"]
    return out


def _turned_cut(real_cut):
    """particle_art._cut with a rotated record turned the other way."""
    def cut(pixels, record):
        result = real_cut(pixels, record)
        if result is None or not record.get("rotated"):
            return result
        rgba, hx, hy = result
        return np.ascontiguousarray(np.rot90(rgba, 2)), hx, hy
    return cut


# name -> (graphic-map transform or None, turn rotated atlas frames?)
INJECTS = {
    "old-table": (_old_table, False),
    "sails-low": (_sails_low, False),
    "turned": (None, True),
}
COARSE_INJECT = "old-table"
# Which frames each inject may change. Frame i (open water) and the control are in none.
INJECT_REACH = {
    "old-table": frozenset(SAIL_FRAMES) | {"g"},
    "sails-low": frozenset(SAIL_FRAMES),
    "turned": frozenset({"h"}),
}


@contextmanager
def _patched(inject: str | None):
    real_map = unit_sprites.graphic_map
    real_cut = particle_art._cut
    unit_sprites.clear_caches()
    if inject is not None:
        transform, turn = INJECTS[inject]
        if transform is not None:
            table = transform(real_map())
            unit_sprites.graphic_map = lambda: table
        if turn:
            particle_art._cut = _turned_cut(real_cut)
    try:
        yield
    finally:
        unit_sprites.graphic_map = real_map
        particle_art._cut = real_cut
        unit_sprites.clear_caches()


# --------------------------------------------------------------- measurement


def _mean_row(mask: np.ndarray) -> float | None:
    ys = np.nonzero(mask)[0]
    return None if not len(ys) else float(ys.mean())


def _sail_report(images: dict[str, np.ndarray], bare: dict[str, np.ndarray]) -> list[str]:
    """Per boat frame, off this run's pixels: the hull (bare masts included) is
    what the old table draws over open water, the sails are what this run draws
    that it does not. The lift is how far the sails' mean row sits above the
    hulls' mean row; both rows of boats hold two each, so it compares like with
    like. `below` counts sail px under the lowest hull row."""
    lines = []
    for fid in _BOATS:
        hull = rp.changed_pixels(bare[fid], bare["i"])
        sails = rp.changed_pixels(images[fid], bare[fid])
        hull_row, sail_row = _mean_row(hull), _mean_row(sails)
        if hull_row is None or sail_row is None:
            lines.append(f"{fid}: hull {int(hull.sum())}px, sails {int(sails.sum())}px")
            continue
        below = int(sails[int(np.nonzero(hull.any(axis=1))[0][-1]) + 1 :].sum())
        lines.append(
            f"{fid}: hull {int(hull.sum())}px, sails {int(sails.sum())}px, sails' mean row "
            f"{hull_row - sail_row:+.1f}px above the hulls', {below}px below the lowest hull row"
        )
    return lines


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    boats = "Four boats of one kind on open water, each facing a different way. They belong to the {} player."
    frames = (
        *(
            rp.Frame(fid, f"{fid}.png", "raw", caption=boats.format(colour))
            for fid, (_const, _player, colour) in _BOATS.items()
        ),
        rp.Frame("g", "g.png", "raw", caption="A patch of sandy ground, angled view."),
        rp.Frame("h", "h.png", "raw", caption="Three panels: three consecutive moments of one small animation."),
        rp.Frame("i", "i.png", "raw", caption="Open water, same framing and zoom as the boat frames."),
    )
    seat = (
        "A boat's mast is the upright pole rising from its deck, and a sail is the sheet of cloth it "
        "carries. First, for each boat, describe where its cloth hangs: against which part of the mast, "
        "and how far its lower edge is from the deck and from the water. Then the closed question. "
        "SEATED if on every boat the cloth hangs from the upper part of the masts and stays above the "
        "deck. DROPPED if on any boat cloth hangs low: lying across the side of the hull or reaching the "
        "water at or below the hull's bottom edge, with bare mast standing above it. NO-CLOTH if no boat "
        "carries any cloth at all."
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                "First describe each boat in each frame: its hull, and everything that stands above the "
                "deck. Then: does any boat have no cloth sail at all, only a hull with at most bare poles "
                "or ropes above it? EVERY-BOAT-RIGGED if every boat in every frame carries at least one "
                "cloth sail, SOME-BARE if at least one boat carries none."
            ),
            frames=SAIL_FRAMES,
            answers=("EVERY-BOAT-RIGGED", "SOME-BARE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=seat,
            frames=SAIL_FRAMES,
            answers=("SEATED", "DROPPED", "NO-CLOTH", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q3",
            question=(
                "Each frame's caption names the colour of the player who owns its boats. First, for each "
                "frame, name any strongly coloured marking painted on the boats (an emblem on a sail, a "
                "flag, coloured trim), ignoring the water, bare wood and plain cream cloth. Then: MATCH if "
                "in every frame the boats carry markings in that frame's stated colour, ABSENT if in at "
                "least one frame the boats show no coloured marking at all, OTHER if some frame's markings "
                "are clearly a different colour from the one its caption states."
            ),
            frames=(*SAIL_FRAMES, CONTROL_FRAME),
            answers=("MATCH", "ABSENT", "OTHER", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q4",
            question=seat,
            frames=(CONTROL_FRAME,),
            answers=("SEATED", "DROPPED", "NO-CLOTH", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q5",
            question=(
                "The ground is bare sand. First describe every mark or object drawn on it, with its rough "
                "size and colour. Then: SCATTERED if there are only a few small separate marks (tiny "
                "winged shapes, small dark spots), BLOCK if there is any solid flat diamond shape of one "
                "colour about the size of a ground tile, EMPTY if there is nothing but sand."
            ),
            frames=("g",),
            answers=("SCATTERED", "BLOCK", "EMPTY", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q6",
            question=(
                "The three panels are consecutive moments of one animation of a few small flying insects "
                "over flat ground. Each insect is a small bright shape, and dark blots on the ground are "
                "shadows. First describe, panel by panel, roughly where the bright shapes and the dark "
                "blots are. Then: STEADY if the middle panel has them in nearly the same places as the "
                "left and right panels, JUMPED if the middle panel's layout is clearly different from "
                "both neighbours (bright shapes where the others have empty ground or dark blots)."
            ),
            frames=("h",),
            answers=("STEADY", "JUMPED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q7",
            question=(
                "First describe everything drawn in this frame. Then: is there any object in it (a boat, "
                "a mark, anything that is not water), or only water? OBJECTS-PRESENT or WATER-ONLY."
            ),
            frames=("i",),
            answers=("OBJECTS-PRESENT", "WATER-ONLY", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map object review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from an isometric map editor for a strategy game. Each frame is a "
            "crop, enlarged by a whole-number factor with no smoothing. Judge only what you can actually see."
        ),
        injects=tuple(INJECTS),
    )


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
    """Spec coupling, frame band, determinism, each inject's reach per frame
    against INJECT_REACH, and the sail measurements each run's answers rest on."""
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
            failures.append(f"frame {frame.id}: two clean renders differ")
        try:
            scaled, _f = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))
    failures += [
        f"frame {fid}: nothing is drawn on the water"
        for fid in _BOATS if not rp.changed_pixels(clean[fid], clean["i"]).any()
    ]
    if _loop_ink_outside_box():
        failures.append(f"frame h: {_loop_ink_outside_box()}px of the loop frames fall outside its fixed crop")
    bare = _frame_images(COARSE_INJECT)
    for line in _sail_report(clean, bare):
        print(f"measure: clean {line}")
    reach = {}
    for name in INJECTS:
        injected = bare if name == COARSE_INJECT else _frame_images(name)
        touched = {}
        for frame in spec.frames:
            mask = rp.changed_pixels(clean[frame.id], injected[frame.id])
            n = int(np.count_nonzero(mask))
            if n:
                touched[frame.id] = (n, rp.largest_component_px(mask))
        if set(touched) != INJECT_REACH[name]:
            failures.append(
                f"inject {name}: changed frames {sorted(touched)}, expected {sorted(INJECT_REACH[name])}"
            )
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
        if name == "sails-low":
            for line in _sail_report(injected, bare):
                print(f"measure: {name} {line}")
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
        raise SystemExit("needs a configured AoE2:DE install: the pack judges sprite and particle art")
    if args.check:
        check()
        return
    for path in generate(args.out_dir, args.inject, args.replica):
        print(f"wrote {path}")
    print(f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory "
          f"{grp._run_token(PACK_ID, args.inject, args.replica)}")


if __name__ == "__main__":
    main()
