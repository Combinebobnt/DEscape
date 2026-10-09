#!/usr/bin/env python3
"""Review pack for per-civ and per-age building art (GH #48): does each owner's
building draw the architecture and starting age the scenario stores, does
Gaia keep the Gaia-table art, and is a town centre drawn as one age.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_pasture_review_pack.py. Needs a real AoE2:DE
install (sprites) and the examples/ corpus; renders off-engine via
render.composite_rect_iso, Stepped, on each file's own terrain and elevation,
with only the units under review kept so nothing else crowds the crop.

Each style question pairs a scene with two reference frames: the same building
types on blank ground, drawn with an explicit (art_civ, age), so the reviewer
matches styles rather than naming them. References never read the scenario,
so no inject can move them (`--check` asserts it).

Run:
    tools/gen_civ_art_review_pack.py
    tools/gen_civ_art_review_pack.py --replica 1
    tools/gen_civ_art_review_pack.py --inject arch-from-civ
    tools/gen_civ_art_review_pack.py --check      # writes nothing

Writes build/review_pack/civ_art/<opaque token>/ (or --out-dir), gitignored.
"""

from __future__ import annotations

import argparse
import copy
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

from descape import asset_source, civ_art, render, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit import review_pack as rp

PACK_ID = "civ_art"
OUT_DIR = grp.OUT_DIR
TILE_PX = grp.TILE_PX
EXAMPLES = ROOT / "examples"
JOAN_2 = "2_Joan_coop_2_v0_15.aoe2scenario"
OLD_ALLIES = "old-allies-final-v2.aoe2scenario"

HOUSE, BARRACKS, MILL, TOWN_CENTRE = 70, 12, 68, 109
# DESERT_SAND, under every reference group.
REFERENCE_TERRAIN = 14

# Joan co-op 2's P2 (civ Teutons, architecture Franks, Feudal) owns no
# building, so the scene places three probe buildings for P2 on P4's land.
_PROBES = ((HOUSE, 80.0, 40.0), (BARRACKS, 84.5, 40.5), (MILL, 81.0, 44.0))

# scene id -> (file, player_id, unit picker). A picker takes the player's units
# and returns the ones to keep; None means "place _PROBES for this player".
_SCENES = {
    # 1: Joan P2, architecture overriding civ.
    "s1": (JOAN_2, 2, None),
    # 2: old-allies P7, HUN-CIV civ with MONGOL-CIV architecture: houses and a mill.
    "s2": (OLD_ALLIES, 7, lambda u: u.unit_const in (HOUSE, MILL) and 160 <= u.x <= 171 and 74 <= u.y <= 80),
    # 3: Joan P4 (Franks, Castle Age): a row of houses.
    "s3": (JOAN_2, 4, lambda u: u.unit_const == HOUSE and 98 <= u.x <= 108 and 46 <= u.y <= 50),
    # 4: old-allies Gaia (stored Bohemians, Feudal): a cluster of houses.
    "s4": (OLD_ALLIES, 0, lambda u: u.unit_const == HOUSE and 103 <= u.x <= 117 and 113 <= u.y <= 116),
    # 5: Joan P4's town centre (Franks, Castle Age).
    "s5": (JOAN_2, 4, lambda u: u.unit_const == TOWN_CENTRE and abs(u.x - 89.0) < 1 and abs(u.y - 24.0) < 1),
    # 6: old-allies Gaia's Dark Age town centre, the specificity frame.
    "s6": (OLD_ALLIES, 0, lambda u: u.unit_const == TOWN_CENTRE and abs(u.x - 108.0) < 1 and abs(u.y - 120.0) < 1),
}

# reference id -> (consts in a row, (art_civ, age)). Each pair's two sets differ.
_REFERENCES = {
    "r1a": ((HOUSE, BARRACKS, MILL), (4, 3)),       # Teutons Feudal: east
    "r1b": ((HOUSE, BARRACKS, MILL), (2, 3)),       # Franks Feudal: west
    "r2a": ((HOUSE, MILL), (12, 3)),                # Mongols Feudal: asia
    "r2b": ((HOUSE, MILL), (17, 3)),                # Huns Feudal: east
    "r3a": ((HOUSE, HOUSE), (2, 4)),                # Franks Castle: west age3
    "r3b": ((HOUSE, HOUSE), (2, 3)),                # Franks Feudal: west age2
    "r4a": ((HOUSE, HOUSE), (None, 2)),             # the Gaia table
    "r4b": ((HOUSE, HOUSE), (39, 3)),               # Bohemians Feudal: slav
}


# ----------------------------------------------------------------- injects


def _arch_from_civ(real):
    def resolve(civilization, architecture, starting_age):
        return real(civilization, None, starting_age)
    return resolve


def _tc_part(piece: dict) -> str:
    """"main", "back", "center" or "front", from b_<set>_town_center_age<n>_<part>_x1."""
    return piece["file_name"].rsplit("_", 2)[-2].lower()


# The TC's body. Not the .dat parent: that is the small back piece, and "main" is an annex.
TC_MAIN_PART = "main"


def _dark_town_centre_annexes(real):
    def resolve_entry(unit_const, art_civ, age):
        entry = real(unit_const, art_civ, age)
        if unit_const != TOWN_CENTRE or entry is None or "pieces" not in entry:
            return entry
        out = copy.deepcopy(entry)
        for piece in out["pieces"]:
            part = _tc_part(piece)
            if part != TC_MAIN_PART:
                piece["file_name"] = f"b_dark_town_center_age1_{part}_x1"
        return out
    return resolve_entry


def _gaia_as_player(real):
    def player_art(loaded, pending=None):
        art = list(real(loaded, pending))
        from descape import player_fields

        specs = {s.field_id: s for s in player_fields.specs_for(loaded)}
        values = [
            player_fields.current_value(loaded, specs[f], 0) if f in specs else None
            for f in ("civilization", "architecture", "starting_age")
        ]
        art[0] = civ_art.resolve(*values)
        return tuple(art)
    return player_art


# name -> (module, attribute, wrapper)
INJECTS = {
    "arch-from-civ": (civ_art, "resolve", _arch_from_civ),
    "age-forced-dark": (civ_art, "art_age", lambda _real: lambda _value: civ_art.DARK_AGE),
    "tc-dark-annexes": (unit_sprites, "resolve_entry", _dark_town_centre_annexes),
    "gaia-as-player": (civ_art, "player_art", _gaia_as_player),
}
COARSE_INJECT = "arch-from-civ"


@contextmanager
def _patched(inject: str | None):
    unit_sprites._scaled_cache.clear()
    saved = None
    if inject is not None:
        module, name, wrap = INJECTS[inject]
        saved = (module, name, getattr(module, name))
        setattr(module, name, wrap(saved[2]))
    try:
        yield
    finally:
        if saved is not None:
            setattr(saved[0], saved[1], saved[2])
        unit_sprites._scaled_cache.clear()


# --------------------------------------------------------------- rendering


@cache
def _loaded(name: str):
    return load_map_and_units(str(EXAMPLES / name))


def _unit(const: int, x: float, y: float, ref: int):
    return SimpleNamespace(
        unit_const=const, x=x, y=y, z=0.0, rotation=0.0, reference_id=ref, status=2, garrisoned_in_id=-1,
    )


def _render_units(base, units_by_player: list[list], player_art) -> np.ndarray:
    """Stepped composite of `units_by_player` on `base`'s own terrain, cropped
    to the units' footprints plus a tile, with headroom above for tall art."""
    scn = SimpleNamespace(
        map_manager=base.map_manager,
        unit_manager=SimpleNamespace(units=units_by_player),
        team_indices=base.team_indices,
        player_colors=base.player_colors,
        player_art=player_art,
        _base=base,
    )
    mm = scn.map_manager
    elevations = np.array(
        [[int(mm.get_tile(x, y).elevation) for x in range(mm.map_width)] for y in range(mm.map_height)],
        dtype=np.int64,
    )
    max_elev = int(elevations.max())
    proj = grp.ig.canvas_size_and_origin(
        mm.map_width, mm.map_height, TILE_PX, grp.ig.MIN_ELEVATION, grp.ig.MAX_ELEVATION, elev_step_pct=50
    )
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)
    units_by_tile = render._units_by_tile(scn)
    bboxes = render.merge_sprite_bboxes(
        render._building_bboxes_iso(scn, mm.map_width, mm.map_height, proj, elevations), sprites
    )
    tiles = [t for units in units_by_player for u in units for t in render.unit_occupied_tiles(u, mm.map_width, mm.map_height)]
    xs, ys = [t[0] for t in tiles], [t[1] for t in tiles]
    corners = [(min(xs) - 1, min(ys) - 1), (max(xs) + 1, max(ys) + 1), (min(xs) - 1, max(ys) + 1), (max(xs) + 1, min(ys) - 1)]
    x0, y0, x1, y1 = grp._tiles_bbox_px(proj, corners, max_elev)
    y0 = max(0, y0 - 5 * proj.half_h)
    return render.composite_rect_iso(
        scn, x0, y0, x1, y1, elevations, proj, TILE_PX, units_by_tile, bboxes, with_units=True, sprites=sprites,
    )


def _scene(scene_id: str) -> np.ndarray:
    name, player_id, picker = _SCENES[scene_id]
    loaded = _loaded(name)
    units = [[] for _ in loaded.unit_manager.units]
    if picker is None:
        units[player_id] = [_unit(c, x, y, 7000 + i) for i, (c, x, y) in enumerate(_PROBES)]
    else:
        units[player_id] = [u for u in loaded.unit_manager.units[player_id] if picker(u)]
    if not units[player_id]:
        raise SystemExit(f"scene {scene_id}: no unit picked from {name} P{player_id}")
    # Read through civ_art at render time, so a patched derivation reaches the scene.
    return _render_units(loaded, units, civ_art.player_art(loaded))


@cache
def _blank():
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = REFERENCE_TERRAIN
    return base


def _reference(ref_id: str) -> np.ndarray:
    consts, art = _REFERENCES[ref_id]
    base = _blank()
    units = [[] for _ in base.unit_manager.units]
    x = 60.0
    for i, const in enumerate(consts):
        span = render.tile_span(const, render.NON_BUILDING_SPAN)[0]
        units[1].append(_unit(const, x + span / 2, 60.0 + span / 2, 8000 + i))
        x += span + 1
    player_art = (civ_art.GAIA_ART, art, *(civ_art.GAIA_ART,) * (len(units) - 2))
    return _render_units(base, units, player_art)


# frame id -> render callable
_FRAMES = {
    "scene_1": lambda: _scene("s1"), "ref_1a": lambda: _reference("r1a"), "ref_1b": lambda: _reference("r1b"),
    "scene_2": lambda: _scene("s2"), "ref_2a": lambda: _reference("r2a"), "ref_2b": lambda: _reference("r2b"),
    "scene_3": lambda: _scene("s3"), "ref_3a": lambda: _reference("r3a"), "ref_3b": lambda: _reference("r3b"),
    "scene_4": lambda: _scene("s4"), "ref_4a": lambda: _reference("r4a"), "ref_4b": lambda: _reference("r4b"),
    "whole_1": lambda: _scene("s5"),
    "whole_2": lambda: _scene("s6"),
}
REFERENCE_FRAMES = tuple(f for f in _FRAMES if f.startswith("ref_"))


# --------------------------------------------------------------- the spec


def _style_check(n: int) -> rp.Check:
    return rp.Check(
        id=f"set_{n}",
        question=(
            f"Frame scene_{n} shows a group of buildings on a map. Frames ref_{n}a and ref_{n}b each show "
            "buildings of the same kinds on plain sand, drawn in two different looks. First describe the "
            "scene's buildings: roof shape and colour, wall material, decoration. Then the closed question: "
            f"are the scene's buildings drawn in the same look as ref_{n}a's or as ref_{n}b's? Judge the "
            "building art only, not the ground, the size of the crop or the team colour. REF-A, REF-B, or "
            "NEITHER if the scene matches neither reference."
        ),
        frames=(f"scene_{n}", f"ref_{n}a", f"ref_{n}b"),
        answers=("REF-A", "REF-B", "NEITHER", "CANNOT-TELL"),
    )


def _whole_check(n: int) -> rp.Check:
    return rp.Check(
        id=f"whole_{n}",
        question=(
            "This frame shows one large building made of several overlapping parts. First describe its "
            "parts: what each is made of (wood, thatch, stone, brick, tile) and how they join. Then: is it "
            "drawn as one consistent construction, or are some of its parts clearly in a different "
            "material or building style from the rest, as if two different buildings were overlaid? "
            "CONSISTENT or MIXED."
        ),
        frames=(f"whole_{n}",),
        answers=("CONSISTENT", "MIXED", "CANNOT-TELL"),
    )


def _spec() -> rp.PackSpec:
    frames = []
    for n in (1, 2, 3, 4):
        frames += [
            rp.Frame(f"scene_{n}", f"scene_{n}.png", "raw", caption=f"Scene {n}: buildings on a map."),
            rp.Frame(f"ref_{n}a", f"ref_{n}a.png", "raw", caption=f"Reference A for scene {n}."),
            rp.Frame(f"ref_{n}b", f"ref_{n}b.png", "raw", caption=f"Reference B for scene {n}."),
        ]
    frames += [
        rp.Frame("whole_1", "whole_1.png", "raw", caption="One large building."),
        rp.Frame("whole_2", "whole_2.png", "raw", caption="Another large building."),
    ]
    checks = (*(_style_check(n) for n in (1, 2, 3, 4)), _whole_check(1), _whole_check(2))
    return rp.PackSpec(
        id=PACK_ID,
        title="Building art review",
        frames=tuple(frames),
        checks=checks,
        preamble=(
            "These are offscreen renders from an isometric map editor for a strategy game. Each frame is a "
            "crop, enlarged by a whole-number factor with no smoothing. Buildings carry their owner's team "
            "colour, which is not what any question is about. Judge only what you can actually see."
        ),
        injects=tuple(INJECTS),
    )


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    with _patched(inject):
        return {frame_id: render_frame() for frame_id, render_frame in _FRAMES.items()}


def _whole_1_tc_parts(inject: str | None, keep: frozenset[str]) -> np.ndarray:
    """whole_1 under `inject`, its town centre cut down to the `keep` parts."""
    with _patched(inject):
        real = unit_sprites.resolve_entry

        def resolve_entry(unit_const, art_civ, age):
            entry = real(unit_const, art_civ, age)
            if unit_const != TOWN_CENTRE or entry is None or "pieces" not in entry:
                return entry
            return {**entry, "pieces": [p for p in entry["pieces"] if _tc_part(p) in keep]}

        unit_sprites.resolve_entry = resolve_entry
        try:
            return _scene("s5")
        finally:
            unit_sprites.resolve_entry = real


def _tc_annex_failures() -> list[str]:
    """tc-dark-annexes must change every whole_1 TC piece but the main one, which stays pixel-identical,
    or whole_1 never shows a mixed construction (2026-10-07 review: the whole TC went Dark Age)."""
    name, player_id, _picker = _SCENES["s5"]
    art = civ_art.player_art(_loaded(name))[player_id]
    clean = unit_sprites.resolve_entry(TOWN_CENTRE, *art)
    with _patched("tc-dark-annexes"):
        injected = unit_sprites.resolve_entry(TOWN_CENTRE, *art)
    parts = frozenset(_tc_part(p) for p in clean["pieces"])
    failures = [] if TC_MAIN_PART in parts else [f"whole_1's town centre has no {TC_MAIN_PART} piece: {sorted(parts)}"]
    for before, after in zip(clean["pieces"], injected["pieces"], strict=True):
        part = _tc_part(before)
        if (before == after) != (part == TC_MAIN_PART):
            failures.append(f"inject tc-dark-annexes: whole_1's {part} piece {'unchanged' if before == after else 'changed'}")
    main, annexes = frozenset({TC_MAIN_PART}), parts - {TC_MAIN_PART}
    main_px = rp.changed_pixels(_whole_1_tc_parts(None, main), _whole_1_tc_parts("tc-dark-annexes", main))
    annex_px = rp.changed_pixels(_whole_1_tc_parts(None, annexes), _whole_1_tc_parts("tc-dark-annexes", annexes))
    if main_px.any():
        failures.append(f"inject tc-dark-annexes: moved {int(main_px.sum())}px of whole_1's main piece")
    if not annex_px.any():
        failures.append("inject tc-dark-annexes: left whole_1's annexes unchanged")
    print(f"tc-dark-annexes: main piece {int(main_px.sum())}px changed, annexes {sorted(annexes)} {int(annex_px.sum())}px")
    return failures


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
    """Spec coupling, frame band, each inject's reach per frame, that no
    inject moves a reference frame, and that tc-dark-annexes spares the TC's
    main piece."""
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
    failures += [
        f"references {n}a and {n}b are identical: the pair discriminates nothing"
        for n in (1, 2, 3, 4)
        if np.array_equal(clean[f"ref_{n}a"], clean[f"ref_{n}b"])
    ]
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
        moved = sorted(set(touched) & set(REFERENCE_FRAMES))
        if moved:
            failures.append(f"inject {name}: moved reference frame(s) {moved}")
        if "whole_2" in touched and name != "gaia-as-player":
            failures.append(f"inject {name}: reached the specificity frame whole_2")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    coarse = reach[COARSE_INJECT]
    if not any(
        comp < coarse.get(fid, (0, 0))[1]
        for name, t in reach.items() if name != COARSE_INJECT
        for fid, (_px, comp) in t.items()
    ) and not any(
        sum(px for px, _c in t.values()) < sum(px for px, _c in coarse.values())
        for name, t in reach.items() if name != COARSE_INJECT
    ):
        failures.append("no inject is smaller than the coarse one")
    failures += _tc_annex_failures()
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
        raise SystemExit("needs a configured AoE2:DE install: the pack judges sprite art")
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
