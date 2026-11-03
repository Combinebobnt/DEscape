#!/usr/bin/env python3
"""Review pack for the buildings a player's civilization and starting age
upgrade (TASK-205): does each building draw the const its owner's techs
upgrade it to, and does a Satrapy town centre draw as one age.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), and the
renderer of tools/gen_civ_art_review_pack.py, which keeps its own pack so its
recorded verdicts stand. The scenes come from a fixture written through the
real write path: examples/blank_map.aoe2scenario, eight players set through
OptionsEditModel, buildings placed through UnitEditModel (walls and the gate
through descape/wall_run.py's plans), then write_scenario() to
build/task205/fixture.aoe2scenario (`--fixture-only`, the AoE2:DE editor
check's input), read back before any frame renders. A pack run renders from
its own scratch copy, so it never rewrites the file a tester has open.

Each style question pairs a scene with two references on blank ground, the
scene's units redrawn with explicit (art_civ, age, tech_civ) and consts no
inject's table change reaches (`_REFERENCES` says how each stays put;
`--check` asserts none moves).

Every inject swaps whole buildings, so a localized inject is as large as the
coarse one on the frame it shares with it. Size is not what this pack
resolves: `--check` asserts that each localized inject reaches one frame.

Run:
    tools/gen_start_upgrade_review_pack.py --fixture-only
    tools/gen_start_upgrade_review_pack.py
    tools/gen_start_upgrade_review_pack.py --replica 1
    tools/gen_start_upgrade_review_pack.py --inject no-refire
    tools/gen_start_upgrade_review_pack.py --check      # writes only the fixture

Writes build/review_pack/buildings_b/<opaque token>/ (or --out-dir), gitignored.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from contextlib import contextmanager
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_civ_art_review_pack as civ_pack
import gen_review_pack as grp
import gen_seam_eyeball as seam_eyeball
import numpy as np

from descape import asset_source, batch_api, civ_art, player_fields, render, unit_sprites, wall_run
from descape.edit_history import EditHistory
from descape.options_model import OptionsEditModel
from descape.scenario_io import is_workshop_subscribed, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.unit_model import UnitEditModel
from testkit import review_pack as rp

PACK_ID = "buildings_b"  # reviewers read it in every frame path, so it names nothing under review
OUT_DIR = grp.OUT_DIR
SOURCE = ROOT / "examples" / "blank_map.aoe2scenario"
FIXTURE = ROOT / "build" / "task205" / "fixture.aoe2scenario"

MONASTERY, FORTIFIED_CHURCH, MILL = 104, 1806, 68
STONE_WALL, FORTIFIED_WALL, STONE_GATE, FORTIFIED_GATE = 117, 155, 64, 63
WATCH_TOWER, GUARD_TOWER, KEEP = 79, 234, 235
SATRAPY_TC, TOWN_CENTRE = 2275, 109
SETTLEMENT, SETTLEMENT_2 = 2556, 2558
SIEGE_WORKSHOP, SIEGE_WORKSHOP_2 = 49, 150
ARMENIANS, POLES, BRITONS, GOTHS, ACHAEMENIDS, FRANKS, TATARS, CUMANS = 44, 38, 1, 3, 46, 2, 33, 34
FEUDAL, CASTLE, IMPERIAL, POST_IMPERIAL = 3, 4, 5, 6

# player -> (civilization, architecture, starting age, [(const, tile x, tile y)]): the plan's Verification
# fixture. P3 and P5's walls and gate are placed by _place_walls(), not listed here. P6's 109 is the
# specificity town centre, which no inject reaches.
PLAYERS = {
    1: ("ARMENIANS", "ARMENIANS", FEUDAL, [(MONASTERY, 10, 10)]),
    2: ("POLES", "POLES", CASTLE, [(MILL, 10, 22)]),
    3: ("BRITONS", "BRITONS", IMPERIAL, [(WATCH_TOWER, 24, 34)]),
    4: ("BRITONS", "BRITONS", POST_IMPERIAL, [(WATCH_TOWER, 10, 46)]),
    5: ("GOTHS", "GOTHS", IMPERIAL, []),
    6: ("ACHAEMENIDS", "ACHAEMENIDS", CASTLE, [(SATRAPY_TC, 10, 70), (TOWN_CENTRE, 18, 70)]),
    7: ("FRANKS", "ARMENIANS", FEUDAL, [(MONASTERY, 10, 82), (SETTLEMENT, 16, 82)]),
    8: ("TATARS", "TATARS", 2, [(SIEGE_WORKSHOP, 10, 94)]),
}
# player -> (wall tiles along +x at y, gate low tile or None)
WALLS = {3: ((10, 13, 18, 21), 34, (14, 34)), 5: ((10, 17, None, None), 58, None)}


# ----------------------------------------------------------------- fixture


def _civ_token(name: str) -> str:
    from AoE2ScenarioParser.datasets.object_support import Civilization

    return Civilization[name].value


def _wall_tiles(spec) -> list[tuple[int, int]]:
    (a0, a1, b0, b1), y, _gate = spec
    tiles = [(x, y) for x in range(a0, a1 + 1)]
    if b0 is not None:
        tiles += [(x, y) for x in range(b0, b1 + 1)]
    return tiles


def _place_walls(model, loaded, player: int) -> None:
    mm = loaded.map_manager
    spec = WALLS[player]
    gate = spec[2]
    if gate is not None:
        span = render.tile_span(STONE_GATE, render.NON_BUILDING_SPAN)
        gate_tiles = [(gate[0] + i, gate[1] + j) for i in range(span[0]) for j in range(span[1])]
        tiles, walls = wall_run.wall_scene(loaded, mm.map_width, mm.map_height)
        plan = wall_run.plan_gate_over_walls(gate_tiles, existing_tiles=tiles, existing_walls=walls)
        wall_run.apply_gate_plan(model, player, STONE_GATE, *render.span_anchor(*gate, *span), plan)
    tiles, walls = wall_run.wall_scene(loaded, mm.map_width, mm.map_height)
    plan = wall_run.plan_wall_run(_wall_tiles(spec), unit_const=STONE_WALL, existing_tiles=tiles, existing_walls=walls)
    wall_run.apply_wall_plan(model, player, plan)


def write_fixture(path: Path = FIXTURE) -> Path:
    """The fixture, through the real write path, read back and checked."""
    if is_workshop_subscribed(path):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {path}")
    loaded = batch_api.load(SOURCE)  # activates the file's game dataset (AGENTS.md)
    options = OptionsEditModel(loaded)
    options.set_value(player_fields.PLAYER_COUNT_FIELD_ID, len(PLAYERS))
    for player, (civ, arch, age, _units) in PLAYERS.items():
        options.set_value(player_fields.player_field_id("civilization", player), _civ_token(civ))
        options.set_value(player_fields.player_field_id("architecture", player), _civ_token(arch))
        options.set_value(player_fields.player_field_id("starting_age", player), age)
    units = UnitEditModel(loaded)
    units.begin_unit_edit(sorted(PLAYERS))
    for player, (_civ, _arch, _age, placed) in PLAYERS.items():
        for const, tx, ty in placed:
            units.add(player, const, *render.span_anchor(tx, ty, *render.tile_span(const, render.NON_BUILDING_SPAN)))
        if player in WALLS:
            _place_walls(units, loaded, player)
    units.commit_unit_edit("Place the TASK-205 fixture", EditHistory())
    path.parent.mkdir(parents=True, exist_ok=True)
    write_scenario(loaded, path, options=options, units=units, backup=False)
    _check_readback(path)
    return path


def _check_readback(path: Path) -> None:
    back = load_map_and_units(path)
    # Stored values, not civ_art's derivation: an inject may patch that.
    specs = {s.field_id: s for s in player_fields.specs_for(back)}
    failures = []
    for player, (civ, arch, age, placed) in PLAYERS.items():
        want = (_civ_token(civ), _civ_token(arch), age)
        got = tuple(player_fields.current_value(back, specs[f], player) for f in ("civilization", "architecture", "starting_age"))
        if got != want:
            failures.append(f"P{player}: stored {got}, wanted {want}")
        consts = sorted(u.unit_const for u in back.unit_manager.units[player])
        wanted = sorted([c for c, _x, _y in placed] + ([STONE_WALL] * len(_wall_tiles(WALLS[player])) if player in WALLS else [])
                        + ([STONE_GATE] if player in WALLS and WALLS[player][2] else []))
        if consts != wanted:
            failures.append(f"P{player}: units {consts}, wanted {wanted}")
    if failures:
        raise SystemExit(f"fixture read-back of {path}:\n  " + "\n  ".join(failures))


@cache
def _fixture():
    scratch = Path(tempfile.mkdtemp(prefix="t205_fixture_")) / FIXTURE.name
    return load_map_and_units(str(write_fixture(scratch)))


# ----------------------------------------------------------------- injects

# The rows the non-age techs add to age_upgrades: Upgrade Rams, Settlement, and an Imperial start's towers,
# walls and gates (measured 2026-11-03 against the pre-TASK-205 table). Also tools/release_render_diff.py's
# start-upgrades patch, which imports age_techs_only().
START_UPGRADE_ROWS = frozenset({
    49, 64, 78, 79, 81, 88, 91, 95, 117, 487, 490, 659, 661, 663, 665, 667, 669, 671, 673, 2556, 2558,
})


def age_techs_only(art: unit_sprites.BuildingArt) -> unit_sprites.BuildingArt:
    """The pre-TASK-205 table: age techs 101-103 at ages 3..5, no civ rows, no per-age annexes, no Post-Imperial."""
    rows = {
        const: {age: target for age, target in by_age.items() if 3 <= age <= 5}
        for const, by_age in art.age_upgrades.items() if const not in START_UPGRADE_ROWS
    }
    return art._replace(age_upgrades={c: r for c, r in rows.items() if r}, civ_upgrades={}, age_art={}, max_age=5)


def _no_refire(art: unit_sprites.BuildingArt) -> unit_sprites.BuildingArt:
    """A civ's Fortified Church row keeps only its first firing: later ages take the generic monastery row."""
    generic = art.age_upgrades.get(MONASTERY, {})
    civ_rows = {}
    for civ, civ_row in art.civ_upgrades.items():
        rows = dict(civ_row)
        if MONASTERY in rows:
            rows[MONASTERY] = {a: (t if a < FEUDAL else generic.get(a, MONASTERY)) for a, t in rows[MONASTERY].items()}
        civ_rows[civ] = rows
    return art._replace(civ_upgrades=civ_rows)


def _table(transform):
    def wrap(real):
        table = transform(real())
        return lambda: table
    return wrap


def _tech_from_architecture(real):
    def resolve(civilization, architecture, starting_age, ror=False):
        art_civ, age, _tech = real(civilization, architecture, starting_age, ror=ror)
        return art_civ, age, art_civ
    return resolve


# name -> (module, attribute, wrapper)
INJECTS = {
    "age-techs-only": (unit_sprites, "building_art", _table(age_techs_only)),
    "tech-civ-from-architecture": (civ_art, "resolve", _tech_from_architecture),
    "no-refire": (unit_sprites, "building_art", _table(_no_refire)),
    "post-imperial-as-imperial": (civ_art, "art_age", lambda real: lambda value: min(real(value), civ_art.IMPERIAL_AGE)),
    "satrapy-dark-annexes": (unit_sprites, "building_art", _table(lambda art: art._replace(age_art={}))),
}
COARSE_INJECT = "age-techs-only"


@contextmanager
def _patched(inject: str | None):
    unit_sprites._scaled_cache.clear()
    unit_sprites.clear_slice_cache()
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
        unit_sprites.clear_slice_cache()


# --------------------------------------------------------------- rendering

# Half-tiles above the footprints: a Keep and a Fortified Church outgrow the civ_art pack's 5.
HEADROOM = 10

# scene id -> (player, consts kept). The units come from the fixture.
_SCENES = {
    "s1": (1, {MONASTERY}),
    "s2": (7, {MONASTERY}),
    "s3": (4, {WATCH_TOWER}),
    "s4": (3, {STONE_WALL, STONE_GATE, WATCH_TOWER}),
    "s5": (8, {SIEGE_WORKSHOP}),
    "s6": (7, {SETTLEMENT}),
    "w1": (6, {SATRAPY_TC}),
    "w2": (6, {TOWN_CENTRE}),
}

# reference id -> (scene whose units it redraws, {placed const: const drawn}, (art_civ, age, tech_civ)).
# The drawn consts at age 2 with no tech civ have no upgrade row, so no inject's table change reaches them; r1b/r2b
# draw 104 at Feudal through the generic age-tech row (104 -> 30), which every inject keeps.
_REFERENCES = {
    "r1a": ("s1", {MONASTERY: FORTIFIED_CHURCH}, (ARMENIANS, 2, None)),
    "r1b": ("s1", {MONASTERY: MONASTERY}, (ARMENIANS, FEUDAL, None)),
    "r2a": ("s2", {MONASTERY: FORTIFIED_CHURCH}, (ARMENIANS, 2, None)),
    "r2b": ("s2", {MONASTERY: MONASTERY}, (ARMENIANS, FEUDAL, None)),
    "r3a": ("s3", {WATCH_TOWER: KEEP}, (BRITONS, 2, None)),
    "r3b": ("s3", {WATCH_TOWER: GUARD_TOWER}, (BRITONS, 2, None)),
    "r4a": ("s4", {STONE_WALL: FORTIFIED_WALL, STONE_GATE: FORTIFIED_GATE, WATCH_TOWER: GUARD_TOWER}, (BRITONS, 2, None)),
    "r4b": ("s4", {}, (BRITONS, 2, None)),
    # The generic row draws 49 as 150 from age 2; Cumans' row starts at age 4, so r5a draws 49 itself.
    "r5a": ("s5", {}, (TATARS, 2, CUMANS)),
    "r5b": ("s5", {SIEGE_WORKSHOP: SIEGE_WORKSHOP_2}, (TATARS, 2, None)),
    "r6a": ("s6", {SETTLEMENT: SETTLEMENT_2}, (ARMENIANS, 2, None)),
    "r6b": ("s6", {}, (ARMENIANS, 2, None)),
}


def _scene_units(scene_id: str) -> tuple[int, list]:
    player, consts = _SCENES[scene_id]
    loaded = _fixture()
    kept = [u for u in loaded.unit_manager.units[player] if u.unit_const in consts]
    if {u.unit_const for u in kept} != consts:
        raise SystemExit(f"scene {scene_id}: P{player} has {sorted(u.unit_const for u in kept)}, wanted {sorted(consts)}")
    return player, kept


def _scene(scene_id: str) -> np.ndarray:
    player, kept = _scene_units(scene_id)
    loaded = _fixture()
    units = [[] for _ in loaded.unit_manager.units]
    units[player] = kept
    # Read through civ_art at render time, so a patched derivation reaches the scene.
    return civ_pack._render_units(loaded, units, civ_art.player_art(loaded), headroom=HEADROOM)


def _reference(ref_id: str) -> np.ndarray:
    scene_id, drawn, art = _REFERENCES[ref_id]
    _player, kept = _scene_units(scene_id)
    base = civ_pack._blank()
    units = [[] for _ in base.unit_manager.units]
    units[1] = [
        civ_pack._unit(drawn.get(u.unit_const, u.unit_const), u.x, u.y, 9000 + i) for i, u in enumerate(kept)
    ]
    for ref, u in zip(units[1], kept, strict=True):
        ref.rotation = u.rotation
    player_art = (civ_art.GAIA_ART, art, *(civ_art.GAIA_ART,) * (len(units) - 2))
    return civ_pack._render_units(base, units, player_art, headroom=HEADROOM)


STYLE_SETS = (1, 2, 3, 4, 5, 6)
_FRAMES = {
    **{f"scene_{n}": (lambda n=n: _scene(f"s{n}")) for n in STYLE_SETS},
    **{f"ref_{n}{s}": (lambda n=n, s=s: _reference(f"r{n}{s}")) for n in STYLE_SETS for s in "ab"},
    "whole_1": lambda: _scene("w1"),
    "whole_2": lambda: _scene("w2"),
}
REFERENCE_FRAMES = tuple(f for f in _FRAMES if f.startswith("ref_"))
SPECIFICITY_FRAME = "whole_2"


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = []
    for n in STYLE_SETS:
        frames += [
            rp.Frame(f"scene_{n}", f"scene_{n}.png", "raw", caption=f"Scene {n}: buildings on a map."),
            rp.Frame(f"ref_{n}a", f"ref_{n}a.png", "raw", caption=f"Reference A for scene {n}."),
            rp.Frame(f"ref_{n}b", f"ref_{n}b.png", "raw", caption=f"Reference B for scene {n}."),
        ]
    frames += [
        rp.Frame("whole_1", "whole_1.png", "raw", caption="One large building."),
        rp.Frame("whole_2", "whole_2.png", "raw", caption="Another large building."),
    ]
    checks = (*(civ_pack._style_check(n) for n in STYLE_SETS), civ_pack._whole_check(1), civ_pack._whole_check(2))
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
    """Spec coupling, frame band, that each reference pair differs, each inject's reach per frame, that no
    inject moves a reference or the specificity frame, and that each localized inject reaches one frame."""
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
        for n in STYLE_SETS
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
        moved = sorted(set(touched) & {*REFERENCE_FRAMES, SPECIFICITY_FRAME})
        if moved:
            failures.append(f"inject {name}: moved reference or specificity frame(s) {moved}")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    # Whole-building swaps (module docstring): a localized inject is localized by frame count, not by size.
    coarse = set(reach[COARSE_INJECT])
    for name, touched in reach.items():
        if name != COARSE_INJECT and len(touched) != 1:
            failures.append(f"inject {name}: reached {sorted(touched)}, not one frame")
    if len(coarse) < 2:
        failures.append(f"coarse inject {COARSE_INJECT} reached {sorted(coarse)}: not coarse")
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
    parser.add_argument("--fixture-only", action="store_true", help="write and read back the fixture, nothing else")
    args = parser.parse_args()
    if args.fixture_only:
        print(f"wrote {write_fixture()}")
        return
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
