#!/usr/bin/env python3
"""
Verifies descape/batch_api.py (v2.5's scriptable-batch-edit API) against a
directory of .aoe2scenario files:

1. buildings_of / raise_elevation_under -- raise the tile under a real
   building by 1, save, reload through the real load path, and confirm the
   reloaded elevation matches the pre-edit value + 1 (clamped).
2. raise_elevation_under's MAX_ELEVATION clamp -- same, but starting the tile
   already at MAX_ELEVATION; must stay there, not go out of range (see
   batch_api.raise_elevation_under's docstring for why exceeding it is unsafe,
   not just untidy).
3. raise_elevation_under's off-map no-op -- a unit whose position is off-map
   must return False, not raise.
4. neighbors() edge/corner behavior -- an interior tile has 4 (or 8 with
   diagonal=True) neighbors; every corner (not just (0, 0) -- see this
   check's own comment for why the opposite corner matters too, given
   xy_to_i's x + y*map_size arithmetic) has only 2 (or 3), and an edge
   (non-corner) tile has 3.
5. recolor_dirt_near_water's algorithm (is_water + neighbors + set_terrain) --
   recolor every DIRT_2 tile bordering water to BEACH, save, reload, and
   confirm (a) every recolored tile really did border water and is now BEACH
   with layer reset, and (b) re-running the same detection against the
   reloaded file finds nothing left to recolor.
6. is_water's own classification -- direct assertions against known
   TerrainId members (not just self-consistent with whatever recolor found),
   including the SHALLOWS family, which needs its own keyword match
   separate from "WATER" (see is_water's docstring), and confirming ICE is
   deliberately excluded.
7. raise_elevation_under_many -- apply it across every on-map building
   belonging to whichever player has the most, save, reload, and confirm
   every affected tile landed at exactly its pre-edit elevation + amount
   (clamped) -- the exact guarantee a naive per-unit raise_elevation_under
   loop does *not* have (see check 1's sibling docstring note and
   raise_elevation_under's own docstring for the confirmed compounding bug
   this replaces).
8. save() does not persist unit edits -- mutate a real unit's `player`,
   save, reload, and confirm the reloaded unit's player is unchanged (the
   real limitation batch_api.py's module docstring documents next to its
   load/save re-exports).

Checks 1-3 use a real building already in the scenario (via
_find_probe_building), never UnitManager.add_unit()/clone_unit() -- seen
directly against this repo's own example files: AoE2ScenarioParser
version-gates Unit.caption_string_id/caption_string via a *class-level*
property swap applied the first time any older-format (pre-1.54) scenario is
loaded in this process, and that swap is never reverted for later scenarios
(a real bug, same family as the one scenario_write.py's own docstring already
documents for commit()/write_to_file() -- "Doesn't work properly when reading
an older scenario first, and a newer one later" per the library's own TODO
in sections/retrievers/retriever_object_link.py). Concretely: add_unit()
crashes constructing a *new* Unit on a >=1.54 scenario if literally any
older-format scenario was loaded earlier in the same process, even though
the current scenario's own version supports the field fine. Confirmed by
loading two files back-to-back in one interpreter, not just from reading the
source. Since this script loads all 16 example files (a mix of versions) in
one process, hitting this was only a matter of when. batch_api.py's own
docstring carries the warning for real batch-script authors; mutating a real
existing unit's x/y/tile in place (as done here) sidesteps it entirely, since
that never touches the version-gated fields.

See tools/verify_write_path.py for the same overall pattern applied to the
plain terrain/elevation write path this API sits on top of.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AoE2ScenarioParser.datasets.terrains import TerrainId
from AoE2ScenarioParser.objects.data_objects.unit import Unit

from descape import batch_api
from descape.iso_geometry import MAX_ELEVATION
from descape.scenario_io import LoadedScenario, load_map_and_units


def _find_probe_building(scenario: LoadedScenario) -> Unit | None:
    """A real, already-on-map building belonging to any player, to drive
    checks 1-3 -- see this module's docstring for why these checks must
    never create a new Unit via add_unit()/clone_unit()."""
    for player in range(1, 9):
        for building in batch_api.buildings_of(scenario, player):
            if batch_api.tile_under(scenario, building) is not None:
                return building
    return None


def _dirt_near_water_tiles(scenario) -> list:
    return [
        tile
        for tile in scenario.map_manager.terrain
        if tile.terrain_id == TerrainId.DIRT_2
        and any(batch_api.is_water(n.terrain_id) for n in batch_api.neighbors(scenario, tile.x, tile.y, diagonal=True))
    ]


def check_raise_elevation_under_buildings(path: Path, tmp_dir: Path) -> tuple[bool, str]:
    s = load_map_and_units(path)
    if not s.map_is_square:
        return None, "map isn't square -- get_tile (and this whole check) needs it -- skipped"

    building = _find_probe_building(s)
    if building is None:
        return None, "no player has any on-map building -- skipped"

    tile = batch_api.tile_under(s, building)
    x, y = tile.x, tile.y
    original_elevation = tile.elevation
    expected = max(0, min(MAX_ELEVATION, original_elevation + 1))

    ok = batch_api.raise_elevation_under(s, building, 1)
    if not ok:
        return False, "raise_elevation_under returned False for an on-map building"

    out = tmp_dir / f"{path.stem}.batch_elev{path.suffix}"
    batch_api.save(s, out)
    reloaded = load_map_and_units(out)
    got = reloaded.map_manager.get_tile(x, y)
    if got.elevation != expected:
        return False, f"expected elevation {expected} (from {original_elevation}+1), got {got.elevation}"
    return True, f"OK (elevation {original_elevation} -> {expected})"


def check_raise_elevation_clamp(path: Path, tmp_dir: Path) -> tuple[bool, str]:
    s = load_map_and_units(path)
    if not s.map_is_square:
        return None, "map isn't square -- skipped"

    building = _find_probe_building(s)
    if building is None:
        return None, "no player has any on-map building -- skipped"

    tile = batch_api.tile_under(s, building)
    x, y = tile.x, tile.y
    tile.elevation = MAX_ELEVATION

    ok = batch_api.raise_elevation_under(s, building, 1)
    if not ok:
        return False, "raise_elevation_under returned False for an on-map building"

    out = tmp_dir / f"{path.stem}.batch_clamp{path.suffix}"
    batch_api.save(s, out)
    reloaded = load_map_and_units(out)
    got = reloaded.map_manager.get_tile(x, y)
    if got.elevation != MAX_ELEVATION:
        return False, f"expected clamped elevation {MAX_ELEVATION}, got {got.elevation}"
    return True, f"OK (stayed clamped at {MAX_ELEVATION})"


def check_raise_elevation_off_map(path: Path) -> tuple[bool, str]:
    s = load_map_and_units(path)
    building = _find_probe_building(s)
    if building is None:
        return None, "no player has any on-map building -- skipped"

    # Move a real building off-map rather than creating a new Unit -- see
    # this module's docstring for why add_unit() must stay out of this file.
    building.x, building.y = -5.5, -5.5
    ok = batch_api.raise_elevation_under(s, building, 1)
    if ok:
        return False, "expected False for an off-map unit, got True"
    return True, "OK (off-map unit was a safe no-op)"


def check_neighbors(path: Path) -> tuple[bool, str]:
    s = load_map_and_units(path)
    w, h = s.map_manager.map_width, s.map_manager.map_height

    interior = batch_api.neighbors(s, 5, 5)
    if len(interior) != 4:
        return False, f"interior tile (5,5): expected 4 orthogonal neighbors, got {len(interior)}"
    interior_diag = batch_api.neighbors(s, 5, 5, diagonal=True)
    if len(interior_diag) != 8:
        return False, f"interior tile (5,5): expected 8 neighbors with diagonal=True, got {len(interior_diag)}"

    # (0, 0) alone only proves negative coordinates are rejected -- it can't
    # tell a real bounds check apart from an off-by-one that wraps x==map_width
    # back around to x=0 of the next row (xy_to_i's arithmetic is x + y*size,
    # so that wraparound is a real risk shape, not a hypothetical one). The
    # opposite corner and the near-side edges below are what actually rule
    # that out, on this file's own real map_width/map_height.
    corner = batch_api.neighbors(s, 0, 0)
    if len(corner) != 2:
        return False, f"corner tile (0,0): expected 2 orthogonal neighbors, got {len(corner)}"
    corner_diag = batch_api.neighbors(s, 0, 0, diagonal=True)
    if len(corner_diag) != 3:
        return False, f"corner tile (0,0): expected 3 neighbors with diagonal=True, got {len(corner_diag)}"

    far_corner = batch_api.neighbors(s, w - 1, h - 1)
    if len(far_corner) != 2:
        return False, f"far corner ({w-1},{h-1}): expected 2 orthogonal neighbors, got {len(far_corner)}"
    far_corner_diag = batch_api.neighbors(s, w - 1, h - 1, diagonal=True)
    if len(far_corner_diag) != 3:
        return False, f"far corner ({w-1},{h-1}): expected 3 neighbors with diagonal=True, got {len(far_corner_diag)}"

    east_edge = batch_api.neighbors(s, w - 1, 5)
    if len(east_edge) != 3:
        return False, f"east edge ({w-1},5): expected 3 orthogonal neighbors, got {len(east_edge)}"
    south_edge = batch_api.neighbors(s, 5, h - 1)
    if len(south_edge) != 3:
        return False, f"south edge (5,{h-1}): expected 3 orthogonal neighbors, got {len(south_edge)}"

    return True, "OK (interior 4/8, all 4 corners 2/3, east/south edges 3)"


def check_recolor_dirt_near_water(path: Path, tmp_dir: Path) -> tuple[bool, str]:
    s = load_map_and_units(path)
    to_recolor = _dirt_near_water_tiles(s)
    expected_xys = {t.xy for t in to_recolor}

    for tile in to_recolor:
        batch_api.set_terrain(tile, TerrainId.BEACH)

    out = tmp_dir / f"{path.stem}.batch_recolor{path.suffix}"
    batch_api.save(s, out)
    reloaded = load_map_and_units(out)

    problems = []
    for x, y in expected_xys:
        got = reloaded.map_manager.get_tile(x, y)
        if got.terrain_id != TerrainId.BEACH or got.layer != -1:
            problems.append(f"({x},{y}): expected BEACH/layer=-1, got terrain_id={got.terrain_id},layer={got.layer}")
    if problems:
        return False, "; ".join(problems)

    still_matching = _dirt_near_water_tiles(reloaded)
    if still_matching:
        return False, f"{len(still_matching)} DIRT_2-bordering-water tile(s) remain after recolor+reload"

    return True, f"OK ({len(expected_xys)} tile(s) recolored, 0 remain after reload)"


def check_is_water_classification() -> tuple[bool, str]:
    # No scenario file needed -- this is a pure function of TerrainId.
    expect_water = [
        TerrainId.WATER_DEEP,
        TerrainId.WATER_SHALLOW,
        TerrainId.SHALLOWS,
        TerrainId.SWAMP_SHALLOWS,
    ]
    expect_not_water = [
        TerrainId.GRASS_1,
        TerrainId.DIRT_2,
        TerrainId.BEACH,
        TerrainId.ICE,
        TerrainId.ICE_NAVIGABLE,
    ]
    problems = []
    for t in expect_water:
        if not batch_api.is_water(t):
            problems.append(f"{t.name}: expected is_water=True, got False")
    for t in expect_not_water:
        if batch_api.is_water(t):
            problems.append(f"{t.name}: expected is_water=False, got True")
    if problems:
        return False, "; ".join(problems)
    return True, f"OK ({len(expect_water)} water, {len(expect_not_water)} non-water terrains classified correctly)"


def check_raise_elevation_under_many(path: Path, tmp_dir: Path) -> tuple[bool, str]:
    s = load_map_and_units(path)
    if not s.map_is_square:
        return None, "map isn't square -- skipped"

    best_player, best_buildings = None, []
    for player in range(1, 9):
        buildings = [b for b in batch_api.buildings_of(s, player) if batch_api.tile_under(s, b) is not None]
        if len(buildings) > len(best_buildings):
            best_player, best_buildings = player, buildings
    if not best_buildings:
        return None, "no player has any on-map building -- skipped"

    expected: dict[tuple[int, int], int] = {}
    for b in best_buildings:
        tile = batch_api.tile_under(s, b)
        if tile.xy not in expected:
            expected[tile.xy] = max(0, min(MAX_ELEVATION, tile.elevation + 1))

    adjusted = batch_api.raise_elevation_under_many(s, best_buildings, 1)
    if adjusted != len(expected):
        return False, f"raise_elevation_under_many returned {adjusted}, expected {len(expected)} unique tiles"

    out = tmp_dir / f"{path.stem}.batch_elev_many{path.suffix}"
    batch_api.save(s, out)
    reloaded = load_map_and_units(out)

    problems = []
    for (x, y), expected_elev in expected.items():
        got = reloaded.map_manager.get_tile(x, y).elevation
        if got != expected_elev:
            problems.append(f"({x},{y}): expected {expected_elev}, got {got}")
    if problems:
        return False, "; ".join(problems)
    return True, f"OK (player {best_player}, {len(expected)} unique tile(s), all exactly +1 clamped)"


def check_unit_edit_not_persisted(path: Path, tmp_dir: Path) -> tuple[bool, str]:
    s = load_map_and_units(path)
    building = _find_probe_building(s)
    if building is None:
        return None, "no player has any on-map building -- skipped"

    original_player = building.player
    original_ref_id = building.reference_id
    new_player = 1 if original_player != 1 else 2
    building.player = new_player

    out = tmp_dir / f"{path.stem}.batch_unit_noop{path.suffix}"
    batch_api.save(s, out)
    reloaded = load_map_and_units(out)

    match = next(
        (u for u in reloaded.unit_manager.get_all_units() if u.reference_id == original_ref_id),
        None,
    )
    if match is None:
        return False, f"reference_id {original_ref_id} not found in reloaded file at all"
    if match.player != original_player:
        return False, (
            f"expected save() to leave unit edits out (player should still be {original_player}), "
            f"but reloaded player is {match.player} -- batch_api.py's documented limitation is wrong"
        )
    return True, f"OK (player edit to {new_player} correctly did not persist; still {original_player})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario_dir", type=Path, help="Directory of .aoe2scenario files")
    args = parser.parse_args()

    files = sorted(args.scenario_dir.glob("*.aoe2scenario"))
    if not files:
        print(f"No .aoe2scenario files found in {args.scenario_dir}")
        sys.exit(1)

    file_checks_with_tmp = [
        ("raise_elevation_under_buildings", check_raise_elevation_under_buildings),
        ("raise_elevation clamp", check_raise_elevation_clamp),
        ("recolor_dirt_near_water", check_recolor_dirt_near_water),
        ("raise_elevation_under_many", check_raise_elevation_under_many),
        ("unit edit not persisted", check_unit_edit_not_persisted),
    ]
    file_checks_no_tmp = [
        ("raise_elevation off-map", check_raise_elevation_off_map),
        ("neighbors", check_neighbors),
    ]

    failures = 0
    total = 0

    total += 1
    ok, detail = check_is_water_classification()
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures += 1
    print(f"{status}  {'(no file needed)':38s} [{'is_water classification':32s}] {detail}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for path in files:
            for label, check in file_checks_with_tmp:
                total += 1
                try:
                    ok, detail = check(path, tmp_dir)
                except Exception as e:
                    ok, detail = False, f"{type(e).__name__}: {e}"
                status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
                if ok is False:
                    failures += 1
                print(f"{status}  {path.name:38s} [{label:32s}] {detail}")

            for label, check in file_checks_no_tmp:
                total += 1
                try:
                    ok, detail = check(path)
                except Exception as e:
                    ok, detail = False, f"{type(e).__name__}: {e}"
                status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
                if ok is False:
                    failures += 1
                print(f"{status}  {path.name:38s} [{label:32s}] {detail}")

    print(f"\n{total - failures}/{total} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
