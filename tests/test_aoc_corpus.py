"""TASK-430 stage 2: the write-path hard rules re-measured on real Age of Kings / The Conquerors files with that game's own data.

Every `.scn`/`.scx` under the scenario dir (searched recursively, as
tests/test_legacy_corpus.py does) whose format the `aoc` game claims is
loaded once, with the derived root of the install AOC_INSTALL_PATH names
built once per module and `aoc` active. Skips without that variable or
without such files:

    AOC_INSTALL_PATH=<install> pytest -m corpus tests/test_aoc_corpus.py --scenario-dir <folder>

Figures only: nothing writes. docs/WRITE_RULES.md quotes the counts measured
on 538 files (156 AoK, 382 AoC) on 2026-11-27, beside the census of the same
files under AoE2 (DE) data; these tests hold the rules, not the counts.
"""

from __future__ import annotations

import collections
import functools
import math
import os
from pathlib import Path

import pytest

from descape import (
    asset_source,
    cliff_catalog,
    game_dataset,
    games,
    gate_orientation,
    legacy_content,
    legacy_import,
    scenario_format,
    scenario_io,
    terrain_classes,
    terrain_palette,
    unit_rotation,
    unit_sprites,
)

pytestmark = pytest.mark.corpus

ZERO = 1e-4
# Read at import: conftest unsets it inside every test, so no test configures the 1999 game by accident.
_INSTALL = os.environ.get(legacy_content.INSTALL_ENV)
# 2 of 57,823 ANGLE placements measured (2,226 of 60,169 under AoE2 data); a regenerated table that loses type 70 fails this.
ANGLE_COUNTEREXAMPLE_CEILING = 0.001
# 1,707 of 90,544 walls store neither encoding.
WALL_OTHER_CEILING = 0.05
SNOW_TERRAINS = (33, 34)


@pytest.fixture(scope="module")
def aoc_files(request) -> tuple[Path, ...]:
    import test_legacy_corpus

    if not _INSTALL:
        pytest.skip(f"set {legacy_content.INSTALL_ENV} to an Age of Kings / The Conquerors install")
    _root, candidates = test_legacy_corpus._legacy_candidates(request.config)
    seen, files = set(), []
    for path in candidates:
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        # The figures below are AoK/AoC's; band 1.23 (scn-up) has its own in docs/WRITE_RULES.md (435 const-302 misses).
        if scenario_format.detect_format(path.read_bytes()).game in games.AOC.formats - {scenario_format.UP}:
            files.append(path)
    if not files:
        pytest.skip("no Age of Kings / The Conquerors files in the scenario dir")
    return tuple(files)


@pytest.fixture(scope="module")
def roots_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("aoc_game_roots")


@pytest.fixture
def census(aoc_files, roots_dir) -> dict:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(legacy_content.INSTALL_ENV, _INSTALL)
        mp.setattr(legacy_content, "CACHE_ROOT", roots_dir)
        asset_source.clear_install_caches(None)
        assert legacy_content.ensure_root() is not None, f"{_INSTALL} is not an install validate_install() accepts"
        asset_source.clear_install_caches(None)
        yield _census(aoc_files)
    asset_source.clear_install_caches(None)  # configured() was cached True under the variable just restored


def _is_int(value: float) -> bool:
    return abs(value - round(value)) < ZERO


def _radian_index(value: float) -> int | None:
    k = value / (2 * math.pi / 5)
    return round(k) if _is_int(k) and 0 <= round(k) <= 4 else None


@functools.cache
def _census(files: tuple[Path, ...]) -> dict:
    """One pass over every file with `aoc` active: a Counter per rule."""
    out = collections.defaultdict(collections.Counter)
    with game_dataset.activated(game_dataset.AOC):
        gates = {c for group in gate_orientation.groups().values() for c in group}
        cliffs = unit_sprites.cliff_consts()
        graphics = unit_sprites.graphic_map()
        connectors = unit_sprites.wall_connector_consts()
        out["names"] = {t: terrain_palette.name_for_terrain_id(t) for t in SNOW_TERRAINS}
        out["families"] = {t: (terrain_classes.terrain_family(t), terrain_classes.terrain_climate(t)) for t in SNOW_TERRAINS}
        for path in files:
            loaded = scenario_io.load_map_and_units(path)
            out["file_game"][game_dataset.file_game(loaded).id] += 1
            for tile in loaded.map_manager.terrain:
                if tile.terrain_id in SNOW_TERRAINS:
                    out["snow_tiles"][tile.terrain_id] += 1
            tiles = {(int(u.x), int(u.y)) for units in loaded.unit_manager.units for u in units if u.unit_const in connectors}
            for units in loaded.unit_manager.units:
                for unit in units:
                    _count_unit(out, unit, gates, cliffs, graphics, tiles, path.name)
    return out


def _count_unit(out, unit, gates, cliffs, graphics, wall_tiles, name) -> None:
    const, rotation = unit.unit_const, unit.rotation
    if unit_sprites.rotation_variant_eligible(const):
        stored = round(rotation) if _is_int(rotation) and 0 <= round(rotation) <= 4 else _radian_index(rotation)
        encoding = "literal" if _is_int(rotation) and 0 <= round(rotation) <= 4 else "radians" if stored is not None else "other"
        out["walls"][encoding] += 1
        out["walls"]["frame0"] += unit.initial_animation_frame == 0
        derived = unit_sprites.wall_variant_from_neighbours(unit_sprites.neighbour_mask(int(unit.x), int(unit.y), wall_tiles))
        if derived is not None and stored is not None:
            out["walls"]["derivable"] += 1
            out["walls"]["agree"] += derived == stored
    elif const in gates:
        out["gates"]["n"] += 1
        out["gates"]["zero"] += abs(rotation) < ZERO
        if not -ZERO <= rotation <= 2 * math.pi + ZERO:
            out["gate_bad"][(name, const, rotation)] += 1
    elif const in cliffs:
        out["cliffs"]["n"] += 1
        if not (_is_int(rotation) and round(rotation) == unit.initial_animation_frame):
            out["cliff_bad"][(name, const, rotation, unit.initial_animation_frame)] += 1
        elif not 0 <= unit.initial_animation_frame < cliff_catalog.frame_count(const):
            out["cliff_bad"][(name, const, "frame", unit.initial_animation_frame)] += 1
    elif unit_rotation.rotation_is_angle(const):
        out["angle"]["n"] += 1
        if _is_int(rotation) and 0 <= round(rotation) < unit_rotation.angle_count_for(const) and abs(rotation) > ZERO:
            out["angle"]["counterexamples"] += 1
    if const in legacy_import.TREE_CONSTS:
        out["trees"]["n"] += 1
        count = graphics.get(const, {}).get("angle_count", 0)
        if not (_is_int(rotation) and 0 <= round(rotation) < count):
            out["tree_bad"][(name, const, rotation, count)] += 1


def test_every_file_shows_with_the_1999_data_once_its_install_is_set(census):
    """D4: configured, Auto's game is the file's own, never the stand-in."""
    assert set(census["file_game"]) == {games.AOC.id}, dict(census["file_game"])


def test_the_angle_whitelist_holds_with_the_games_own_data(census):
    """The 1999 .dat's own type 70: AoE2's whitelist met 2,226 integer-index counterexamples on these files."""
    seen, bad = census["angle"]["n"], census["angle"]["counterexamples"]
    assert seen, "no ANGLE placement, so this proves nothing"
    assert bad / seen < ANGLE_COUNTEREXAMPLE_CEILING, f"{bad}/{seen}"


def test_cliffs_store_their_frame_inside_the_1999_frame_count(census):
    assert census["cliffs"]["n"], "no cliff placement"
    assert not census["cliff_bad"], sorted(census["cliff_bad"])[:10]


def test_trees_store_a_variant_inside_their_1999_graphic(census):
    """Stage 3's premise for drawing a header-2 file's trees as stored under the 1999 art (legacy_tree_variants)."""
    assert census["trees"]["n"], "no tree placement"
    assert not census["tree_bad"], sorted(census["tree_bad"])[:10]


def test_walls_store_both_encodings(census):
    """variant_index() must read both: neither encoding goes away with the game's own data."""
    walls = census["walls"]
    total = walls["literal"] + walls["radians"] + walls["other"]
    assert walls["literal"] and walls["radians"], dict(walls)
    assert walls["other"] / total < WALL_OTHER_CEILING, dict(walls)


def test_gates_store_zero_or_an_angle(census):
    """A 1999 gate may store a real angle (DE's gates store 0 or 7), never a value outside one turn."""
    assert census["gates"]["n"], "no gate placement"
    assert not census["gate_bad"], sorted(census["gate_bad"])[:10]


def test_snow_dirt_and_snow_grass_are_1999_terrains(census):
    """Ids 33 and 34, obsolete in AoE2:DE, are snow land in the 1999 game and named from its .dat."""
    assert census["families"] == dict.fromkeys(SNOW_TERRAINS, ("land", "snow"))
    assert all(not name.startswith("OBSOLETE") for name in census["names"].values()), census["names"]
    assert census["snow_tiles"], "no tile of either"
