"""TASK-431 stage 2: the write-path hard rules re-measured on real HD Edition files with HD Edition's own data.

The scenarios come from the install HD_INSTALL_PATH names (its own templates
span every HD band) plus the scenario dir, both searched recursively and
counted once per resolved path. The derived root is built once per module.
Header-2 HD files (`scn-hd`, bands 1.21, 1.24 and 1.26) are read with `hd`
active, as Auto shows them; header-3 HD files (`hd-aoe2scenario`) with `hd`
hand-picked, as Settings > Advanced > Game data shows them (that the lock
then refuses their edits is tests/test_games.py's). The header-2 files are
all map templates with no wall, gate or ANGLE placement, so those rules rest
on the header-3 files. Skips without the
variable or without such files:

    HD_INSTALL_PATH=<install> pytest -m corpus tests/test_hd_corpus.py --scenario-dir <folder>

Figures only: nothing writes. docs/WRITE_RULES.md quotes the counts measured
on 28 header-2 and 37 header-3 files on 2026-11-27; these tests hold the
rules, not the counts.
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
    hd_content,
    legacy_import,
    object_catalog,
    scenario_format,
    scenario_io,
    unit_rotation,
    unit_sprites,
)

pytestmark = pytest.mark.corpus

ZERO = 1e-4
# Read at import: conftest unsets it inside every test, so no test configures HD Edition by accident.
_INSTALL = os.environ.get(hd_content.INSTALL_ENV)
_SUFFIXES = {".scn", ".scx", ".scx2", ".aoe2scenario"}
ANGLE_COUNTEREXAMPLE_CEILING = 0.001
WALL_OTHER_CEILING = 0.05
WALL_CLASS, CLIFF_CLASS = 27, 34


def _candidates(config: pytest.Config) -> list[Path]:
    override = config.getoption("--scenario-dir")
    roots = [Path(_INSTALL), Path(override) if override else Path(__file__).resolve().parent.parent / "examples"]
    seen, out = set(), []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in _SUFFIXES and path.is_file() and path.resolve() not in seen:
                seen.add(path.resolve())
                out.append(path)
    return out


@pytest.fixture(scope="module")
def hd_files(request) -> dict[str, tuple[Path, ...]]:
    if not _INSTALL:
        pytest.skip(f"set {hd_content.INSTALL_ENV} to an Age of Empires II: HD Edition install")
    files = collections.defaultdict(list)
    for path in _candidates(request.config):
        label = scenario_format.detect_format(path.read_bytes()).game
        if label in (scenario_format.HD, scenario_format.HD_DE):
            files[label].append(path)
    if not files[scenario_format.HD]:
        pytest.skip("no header-2 HD Edition files under the install or the scenario dir")
    return {label: tuple(paths) for label, paths in files.items()}


@pytest.fixture(scope="module")
def roots_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("hd_game_roots")


@pytest.fixture
def census(hd_files, roots_dir) -> dict:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(hd_content.INSTALL_ENV, _INSTALL)
        mp.setattr(hd_content, "CACHE_ROOT", roots_dir)
        asset_source.clear_install_caches(None)
        assert hd_content.ensure_root() is not None, f"{_INSTALL} is not an install validate_install() accepts"
        asset_source.clear_install_caches(None)
        yield _census(hd_files[scenario_format.HD], hd_files.get(scenario_format.HD_DE, ()))
    asset_source.clear_install_caches(None)  # configured() was cached True under the variable just restored


def _is_int(value: float) -> bool:
    return abs(value - round(value)) < ZERO


def _radian_index(value: float) -> int | None:
    k = value / (2 * math.pi / 5)
    return round(k) if _is_int(k) and 0 <= round(k) <= 4 else None


@functools.cache
def _census(header2: tuple[Path, ...], header3: tuple[Path, ...]) -> dict[str, dict]:
    """One pass over every file with `hd` active: a Counter per rule, per header (2 and 3)."""
    out = {2: collections.defaultdict(collections.Counter), 3: collections.defaultdict(collections.Counter)}
    with game_dataset.activated(game_dataset.HD):
        gates = {c for group in gate_orientation.groups().values() for c in group}
        cliffs = unit_sprites.cliff_consts()
        graphics = unit_sprites.graphic_map()
        connectors = unit_sprites.wall_connector_consts()
        objects = object_catalog.dat_objects()
        for header, paths in ((2, header2), (3, header3)):
            for path in paths:
                loaded = scenario_io.load_map_and_units(path)
                out[header]["file_game"][game_dataset.file_game(loaded).id] += 1
                out[header]["bands"][loaded.source_format.data_version] += 1
                tiles = {(int(u.x), int(u.y)) for units in loaded.unit_manager.units for u in units if u.unit_const in connectors}
                for units in loaded.unit_manager.units:
                    for unit in units:
                        _count_unit(out[header], unit, gates, cliffs, graphics, objects, tiles, path.name)
    return out


def _count_unit(out, unit, gates, cliffs, graphics, objects, wall_tiles, name) -> None:
    const, rotation = unit.unit_const, unit.rotation
    out["placements"]["n"] += 1
    entry = objects.get(const)
    if entry is None:
        out["unknown"][const] += 1
    elif entry["class"] == WALL_CLASS and graphics.get(const, {}).get("angle_count") == 5:
        out["wall_class_placed"][const] += 1
    elif entry["class"] == CLIFF_CLASS:
        out["cliff_class_placed"][const] += 1
    if unit_sprites.rotation_variant_eligible(const):
        stored = round(rotation) if _is_int(rotation) and 0 <= round(rotation) <= 4 else _radian_index(rotation)
        encoding = "literal" if _is_int(rotation) and 0 <= round(rotation) <= 4 else "radians" if stored is not None else "other"
        out["walls"][encoding] += 1
        out["walls"]["frame0"] += unit.initial_animation_frame == 0
        out["wall_consts"][const] += 1
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
            out["angle_bad"][const] += 1
    if const in legacy_import.TREE_CONSTS:
        out["trees"]["n"] += 1
        count = graphics.get(const, {}).get("angle_count", 0)
        if not (_is_int(rotation) and 0 <= round(rotation) < count):
            out["tree_bad"][(name, const, rotation, count)] += 1


def _merged(census: dict, key: str) -> collections.Counter:
    return census[2][key] + census[3][key]


def _with_hd(table) -> set[int]:
    with game_dataset.activated(game_dataset.HD):
        return set(table())


def test_header2_files_show_with_hd_data_once_its_install_is_set(census):
    """H3/H5: configured, Auto's game for a header-2 HD file is `hd`, never the stand-in."""
    assert set(census[2]["file_game"]) == {games.HD.id}, dict(census[2]["file_game"])
    assert set(census[2]["bands"]) <= {"1.21", "1.24", "1.26"}, dict(census[2]["bands"])


def test_every_placed_const_is_in_the_hd_dat(census):
    """Nothing in an HD file draws as the unknown-const mark under HD's own data."""
    assert not census[2]["unknown"] and not census[3]["unknown"], (dict(census[2]["unknown"]), dict(census[3]["unknown"]))


def test_the_angle_whitelist_holds_with_the_games_own_data(census):
    seen, bad = sum(census[h]["angle"]["n"] for h in (2, 3)), _merged(census, "angle_bad")
    assert seen, "no ANGLE placement, so this proves nothing"
    assert sum(bad.values()) / seen < ANGLE_COUNTEREXAMPLE_CEILING, dict(bad)


def test_cliffs_are_class_34_and_store_their_frame(census):
    """H11's cliff set (class 34) covers every placed cliff, each storing its frame inside the HD frame count."""
    assert sum(census[h]["cliffs"]["n"] for h in (2, 3)), "no cliff placement"
    assert set(_merged(census, "cliff_class_placed")) <= _with_hd(unit_sprites.cliff_consts)
    assert not _merged(census, "cliff_bad"), sorted(_merged(census, "cliff_bad"))[:10]


def test_trees_store_a_variant_inside_their_hd_graphic(census):
    """H8's premise: HD art draws a stored legacy tree variant as stored, header 2 and header 3 alike."""
    assert census[2]["trees"]["n"] and census[3]["trees"]["n"], "no tree placement"
    assert not _merged(census, "tree_bad"), sorted(_merged(census, "tree_bad"))[:10]


def test_walls_are_the_5_angle_class_27_consts_and_store_both_encodings(census):
    """H11's wall set: every placed 5-angle class-27 const is a shape-variant wall, and variant_index() must read both encodings."""
    walls = _merged(census, "walls")
    total = walls["literal"] + walls["radians"] + walls["other"]
    assert walls["literal"] and walls["radians"], dict(walls)
    assert walls["other"] / total < WALL_OTHER_CEILING, dict(walls)
    assert walls["frame0"] == total, dict(walls)
    assert set(_merged(census, "wall_class_placed")) == set(_merged(census, "wall_consts"))


def test_gates_store_zero_or_an_angle(census):
    assert sum(census[h]["gates"]["n"] for h in (2, 3)), "no gate placement"
    assert not _merged(census, "gate_bad"), sorted(_merged(census, "gate_bad"))[:10]
