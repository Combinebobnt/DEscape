"""GH #58: a data mod's derived tables replace every generated table while it is active.

The overlay rides game_dataset (GH #67's switching layer): set_data_mod() names
a folder of tables, and every loader and cache follows it. The sentinel test
writes one changed value per table into a synthetic mod folder and reads each
back through its public accessor, then switches the mod off and expects the
install's values again: an accessor whose cache ignored the switch fails here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from descape import (
    asset_source,
    dat_derive,
    game_dataset,
    garrison,
    object_catalog,
    terrain_blend,
    terrain_classes,
    terrain_units,
    unit_rotation,
    unit_sprites,
    unit_stats_table,
)
from descape.terrain_palette import BUILDING_TILE_SPANS, TREE_UNIT_IDS

DESCAPE = Path(game_dataset.__file__).resolve().parent
SPEARMAN, TOWN_CENTER, CASTLE = 93, 109, 82
NEW_UNIT = 9990  # no install table has it
NEW_TREE = 9991


def _load(folder: Path, name: str) -> dict:
    return json.loads((folder / name).read_text())


def _save(folder: Path, name: str, data: dict) -> None:
    (folder / name).write_text(json.dumps(data))


@pytest.fixture()
def mod_tables(tmp_path) -> game_dataset.DataModTables:
    """A copy of the AoE2 tables with one sentinel per table."""
    folder = tmp_path / "tables"
    folder.mkdir()
    for name in dat_derive.TABLE_NAMES:
        shutil.copy(DESCAPE / name, folder / name)
    stats = _load(folder, "unit_stats.json")
    stats["stats"][str(SPEARMAN)]["hp"] = 4242
    _save(folder, "unit_stats.json", stats)
    catalog = _load(folder, "object_catalog.json")
    catalog["objects"][str(NEW_UNIT)] = {**catalog["objects"][str(SPEARMAN)], "code": "MODUNIT", "string_id": -1}
    _save(folder, "object_catalog.json", catalog)
    graphics = _load(folder, "unit_graphic_map.json")
    graphics["graphics"][str(SPEARMAN)]["file_name"] = "sentinel_x1"
    graphics["graphics"][str(NEW_UNIT)] = graphics["graphics"][str(SPEARMAN)]
    _save(folder, "unit_graphic_map.json", graphics)
    render_data = _load(folder, "unit_render_data.json")
    render_data["buildings"][str(TOWN_CENTER)] = [5, 5]
    _save(folder, "unit_render_data.json", render_data)
    trees = _load(folder, "tree_unit_ids.json")
    trees["ids"][str(NEW_TREE)] = "MODTREE"
    _save(folder, "tree_unit_ids.json", trees)
    hosts = _load(folder, "garrison_table.json")
    hosts["hosts"][str(CASTLE)]["cap"] = 77
    _save(folder, "garrison_table.json", hosts)
    classes = _load(folder, "terrain_classes.json")
    classes["classes"]["0"]["family"] = "sentinel"
    _save(folder, "terrain_classes.json", classes)
    textures = _load(folder, "terrain_texture_map.json")
    textures["mapping"]["0"] = "sentinel.dds"
    _save(folder, "terrain_texture_map.json", textures)
    blend = _load(folder, "terrain_blend_map.json")
    blend["terrains"]["0"]["priority"] = 4321
    _save(folder, "terrain_blend_map.json", blend)
    art = _load(folder, "building_art_map.json")
    art["civ_names"]["1"] = "Sentinelians"
    _save(folder, "building_art_map.json", art)
    units = _load(folder, "terrain_unit_map.json")
    first = next(iter(units["terrains"]))
    units["terrains"][first] = [{**units["terrains"][first][0], "id": NEW_TREE}]
    _save(folder, "terrain_unit_map.json", units)
    return game_dataset.DataModTables("sentinel", folder, tmp_path / "mod", "Sentinel Mod", frozenset({NEW_UNIT}))


def _readings() -> dict:
    """One value per table, each through a public accessor; the two imported names by binding."""
    return {
        "unit_stats": unit_stats_table.unit_stats(SPEARMAN)["hp"],
        "object_catalog": NEW_UNIT in {entry.id for entry in object_catalog.objects()},
        "rotation": unit_rotation.rotation_is_angle(NEW_UNIT),
        "unit_graphic_map": unit_sprites.graphic_map()[SPEARMAN]["file_name"],
        "unit_render_data": BUILDING_TILE_SPANS[TOWN_CENTER],
        "tree_unit_ids": NEW_TREE in TREE_UNIT_IDS,
        "garrison_table": garrison.capacity(CASTLE),
        "terrain_classes": terrain_classes.terrain_family(0),
        "terrain_texture_map": asset_source._terrain_texture_map().get(0),
        "terrain_blend_map": terrain_blend.BLEND_TABLE[0].priority,
        "building_art_map": unit_sprites.building_art().civ_names[1],
        "terrain_unit_map": NEW_TREE in terrain_units.TREE_CONSTS,
    }


def test_every_table_follows_the_data_mod_and_back(mod_tables):
    install = _readings()
    assert game_dataset.set_data_mod(mod_tables)
    modded = _readings()
    assert modded == {
        "unit_stats": 4242,
        "object_catalog": True,
        "rotation": True,
        "unit_graphic_map": "sentinel_x1",
        "unit_render_data": (5, 5),
        "tree_unit_ids": True,
        "garrison_table": 77,
        "terrain_classes": "sentinel",
        "terrain_texture_map": "sentinel.dds",
        "terrain_blend_map": 4321,
        "building_art_map": "Sentinelians",
        "terrain_unit_map": True,
    }
    assert all(install[key] != modded[key] for key in install), "a sentinel equals the install's value"
    assert game_dataset.set_data_mod(None)
    assert _readings() == install


def test_the_mod_is_inert_under_return_of_rome(mod_tables):
    ror_stats = json.loads(game_dataset.table_path("unit_stats.json", game_dataset.ROR).read_text())["stats"]
    game_dataset.set_data_mod(mod_tables)
    game_dataset.set_active(game_dataset.ROR)
    assert game_dataset.data_mod() is None and game_dataset.cache_key() == game_dataset.ROR
    assert unit_stats_table.unit_stats(SPEARMAN)["hp"] == ror_stats[str(SPEARMAN)]["hp"], "RoR's own Hoplite"
    game_dataset.set_active(game_dataset.AOE2)
    assert unit_stats_table.unit_stats(SPEARMAN)["hp"] == 4242, "the mod applies again on AoE2"


def test_explicit_aoe2_reads_stay_on_the_install_tables(mod_tables):
    game_dataset.set_data_mod(mod_tables)
    assert game_dataset.table_path("object_catalog.json", game_dataset.AOE2) == DESCAPE / "object_catalog.json"
    assert game_dataset.table_path("object_catalog.json") == mod_tables.tables_dir / "object_catalog.json"
    assert game_dataset.table_path("cliff_connectivity.json") == DESCAPE / "cliff_connectivity.json", "shared"


def test_two_mods_with_one_key_keep_separate_caches(mod_tables, tmp_path):
    """Byte-identical .dat files share a cache key but not their strings or folder: the overlay is the cache identity."""
    twin = game_dataset.DataModTables(mod_tables.key, mod_tables.tables_dir, tmp_path / "twin", "Twin")
    game_dataset.set_data_mod(mod_tables)
    first = game_dataset.cache_key()
    assert game_dataset.set_data_mod(twin) and game_dataset.cache_key() != first


def test_generation_moves_on_every_effective_change(mod_tables):
    start = game_dataset.generation()
    assert game_dataset.set_data_mod(mod_tables) and game_dataset.generation() == start + 1
    assert not game_dataset.set_data_mod(mod_tables) and game_dataset.generation() == start + 1
    game_dataset.set_active(game_dataset.ROR)
    assert game_dataset.generation() == start + 2
    assert not game_dataset.set_data_mod(None), "inert under RoR: nothing in effect changed"
    assert game_dataset.generation() == start + 2


def test_mod_only_rows_are_tagged_and_keep_their_category(mod_tables):
    game_dataset.set_data_mod(mod_tables)
    row = next(entry for entry in object_catalog.objects() if entry.id == NEW_UNIT)
    assert row.name.endswith(object_catalog.DATA_MOD_TAG) and row.category == "Units"
    spearman = next(entry for entry in object_catalog.objects() if entry.id == SPEARMAN)
    assert object_catalog.DATA_MOD_TAG not in spearman.name


def test_mod_strings_override_the_install_key_by_key(tmp_path, monkeypatch, mod_tables):
    install = tmp_path / "install"
    base = install / asset_source.STRINGS_SUBPATH_TEMPLATE.format(lang="en")
    base.parent.mkdir(parents=True)
    base.write_text('1 "Install one"\n2 "Install two"\n3 "Install three"\n', encoding="utf-8")
    folder = mod_tables.root / "Resources" / "en" / "strings" / "key-value"
    folder.mkdir(parents=True)
    (folder / "key-value-strings-utf8.txt").write_text('2 "Mod full two"\n3 "Mod full three"\n', encoding="utf-8")
    (folder / "key-value-modded-strings-utf8.txt").write_text('3 "Mod three"\n', encoding="utf-8")
    monkeypatch.setenv("AOE2DE_INSTALL_PATH", str(install))
    asset_source.clear_install_caches()
    game_dataset.set_data_mod(mod_tables)
    assert [asset_source.resource_string(k, "en") for k in (1, 2, 3)] == ["Install one", "Mod full two", "Mod three"]
    game_dataset.set_data_mod(None)
    assert asset_source.resource_string(3, "en") == "Install three"

