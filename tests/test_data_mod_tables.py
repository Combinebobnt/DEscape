"""GH #58: a data mod's derived tables replace every generated table while it is active.

The overlay rides game_dataset (GH #67's switching layer): set_data_mod() names
a folder of tables, and every loader and cache follows it. The sentinel test
writes one changed value per table into a synthetic mod folder and reads each
back through its public accessor, then switches the mod off and expects the
install's values again: an accessor whose cache ignored the switch fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from descape import (
    asset_source,
    data_mods,
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
from testkit import data_mod_fixture
from testkit.data_mod_fixture import ARCHER, CASTLE, NEW_TREE, NEW_UNIT, SPEARMAN, TOWN_CENTER

DESCAPE = Path(game_dataset.__file__).resolve().parent


@pytest.fixture()
def mod_tables(tmp_path) -> game_dataset.DataModTables:
    return data_mod_fixture.mod_tables(tmp_path)


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



# --- GH #265: Settings > Mods' unchanged objects and "(data mod)" label ---------

CROSSBOWMAN, LONGBOWMAN, KNIGHT, MILITIA, BARRACKS = 7, 8, 38, 74, 12


def _edit_rows(mod: game_dataset.DataModTables) -> None:
    """Militia repurposed and a Knight with another icon (both changed); a Longbowman
    other only in civs, a Crossbowman only in graphic_id, Barracks only in its garrison flag (all unchanged)."""
    folder = mod.tables_dir
    catalog = data_mod_fixture.load(folder, "object_catalog.json")
    objects = catalog["objects"]
    objects[str(MILITIA)]["code"] = "MODWORKER"
    objects[str(KNIGHT)]["icon"] = 999
    objects[str(LONGBOWMAN)]["civs"] = [99]
    data_mod_fixture.save(folder, "object_catalog.json", catalog)
    graphics = data_mod_fixture.load(folder, "unit_graphic_map.json")
    graphics["graphics"][str(CROSSBOWMAN)]["graphic_id"] += 1000
    graphics["graphics"][str(BARRACKS)]["garrison_flags"][0]["file_name"] = "b_misc_garrison_flag_x1"
    data_mod_fixture.save(folder, "unit_graphic_map.json", graphics)


def test_unchanged_ids_are_the_aoe2_objects_the_mod_left_alone(mod_tables):
    _edit_rows(mod_tables)
    game_dataset.set_data_mod(mod_tables)
    unchanged = data_mods.unchanged_ids()
    assert {ARCHER, LONGBOWMAN, CROSSBOWMAN, BARRACKS} <= unchanged
    assert not {SPEARMAN, NEW_UNIT, MILITIA, KNIGHT} & unchanged


def test_a_name_the_mods_strings_define_is_changed(mod_tables):
    archer_string = object_catalog.dat_objects()[ARCHER]["string_id"]
    folder = mod_tables.root / "resources" / "en" / "strings" / "key-value"
    folder.mkdir(parents=True)
    (folder / "key-value-modded-strings-utf8.txt").write_text(f'{archer_string} "Bowman"\n', encoding="utf-8")
    game_dataset.set_data_mod(mod_tables)
    assert archer_string in asset_source.data_mod_string_ids("en")
    assert ARCHER not in data_mods.unchanged_ids()


def test_unchanged_ids_is_empty_with_no_mod_and_under_return_of_rome(mod_tables):
    assert data_mods.unchanged_ids() == frozenset()
    game_dataset.set_data_mod(mod_tables)
    assert ARCHER in data_mods.unchanged_ids()
    game_dataset.set_active(game_dataset.ROR)
    assert data_mods.unchanged_ids() == frozenset()


def test_the_data_mod_label_follows_its_setting(mod_tables):
    from descape import settings

    game_dataset.set_data_mod(mod_tables)

    def name() -> str:
        return object_catalog.name_for(object_catalog.objects(), NEW_UNIT)

    assert name().endswith(object_catalog.DATA_MOD_TAG)
    settings.set_data_mod_label_shown(False)
    object_catalog.objects.cache_clear()
    assert name() and object_catalog.DATA_MOD_TAG not in name()
    settings.set_data_mod_label_shown(True)
    object_catalog.objects.cache_clear()
    assert name().endswith(object_catalog.DATA_MOD_TAG)
