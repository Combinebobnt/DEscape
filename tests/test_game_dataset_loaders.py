"""GH #67: every generated-table loader follows the active game dataset.

Const 93 is the anchor: Spearman (PKEMN) in the AoE2 .dat, Hoplite in the
Return of Rome one. Town Center 109 is 4x4 in AoE2 and 3x3 in RoR. Each test
switches to RoR and back, so it also checks nothing stays cached across a
switch, and that the tables other modules imported by name are refilled in
place rather than rebound."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from descape import (
    asset_source,
    batch_api,
    game_dataset,
    garrison,
    object_catalog,
    terrain_blend,
    terrain_palette,
    terrain_units,
    unit_kind,
    unit_rotation,
    unit_sprites,
    unit_stats_table,
)

HOPLITE_SPEARMAN = 93
TOWN_CENTER = 109


def _ror() -> None:
    game_dataset.set_active(game_dataset.ROR)


def _aoe2() -> None:
    game_dataset.set_active(game_dataset.AOE2)


def test_object_names_follow_the_dataset():
    assert object_catalog.display_name(HOPLITE_SPEARMAN) == "Spearman"
    _ror()
    assert object_catalog.display_name(HOPLITE_SPEARMAN) == "Hoplite"
    assert object_catalog.object_name(HOPLITE_SPEARMAN) == "Hoplite"
    _aoe2()
    assert object_catalog.display_name(HOPLITE_SPEARMAN) == "Spearman"


def test_the_ror_object_list_comes_from_the_ror_dat_not_the_aoe2_enums():
    _ror()
    listed = {entry.id: entry for entry in object_catalog.objects()}
    assert listed[HOPLITE_SPEARMAN].name == "Hoplite"
    assert listed[HOPLITE_SPEARMAN].category == "Units"
    assert listed[TOWN_CENTER].category == "Buildings"
    dat = object_catalog.dat_objects()
    spawners = set(object_catalog._ROR_EDITOR_LISTED_DAT_ONLY)
    assert all(not dat[id_].get("hidden") for id_ in listed.keys() - spawners), "an editor-hidden RoR object was listed"
    assert 2607 not in listed or dat[2607].get("hidden") is False, "AoE2's Gunpowder Keg exception leaked into RoR"
    assert object_catalog.combined_object_name(HOPLITE_SPEARMAN) == "Hoplite"


def test_the_ror_list_adds_the_hidden_invisible_spawners_its_editor_shows():
    """GH #67: the RoR .dat hides Invisible Spawner A-J, its in-game editor lists them under Other."""
    _ror()
    dat = object_catalog.dat_objects()
    listed = {entry.id: entry for entry in object_catalog.objects()}
    for id_ in object_catalog._ROR_EDITOR_LISTED_DAT_ONLY:
        assert dat[id_]["hidden"] is True
        assert listed[id_].category == "Others"
        assert listed[id_].hidden is False


def test_the_ror_list_leaves_out_what_its_editor_does_not_show():
    """GH #67: wolf 89 and pyramid 689 (both nameless) are unhidden in the RoR .dat but absent from its editor."""
    _ror()
    dat = object_catalog.dat_objects()
    listed = {entry.id for entry in object_catalog.objects()}
    for id_ in object_catalog._ROR_EDITOR_UNLISTED:
        assert not dat[id_].get("hidden")
        assert id_ not in listed


def test_the_ror_spawners_stay_out_of_the_aoe2_list():
    listed = {entry.id for entry in object_catalog.objects()}
    assert listed.isdisjoint(object_catalog._ROR_EDITOR_LISTED_DAT_ONLY)


def test_the_ror_tech_list_comes_from_the_ror_dat():
    aoe2_count = len(object_catalog.techs())
    _ror()
    techs = object_catalog.techs()
    assert len(techs) == len(object_catalog.dat_techs()) != aoe2_count
    assert {t.category for t in techs} == {"Techs"}


def test_sprites_stats_and_rotation_follow_the_dataset():
    assert unit_sprites.graphic_map()[HOPLITE_SPEARMAN]["file_name"] == "u_inf_spearman_idleA_x1"
    assert unit_stats_table.unit_stats(HOPLITE_SPEARMAN)["hp"] == 45
    _ror()
    assert unit_sprites.graphic_map()[HOPLITE_SPEARMAN]["file_name"] == "u_all_hoplite_idleA_x1"
    assert unit_stats_table.unit_stats(HOPLITE_SPEARMAN)["hp"] == 120
    assert unit_rotation.rotation_is_angle(HOPLITE_SPEARMAN)
    _aoe2()
    assert unit_sprites.graphic_map()[HOPLITE_SPEARMAN]["file_name"] == "u_inf_spearman_idleA_x1"


def test_ror_particle_units_match_aoe2s():
    """GH #67: RoR's Butterflies and Seagulls use the same graphics and effects as AoE2's, so both tables carry them."""
    aoe2 = {k: e for k, e in unit_sprites.graphic_map().items() if e.get("particle")}
    _ror()
    ror = {k: e for k, e in unit_sprites.graphic_map().items() if e.get("particle")}
    assert set(aoe2) == {303, 1608, 1609, 1610}
    assert ror == aoe2


def test_imported_tables_are_refilled_in_place():
    spans, trees, tree_consts = terrain_palette.BUILDING_TILE_SPANS, terrain_palette.TREE_UNIT_IDS, terrain_units.TREE_CONSTS
    blend = terrain_blend.BLEND_TABLE
    aoe2_trees = set(trees)
    assert spans[TOWN_CENTER] == (4, 4)
    _ror()
    assert terrain_palette.BUILDING_TILE_SPANS is spans and terrain_palette.TREE_UNIT_IDS is trees
    assert terrain_units.TREE_CONSTS is tree_consts and terrain_blend.BLEND_TABLE is blend
    assert spans[TOWN_CENTER] == (3, 3)
    assert trees != aoe2_trees and len(trees) == 50
    assert tree_consts <= trees
    assert terrain_palette.terrain_ids(), "terrain names were rebuilt"
    _aoe2()
    assert spans[TOWN_CENTER] == (4, 4) and trees == aoe2_trees


def test_garrison_and_unit_kind_follow_the_dataset():
    aoe2_hosts = sum(1 for const in range(3000) if garrison.can_hold(const))
    aoe2_walls = unit_kind.wall_consts()
    _ror()
    assert sum(1 for const in range(3000) if garrison.can_hold(const)) != aoe2_hosts
    assert unit_kind.wall_consts(), "derived from the RoR tables"
    _aoe2()
    assert unit_kind.wall_consts() == aoe2_walls


def test_ror_gaia_only_follows_its_editors_player_lists():
    """GH #67: what RoR's editor lists only for GAIA goes to GAIA on placement; AoE2's rule is unchanged."""
    deer, lion, gazelle, discovery, gold, keg = 65, 1029, 22, 16, 66, 2607
    assert unit_kind.placement_owner(deer, 1) == 1, "AoE2 lets a player own a Deer"
    assert keg in unit_kind.gaia_only_consts()
    _ror()
    gaia_only = unit_kind.gaia_only_consts()
    assert {deer, lion, gazelle, discovery, gold} <= gaia_only
    assert unit_kind.placement_owner(deer, 1) == 0
    ornlu, revealer, furious, granary, bowman = 707, 0, 860, 1089, 4
    assert not {ornlu, revealer, furious, granary, bowman} & gaia_only
    assert not set(unit_kind._ROR_PLAYER_OTHERS) & gaia_only


def test_batch_api_is_building_reads_the_ror_dat():
    unit = SimpleNamespace(unit_const=HOPLITE_SPEARMAN)
    assert not batch_api.is_building(unit)
    town_center = SimpleNamespace(unit_const=TOWN_CENTER)
    assert batch_api.is_building(town_center)
    _ror()
    assert batch_api.is_building(town_center)
    assert not batch_api.is_building(unit)


@pytest.fixture()
def ror_install(tmp_path, monkeypatch):
    monkeypatch.setenv("AOE2DE_INSTALL_PATH", str(tmp_path))
    asset_source.clear_install_caches()
    return tmp_path


def test_ror_strings_are_read_over_the_base_strings(ror_install):
    base = ror_install / asset_source.STRINGS_SUBPATH_TEMPLATE.format(lang="en")
    base.parent.mkdir(parents=True)
    base.write_text('5078 "Spearman"\n305188 "Base Hoplite"\n1 "Only base"\n', encoding="utf-8")
    ror = ror_install / game_dataset.ROR_SUBPATH / asset_source.ROR_STRINGS_SUBPATH_TEMPLATE.format(lang="en")
    ror.parent.mkdir(parents=True)
    ror.write_text('305188 "Hoplite"\n', encoding="utf-8")
    assert asset_source.resource_string(305188, "en") == "Base Hoplite"
    _ror()
    assert asset_source.resource_string(305188, "en") == "Hoplite"
    assert asset_source.resource_string(1, "en") == "Only base"
    assert object_catalog.display_name(HOPLITE_SPEARMAN) == "Hoplite"


def test_ror_icons_and_textures_come_from_the_ror_root_first(ror_install):
    for root in (ror_install, ror_install / game_dataset.ROR_SUBPATH):
        folder = root / asset_source.UNIT_ICON_DIR
        folder.mkdir(parents=True)
        (folder / asset_source.UNIT_ICON_NAME.format(icon=16)).write_bytes(b"x")
    (ror_install / asset_source.UNIT_ICON_DIR / asset_source.UNIT_ICON_NAME.format(icon=31)).write_bytes(b"x")
    assert asset_source.unit_icon_path(16).parent == ror_install / asset_source.UNIT_ICON_DIR
    _ror()
    assert asset_source.unit_icon_path(16).parent == ror_install / game_dataset.ROR_SUBPATH / asset_source.UNIT_ICON_DIR
    assert asset_source.unit_icon_path(31).parent == ror_install / asset_source.UNIT_ICON_DIR, "falls back to the install"


def _atlas(*textures: tuple[str, str]) -> str:
    entries = [{"RefName": ref, "FileName": name} for ref, name in textures]
    return json.dumps({"AtlasTextures": [{"AtlasDef": {"Name": "a", "Textures": entries}}]})


def test_ror_portraits_follow_its_icon_table(ror_install):
    """GH #67: RoR's icons.json renumbers portraits; an index names a RoR atlas file or an AoE2 one."""
    ror_ui = ror_install / game_dataset.ROR_SUBPATH / asset_source.UI_DIR
    aoe2_ui = ror_install / asset_source.UI_DIR
    for path in (
        ror_ui / "textures/ingame/units/016_50730.dds",
        ror_ui / "textures/ingame/units/001_50730.dds",
        aoe2_ui / "textures/ingame/units/589_50730.DDS",
        aoe2_ui / "textures/ingame/units/294_50730.dds",
        aoe2_ui / "textures/ingame/buildings/052_aqueduct.dds",
        aoe2_ui / "textures/ingame/units/146_50730.dds",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    table = {
        "Units": {"016": "UnitIcons016", "001": "UnitIcons29450730", "146": "UnitIcons58950730", "099": "Nowhere"},
        "Buildings": {"080": "BuildingIcons052Aqueduct"},
    }
    (ror_ui / asset_source.ICON_TABLE_NAME).write_text(json.dumps(table), encoding="utf-8")
    (ror_ui / asset_source.ROR_MATERIALS_NAME).write_text(
        _atlas(("UnitIcons016", "textures/ingame/units/016_50730.dds"), ("UnitIcons001", "textures/ingame/units/001_50730.dds")),
        encoding="utf-8",
    )
    (aoe2_ui / asset_source.MATERIALS_NAME).write_text(
        _atlas(
            ("UnitIcons29450730", "textures/ingame/units/294_50730.dds"),
            ("UnitIcons58950730", "textures/ingame/units/589_50730.dds"),
            ("BuildingIcons052Aqueduct", "textures/ingame/buildings/052_aqueduct.dds"),
        ),
        encoding="utf-8",
    )
    assert asset_source.unit_icon_path(146).name == "146_50730.dds", "AoE2 ignores the RoR table"
    _ror()
    assert asset_source.unit_icon_path(16) == ror_ui / "textures/ingame/units/016_50730.dds"
    assert asset_source.unit_icon_path(1) == aoe2_ui / "textures/ingame/units/294_50730.dds", "the table beats RoR's own 001"
    assert asset_source.unit_icon_path(146) == aoe2_ui / "textures/ingame/units/589_50730.DDS", "file name case varies"
    assert asset_source.unit_icon_path(80, building=True) == aoe2_ui / "textures/ingame/buildings/052_aqueduct.dds"
    assert asset_source.unit_icon_path(99) is None, "an unknown RefName falls back to the index lookup"


def test_ror_sld_lookup_prefers_the_ror_graphics(ror_install):
    base = ror_install / unit_sprites.GRAPHICS_SUBPATH
    ror = ror_install / game_dataset.ROR_SUBPATH / unit_sprites.GRAPHICS_SUBPATH
    ror.mkdir(parents=True)
    (ror / "u_all_hoplite_idleA_x1.sld").write_bytes(b"")
    assert unit_sprites._sld_path(base, "u_all_hoplite_idleA_x1") == base / "u_all_hoplite_idleA_x1.sld"
    _ror()
    assert unit_sprites._sld_path(base, "u_all_hoplite_idleA_x1") == ror / "u_all_hoplite_idleA_x1.sld"
    assert unit_sprites._sld_path(base, "only_base") == base / "only_base.sld"


def test_ror_sld_bytes_reread_the_ror_file_after_eviction(ror_install):
    """GH #67: a decode whose walked bytes left the cache re-reads the file the walk read, not the install's."""
    base = ror_install / unit_sprites.GRAPHICS_SUBPATH
    ror = ror_install / game_dataset.ROR_SUBPATH / unit_sprites.GRAPHICS_SUBPATH
    base.mkdir(parents=True)
    ror.mkdir(parents=True)
    (base / "u_all_bowman_idleA_x1.sld").write_bytes(b"aoe2")
    (ror / "u_all_bowman_idleA_x1.sld").write_bytes(b"ror")
    _ror()
    unit_sprites._sld_bytes_cache.clear()
    assert unit_sprites._sld_bytes(base, "u_all_bowman_idleA_x1") == b"ror"
