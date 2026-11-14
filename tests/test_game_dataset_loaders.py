"""GH #67: every generated-table loader follows the active game dataset.

Const 93 is the anchor: Spearman (PKEMN) in the AoE2 .dat, Hoplite in the
Return of Rome one. Town Center 109 is 4x4 in AoE2 and 3x3 in RoR. Each test
switches to RoR and back, so it also checks nothing stays cached across a
switch, and that the tables other modules imported by name are refilled in
place rather than rebound."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import (
    asset_source,
    batch_api,
    game_dataset,
    games,
    garrison,
    mod_graphics,
    object_catalog,
    placement_rules,
    terrain_blend,
    terrain_catalog,
    terrain_palette,
    terrain_units,
    trigger_fields,
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
    shown = {id_ for id_, entry in listed.items() if not entry.hidden}
    assert all(not dat[id_].get("hidden") for id_ in shown - spawners), "an editor-hidden RoR object was listed as shown"
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


def test_both_games_list_the_invisible_spawners_under_other():
    """GH #67 measured them in RoR's editor, GH #32 in AoE2's 1.59 editor (same ids)."""
    spawners = set(object_catalog._ROR_EDITOR_LISTED_DAT_ONLY)
    aoe2 = {entry.id: entry.category for entry in object_catalog.objects()}
    assert all(aoe2.get(const) == "Others" for const in spawners)
    _ror()
    ror = {entry.id: entry.category for entry in object_catalog.objects()}
    assert all(ror.get(const) == "Others" for const in spawners)


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
    """GH #67, GH #311: RoR's Butterflies, Seagulls, Smoke and Flames use the same graphics and
    effects as AoE2's (measured on the RoR .dat 2026-11-02), so both tables carry the same entries."""
    aoe2 = {k: e for k, e in unit_sprites.graphic_map().items() if e.get("particle")}
    _ror()
    ror = {k: e for k, e in unit_sprites.graphic_map().items() if e.get("particle")}
    assert sorted(ror) == [303, 1308, 1333, 1334, 1335, 1336, 1608, 1609, 1610]
    assert ror == aoe2


def test_blend_priorities_follow_the_dataset_both_ways():
    """BLEND_TABLE is refilled in place, so the priority lookup keyed on its
    identity must be dropped too, or the first game's order sticks."""

    def priorities():
        prio, _btype = terrain_blend._luts()
        return prio.tolist()

    aoe2 = priorities()
    _ror()
    ror = priorities()
    assert ror == [terrain_blend.BLEND_TABLE[t].priority if t in terrain_blend.BLEND_TABLE else -1 for t in range(len(ror))]
    assert ror != aoe2
    _aoe2()
    assert priorities() == aoe2


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
    """GH #67: what RoR's editor lists only for GAIA goes to GAIA on placement; AoE2 has its own rule (GH #32)."""
    deer, lion, gazelle, discovery, gold, keg = 65, 1029, 22, 16, 66, 2607
    assert discovery not in unit_kind.gaia_only_consts(), "AoE2's 16 is not Discovery"
    assert keg in unit_kind.gaia_only_consts()
    _ror()
    gaia_only = unit_kind.gaia_only_consts()
    assert {deer, lion, gazelle, discovery, gold} <= gaia_only
    assert unit_kind.placement_owner(deer, 1) == 0
    ornlu, revealer, furious, granary, bowman = 707, 0, 860, 1089, 4
    assert not {ornlu, revealer, furious, granary, bowman} & gaia_only
    assert not set(unit_kind._ROR_PLAYER_OTHERS) & gaia_only
    assert unit_kind.owner_refused(gazelle, 1) and not unit_kind.owner_refused(gazelle, 0)
    assert not unit_kind.owner_refused(granary, 1)


def test_batch_api_is_building_reads_the_ror_dat():
    unit = SimpleNamespace(unit_const=HOPLITE_SPEARMAN)
    assert not batch_api.is_building(unit)
    town_center = SimpleNamespace(unit_const=TOWN_CENTER)
    assert batch_api.is_building(town_center)
    _ror()
    assert batch_api.is_building(town_center)
    assert not batch_api.is_building(unit)


def test_batch_api_load_passes_its_keyword_options_through():
    """batch_api.load was load_map_and_units itself before GH #67; scripts pass its options."""
    from descape import scenario_io

    loaded = batch_api.load(scenario_io.BLANK_TEMPLATE_PATH, fast_terrain=False, fast_units=False)
    assert loaded.map_manager.map_width > 0
    assert game_dataset.active() == game_dataset.AOE2


@pytest.fixture()
def ror_install(tmp_path, monkeypatch):
    monkeypatch.setenv("AOE2DE_INSTALL_PATH", str(tmp_path))
    asset_source.clear_install_caches()
    return tmp_path


def test_ror_strings_are_read_over_the_base_strings(ror_install):
    base = ror_install / asset_source.STRINGS_SUBPATH_TEMPLATE.format(lang="en")
    base.parent.mkdir(parents=True)
    base.write_text('5078 "Spearman"\n305188 "Base Hoplite"\n1 "Only base"\n', encoding="utf-8")
    ror = ror_install / games.ROR.root_subpath / asset_source.ROR_STRINGS_SUBPATH_TEMPLATE.format(lang="en")
    ror.parent.mkdir(parents=True)
    ror.write_text('305188 "Hoplite"\n', encoding="utf-8")
    assert asset_source.resource_string(305188, "en") == "Base Hoplite"
    _ror()
    assert asset_source.resource_string(305188, "en") == "Hoplite"
    assert asset_source.resource_string(1, "en") == "Only base"
    assert object_catalog.display_name(HOPLITE_SPEARMAN) == "Hoplite"


def _both_rows(table, id_: int) -> tuple[dict, dict]:
    """id_'s (AoE2 row, RoR row) from an object_catalog.dat_*() reader, read at test time."""
    aoe2 = table()[id_]
    _ror()
    ror = table()[id_]
    _aoe2()
    return aoe2, ror


def _write_base_string(install, key: int, text: str) -> None:
    base = install / asset_source.STRINGS_SUBPATH_TEMPLATE.format(lang="en")
    base.parent.mkdir(parents=True)
    base.write_text(f'{key} "{text}"\n', encoding="utf-8")


def test_without_ror_strings_a_ror_tech_keeps_its_own_name(ror_install):
    aoe2, ror = _both_rows(object_catalog.dat_techs, 101)
    assert aoe2["string_id"] != ror["string_id"]
    assert ror["code"] == "Tool Age"
    _write_base_string(ror_install, ror["string_id"], "Keep")
    _ror()
    assert object_catalog.tech_name(101) == "Tool Age"
    assert trigger_fields.resolve_reference("TechInfo", 101) == "Tool Age"


def test_without_ror_strings_a_row_both_games_share_takes_the_install_name(ror_install):
    aoe2, ror = _both_rows(object_catalog.dat_objects, 102)
    assert (aoe2["code"], aoe2["string_id"]) == (ror["code"], ror["string_id"])
    _write_base_string(ror_install, ror["string_id"], "Stone Mine")
    _ror()
    assert object_catalog.object_name(102) == "Stone Mine"


def test_without_ror_strings_a_reused_slot_keeps_its_ror_code(ror_install):
    aoe2, ror = _both_rows(object_catalog.dat_objects, 751)
    assert aoe2["string_id"] == ror["string_id"]
    assert aoe2["code"] != ror["code"]
    assert ror["code"] == "Clubman"
    _write_base_string(ror_install, ror["string_id"], "Eagle Scout")
    _ror()
    assert object_catalog.object_name(751) == "Clubman"


def test_without_ror_strings_a_ror_terrain_keeps_its_panel_label(ror_install):
    aoe2, ror = _both_rows(object_catalog.dat_terrains, 40)
    assert aoe2["string_id"] != ror["string_id"]
    _write_base_string(ror_install, ror["string_id"], "Temple Guard")
    _ror()
    label = terrain_catalog.display_name(terrain_palette.name_for_terrain_id(40))
    assert label != "Temple Guard"
    assert placement_rules._terrain_name(40) == label


def test_with_ror_strings_a_key_they_lack_still_reads_the_base_file(ror_install):
    _aoe2_row, ror_row = _both_rows(object_catalog.dat_techs, 101)
    _write_base_string(ror_install, ror_row["string_id"], "Base text")
    assert not game_dataset.has_own_strings("en", games.ROR)
    assert game_dataset.has_own_strings("en"), "the active game, AoE2: the install's base file"
    ror = ror_install / games.ROR.root_subpath / asset_source.ROR_STRINGS_SUBPATH_TEMPLATE.format(lang="en")
    ror.parent.mkdir(parents=True)
    ror.write_text('1 "Unrelated"\n', encoding="utf-8")
    asset_source.clear_install_caches()
    assert game_dataset.has_own_strings("en", games.ROR)
    _ror()
    assert object_catalog.tech_name(101) == "Base text"


def test_ror_icons_and_textures_come_from_the_ror_root_first(ror_install):
    for root in (ror_install, ror_install / games.ROR.root_subpath):
        folder = root / asset_source.UNIT_ICON_DIR
        folder.mkdir(parents=True)
        (folder / asset_source.UNIT_ICON_NAME.format(icon=16)).write_bytes(b"x")
    (ror_install / asset_source.UNIT_ICON_DIR / asset_source.UNIT_ICON_NAME.format(icon=31)).write_bytes(b"x")
    assert asset_source.unit_icon_path(16).parent == ror_install / asset_source.UNIT_ICON_DIR
    _ror()
    assert asset_source.unit_icon_path(16).parent == ror_install / games.ROR.root_subpath / asset_source.UNIT_ICON_DIR
    assert asset_source.unit_icon_path(31).parent == ror_install / asset_source.UNIT_ICON_DIR, "falls back to the install"


def _atlas(*textures: tuple[str, str]) -> str:
    entries = [{"RefName": ref, "FileName": name} for ref, name in textures]
    return json.dumps({"AtlasTextures": [{"AtlasDef": {"Name": "a", "Textures": entries}}]})


def test_ror_portraits_follow_its_icon_table(ror_install):
    """GH #67: RoR's icons.json renumbers portraits; an index names a RoR atlas file or an AoE2 one."""
    ror_ui = ror_install / games.ROR.root_subpath / asset_source.UI_DIR
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
    # samefile: on case-insensitive APFS the lowercase name as written already exists.
    assert asset_source.unit_icon_path(146).samefile(aoe2_ui / "textures/ingame/units/589_50730.DDS"), "file name case varies"
    assert asset_source.unit_icon_path(80, building=True) == aoe2_ui / "textures/ingame/buildings/052_aqueduct.dds"
    assert asset_source.unit_icon_path(99) is None, "an unknown RefName falls back to the index lookup"


# --- GH #363: a data mod's portraits ------------------------------------------------------

PID = "7656"
DESCAPE = Path(game_dataset.__file__).resolve().parent


def _ui_file(root: Path, rel: str, color: tuple[int, int, int, int] | None = None) -> Path:
    """root/widgetui/<rel>: a 4 px PNG under the .dds name when color is given (Pillow reads it), else a stub."""
    path = root / asset_source.UI_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if color is None:
        path.write_bytes(b"x")
    else:
        from PIL import Image

        Image.new("RGBA", (4, 4), color).save(path, format="PNG")
    return path


def _ui_json(root: Path, name: str, text: str) -> None:
    folder = root / asset_source.UI_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding="utf-8")


def _icon_table(root: Path, units: dict | None = None, buildings: dict | None = None) -> None:
    _ui_json(root, asset_source.ICON_TABLE_NAME, json.dumps({"Units": units or {}, "Buildings": buildings or {}}))


@pytest.fixture()
def portrait_mods(tmp_path, monkeypatch):
    """An install and a profile with a data mod in effect (disabled in game, as every data
    mod is; its tables are the AoE2 ones) and two art mods, High over Low; Mod Art on."""
    mods = tmp_path / "profile" / PID / "mods"
    found = SimpleNamespace(
        install=tmp_path / "install",
        data=mods / "subscribed" / "1_Data",
        high=mods / "subscribed" / "2_High",
        low=mods / "subscribed" / "3_Low",
    )
    for root in (found.install, found.data, found.high, found.low):
        root.mkdir(parents=True)
    status = {"Mods": [
        {"Enabled": False, "Path": "subscribed//1_Data", "Priority": 3},
        {"Enabled": True, "Path": "subscribed//2_High", "Priority": 1},
        {"Enabled": True, "Path": "subscribed//3_Low", "Priority": 2},
    ]}
    (mods / "mod-status.json").write_text(json.dumps(status))
    monkeypatch.setenv("AOE2DE_INSTALL_PATH", str(found.install))
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(tmp_path / "profile"))
    asset_source.clear_install_caches()
    mod_graphics.set_enabled(True)
    found.tables = game_dataset.DataModTables("key", DESCAPE, found.data, "Stand-in")
    game_dataset.set_data_mod(found.tables)
    return found


def test_portrait_layers_are_the_data_mod_then_the_drawn_art_mods(portrait_mods):
    m = portrait_mods
    assert asset_source.portrait_layers() == (m.data, m.high, m.low)
    mod_graphics.set_included({"subscribed//1_Data"})
    assert asset_source.portrait_layers() == (m.data, m.high, m.low), "the data mod is never listed twice"
    mod_graphics.set_excluded({"subscribed//2_High"})
    assert asset_source.portrait_layers() == (m.data, m.low)
    mod_graphics.set_enabled(False)
    assert asset_source.portrait_layers() == (m.data,)


def test_a_data_mods_table_and_materials_name_its_own_files(portrait_mods):
    m = portrait_mods
    unit = _ui_file(m.data, "textures/ingame/units/EmpUnitIcons_001.dds")
    building = _ui_file(m.data, "textures/ingame/buildings/EmpBuildingIcons_002.dds")
    _icon_table(m.data, {"1000": "EmpUnit1"}, {"118": "EmpBuilding2"})
    _ui_json(m.data, asset_source.MATERIALS_NAME, _atlas(
        ("EmpUnit1", "textures/ingame/units/EmpUnitIcons_001.dds"),
        ("EmpBuilding2", "textures/ingame/buildings/EmpBuildingIcons_002.dds"),
    ))
    assert asset_source.unit_icon_path(1000) == unit
    assert asset_source.unit_icon_path(118, building=True) == building


def test_an_unknown_refname_falls_to_the_index_lookup(portrait_mods):
    m = portrait_mods
    install_unit = _ui_file(m.install, "textures/ingame/units/005_50730.dds")
    _icon_table(m.data, {"005": "Nowhere"})
    _ui_json(m.data, asset_source.MATERIALS_NAME, _atlas(("Elsewhere", "textures/ingame/units/005_50730.dds")))
    assert asset_source.unit_icon_path(5) == install_unit


def test_an_index_the_mod_lacks_falls_to_the_install_portrait(portrait_mods):
    m = portrait_mods
    install_unit = _ui_file(m.install, "textures/ingame/units/017_50730.dds")
    install_building = _ui_file(m.install, "textures/ingame/buildings/007_castle.dds")
    _ui_file(m.data, "textures/ingame/units/mod.dds")
    _icon_table(m.data, {"1000": "Mod"})
    _ui_json(m.data, asset_source.MATERIALS_NAME, _atlas(("Mod", "textures/ingame/units/mod.dds")))
    assert asset_source.unit_icon_path(17) == install_unit
    assert asset_source.unit_icon_path(7, building=True) == install_building


def test_a_table_entry_resolves_through_the_installs_materials_and_files(portrait_mods):
    """GB's 921 shared portraits: the mod's table names an install RefName and an install file."""
    m = portrait_mods
    install_unit = _ui_file(m.install, "textures/ingame/units/294_50730.dds")
    _icon_table(m.data, {"016": "UnitIcons29450730"})
    _ui_json(m.install, asset_source.MATERIALS_NAME, _atlas(("UnitIcons29450730", "textures/ingame/units/294_50730.dds")))
    assert asset_source.unit_icon_path(16) == install_unit


def test_with_no_data_mod_or_under_ror_the_install_portrait_shows(portrait_mods):
    m = portrait_mods
    install_unit = _ui_file(m.install, "textures/ingame/units/017_50730.dds")
    mod_unit = _ui_file(m.data, "textures/ingame/units/mod.dds")
    _icon_table(m.data, {"017": "Mod"})
    _ui_json(m.data, asset_source.MATERIALS_NAME, _atlas(("Mod", "textures/ingame/units/mod.dds")))
    assert asset_source.unit_icon_path(17) == mod_unit
    game_dataset.set_data_mod(None)
    assert asset_source.portrait_layers() == ()
    assert asset_source.unit_icon_path(17) == install_unit
    game_dataset.set_data_mod(m.tables)
    _ror()
    assert asset_source.portrait_layers() == ()
    assert asset_source.unit_icon_path(17) == install_unit


def test_the_installs_own_icon_table_is_never_read(portrait_mods):
    """A real install ships widgetui/icons.json; AoE2 portraits still go by file name."""
    m = portrait_mods
    install_unit = _ui_file(m.install, "textures/ingame/units/017_50730.dds")
    _ui_file(m.install, "textures/ingame/units/elsewhere.dds")
    _icon_table(m.install, {"017": "Elsewhere"})
    _ui_json(m.install, asset_source.MATERIALS_NAME, _atlas(("Elsewhere", "textures/ingame/units/elsewhere.dds")))
    assert asset_source.unit_icon_path(17) == install_unit, "a data mod with no table"
    game_dataset.set_data_mod(None)
    assert asset_source.unit_icon_path(17) == install_unit, "no data mod"


def test_wrong_shaped_or_escaping_mod_tables_count_as_absent(portrait_mods):
    m = portrait_mods
    install_unit = _ui_file(m.install, "textures/ingame/units/017_50730.dds")
    outside = m.data.parent / "outside.dds"
    outside.write_bytes(b"x")
    _ui_file(m.data, "textures/ingame/units/002_ok.dds")
    # "²" passes str.isdigit() but not int(): skipped, never a ValueError.
    table = {"Units": {"017": ["x"], "018": "Up", "019": "Num", "²": "Ok"}, "Buildings": []}
    _ui_json(m.data, asset_source.ICON_TABLE_NAME, json.dumps(table))
    _ui_json(m.data, asset_source.MATERIALS_NAME, json.dumps({"AtlasTextures": [
        3, {"AtlasDef": []}, {"AtlasDef": {"Textures": [7, {"RefName": "Num", "FileName": 5}, {"RefName": ["x"], "FileName": "a"}]}},
        {"AtlasDef": {"Textures": [{"RefName": "Up", "FileName": "../../outside.dds"}]}},
        {"AtlasDef": {"Textures": [{"RefName": "Ok", "FileName": "textures/ingame/units/002_ok.dds"}]}},
    ]}))
    assert asset_source.unit_icon_path(17) == install_unit
    assert asset_source.unit_icon_path(18) is None, "a FileName never leaves the mod's widgetui"
    assert asset_source.unit_icon_path(19) is None
    assert asset_source.unit_icon_path(7, building=True) is None


def test_a_mods_widgetui_and_table_names_match_case_insensitively(portrait_mods):
    m = portrait_mods
    folder = m.data / "WidgetUI" / "Textures" / "InGame" / "Units"
    folder.mkdir(parents=True)
    (folder / "Mod.DDS").write_bytes(b"x")
    (m.data / "WidgetUI" / "Icons.json").write_text(json.dumps({"Units": {"1000": "Mod"}}))
    (m.data / "WidgetUI" / "Materials.JSON").write_text(_atlas(("Mod", "textures/ingame/units/mod.dds")))
    assert asset_source.unit_icon_path(1000).samefile(folder / "Mod.DDS")


def test_an_art_mod_counts_only_while_mod_art_draws_it(portrait_mods):
    m = portrait_mods
    high = _ui_file(m.high, "textures/ingame/units/017_50730.dds")
    low = _ui_file(m.low, "textures/ingame/units/017_50730.dds")
    install_unit = _ui_file(m.install, "textures/ingame/units/017_50730.dds")
    assert asset_source.unit_icon_path(17) == high
    mod_graphics.set_excluded({"subscribed//2_High"})
    assert asset_source.unit_icon_path(17) == low, "an opted-out mod is ignored"
    mod_graphics.set_enabled(False)
    assert asset_source.unit_icon_path(17) == install_unit


def test_the_data_mod_beats_an_art_mod(portrait_mods):
    """Its icons.json replaces the art mod's whole, and its loose files come first."""
    m = portrait_mods
    data_unit = _ui_file(m.data, "textures/ingame/units/data.dds")
    _ui_file(m.high, "textures/ingame/units/art.dds")
    _icon_table(m.data, {"1000": "Data"})
    _icon_table(m.high, {"1000": "Art", "1001": "Art"})
    _ui_json(m.high, asset_source.MATERIALS_NAME, _atlas(
        ("Data", "textures/ingame/units/data.dds"), ("Art", "textures/ingame/units/art.dds"),
    ))
    assert asset_source.unit_icon_path(1000) == data_unit
    assert asset_source.unit_icon_path(1001) is None, "the art mod's table is not read"
    data_loose = _ui_file(m.data, "textures/ingame/units/017_50730.dds")
    _ui_file(m.high, "textures/ingame/units/017_50730.dds")
    assert asset_source.unit_icon_path(17) == data_loose


def test_the_first_layers_materials_replace_later_ones(portrait_mods):
    m = portrait_mods
    first = _ui_file(m.high, "textures/ingame/units/first.dds")
    _ui_file(m.high, "textures/ingame/units/later.dds")
    _icon_table(m.high, {"1000": "First", "1001": "Later"})
    _ui_json(m.data, asset_source.MATERIALS_NAME, _atlas(("First", "textures/ingame/units/first.dds")))
    _ui_json(m.high, asset_source.MATERIALS_NAME, _atlas(("Later", "textures/ingame/units/later.dds")))
    _ui_json(m.install, asset_source.MATERIALS_NAME, _atlas(("Later", "textures/ingame/units/later.dds")))
    assert asset_source.unit_icon_path(1000) == first, "the file comes from the first folder that has it"
    assert asset_source.unit_icon_path(1001) is None
    more = _ui_file(m.low, "textures/ingame/units/more.dds")
    _ui_json(m.low, asset_source.ROR_MATERIALS_NAME, _atlas(("First", "textures/ingame/units/more.dds")))
    asset_source.clear_art_caches()
    assert asset_source.unit_icon_path(1000) == more, "morematerials beats materials"


def test_loose_portraits_in_a_mod_without_a_table_beat_the_installs(portrait_mods):
    m = portrait_mods
    _ui_file(m.install, "textures/ingame/units/017_50730.dds")
    _ui_file(m.install, "textures/ingame/buildings/007_castle.dds")
    unit = _ui_file(m.data, "textures/ingame/units/017_50730.DDS")
    building = _ui_file(m.data, "textures/ingame/buildings/007_fort.dds")
    assert asset_source.unit_icon_path(17).samefile(unit), "file name case varies"
    assert asset_source.unit_icon_path(7, building=True) == building


def test_a_mod_art_toggle_with_clear_art_caches_changes_the_cached_portrait(portrait_mods):
    m = portrait_mods
    _ui_file(m.high, "textures/ingame/units/017_50730.dds", (255, 0, 0, 255))
    _ui_file(m.install, "textures/ingame/units/017_50730.dds", (0, 0, 255, 255))
    assert tuple(asset_source.get_unit_icon(17)[0, 0]) == (255, 0, 0, 255)
    mod_graphics.set_enabled(False)
    assert tuple(asset_source.get_unit_icon(17)[0, 0]) == (255, 0, 0, 255), "cached until the viewer clears"
    asset_source.clear_art_caches()
    assert tuple(asset_source.get_unit_icon(17)[0, 0]) == (0, 0, 255, 255)


def test_ror_sld_lookup_prefers_the_ror_graphics(ror_install):
    base = ror_install / unit_sprites.GRAPHICS_SUBPATH
    ror = ror_install / games.ROR.root_subpath / unit_sprites.GRAPHICS_SUBPATH
    ror.mkdir(parents=True)
    (ror / "u_all_hoplite_idleA_x1.sld").write_bytes(b"")
    assert unit_sprites._sld_path(base, "u_all_hoplite_idleA_x1") == base / "u_all_hoplite_idleA_x1.sld"
    _ror()
    assert unit_sprites._sld_path(base, "u_all_hoplite_idleA_x1") == ror / "u_all_hoplite_idleA_x1.sld"
    assert unit_sprites._sld_path(base, "only_base") == base / "only_base.sld"


def test_ror_sld_bytes_reread_the_ror_file_after_eviction(ror_install):
    """GH #67: a decode whose walked bytes left the cache re-reads the file the walk read, not the install's."""
    base = ror_install / unit_sprites.GRAPHICS_SUBPATH
    ror = ror_install / games.ROR.root_subpath / unit_sprites.GRAPHICS_SUBPATH
    base.mkdir(parents=True)
    ror.mkdir(parents=True)
    (base / "u_all_bowman_idleA_x1.sld").write_bytes(b"aoe2")
    (ror / "u_all_bowman_idleA_x1.sld").write_bytes(b"ror")
    _ror()
    unit_sprites._sld_bytes_cache.clear()
    assert unit_sprites._sld_bytes(base, "u_all_bowman_idleA_x1") == b"ror"
