"""Terrain catalog coverage -- Qt-free and install-free, so it runs in the
default tier the same way tests/test_brush.py does.

The load-bearing assertion is that every TerrainId lands in exactly one
category with nothing falling through: _category_for() raises rather than
offering an "Other" bucket, so a new enum member fails here loudly instead
of appearing in a catch-all nobody reads. Only a data mod's own terrains
(GH #262) go under "Other", and those are placed by their own rule (GH #271).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from AoE2ScenarioParser.datasets.terrains import TerrainId

from descape import dat_derive, game_dataset, object_catalog, terrain_catalog, terrain_palette
from descape.terrain_catalog import CATEGORY_MODDABLE, CATEGORY_OTHER, CATEGORY_UNUSED, display_name, terrains
from testkit import data_mod_fixture
from testkit.data_mod_fixture import MOON_GRASS, SW_FARM, SW_JUNGLE, SW_LAKE, SW_METAL, SW_SANDBANK

_HIDDEN_PREFIXES = ("MODDABLE_", "OBSOLETE_", "BLACK", "CORRUPTION", "RESERVED")


# The .dat-only terrains (the Sept 2026 patch), named from object_catalog.json.
_DAT_ONLY = {131: "FOREST_SPRUCE", 132: "FOREST_SPRUCE_SNOW", 133: "FOREST_OAK_GREEN"}
_ALL_IDS = {t.value for t in TerrainId} | set(_DAT_ONLY)


def test_every_terrain_id_has_exactly_one_entry():
    entries = terrains()
    assert len(entries) == len(_ALL_IDS)
    assert len({e.id for e in entries}) == len(entries)
    assert {e.id for e in entries} == _ALL_IDS
    assert set(terrain_palette.terrain_ids()) == _ALL_IDS


def test_the_dat_only_terrains_are_visible_forest():
    by_id = {e.id: e for e in terrains()}
    for terrain_id, name in _DAT_ONLY.items():
        assert terrain_id not in {t.value for t in TerrainId}, "only tests anything for a dat-only id"
        assert by_id[terrain_id].name == name
        assert by_id[terrain_id].category == "Forest"
        assert by_id[terrain_id].hidden is False
        assert terrain_palette.name_for_terrain_id(terrain_id) == name
        # The no-install colour fallback is keyed by the same names, so it covers them too.
        assert terrain_palette._TERRAIN_COLOR_CACHE[terrain_id] == terrain_palette._color_for_name("FOREST")


def test_enum_style_name():
    assert terrain_palette.enum_style_name("Forest, Spruce Snow") == "FOREST_SPRUCE_SNOW"
    assert terrain_palette.enum_style_name("  OBSOLETE (Road, Gravel)") == "OBSOLETE_ROAD_GRAVEL"


def test_no_terrain_falls_through_to_a_catch_all():
    """_category_for raises on an unmatched name, so this is really "does
    building the catalog succeed" -- stated as its own case because that is
    the invariant a new enum member would break."""
    assert all(e.category for e in terrains())


def test_the_hidden_set_is_exactly_the_junk_entries():
    hidden = {e.name for e in terrains() if e.hidden}
    expected = {t.name for t in TerrainId if t.name.startswith(_HIDDEN_PREFIXES)}
    assert hidden == expected
    # 9 MODDABLE_ + 9 OBSOLETE_ + BLACK/BLACK_WALKABLE + CORRUPTION + RESERVED.
    assert len(hidden) == 22


def test_hidden_is_exactly_the_unused_and_moddable_categories():
    assert {e.name for e in terrains() if e.hidden} == {
        e.name for e in terrains() if e.category in (CATEGORY_UNUSED, CATEGORY_MODDABLE)
    }
    assert {e.name for e in terrains() if e.category == CATEGORY_MODDABLE} == {
        t.name for t in TerrainId if t.name.startswith("MODDABLE_")
    }
    assert not any(e.mod for e in terrains())


def test_category_counts():
    counts = {}
    for entry in terrains():
        counts[entry.category] = counts.get(entry.category, 0) + 1
    assert counts == {
        "Beach": 12,
        "Desert": 3,
        "Dirt": 6,
        "Farms": 15,  # FARM 5 + PASTURE 5 + RICE 5
        "Forest": 31,  # FOREST 24 + UNDERBRUSH 4 + .dat-only 131-133
        "Grass": 10,
        "Road & Rock": 7,  # GRAVEL 2 + ROAD 4 + ROCK 1
        "Snow & Ice": 10,  # ICE 3 + SNOW 7
        "Water": 18,  # SHALLOWS 4 + SWAMP 2 + WATER 12
        CATEGORY_UNUSED: 13,  # OBSOLETE_ 9 + BLACK 2 + CORRUPTION + RESERVED
        CATEGORY_MODDABLE: 9,
    }
    assert sum(counts.values()) == len(_ALL_IDS)


def test_the_junk_prefixes_win_over_the_bare_ones_they_contain():
    """_PREFIX_CATEGORIES is ordered for exactly this: without it,
    OBSOLETE_UNDERBRUSH_JUNGLE would read as Forest and MODDABLE_GRASS_1 as
    Grass, and 18 junk entries would show by default."""
    by_name = {e.name: e for e in terrains()}
    for name, entry in by_name.items():
        if name.startswith(("MODDABLE_", "OBSOLETE_")):
            assert entry.category == (CATEGORY_MODDABLE if name.startswith("MODDABLE_") else CATEGORY_UNUSED), name
    # The two that actually collide, named so the case can't silently stop
    # testing anything if the enum drops them.
    assert "OBSOLETE_UNDERBRUSH_JUNGLE" in by_name
    assert "MODDABLE_GRASS_1" in by_name


def test_an_unmatched_name_raises_rather_than_bucketing():
    with pytest.raises(KeyError):
        terrain_catalog._category_for("SOMETHING_NEW_AND_UNMAPPED")


def test_a_data_mods_own_terrain_goes_under_other(tmp_path):
    """GH #262: Galactic Battlegrounds' .dat names terrain SW_TERR_METAL1, and
    File > New Map, which reads this catalog, raised on it."""
    descape_dir = Path(game_dataset.__file__).resolve().parent
    folder = tmp_path / "tables"
    folder.mkdir()
    for name in dat_derive.TABLE_NAMES:
        shutil.copy(descape_dir / name, folder / name)
    catalog = json.loads((folder / "object_catalog.json").read_text())
    catalog["terrains"]["140"] = {"code": "SW_TERR_METAL1", "hidden": False, "string_id": -1}
    (folder / "object_catalog.json").write_text(json.dumps(catalog))
    game_dataset.set_data_mod(game_dataset.DataModTables("gbg", folder, tmp_path / "mod", "GBG", frozenset()))
    by_id = {e.id: e for e in terrains()}
    assert (by_id[140].name, by_id[140].category, by_id[140].hidden) == ("SW_TERR_METAL1", CATEGORY_OTHER, False)
    assert by_id[0].category == "Grass", "the game's own names keep their categories"
    game_dataset.set_data_mod(None)
    assert 140 not in {e.id for e in terrains()}
    with pytest.raises(KeyError):
        terrain_catalog._category_for("SW_TERR_METAL1")


@pytest.fixture
def gb_like_mod(tmp_path):
    mod = data_mod_fixture.mod_tables(tmp_path)
    data_mod_fixture.add_terrains(mod)
    return mod


def test_a_data_mods_new_terrains_are_placed_by_the_game_editors_rule(gb_like_mod):
    """GH #271: tree terrain units -> Forest, then the water and beach families,
    then the prefix table on the name and on the name minus its first word."""
    game_dataset.set_data_mod(gb_like_mod)
    by_id = {e.id: e for e in terrains()}
    assert {terrain_id: by_id[terrain_id].category for terrain_id in (SW_FARM, SW_JUNGLE, SW_LAKE, SW_SANDBANK, SW_METAL)} == {
        SW_FARM: "Farms",
        SW_JUNGLE: "Forest",
        SW_LAKE: "Water",
        SW_SANDBANK: "Beach",
        SW_METAL: CATEGORY_OTHER,
    }
    assert by_id[SW_FARM].name == "SW_FARM", "a new id keeps its code-based name"
    assert all(by_id[terrain_id].mod and not by_id[terrain_id].hidden for terrain_id in (SW_FARM, SW_JUNGLE, SW_METAL))
    assert not by_id[0].mod and by_id[0].category == "Grass", "an untouched AoE2 row keeps the stock rule"


def test_a_changed_id_shows_the_mods_name_and_hidden_flag(gb_like_mod):
    game_dataset.set_data_mod(gb_like_mod)
    assert MOON_GRASS in object_catalog.data_mod_terrain_ids()
    entry = next(e for e in terrains() if e.id == MOON_GRASS)
    assert (entry.name, entry.category, entry.hidden, entry.mod) == ("MOON_GRASS", "Grass", False, True)
    assert terrain_palette.name_for_terrain_id(MOON_GRASS) == "MOON_GRASS"
    game_dataset.set_data_mod(None)
    entry = next(e for e in terrains() if e.id == MOON_GRASS)
    assert (entry.name, entry.category, entry.hidden, entry.mod) == ("MODDABLE_GRASS_1", CATEGORY_MODDABLE, True, False)


@pytest.mark.usefixtures("fake_install")
def test_a_string_the_mod_redefines_renames_an_unchanged_row(gb_like_mod):
    """The mod's strings over the install's: resource_string() reads none without an install."""
    road = TerrainId.ROAD.value  # no terrain units, so the name decides
    road_string = object_catalog.dat_terrains()[road]["string_id"]
    folder = gb_like_mod.root / "resources" / "en" / "strings" / "key-value"
    folder.mkdir(parents=True)
    (folder / "key-value-modded-strings-utf8.txt").write_text(f'{road_string} "Lava Field"\n', encoding="utf-8")
    game_dataset.set_data_mod(gb_like_mod)
    entry = next(e for e in terrains() if e.id == road)
    assert (entry.name, entry.category, entry.mod) == ("LAVA_FIELD", CATEGORY_OTHER, True)
    assert 1 not in object_catalog.data_mod_terrain_ids(), "a row whose string the mod leaves alone is not the mod's"


@pytest.mark.usefixtures("fake_install")
def test_a_full_strings_copy_renames_only_what_it_changes(gb_like_mod):
    """A mod shipping the whole key-value-strings-utf8.txt repeats every stock string: those stay stock."""
    from descape import asset_source

    road = TerrainId.ROAD.value
    road_string = object_catalog.dat_terrains()[road]["string_id"]
    install_file = asset_source.get_install_path() / asset_source.STRINGS_SUBPATH_TEMPLATE.format(lang="en")
    install_file.parent.mkdir(parents=True, exist_ok=True)
    with install_file.open("a", encoding="utf-8") as f:
        f.write(f'\n{road_string} "Road"\n')
    folder = gb_like_mod.root / "resources" / "en" / "strings" / "key-value"
    folder.mkdir(parents=True)
    (folder / "key-value-strings-utf8.txt").write_text(f'{road_string} "Road"\n', encoding="utf-8")
    game_dataset.set_data_mod(gb_like_mod)
    assert road not in object_catalog.data_mod_terrain_ids()
    assert not next(e for e in terrains() if e.id == road).mod
    for full, modded, counted in (('"Lava"', None, True), ('"Lava"', '"Road"', False)):
        (folder / "key-value-strings-utf8.txt").write_text(f"{road_string} {full}\n", encoding="utf-8")
        if modded:
            (folder / "key-value-modded-strings-utf8.txt").write_text(f"{road_string} {modded}\n", encoding="utf-8")
        asset_source.clear_install_caches()
        object_catalog.data_mod_terrain_ids.cache_clear()
        assert (road in object_catalog.data_mod_terrain_ids()) is counted, (full, modded)


@pytest.mark.usefixtures("fake_install")
def test_a_non_english_string_keeps_its_letters(gb_like_mod):
    road = TerrainId.ROAD.value
    folder = gb_like_mod.root / "resources" / "en" / "strings" / "key-value"
    folder.mkdir(parents=True)
    road_string = object_catalog.dat_terrains()[road]["string_id"]
    (folder / "key-value-modded-strings-utf8.txt").write_text(f'{road_string} "Forêt lunaire"\n', encoding="utf-8")
    game_dataset.set_data_mod(gb_like_mod)
    entry = next(e for e in terrains() if e.id == road)
    assert (entry.name, display_name(entry.name)) == ("FORÊT_LUNAIRE", "Forêt Lunaire")


def test_data_mod_terrain_ids_is_empty_without_a_mod_and_under_return_of_rome(gb_like_mod):
    assert object_catalog.data_mod_terrain_ids() == frozenset()
    game_dataset.set_data_mod(gb_like_mod)
    assert {SW_FARM, SW_METAL, MOON_GRASS} <= object_catalog.data_mod_terrain_ids()
    game_dataset.set_active(game_dataset.ROR)
    assert object_catalog.data_mod_terrain_ids() == frozenset()
    assert not any(e.mod for e in terrains())


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("GRASS_1", "Grass 1"),
        ("GRASS_FLOWERS_2", "Grass Flowers 2"),
        ("BEACH_NON_NAVIGABLE_WET_GRAVEL", "Beach Non Navigable Wet Gravel"),
    ],
)
def test_display_name_is_readable(raw: str, shown: str):
    assert display_name(raw) == shown
