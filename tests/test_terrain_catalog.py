"""Terrain catalog coverage -- Qt-free and install-free, so it runs in the
default tier the same way tests/test_brush.py does.

The load-bearing assertion is that every TerrainId lands in exactly one
category with nothing falling through: _category_for() raises rather than
offering an "Other" bucket, so a new enum member fails here loudly instead
of appearing in a catch-all nobody reads.
"""

from __future__ import annotations

import pytest
from AoE2ScenarioParser.datasets.terrains import TerrainId

from descape import terrain_catalog, terrain_palette
from descape.terrain_catalog import CATEGORY_UNUSED, display_name, terrains

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


def test_hidden_is_exactly_the_unused_category():
    assert {e.name for e in terrains() if e.hidden} == {
        e.name for e in terrains() if e.category == CATEGORY_UNUSED
    }


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
        CATEGORY_UNUSED: 22,
    }
    assert sum(counts.values()) == len(_ALL_IDS)


def test_the_junk_prefixes_win_over_the_bare_ones_they_contain():
    """_PREFIX_CATEGORIES is ordered for exactly this: without it,
    OBSOLETE_UNDERBRUSH_JUNGLE would read as Forest and MODDABLE_GRASS_1 as
    Grass, and 18 junk entries would show by default."""
    by_name = {e.name: e for e in terrains()}
    for name, entry in by_name.items():
        if name.startswith(("MODDABLE_", "OBSOLETE_")):
            assert entry.category == CATEGORY_UNUSED, name
    # The two that actually collide, named so the case can't silently stop
    # testing anything if the enum drops them.
    assert "OBSOLETE_UNDERBRUSH_JUNGLE" in by_name
    assert "MODDABLE_GRASS_1" in by_name


def test_an_unmatched_name_raises_rather_than_bucketing():
    with pytest.raises(KeyError):
        terrain_catalog._category_for("SOMETHING_NEW_AND_UNMAPPED")


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
