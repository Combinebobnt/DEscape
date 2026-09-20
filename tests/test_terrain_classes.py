"""Terrain classification coverage -- Qt-free and install-free: the JSON is
committed, so nothing here needs a real AoE2DE install.

The point of this file is that the classification beats a name-based guess
on real cases, in BOTH directions. Those cases are named individually rather
than checked in bulk, because a bulk "every entry has a family" assertion
would still pass if a regeneration silently flipped BEACH_NON_NAVIGABLE to
beach -- which would turn a shoreline ships deliberately cannot dock at into
one they can.
"""

from __future__ import annotations

import pytest
from AoE2ScenarioParser.datasets.terrains import TerrainId

from descape import batch_api, terrain_classes
from descape.terrain_classes import (
    DEFAULT_BEACH_ID,
    auto_beach_for,
    beach_terrains,
    is_beach_family,
    is_water_family,
    terrain_climate,
    terrain_family,
    water_terrains,
)

_BEACH = 2
_BEACH_ICE = 37
_ICE_NAVIGABLE = 26
_ICE = 35
_ICE_SOFT = 127
_GRASS_1 = 0
_WATER_DEEP = 22
_SHALLOWS = 4
_SHALLOWS_AZURE = 59
_FOREST_MANGROVE = 55
_FOREST_REEDS_SHALLOWS = 90
_RICE_FARM = 63
_BEACH_NON_NAVIGABLE = (79, 80, 81, 82)

_FAMILIES = {
    "medium_water",
    "deep_water",
    "shallow_water",
    "shallows",
    "beach",
    "land",
    "obsolete",
}
_CLIMATES = {"temperate", "ice", "snow"}


def test_every_enabled_terrain_id_is_classified():
    classes = terrain_classes._classes()
    assert len(classes) == 131
    # Every TerrainId the app can offer has a classification, so no paint
    # path ever has to handle a gap.
    assert {t.value for t in TerrainId} <= set(classes)


def test_every_entry_has_a_known_family_and_climate():
    for terrain_id, entry in terrain_classes._classes().items():
        assert entry["family"] in _FAMILIES, terrain_id
        assert entry["climate"] in _CLIMATES, terrain_id


def test_an_unknown_id_reports_unknown_rather_than_raising():
    """Called once per tile inside a paint loop -- a raise mid-stroke would
    leave EditHistory's snapshot uncleared and wedge every later edit."""
    assert terrain_family(9999) == terrain_classes.UNKNOWN
    assert terrain_climate(-1) == terrain_classes.UNKNOWN
    assert not is_water_family(9999)
    assert not is_beach_family(9999)


@pytest.mark.parametrize("terrain_id", _BEACH_NON_NAVIGABLE)
def test_the_non_navigable_beaches_are_land_not_beach(terrain_id: int):
    """A name match on "BEACH" gets all four of these wrong. They are the
    reason this table exists at all."""
    assert terrain_family(terrain_id) == "land"
    assert not is_beach_family(terrain_id)
    assert not is_water_family(terrain_id)


@pytest.mark.parametrize(
    "terrain_id", [_FOREST_MANGROVE, _FOREST_REEDS_SHALLOWS, _RICE_FARM, 64, 65, 66, 67]
)
def test_the_wet_terrains_with_dry_names_are_shallows(terrain_id: int):
    """The other direction: no WATER/SHALLOW substring, but the game counts
    them as water."""
    assert terrain_family(terrain_id) == "shallows"
    assert is_water_family(terrain_id)


def test_shallows_count_as_water():
    """Pinned in both directions because both readings are defensible and
    the difference is user-visible: excluding shallows would leave SHALLOWS
    and SHALLOWS_AZURE, the two most natural AoE2 shoreline waters, with no
    auto-beach at all."""
    assert is_water_family(_SHALLOWS)
    assert is_water_family(_SHALLOWS_AZURE)
    assert not is_water_family(_GRASS_1)
    assert not is_water_family(_BEACH)
    for terrain_id in _BEACH_NON_NAVIGABLE:
        assert not is_water_family(terrain_id)


def test_the_beach_family_is_eight_temperate_plus_one_ice():
    beaches = beach_terrains()
    assert set(beaches) == {2, 37, 51, 52, 53, 91, 107, 108, 109}
    ice = [t for t in beaches if terrain_climate(t) == "ice"]
    assert ice == [_BEACH_ICE]
    # No snow beach exists at all -- the documented reason auto_beach_for's
    # step 3 is a hardcoded default rather than a derivation.
    assert not [t for t in beaches if terrain_climate(t) == "snow"]


def test_water_terrains_covers_every_water_family():
    waters = water_terrains()
    assert _SHALLOWS in waters and _WATER_DEEP in waters
    assert _ICE_NAVIGABLE in waters
    assert all(is_water_family(t) for t in waters)


# -- the auto rule -----------------------------------------------------------


def test_auto_beach_keys_on_the_land_tile_first():
    """Ice land is the case with exactly one beach in its climate."""
    assert auto_beach_for(_WATER_DEEP, _ICE) == _BEACH_ICE
    assert auto_beach_for(_WATER_DEEP, _ICE_SOFT) == _BEACH_ICE


def test_auto_beach_falls_back_to_the_waters_climate():
    """Snow land has no beach of its own, so ICE_NAVIGABLE is what decides
    -- the case a land-only rule would miss."""
    snow_land = next(
        t.value
        for t in TerrainId
        if terrain_climate(t.value) == "snow" and terrain_family(t.value) == "land"
    )
    assert auto_beach_for(_ICE_NAVIGABLE, snow_land) == _BEACH_ICE


def test_auto_beach_defaults_to_plain_beach_for_temperate():
    assert auto_beach_for(_WATER_DEEP, _GRASS_1) == _BEACH
    assert auto_beach_for(_WATER_DEEP) == DEFAULT_BEACH_ID


def test_snow_land_against_ordinary_water_gets_a_sand_beach():
    """The documented gap, asserted so it stays a known cost rather than
    becoming a surprise: the .dat carries no snow beach."""
    snow_land = next(
        t.value
        for t in TerrainId
        if terrain_climate(t.value) == "snow" and terrain_family(t.value) == "land"
    )
    assert auto_beach_for(_WATER_DEEP, snow_land) == _BEACH


# -- divergence from batch_api ------------------------------------------------


def test_the_divergence_from_batch_api_is_pinned():
    """batch_api.is_water() is a name-keyword classifier with its own pinned
    assertions in tools/verify_batch_api.py and tests/migration_manifest.py,
    and its documented intent deliberately excludes ice. Unifying the two
    would be a behaviour change to a published API. This test is what makes
    that divergence a recorded decision rather than a bug someone "fixes"."""
    disagree = {
        t.value
        for t in TerrainId
        if batch_api.is_water(t.value) != is_water_family(t.value)
    }
    assert disagree == {26, 28, 55, 63, 64, 65, 66, 67}
    # 26 ICE_NAVIGABLE: batch_api excludes ice on purpose (walkable ground
    # for land units), the .dat calls it shallows.
    # 28 WATER_2D_BRIDGE: batch_api's name match says water, the .dat says
    # land -- correctly, it is a bridge deck you walk on. The one case where
    # the .dat is the safer answer for THIS feature: painting a bridge tile
    # must not offer an auto-beach.
    # 55 FOREST_MANGROVE and 63-67 RICE_FARM*: no water word in the name,
    # but the game counts them as shallows.
    assert not is_water_family(28)
    assert batch_api.is_water(28)
    # FOREST_REEDS_SHALLOWS is NOT in the set: its name carries "SHALLOW",
    # so the keyword classifier happens to agree with the table here.
    assert batch_api.is_water(_FOREST_REEDS_SHALLOWS)
    assert is_water_family(_FOREST_REEDS_SHALLOWS)
