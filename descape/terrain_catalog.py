"""Terrain ids grouped into categories for the visual browser. Pure data,
no Qt, so it stays testable in the default (install-free) tier -- the same
split object_catalog.py has from constant_picker.py.

The category table is prefix-matched against the enum's own names, in
order. Order is load-bearing: OBSOLETE_ and MODDABLE_ must match before the
bare prefixes they contain (OBSOLETE_UNDERBRUSH_JUNGLE, MODDABLE_GRASS_1),
or those 18 entries would land under Forest and Grass instead of being
filtered out of the default view.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from AoE2ScenarioParser.datasets.terrains import TerrainId

from descape.terrain_palette import name_for_terrain_id

CATEGORY_UNUSED = "Unused / moddable"

# (prefix, category). Matched in order, first hit wins.
_PREFIX_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("MODDABLE_", CATEGORY_UNUSED),
    ("OBSOLETE_", CATEGORY_UNUSED),
    ("BLACK", CATEGORY_UNUSED),
    ("CORRUPTION", CATEGORY_UNUSED),
    ("RESERVED", CATEGORY_UNUSED),
    ("BEACH", "Beach"),
    ("DESERT", "Desert"),
    ("DIRT", "Dirt"),
    ("FARM", "Farms"),
    ("PASTURE", "Farms"),
    ("RICE", "Farms"),
    ("FOREST", "Forest"),
    ("UNDERBRUSH", "Forest"),
    ("GRASS", "Grass"),
    ("GRAVEL", "Road & Rock"),
    ("ROAD", "Road & Rock"),
    ("ROCK", "Road & Rock"),
    ("ICE", "Snow & Ice"),
    ("SNOW", "Snow & Ice"),
    ("SHALLOWS", "Water"),
    ("SWAMP", "Water"),
    ("WATER", "Water"),
)


@dataclass(frozen=True)
class TerrainEntry:
    """One row of the browser. `hidden` entries (the MODDABLE_/OBSOLETE_
    junk, plus BLACK/CORRUPTION/RESERVED) stay out of the tree until the
    "show unused" box is ticked -- unlike CatalogBrowseDialog's own hidden
    flag, this one has a real meaning from day one, which is why the
    checkbox is offered at all."""

    id: int
    name: str
    category: str
    hidden: bool


def _category_for(name: str) -> str:
    for prefix, category in _PREFIX_CATEGORIES:
        if name.startswith(prefix):
            return category
    # Deliberately not an "Other" bucket: a new enum member falling through
    # should fail a test, not quietly appear in a catch-all nobody reads.
    raise KeyError(f"No terrain category for {name!r} -- add a _PREFIX_CATEGORIES row")


def display_name(name: str) -> str:
    """The enum name as a label: GRASS_FLOWERS_2 -> "Grass Flowers 2".
    Display only -- nothing reads a terrain back by this string, and
    terrain_palette.name_for_terrain_id stays the source of the raw name."""
    return name.replace("_", " ").title()


@lru_cache(maxsize=1)
def terrains() -> tuple[TerrainEntry, ...]:
    """Every TerrainId as a catalog row, id order. Built from
    name_for_terrain_id rather than re-deriving a second id->name map."""
    return tuple(
        TerrainEntry(
            id=terrain.value,
            name=name_for_terrain_id(terrain.value),
            category=_category_for(terrain.name),
            hidden=_category_for(terrain.name) == CATEGORY_UNUSED,
        )
        for terrain in sorted(TerrainId, key=lambda t: t.value)
    )
