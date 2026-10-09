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

from descape import object_catalog
from descape.terrain_palette import is_dat_only_terrain, name_for_terrain_id, terrain_ids

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


def _entry(terrain_id: int, dat_terrains: dict[int, dict]) -> TerrainEntry:
    name = name_for_terrain_id(terrain_id)
    category = _category_for(name)
    if is_dat_only_terrain(terrain_id):
        hidden = bool(dat_terrains.get(terrain_id, {}).get("hidden"))
    else:
        hidden = category == CATEGORY_UNUSED
    return TerrainEntry(id=terrain_id, name=name, category=category, hidden=hidden)


@lru_cache(maxsize=1)
def terrains() -> tuple[TerrainEntry, ...]:
    """Every terrain_palette.terrain_ids() id as a catalog row, id order:
    TerrainId plus the .dat-only ids (131-133). Built from
    name_for_terrain_id rather than re-deriving a second id->name map. A
    dat-only id is hidden when the .dat hides it from the editor; an enum id
    by its prefix, as before."""
    dat_terrains = object_catalog.dat_terrains()
    return tuple(_entry(terrain_id, dat_terrains) for terrain_id in terrain_ids())
