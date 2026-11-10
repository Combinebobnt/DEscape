"""Terrain ids grouped into categories for the visual browser. Pure data,
no Qt, so it stays testable in the default (install-free) tier -- the same
split object_catalog.py has from constant_picker.py.

The category table is prefix-matched against the enum's own names, in
order. Order is load-bearing: OBSOLETE_ and MODDABLE_ must match before the
bare prefixes they contain (OBSOLETE_UNDERBRUSH_JUNGLE, MODDABLE_GRASS_1),
or those 18 entries would land under Forest and Grass instead of being
filtered out of the default view.

A data mod's own terrains (GH #271: new ids, or ids it changed) are placed by
_mod_category() instead, the game editor's own rule first.
"""

from __future__ import annotations

from dataclasses import dataclass

from descape import game_dataset, object_catalog, terrain_classes, terrain_units
from descape.terrain_palette import is_dat_only_terrain, name_for_terrain_id, terrain_ids

CATEGORY_UNUSED = "Unused"
CATEGORY_MODDABLE = "Moddable"
# Settings > Mods' one group for every data mod terrain (GH #271).
CATEGORY_MOD_GROUP = "Mod terrains"
# A data mod's own terrain whose name no prefix matches (GH #262).
CATEGORY_OTHER = "Other"

_HIDDEN_CATEGORIES = frozenset({CATEGORY_UNUSED, CATEGORY_MODDABLE})

# (prefix, category). Matched in order, first hit wins.
_PREFIX_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("MODDABLE_", CATEGORY_MODDABLE),
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
    # A data mod's own terrain: an id it adds or changes (GH #271).
    mod: bool = False


def _prefix_category(name: str, *, junk: bool = True) -> str:
    """The first _PREFIX_CATEGORIES hit for name, or ""; junk=False skips the Unused and Moddable rows."""
    for prefix, category in _PREFIX_CATEGORIES:
        if (junk or category not in _HIDDEN_CATEGORIES) and name.startswith(prefix):
            return category
    return ""


def _category_for(name: str) -> str:
    category = _prefix_category(name)
    if category:
        return category
    # A data mod names its own .dat-only terrains (Galactic Battlegrounds' SW_TERR_METAL1).
    if game_dataset.data_mod() is not None:
        return CATEGORY_OTHER
    # Deliberately no "Other" bucket on the game's own data: a new enum member
    # falling through should fail a test, not quietly appear in a catch-all nobody reads.
    raise KeyError(f"No terrain category for {name!r} -- add a _PREFIX_CATEGORIES row")


def _mod_category(terrain_id: int, name: str) -> str:
    """A data mod terrain's category (GH #271), from the mod's tables: Forest when
    its terrain units are trees (the game editor's Forest filter), Water or Beach by
    its family, then the prefix table on the name and on the name minus its first
    word (SW_FARM -> FARM), else Other."""
    if terrain_units.is_tree_terrain(terrain_id):
        return "Forest"
    if terrain_classes.is_water_family(terrain_id):
        return "Water"
    if terrain_classes.is_beach_family(terrain_id):
        return "Beach"
    for candidate in (name, name.partition("_")[2]):
        category = _prefix_category(candidate, junk=False)
        if category:
            return category
    return CATEGORY_OTHER


def display_name(name: str) -> str:
    """The enum name as a label: GRASS_FLOWERS_2 -> "Grass Flowers 2".
    Display only -- nothing reads a terrain back by this string, and
    terrain_palette.name_for_terrain_id stays the source of the raw name."""
    return name.replace("_", " ").title()


def _entry(terrain_id: int, dat_terrains: dict[int, dict], mod_ids: frozenset[int]) -> TerrainEntry:
    name = name_for_terrain_id(terrain_id)
    mod = terrain_id in mod_ids
    category = _mod_category(terrain_id, name) if mod else _category_for(name)
    if mod or is_dat_only_terrain(terrain_id):
        hidden = bool(dat_terrains.get(terrain_id, {}).get("hidden"))
    else:
        hidden = category in _HIDDEN_CATEGORIES
    return TerrainEntry(id=terrain_id, name=name, category=category, hidden=hidden, mod=mod)


@game_dataset.cache(maxsize=1)
def terrains() -> tuple[TerrainEntry, ...]:
    """Every terrain_palette.terrain_ids() id as a catalog row, id order:
    TerrainId plus the .dat-only ids (131-133). Built from
    name_for_terrain_id rather than re-deriving a second id->name map. A
    dat-only id or a data mod terrain is hidden when the .dat hides it from
    the editor; any other enum id by its prefix, as before."""
    dat_terrains = object_catalog.dat_terrains()
    mod_ids = object_catalog.data_mod_terrain_ids()
    return tuple(_entry(terrain_id, dat_terrains, mod_ids) for terrain_id in terrain_ids())
