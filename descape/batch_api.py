"""v2.5's scriptable-batch-edit API: a small set of ergonomic helpers on top
of the same scenario_io/scenario_write/elevation_tools object model the GUI
already uses, for one-off Python scripts that do things like "raise elevation
under every Player 3 building by 1" or "recolor every DIRT_2 tile bordering
water to BEACH" -- the in-game editor has no scripting surface at all, so
this is the one capability category with no in-game equivalent to match.

Nothing here is a new capability the object model didn't already have --
AoE2ScenarioParser's managers already expose per-tile/per-unit mutation, and
scenario_write.write_scenario() already persists terrain edits made that way
(see the IMPORTANT note by the load/save re-exports below for the real limit
on that: unit edits don't persist yet). What's missing without this module is
the glue a script author would otherwise have to rebuild every time (which
unit_consts are buildings, adjacent-tile lookup, what counts as "water", the
correct elevation clamp, and the order-safe way to bulk-edit many units at
once) -- see each function's docstring for which upstream fact it's standing
in for.

See batch_scripts/ for runnable examples.

Known AoE2ScenarioParser gotcha for scripts that process many files in one
process (e.g. "apply this edit to every scenario in my mod folder"): calling
UnitManager.add_unit()/clone_unit() to create a *new* unit can raise
UnsupportedAttributeError on caption_string_id/caption_string if any
older-format (pre-1.54/1.55) scenario was loaded earlier in the same process
-- even when the *current* scenario's own version supports those fields
fine. Confirmed directly (two files loaded back-to-back in one interpreter,
not just read from source): AoE2ScenarioParser version-gates those two
fields via a class-level property swap the first time an unsupported version
is seen, and per the library's own TODO in
sections/retrievers/retriever_object_link.py, that swap is never reverted
for later scenarios in the same process. Same family of bug as the one
scenario_write.py's own docstring documents for commit()/write_to_file() --
just hitting Unit construction instead of scenario serialization.

Passing explicit caption_string_id=None, caption_string=None to
add_unit()/clone_unit() avoids the crash *at construction* (confirmed: the
swapped setter only raises for a non-None value), but the swap also poisons
the *getter* unconditionally -- confirmed separately: merely repr()'ing or
logging the resulting unit afterward still raises, since Unit.__repr__ reads
caption_string_id directly. So the explicit-None workaround only helps a
script that never reads those two fields back (including indirectly, e.g. no
debug print of the unit) for the rest of the process. A script that does
need to read them back, or that calls add_unit()/clone_unit() without
thinking about this ahead of time, should process each file in its own
subprocess instead. Not something this module works around itself -- that
would mean patching the vendored library, out of scope here.
"""

from __future__ import annotations

from AoE2ScenarioParser.datasets.buildings import BuildingInfo
from AoE2ScenarioParser.datasets.terrains import TerrainId
from AoE2ScenarioParser.objects.data_objects.terrain_tile import TerrainTile
from AoE2ScenarioParser.objects.data_objects.unit import Unit

from descape.elevation_tools import set_tile_elevation
from descape.iso_geometry import MAX_ELEVATION
from descape.scenario_io import LoadedScenario, load_map_and_units
from descape.scenario_write import write_scenario

# Re-exported under this module so a batch script only needs `from descape
# import batch_api` and never has to know which of scenario_io/scenario_write
# actually implements load/save.
#
# IMPORTANT: save() (scenario_write.write_scenario) only ever writes Map
# terrain -- Units and the trigger tail pass through byte-for-byte from the
# original file regardless of what's in memory. buildings_of()/tile_under()
# hand back real, mutable Unit objects, but editing one (player, x/y,
# rotation, ...) and calling save() silently drops that edit -- confirmed
# directly: mutating a unit's `player`, saving, and reloading shows the
# original player, not the edited one. Only tile.terrain_id/elevation/layer
# mutations persist. A future version might extend the write path to cover
# Units too, but nothing here does yet.
load = load_map_and_units
save = write_scenario

# Every unit_const that's a building, per AoE2ScenarioParser's own
# BuildingInfo dataset -- Unit itself carries no is-this-a-building flag,
# just the raw unit_const int, so this is the one place that fact lives.
_BUILDING_IDS: frozenset[int] = frozenset(b.ID for b in BuildingInfo)

_ORTHOGONAL_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_DIAGONAL_OFFSETS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def is_building(unit: Unit) -> bool:
    """True if unit's unit_const identifies a building, not a regular unit/hero/other."""
    return unit.unit_const in _BUILDING_IDS


def buildings_of(scenario: LoadedScenario, player: int) -> list[Unit]:
    """All of player's placed buildings (see is_building)."""
    return [u for u in scenario.unit_manager.get_player_units(player) if is_building(u)]


def tile_under(scenario: LoadedScenario, unit: Unit) -> TerrainTile | None:
    """The terrain tile unit currently stands on. None if unit.tile falls
    outside the map (get_tile_safe) -- a batch script shouldn't crash on a
    unit some other edit left out of bounds."""
    t = unit.tile
    return scenario.map_manager.get_tile_safe(t.x, t.y)


def neighbors(scenario: LoadedScenario, x: int, y: int, diagonal: bool = False) -> list[TerrainTile]:
    """The (up to) 4 orthogonally adjacent tiles, or 8 with diagonal=True.
    Tiles that fall off the map edge are silently omitted rather than
    raising -- MapManager itself has no neighbor-lookup helper at all."""
    offsets = _ORTHOGONAL_OFFSETS + _DIAGONAL_OFFSETS if diagonal else _ORTHOGONAL_OFFSETS
    mm = scenario.map_manager
    result = []
    for dx, dy in offsets:
        tile = mm.get_tile_safe(x + dx, y + dy)
        if tile is not None:
            result.append(tile)
    return result


def is_water(terrain_id: int) -> bool:
    """True for any water or shallows-family terrain (open water, all
    WATER_* variants, SHALLOWS/SWAMP_SHALLOWS/etc). Keyword match against
    TerrainId's own enum name -- the same approach descape/terrain_palette.py
    uses for color classification, since AoE2ScenarioParser has no semantic
    terrain-family dataset of its own. "SHALLOW" has to be checked
    separately from "WATER": confirmed directly against the live TerrainId
    enum that terrain names like SHALLOWS/SHALLOWS_AZURE/SWAMP_SHALLOWS
    contain "SHALLOW" but not "WATER" -- they're a distinct naming family,
    not a WATER_* variant, even though they're ship-navigable water for
    gameplay purposes same as any other water tile. Deliberately excludes
    ICE/ICE_NAVIGABLE/BEACH_ICE:
    despite being frozen water, AoE2 treats ice as walkable solid ground for
    land units, not water, so a "borders water" bulk edit shouldn't treat it
    as one."""
    name = TerrainId(terrain_id).name
    return "WATER" in name or "SHALLOW" in name


def raise_elevation_under(scenario: LoadedScenario, unit: Unit, amount: int = 1) -> bool:
    """Raises (or lowers, for a negative amount) the elevation of the tile
    `unit` stands on, propagating to neighbors the same way the Elevate tool
    does (elevation_tools.set_tile_elevation). Result is clamped to
    [0, MAX_ELEVATION] -- the same clamp viewer.py's Elevate tool applies;
    going negative crashes scenario_write.py's raw byte patch at save time,
    and going above MAX_ELEVATION breaks the Stepped-mode renderer's
    fixed-range canvas sizing. Returns False (no-op) if unit's tile is
    off-map.

    Reads the tile's CURRENT elevation at call time -- calling this in a
    loop over many units whose tiles are shared or adjacent will compound,
    since one call's neighbor propagation can shift a tile a previous call
    already touched before its own turn. Confirmed directly, not
    hypothetical: looping this over a real file's 156 same-player buildings
    left 30 of their tiles at +2 instead of +1. Use
    raise_elevation_under_many for a bulk edit across many units instead --
    it computes every tile's target from one pre-edit snapshot, so it's
    immune to processing order."""
    tile = tile_under(scenario, unit)
    if tile is None:
        return False
    new_elevation = max(0, min(MAX_ELEVATION, tile.elevation + amount))
    set_tile_elevation(scenario.map_manager, tile.x, tile.y, new_elevation)
    return True


def raise_elevation_under_many(scenario: LoadedScenario, units: list[Unit], amount: int = 1) -> int:
    """Like raise_elevation_under, but safe for many units at once when
    their tiles may be shared or adjacent -- which looping raise_elevation_under
    per unit is not (see its own docstring for the confirmed compounding
    bug this avoids). Every affected tile's target elevation is computed
    from a single pre-edit snapshot (deduping tiles shared by more than one
    unit first), then applied. Confirmed exact (zero deviation) across
    thousands of real adjacent building tiles in this repo's own example
    files: every unique tile's final elevation matched its pre-edit value +
    amount (clamped), regardless of processing order. Not a structural
    guarantee for arbitrary inputs, though -- it relies on the clamp
    compressing nearby differences rather than expanding them, and on the
    source terrain already being in-range; it's an empirically strong result
    on real scenario data, not a proof. Off-map units are silently skipped,
    same as raise_elevation_under. Returns the number of unique on-map tiles
    adjusted."""
    targets: dict[tuple[int, int], int] = {}
    for unit in units:
        tile = tile_under(scenario, unit)
        if tile is None:
            continue
        if tile.xy not in targets:
            targets[tile.xy] = max(0, min(MAX_ELEVATION, tile.elevation + amount))
    for (x, y), target_elevation in targets.items():
        set_tile_elevation(scenario.map_manager, x, y, target_elevation)
    return len(targets)


def set_terrain(tile: TerrainTile, terrain_id: int) -> None:
    """Sets tile's terrain_id and clears a stale `layer` (double-terrain
    blend) left over from whatever the tile used to be -- render.py doesn't
    draw `layer`, but the game does, so leaving it set after changing
    terrain_id would make this terrain a lie about what the game actually
    shows (same fix viewer.py's Terrain tool already applies per-click; see
    its own comment in ViewerWindow's mouse handler)."""
    tile.terrain_id = terrain_id
    tile.layer = -1
