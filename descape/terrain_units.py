"""Auto-placing GAIA trees/eye-candy when Draw/Paint Can paint terrain --
mirrors AoE2:DE's in-game editor's Eye Candy option (docs/
INGAME_EDITOR_REFERENCE.md:183).

Placement model, measured against the .dat's own terrain_unit_map.json table
and the examples/ corpus: per tile, walk the terrain's unit list in order,
rolling `rng.random() * 1000 < density` for each and placing the first that
succeeds (at most one unit per tile). This is the only reading under which a
terrain with a non-monotonic density list (e.g. FOREST_DRY_SOUTH_AMERICAN)
makes every one of its species reachable, and it matches every
density-1000 forest's 100% corpus occupancy.

Leaf module: reads its own JSON assets directly, stdlib only (plus
unit_rotation, itself Qt-free/stdlib-only). No Qt, no descape.settings, no
genieutils. Deliberately not terrain_palette: its TREE_UNIT_IDS is scoped to
trees for the "Show Trees" filter and the minimap-color special-case, and
using it here as "every const this feature can place" would leave eye-candy
doodads (grass tufts, jungle underbrush) un-removed on repaint once Eye
candy is in scope -- this module builds its own TREE_CONSTS/DOODAD_CONSTS
from tree_unit_ids.json instead.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from descape import unit_rotation

_TERRAIN_UNIT_MAP_PATH = Path(__file__).resolve().parent / "terrain_unit_map.json"
_TREE_UNIT_IDS_PATH = Path(__file__).resolve().parent / "tree_unit_ids.json"

# terrain_id -> [{"id", "density", "centering"}, ...], in roll order.
_TERRAIN_UNITS: dict[int, list[dict]] = {
    int(tid): units
    for tid, units in json.loads(_TERRAIN_UNIT_MAP_PATH.read_text())["terrains"].items()
}

_TREE_IDS: frozenset[int] = frozenset(
    int(uid) for uid in json.loads(_TREE_UNIT_IDS_PATH.read_text())["ids"]
)

_ALL_TERRAIN_UNIT_CONSTS: frozenset[int] = frozenset(
    spec["id"] for units in _TERRAIN_UNITS.values() for spec in units
)

# Every const terrain_unit_map.json can place, partitioned by "is a tree" --
# not "the const exists in tree_unit_ids.json", but restricted to consts this
# feature actually places, so a future terrain-table addition outside either
# set fails a test rather than silently falling into the wrong bucket.
TREE_CONSTS: frozenset[int] = _ALL_TERRAIN_UNIT_CONSTS & _TREE_IDS
DOODAD_CONSTS: frozenset[int] = _ALL_TERRAIN_UNIT_CONSTS - _TREE_IDS


def is_tree_terrain(terrain_id: int) -> bool:
    """True if painting this terrain can place a tree (the Trees checkbox)."""
    return any(spec["id"] in TREE_CONSTS for spec in _TERRAIN_UNITS.get(terrain_id, ()))


def is_doodad_terrain(terrain_id: int) -> bool:
    """True if painting this terrain can place an eye-candy doodad (the Eye
    candy checkbox) -- grass tufts, jungle underbrush, and the like."""
    return any(spec["id"] in DOODAD_CONSTS for spec in _TERRAIN_UNITS.get(terrain_id, ()))


def _enabled_consts(*, trees: bool, doodads: bool) -> frozenset[int]:
    consts: frozenset[int] = frozenset()
    if trees:
        consts = consts | TREE_CONSTS
    if doodads:
        consts = consts | DOODAD_CONSTS
    return consts


def roll_unit(terrain_id: int, *, trees: bool, doodads: bool, rng: random.Random) -> tuple[int, int] | None:
    """Walks terrain_id's unit list in roll order, skipping any const outside
    the enabled categories, and returns the first (unit_const, centering) to
    succeed its density roll (`rng.random() * 1000 < density`) -- or None if
    every enabled entry failed its roll (or the terrain has no entries at
    all). See the plan's "Placement model adopted" for why walking in order
    and stopping at the first success is the reading the game's own
    non-monotonic density lists (e.g. FOREST_DRY_SOUTH_AMERICAN) require.

    A disabled-category const consumes no roll -- it is skipped outright, not
    rolled and discarded -- so toggling Eye candy off never changes which
    random draws the enabled tree entries see.
    """
    enabled = _enabled_consts(trees=trees, doodads=doodads)
    for spec in _TERRAIN_UNITS.get(terrain_id, ()):
        const = spec["id"]
        if const not in enabled:
            continue
        if rng.random() * 1000 < spec["density"]:
            return const, spec["centering"]
    return None


def variant_for(const: int, rng: random.Random) -> int:
    """A random graphic-variant index in range for `const`'s standing
    graphic: `rng.randrange(angle_count)`. Deliberately not
    unit_sprites.sld_frame_count() -- that returns None with no game install
    configured, which would make the same paint gesture write different
    files on different machines."""
    return rng.randrange(unit_rotation.angle_count_for(const))


@dataclass
class UnitAddSpec:
    """One planned placement -- enough for UnitEditModel.add(player=GAIA, ...).
    `rotation` and `initial_animation_frame` are always the same variant
    index (both float and int forms), matching the corpus's 100% agreement
    on that pairing for tree/doodad placements."""

    x: float
    y: float
    unit_const: int
    rotation: float
    initial_animation_frame: int


@dataclass
class TerrainUnitPlan:
    """What one gesture's terrain change means for GAIA trees/doodads --
    the pure planner's whole output. `removes` holds the actual (tracked)
    Unit objects to delete; `adds` holds specs for units to place."""

    adds: list[UnitAddSpec] = field(default_factory=list)
    removes: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.adds or self.removes)


def plan_terrain_units(
    changes: Sequence[tuple[int, tuple, tuple]],
    tiles_by_index: dict,
    existing: dict,
    *,
    trees: bool,
    doodads: bool,
    rng: random.Random,
) -> TerrainUnitPlan:
    """The pure planner: no model mutation, no Qt, no undo bookkeeping.

    `changes` is a TileDiffRecord.changes list -- (tile_index, old_state,
    new_state) triples, each state a (terrain_id, elevation, layer) tuple.
    Only entries whose terrain_id actually changed are considered:
    repainting a tile with the terrain it already has plans nothing, so a
    stroke that only resets `layer` never touches trees.

    `tiles_by_index` maps a changed tile's index to its (x, y) tile
    coordinate -- kept out of this module so it stays free of map-width
    arithmetic. `existing` is a {(x, y): [unit, ...]} index over GAIA units
    whose const is in the enabled categories, built once for the whole
    gesture by the caller; this function removes from it as it plans, so
    that index stays correct for any later caller in the same gesture.
    """
    plan = TerrainUnitPlan()
    if not trees and not doodads:
        return plan
    enabled = _enabled_consts(trees=trees, doodads=doodads)

    for index, old, new in changes:
        old_terrain_id = old[0]
        new_terrain_id = new[0]
        if old_terrain_id == new_terrain_id:
            continue
        x, y = tiles_by_index[index]

        for unit in list(existing.get((x, y), ())):
            if unit.unit_const in enabled:
                plan.removes.append(unit)
                existing[(x, y)].remove(unit)

        rolled = roll_unit(new_terrain_id, trees=trees, doodads=doodads, rng=rng)
        if rolled is None:
            continue
        const, centering = rolled
        if centering:
            px, py = x + 0.5, y + 0.5
        else:
            px, py = x + rng.uniform(0.1, 0.9), y + rng.uniform(0.1, 0.9)
        variant = variant_for(const, rng)
        plan.adds.append(
            UnitAddSpec(
                x=px,
                y=py,
                unit_const=const,
                rotation=float(variant),
                initial_animation_frame=variant,
            )
        )

    return plan
