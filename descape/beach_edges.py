"""The automatic shoreline: given the tiles a touch just painted, lay a ring
of beach terrain between them and the other side of the shore.

Two modes, picked by the painted terrain (`ring_mode()`, which the viewer's
Auto beach gate reads too, so the two cannot disagree):

- **water**: the core is water, and the ring turns the land around it into
  beach, never touching other water.
- **land** (GH #259): the core is land, and the ring turns the water around
  it into beach, never touching land or existing beach. The beach goes
  outward, onto the water, so the core is exactly what the brush covered,
  the same contract as water mode. A beach, a bridge deck or a
  non-navigable shore lays no ring at all.

Qt-free and standalone, in the shape of descape/fill_tools.py -- and for the
same reason it indexes mm.terrain[y * width + x] directly rather than going
through get_tile/get_tile_safe: get_tile raises ValueError on every
coordinate of a non-square map, and a raise partway through would leave
EditHistory's stroke snapshot uncleared, wedging every later edit.
fill_tools.py's own docstring covers that trap in full. Duck-typed on
map_width/map_height/terrain, so the tests need no real MapManager.

No EditHistory involvement: this mutates tiles, and the caller's already-open
stroke captures them for free, because begin_stroke snapshots the whole
terrain array and commit_stroke diffs it.

**The ring is bounded to the tiles this touch painted**, never recomputed
over all water on the map. A global recompute would add shoreline to lakes
the user never touched and fight any deliberately beach-less coast
elsewhere. Bounded is also what makes dragging behave: at each touch the
ring moves with the brush, the tile ahead becomes beach and then water on
the next touch, and the tiles beside it stay beach, with no trailing smear.

The honest cost of that bound, documented rather than left to be discovered:
paint a pond (it gets a ring), then paint an annulus of water around it
whose core misses that ring, and the old beach is left fully enclosed by
water with nothing to recompute it. It is user-fixable by painting over it.
Land mode has the mirror case: paint an island (it gets a ring), then a ring
of land around it whose core misses that beach, and the beach is left fully
enclosed by land.
"""

from __future__ import annotations

from descape import terrain_classes
from descape.batch_api import set_terrain

# Water mode: families a ring tile is never allowed to overwrite. Everything
# else is rewritten, INCLUDING existing beach -- normalizing an old shoreline to the
# current beach terrain is the deliberate choice, so the ring always reads
# as one uniform coast.
_PROTECTED_FAMILIES = frozenset(
    {"medium_water", "deep_water", "shallow_water", "shallows"}
)

# Terrains 79-82 BEACH_NON_NAVIGABLE and its _WET_SAND/_WET_GRAVEL/_WET_ROCK
# siblings. The .dat classifies these as family `land`, so water mode's
# overwrite policy above would convert them -- and turning one into BEACH makes a
# shoreline ships can dock at out of one the map author deliberately made so
# they could not. That is a gameplay change, not a cosmetic one. Carved out
# as a curated id set with the measured reason, rather than by reintroducing
# a name-substring rule that this whole module exists to avoid.
PROTECTED_TERRAIN_IDS = frozenset({79, 80, 81, 82})

# Family-land picks that lay no land ring: 28 WATER_2D_BRIDGE (land in the
# .dat, but a bridge deck must not shore its own river) and the
# non-navigable shores above, which are a shore already.
NO_RING_LAND_IDS = frozenset({28}) | PROTECTED_TERRAIN_IDS

# Shallows-family features a land ring never converts: 55 FOREST_MANGROVE,
# 63-67 RICE_FARM and its growth stages, 90 FOREST_REEDS_SHALLOWS. Those are
# placed deliberately, not open water, so a grass stroke beside a paddy must
# not eat it. Plain shallows and ICE_NAVIGABLE do convert.
KEPT_FEATURE_WATER_IDS = frozenset({55, 63, 64, 65, 66, 67, 90})


def ring_mode(terrain_id: int) -> str | None:
    """"water", "land", or None for a pick that lays no ring (a beach, a
    bridge, a non-navigable shore, an obsolete or unknown id)."""
    if terrain_classes.is_water_family(terrain_id):
        return "water"
    if terrain_classes.terrain_family(terrain_id) == "land" and terrain_id not in NO_RING_LAND_IDS:
        return "land"
    return None


def ring_tiles(
    core, width: int, map_width: int, map_height: int
) -> list[tuple[int, int]]:
    """Every tile within Chebyshev distance `width` of a core tile, minus the
    core itself, clipped to the map.

    Chebyshev (8-connected), not orthogonal: a 4-connected ring leaves
    diagonal shorelines with land tiles touching water at a corner. At width
    2 that means the (+/-2, +/-2) corner IS included.

    Width and height are checked separately, never as one flat-index clamp,
    so a ring straddling an edge can never wrap onto the next row -- the same
    clipping shape brush.brush_tiles uses.
    """
    if width <= 0:
        return []
    core_set = set(core)
    out: dict[tuple[int, int], None] = {}
    for cx, cy in core_set:
        for dy in range(-width, width + 1):
            for dx in range(-width, width + 1):
                tile = (cx + dx, cy + dy)
                if tile in core_set:
                    continue
                if not (0 <= tile[0] < map_width and 0 <= tile[1] < map_height):
                    continue
                out[tile] = None
    return list(out)


def _water_ring_target(current: int, water_id: int, beach_id: int | None) -> int | None:
    if current in PROTECTED_TERRAIN_IDS:
        return None
    if terrain_classes.terrain_family(current) in _PROTECTED_FAMILIES:
        return None
    return beach_id if beach_id is not None else terrain_classes.auto_beach_for(water_id, current)


def _land_ring_target(current: int, land_id: int, beach_id: int | None) -> int | None:
    if not terrain_classes.is_water_family(current) or current in KEPT_FEATURE_WATER_IDS:
        return None
    return beach_id if beach_id is not None else terrain_classes.auto_beach_for(current, land_id)


def apply_beach_ring(mm, core, painted_id: int, beach_id: int | None, width: int) -> list[int]:
    """Lay a beach ring around `core`, the (x, y) tiles this touch just set
    to `painted_id`, in the mode `ring_mode(painted_id)` names (none for a
    pick that lays no ring). Returns the changed flat indices, the way
    fill_tools.flood_fill_terrain does.

    `beach_id=None` means Auto: each ring tile's beach is derived per tile
    from the land and water on its two sides, so one stroke crossing from
    ice into grass gets BEACH_ICE on the ice side and BEACH on the grass side.

    The viewer ignores the return value (the stroke diff already drives the
    repaint), but returning it keeps the headless tests clean and gives
    batch scripts a usable entry point.
    """
    mode = ring_mode(painted_id)
    if mode is None:
        return []
    ring_target = _water_ring_target if mode == "water" else _land_ring_target
    map_width, map_height = mm.map_width, mm.map_height
    terrain = mm.terrain
    changed: list[int] = []
    for x, y in ring_tiles(core, width, map_width, map_height):
        index = y * map_width + x
        tile = terrain[index]
        current = tile.terrain_id
        target = ring_target(current, painted_id, beach_id)
        if target is None:
            continue
        if current == target:
            # Idempotent: a second identical call reports no change, so a
            # repeated touch cannot push a phantom undo step.
            continue
        # set_terrain, not a bare terrain_id assignment: it clears `layer`
        # too. Roughly 1 in 4 tiles in the real examples/ files carry a
        # genuine second terrain there, and a beach left holding a stale
        # blend renders fine here and wrong in-game.
        set_terrain(tile, target)
        changed.append(index)
    return changed
