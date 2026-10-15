"""Region copy/paste payload -- the Select tool's clipboard (phase 2.8).

Qt-free, matching the house style of brush.py/fill_tools.py/elevation_tools.py:
this is plain data plus pure functions over MapManager/UnitManager, so it is
testable headlessly and importable without a running QApplication.

A RegionBlock is one immutable snapshot of a rectangular region -- terrain,
elevation and units -- captured once by copy_region() and replayed any number
of times by paste_terrain()/elevation_targets(). One type serves as both the
clipboard entry and (for a later phase) a floating piece, so a future
float-and-drag paste needs no second type.

copy_units() makes the same type without tiles (GH #192, Units mode's Copy
and Cut): RegionBlock.has_tiles tells the two apart, and only the unit half
(unit_paste_targets()) may be called on a block without them.

Units are dropped by copy_region()'s own bounds check, not gathered via
unit_pick.units_in_rect(): that index is built lazily on entering Units mode
(viewer.py's _rebuild_unit_index()) and is current or None in Terrain mode,
where the Select tool lives: it may survive outside Units mode, but it is
often None there, so a pick-index lookup could silently return no units here.
A direct walk over the 9 player lists costs one pass over the whole
document (tens of thousands of units at the largest) for one human-paced
gesture, which is free at this density.

A copied unit's garrisoned_in_id is not carried verbatim: it names a
reference_id in the *source* file, and paste mints new reference ids, so
copying it over would point at the wrong unit or at nothing. Instead
copy_region() resolves it to garrison_slot, an index into this same
RegionBlock's own units tuple (or -1), and unit_paste_targets() turns that
back into a real id once the holder's own paste has minted one. A link
whose holder wasn't captured in the same region, is ambiguous, or is
self-referential resolves to -1 rather than guessing.

caption_string_id/caption_string are read defensively (a version before
1.54/1.55 has no such field at all, and the library raises
UnsupportedAttributeError reading one rather than returning an absent
value -- descape/unit_model.py's own module docstring documents the same
poisoning behaviour). Falling back to add()'s own defaults (-1, "") on a
version that doesn't support them is exactly what paste would produce
anyway: unit_model.UnitEditModel.add()'s docstring notes that
commit()'s write path already silently drops a field whose Support range
excludes the scenario version, regardless of what value it's given.
This read also catches plain AttributeError: a unit parsed while its class
was poisoned (library_compat.depoison() only fixes the class, not objects
already parsed under the old one) has no instance attribute at all for a
disabled link, and reading it raises AttributeError instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError
from AoE2ScenarioParser.objects.managers.map_manager import MapManager
from AoE2ScenarioParser.objects.managers.unit_manager import UnitManager

from descape.unit_model import capture_flag_of


@dataclass(frozen=True)
class RegionUnit:
    player: int
    unit_const: int
    dx: float  # relative to the source region's tx0
    dy: float  # relative to the source region's ty0
    z: float
    # Verbatim at copy and paste; only region_transform.transform_block() derives a new one (AGENTS.md).
    rotation: float
    status: int
    initial_animation_frame: int
    caption_string_id: int
    caption_string: str
    garrison_slot: int  # index into RegionBlock.units, or -1 -- resolved at copy time
    capture_flag: int = -1  # 1.59's per-unit capture setting, -1 (Default) before it


@dataclass(frozen=True)
class RegionBlock:
    """One clipboard snapshot: a tile rectangle with the units on it, or
    units alone (GH #192, copy_units()).

    A units-only block has empty tile tuples, and has_tiles is the one test
    for it: a copied region is never empty (normalize_region() returns None
    instead), so empty tuples can only mean "no tiles were copied". Its
    width and height are then the copied units' anchor-tile bounding box."""

    width: int
    height: int
    terrain_ids: tuple[int, ...]  # row-major, len == width * height
    elevations: tuple[int, ...]  # row-major, len == width * height
    # Row-major, len == width * height. Each tile's layered terrain (-1 = none),
    # which paste_terrain() writes back verbatim beside terrain_ids.
    layers: tuple[int, ...]
    units: tuple[RegionUnit, ...]

    @property
    def has_tiles(self) -> bool:
        return bool(self.terrain_ids)


def normalize_region(
    anchor: tuple[int, int], other: tuple[int, int], map_width: int, map_height: int
) -> tuple[int, int, int, int] | None:
    """The half-open tile rectangle spanning `anchor` and `other`, inclusive
    at both ends (the two points are tiles, not corners -- dragging from one
    tile to its neighbour selects both; a press-release with no movement
    selects that one tile), clamped by intersection with the map rather than
    refused -- a drag legitimately runs off the map edge. None only if the
    intersection is empty (an anchor or endpoint entirely off-map)."""
    ax, ay = anchor
    bx, by = other
    tx0, ty0 = min(ax, bx), min(ay, by)
    tx1, ty1 = max(ax, bx) + 1, max(ay, by) + 1
    tx0, ty0 = max(0, tx0), max(0, ty0)
    tx1, ty1 = min(map_width, tx1), min(map_height, ty1)
    if tx0 >= tx1 or ty0 >= ty1:
        return None
    return tx0, ty0, tx1, ty1


def copy_region(
    mm: MapManager, unit_manager: UnitManager, tx0: int, ty0: int, tx1: int, ty1: int
) -> RegionBlock:
    """Snapshots the half-open tile rectangle [tx0, tx1) x [ty0, ty1).

    Indexes mm.terrain directly (mm.terrain[y * mm.map_width + x]), not via
    mm.get_tile(), which raises on a non-square map -- the same reason
    fill_tools.py and ViewerWindow.pick_tile_value already do so.
    """
    width, height = tx1 - tx0, ty1 - ty0
    terrain_ids = []
    elevations = []
    layers = []
    for y in range(ty0, ty1):
        for x in range(tx0, tx1):
            tile = mm.terrain[y * mm.map_width + x]
            terrain_ids.append(tile.terrain_id)
            elevations.append(tile.elevation)
            layers.append(tile.layer)

    captured = [
        u
        for player_units in unit_manager.units
        for u in player_units
        if tx0 <= int(u.x) < tx1 and ty0 <= int(u.y) < ty1
    ]
    return RegionBlock(
        width=width,
        height=height,
        terrain_ids=tuple(terrain_ids),
        elevations=tuple(elevations),
        layers=tuple(layers),
        units=_region_units(captured, tx0, ty0),
    )


def _region_units(captured, tx0: int, ty0: int) -> tuple[RegionUnit, ...]:
    """The RegionUnit of every unit in `captured`, in that order, with dx/dy
    relative to (tx0, ty0) and garrison links resolved to slots in the
    returned tuple: copy_region()'s and copy_units()' shared unit half."""
    captions = []
    for u in captured:
        try:
            captions.append((u.caption_string_id, u.caption_string))
        except (UnsupportedAttributeError, AttributeError):
            captions.append((-1, ""))

    # reference_id -> slot, but only for a reference_id captured exactly
    # once: a duplicate must resolve every link pointing at it to -1 rather
    # than let dict last-wins silently pick one holder over the other.
    slot_by_ref: dict[int, int] = {}
    duplicate_refs: set[int] = set()
    for slot, u in enumerate(captured):
        if u.reference_id in slot_by_ref:
            duplicate_refs.add(u.reference_id)
        else:
            slot_by_ref[u.reference_id] = slot

    units = []
    for u, (caption_string_id, caption_string) in zip(captured, captions, strict=True):
        holder_ref = u.garrisoned_in_id
        if holder_ref == -1 or holder_ref in duplicate_refs or holder_ref == u.reference_id:
            garrison_slot = -1
        else:
            garrison_slot = slot_by_ref.get(holder_ref, -1)
        units.append(
            RegionUnit(
                player=u.player,
                unit_const=u.unit_const,
                dx=u.x - tx0,
                dy=u.y - ty0,
                z=u.z,
                rotation=u.rotation,
                status=u.status,
                initial_animation_frame=u.initial_animation_frame,
                caption_string_id=caption_string_id,
                caption_string=caption_string,
                garrison_slot=garrison_slot,
                # -1 on a pre-1.59 unit, whose poisoned link raises on read.
                capture_flag=capture_flag_of(u),
            )
        )
    return tuple(units)


def copy_units(unit_manager: UnitManager, units) -> RegionBlock | None:
    """A units-only block of `units` and everything garrisoned inside them,
    transitively (GH #192). None when no passed unit is in the document.

    Units are matched by identity, never by key: two units can share a
    reference_id, and a key match would pull in the unselected twin. Like
    copy_region() this walks the nine lists directly, so it needs no pick
    index and no edit model. Slots follow list order.

    The origin and size are the passed units' anchor-tile bounding box
    alone: an occupant stored on some other tile rides along at its own
    offset, but cannot shift the group off the paste cursor."""
    wanted = {id(u) for u in units}
    chosen = [u for player_units in unit_manager.units for u in player_units if id(u) in wanted]
    if not chosen:
        return None
    tx0, ty0 = min(int(u.x) for u in chosen), min(int(u.y) for u in chosen)
    tx1, ty1 = max(int(u.x) for u in chosen), max(int(u.y) for u in chosen)

    taken = {id(u) for u in chosen}
    holder_refs = {u.reference_id for u in chosen}
    grew = True
    while grew:
        grew = False
        for player_units in unit_manager.units:
            for u in player_units:
                inside = u.garrisoned_in_id
                if id(u) not in taken and inside != -1 and inside in holder_refs:
                    taken.add(id(u))
                    holder_refs.add(u.reference_id)
                    grew = True
    captured = [u for player_units in unit_manager.units for u in player_units if id(u) in taken]
    return RegionBlock(
        width=tx1 - tx0 + 1,
        height=ty1 - ty0 + 1,
        terrain_ids=(),
        elevations=(),
        layers=(),
        units=_region_units(captured, tx0, ty0),
    )


def centred_anchor(block: RegionBlock, hx: int, hy: int) -> tuple[int, int]:
    """The anchor that centres `block`'s bounding box on tile (hx, hy), in
    whole tiles so every unit keeps its sub-tile offset. An even size puts
    the extra tile on the high side. Not clamped to any map."""
    return hx - (block.width - 1) // 2, hy - (block.height - 1) // 2


def _clipped_bounds(block: RegionBlock, tx0: int, ty0: int, map_width: int, map_height: int):
    """The (x0, y0, x1, y1) absolute, half-open, on-map subrange of a paste
    anchored at (tx0, ty0) -- writes the part that fits rather than refusing
    a paste that runs off the map edge."""
    x0 = max(0, tx0)
    y0 = max(0, ty0)
    x1 = min(map_width, tx0 + block.width)
    y1 = min(map_height, ty0 + block.height)
    return x0, y0, x1, y1


def clipped_indices(block: RegionBlock, tx0: int, ty0: int, map_width: int, map_height: int) -> list[int]:
    """Flat `y * map_width + x` index of every on-map tile a paste anchored at
    (tx0, ty0) writes: paste_terrain()'s and elevation_targets()' shared
    rectangle. Elevation's skirt, which set_tiles_elevation() indexes by
    map_size, only runs on square maps, where the two agree."""
    x0, y0, x1, y1 = _clipped_bounds(block, tx0, ty0, map_width, map_height)
    return [y * map_width + x for y in range(y0, y1) for x in range(x0, x1)]


def paste_terrain(mm: MapManager, block: RegionBlock, tx0: int, ty0: int, *, carry_layer: bool = True) -> None:
    """Writes block's terrain_ids and layers into the map anchored at
    (tx0, ty0), clipped to the map's bounds. Each (terrain_id, layer) pair is
    copied verbatim, never normalised. carry_layer=False writes every layer
    as -1 instead: for a file whose terrain has no layer field, where a
    pasted layer would draw and never save."""
    x0, y0, x1, y1 = _clipped_bounds(block, tx0, ty0, mm.map_width, mm.map_height)
    for y in range(y0, y1):
        row = (y - ty0) * block.width
        for x in range(x0, x1):
            tile = mm.terrain[y * mm.map_width + x]
            tile.terrain_id = block.terrain_ids[row + (x - tx0)]
            tile.layer = block.layers[row + (x - tx0)] if carry_layer else -1


def elevation_targets(
    block: RegionBlock, tx0: int, ty0: int, map_width: int, map_height: int
) -> list[tuple[int, int, int]]:
    """(x, y, elevation) for every on-map tile block's elevations would land
    on when pasted at (tx0, ty0) -- feeds elevation_tools.set_tiles_elevation,
    whose footprint-wide `xys` guard is what keeps the pasted interior
    byte-exact (see the plan's own verification of this claim)."""
    x0, y0, x1, y1 = _clipped_bounds(block, tx0, ty0, map_width, map_height)
    targets = []
    for y in range(y0, y1):
        row = (y - ty0) * block.width
        targets.extend((x, y, block.elevations[row + (x - tx0)]) for x in range(x0, x1))
    return targets


@dataclass(frozen=True)
class PasteTarget:
    slot: int  # index into block.units
    unit: RegionUnit
    x: float
    y: float
    holder_slot: int  # -1, or the slot of a holder that is itself in this list


def unit_paste_targets(
    block: RegionBlock, tx0: int, ty0: int, map_width: int, map_height: int
) -> list[PasteTarget]:
    """A PasteTarget for every RegionUnit whose translated position lands
    on-map, in holder-before-occupant order (a stable Kahn topological sort
    over garrison_slot, tiebroken by original slot order) so the caller can
    add each unit in list order and look its holder's freshly minted id up
    by slot.

    garrison_slot links into slots dropped by the on-map clip, or into a
    cycle (mutual or longer -- only reachable on a hand-built or foreign
    file, since copy_region() never produces one), are both severed to -1
    here rather than left to loop forever or point at nothing."""
    survivors: dict[int, tuple[float, float]] = {}
    for slot, u in enumerate(block.units):
        x, y = tx0 + u.dx, ty0 + u.dy
        # The sign is tested on the float: int() truncates toward zero, so -0.5 would pass as tile 0.
        if x >= 0 and y >= 0 and int(x) < map_width and int(y) < map_height:
            survivors[slot] = (x, y)

    emitted: set[int] = set()
    broken: set[int] = set()
    ordered: list[int] = []
    remaining = list(survivors)
    while remaining:
        ready = [s for s in remaining if block.units[s].garrison_slot == -1
                 or block.units[s].garrison_slot not in survivors
                 or block.units[s].garrison_slot in emitted]
        still = [s for s in remaining if s not in ready]
        if not ready:
            ready, still = still, []
            broken.update(ready)
        ordered.extend(ready)
        emitted.update(ready)
        remaining = still

    targets = []
    for slot in ordered:
        u = block.units[slot]
        x, y = survivors[slot]
        holder_slot = -1 if slot in broken or u.garrison_slot not in survivors else u.garrison_slot
        targets.append(PasteTarget(slot=slot, unit=u, x=x, y=y, holder_slot=holder_slot))
    return targets


def translated_region(
    region: tuple[int, int, int, int], dx: int, dy: int, map_width: int, map_height: int
) -> tuple[int, int, int, int] | None:
    """A committed half-open rect shifted by (dx, dy) and clamped by
    intersection with the map, exactly as normalize_region() clamps rather
    than refusing -- a move legitimately drags off the map edge. None only
    when the translated rect lands entirely off-map.

    This feeds the OVERLAY only. The block itself is always re-pasted at its
    own unclipped anchor, which is what lets a region dragged partly off the
    map and back come back whole."""
    tx0, ty0, tx1, ty1 = region
    tx0, ty0, tx1, ty1 = tx0 + dx, ty0 + dy, tx1 + dx, ty1 + dy
    tx0, ty0 = max(0, tx0), max(0, ty0)
    tx1, ty1 = min(map_width, tx1), min(map_height, ty1)
    if tx0 >= tx1 or ty0 >= ty1:
        return None
    return tx0, ty0, tx1, ty1


@dataclass(frozen=True)
class PasteMove:
    """Everything a move needs to replay the paste that armed it.

    `anchor` is the UNCLIPPED anchor the paste was made at, not the clamped
    region the selection was moved to: every commit re-pastes the full
    snapshot at an unclipped anchor and _clipped_bounds() writes the part that
    fits, so no intermediate off-map position can lose content.

    The category flags are the ones the paste actually used, not whatever the
    checkboxes say now -- a move replays the paste that happened, it is not a
    second chance to change what gets pasted.

    Lives here rather than in viewer.py so its invariants stay testable
    without Qt, matching RegionBlock's own placement.

    A turn or flip (GH #197) keeps `source`, the block as first pasted, and
    `element`, the D4 element accumulated since, and re-derives `block` from
    the two each time, so nothing drifts. `pivot2` is the doubled centre the
    turns are about: set by the first turn, shifted by a move, never
    re-derived. All three default to "no turn yet"."""

    block: RegionBlock
    anchor: tuple[int, int]
    do_terrain: bool
    do_elevation: bool
    do_units: bool
    record: object
    source: RegionBlock | None = None
    element: str = "id"
    pivot2: tuple[int, int] | None = None
