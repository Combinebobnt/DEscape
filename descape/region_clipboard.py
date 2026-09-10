"""Region copy/paste payload -- the Select tool's clipboard (phase 2.8).

Qt-free, matching the house style of brush.py/fill_tools.py/elevation_tools.py:
this is plain data plus pure functions over MapManager/UnitManager, so it is
testable headlessly and importable without a running QApplication.

A RegionBlock is one immutable snapshot of a rectangular region -- terrain,
elevation and units -- captured once by copy_region() and replayed any number
of times by paste_terrain()/elevation_targets(). One type serves as both the
clipboard entry and (for a later phase) a floating piece, so a future
float-and-drag paste needs no second type.

Units are dropped by copy_region()'s own bounds check, not gathered via
unit_pick.units_in_rect(): that index is built lazily on entering Units mode
(viewer.py's _rebuild_unit_index()) and is None while in Terrain mode, where
the Select tool lives -- a pick-index lookup would silently return no units
here. A direct walk over the 9 player lists costs one pass over the whole
document (tens of thousands of units at the largest) for one human-paced
gesture, which is free at this density.

A copied unit's garrisoned_in_id is dropped, not carried: it names a
reference_id in the *source* file, and paste mints new reference ids, so
carrying it over would point at the wrong unit or at nothing. Remapping
within-region garrison links is a legitimate follow-up, not handled here.

caption_string_id/caption_string are read defensively (a version before
1.54/1.55 has no such field at all, and the library raises
UnsupportedAttributeError reading one rather than returning an absent
value -- descape/unit_model.py's own module docstring documents the same
poisoning behaviour). Falling back to add()'s own defaults (-1, "") on a
version that doesn't support them is exactly what paste would produce
anyway: unit_model.UnitEditModel.add()'s docstring notes that
commit()'s write path already silently drops a field whose Support range
excludes the scenario version, regardless of what value it's given.
"""

from __future__ import annotations

from dataclasses import dataclass

from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError
from AoE2ScenarioParser.objects.managers.map_manager import MapManager
from AoE2ScenarioParser.objects.managers.unit_manager import UnitManager


@dataclass(frozen=True)
class RegionUnit:
    player: int
    unit_const: int
    dx: float  # relative to the source region's tx0
    dy: float  # relative to the source region's ty0
    z: float
    rotation: float  # verbatim -- never transformed, see AGENTS.md's hard rule
    status: int
    initial_animation_frame: int
    caption_string_id: int
    caption_string: str


@dataclass(frozen=True)
class RegionBlock:
    width: int
    height: int
    terrain_ids: tuple[int, ...]  # row-major, len == width * height
    elevations: tuple[int, ...]  # row-major, len == width * height
    layers: tuple[int, ...]  # row-major, len == width * height -- captured for
    # symmetry with terrain_ids/elevations, but never written back on paste:
    # paste_terrain() always resets layer to -1, matching every other
    # terrain-write path in this tool (batch_api.set_terrain,
    # on_edit_stroke_tile).
    units: tuple[RegionUnit, ...]


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

    units = []
    for player_units in unit_manager.units:
        for u in player_units:
            if tx0 <= int(u.x) < tx1 and ty0 <= int(u.y) < ty1:
                try:
                    caption_string_id, caption_string = u.caption_string_id, u.caption_string
                except UnsupportedAttributeError:
                    caption_string_id, caption_string = -1, ""
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
                    )
                )

    return RegionBlock(
        width=width,
        height=height,
        terrain_ids=tuple(terrain_ids),
        elevations=tuple(elevations),
        layers=tuple(layers),
        units=tuple(units),
    )


def _clipped_bounds(block: RegionBlock, tx0: int, ty0: int, map_width: int, map_height: int):
    """The (x0, y0, x1, y1) absolute, half-open, on-map subrange of a paste
    anchored at (tx0, ty0) -- writes the part that fits rather than refusing
    a paste that runs off the map edge."""
    x0 = max(0, tx0)
    y0 = max(0, ty0)
    x1 = min(map_width, tx0 + block.width)
    y1 = min(map_height, ty0 + block.height)
    return x0, y0, x1, y1


def paste_terrain(mm: MapManager, block: RegionBlock, tx0: int, ty0: int) -> None:
    """Writes block's terrain_ids into the map anchored at (tx0, ty0),
    clipped to the map's bounds. Always resets layer to -1 -- see
    RegionBlock.layers' own comment."""
    x0, y0, x1, y1 = _clipped_bounds(block, tx0, ty0, mm.map_width, mm.map_height)
    for y in range(y0, y1):
        row = (y - ty0) * block.width
        for x in range(x0, x1):
            tile = mm.terrain[y * mm.map_width + x]
            tile.terrain_id = block.terrain_ids[row + (x - tx0)]
            tile.layer = -1


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
        for x in range(x0, x1):
            targets.append((x, y, block.elevations[row + (x - tx0)]))
    return targets


def unit_paste_targets(
    block: RegionBlock, tx0: int, ty0: int, map_width: int, map_height: int
) -> list[tuple[RegionUnit, float, float]]:
    """(unit, x, y) for every RegionUnit whose translated position lands
    on-map -- dropped with the same clip terrain/elevation use, per (unit,
    absolute x, absolute y) so the caller need not re-derive the translation."""
    targets = []
    for u in block.units:
        x, y = tx0 + u.dx, ty0 + u.dy
        if 0 <= int(x) < map_width and 0 <= int(y) < map_height:
            targets.append((u, x, y))
    return targets
