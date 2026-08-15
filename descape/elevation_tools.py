"""Single-tile elevation editing, on top of MapManager.set_elevation.

Confirmed directly against AoE2ScenarioParser's source (objects/managers/
map_manager.py): set_elevation()'s single-point branch (x1 == x2 and y1 == y2,
which is every call v2's brush-size-1 elevation tools make) never actually
assigns the target tile's own `elevation` -- only the multi-tile rectangle
branch does that, before running the neighbor-propagation recursion. Verified
empirically too: calling set_elevation(tile.elevation + 1, x, y) on an
isolated tile is a silent no-op for that tile. Confirmed on this repo's own
example files, not a hypothetical reading of the code.

set_tile_elevation() below is what the single-point branch would need to do to
behave like the rectangle branch: assign the tile's elevation directly, then
run the same MapManager._elevation_tile_recursion() propagation the library's
own rectangle branch uses.
Reaching into that private method mirrors this project's existing, documented
precedent for touching AoE2ScenarioParser internals when its public API
doesn't cover a case we need -- see descape/scenario_io.py's module docstring.
"""

from __future__ import annotations

from collections.abc import Sequence

from AoE2ScenarioParser.objects.managers.map_manager import MapManager


def set_tile_elevation(mm: MapManager, x: int, y: int, elevation: int) -> None:
    """Sets tile (x, y)'s own elevation to `elevation` and propagates to
    neighbors exactly like the in-game brush / MapManager.set_elevation's
    multi-tile branch does. Requires mm.map_width == mm.map_height (see
    LoadedScenario.map_is_square) -- MapManager.get_tile raises otherwise,
    same constraint set_elevation itself has."""
    tile = mm.get_tile(x, y)
    tile.elevation = elevation
    mm._elevation_tile_recursion(tile, {tile.xy})


def set_tiles_elevation(mm: MapManager, targets: Sequence[tuple[int, int, int]]) -> None:
    """Multi-tile counterpart to set_tile_elevation() above, for a brush
    footprint -- every (x, y, elevation) in `targets` is assigned first, then
    _elevation_tile_recursion() is run once per target tile with `xys` set to
    the WHOLE footprint, exactly mirroring MapManager.set_elevation()'s own
    multi-tile rectangle branch (map_manager.py's `source_tiles`/`xys`/
    `edge_tiles` construction).

    Do NOT build this by calling set_tile_elevation() once per target: that
    passes xys={tile.xy}, a single tile, so each call's propagation is free
    to rewrite any OTHER target tile a previous call in the same loop already
    set -- _elevation_tile_recursion()'s `(new_x, new_y) not in xys` guard is
    exactly what stops that, and it only works if xys is the full set up
    front. Confirmed empirically: on flat ground the two approaches happen to
    agree, but on ordinary uneven terrain the per-tile-loop version badly
    over-smooths (observed flood-filling a uniform plateau across a region
    several tiles wider than the brush, instead of a clean per-tile delta).

    Requires mm.map_width == mm.map_height, same as set_tile_elevation()."""
    footprint = {(x, y) for x, y, _ in targets}
    tiles = []
    for x, y, elevation in targets:
        tile = mm.get_tile(x, y)
        tile.elevation = elevation
        tiles.append(tile)
    for tile in tiles:
        mm._elevation_tile_recursion(tile, footprint)
