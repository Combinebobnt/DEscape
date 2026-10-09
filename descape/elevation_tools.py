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
run the same propagation the library's own rectangle branch uses
(MapManager._elevation_tile_recursion()). That propagation is vendored below
as _elevation_tile_recursion() so it can report every tile it writes, which
the viewer's live stroke repaint needs; the library's copy returns nothing.
tests/test_elevation_tools.py checks the two agree on randomized maps.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

from AoE2ScenarioParser.helper.maffs import sign
from AoE2ScenarioParser.objects.managers.map_manager import MapManager


# Adapted from AoE2ScenarioParser (https://github.com/KSneijders/AoE2ScenarioParser),
# v0.8.3 (commit b763e2e3; unchanged through v0.9.4), AoE2ScenarioParser/objects/managers/map_manager.py,
# MapManager._elevation_tile_recursion. GPL-3.0, the same license as DEscape;
# copyright for the original remains with the AoE2ScenarioParser authors.
# Changed only to take `mm` explicitly, add every written index to `touched`,
# and call the optional `before_write(index)` just before each write.
def _elevation_tile_recursion(
    mm: MapManager,
    source_tile,
    xys,
    touched: set[int],
    visited=None,
    before_write: Callable[[int], None] | None = None,
) -> None:
    visited = set() if visited is None else visited.copy()
    x, y = source_tile.xy
    visited.add((x, y))
    size = mm.map_size
    for nx, ny in itertools.product(range(-1, 2), repeat=2):
        new_x, new_y = x + nx, y + ny
        if (nx or ny) and (new_x, new_y) not in xys and (new_x, new_y) not in visited:
            other = mm.get_tile_safe(new_x, new_y)
            if other is None:
                continue
            i = new_y * size + new_x
            behind = mm.get_tile_safe(x + nx * 2, y + ny * 2)
            if behind is not None and other.elevation < source_tile.elevation == behind.elevation:
                if before_write is not None:
                    before_write(i)
                other.elevation = source_tile.elevation
                touched.add(i)
            elif abs(other.elevation - source_tile.elevation) > 1:
                if before_write is not None:
                    before_write(i)
                other.elevation = source_tile.elevation + int(sign(other.elevation, source_tile.elevation))
                touched.add(i)
                _elevation_tile_recursion(mm, other, xys, touched, visited, before_write)


def set_tile_elevation(mm: MapManager, x: int, y: int, elevation: int) -> set[int]:
    """Sets tile (x, y)'s own elevation to `elevation` and propagates to
    neighbors exactly like the in-game brush / MapManager.set_elevation's
    multi-tile branch does. Returns every flat index written, the target
    included. Requires mm.map_width == mm.map_height (see
    LoadedScenario.map_is_square) -- MapManager.get_tile raises otherwise,
    same constraint set_elevation itself has."""
    tile = mm.get_tile(x, y)
    tile.elevation = elevation
    touched = {y * mm.map_size + x}
    _elevation_tile_recursion(mm, tile, {tile.xy}, touched)
    return touched


def set_tiles_elevation(
    mm: MapManager,
    targets: Sequence[tuple[int, int, int]],
    before_write: Callable[[int], None] | None = None,
) -> set[int]:
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

    Returns every flat index written: all targets (even ones already at
    their value) plus every propagated tile. Requires mm.map_width ==
    mm.map_height, same as set_tile_elevation().

    `before_write`, when given, is called with a propagated tile's flat index
    just before each write to it (possibly more than once per tile), so a
    scoped EditHistory stroke can capture it. It is not called for the
    targets: those are the caller's own write set, known up front."""
    footprint = {(x, y) for x, y, _ in targets}
    size = mm.map_size
    touched = {y * size + x for x, y in footprint}
    tiles = []
    for x, y, elevation in targets:
        tile = mm.get_tile(x, y)
        tile.elevation = elevation
        tiles.append(tile)
    for tile in tiles:
        _elevation_tile_recursion(mm, tile, footprint, touched, before_write=before_write)
    return touched
