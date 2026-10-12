"""Flood-fill (Paint Can) terrain painting.

Iterative, non-recursive traversal -- a stack of flat tile indices, marked
visited at push time rather than at pop time. That invariant is what bounds
the stack at map_width * map_height and makes a Python list usable as a LIFO
worklist without ever needing to revisit or grow past that size. Shape lifted
from a proven ACS implementation (orc_slayer's FloodFill), not the recursive
form flood fill is usually described with.

Deliberately never calls MapManager.get_tile()/get_tile_safe(): get_tile
raises ValueError on every coordinate of a non-square map (it goes through
map_size, which itself raises when width != height), and get_tile_safe just
swallows that into None for every tile, not only off-map ones -- confirmed
directly against AoE2ScenarioParser's source. A raise partway through a fill
would leave EditHistory's in-progress stroke snapshot uncleared (no
try/finally in EditHistory.apply()), wedging every later edit. Indexing
mm.terrain[y * mm.map_width + x] directly has no such failure mode.

Duck-typed on purpose: only mm.map_width/map_height/terrain and each tile's
terrain_id/layer are touched, so a plain fake object works for tests without
constructing a real AoE2ScenarioParser MapManager -- the same fake-friendly
contract descape/edit_history.py documents for its own tile parameter.

descape/batch_api.py must not import this module back -- a scriptable fill
is reached directly as descape.fill_tools.flood_fill_terrain (or flood_fill
with a terrain_target target), the same way elevation_tools is used
standalone.
"""

from __future__ import annotations

from collections.abc import Callable

from AoE2ScenarioParser.objects.managers.map_manager import MapManager

from descape.batch_api import _ORTHOGONAL_OFFSETS
from descape.terrain_target import CLEAR, LAYER, TERRAIN, write_tile


def contiguous_region(mm: MapManager, x: int, y: int, *, match_layer: bool = False) -> list[int]:
    """Flat indices (mm.terrain's own row-major indexing, y * map_width + x)
    of the 4-connected region of tiles sharing (x, y)'s terrain_id, including
    (x, y) itself. [] if (x, y) is off-map. Reads mm.terrain but mutates
    nothing -- safe to call as a pure query.

    match_layer=True bounds the region by the (terrain_id, layer) pair
    instead, so it stops at a base boundary and at a different layer."""
    width, height = mm.map_width, mm.map_height
    if not (0 <= x < width and 0 <= y < height):
        return []
    terrain = mm.terrain
    start = y * width + x
    source_id = terrain[start].terrain_id
    source_layer = terrain[start].layer

    visited = bytearray(width * height)
    visited[start] = 1
    stack = [start]
    region: list[int] = []
    while stack:
        i = stack.pop()
        region.append(i)
        cy, cx = divmod(i, width)
        for dx, dy in _ORTHOGONAL_OFFSETS:
            nx, ny = cx + dx, cy + dy
            # Explicit column/row bounds checks, not i +/- 1 or i +/- width:
            # without them, x=0's "left" neighbour silently wraps onto the
            # previous row's last tile (and x=width-1's "right" neighbour
            # onto the next row's first) -- the classic flood-fill wraparound
            # bug that flat-index arithmetic invites.
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            j = ny * width + nx
            if visited[j]:
                continue
            tile = terrain[j]
            if tile.terrain_id != source_id:
                continue
            if match_layer and tile.layer != source_layer:
                continue
            # Marked visited at push, not at pop -- each index can enter the
            # stack at most once, which is what bounds it at width * height
            # and makes the plain-list stack safe without a separate cap.
            visited[j] = 1
            stack.append(j)
    return region


def fill_indices(mm: MapManager, x: int, y: int, target: str, terrain_id: int) -> list[int]:
    """Exactly the flat indices a Paint Can click at (x, y) would change for
    `target` (terrain_target.TERRAIN, LAYER or CLEAR), in traversal order.
    [] for a click that changes nothing, and for an off-map one. A pure
    query: reads mm.terrain, mutates nothing.

    TERRAIN: the clicked terrain_id region. When that region already has
    terrain_id, only its tiles carrying a layered terrain, which a terrain
    write clears.
    LAYER: the region sharing the clicked (terrain_id, layer) pair, so a fill
    never crosses a base boundary or overwrites a different layer. [] when
    the clicked tile already has the layer this would write.
    CLEAR: that same pair region, [] when the clicked tile has no layer."""
    if target not in (TERRAIN, LAYER, CLEAR):
        raise ValueError(f"unknown terrain target {target!r}")
    width, height = mm.map_width, mm.map_height
    if not (0 <= x < width and 0 <= y < height):
        return []
    clicked = mm.terrain[y * width + x]
    if target == TERRAIN:
        region = contiguous_region(mm, x, y)
        if clicked.terrain_id != terrain_id:
            return region
        # One pass over mm.terrain: per-index reads go through UuidList's slow __getitem__.
        layers = [tile.layer for tile in mm.terrain]
        return [i for i in region if layers[i] != -1]
    if target == LAYER:
        # batch_api.set_layer's rule: a layer equal to the base is none.
        value = -1 if terrain_id == clicked.terrain_id else terrain_id
        if clicked.layer == value:
            return []
    elif clicked.layer == -1:
        return []
    return contiguous_region(mm, x, y, match_layer=True)


def flood_fill(mm: MapManager, x: int, y: int, target: str, terrain_id: int) -> list[int]:
    """Paint Can's write: terrain_target.write_tile() over
    fill_indices(mm, x, y, target, terrain_id). Returns the flat indices
    written, in traversal order, [] when nothing changed."""
    region = fill_indices(mm, x, y, target, terrain_id)
    terrain = mm.terrain
    for i in region:
        write_tile(terrain[i], target, terrain_id)
    return region


def flood_fill_terrain(mm: MapManager, x: int, y: int, terrain_id: int) -> list[int]:
    """flood_fill() on the Terrain target: paints terrain_id over the clicked
    region and clears its layered terrain, as every terrain write does.
    Returns the flat indices written, in traversal order.

    On a region that already has terrain_id it writes only the tiles that
    carry a layered terrain, clearing it, and returns [] when there are
    none (or (x, y) is off-map), so a no-op fill leaves no dirty tile and no
    undo record."""
    return flood_fill(mm, x, y, TERRAIN, terrain_id)


def connected_regions(mm: MapManager, passable: Callable[[int], bool]) -> list[list[int]]:
    """Every 4-connected region of tiles whose flat index satisfies
    `passable(i)`, each as a list of flat indices. The sibling of
    contiguous_region() for a predicate rather than exact terrain_id
    equality, which other callers depend on and so stays unbent. Same
    push-time visited marking and explicit bounds checks; reads only."""
    width, height = mm.map_width, mm.map_height
    visited = bytearray(width * height)
    regions: list[list[int]] = []
    for start in range(width * height):
        if visited[start]:
            continue
        visited[start] = 1
        if not passable(start):
            continue
        stack = [start]
        region: list[int] = []
        while stack:
            i = stack.pop()
            region.append(i)
            cy, cx = divmod(i, width)
            for dx, dy in _ORTHOGONAL_OFFSETS:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                j = ny * width + nx
                if visited[j]:
                    continue
                visited[j] = 1
                if passable(j):
                    stack.append(j)
        regions.append(region)
    return regions
