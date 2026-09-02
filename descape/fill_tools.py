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
is reached directly as descape.fill_tools.flood_fill_terrain, the same way
elevation_tools is used standalone.
"""

from __future__ import annotations

from AoE2ScenarioParser.objects.managers.map_manager import MapManager

from descape.batch_api import _ORTHOGONAL_OFFSETS, set_terrain


def contiguous_region(mm: MapManager, x: int, y: int) -> list[int]:
    """Flat indices (mm.terrain's own row-major indexing, y * map_width + x)
    of the 4-connected region of tiles sharing (x, y)'s terrain_id, including
    (x, y) itself. [] if (x, y) is off-map. Reads mm.terrain but mutates
    nothing -- safe to call as a pure query."""
    width, height = mm.map_width, mm.map_height
    if not (0 <= x < width and 0 <= y < height):
        return []
    terrain = mm.terrain
    start = y * width + x
    source_id = terrain[start].terrain_id

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
            if terrain[j].terrain_id != source_id:
                continue
            # Marked visited at push, not at pop -- each index can enter the
            # stack at most once, which is what bounds it at width * height
            # and makes the plain-list stack safe without a separate cap.
            visited[j] = 1
            stack.append(j)
    return region


def flood_fill_terrain(mm: MapManager, x: int, y: int, terrain_id: int) -> list[int]:
    """Paints terrain_id over contiguous_region(mm, x, y) via
    batch_api.set_terrain (terrain_id + layer reset to -1, same as every
    other terrain-write path in this tool). Returns the flat indices
    written, in traversal order.

    Returns [] and writes nothing if (x, y) is off-map, or if the region
    already has terrain_id -- a deliberate early-out, not just an
    optimization: without it, set_terrain's layer = -1 would still produce
    real dirty tiles (and a phantom undo record) on any same-terrain region
    whose tiles carry a real second terrain via `layer` (true for roughly 1
    in 4 tiles in this project's real example files). The accepted
    consequence is that Paint Can will not normalize a stale `layer` on a
    region that already has the selected terrain -- consistent with
    terrain-blending being an explicit non-goal elsewhere in this tool."""
    width, height = mm.map_width, mm.map_height
    if not (0 <= x < width and 0 <= y < height):
        return []
    if mm.terrain[y * width + x].terrain_id == terrain_id:
        return []

    region = contiguous_region(mm, x, y)
    terrain = mm.terrain
    for i in region:
        set_terrain(terrain[i], terrain_id)
    return region
