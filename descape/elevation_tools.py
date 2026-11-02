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

The propagation's gap-fill branch raises a tile without recursing from it, so
the raised tile can end up 2+ levels from one of its own neighbours: an
illegal step, which crashes the game on load. The gap fill stays (it is the
in-game brush's shape), and set_tiles_elevation() repairs what it leaves: a
call on a legal map never leaves a new illegal step. Only a block whose own
tiles hold such a step, which the targets keep, can still carry one.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import numpy as np
from AoE2ScenarioParser.helper.maffs import sign
from AoE2ScenarioParser.objects.managers.map_manager import MapManager

from descape.iso_geometry import MAX_ELEVATION, MIN_ELEVATION

# Stands in for +/-inf in cone_clamp()'s int64 arrays.
_UNBOUNDED = 1 << 40

# The four 8-neighbour pair directions, as (slice of a tile, slice of its neighbour).
_PAIR_SLICES = (
    ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None))),
    ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1))),
)


# Adapted from AoE2ScenarioParser (https://github.com/KSneijders/AoE2ScenarioParser),
# v0.8.3 (commit b763e2e3; unchanged through v0.9.4), AoE2ScenarioParser/objects/managers/map_manager.py,
# MapManager._elevation_tile_recursion. GPL-3.0, the same license as DEscape;
# copyright for the original remains with the AoE2ScenarioParser authors.
# Changed only to take `mm` explicitly, add every written index to `touched`,
# call the optional `before_write(index)` just before each write, and append
# each gap-filled index to the optional `gap_filled`.
def _elevation_tile_recursion(
    mm: MapManager,
    source_tile,
    xys,
    touched: set[int],
    visited=None,
    before_write: Callable[[int], None] | None = None,
    gap_filled: list[int] | None = None,
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
                if gap_filled is not None:
                    gap_filled.append(i)
            elif abs(other.elevation - source_tile.elevation) > 1:
                if before_write is not None:
                    before_write(i)
                other.elevation = source_tile.elevation + int(sign(other.elevation, source_tile.elevation))
                touched.add(i)
                _elevation_tile_recursion(mm, other, xys, touched, visited, before_write, gap_filled)


def _propagate(
    mm: MapManager,
    targets: Sequence[tuple[int, int, int]],
    touched: set[int],
    before_write: Callable[[int], None] | None = None,
    gap_filled: list[int] | None = None,
) -> None:
    """MapManager.set_elevation()'s multi-tile branch for an arbitrary
    footprint, with no repair: assigns every target, then runs one recursion
    per target with the whole footprint as `xys`."""
    footprint = {(x, y) for x, y, _ in targets}
    size = mm.map_size
    touched.update(y * size + x for x, y in footprint)
    tiles = []
    for x, y, elevation in targets:
        tile = mm.get_tile(x, y)
        tile.elevation = elevation
        tiles.append(tile)
    for tile in tiles:
        _elevation_tile_recursion(mm, tile, footprint, touched, before_write=before_write, gap_filled=gap_filled)


def _cone_floor(fixed_mask: np.ndarray, fixed_value: np.ndarray, radius: int) -> np.ndarray:
    # max(v - d) over the fixed tiles within `radius`: one 3x3 dilation per pass, stepping down 1.
    floor = np.where(fixed_mask, fixed_value, -_UNBOUNDED)
    for _ in range(radius):
        wide = floor.copy()
        wide[:, 1:] = np.maximum(wide[:, 1:], floor[:, :-1])
        wide[:, :-1] = np.maximum(wide[:, :-1], floor[:, 1:])
        tall = wide.copy()
        tall[1:] = np.maximum(tall[1:], wide[:-1])
        tall[:-1] = np.maximum(tall[:-1], wide[1:])
        floor = np.maximum(floor, tall - 1)
    return floor


def _cone_bounds(fixed_mask: np.ndarray, fixed_value: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    return _cone_floor(fixed_mask, fixed_value, radius), -_cone_floor(fixed_mask, -fixed_value, radius)


def cone_clamp(base: np.ndarray, fixed_mask: np.ndarray, fixed_value: np.ndarray, radius: int) -> np.ndarray:
    """`base` clamped into the cones (Chebyshev slope 1, out to `radius`)
    around the fixed tiles, which take their value. Legal when `base` is and
    the fixed values are consistent; where they conflict the upper cone wins.
    Tiles outside the arrays do not exist here: the caller sizes the window."""
    floor, ceiling = _cone_bounds(fixed_mask, fixed_value, radius)
    return np.where(fixed_mask, fixed_value, np.minimum(np.maximum(base, floor), ceiling))


def _read_window(
    mm: MapManager, box: tuple[int, int, int, int], margin: int, pre_call: dict[int, int]
) -> tuple[int, int, np.ndarray, np.ndarray]:
    # `box` widened by `margin` and clipped to the map: its corner, its elevations, and those before the call.
    size = mm.map_size
    x0, y0 = max(0, box[0] - margin), max(0, box[1] - margin)
    x1, y1 = min(size, box[2] + margin), min(size, box[3] + margin)
    terrain = mm.terrain
    now = np.array(
        [[t.elevation for t in terrain[y * size + x0 : y * size + x1]] for y in range(y0, y1)], dtype=np.int64
    )
    before = now.copy()
    for i, elevation in pre_call.items():
        y, x = divmod(i, size)
        before[y - y0, x - x0] = elevation
    return x0, y0, now, before


def _has_new_step(now: np.ndarray, before: np.ndarray) -> bool:
    return any(
        np.any((np.abs(now[a] - now[b]) > 1) & (np.abs(before[a] - before[b]) <= 1)) for a, b in _PAIR_SLICES
    )


def _repair_new_steps(
    mm: MapManager,
    targets: Sequence[tuple[int, int, int]],
    touched: set[int],
    gap_filled: list[int],
    pre_call: dict[int, int],
    before_write: Callable[[int], None] | None,
) -> None:
    # Re-derives the call's result as a cone clamp of the pre-call map from the targets and the
    # gap fills consistent with them. Every pre_call key is in `touched`, so in the window.
    size = mm.map_size
    rows, columns = zip(*(divmod(i, size) for i in touched), strict=True)
    box = (min(columns), min(rows), max(columns) + 1, max(rows) + 1)
    _x0, _y0, now, before = _read_window(mm, box, 1, pre_call)
    if not _has_new_step(now, before):
        return

    # Past `radius` from every fixed tile the cones are looser than any window value, so the
    # clamp changes nothing there. Out-of-range files can need more than the legal range.
    radius = MAX_ELEVATION - MIN_ELEVATION
    x0, y0, now, before = _read_window(mm, box, radius, pre_call)
    spread = int(max(now.max(), before.max()) - min(now.min(), before.min()))
    if spread > radius:
        radius = spread
        x0, y0, now, before = _read_window(mm, box, radius, pre_call)

    fixed_mask = np.zeros(now.shape, dtype=bool)
    for x, y, _ in targets:
        fixed_mask[y - y0, x - x0] = True
    floor, ceiling = _cone_bounds(fixed_mask, now, radius)
    kept: list[tuple[int, int, int]] = []
    for i in dict.fromkeys(gap_filled):
        y, x = divmod(i, size)
        row, column = y - y0, x - x0
        value = int(now[row, column])
        if floor[row, column] <= value <= ceiling[row, column] and all(
            abs(value - other) <= max(abs(x - other_x), abs(y - other_y)) for other_x, other_y, other in kept
        ):
            kept.append((x, y, value))
            fixed_mask[row, column] = True

    clamped = cone_clamp(before, fixed_mask, now, radius)
    terrain = mm.terrain
    for row, column in np.argwhere(clamped != now).tolist():
        i = (row + y0) * size + column + x0
        if before_write is not None:
            before_write(i)
        terrain[i].elevation = int(clamped[row, column])
        touched.add(i)


def set_tile_elevation(mm: MapManager, x: int, y: int, elevation: int) -> set[int]:
    """Sets tile (x, y)'s own elevation to `elevation` and propagates to
    neighbors like the in-game brush / MapManager.set_elevation's multi-tile
    branch does, then repairs any illegal step the gap fill left
    (set_tiles_elevation). Returns every flat index written, the target
    included. Requires mm.map_width == mm.map_height (see
    LoadedScenario.map_is_square) -- MapManager.get_tile raises otherwise,
    same constraint set_elevation itself has."""
    return set_tiles_elevation(mm, [(x, y, elevation)])


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

    Never leaves a new illegal step on a legal map, unless `targets` hold one
    between themselves. The gap-fill branch can (see the module docstring),
    so when a gap fill left one, the result is redone as a cone clamp of the
    pre-call map from the targets and the gap fills consistent with them.
    Otherwise the propagation's result stands untouched: the gap fill stays
    because it is the in-game brush's shape, pinned by
    tests/test_elevation_tools.py's _GAP_FILL_EXTRA.

    Returns every flat index written: all targets (even ones already at
    their value) plus every propagated or repaired tile. Requires
    mm.map_width == mm.map_height, same as set_tile_elevation().

    `before_write`, when given, is called with a propagated or repaired
    tile's flat index just before each write to it (possibly more than once
    per tile), so a scoped EditHistory stroke can capture it. It is not
    called for the targets: those are the caller's own write set, known up
    front."""
    size = mm.map_size
    pre_call = {y * size + x: mm.get_tile(x, y).elevation for x, y, _ in targets}
    terrain = mm.terrain

    def record(i: int) -> None:
        pre_call.setdefault(i, terrain[i].elevation)
        if before_write is not None:
            before_write(i)

    touched: set[int] = set()
    gap_filled: list[int] = []
    _propagate(mm, targets, touched, record, gap_filled)
    if gap_filled:
        _repair_new_steps(mm, targets, touched, gap_filled, pre_call, before_write)
    return touched
