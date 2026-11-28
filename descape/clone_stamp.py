"""The Clone Stamp's source, its offset and a stroke's elevation seam. Qt-free.

A Ctrl+click sets the anchor (`CloneSource.set_anchor`); the first accepted
plain press after it fixes `offset = press - anchor`, which every later
stroke keeps until the next Ctrl+click. A stroke copies each footprint
tile's source, `tile - offset`, from the map as it was at the press (the
stroke's begin_stroke() snapshot), never live, so a stroke whose source and
destination overlap cannot copy its own output forward. A footprint tile
whose source is off the map is skipped, never wrapped or clamped.

Terrain and layer copy verbatim as a pair, as `region_clipboard.paste_terrain()`
does. Elevation is raw-assigned on the cloned tiles, and every other tile is
clamped into the Chebyshev cones of the cloned set K: for a tile t not in K,
`L(t) = max(e_k - d(k, t))`, `U(t) = min(e_k + d(k, t))` over k in K, and the
new value is `min(max(e0(t), L), U)`, e0 being the press-time map. "No
8-neighbour step above 1" is exactly "1-Lipschitz in Chebyshev distance",
and a clamp of three 1-Lipschitz functions is one, so a legal source and a
legal map leave no new illegal pair, while every cloned tile keeps its
source value. The clamp is `elevation_tools.cone_clamp()`, not
`set_tiles_elevation()`, whose gap fill would rewrite tiles an earlier dab
cloned. Bounds only tighten as K grows, so a clamp per dab equals one at
the end. Where the source already holds an illegal pair the cones conflict
and the upper one wins, so new pairs can appear within the radius of a
copied one; `CloneStroke.violations()` reports them.

Duck-typed on `mm.map_width`, `map_height` and `terrain` (each tile's
`terrain_id`, `layer`, `elevation`), and indexes `mm.terrain` flat:
`MapManager.get_tile()` raises on a non-square map.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from descape.elevation_tools import cone_clamp
from descape.iso_geometry import MAX_ELEVATION, MIN_ELEVATION

Tile = tuple[int, int]
# (x, y, terrain_id, layer, elevation): a destination tile and what its source held at the press.
Target = tuple[int, int, int, int, int]


@dataclass
class CloneSource:
    """The anchor a Ctrl+click set and the offset the first press after it fixed."""

    anchor: Tile | None = None
    offset: Tile | None = None

    def set_anchor(self, tile: Tile) -> None:
        self.anchor = (int(tile[0]), int(tile[1]))
        self.offset = None

    def fix_offset(self, press: Tile) -> None:
        """Fixes the offset from `press`; a no-op once fixed or with no anchor."""
        if self.anchor is not None and self.offset is None:
            self.offset = (press[0] - self.anchor[0], press[1] - self.anchor[1])

    def source_of(self, tile: Tile) -> Tile | None:
        if self.offset is None:
            return None
        return (tile[0] - self.offset[0], tile[1] - self.offset[1])

    def marker_origin(self, hover: Tile | None) -> Tile | None:
        """Where the next dab samples from: the anchor until the offset is
        fixed, then `hover - offset` (None with no hover)."""
        if self.anchor is None:
            return None
        if self.offset is None:
            return self.anchor
        return None if hover is None else self.source_of(hover)

    def clear(self) -> None:
        self.anchor = None
        self.offset = None


def clone_targets(
    footprint: Iterable[Tile], offset: Tile, start_state: Callable[[int], tuple[int, int, int]], width: int, height: int
) -> list[Target]:
    """Each on-map footprint tile whose source is on the map, with the source's
    `(terrain_id, layer, elevation)` as `start_state(flat index)` reports it
    (`EditHistory.stroke_start_state`'s `(terrain_id, elevation, layer)`)."""
    dx, dy = offset
    targets = []
    for x, y in footprint:
        sx, sy = x - dx, y - dy
        if not (0 <= x < width and 0 <= y < height and 0 <= sx < width and 0 <= sy < height):
            continue
        terrain_id, elevation, layer = start_state(sy * width + sx)
        targets.append((x, y, terrain_id, layer, elevation))
    return targets


class CloneStroke:
    """One elevation-carrying stroke's clamp state: the press-time elevations
    `e0` (a `(height, width)` grid), the cloned mask and values, and the
    elevations the stroke has written so far (`current`)."""

    def __init__(self, e0: np.ndarray) -> None:
        self.e0 = np.asarray(e0, dtype=np.int64)
        self.height, self.width = self.e0.shape
        self.cloned = np.zeros(self.e0.shape, dtype=bool)
        self.cloned_value = np.zeros(self.e0.shape, dtype=np.int64)
        self.current = self.e0.copy()
        # A cloned tile binds a tile only nearer than |e_k - e0(t)|, and every cloned value is an e0 value,
        # so e0's spread bounds the reach. At least the legal range; more where a file holds out-of-range values.
        spread = int(self.e0.max() - self.e0.min()) if self.e0.size else 0
        self.radius = max(MAX_ELEVATION - MIN_ELEVATION, spread)

    @classmethod
    def from_tiles(cls, terrain, width: int, height: int) -> CloneStroke:
        """Built from the map's tiles, read right after begin_stroke() (so equal to its snapshot)."""
        e0 = np.fromiter((t.elevation for t in terrain), dtype=np.int64, count=width * height)
        return cls(e0.reshape(height, width))

    def add(self, targets: Iterable[Target]) -> None:
        for x, y, _terrain_id, _layer, elevation in targets:
            self.cloned[y, x] = True
            self.cloned_value[y, x] = elevation
            self.current[y, x] = elevation

    def _box(self, box: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = box
        return max(0, x0 - margin), max(0, y0 - margin), min(self.width, x1 + margin), min(self.height, y1 + margin)

    def relax(self, box: tuple[int, int, int, int]) -> dict[int, int]:
        """`{flat index: elevation}` for every non-cloned tile within the radius
        of `box` (x0, y0, x1, y1, exclusive) whose clamped value differs from
        `current`. Reads bounds from `box` plus twice the radius, the reach of
        any cloned tile that can bind a tile in the kept window."""
        r = self.radius
        rx0, ry0, rx1, ry1 = self._box(box, 2 * r)
        kx0, ky0, kx1, ky1 = self._box(box, r)
        read = (slice(ry0, ry1), slice(rx0, rx1))
        clamped = cone_clamp(self.e0[read], self.cloned[read], self.cloned_value[read], r)
        kept = clamped[ky0 - ry0 : ky1 - ry0, kx0 - rx0 : kx1 - rx0]
        changed = (kept != self.current[ky0:ky1, kx0:kx1]) & ~self.cloned[ky0:ky1, kx0:kx1]
        return {
            (row + ky0) * self.width + column + kx0: int(kept[row, column])
            for row, column in np.argwhere(changed).tolist()
        }

    def violations(self, indices: Iterable[int]) -> list[tuple[int, int]]:
        """Every 8-neighbour pair `(a, b)`, a < b, more than one level apart in
        `current` with at least one end in `indices`."""
        width, height = self.width, self.height
        found: set[tuple[int, int]] = set()
        flat = self.current.ravel()
        for i in indices:
            y, x = divmod(i, width)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if (dx or dy) and 0 <= nx < width and 0 <= ny < height:
                        j = ny * width + nx
                        if abs(int(flat[i]) - int(flat[j])) > 1:
                            found.add((min(i, j), max(i, j)))
        return sorted(found)


def apply_clone(
    mm, stroke: CloneStroke | None, targets: list[Target], *, terrain: bool, elevation: bool, carry_layer: bool
) -> tuple[list[Tile], set[int]]:
    """Writes one dab: each target's terrain and layer (layer -1 without
    `carry_layer`) when `terrain`; with `elevation`, each target's elevation
    raw, then the cone clamp around them (`stroke` required). Returns the
    cloned tiles and every flat index written, the clamp's ring included."""
    width = mm.map_width
    tiles = mm.terrain
    written: set[int] = set()
    for x, y, terrain_id, layer, value in targets:
        i = y * width + x
        tile = tiles[i]
        if terrain:
            tile.terrain_id = terrain_id
            tile.layer = layer if carry_layer else -1
        if elevation:
            tile.elevation = value
        written.add(i)
    if elevation and targets:
        stroke.add(targets)
        xs = [t[0] for t in targets]
        ys = [t[1] for t in targets]
        for i, value in stroke.relax((min(xs), min(ys), max(xs) + 1, max(ys) + 1)).items():
            tiles[i].elevation = value
            stroke.current.flat[i] = value
            written.add(i)
    return [(t[0], t[1]) for t in targets], written
