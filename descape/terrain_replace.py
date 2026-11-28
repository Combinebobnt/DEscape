"""The Replace Brush's match and write: paint a terrain only over the tiles that
match the one the stroke was pressed on. Qt-free.

A tile matches on its stored data, never on how it renders: terrain blending
draws a neighbour's terrain over a tile's edge, so matching on the picture
would make a tile match because of what is next to it. The relation is Paint
Can's (`fill_tools.fill_indices()`): on the Terrain target the base
`terrain_id` alone, on Layer and Clear layer the `(terrain_id, layer)` pair,
so the two tools never disagree about what "the same terrain" is. Global
Flood needs no mask taken at press: the only tiles a stroke changes are ones
it wrote, and a stroke never writes a tile twice, so checking each tile as it
is written paints exactly what a press-time mask would.

Duck-typed on `mm.map_width`, `map_height`, `terrain` and each tile's
`terrain_id`/`layer`, like `fill_tools`, and indexes `mm.terrain` directly for
the same reason (`get_tile()` raises on a non-square map). `batch_api` must
not import this module back.
"""

from __future__ import annotations

from dataclasses import dataclass

from descape.fill_tools import contiguous_region
from descape.terrain_target import CLEAR, LAYER, TERRAIN, write_tile

FLOOD_GLOBAL, FLOOD_CONTIGUOUS = "global", "contiguous"
FLOODS = (FLOOD_GLOBAL, FLOOD_CONTIGUOUS)


def reference(tile) -> tuple[int, int]:
    """The press tile's `(terrain_id, layer)`, the stroke's reference."""
    return (tile.terrain_id, tile.layer)


def matches(tile, ref: tuple[int, int], target: str) -> bool:
    """Whether `tile` is "the same terrain" as `ref` for `target`."""
    if target == TERRAIN:
        return tile.terrain_id == ref[0]
    if target in (LAYER, CLEAR):
        return (tile.terrain_id, tile.layer) == ref
    raise ValueError(f"unknown terrain target {target!r}")


def contiguous(mm, x: int, y: int, target: str) -> frozenset[int]:
    """Flat indices of the matching region connected to (x, y); empty off the map."""
    return frozenset(contiguous_region(mm, x, y, match_layer=target != TERRAIN))


def replace_tiles(mm, tiles, ref, target: str, terrain_id: int, region=None) -> list[int]:
    """Writes `terrain_id` for `target` over each matching `(x, y)` of `tiles`,
    in order, and returns the flat indices it changed. Skips an off-map tile,
    one that does not match `ref` (None matches nothing), one outside `region`
    when given, and one the write would leave as it was."""
    if target not in (TERRAIN, LAYER, CLEAR):
        raise ValueError(f"unknown terrain target {target!r}")
    if ref is None:
        return []
    width, height = mm.map_width, mm.map_height
    terrain = mm.terrain
    changed: list[int] = []
    for x, y in tiles:
        if not (0 <= x < width and 0 <= y < height):
            continue
        index = y * width + x
        if region is not None and index not in region:
            continue
        tile = terrain[index]
        if not matches(tile, ref, target):
            continue
        before = (tile.terrain_id, tile.layer)
        write_tile(tile, target, terrain_id)
        if (tile.terrain_id, tile.layer) != before:
            changed.append(index)
    return changed


@dataclass
class ReplaceStroke:
    """One stroke's rule: the Flood read at its start, then the reference and
    the Contiguous region taken once, from the press tile (`press()`)."""

    flood: str
    pressed: bool = False
    ref: tuple[int, int] | None = None
    region: frozenset[int] | None = None


def press(stroke: ReplaceStroke, mm, x: int, y: int, target: str) -> None:
    """The stroke's first touch, at the press tile. Off the map (Flat accepts a
    press on the map rect's far edge, x == map_width, whose flat index would
    wrap to the next row) the reference stays None, which matches nothing."""
    stroke.pressed = True
    if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
        return
    stroke.ref = reference(mm.terrain[y * mm.map_width + x])
    if stroke.flood == FLOOD_CONTIGUOUS:
        stroke.region = contiguous(mm, x, y, target)
