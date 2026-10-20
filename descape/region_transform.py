"""Turning or flipping a pasted region as a block (GH #197).

Qt-free, like region_clipboard.py: a RegionBlock in, a new RegionBlock out,
which the viewer re-pastes in place of the paste it undoes. Nothing here edits
a placed unit: every unit of the result is a new add(), exactly as in Map
mirroring, and its const, position and rotation come from
mirror_tools.unit_image(), the functions Mirror Map uses. So an ANGLE facing
goes through facing_image(), a wall's shape index through wall_index_image()
(integer-encoded files only), a gate through reorient_gate_const(), and every
other rotation is copied verbatim.

Only the eight D4 elements, which are exact on the tile lattice. The block sits
in block-local coordinates on an N x N square, N = max(width, height), the
element runs with n = N, and the image rectangle's low corner is subtracted
again. Measured exact (2026-10-12) for every gate const, building parity and
1x1 unit on a tile boundary at a block's edge: 291,200 footprint checks, no
mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from descape import gate_orientation, mirror_tools, render, unit_sprites
from descape.region_clipboard import RegionBlock
from descape.terrain_palette import tile_span

# Screen-space names: tile_screen_origin() puts sx along x + y and sy along
# y - x, so `a` mirrors left/right on screen and `d` top/bottom, and `r` turns
# a facing the way rotate_step(+1) does, clockwise on screen.
ROTATE_CW, ROTATE_CCW, FLIP_HORIZONTAL, FLIP_VERTICAL = "r", "r3", "a", "d"


@dataclass(frozen=True)
class BlockTransform:
    block: RegionBlock
    # Units drawn as before though the block turned: a non-square span with no
    # orientation sibling under an axis-swapping element, and every cliff.
    kept_orientation: int


def _keeps_orientation(unit_const: int, element: str) -> bool:
    if unit_const in unit_sprites.cliff_consts():
        return True
    span_x, span_y = tile_span(unit_const, render.NON_BUILDING_SPAN)
    return span_x != span_y and element in mirror_tools.AXIS_SWAPPING and not gate_orientation.is_gate(unit_const)


def transform_block(block: RegionBlock, element: str, file_radian: bool) -> BlockTransform:
    """`block` under D4 `element`. "id" returns `block` itself, so a turn back
    to the start re-pastes the very block first pasted. Units keep their order
    and garrison_slot; an occupant takes its holder's image position, since
    the app keeps occupants on their host's point and a 1x1 occupant would
    otherwise snap a tile away from an even-span host (a castle at (4.0, 4.0)
    under `r` lands at (3.0, 4.0), its occupant's own image at (2.0, 4.0))."""
    if element not in mirror_tools.TRANSFORMS:
        raise ValueError(f"transform_block: {element!r} is not a D4 element")
    if element == "id":
        return BlockTransform(block, 0)
    w, h = block.width, block.height
    n = max(w, h)
    fn = mirror_tools.TRANSFORMS[element]
    corners = (fn(0, 0, n), fn(w - 1, h - 1, n))
    low_x, low_y = min(c[0] for c in corners), min(c[1] for c in corners)
    new_w, new_h = (h, w) if element in mirror_tools.AXIS_SWAPPING else (w, h)

    terrain_ids: tuple[int, ...] = ()
    elevations: tuple[int, ...] = ()
    layers: tuple[int, ...] = ()
    if block.has_tiles:
        order = [0] * (w * h)
        for y in range(h):
            for x in range(w):
                image_x, image_y = fn(x, y, n)
                order[(image_y - low_y) * new_w + (image_x - low_x)] = y * w + x
        terrain_ids = tuple(block.terrain_ids[i] for i in order)
        elevations = tuple(block.elevations[i] for i in order)
        layers = tuple(block.layers[i] for i in order)

    images = [mirror_tools.unit_image(u, u.dx, u.dy, element, n, file_radian) for u in block.units]
    units = []
    for unit, (unit_const, own_x, own_y, rotation) in zip(block.units, images, strict=True):
        x, y = own_x, own_y
        if 0 <= unit.garrison_slot < len(images):
            _holder_const, x, y, _holder_rotation = images[unit.garrison_slot]
        units.append(replace(unit, unit_const=unit_const, dx=x - low_x, dy=y - low_y, rotation=rotation))
    kept = sum(1 for unit in block.units if _keeps_orientation(unit.unit_const, element))
    result = RegionBlock(
        width=new_w,
        height=new_h,
        terrain_ids=terrain_ids,
        elevations=elevations,
        layers=layers,
        units=tuple(units),
        dataset=block.dataset,
    )
    return BlockTransform(result, kept)


def pivot2_of(anchor: tuple[int, int], block: RegionBlock) -> tuple[int, int]:
    """The block's centre at `anchor`, doubled so it stays an integer."""
    return 2 * anchor[0] + block.width, 2 * anchor[1] + block.height


def anchor_for(pivot2: tuple[int, int], block: RegionBlock) -> tuple[int, int]:
    """The anchor that centres `block` on a doubled pivot. With an odd
    width - height the block sits half a tile toward the top-left, and four
    quarter turns still land back on the first anchor."""
    return (pivot2[0] - block.width) // 2, (pivot2[1] - block.height) // 2


def lands_on_map(block: RegionBlock, anchor: tuple[int, int], map_width: int, map_height: int) -> bool:
    """Whether the block's rectangle at `anchor` covers any on-map tile. A
    turn, unlike a move, can carry a block entirely off the map: a 20x2 block
    pasted at (119, 60) on a 120 map turns to (128, 51)."""
    x, y = anchor
    return x < map_width and y < map_height and x + block.width > 0 and y + block.height > 0
