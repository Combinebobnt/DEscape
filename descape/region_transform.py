"""Turning or flipping a pasted region as a block (GH #197), and a unit
selection or Select region in place (TASK-256).

Qt-free, like region_clipboard.py, and nothing here writes. transform_block()
takes a RegionBlock and returns a new one, which the viewer re-pastes in place
of the paste it undoes, so every unit of the result is a new add(), exactly as
in Map mirroring. plan_in_place() plans the same images for units that already
exist, which the viewer writes through UnitEditModel's guarded setters, keeping
each unit's reference_id. Both take each unit's const, position and rotation
from mirror_tools.unit_image(), the functions Mirror Map uses. So an ANGLE
facing goes through facing_image(), a wall's shape index through
wall_index_image() (integer-encoded files only), a gate through
reorient_gate_const(), and every other rotation is copied verbatim.

Only the eight D4 elements, which are exact on the tile lattice. The block sits
in block-local coordinates on an N x N square, N = max(width, height), the
element runs with n = N, and the image rectangle's low corner is subtracted
again. Measured exact (2026-10-12) for every gate const, building parity and
1x1 unit on a tile boundary at a block's edge: 291,200 footprint checks, no
mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

from descape import gate_orientation, mirror_tools, render, unit_rotation, unit_sprites
from descape.region_clipboard import RegionBlock
from descape.terrain_palette import tile_span

# Screen-space names: tile_screen_origin() puts sx along x + y and sy along
# y - x, so `a` mirrors left/right on screen and `d` top/bottom, and `r` turns
# a facing the way rotate_step(+1) does, clockwise on screen.
ROTATE_CW, ROTATE_CCW, FLIP_HORIZONTAL, FLIP_VERTICAL = "r", "r3", "a", "d"
# Top-down Flat draws x right and y down, so there the same two flips are mx and my.
_TOP_DOWN_FLIPS = {FLIP_HORIZONTAL: "mx", FLIP_VERTICAL: "my"}


def flip_element(direction: str, top_down: bool) -> str:
    """The D4 element that mirrors the screen in flip `direction`
    (FLIP_HORIZONTAL or FLIP_VERTICAL): the direction itself in the
    isometric views, mx or my in top-down Flat. Rotations need no such map:
    the views differ by a rotation, which keeps the turn sense."""
    if direction not in _TOP_DOWN_FLIPS:
        raise ValueError(f"flip_element: {direction!r} is not a flip direction")
    return _TOP_DOWN_FLIPS[direction] if top_down else direction


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


def _frame(element: str, w: int, h: int) -> tuple[int, int, int, int, int]:
    """(n, low_x, low_y, new_w, new_h): the padded square's size, the image
    rectangle's low corner on it, and the image's size."""
    n = max(w, h)
    fn = mirror_tools.TRANSFORMS[element]
    corners = (fn(0, 0, n), fn(w - 1, h - 1, n))
    low_x, low_y = min(c[0] for c in corners), min(c[1] for c in corners)
    new_w, new_h = (h, w) if element in mirror_tools.AXIS_SWAPPING else (w, h)
    return n, low_x, low_y, new_w, new_h


def unit_images(units, element: str, w: int, h: int, file_radian: bool) -> list[tuple[int, float, float, float]]:
    """(const, x, y, rotation) of each unit's image under `element`, in the
    image rectangle's own coordinates, for a w x h block. `units` carry
    unit_const, rotation, a block-local dx/dy and a garrison_slot (a
    RegionUnit, or a LocalUnit for a placed one). An occupant takes its
    holder's image position, since the app keeps occupants on their host's
    point and a 1x1 occupant would otherwise snap a tile away from an
    even-span host (a castle at (4.0, 4.0) under `r` lands at (3.0, 4.0), its
    occupant's own image at (2.0, 4.0))."""
    n, low_x, low_y, _new_w, _new_h = _frame(element, w, h)
    images = [mirror_tools.unit_image(u, u.dx, u.dy, element, n, file_radian) for u in units]
    result = []
    for unit, (unit_const, own_x, own_y, rotation) in zip(units, images, strict=True):
        x, y = own_x, own_y
        if 0 <= unit.garrison_slot < len(images):
            _holder_const, x, y, _holder_rotation = images[unit.garrison_slot]
        result.append((unit_const, x - low_x, y - low_y, rotation))
    return result


def transform_block(block: RegionBlock, element: str, file_radian: bool) -> BlockTransform:
    """`block` under D4 `element`. "id" returns `block` itself, so a turn back
    to the start re-pastes the very block first pasted. Units keep their order
    and garrison_slot, and take unit_images()' answer."""
    if element not in mirror_tools.TRANSFORMS:
        raise ValueError(f"transform_block: {element!r} is not a D4 element")
    if element == "id":
        return BlockTransform(block, 0)
    w, h = block.width, block.height
    n, low_x, low_y, new_w, new_h = _frame(element, w, h)
    fn = mirror_tools.TRANSFORMS[element]

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

    images = unit_images(block.units, element, w, h, file_radian)
    units = [
        replace(unit, unit_const=unit_const, dx=x, dy=y, rotation=rotation)
        for unit, (unit_const, x, y, rotation) in zip(block.units, images, strict=True)
    ]
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
    return _centred(pivot2, block.width, block.height)


def _centred(pivot2: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    return (pivot2[0] - width) // 2, (pivot2[1] - height) // 2


class LocalUnit(NamedTuple):
    """What unit_images() reads of a placed unit, in block-local coordinates."""

    unit_const: int
    rotation: float
    dx: float
    dy: float
    garrison_slot: int


@dataclass(frozen=True)
class FirstUnit:
    """A placed unit as a selection transform first found it (TASK-256).
    `key` is the caller's handle for it, opaque here; `holder` is the index in
    the same list of the unit it is garrisoned in, or -1."""

    key: object
    unit_const: int
    x: float
    y: float
    rotation: float
    holder: int = -1


# InPlaceImage.rotation_write's two kinds: a facing for set_rotation(), a shape index for set_wall_variant().
WRITE_ANGLE, WRITE_WALL = "angle", "wall"


@dataclass(frozen=True)
class InPlaceImage:
    key: object
    unit_const: int
    x: float
    y: float
    # (WRITE_ANGLE, facing), (WRITE_WALL, index), or None to keep the rotation verbatim.
    rotation_write: tuple[str, float] | None


@dataclass(frozen=True)
class InPlacePlan:
    images: tuple[InPlaceImage, ...]
    rect: tuple[int, int, int, int]  # the image rectangle, half-open (tx0, ty0, tx1, ty1)
    block: RegionBlock | None  # the turned tiles, to write at rect's low corner; None without tiles or for id
    kept_orientation: int


def plan_in_place(
    first_units, element: str, rect: tuple[int, int, int, int], file_radian: bool, tiles: RegionBlock | None = None
) -> InPlacePlan:
    """Where `element` sends each of `first_units` (FirstUnits) when the
    half-open `rect` turns about its own centre, as the existing units it
    edits rather than new ones (TASK-256). The images are unit_images()', so
    the per-const rules are the region turn's and Mirror Map's: an ANGLE
    facing through facing_image(), a wall's shape index through
    wall_index_image() in an integer-encoded file only, a gate through
    reorient_gate_const(), every other rotation verbatim (rotation_write None).
    The image rectangle is centred on `rect`'s doubled centre by anchor_for(),
    so a caller that always plans from the same first state and the element
    composed since cannot drift. `tiles` is `rect`'s tiles-only block, turned
    into `block`. "id" plans nothing. Reads only."""
    if element not in mirror_tools.TRANSFORMS:
        raise ValueError(f"plan_in_place: {element!r} is not a D4 element")
    if element == "id":
        return InPlacePlan((), rect, None, 0)
    tx0, ty0, tx1, ty1 = rect
    w, h = tx1 - tx0, ty1 - ty0
    if tiles is not None and (tiles.width, tiles.height) != (w, h):
        raise ValueError(f"plan_in_place: a {tiles.width}x{tiles.height} tiles block for a {w}x{h} rect")
    local = [LocalUnit(u.unit_const, u.rotation, u.x - tx0, u.y - ty0, u.holder) for u in first_units]
    _n, _low_x, _low_y, new_w, new_h = _frame(element, w, h)
    ax, ay = _centred((tx0 + tx1, ty0 + ty1), new_w, new_h)
    images = tuple(
        InPlaceImage(first.key, unit_const, ax + x, ay + y, _rotation_write(first, rotation, file_radian))
        for first, (unit_const, x, y, rotation) in zip(
            first_units, unit_images(local, element, w, h, file_radian), strict=True
        )
    )
    block = None if tiles is None else transform_block(tiles, element, file_radian).block
    kept = sum(1 for first in first_units if _keeps_orientation(first.unit_const, element))
    return InPlacePlan(images, (ax, ay, ax + new_w, ay + new_h), block, kept)


def _rotation_write(first: FirstUnit, rotation: float, file_radian: bool) -> tuple[str, float] | None:
    """How a planned rotation reaches the model: ANGLE consts through
    set_rotation(), a wall whose index changed (integer-encoded file) through
    set_wall_variant(), and nothing else at all."""
    if unit_rotation.rotation_is_angle(first.unit_const):
        return (WRITE_ANGLE, rotation)
    if file_radian or not unit_sprites.rotation_variant_eligible(first.unit_const):
        return None
    if rotation == float(first.rotation):
        return None
    return (WRITE_WALL, int(rotation))


def image_on_map(unit_const: int, x: float, y: float, map_width: int, map_height: int) -> bool:
    """Whether a `unit_const` at (x, y) has its whole footprint on the map."""
    span_x, span_y = tile_span(unit_const, render.NON_BUILDING_SPAN)
    return _axis_on_map(x, span_x, map_width) and _axis_on_map(y, span_y, map_height)


def _axis_on_map(coord: float, span: int, size: int) -> bool:
    if span == 1:
        # The sign is tested on the float: int() truncates toward zero, so -0.5 would pass as tile 0.
        return coord >= 0 and int(coord) < size
    low = render._span_start(coord, span)
    return 0 <= low <= size - span


def lands_on_map(block: RegionBlock, anchor: tuple[int, int], map_width: int, map_height: int) -> bool:
    """Whether the block's rectangle at `anchor` covers any on-map tile. A
    turn, unlike a move, can carry a block entirely off the map: a 20x2 block
    pasted at (119, 60) on a 120 map turns to (128, 51)."""
    x, y = anchor
    return x < map_width and y < map_height and x + block.width > 0 and y + block.height > 0
