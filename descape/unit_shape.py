"""Unit Line and Unit Shape (GH #365): which cells one drag fills, and which
unit stands in each. Qt-free; MapView previews the cells, the viewer commits
them.

A press reads the pick and the tool options once into a ShapeLatch
(latch_for()), held for the whole drag. A wall pick takes the wall branch:
cells are tiles, and the viewer commits them as one wall run, so the wall
shapes come from wall_run like Place Unit's run. Any other pick takes the
unit branch: the drag box is cut into a lattice of cells, each the largest
footprint over the pick plus Spacing, and the shape is rasterized in that
lattice (lattice_cells()). Each cell gets one candidate, centred in its
cell (plan_candidates()); unit_brush.place_candidate() then applies the
rules and the rotation, so this module writes no rotation of its own.

The lattice is box-local: the anchor's cell is local (0, 0) and each axis
is mirrored by the drag direction, so a shape's "base on the anchor's edge"
or "right angle at the anchor" needs no per-quadrant case, and every
footprint stays inside the dragged box (Shift's snaps can grow the box).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from descape import gate_orientation, unit_brush, unit_kind, unit_sprites
from descape.place_rotation import PlaceRotation
from descape.render import NON_BUILDING_SPAN
from descape.shape_tools import (
    RIGHT_TRIANGLE,
    TRIANGLE,
    ellipse_tiles,
    line_tiles,
    rect_tiles,
    snap_line_delta,
    snap_square_delta,
    snap_triangle_delta,
    snap_wall_delta,
    triangle_tiles,
    wall_path_tiles,
)
from descape.terrain_palette import tile_span

WALL = "wall"
UNITS = "units"

LINE = "line"
RECTANGLE = "rectangle"
ELLIPSE = "ellipse"
# Unit Shape's Shape combo, in order: (option value, label).
SHAPE_KINDS: tuple[tuple[str, str], ...] = (
    (RECTANGLE, "Rectangle"),
    (ELLIPSE, "Ellipse"),
    (TRIANGLE, "Triangle"),
    (RIGHT_TRIANGLE, "Right Triangle"),
)
SPACING_MAX = 20


@dataclass(frozen=True)
class ShapeLatch:
    """One drag's pick and options, read once at the press."""

    branch: str  # WALL or UNITS
    consts: tuple[int, ...]
    owner: int  # the chosen owner; placement_owner() still makes a GAIA-only const GAIA's
    W: int
    H: int
    spacing: int
    kind: str  # LINE, or one of SHAPE_KINDS' values
    filled: bool
    random: bool
    rules: bool
    pending: PlaceRotation
    label: str = ""  # the pick's name as the catalog showed it at the press, for a mixed pick's status line


def refusal(consts) -> str | None:
    """Why a pick cannot be drawn as a line or shape, or None."""
    consts = tuple(consts)
    if not consts:
        return "pick an object in the catalog"
    if any(gate_orientation.is_gate(const) for const in consts):
        return "gates are placed one at a time with Place Unit"
    if any(const in unit_kind.cliff_consts() for const in consts):
        return "cliffs are placed as chains with the Cliff tool"
    walls = set(unit_sprites.wall_family_consts())
    picked_walls = {const for const in consts if const in walls}
    if picked_walls and len(picked_walls) < len(set(consts)):
        return "pick walls alone"
    if len(picked_walls) > 1:
        return "pick one wall"
    return None


def latch_for(
    consts,
    owner: int,
    kind: str,
    *,
    filled: bool,
    spacing: int,
    randomize: bool,
    rules: bool,
    pending: PlaceRotation,
) -> ShapeLatch:
    """The latch for a pick refusal() accepts. A wall pick drops Filled and
    Spacing: a wall shape is always its one-tile outline."""
    consts = tuple(dict.fromkeys(consts))
    reason = refusal(consts)
    if reason is not None:
        raise ValueError(reason)
    if consts[0] in unit_sprites.wall_family_consts():
        return ShapeLatch(WALL, consts, owner, 1, 1, 0, kind, False, randomize, rules, pending)
    spans = [tile_span(const, NON_BUILDING_SPAN) for const in consts]
    return ShapeLatch(
        UNITS,
        consts,
        owner,
        max(w for w, _ in spans),
        max(h for _, h in spans),
        max(0, min(int(spacing), SPACING_MAX)),
        kind,
        filled,
        randomize,
        rules,
        pending,
    )


def _axis(a: int, e: int, size: int, spacing: int) -> tuple[int, int]:
    """(direction, cell count) on one axis for anchor `a` and end `e`."""
    d = 1 if e >= a else -1
    return d, max(1, (abs(e - a) + 1 + spacing) // (size + spacing))


def _cell_low(a: int, d: int, k: int, size: int, step: int) -> int:
    """The low tile of local cell `k` on one axis."""
    return a + k * step if d > 0 else a - k * step - (size - 1)


def _local_raster(kind: str, lx: int, ly: int, filled: bool, shift: bool) -> list[tuple[int, int]]:
    """The shape over local cells (0, 0)-(lx, ly), lx, ly >= 0."""
    if shift:
        snap = {LINE: snap_line_delta, TRIANGLE: snap_triangle_delta}.get(kind, snap_square_delta)
        lx, ly = snap(lx, ly)
    w, h = lx + 1, ly + 1
    if kind == LINE:
        return line_tiles(0, 0, lx, ly, w, h)
    if kind == RECTANGLE:
        return rect_tiles(0, 0, lx, ly, w, h, filled=filled)
    if kind == ELLIPSE:
        return ellipse_tiles(0, 0, lx, ly, w, h, filled=filled)
    if kind in (TRIANGLE, RIGHT_TRIANGLE):
        return triangle_tiles(0, 0, lx, ly, w, h, kind=kind, filled=filled)
    raise ValueError(f"unknown shape kind {kind!r}")


def lattice_cells(
    latch: ShapeLatch,
    anchor: tuple[int, int],
    end: tuple[int, int],
    shift: bool,
    map_w: int,
    map_h: int,
) -> list[tuple[int, int]]:
    """The drag's cells as their low-corner tiles, in raster order, less any
    cell wholly off the map. A wall-branch cell is a tile; a wall line is
    Place Unit's run (wall_path_tiles(), Shift = snap_wall_delta()). A cell
    partly off the map stays: plan_candidates()' caller drops a footprint
    that does not fit, and the preview shows its on-map tiles."""
    ax, ay = anchor
    if latch.branch == WALL and latch.kind == LINE:
        dx, dy = end[0] - ax, end[1] - ay
        if shift:
            dx, dy = snap_wall_delta(dx, dy)
        return wall_path_tiles(ax, ay, ax + dx, ay + dy, map_w, map_h)
    W, H, gap = latch.W, latch.H, latch.spacing
    dx_, nx = _axis(ax, end[0], W, gap)
    dy_, ny = _axis(ay, end[1], H, gap)
    cells = []
    for i, j in _local_raster(latch.kind, nx - 1, ny - 1, latch.filled, shift):
        ox = _cell_low(ax, dx_, i, W, W + gap)
        oy = _cell_low(ay, dy_, j, H, H + gap)
        if ox + W > 0 and oy + H > 0 and ox < map_w and oy < map_h:
            cells.append((ox, oy))
    return cells


def cell_block(latch: ShapeLatch, cell: tuple[int, int], map_w: int, map_h: int) -> list[tuple[int, int]]:
    """A cell's W x H tiles, clipped to the map: what the preview shades."""
    ox, oy = cell
    return [
        (x, y)
        for y in range(max(oy, 0), min(oy + latch.H, map_h))
        for x in range(max(ox, 0), min(ox + latch.W, map_w))
    ]


def plan_candidates(latch: ShapeLatch, cells, rng: random.Random) -> list[tuple[int, float, float]]:
    """(const, x, y) per cell: the const drawn from the pick (no draw for a
    single const, as the Unit Brush does), centred in its cell's W x H block
    the way render._span_start() reads a footprint back."""
    out = []
    for ox, oy in cells:
        const = latch.consts[0] if len(latch.consts) == 1 else rng.choice(latch.consts)
        w, h = tile_span(const, NON_BUILDING_SPAN)
        out.append((const, ox + (latch.W - w) // 2 + w / 2, oy + (latch.H - h) // 2 + h / 2))
    return out


def place_cells(scenario, latch: ShapeLatch, cells, add, rng: random.Random) -> unit_brush.BrushStroke:
    """Places the unit branch's candidates through `add` and returns the
    stroke holding what was placed and refused. A candidate whose footprint
    leaves the map is dropped; the rest go through the Unit Brush's own
    tail (unit_brush.place_candidate()), so rules, Free placement, rotation
    and the GAIA-only owner match the brush exactly."""
    if latch.branch != UNITS:
        raise ValueError("place_cells() is the unit branch; walls go through wall_run")
    stroke = unit_brush.BrushStroke(
        consts=latch.consts,
        chosen_owner=latch.owner,
        density=1.0,
        rules=latch.rules,
        randomize=latch.random,
        pending=latch.pending,
        rng=rng,
    )
    mm = scenario.map_manager
    for const, x, y in plan_candidates(latch, cells, rng):
        if unit_brush._fits(x, y, tile_span(const, NON_BUILDING_SPAN), mm.map_width, mm.map_height):
            unit_brush.place_candidate(scenario, stroke, const, x, y, add)
    return stroke
