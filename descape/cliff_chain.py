"""Track B Stage 2 of the 2026-09-02 cliffs plan: the auto-connecting chain
drag's geometry and piece resolution.

No Qt here, the same split `cliff_catalog` and `cliff_connectivity` already
use -- `viewer.py` only feeds cursor tiles in and writes the resulting nodes
out through `UnitEditModel`, so every rule below is testable headless.

**Nodes step on a lattice, not on the cursor tile.** Once the first node's
anchor and span are fixed, each subsequent node sits exactly one footprint
away (`anchor + (dx * span_x, dy * span_y)`), stepping an axis only once the
cursor has actually left the last footprint on that axis. Placing a node
wherever the cursor happened to first leave the previous one instead would
drift: on a diagonal drag with a 3x3 piece the cursor typically exits on one
axis a tile or two before the other, so `cliff_connectivity.adjacency_dir()`
would report a cardinal where a diagonal was drawn, and the dirsets would
wander off the measured table's key space. The walk is a loop rather than a
single step so a fast drag -- which skips whole tiles between mouse-move
events -- still emits the intermediate nodes instead of leaving a gap.

**One suffix (piece size) per stroke.** The plan's "Decisions" addendum says
suffix inherits from an already-placed neighbouring node; since each node's
own neighbour is the previous node in the same stroke, that resolves to a
single size for the whole run, which is what makes the span -- and so the
lattice above -- constant. The stroke's suffix is taken from an existing
map cliff adjacent to the first node if there is one, else from the Piece
picker.
"""

from __future__ import annotations

from dataclasses import dataclass

from descape import cliff_catalog, cliff_connectivity, unit_sprites
from descape.render import span_anchor, unit_tile_bounds
from descape.terrain_palette import tile_span

Bounds = tuple[int, int, int, int]
Direction = tuple[int, int]

_NON_CLIFF_SPAN = (1, 1)

# The lattice walk below advances one footprint per iteration, so this only
# has to exceed the longest run a map allows: a 1-wide piece on the largest
# corpus map is ~480 steps.
_MAX_WALK_STEPS = 512


@dataclass(frozen=True)
class ChainNode:
    """One resolved cliff the stroke will place. `rotation` doubles as
    `initial_animation_frame` -- 18,232/18,232 corpus records agree, per the
    plan's own Context section."""

    unit_const: int
    rotation: int
    x: float
    y: float
    bounds: Bounds
    # False when the node's neighbour shape wasn't in the measured table
    # (an isolated node, or one of the ~5% of shapes too rare to pin) and
    # `rotation` is the Frame picker's own value rather than a resolved
    # one. Surfaced in the status bar so a run that silently kept one frame
    # throughout is legible as a fallback, not as a resolution.
    resolved: bool


@dataclass(frozen=True)
class ExistingCliff:
    """A cliff already on the map: a neighbour for dirset purposes, a blocker
    for placement, and the source the first node inherits its suffix from."""

    unit_const: int
    bounds: Bounds


def existing_cliffs(scenario, map_width: int, map_height: int) -> list[ExistingCliff]:
    """Every cliff already placed, across all players.

    Scanned straight off `unit_manager` rather than through
    `map_view._unit_index`: that pick index is only built in Units mode (see
    `_after_unit_mutation()`'s own mode gate) and is None for the whole life
    of a Terrain-mode Cliff stroke.
    """
    consts = unit_sprites.cliff_consts()
    out: list[ExistingCliff] = []
    for units in scenario.unit_manager.units:
        for unit in units:
            if unit.unit_const not in consts:
                continue
            bounds = unit_tile_bounds(unit, map_width, map_height)
            if bounds is not None:
                out.append(ExistingCliff(unit_const=unit.unit_const, bounds=bounds))
    return out


def _overlap(a: Bounds, b: Bounds) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


class ChainStroke:
    """Accumulates a Cliff drag without mutating anything, then resolves the
    whole run at once in `nodes()`.

    Deliberately accumulate-then-commit, the shape the Convert brush already
    establishes and for the same reason: `UnitEditModel.begin_unit_edit()`
    needs the complete touched-player set up front. A node's own piece is
    also unknowable mid-drag -- its neighbour set isn't complete until the
    drag ends, since the node after it hasn't been drawn yet.
    """

    def __init__(
        self,
        *,
        map_width: int,
        map_height: int,
        existing: list[ExistingCliff],
        fallback_const: int,
        fallback_frame: int,
    ) -> None:
        self._map_width = map_width
        self._map_height = map_height
        self._existing = existing
        # The Piece/Frame picker's own values -- Stage 1's manual choice,
        # which is what an unresolvable node falls back to (the plan's
        # Decisions addendum, third rule).
        self._fallback_const = fallback_const
        self._fallback_frame = fallback_frame
        self._unit_const = fallback_const
        self._span = tile_span(fallback_const, _NON_CLIFF_SPAN)
        self._anchors: list[tuple[int, int]] = []
        # Lattice head: where the walk has reached, which is NOT always the
        # last emitted anchor -- a node overlapping an existing cliff is
        # skipped, but the run still has to continue past it.
        self._walk: tuple[int, int] | None = None

    def add_cursor_tile(self, tile_x: int, tile_y: int) -> None:
        if self._walk is None:
            self._start(tile_x, tile_y)
            return
        ax, ay = self._walk
        span_x, span_y = self._span
        for _ in range(_MAX_WALK_STEPS):
            dx = 1 if tile_x >= ax + span_x else (-1 if tile_x < ax else 0)
            dy = 1 if tile_y >= ay + span_y else (-1 if tile_y < ay else 0)
            if dx == 0 and dy == 0:
                break
            ax, ay = ax + dx * span_x, ay + dy * span_y
            if not self._fits(ax, ay):
                break
            self._walk = (ax, ay)
            self._emit(ax, ay)

    def nodes(self) -> list[ChainNode]:
        """The stroke's placements, resolved against the measured
        `cliff_connectivity` table.

        Rotation follows the plan's Decisions addendum second rule --
        inherit the previous node's face -- read as "keep it when this
        node's own candidate set allows it, else take that set's first
        candidate". Taken literally the rule would carry a straight run's
        frame onto a corner, whose candidates are a disjoint set, writing a
        shape that doesn't connect: the exact failure the tool exists to
        avoid.
        """
        anchors = self._anchors
        bounds = [self._bounds(tx, ty) for tx, ty in anchors]
        suffix = cliff_catalog.piece_suffix(self._unit_const)
        out: list[ChainNode] = []
        previous_rotation: int | None = None
        for i, (tile_x, tile_y) in enumerate(anchors):
            dirset = self._dirset(i, bounds)
            # The empty dirset deliberately never reaches the table, even
            # though it has a row there: at 43% suffix purity it is the
            # weakest entry by a wide margin, and overriding an explicit
            # picker choice with it would regress Stage 1's own click.
            #
            # rotations_for(), never resolve(): the table's top-level
            # rotations belong to whichever piece SIZE was that dirset's
            # corpus majority, and this stroke pins one size of its own.
            # Mixing the two draws a wrong-scale frame -- see
            # rotations_for()'s own docstring for the measurement.
            rotations = (
                cliff_connectivity.rotations_for(dirset, suffix)
                if dirset and suffix is not None
                else None
            )
            if not rotations:
                # No neighbours at all, a neighbour shape too rare to be
                # pinned, or that shape never seen at this piece size. All
                # take the picker's frame rather than a guess; the const
                # stays the stroke's own so a run keeps one size throughout.
                rotation = self._fallback_frame
            else:
                rotation = previous_rotation if previous_rotation in rotations else rotations[0]
            x, y = span_anchor(tile_x, tile_y, *self._span)
            out.append(
                ChainNode(
                    unit_const=self._unit_const,
                    rotation=rotation,
                    x=x,
                    y=y,
                    bounds=bounds[i],
                    resolved=bool(rotations),
                )
            )
            previous_rotation = rotation
        return out

    def _start(self, tile_x: int, tile_y: int) -> None:
        """Fixes the stroke's piece (and so its span) off the first node.

        The suffix comes from an existing map cliff already touching this
        node if there is one, else from the Piece picker. The table's own
        pinned majority -- the plan's stated second fallback -- can't be
        used here: a node's dirset isn't known until the drag ends, and the
        empty-dirset row is the weakest in the table (43% suffix purity)
        where an explicit picker choice is unambiguous.
        """
        probe = self._bounds(tile_x, tile_y, tile_span(self._fallback_const, _NON_CLIFF_SPAN))
        for cliff in self._existing:
            if cliff_connectivity.adjacency_dir(probe, cliff.bounds) is None:
                continue
            suffix = cliff_catalog.piece_suffix(cliff.unit_const)
            inherited = None if suffix is None else cliff_catalog.piece_for_suffix(self._fallback_const, suffix)
            if inherited is not None:
                self._unit_const = inherited
                self._span = tile_span(inherited, _NON_CLIFF_SPAN)
            break
        if not self._fits(tile_x, tile_y):
            return
        self._walk = (tile_x, tile_y)
        self._emit(tile_x, tile_y)

    def _emit(self, tile_x: int, tile_y: int) -> None:
        if (tile_x, tile_y) in self._anchors:
            return
        if any(_overlap(self._bounds(tile_x, tile_y), c.bounds) for c in self._existing):
            return
        self._anchors.append((tile_x, tile_y))

    def _bounds(self, tile_x: int, tile_y: int, span: tuple[int, int] | None = None) -> Bounds:
        """The tile box `render.unit_tile_bounds()` would report for a node
        anchored here. `span_anchor` puts the footprint's lowest tile at the
        anchor tile on both axes, so this is just the span read off it."""
        span_x, span_y = self._span if span is None else span
        return tile_x, tile_x + span_x, tile_y, tile_y + span_y

    def _fits(self, tile_x: int, tile_y: int) -> bool:
        """Whole footprint on-map, not merely the anchor tile: a cliff half
        off the edge would be clamped by `unit_tile_bounds()` into a box
        that no longer matches the lattice the adjacency math assumes."""
        span_x, span_y = self._span
        return (
            0 <= tile_x
            and tile_x + span_x <= self._map_width
            and 0 <= tile_y
            and tile_y + span_y <= self._map_height
        )

    def _dirset(self, index: int, bounds: list[Bounds]) -> frozenset[Direction]:
        """This node's neighbour directions, over both the stroke's own nodes
        and the cliffs already on the map. Operand order matches
        `tools/gen_cliff_connectivity.py`'s scan loop -- own box first, so a
        direction reads self -> neighbour, the sense the pinned table's keys
        were measured in."""
        own = bounds[index]
        dirset: set[Direction] = set()
        for i, other in enumerate(bounds):
            if i == index:
                continue
            direction = cliff_connectivity.adjacency_dir(own, other)
            if direction is not None:
                dirset.add(direction)
        for cliff in self._existing:
            direction = cliff_connectivity.adjacency_dir(own, cliff.bounds)
            if direction is not None:
                dirset.add(direction)
        return frozenset(dirset)
