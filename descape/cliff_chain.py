"""The Cliff tool's auto-connecting chain drag: the path a drag traces, and
the pieces and positions that path resolves to.

No Qt here, the same split `cliff_catalog` and `cliff_connectivity` already
use: `viewer.py` only feeds cursor tiles in and writes the resulting nodes
out through `UnitEditModel`, so every rule below is testable headless.

**The path is a list of cardinal steps.** Each node joins the next one by a
shared footprint edge, never a corner, so a diagonal drag becomes a
staircase. A step is taken when the cursor leaves the head node's realized
footprint, widened to at least 3 tiles per axis so a 1-tile wobble beside a
thin piece does not turn a corner. When the cursor is out on both axes the
larger overshoot goes first. A step that would land on a node already in the
path is not taken. The walk loops, so a fast drag that skips tiles between
mouse-move events still emits every intermediate node.

**Pieces are resolved jointly per junction, not one per stroke.** A corner's
piece size is conditioned on the size of the straights it meets, so a
zig-zag's legs can change size between turns. `cliff_connectivity.
resolve_chain()` scores the whole path's suffixes and rotations over the
measured transitions, then the layout walks from the press tile, placing
each node's footprint where `cliff_connectivity.offset()` says the corpus
puts that neighbour. The path re-resolves after every step, since a new step
can change earlier nodes.

**Contacts get one bounded pass.** After the first layout, a node whose
footprint shares an edge with an existing cliff or a non-adjacent node of
the same stroke takes that direction into its dirset and that neighbour as a
fixed contact, and the path resolves and lays out once more. A node whose
final neighbours still differ from the shape it was resolved for is flagged
unresolved rather than chased further.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from descape import cliff_catalog, cliff_connectivity, unit_sprites
from descape.render import span_anchor, unit_tile_bounds
from descape.terrain_palette import tile_span

Bounds = tuple[int, int, int, int]
Direction = tuple[int, int]

_NON_CLIFF_SPAN = (1, 1)
_CARDINALS: frozenset[Direction] = frozenset({(0, -1), (0, 1), (-1, 0), (1, 0)})
# The step test's minimum box per axis around the head footprint.
_MIN_STEP_BOX = 3

# One walk call advances at most this many steps; the longest run a map
# allows is a 1-wide piece across the largest corpus map, ~480 steps.
_MAX_WALK_STEPS = 512
_PLAN_CACHE_SIZE = 4


@dataclass(frozen=True)
class ChainNode:
    """One cliff the stroke will place. `rotation` doubles as
    `initial_animation_frame` (18,232/18,232 corpus records agree)."""

    unit_const: int
    rotation: int
    x: float
    y: float
    bounds: Bounds
    # False when the node's piece or position is not backed by the measured
    # table: no candidate (it keeps the picked piece and frame), a pair or
    # shape under the table's floor, a neighbour dropped next to it, or final
    # neighbours that differ from the shape it was resolved for.
    resolved: bool
    # The neighbour shape it was resolved for (path plus contacts), and its
    # position on the path: a dropped node leaves a gap in `index`.
    dirset: frozenset[Direction] = frozenset()
    index: int = 0


@dataclass(frozen=True)
class ExistingCliff:
    """A cliff already on the map: a fixed contact and a placement blocker."""

    unit_const: int
    bounds: Bounds


def existing_cliffs(scenario, map_width: int, map_height: int) -> list[ExistingCliff]:
    """Every cliff already placed, across all players.

    Scanned straight off `unit_manager` rather than through
    `map_view._unit_index`: that pick index is current or None outside Units
    mode (see `_after_unit_mutation()`'s own mode gate). It may survive
    there, but a Terrain-mode Cliff stroke cannot count on it being built.
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


def _opposite(d: Direction) -> Direction:
    return -d[0], -d[1]


def _tiles(bounds: Bounds):
    x0, x1, y0, y1 = bounds
    for x in range(x0, x1):
        for y in range(y0, y1):
            yield x, y


def _ring(bounds: Bounds):
    """The tiles touching `bounds` by an edge or a corner, from outside."""
    x0, x1, y0, y1 = bounds
    for x in range(x0 - 1, x1 + 1):
        yield x, y0 - 1
        yield x, y1
    for y in range(y0, y1):
        yield x0 - 1, y
        yield x1, y


class _TileIndex:
    """Tile -> owner keys, so neighbour and overlap probes cost a footprint's
    ring rather than a scan over every other box."""

    def __init__(self) -> None:
        self._owners: dict[tuple[int, int], list] = {}

    def add(self, key, bounds: Bounds) -> None:
        for tile in _tiles(bounds):
            self._owners.setdefault(tile, []).append(key)

    def touching(self, bounds: Bounds) -> set:
        out: set = set()
        for tile in _ring(bounds):
            out.update(self._owners.get(tile, ()))
        return out

    def overlapping(self, bounds: Bounds) -> set:
        out: set = set()
        for tile in _tiles(bounds):
            out.update(self._owners.get(tile, ()))
        return out


def path_dirsets(steps: list[Direction]) -> list[frozenset[Direction]]:
    """Each node's neighbour directions along the path alone: back to the
    previous node and on to the next."""
    n = len(steps) + 1
    out = []
    for i in range(n):
        dirs = set()
        if i > 0:
            dirs.add(_opposite(steps[i - 1]))
        if i < n - 1:
            dirs.add(steps[i])
        out.append(frozenset(dirs))
    return out


def _abut(step: Direction, span_a: tuple[int, int], span_b: tuple[int, int]) -> tuple[int, int]:
    """Flush along `step`, cross-axis 0: the layout when no offset is measured."""
    dx, dy = step
    if dx:
        return (span_a[0] if dx > 0 else -span_b[0]), 0
    return 0, (span_a[1] if dy > 0 else -span_b[1])


class _Layout:
    """One resolve plus layout pass over a path. `bounds` covers every path
    node, dropped or off-map ones included: the walk's step test reads the
    head's."""

    def __init__(self, press, steps, dirsets, resolved, picked_const, picked_frame) -> None:
        self.dirsets = dirsets
        self.consts: list[int] = []
        self.rotations: list[int] = []
        self.ok: list[bool] = []
        self.suffixes: list[int] = []
        for suffix, rotation, ok in resolved:
            const = picked_const if suffix is None else cliff_catalog.piece_for_suffix(picked_const, suffix)
            fallback = const is None or rotation is None
            const = picked_const if const is None else const
            self.consts.append(const)
            self.suffixes.append(cliff_catalog.suffixes().get(const) if suffix is None else suffix)
            self.rotations.append(picked_frame if rotation is None else rotation)
            self.ok.append(ok and not fallback)
        self.spans = [tile_span(c, _NON_CLIFF_SPAN) for c in self.consts]
        lows = [press]
        for i, step in enumerate(steps):
            found = cliff_connectivity.offset(dirsets[i], self.spans[i], step, dirsets[i + 1], self.spans[i + 1])
            if found is None:
                found = _abut(step, self.spans[i], self.spans[i + 1])
                self.ok[i] = self.ok[i + 1] = False
            lows.append((lows[i][0] + found[0], lows[i][1] + found[1]))
        self.lows = lows
        self.bounds: list[Bounds] = [
            (x, x + sx, y, y + sy) for (x, y), (sx, sy) in zip(lows, self.spans, strict=True)
        ]


class ChainStroke:
    """Accumulates a Cliff drag without mutating anything; `nodes()` is what
    a release would place right now, so a preview can show it mid-drag.

    Accumulate-then-commit, the shape the Convert brush already establishes
    and for the same reason: `UnitEditModel.begin_unit_edit()` needs the
    complete touched-player set up front.
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
        # The Piece/Frame picker's values: a lone node's piece, the family
        # every resolved suffix is drawn from, and the fallback for a node
        # with no candidate.
        self._fallback_const = fallback_const
        self._fallback_frame = fallback_frame
        self._allowed = frozenset(
            s for s in range(1, 10) if cliff_catalog.piece_for_suffix(fallback_const, s) is not None
        )
        self._existing_index = _TileIndex()
        for k, cliff in enumerate(existing):
            self._existing_index.add(k, cliff.bounds)
        self._press: tuple[int, int] | None = None
        self._steps: list[Direction] = []
        self._plans: OrderedDict[tuple[Direction, ...], _Layout] = OrderedDict()
        self._nodes: tuple[tuple[Direction, ...], list[ChainNode]] | None = None

    @property
    def steps(self) -> list[Direction]:
        return list(self._steps)

    def set_path(self, press: tuple[int, int], steps: list[Direction]) -> None:
        """Replaces the walk with an explicit path (tests and probes)."""
        self._press = press
        self._steps = list(steps)

    def add_cursor_tile(self, tile_x: int, tile_y: int) -> None:
        if self._press is None:
            span = tile_span(self._fallback_const, _NON_CLIFF_SPAN)
            if self._fits((tile_x, tile_x + span[0], tile_y, tile_y + span[1])):
                self._press = (tile_x, tile_y)
            return
        for _ in range(_MAX_WALK_STEPS):
            options = self._step_options(self._plan().bounds[-1], tile_x, tile_y)
            if not options or not any(self._try_step(d) for d in options):
                break

    def nodes(self) -> list[ChainNode]:
        if self._press is None:
            return []
        key = tuple(self._steps)
        if self._nodes is None or self._nodes[0] != key:
            self._nodes = (key, self._finish(self._plan()))
        return list(self._nodes[1])

    def _step_options(self, head: Bounds, tile_x: int, tile_y: int) -> list[Direction]:
        x0, x1, y0, y1 = self._widened(head)
        over_x = tile_x - x0 if tile_x < x0 else (tile_x - x1 + 1 if tile_x >= x1 else 0)
        over_y = tile_y - y0 if tile_y < y0 else (tile_y - y1 + 1 if tile_y >= y1 else 0)
        # Larger overshoot first; a tie goes to x.
        options = []
        if over_x:
            options.append((-abs(over_x), 0, (1 if over_x > 0 else -1, 0)))
        if over_y:
            options.append((-abs(over_y), 1, (0, 1 if over_y > 0 else -1)))
        return [d for *_, d in sorted(options)]

    @staticmethod
    def _widened(bounds: Bounds) -> Bounds:
        x0, x1, y0, y1 = bounds
        grow_x = max(0, _MIN_STEP_BOX - (x1 - x0))
        grow_y = max(0, _MIN_STEP_BOX - (y1 - y0))
        return x0 - grow_x // 2, x1 + grow_x - grow_x // 2, y0 - grow_y // 2, y1 + grow_y - grow_y // 2

    def _try_step(self, d: Direction) -> bool:
        if self._steps and d == _opposite(self._steps[-1]):
            return False
        self._steps.append(d)
        bounds = self._plan().bounds
        head = bounds[-1]
        if self._fits(head) and not any(_overlap(head, b) for b in bounds[:-1]):
            return True
        self._steps.pop()
        return False

    def _fits(self, bounds: Bounds) -> bool:
        """Whole footprint on-map: `unit_tile_bounds()` would clamp a cliff
        half off the edge into a box the adjacency math no longer matches."""
        x0, x1, y0, y1 = bounds
        return x0 >= 0 and y0 >= 0 and x1 <= self._map_width and y1 <= self._map_height

    def _plan(self) -> _Layout:
        key = tuple(self._steps)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._resolve()
            self._plans[key] = plan
            if len(self._plans) > _PLAN_CACHE_SIZE:
                self._plans.popitem(last=False)
        else:
            self._plans.move_to_end(key)
        return plan

    def _resolve(self) -> _Layout:
        dirsets = path_dirsets(self._steps)
        layout = self._layout(dirsets, None)
        contacts = self._contacts(layout)
        if any(contacts):
            augmented = [d | {c[0] for c in cs} for d, cs in zip(dirsets, contacts, strict=True)]
            entries = [[self._contact_entry(c, layout, augmented) for c in cs] for cs in contacts]
            layout = self._layout(augmented, entries)
        return layout

    def _layout(self, dirsets, contacts) -> _Layout:
        if dirsets == [frozenset()]:
            # A lone click never reaches the table's empty-dirset row (43%
            # suffix purity, its weakest): the picked piece and frame stand.
            resolved = [(None, None, False)]
        else:
            resolved = cliff_connectivity.resolve_chain(
                dirsets, self._steps, allowed_suffixes=self._allowed, contacts=contacts
            )
        return _Layout(self._press, self._steps, dirsets, resolved, self._fallback_const, self._fallback_frame)

    def _contacts(self, layout: _Layout) -> list[list[tuple[Direction, tuple[str, int]]]]:
        """Per node, (direction, owner) of every cardinal contact that is not
        a path neighbour: an existing cliff or a non-adjacent stroke node."""
        index = _TileIndex()
        for i, b in enumerate(layout.bounds):
            index.add(i, b)
        out = []
        for i, b in enumerate(layout.bounds):
            found = []
            for j in sorted(index.touching(b)):
                if abs(j - i) > 1:
                    d = cliff_connectivity.adjacency_dir(b, layout.bounds[j])
                    if d in _CARDINALS:
                        found.append((d, ("node", j)))
            for k in sorted(self._existing_index.touching(b)):
                d = cliff_connectivity.adjacency_dir(b, self._existing[k].bounds)
                if d in _CARDINALS:
                    found.append((d, ("existing", k)))
            out.append(found)
        return out

    def _contact_entry(self, contact, layout: _Layout, augmented) -> cliff_connectivity.Contact:
        d, (kind, j) = contact
        if kind == "node":
            return d, layout.suffixes[j], augmented[j]
        cliff = self._existing[j]
        dirs = {
            cliff_connectivity.adjacency_dir(cliff.bounds, self._existing[k].bounds)
            for k in self._existing_index.touching(cliff.bounds)
            if k != j
        }
        dirs.add(_opposite(d))
        dirs.discard(None)
        return d, cliff_catalog.suffixes().get(cliff.unit_const), frozenset(dirs)

    def _finish(self, layout: _Layout) -> list[ChainNode]:
        """Truncates at the first node off the map, drops a node overlapping
        an existing cliff or an earlier kept node, then flags every node
        whose final neighbours differ from the shape it was resolved for."""
        ok = list(layout.ok)
        kept: list[int] = []
        index = _TileIndex()
        for i, b in enumerate(layout.bounds):
            if not self._fits(b):
                break
            if self._existing_index.overlapping(b) or index.overlapping(b):
                for j in (i - 1, i + 1):
                    if 0 <= j < len(ok):
                        ok[j] = False
                continue
            kept.append(i)
            index.add(i, b)
        out = []
        for i in kept:
            b = layout.bounds[i]
            realized = {cliff_connectivity.adjacency_dir(b, layout.bounds[j]) for j in index.touching(b) if j != i}
            realized |= {
                cliff_connectivity.adjacency_dir(b, self._existing[k].bounds) for k in self._existing_index.touching(b)
            }
            realized.discard(None)
            x, y = span_anchor(*layout.lows[i], *layout.spans[i])
            out.append(
                ChainNode(
                    unit_const=layout.consts[i],
                    rotation=layout.rotations[i],
                    x=x,
                    y=y,
                    bounds=b,
                    resolved=ok[i] and realized == layout.dirsets[i],
                    dirset=layout.dirsets[i],
                    index=i,
                )
            )
        return out
