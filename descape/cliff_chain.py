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

**The drag direction picks the facing.** A piece's size is which way its
face points (`cliff_connectivity.leg_tall()`), and a run keeps its face on
one side, so each path edge is tall or thin by the direction it was drawn in
(`edge_tall()`) and each node's piece is a lookup on its typed arms
(`resolve_shape()`). The same line drawn the other way is the other facing.
A straight alternates its two art twins by footprint position, never by path
index, so a growing drag cannot shuffle them.

**The press tile is a centre.** The first piece is centred on it, clamped
onto the map, and slid off any existing cliff it would overlap while still
covering the press tile. The layout then walks from there, placing each
node's footprint where `cliff_connectivity.offset()` says the corpus puts
that neighbour.

**A press on an existing cliff anchors the run on it.** That cliff is the
path's node 0: fixed, never placed, and the walk steps from it. An end cap
passes its own run's side on, so a run continued in line keeps its size.

**Contacts repeat until they settle.** After the first layout, a node whose
footprint shares an edge with an existing cliff or a non-adjacent node of
the same stroke takes that neighbour's own leg as one more arm, and the path
lays out again. Each pass types its contacts by the touched pieces' shapes
from the pass before and starts over from the path shapes, until a pass
changes no shape, so two nodes that touch each other agree on their shared
edge and a one-drag loop closes. At most `_MAX_CONTACT_PASSES` run; a run
that does not settle keeps its first pass. A node whose widened shape has no
piece keeps its path shape.

**Joins rewrite the old end, narrowly.** An existing end cap or lone piece
that gains exactly one neighbour from the run takes the frame the table
gives its new arms at its own size (`rewrites()`). Where that needs a
different-size piece it is left alone and counted (`unjoined()`).

**An erase re-caps the ends it leaves, as narrowly** (GH #247,
`erase_rewrites()`). A surviving cliff that was joined to an erased one and
is left joined to exactly one neighbour takes the end cap the table gives
that arm at its own size. "Joined" is the drag's own test: a shared edge,
legs of the same type, and each piece's stored frame shaped for that side
(`shaped_arms()`). Where its size has no such cap it is left alone and
counted; a survivor left with no joined neighbour, or with two, is not
touched.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cache

from descape import cliff_catalog, cliff_connectivity, unit_sprites
from descape.cliff_connectivity import DRAG_FACE_SIDE, edge_tall, leg_tall
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
# Contact passes per layout; a run with no fixpoint inside the bound keeps its first pass.
_MAX_CONTACT_PASSES = 4


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
    # table: no piece for its shape (it keeps the picked piece and frame), no
    # measured offset, or an arm it was shaped for with no neighbour on it.
    resolved: bool
    # The cardinal arms it was shaped for (path plus contacts), and its
    # position on the path: a dropped node leaves a gap in `index`.
    arms: frozenset[Direction] = frozenset()
    index: int = 0


@dataclass(frozen=True)
class ExistingCliff:
    """A cliff already on the map: a fixed contact, a placement blocker, and
    what a join may reshape. `rotation` is None when the stored value is no
    frame index; `unit` is the live object a rewrite writes to."""

    unit_const: int
    bounds: Bounds
    rotation: int | None = None
    player: int = 0
    unit: object = field(default=None, compare=False)


@dataclass(frozen=True)
class ChainRewrite:
    """A new frame for the existing cliff at `existing_index`: same const,
    same position."""

    existing_index: int
    rotation: int


def _frame(rotation) -> int | None:
    try:
        return int(float(rotation))
    except (TypeError, ValueError, OverflowError):
        return None


def existing_cliffs(scenario, map_width: int, map_height: int) -> list[ExistingCliff]:
    """Every cliff already placed, across all players.

    Scanned straight off `unit_manager` rather than through
    `map_view._unit_index`: that pick index is current or None outside Units
    mode (see `_after_unit_mutation()`'s own mode gate). It may survive
    there, but a Terrain-mode Cliff stroke cannot count on it being built.
    """
    consts = unit_sprites.cliff_consts()
    out: list[ExistingCliff] = []
    for player, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            if unit.unit_const not in consts:
                continue
            bounds = unit_tile_bounds(unit, map_width, map_height)
            if bounds is not None:
                out.append(
                    ExistingCliff(
                        unit_const=unit.unit_const, bounds=bounds, rotation=_frame(unit.rotation), player=player, unit=unit
                    )
                )
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


def index_cliffs(existing: list[ExistingCliff]) -> _TileIndex:
    """The tile index a `ChainStroke` probes `existing` through. A caller that
    builds many strokes over one cliff list (the hover) builds it once."""
    index = _TileIndex()
    for k, cliff in enumerate(existing):
        index.add(k, cliff.bounds)
    return index


def _cardinal_contacts(existing: list[ExistingCliff], index: _TileIndex, k: int) -> list[tuple[Direction, int]]:
    """(direction from cliff `k`, neighbour index) for every cliff sharing an edge with `k`."""
    bounds = existing[k].bounds
    out = []
    for j in sorted(index.touching(bounds)):
        if j != k:
            d = cliff_connectivity.adjacency_dir(bounds, existing[j].bounds)
            if d in _CARDINALS:
                out.append((d, j))
    return out


@cache
def _frame_arms(suffix: int, frame_limit: int) -> dict[int, frozenset[Direction]]:
    """frame -> the intersection of every arm set whose row at `suffix` gives it."""
    out: dict[int, frozenset[Direction]] = {}
    for arms in cliff_connectivity.shape_tables().rows:
        for frame in cliff_connectivity.rewrite_frames(arms, suffix, frame_limit=frame_limit) or ():
            out[frame] = out[frame] & arms if frame in out else arms
    return out


def shaped_arms(unit_const: int, frame: int | None) -> frozenset[Direction]:
    """The cardinal arms a cliff of `unit_const` stored on `frame` is shaped
    for: what every table row at its suffix that gives that frame agrees on.
    Frames repeat across rows (a 3x3 straight's 1/2 also top both T rows), so
    a T's straight frame is shaped for its two through arms only. Empty for a
    frame no row gives, or no frame at all."""
    suffix = cliff_catalog.suffixes().get(unit_const)
    if suffix is None or frame is None:
        return frozenset()
    return _frame_arms(suffix, cliff_catalog.frame_count(unit_const)).get(frame, frozenset())


def _cliff_rewrite_frames(cliff: ExistingCliff, arms: frozenset[Direction]) -> tuple[int, ...] | None:
    """`rewrite_frames()` for an existing cliff at its own suffix and frame count; None off the table."""
    suffix = cliff_catalog.suffixes().get(cliff.unit_const)
    if suffix is None:
        return None
    return cliff_connectivity.rewrite_frames(arms, suffix, frame_limit=cliff_catalog.frame_count(cliff.unit_const))


def _joined(existing: list[ExistingCliff], k: int, d: Direction, j: int) -> bool:
    """Whether cliff `k` is joined to its `d`-side neighbour `j`: legs of the
    same type there, and each stored frame shaped for its side of the edge."""
    a, b = existing[k], existing[j]
    suffixes = cliff_catalog.suffixes()
    suffix_a, suffix_b = suffixes.get(a.unit_const), suffixes.get(b.unit_const)
    if suffix_a is None or suffix_b is None:
        return False
    back = _opposite(d)
    return (
        leg_tall(suffix_a, d) == leg_tall(suffix_b, back)
        and d in shaped_arms(a.unit_const, a.rotation)
        and back in shaped_arms(b.unit_const, b.rotation)
    )


def joined_arms(
    existing: list[ExistingCliff], index: _TileIndex, k: int, *, skip: frozenset[int] | set[int] = frozenset()
) -> frozenset[Direction]:
    """The cardinal directions in which cliff `k` is joined to a cliff not in `skip`."""
    return frozenset(d for d, j in _cardinal_contacts(existing, index, k) if j not in skip and _joined(existing, k, d, j))


def erase_rewrites(
    existing: list[ExistingCliff], index: _TileIndex, erased: set[int]
) -> tuple[list[ChainRewrite], int]:
    """(rewrites, left alone) for the survivors of erasing `erased` (indices
    into `existing`). A survivor joined to an erased cliff and left joined to
    exactly one other takes the end cap the table gives that arm at its own
    size, unless it is on it already. Where its size has none it is left
    alone and counted. Any other survivor is untouched and uncounted."""
    erased = set(erased)
    touched: set[int] = set()
    for e in erased:
        for d, j in _cardinal_contacts(existing, index, e):
            if j not in erased and _joined(existing, j, _opposite(d), e):
                touched.add(j)
    rewrites: list[ChainRewrite] = []
    left = 0
    for k in sorted(touched):
        surviving = [(d, j) for d, j in _cardinal_contacts(existing, index, k) if j not in erased]
        joined = [(d, j) for d, j in surviving if _joined(existing, k, d, j)]
        arms = frozenset(d for d, _ in joined)
        # A junction's centre: an unjoined contact (a T's branch) still touches it, as in _joins().
        if len(arms) != 1 or len(joined) != len(surviving):
            continue
        cliff = existing[k]
        frames = _cliff_rewrite_frames(cliff, arms)
        if frames is None:
            left += 1
        elif cliff.rotation not in frames:
            rewrites.append(ChainRewrite(k, _twin(frames, (cliff.bounds[0], cliff.bounds[2]))))
    return rewrites, left


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


def _path_arms(steps: list[Direction], side: int) -> list[dict[Direction, bool]]:
    """`path_dirsets()` with each arm typed by its edge: tall or thin."""
    out: list[dict[Direction, bool]] = [{} for _ in range(len(steps) + 1)]
    for i, step in enumerate(steps):
        tall = edge_tall(step, side)
        out[i][step] = tall
        out[i + 1][_opposite(step)] = tall
    return out


def _abut(step: Direction, span_a: tuple[int, int], span_b: tuple[int, int]) -> tuple[int, int]:
    """Flush along `step`, cross-axis 0: the layout when no offset is measured."""
    dx, dy = step
    if dx:
        return (span_a[0] if dx > 0 else -span_b[0]), 0
    return 0, (span_a[1] if dy > 0 else -span_b[1])


def _twin(frames: tuple[int, ...], low: tuple[int, int]) -> int:
    """One of a straight's two art twins, by where the footprint sits."""
    return frames[0] if ((low[0] + low[1]) // 3) % 2 == 0 else frames[-1]


@dataclass(frozen=True)
class _Shape:
    """One path node's piece: its const and span, the frames its shape allows
    (none: the picked frame stands) and the arms it was shaped for."""

    const: int
    span: tuple[int, int]
    suffix: int | None
    frames: tuple[int, ...]
    arms: frozenset[Direction]
    ok: bool


def _off_nominal(shape: _Shape) -> bool:
    """A piece whose real span is not its suffix's nominal one: Short Marble
    Cliff 01 (3x2 where every other family's 01 is 3x3)."""
    nominal = cliff_connectivity.NOMINAL_SPAN.get(shape.suffix)
    return nominal is not None and shape.span != nominal


@dataclass
class _Plan:
    """One resolve plus layout of a path. `bounds` covers every path node,
    the anchor, dropped and off-map ones included: the walk's step test reads
    the head's."""

    side: int
    shapes: list[_Shape]
    lows: list[tuple[int, int]]
    bounds: list[Bounds]
    ok: list[bool]
    # Node 0 of a free start beside an existing cliff: whether it touches
    # one, and whether its shape took that contact in.
    start_touches: bool = False
    start_joined: bool = False


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
        existing_index: _TileIndex | None = None,
        side: int | None = None,
    ) -> None:
        self._map_width = map_width
        self._map_height = map_height
        self._existing = existing
        # The Piece/Frame picker's values: a lone node's piece, the family
        # every resolved suffix is drawn from, and the fallback for a node
        # whose shape has no piece.
        self._fallback_const = fallback_const
        self._fallback_frame = fallback_frame
        self._allowed = frozenset(
            s for s in range(1, 10) if cliff_catalog.piece_for_suffix(fallback_const, s) is not None
        )
        self._frame_limit = cliff_catalog.frame_count(fallback_const)
        self._existing_index = index_cliffs(existing) if existing_index is None else existing_index
        self._existing_arm_cache: dict[int, frozenset[Direction]] = {}
        # A test's override of the side the face goes on; None derives it.
        self._side_override = side
        # The press tile: the first piece's centre, or a tile of the anchor.
        self._press: tuple[int, int] | None = None
        # The existing cliff the press landed on, as an index into `existing`.
        self._anchor: int | None = None
        self._steps: list[Direction] = []
        self._plans: OrderedDict[tuple, _Plan] = OrderedDict()
        self._nodes: tuple[tuple, list[ChainNode]] | None = None

    @property
    def steps(self) -> list[Direction]:
        return list(self._steps)

    @property
    def anchor(self) -> int | None:
        """Index into `existing` of the cliff the run starts on, or None."""
        return self._anchor

    @property
    def side(self) -> int | None:
        """Which side the run's face is on, None before the first step: it
        is derived from the anchor and the path on every resolve, never stored."""
        return self._plan().side if self._press is not None and self._steps else None

    def set_path(self, press: tuple[int, int], steps: list[Direction]) -> None:
        """Replaces the walk with an explicit path (tests and probes)."""
        self._start(press)
        self._steps = list(steps)

    def add_cursor_tile(self, tile_x: int, tile_y: int) -> None:
        if self._press is None:
            if 0 <= tile_x < self._map_width and 0 <= tile_y < self._map_height:
                self._start((tile_x, tile_y))
            return
        for _ in range(_MAX_WALK_STEPS):
            options = self._step_options(self._plan().bounds[-1], tile_x, tile_y)
            if not options or not any(self._try_step(d) for d in options):
                break

    def nodes(self) -> list[ChainNode]:
        if self._press is None:
            return []
        key = self._key()
        if self._nodes is None or self._nodes[0] != key:
            self._nodes = (key, self._finish(self._plan()))
        return list(self._nodes[1])

    def head_bounds(self) -> Bounds | None:
        """The footprint of the run's head, or None when the head is not
        placed (dropped, off the map, or an anchor with no step yet)."""
        nodes = self.nodes()
        return nodes[-1].bounds if nodes and nodes[-1].index == len(self._steps) else None

    def rewrites(self) -> list[ChainRewrite]:
        """New frames for existing cliffs the run joins: only an end cap or
        a lone piece (at most one cardinal neighbour) that gains exactly one
        arm from a placed node, where the table has its new arms at its own
        size and it is not on one of those frames already."""
        return self._joins()[0]

    def unjoined(self) -> int:
        """How many joined ends were left as they were because the join needs
        a different-size piece: end caps, lone pieces and the anchor, and a
        cap a run of the other facing met end-on."""
        return self._joins()[1]

    def _start(self, press: tuple[int, int]) -> None:
        self._press = press
        hits = self._existing_index.overlapping((press[0], press[0] + 1, press[1], press[1] + 1))
        self._anchor = min(hits) if hits else None
        self._steps = []
        self._plans.clear()
        self._nodes = None

    def _key(self) -> tuple:
        return self._press, self._anchor, self._side_override, tuple(self._steps)

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

    def _plan(self) -> _Plan:
        key = self._key()
        plan = self._plans.get(key)
        if plan is None:
            plan = self._resolve()
            self._plans[key] = plan
            if len(self._plans) > _PLAN_CACHE_SIZE:
                self._plans.popitem(last=False)
        else:
            self._plans.move_to_end(key)
        return plan

    # -- which side the face goes on ---------------------------------------

    def _resolve(self) -> _Plan:
        if self._side_override is not None:
            return self._lay(self._side_override)
        if self._anchor is not None:
            return self._lay(self._anchor_side())
        plan = self._lay(DRAG_FACE_SIDE)
        if self._steps and plan.start_touches and not plan.start_joined:
            # A free start beside a cliff whose facing the default side cannot
            # continue: the other side's, if that one joins it.
            other = self._lay(-DRAG_FACE_SIDE)
            if other.start_joined:
                return other
        return plan

    def _anchor_side(self) -> int:
        """An end cap keeps its own run's side. A lone piece takes the side
        whose first edge matches its leg on the leaving side, if it has a cap
        frame for that arm at all. A mid-run piece leaves it to the drag."""
        cliff = self._existing[self._anchor]
        suffix = cliff_catalog.suffixes().get(cliff.unit_const)
        arms = self._existing_arms(self._anchor)
        if suffix is None or len(arms) > 1:
            return DRAG_FACE_SIDE
        if arms:
            (arm,) = arms
            # The run that ends on this cap arrived travelling away from its arm.
            travel, tall = _opposite(arm), leg_tall(suffix, arm)
        elif self._steps and self._rewrite_frames(self._anchor, frozenset({self._steps[0]})) is not None:
            travel, tall = self._steps[0], leg_tall(suffix, self._steps[0])
        else:
            # A thin piece's broadside is tall only nominally: no cap there.
            return DRAG_FACE_SIDE
        return DRAG_FACE_SIDE if edge_tall(travel, DRAG_FACE_SIDE) == tall else -DRAG_FACE_SIDE

    # -- pieces and layout ---------------------------------------------------

    def _lay(self, side: int) -> _Plan:
        path_arms = _path_arms(self._steps, side)
        base = [self._shape(arms) or self._picked(arms) for arms in path_arms]
        if self._anchor is not None:
            base[0] = self._anchor_shape()
        plan = self._layout(side, base)
        first: _Plan | None = None
        for _ in range(_MAX_CONTACT_PASSES):
            contacts = self._contacts(plan)
            if first is None and not any(arms for arms, _ in contacts):
                return plan
            shapes, start_joined = self._contact_shapes(contacts, base, path_arms)
            if first is not None and shapes == plan.shapes:
                return plan
            start_touches = contacts[0][1]
            plan = self._layout(side, shapes)
            plan.start_touches = start_touches
            plan.start_joined = start_touches and start_joined
            if first is None:
                first = plan
        # TASK-453: the last pass's plan gets its check too, so a run that settles there keeps it.
        shapes, _ = self._contact_shapes(self._contacts(plan), base, path_arms)
        return plan if shapes == plan.shapes else first

    def _contact_shapes(self, contacts, base: list[_Shape], path_arms) -> tuple[list[_Shape], bool]:
        """(each node's shape widened by its contacts, whether node 0's was).
        Every pass starts over from the path shapes, so a node that lost its contact reverts."""
        shapes = list(base)
        start_joined = False
        for i, (extra, _) in enumerate(contacts):
            if not extra:
                continue
            # A path arm keeps its edge's type over a contact on the same side.
            found = self._shape({**extra, **path_arms[i]})
            if found is not None:
                shapes[i] = found
                start_joined = start_joined or i == 0
        return shapes, start_joined

    def _shape(self, arms: dict[Direction, bool]) -> _Shape | None:
        """The table's piece for these typed arms, or None when it has none.
        A node with no arm never reaches the table: the picked piece stands."""
        if not arms:
            return None
        found = cliff_connectivity.resolve_shape(arms, allowed_suffixes=self._allowed, frame_limit=self._frame_limit)
        if found is None:
            return None
        suffix, frames = found
        const = cliff_catalog.piece_for_suffix(self._fallback_const, suffix)
        return _Shape(const, tile_span(const, _NON_CLIFF_SPAN), suffix, frames, frozenset(arms), True)

    def _picked(self, arms: dict[Direction, bool]) -> _Shape:
        const = self._fallback_const
        suffix = cliff_catalog.suffixes().get(const)
        return _Shape(const, tile_span(const, _NON_CLIFF_SPAN), suffix, (), frozenset(arms), False)

    def _anchor_shape(self) -> _Shape:
        cliff = self._existing[self._anchor]
        x0, x1, y0, y1 = cliff.bounds
        arms = self._existing_arms(self._anchor) | frozenset(self._steps[:1])
        return _Shape(cliff.unit_const, (x1 - x0, y1 - y0), cliff_catalog.suffixes().get(cliff.unit_const), (), arms, True)

    def _layout(self, side: int, shapes: list[_Shape]) -> _Plan:
        ok = [shape.ok for shape in shapes]
        lows = [self._first_low(shapes[0].span)]
        for i, step in enumerate(self._steps):
            a, b = shapes[i], shapes[i + 1]
            if _off_nominal(a) or _off_nominal(b):
                # The table keys offsets by real span, and Short Marble 01's 3x2 is
                # the s6/s7 corners' there: none measured is its own, so abut flush.
                found = _abut(step, a.span, b.span)
            else:
                found = cliff_connectivity.offset(a.arms, a.span, step, b.arms, b.span)
            if found is None:
                found = _abut(step, a.span, b.span)
                ok[i] = ok[i + 1] = False
            lows.append((lows[i][0] + found[0], lows[i][1] + found[1]))
        bounds = [(x, x + s.span[0], y, y + s.span[1]) for (x, y), s in zip(lows, shapes, strict=True)]
        return _Plan(side=side, shapes=list(shapes), lows=lows, bounds=bounds, ok=ok)

    def _first_low(self, span: tuple[int, int]) -> tuple[int, int]:
        """Node 0's low corner: the anchor's own, else `span` centred on the
        press tile, clamped onto the map, then slid off any existing cliff."""
        if self._anchor is not None:
            x0, _x1, y0, _y1 = self._existing[self._anchor].bounds
            return x0, y0
        low = tuple(
            max(0, min(press - size // 2, limit - size))
            for press, size, limit in zip(self._press, span, (self._map_width, self._map_height), strict=True)
        )
        return self._slid(low, span)

    def _slid(self, low: tuple[int, int], span: tuple[int, int]) -> tuple[int, int]:
        """`low` moved by the smallest shift after which the footprint is on
        the map, clear of every existing cliff and still over the press tile;
        `low` itself when it is clear already or no shift is."""

        def box(x: int, y: int) -> Bounds:
            return x, x + span[0], y, y + span[1]

        if not self._existing_index.overlapping(box(*low)):
            return low
        best = None
        for x in range(self._press[0] - span[0] + 1, self._press[0] + 1):
            for y in range(self._press[1] - span[1] + 1, self._press[1] + 1):
                if not self._fits(box(x, y)) or self._existing_index.overlapping(box(x, y)):
                    continue
                dx, dy = x - low[0], y - low[1]
                # Smallest first; a tie goes to x over y, then to the negative shift.
                rank = (abs(dx) + abs(dy), abs(dy), dx, dy)
                if best is None or rank < best[0]:
                    best = (rank, (x, y))
        return low if best is None else best[1]

    def _contacts(self, plan: _Plan) -> list[tuple[dict[Direction, bool], bool]]:
        """Per node, its cardinal contacts that are not path neighbours, each
        typed by the touched piece's own leg, and whether any is an existing
        cliff. The anchor is the path's node 0, not a contact of its own."""
        index = _TileIndex()
        for i, b in enumerate(plan.bounds):
            index.add(i, b)
        suffixes = cliff_catalog.suffixes()
        out = []
        for i, b in enumerate(plan.bounds):
            found: dict[Direction, bool] = {}
            touches_existing = False
            if i == 0 and self._anchor is not None:
                out.append((found, False))
                continue
            for j in sorted(index.touching(b)):
                suffix = plan.shapes[j].suffix
                if abs(j - i) > 1 and suffix is not None:
                    d = cliff_connectivity.adjacency_dir(b, plan.bounds[j])
                    if d in _CARDINALS:
                        found.setdefault(d, leg_tall(suffix, _opposite(d)))
            for k in sorted(self._existing_index.touching(b)):
                suffix = suffixes.get(self._existing[k].unit_const)
                if k == self._anchor or suffix is None:
                    continue
                d = cliff_connectivity.adjacency_dir(b, self._existing[k].bounds)
                if d in _CARDINALS:
                    found.setdefault(d, leg_tall(suffix, _opposite(d)))
                    touches_existing = True
            out.append((found, touches_existing))
        return out

    def _finish(self, plan: _Plan) -> list[ChainNode]:
        """Truncates at the first node off the map, drops a node overlapping
        an existing cliff or an earlier kept node, then flags every node with
        an arm it was shaped for and no neighbour on it. The anchor is the
        map's own cliff: never placed, never dropped."""
        kept: list[int] = []
        index = _TileIndex()
        for i, b in enumerate(plan.bounds):
            if i == 0 and self._anchor is not None:
                continue
            if not self._fits(b):
                break
            if self._existing_index.overlapping(b) or index.overlapping(b):
                continue
            kept.append(i)
            index.add(i, b)
        out = []
        for i in kept:
            b, shape = plan.bounds[i], plan.shapes[i]
            realized = {cliff_connectivity.adjacency_dir(b, plan.bounds[j]) for j in index.touching(b) if j != i}
            realized |= {
                cliff_connectivity.adjacency_dir(b, self._existing[k].bounds) for k in self._existing_index.touching(b)
            }
            x, y = span_anchor(*plan.lows[i], *shape.span)
            out.append(
                ChainNode(
                    unit_const=shape.const,
                    rotation=_twin(shape.frames, plan.lows[i]) if shape.frames else self._fallback_frame,
                    x=x,
                    y=y,
                    bounds=b,
                    resolved=plan.ok[i] and shape.arms <= realized,
                    arms=shape.arms,
                    index=i,
                )
            )
        return out

    # -- joins onto existing cliffs ----------------------------------------

    def _existing_arms(self, k: int) -> frozenset[Direction]:
        """The cardinal neighbours an existing cliff already has on the map."""
        arms = self._existing_arm_cache.get(k)
        if arms is None:
            contacts = _cardinal_contacts(self._existing, self._existing_index, k)
            arms = self._existing_arm_cache[k] = frozenset(d for d, _ in contacts)
        return arms

    def _rewrite_frames(self, k: int, arms: frozenset[Direction]) -> tuple[int, ...] | None:
        return _cliff_rewrite_frames(self._existing[k], arms)

    def _joins(self) -> tuple[list[ChainRewrite], int]:
        suffixes = cliff_catalog.suffixes()
        # Per existing cliff, each side a placed node touches, and whether the
        # node joins it there: shaped for that arm, with a leg of the same type.
        gained: dict[int, dict[Direction, bool]] = {}
        for node in self.nodes():
            node_suffix = suffixes.get(node.unit_const)
            for k in self._existing_index.touching(node.bounds):
                d = cliff_connectivity.adjacency_dir(self._existing[k].bounds, node.bounds)
                if d not in _CARDINALS:
                    continue
                suffix, back = suffixes.get(self._existing[k].unit_const), _opposite(d)
                joined = (
                    back in node.arms
                    and suffix is not None
                    and node_suffix is not None
                    and leg_tall(suffix, d) == leg_tall(node_suffix, back)
                )
                sides = gained.setdefault(k, {})
                sides[d] = sides.get(d, False) or joined
        rewrites: list[ChainRewrite] = []
        unjoined = 0
        for k in sorted(gained):
            cliff = self._existing[k]
            prior = self._existing_arms(k)
            new = set(gained[k]) - prior
            # A run passing beside a straight or a corner reshapes nothing.
            if not new or (len(prior) > 1 and k != self._anchor):
                continue
            frames = self._rewrite_frames(k, prior | new)
            if frames is None:
                unjoined += 1
            elif len(prior) <= 1 and len(new) == 1 and cliff.rotation not in frames:
                if gained[k][next(iter(new))]:
                    rewrites.append(ChainRewrite(k, _twin(frames, (cliff.bounds[0], cliff.bounds[2]))))
                else:
                    # A run of the other facing met it end-on: two runs abut on their caps.
                    unjoined += 1
        return rewrites, unjoined
