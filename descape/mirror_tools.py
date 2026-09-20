"""Map mirroring (symmetry generator): plan_mirror() for terrain and
elevation (Stage 1), plan_mirror_units() for the units on top of it (Stage 2).

Follows fill_tools.py/brush.py's shape: pure index math, no PyQt5, and no
AoE2ScenarioParser import either -- duck-typed on
mm.map_width/map_height/terrain and each tile's
terrain_id/elevation/layer, so a plain fake object works for tests the same
way tests/test_fill_tools.py's FakeMapManager does. Square maps only (the
whole app-wide rule -- see scenario_io.LoadedScenario.map_is_square); callers
must gate on that before calling plan_mirror, same as every other terrain
tool.

The eight transforms are the dihedral group D4 acting on a square tile
lattice, so every one of the nine non-trivial subgroups below is exact in
tile space -- no resampling, no rounding. Decision 3 in the plan is why the
mode labels are phrased in SCREEN terms (the isometric projection's tips are
W=(0,0), N=(n-1,0), E=(n-1,n-1), S=(0,n-1)) rather than "obvious" array
directions: `a` is the true left/right mirror, `d` the true top/bottom
mirror, and `mx`/`my`/`r`/`r3` are valid tile-space operations that do NOT
correspond to a screen-axis-aligned mirror or rotation.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache

from descape import gate_orientation, render, unit_sprites
from descape.edit_history import TileState, tile_state
from descape.terrain_palette import tile_span

# --------------------------------------------------------------------------
# The eight (x, y, n) -> (x, y) transforms, D4 acting on an n x n lattice.
# --------------------------------------------------------------------------


def _id(x: int, y: int, n: int) -> tuple[int, int]:
    return (x, y)


def _r(x: int, y: int, n: int) -> tuple[int, int]:
    return (n - 1 - y, x)


def _r2(x: int, y: int, n: int) -> tuple[int, int]:
    return (n - 1 - x, n - 1 - y)


def _r3(x: int, y: int, n: int) -> tuple[int, int]:
    return (y, n - 1 - x)


def _mx(x: int, y: int, n: int) -> tuple[int, int]:
    return (n - 1 - x, y)


def _my(x: int, y: int, n: int) -> tuple[int, int]:
    return (x, n - 1 - y)


def _d(x: int, y: int, n: int) -> tuple[int, int]:
    return (y, x)


def _a(x: int, y: int, n: int) -> tuple[int, int]:
    return (n - 1 - y, n - 1 - x)


TRANSFORMS: dict[str, Callable[[int, int, int], tuple[int, int]]] = {
    "id": _id,
    "r": _r,
    "r2": _r2,
    "r3": _r3,
    "mx": _mx,
    "my": _my,
    "d": _d,
    "a": _a,
}

# The same eight transforms, in doubled centre-relative (u, v) space
# (u = 2x-(n-1), v = 2y-(n-1)): "n-1" drops out entirely since it is the
# lattice's own centre offset, which doubled coordinates are built to
# cancel. Used only by _derive_slice_labels below -- plan_mirror itself
# stays in (x, y, n) form throughout, per the plan's own instruction, so
# tests can assert e.g. a(0, 0, n) == (n-1, n-1) directly.
_UV_TRANSFORMS: dict[str, Callable[[int, int], tuple[int, int]]] = {
    "id": lambda u, v: (u, v),
    "r": lambda u, v: (-v, u),
    "r2": lambda u, v: (-u, -v),
    "r3": lambda u, v: (v, -u),
    "mx": lambda u, v: (-u, v),
    "my": lambda u, v: (u, -v),
    "d": lambda u, v: (v, u),
    "a": lambda u, v: (-v, -u),
}


def doubled(x: int, y: int, n: int) -> tuple[int, int]:
    """(u, v) = (2x-(n-1), 2y-(n-1)) -- integer, centre-relative, and never
    zero for even n (every standard AoE2 map size). Used only for
    preferred_domain predicates, never for the transforms themselves."""
    return (2 * x - (n - 1), 2 * y - (n - 1))


# --------------------------------------------------------------------------
# Slice-label derivation -- cosmetic dropdown text only. The live slice
# overlay (viewer.py) is what actually disambiguates for the user, so this
# picks one fixed representative point per mode and classifies each group
# element's image of it into one of 8 compass sectors, rather than hand-
# typing a guessed table that could silently drift from the transforms.
# --------------------------------------------------------------------------

# angle = atan2(v-u, u+v) in degrees: the iso projection's screen_x tracks
# x+y (so u+v after doubling) and screen_y tracks y-x (so v-u), and the map's
# four screen tips West/North/East/South sit at exactly 180/270/0/90 degrees
# in this scheme.
_SECTOR_NAMES = (
    "East",
    "South-East",
    "South",
    "South-West",
    "West",
    "North-West",
    "North",
    "North-East",
)


def _classify_sector(u: int, v: int) -> str:
    angle = math.degrees(math.atan2(v - u, u + v)) % 360
    idx = round(angle / 45) % 8
    return _SECTOR_NAMES[idx]


def _disambiguate(labels: tuple[str, ...]) -> tuple[str, ...]:
    """Appends a deterministic " (N)" suffix to every repeated label, so the
    Source-slice combo never shows two textually-identical entries. Needed
    for mode 9 only -- see _derive_slice_labels' docstring for why 8-way's
    eight group elements can only ever land in 4 distinct compass sectors,
    not 8, and that this is a fact about D4's group structure, not a search
    that gave up too early."""
    counts = Counter(labels)
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        if counts[label] == 1:
            out.append(label)
            continue
        seen[label] = seen.get(label, 0) + 1
        out.append(f"{label} ({seen[label]})")
    return tuple(out)


def _derive_slice_labels(
    group: tuple[str, ...], preferred_domain: Callable[[int, int], bool]
) -> tuple[str, ...]:
    """Searches small integer (u, v) candidates (both signs -- several modes'
    domains require a negative coordinate) for one satisfying
    preferred_domain whose group-element images classify into as many
    distinct compass sectors as possible, preferring a fully-distinct set.

    Full distinctness is achievable for every mode except 9 (8-way): D4's
    rotation subgroup {id, r, r2, r3} occupies one whole compass-sector
    parity class by construction (rotating a point by 90 degrees in (u, v)
    space rotates its screen angle by exactly 90 degrees too, i.e. by 2
    sectors), and the reflection coset {mx, my, d, a} always lands in the
    SAME parity class (angle(d(p)) = -angle(p) is an identity, and negating
    an angle about 0 degrees -- itself a sector centre -- preserves sector
    parity). So mode 9's eight elements can only ever occupy 4 distinct
    sectors, each exactly twice, for any point and any of its four
    reflections used as the coset representative. That is a fact about D4's
    group structure under this exact classification scheme, not a search
    that gave up too early -- confirmed by exhaustive search over a wide
    integer range before this function was written this way. _disambiguate()
    is what keeps the dropdown showing 8 distinct strings anyway.
    """
    first: tuple[str, ...] | None = None
    candidates = list(range(1, 15)) + list(range(-1, -15, -1))
    for u0 in candidates:
        for v0 in candidates:
            if u0 == 0 or v0 == 0:
                continue
            if not preferred_domain(u0, v0):
                continue
            labels = tuple(_classify_sector(*_UV_TRANSFORMS[name](u0, v0)) for name in group)
            if first is None:
                first = labels
            if len(set(labels)) == len(group):
                return labels
    if first is None:
        raise AssertionError(f"no representative (u, v) found for domain of group {group!r}")
    return _disambiguate(first)


# --------------------------------------------------------------------------
# The nine modes -- Decision 2's table, exactly. Order matters within each
# group tuple: it is the Source-slice combo's slice order (slice_index
# indexes into it).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MirrorMode:
    mode_id: int
    label: str
    group: tuple[str, ...]
    preferred_domain: Callable[[int, int], bool]
    slice_labels: tuple[str, ...]


def _mode(mode_id: int, label: str, group: tuple[str, ...], domain: Callable[[int, int], bool]) -> MirrorMode:
    return MirrorMode(mode_id, label, group, domain, _derive_slice_labels(group, domain))


MODES: tuple[MirrorMode, ...] = (
    _mode(1, "Mirror West ↔ East (left/right on screen)", ("id", "a"), lambda u, v: u + v < 0),
    _mode(2, "Mirror North ↔ South (top/bottom on screen)", ("id", "d"), lambda u, v: v - u < 0),
    _mode(3, "Mirror West ↔ North", ("id", "mx"), lambda u, v: u < 0),
    _mode(4, "Mirror West ↔ South", ("id", "my"), lambda u, v: v < 0),
    _mode(5, "180° rotation", ("id", "r2"), lambda u, v: u + v < 0),
    _mode(6, "4-way rotational (90°)", ("id", "r", "r2", "r3"), lambda u, v: u > 0 and v > 0),
    _mode(
        7,
        "4-way reflective, screen axes",
        ("id", "d", "a", "r2"),
        lambda u, v: u + v > 0 and v - u < 0,
    ),
    _mode(8, "4-way reflective, map axes", ("id", "mx", "my", "r2"), lambda u, v: u > 0 and v > 0),
    _mode(
        9,
        "8-way (full symmetry)",
        ("id", "r", "r2", "r3", "mx", "my", "d", "a"),
        lambda u, v: u >= 0 and v >= 0 and v <= u,
    ),
)

MODE_BY_ID: dict[int, MirrorMode] = {mode.mode_id: mode for mode in MODES}


# --------------------------------------------------------------------------
# The core routine.
# --------------------------------------------------------------------------


@dataclass
class MirrorPlan:
    # Flat destination index + new (terrain_id, elevation, layer), already
    # filtered to tiles that actually differ from their current state.
    changes: list[tuple[int, TileState]]
    # {d : source(d) == d} -- derived as a byproduct of the same per-tile
    # loop that builds `changes`, not redefined via preferred_domain: on a
    # degenerate orbit (non-trivial stabiliser -- on-axis tiles, an odd-n
    # map's centre under C4) that would silently disagree with what the
    # gather actually treated as the source.
    source_indices: frozenset[int]
    # (index_a, index_b) flat-index pairs, index_a < index_b, of every
    # 8-connected neighbour pair whose post-mirror elevations differ by more
    # than 1. Empty by construction when do_elevation is False -- nothing
    # about elevation changed, so nothing to flag.
    elevation_violations: list[tuple[int, int]]


def plan_mirror(mm, mode_id: int, slice_index: int, do_terrain: bool, do_elevation: bool) -> MirrorPlan:
    """Computes (but does not apply) a whole-map mirror under MODE_BY_ID
    [mode_id], sourced from that mode's slice_index-th group element.

    Everything is computed before any mutation, per the plan's rationale:
    EditHistory.apply() has no try/finally, so a mutator that raises midway
    would leave its stroke snapshot uncleared and wedge every later edit.
    The mutator this function's caller hands to apply() must therefore be a
    bare assignment loop over `changes` that cannot raise -- all the work
    that CAN fail (or merely take a while) happens here instead, including
    the elevation seam check, which runs against a virtual post-mirror
    elevation array rather than mutating mm.terrain and rolling back.
    """
    mode = MODE_BY_ID[mode_id]
    width, height = mm.map_width, mm.map_height
    n = width  # caller must gate on map_is_square; square is assumed from here on
    terrain = mm.terrain
    group_fns = [TRANSFORMS[name] for name in mode.group]
    source_fn = group_fns[slice_index]
    preferred_domain = mode.preferred_domain

    changes: list[tuple[int, TileState]] = []
    source_indices: set[int] = set()
    # Virtual post-mirror elevation grid for the seam check -- starts as a
    # plain copy of the current elevations and is overwritten in place below
    # only where do_elevation is True, so no tile in mm.terrain is ever
    # mutated by this function.
    new_elevation = [tile.elevation for tile in terrain]

    for y in range(height):
        for x in range(width):
            d_idx = y * width + x
            orbit = {g(x, y, n) for g in group_fns}
            rep = min(
                orbit,
                key=lambda t: (0 if preferred_domain(*doubled(t[0], t[1], n)) else 1, t[1], t[0]),
            )
            src_x, src_y = source_fn(*rep, n)
            if (src_x, src_y) == (x, y):
                source_indices.add(d_idx)

            src_idx = src_y * width + src_x
            src_tile = terrain[src_idx]
            dst_tile = terrain[d_idx]
            old_state = tile_state(dst_tile)

            new_terrain_id = src_tile.terrain_id if do_terrain else old_state[0]
            new_layer = src_tile.layer if do_terrain else old_state[2]
            new_elevation_value = src_tile.elevation if do_elevation else old_state[1]
            if do_elevation:
                new_elevation[d_idx] = new_elevation_value

            new_state = (new_terrain_id, new_elevation_value, new_layer)
            if new_state != old_state:
                changes.append((d_idx, new_state))

    elevation_violations: list[tuple[int, int]] = []
    if do_elevation:
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                elevation = new_elevation[idx]
                for dy, dx in itertools.product(range(-1, 2), repeat=2):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    n_idx = ny * width + nx
                    if n_idx <= idx:
                        continue  # the pair (n_idx, idx) already visited this pair from its own side
                    if abs(elevation - new_elevation[n_idx]) > 1:
                        elevation_violations.append((idx, n_idx))

    return MirrorPlan(changes, frozenset(source_indices), elevation_violations)


# --------------------------------------------------------------------------
# Stage 2: units.
#
# Imports render (Qt-free, numpy only) for the footprint geometry -- span
# parity and the diagonal gates' sparse tile sets live there, and a second
# copy here would be a silent drift source. The module's own "no PyQt5, no
# AoE2ScenarioParser" rule is unaffected; units are duck-typed here too.
# --------------------------------------------------------------------------


# Continuous counterparts of TRANSFORMS, acting on a unit's float position
# rather than on a tile index: a tile t spans [t, t+1), so the reflection that
# sends tile t to tile n-1-t sends coordinate X to n-X. Every element is the
# same permutation of the lattice either way -- only the "-1" differs, and it
# is the tile-index convention, not part of the geometry.
POSITION_TRANSFORMS: dict[str, Callable[[float, float, int], tuple[float, float]]] = {
    "id": lambda x, y, n: (x, y),
    "r": lambda x, y, n: (n - y, x),
    "r2": lambda x, y, n: (n - x, n - y),
    "r3": lambda x, y, n: (y, n - x),
    "mx": lambda x, y, n: (n - x, y),
    "my": lambda x, y, n: (x, n - y),
    "d": lambda x, y, n: (y, x),
    "a": lambda x, y, n: (n - y, n - x),
}

# The four elements that exchange the x and y axes. A footprint whose span is
# not square cannot be expressed under one of these (no const carries the
# swapped shape), and a wall's stored run-direction index has to swap with
# them -- Resolution 1's "axis-swapping class".
AXIS_SWAPPING = frozenset({"r", "r3", "d", "a"})

# Wall variant indices, from unit_sprites.wall_variant_from_neighbours: 0 is a
# run along +-x, 1 a run along +-y. 2 (tower/corner/junction) is fixed by
# every element, and 3/4 have no mask correspondence at all, so both pass
# through -- the same measured decision wall_variant_from_neighbours makes by
# returning None for mask 0.
_WALL_AXIS_SWAP = {0.0: 1.0, 1.0: 0.0}
_WALL_ANGLE_COUNT = 5


@lru_cache(maxsize=1)
def _element_names() -> dict[tuple[tuple[int, int], ...], str]:
    """Lookup from an element's action on a fixed probe lattice to its name,
    so composition and inversion are derived from TRANSFORMS themselves rather
    than from a second hand-typed multiplication table."""
    probes = [(0, 0), (1, 0), (0, 1), (2, 1)]
    return {tuple(fn(x, y, 5) for x, y in probes): name for name, fn in TRANSFORMS.items()}


def compose(after: str, before: str) -> str:
    """The single D4 element equal to `after` applied to `before`'s result."""
    probes = [(0, 0), (1, 0), (0, 1), (2, 1)]
    signature = tuple(TRANSFORMS[after](*TRANSFORMS[before](x, y, 5), 5) for x, y in probes)
    return _element_names()[signature]


def invert(name: str) -> str:
    """`name`'s inverse in D4 -- every reflection is its own, the rotations pair up."""
    return next(other for other in TRANSFORMS if compose(name, other) == "id")


# A gate's four orientations by RUN DIRECTION, in gate_orientation's own cycle
# order (A, C, B, D = 0, 45, 90, 135 degrees in tile space). Direction is mod
# 180 degrees -- a gate runs along a line, it does not point along it.
_RUN_DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1))


def _linear_part(name: str) -> Callable[[int, int], tuple[int, int]]:
    """`name`'s action on a direction vector: the transform minus its own
    translation, measured from TRANSFORMS rather than hand-typed."""
    fn = TRANSFORMS[name]
    n = 16
    ox, oy = fn(0, 0, n)

    def apply(dx: int, dy: int) -> tuple[int, int]:
        x, y = fn(dx, dy, n)
        return (x - ox, y - oy)

    return apply


@cache
def gate_orientation_map(name: str) -> tuple[int, ...]:
    """Where `name` sends each gate orientation index (0 ne, 1 e, 2 se, 3 n).

    Derived from TRANSFORMS' own linear action on the four run directions, so
    it cannot drift from the transforms the positions use. Reflections come
    out as transpositions rather than cyclic shifts, which is why
    gate_orientation.cycle_const() cannot express them.
    """
    linear = _linear_part(name)
    mapped = []
    for dx, dy in _RUN_DIRECTIONS:
        vx, vy = linear(dx, dy)
        if (vx, vy) not in _RUN_DIRECTIONS:
            vx, vy = -vx, -vy  # same line, opposite sense
        mapped.append(_RUN_DIRECTIONS.index((vx, vy)))
    return tuple(mapped)


def reorient_gate_const(unit_const: int, name: str) -> int:
    """The orientation sibling a gate becomes under D4 element `name`.

    Raises for a non-gate, matching gate_orientation.cycle_const()'s
    loud-refusal contract. The six 1x1 corner groups pass through verbatim:
    all four of their siblings share one graphic and a 1x1 footprint is
    invariant under every element, so remapping would churn the stored const
    for no visual effect.
    """
    siblings = gate_orientation.orientation_siblings(unit_const)
    if siblings is None:
        raise ValueError(f"unit_const {unit_const} is not a gate: it has no orientation siblings")
    if all(tile_span(c, render.NON_BUILDING_SPAN) == (1, 1) for c in siblings):
        return unit_const
    return siblings[gate_orientation_map(name)[siblings.index(unit_const)]]


@dataclass(frozen=True)
class UnitImage:
    """One unit to create: a source unit's image under `element`."""
    source: object
    element: str
    player: int
    unit_const: int
    x: float
    y: float
    rotation: float


@dataclass
class UnitMirrorPlan:
    # New units to add, in emit order, and the destination-slice units to
    # remove first. Both empty when the map is already symmetric.
    images: list[UnitImage]
    removals: list[object]
    # Refusal lists. Any non-empty one means "do not apply" -- the caller
    # reports them rather than repairing, exactly as Stage 1's elevation seam
    # check does, since every repair would break the symmetry that was asked
    # for.
    garrisoned_blockers: list[object]
    straddling: list[object]
    # Report-only: a source unit whose footprint is not square cannot be
    # reflected under an axis-swapping element, because no const carries the
    # swapped shape. Its image keeps the original footprint orientation.
    # Gates are the one family that CAN (their four orientations are four
    # consts), so they are reoriented rather than listed here.
    unsquare_spans: list[object]
    # Report-only: gates whose image carries a different orientation const
    # than its source (Stage 2b).
    gates: list[object]

    @property
    def blocked(self) -> bool:
        return bool(self.garrisoned_blockers or self.straddling)


def _key(x: float, y: float) -> tuple[int, int]:
    """Position identity for orbit dedup. Rounded rather than compared raw:
    positions are float32-derived, and an on-axis unit's image must land on
    itself exactly."""
    return (round(x * 1e6), round(y * 1e6))


def _image_position(
    unit_const: int, x: float, y: float, n: int, element: str, new_const: int | None = None
) -> tuple[float, float]:
    """Where `element` sends a unit at (x, y).

    A span-1x1 unit transforms continuously, keeping its arbitrary sub-tile
    position. Anything larger is a building, whose anchor is derived from the
    reflected footprint's own low corner instead -- the same span_low_corner/
    span_anchor round trip UnitEditModel.set_unit_const() uses, and the only
    correct treatment of the diagonal gates' sparse footprints.

    `new_const` is the reoriented gate sibling (Stage 2b), whose span is the
    one the anchor has to satisfy: ne is (4, 1) and se is (1, 4), so reusing
    the source's span would leave the image half a footprint off its tiles.
    """
    span_x, span_y = tile_span(unit_const, render.NON_BUILDING_SPAN)
    if span_x == 1 and span_y == 1:
        image_x, image_y = POSITION_TRANSFORMS[element](x, y, n)
        # A tile spans [t, t+1), so a coordinate sitting exactly ON a tile
        # boundary (2.5% of 1x1 corpus placements) reflects onto the far
        # boundary, which int() reads as the NEXT tile along. Snapping it back
        # is what keeps the image's tile the reflection of the source's --
        # without it a mirrored forest sits one tile off its own terrain, and
        # mirroring twice drifts.
        tile_x, tile_y = TRANSFORMS[element](int(x), int(y), n)
        return (
            image_x - 1.0 if int(image_x) != tile_x else image_x,
            image_y - 1.0 if int(image_y) != tile_y else image_y,
        )
    fn = TRANSFORMS[element]
    low_x = render._span_start(x, span_x)
    low_y = render._span_start(y, span_y)
    corners = [fn(low_x, low_y, n), fn(low_x + span_x - 1, low_y + span_y - 1, n)]
    target_span = tile_span(unit_const if new_const is None else new_const, render.NON_BUILDING_SPAN)
    return render.span_anchor(
        min(c[0] for c in corners), min(c[1] for c in corners), *target_span
    )


def _image_rotation(unit, element: str, file_radian: bool) -> float:
    """A wall's stored run-direction index swaps with the axis, in a file that
    encodes it as a literal index; every other rotation passes through
    verbatim.

    In a radian-encoded file the stored value carries no shape information
    (measured: mask 1100 splits 142/101/79/85/82 across the five indices), and
    both DEscape and the game derive the shape from connectivity there, so
    there is nothing meaningful to permute. Non-wall rotations are verbatim
    because nothing has measured what a mirror should do to a real facing
    angle: a mirrored archer keeps the direction it faced, which is a known
    Stage 2 gap rather than a derived answer.
    """
    rotation = float(unit.rotation)
    if element not in AXIS_SWAPPING or file_radian:
        return rotation
    if not unit_sprites.rotation_variant_eligible(unit.unit_const):
        return rotation
    index = float(unit_sprites.variant_index(rotation, _WALL_ANGLE_COUNT))
    return _WALL_AXIS_SWAP.get(index, rotation)


def _rotated_owner(player: int, player_ids: Sequence[int], steps: int) -> int:
    """`player` advanced `steps` positions along the scenario's own defined
    player list. GAIA (0) is fixed, and so is any owner outside that list --
    rotating one would invent a player the file does not define."""
    if steps == 0 or player == 0 or player not in player_ids:
        return player
    return player_ids[(player_ids.index(player) + steps) % len(player_ids)]


def plan_mirror_units(
    mm,
    mode_id: int,
    slice_index: int,
    units_by_player: Sequence[Sequence],
    source_indices: frozenset[int],
    player_ids: Sequence[int] = (),
    ownership_steps: int = 0,
    referencing: Callable[[object], Sequence] | None = None,
) -> UnitMirrorPlan:
    """Computes (but does not apply) the unit half of a mirror.

    `source_indices` is plan_mirror()'s own output, passed in rather than
    recomputed: the two halves must agree about which tiles are the source,
    and a second derivation could disagree on a degenerate orbit.

    A unit belongs to the source slice by its ANCHOR tile ((int(x), int(y))),
    matching region_clipboard's convention -- a footprint-based test would let
    one building belong to two slices at once, which makes "one unit, one
    orbit" ill-defined. A building anchored just outside the slice whose
    footprint spills in is caught by the straddle sweep instead.

    Every unit outside the source slice is removed and re-created from the
    source, so the destination slices end up an exact copy rather than a
    merge. `referencing` is UnitEditModel.referencing: a removal it refuses
    (a garrisoned unit) is reported, never forced.
    """
    mode = MODE_BY_ID[mode_id]
    n = mm.map_width
    width, height = mm.map_width, mm.map_height
    source_element = mode.group[slice_index]
    # The element that carries a SOURCE unit onto each destination slice:
    # destination slice h is sourced through h . source^-1, which is the whole
    # group again but re-indexed, so a non-zero slice_index still yields
    # identity first (a source unit's own position).
    inverse_source = invert(source_element)
    elements = [compose(h, inverse_source) for h in mode.group]

    sources: list[object] = []
    removals: list[object] = []
    for units in units_by_player:
        for unit in units:
            tx, ty = int(unit.x), int(unit.y)
            if not (0 <= tx < width and 0 <= ty < height):
                continue  # already off-map; nothing this operation can place
            (sources if ty * width + tx in source_indices else removals).append(unit)

    garrisoned_blockers = []
    if referencing is not None:
        kept = {id(unit) for unit in sources}
        garrisoned_blockers.extend(
            unit for unit in removals if any(id(other) in kept for other in referencing(unit))
        )

    eligible_rotations = [
        float(unit.rotation)
        for units in units_by_player
        for unit in units
        if unit_sprites.rotation_variant_eligible(unit.unit_const)
    ]
    file_radian = unit_sprites.file_is_radian(eligible_rotations, _WALL_ANGLE_COUNT)

    source_tiles: set[tuple[int, int]] = set()
    for unit in sources:
        source_tiles.update(render.unit_occupied_tiles(unit, width, height) or ())

    images: list[UnitImage] = []
    straddling: list[object] = []
    unsquare_spans: list[object] = []
    gates: list[object] = []
    for unit in sources:
        span_x, span_y = tile_span(unit.unit_const, render.NON_BUILDING_SPAN)
        is_gate = gate_orientation.is_gate(unit.unit_const)
        seen = {_key(unit.x, unit.y)}
        reported_span = False
        for index, element in enumerate(elements):
            # Stage 2b: a gate's orientation lives in its const, so the image
            # gets the sibling the element maps it to -- and its anchor comes
            # from that sibling's own span, which is why the const is resolved
            # before the position.
            new_const = reorient_gate_const(unit.unit_const, element) if is_gate else unit.unit_const
            x, y = _image_position(unit.unit_const, unit.x, unit.y, n, element, new_const)
            if _key(x, y) in seen and new_const == unit.unit_const:
                continue  # a degenerate orbit: this element maps the unit onto itself
            seen.add(_key(x, y))
            if new_const != unit.unit_const and unit not in gates:
                gates.append(unit)
            if span_x != span_y and element in AXIS_SWAPPING and not is_gate and not reported_span:
                unsquare_spans.append(unit)
                reported_span = True
            tiles = render.occupied_tiles_for(new_const, x, y, width, height)
            if tiles and source_tiles.intersection(tiles):
                # An image overlapping a KEPT source unit means the source's
                # own footprint crosses the symmetry axis: dedup cannot catch
                # it (the two anchors genuinely differ), and no repair exists
                # that preserves the symmetry. Reported once per unit however
                # many of its images overlap.
                if unit not in straddling:
                    straddling.append(unit)
                continue
            images.append(
                UnitImage(
                    source=unit,
                    element=element,
                    player=_rotated_owner(unit.player, player_ids, index * ownership_steps),
                    unit_const=new_const,
                    x=x,
                    y=y,
                    rotation=_image_rotation(unit, element, file_radian),
                )
            )

    return UnitMirrorPlan(
        images=images,
        removals=removals,
        garrisoned_blockers=garrisoned_blockers,
        straddling=straddling,
        unsquare_spans=unsquare_spans,
        gates=gates,
    )
