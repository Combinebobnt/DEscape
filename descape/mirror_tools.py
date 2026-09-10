"""Map mirroring (symmetry generator) for the Terrain/Elevation half of the
feature (Stage 1). Units are Stage 2, blocked on phase 3.5 (a Units write
path and a fourth EditHistory record type don't exist yet); this module
never touches them.

Follows fill_tools.py/brush.py's shape: pure index math, no PyQt5, no
descape.settings, and no AoE2ScenarioParser import either -- duck-typed on
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
from dataclasses import dataclass
from typing import Callable

from descape.edit_history import TileState, tile_state

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
