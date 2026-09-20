"""Random unit distribution: scatter units across a caller-supplied tile set.

e.g. forty GAIA fish at reproducible random positions across a pond. No
in-game editor equivalent exists (Map Copy is deterministic, and the only
"random" in its Units tab is rotation during manual placement).

**Region-building is the caller's job.** scatter_units() takes an explicit
tile set and never builds one: a predicate over mm.terrain, a BFS over an
is_water() set, or a rectangle comprehension. fill_tools.contiguous_region()
keys on exact terrain_id equality, so it can't express "a body of water" (a
bay of WATER + WATER_DEEP + SHALLOWS comes back as three regions).

**rotation is a pass-through, never randomized.** For ~65% of GAIA objects
(and every wall and gate) `rotation` is a graphic-variant index, not an
angle -- see AGENTS.md's hard rules. random.uniform(0, 2 * math.pi) is
therefore banned here: it would write variant indices nobody has measured.
A caller who has confirmed a set of values valid for its const passes them
as rotation_choices; otherwise every unit gets 0.0.
initial_animation_frame gets the same treatment, and is the field to vary
if the goal is forty fish not animating in lockstep.

**Determinism.** A local random.Random(seed), never module-global random.*,
and the eligible tiles are sorted before any draw, so the same seed gives
the same placements whether tiles arrive as a list, a set or a generator.

**Undo is the caller's job too.** Wrap the call in
units.begin_unit_edit([player]) / units.commit_unit_edit(label, history) for
one undo step; scatter never touches EditHistory.

Leaf module: batch_api.py must not import fill_tools.py, so a scatter helper
living inside batch_api could never be combined with it. A batch script
imports both.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from AoE2ScenarioParser.objects.data_objects.unit import Unit

from descape.scenario_io import LoadedScenario
from descape.unit_model import UnitEditModel

# Strictly below 0.5 so a jittered unit's int(x)/int(y) never leaves its tile.
MAX_JITTER = 0.499


class _Spec(NamedTuple):
    """The five fields UnitEditModel.add_many() reads per unit."""

    x: float
    y: float
    unit_const: int
    rotation: float
    initial_animation_frame: int


def occupied_tiles(scenario: LoadedScenario) -> set[tuple[int, int]]:
    """Every on-map tile covered by an existing unit's footprint, all nine
    player lists. Footprints come from render.unit_tile_bounds(), the same
    oracle unit_pick.build_index() uses; off-map units cover nothing."""
    # Deferred: render is heavy, and only avoid_occupied needs it.
    from descape.render import unit_tile_bounds

    mm = scenario.map_manager
    width, height = mm.map_width, mm.map_height
    covered: set[tuple[int, int]] = set()
    for player_units in scenario.unit_manager.units:
        for unit in player_units:
            bounds = unit_tile_bounds(unit, width, height)
            if bounds is None:
                continue
            x0, x1, y0, y1 = bounds
            covered.update((x, y) for y in range(y0, y1) for x in range(x0, x1))
    return covered


def _normalize_tiles(tiles: Iterable, width: int, height: int) -> list[tuple[int, int]]:
    """Sorted, de-duplicated, on-map (x, y) pairs. Filters against the map
    here because nothing downstream does: unit_fields' coordinate limits are
    +/-2**15, not map bounds."""
    eligible: set[tuple[int, int]] = set()
    for tile in tiles:
        x, y = tile if isinstance(tile, tuple) else (tile.x, tile.y)
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            eligible.add((x, y))
    return sorted(eligible)


def _requested_count(
    rng: random.Random,
    count: int | tuple[int, int] | None,
    density: float | None,
    eligible: int,
) -> int:
    if (count is None) == (density is None):
        raise ValueError("pass exactly one of count or density")
    if density is not None:
        if density < 0:
            raise ValueError(f"density must be >= 0, got {density}")
        return math.ceil(density * eligible)
    if isinstance(count, tuple):
        lo, hi = count
        if not 0 <= lo <= hi:
            raise ValueError(f"count range must satisfy 0 <= lo <= hi, got {count}")
        return rng.randint(lo, hi)
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    return count


def _pick_tiles(
    rng: random.Random, eligible: list[tuple[int, int]], wanted: int, min_spacing: int
) -> list[tuple[int, int]]:
    """Without replacement, so at most one unit per tile. min_spacing > 1 is
    greedy over a shuffled order and may return fewer than `wanted`."""
    if min_spacing <= 1:
        return rng.sample(eligible, min(wanted, len(eligible)))
    order = eligible[:]
    rng.shuffle(order)
    reach = min_spacing - 1
    blocked: set[tuple[int, int]] = set()
    picked: list[tuple[int, int]] = []
    for x, y in order:
        if len(picked) >= wanted:
            break
        if (x, y) in blocked:
            continue
        picked.append((x, y))
        blocked.update((x + dx, y + dy) for dy in range(-reach, reach + 1) for dx in range(-reach, reach + 1))
    return picked


def scatter_units(
    scenario: LoadedScenario,
    units: UnitEditModel,
    tiles: Iterable,
    unit_consts: int | Sequence[int],
    *,
    player: int = 0,
    count: int | tuple[int, int] | None = None,
    density: float | None = None,
    weights: Sequence[float] | None = None,
    seed: int | None = None,
    jitter: float = 0.0,
    min_spacing: int = 0,
    avoid_occupied: bool = False,
    rotation_choices: Sequence[float] | None = None,
    animation_frames: Sequence[int] | None = None,
) -> list[Unit]:
    """Places units for `player` on randomly chosen tiles from `tiles`
    ((x, y) pairs or TerrainTiles) and returns the units actually placed.

    Exactly one of `count` (an int, or an inclusive (lo, hi) range drawn
    from the same RNG) or `density` (a fraction of eligible tiles, rounded
    up) must be given. The requested number is clamped to what is eligible,
    so asking for 200 fish in a 30-tile pond places 30; the returned list is
    the truthful count.

    `unit_consts` is one const or a mix, chosen per unit (by `weights` if
    given). Consts are passed through unchecked, same as add().

    Units sit at tile centres (x + 0.5) unless `jitter` > 0 opts into
    x + 0.5 +/- jitter, clamped to MAX_JITTER. z is always 0.0, never derived
    from terrain elevation. `min_spacing` is a Chebyshev tile distance,
    best-effort. `avoid_occupied` drops tiles under any existing unit's
    footprint (see occupied_tiles()).
    """
    if jitter < 0:
        raise ValueError(f"jitter must be >= 0, got {jitter}")
    consts = [unit_consts] if isinstance(unit_consts, int) else list(unit_consts)
    if not consts:
        raise ValueError("unit_consts must not be empty")
    if weights is not None and len(weights) != len(consts):
        raise ValueError(f"weights has {len(weights)} entries for {len(consts)} unit_consts")
    if rotation_choices is not None and not rotation_choices:
        raise ValueError("rotation_choices must be None or non-empty")
    if animation_frames is not None and not animation_frames:
        raise ValueError("animation_frames must be None or non-empty")

    mm = scenario.map_manager
    eligible = _normalize_tiles(tiles, mm.map_width, mm.map_height)
    if avoid_occupied:
        covered = occupied_tiles(scenario)
        eligible = [t for t in eligible if t not in covered]

    rng = random.Random(seed)
    wanted = _requested_count(rng, count, density, len(eligible))
    picked = _pick_tiles(rng, eligible, wanted, min_spacing)
    jitter = min(jitter, MAX_JITTER)

    specs = []
    for x, y in picked:
        if len(consts) == 1:
            const = consts[0]
        elif weights is not None:
            const = rng.choices(consts, weights=weights)[0]
        else:
            const = rng.choice(consts)
        # Verbatim caller-supplied values only; see the module docstring.
        rotation = rng.choice(rotation_choices) if rotation_choices is not None else 0.0
        frame = rng.choice(animation_frames) if animation_frames is not None else 0
        ux, uy = x + 0.5, y + 0.5
        if jitter:
            ux += rng.uniform(-jitter, jitter)
            uy += rng.uniform(-jitter, jitter)
        specs.append(_Spec(ux, uy, const, rotation, frame))

    if not specs:
        return []
    return units.add_many(player, specs)
