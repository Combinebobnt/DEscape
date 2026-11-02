"""Coverage for descape/elevation_tools.py's set_tiles_elevation() -- the
multi-tile counterpart to set_tile_elevation() added for the brush feature.

Needs a real AoE2ScenarioParser MapManager (the propagation is a vendored copy
of the library's own private method, checked against the original below, and
a fake object can't stand in for either), loaded from the shipped blank template -- same fixture
test_write_path.py uses -- but otherwise Qt-free, no gui marker.

The central claim under test: calling set_tile_elevation() once per tile in a
loop is NOT equivalent to set_tiles_elevation() on the same targets, because
_elevation_tile_recursion()'s `xys` argument (the tiles it must never
overwrite) is a single tile for the former and the whole footprint for the
latter. See set_tiles_elevation()'s own docstring and this project's
implementation plan for the empirical finding that motivated this split.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from descape.elevation_tools import _propagate, cone_clamp, set_tile_elevation, set_tiles_elevation
from descape.iso_geometry import MAX_ELEVATION
from descape.map_analysis import check_elevation
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

import conftest

# A 5x5 circle brush footprint (BRUSH_SHAPE_CIRCLE, size 5), transcribed
# rather than imported from descape.brush -- this module's own contract
# shouldn't depend on brush.py's offsets staying exactly this shape.
_CIRCLE5 = [
    (-1, -2), (0, -2), (1, -2),
    (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
    (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
    (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
    (-1, 2), (0, 2), (1, 2),
]
_CIRCLE5_CORNERS = [(-2, -2), (2, -2), (-2, 2), (2, 2)]


def _load():
    return load_map_and_units(BLANK_TEMPLATE_PATH)


def test_circle_footprint_delta_one_leaves_corners_and_outside_untouched() -> None:
    scenario = _load()
    mm = scenario.map_manager
    cx, cy = 40, 40
    targets = [(cx + dx, cy + dy, 1) for dx, dy in _CIRCLE5]
    set_tiles_elevation(mm, targets)

    for dx, dy in _CIRCLE5:
        assert mm.get_tile(cx + dx, cy + dy).elevation == 1
    for dx, dy in _CIRCLE5_CORNERS:
        assert mm.get_tile(cx + dx, cy + dy).elevation == 0

    for y in range(cy - 4, cy + 5):
        for x in range(cx - 4, cx + 5):
            if (x - cx, y - cy) in _CIRCLE5:
                continue
            assert mm.get_tile(x, y).elevation == 0, f"leaked outside the footprint at ({x}, {y})"


def test_matches_set_elevation_for_a_rectangle() -> None:
    # set_tiles_elevation is meant to reproduce MapManager.set_elevation's
    # own multi-tile rectangle branch, just for an arbitrary tile set
    # instead of only a rectangle -- confirm the two agree on a rectangle.
    left = _load()
    right = _load()
    x1, y1, x2, y2 = 30, 30, 33, 33  # 4x4 block

    left.map_manager.set_elevation(3, x1, y1, x2, y2)

    targets = [(x, y, 3) for y in range(y1, y2 + 1) for x in range(x1, x2 + 1)]
    set_tiles_elevation(right.map_manager, targets)

    lmm, rmm = left.map_manager, right.map_manager
    for y in range(y1 - 3, y2 + 4):
        for x in range(x1 - 3, x2 + 4):
            assert lmm.get_tile(x, y).elevation == rmm.get_tile(x, y).elevation, f"mismatch at ({x}, {y})"


def test_large_delta_radiates_a_ramp_but_keeps_the_plateau_shape() -> None:
    # Documents the accepted, desired behavior (matches the in-game editor):
    # a big single-stroke jump legally ramps down outside the brush rather
    # than leaving an illegal cliff. The PLATEAU stays exactly the circle;
    # what spreads is the ramp around it.
    scenario = _load()
    mm = scenario.map_manager
    cx, cy = 40, 40
    targets = [(cx + dx, cy + dy, 5) for dx, dy in _CIRCLE5]
    set_tiles_elevation(mm, targets)

    for dx, dy in _CIRCLE5:
        assert mm.get_tile(cx + dx, cy + dy).elevation == 5
    # The ramp is real: at least one tile just outside the footprint is
    # non-zero (an illegal-cliff-avoiding implementation could not leave
    # every neighbor at 0 next to a level-5 plateau).
    assert any(mm.get_tile(cx + dx, cy + dy).elevation != 0 for dx, dy in _CIRCLE5_CORNERS)
    # And every adjacent step is legal (AoE2's own +/-1 rule).
    for y in range(cy - 6, cy + 7):
        for x in range(cx - 6, cx + 7):
            e = mm.get_tile(x, y).elevation
            for nx, ny in ((x + 1, y), (x, y + 1)):
                if cx - 6 <= nx <= cx + 6 and cy - 6 <= ny <= cy + 6:
                    assert abs(e - mm.get_tile(nx, ny).elevation) <= 1


def test_per_tile_loop_diverges_from_batched_on_uneven_terrain() -> None:
    # The regression this whole function exists to fix: looping
    # set_tile_elevation() (single-tile xys) over a footprint lets each
    # call's propagation rewrite an EARLIER call's already-set tile, because
    # xys only ever excludes that one call's own tile. On flat ground the
    # two approaches happen to agree (nothing to propagate into); this test
    # seeds a bumpy starting elevation so they don't.
    looped = _load()
    batched = _load()
    cx, cy = 40, 40
    footprint = [(cx + dx, cy + dy) for dx in range(-2, 3) for dy in range(-2, 3)]

    # A deterministic, non-flat pattern (not random -- keeps the test
    # reproducible) with real +/-1 steps between neighbors so it's a legal
    # starting map, seeded well outside the footprint too so the recursion
    # has real terrain to propagate into on every side.
    for mm in (looped.map_manager, batched.map_manager):
        for y in range(cy - 6, cy + 7):
            for x in range(cx - 6, cx + 7):
                mm.get_tile(x, y).elevation = (x + y) % 3

    for x, y in footprint:
        cur = looped.map_manager.get_tile(x, y).elevation
        set_tile_elevation(looped.map_manager, x, y, min(7, cur + 1))

    targets = [(x, y, min(7, batched.map_manager.get_tile(x, y).elevation + 1)) for x, y in footprint]
    set_tiles_elevation(batched.map_manager, targets)

    looped_grid = [looped.map_manager.get_tile(x, y).elevation for x, y in footprint]
    batched_grid = [batched.map_manager.get_tile(x, y).elevation for x, y in footprint]
    assert looped_grid != batched_grid, "expected the two approaches to diverge on uneven terrain"

    # The batched result is the one that's actually correct: every footprint
    # tile is exactly its own pre-edit elevation + 1.
    for x, y in footprint:
        expected = min(7, ((x + y) % 3) + 1)
        assert batched.map_manager.get_tile(x, y).elevation == expected


def _library_set_tiles_elevation(mm, targets) -> None:
    """set_tiles_elevation() as it was before the recursion was vendored:
    the library's own private method, the oracle for the copy."""
    footprint = {(x, y) for x, y, _ in targets}
    tiles = []
    for x, y, elevation in targets:
        tile = mm.get_tile(x, y)
        tile.elevation = elevation
        tiles.append(tile)
    for tile in tiles:
        mm._elevation_tile_recursion(tile, footprint)


def _seed_elevations(mms, rng, cx, cy, radius, legal) -> None:
    # legal=True: neighbours differ by at most 1, the maps real files hold.
    # legal=False: arbitrary 0..7, so the `abs(diff) > 1` recursive branch fires often.
    grid = {}
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if legal:
                base = grid.get((x - 1, y), grid.get((x, y - 1), rng.randint(0, 7)))
                grid[(x, y)] = max(0, min(7, base + rng.choice((-1, 0, 1))))
                if (x, y - 1) in grid and abs(grid[(x, y)] - grid[(x, y - 1)]) > 1:
                    grid[(x, y)] = grid[(x, y - 1)] + (1 if grid[(x, y)] > grid[(x, y - 1)] else -1)
            else:
                grid[(x, y)] = rng.randint(0, 7)
    for mm in mms:
        for (x, y), e in grid.items():
            mm.get_tile(x, y).elevation = e


@pytest.mark.parametrize("legal", [True, False], ids=["legal", "cliffy"])
@pytest.mark.parametrize("seed", range(12))
def test_vendored_recursion_matches_the_library_and_reports_every_change(seed, legal) -> None:
    rng = random.Random(seed)
    ours, theirs = _load(), _load()
    omm, tmm = ours.map_manager, theirs.map_manager
    cx, cy = rng.randint(20, 100), rng.randint(20, 100)
    _seed_elevations((omm, tmm), rng, cx, cy, 9, legal)
    before = [t.elevation for t in omm.terrain]

    size = rng.choice((1, 3, 5, 9))
    half = size // 2
    footprint = [(cx + dx, cy + dy) for dx in range(-half, half + 1) for dy in range(-half, half + 1)]
    mode = rng.choice(("raise", "lower", "level"))
    level = rng.randint(0, 7)

    def target(x, y):
        e = omm.get_tile(x, y).elevation
        return {"raise": min(7, e + 1), "lower": max(0, e - 1), "level": level}[mode]

    targets = [(x, y, target(x, y)) for x, y in footprint]
    touched: set[int] = set()
    _propagate(omm, targets, touched)
    _library_set_tiles_elevation(tmm, targets)

    after = [t.elevation for t in omm.terrain]
    assert after == [t.elevation for t in tmm.terrain]
    changed = {i for i, (b, a) in enumerate(zip(before, after, strict=True)) if b != a}
    assert changed, "the fixture changed nothing, so the reporting check is vacuous"
    assert changed <= touched
    assert {y * omm.map_size + x for x, y in footprint} <= touched


@pytest.mark.parametrize("seed", range(6))
def test_single_tile_vendored_recursion_matches_the_library(seed) -> None:
    rng = random.Random(1000 + seed)
    ours, theirs = _load(), _load()
    omm, tmm = ours.map_manager, theirs.map_manager
    cx, cy = rng.randint(20, 100), rng.randint(20, 100)
    _seed_elevations((omm, tmm), rng, cx, cy, 6, seed % 2 == 0)
    before = [t.elevation for t in omm.terrain]
    level = rng.randint(0, 7)

    touched: set[int] = set()
    _propagate(omm, [(cx, cy, level)], touched)
    tile = tmm.get_tile(cx, cy)
    tile.elevation = level
    tmm._elevation_tile_recursion(tile, {tile.xy})

    after = [t.elevation for t in omm.terrain]
    assert after == [t.elevation for t in tmm.terrain]
    assert {i for i, (b, a) in enumerate(zip(before, after, strict=True)) if b != a} <= touched


# TASK-257: tiles the gap-fill rule raises outside a hollow or thin brush, sizes 1-9, raising
# flat ground by 1 (user decision 2026-10-13: the propagation stays). Measured, not derived.
_GAP_FILL_EXTRA = {
    ("square_outline", 1): [0, 0, 1, 4, 4, 4, 4, 4, 4],
    ("square_outline", 2): [0, 0, 0, 0, 1, 4, 4, 4, 4],
    ("square_outline", 3): [0, 0, 0, 0, 0, 0, 1, 4, 4],
    ("square_outline", 4): [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ("circle_outline", 1): [0, 0, 1, 4, 4, 0, 0, 0, 0],
    ("circle_outline", 2): [0, 0, 0, 0, 1, 4, 4, 0, 0],
    ("circle_outline", 3): [0, 0, 0, 0, 0, 0, 1, 4, 4],
    ("circle_outline", 4): [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ("plus", 1): [0, 0, 0, 0, 4, 4, 4, 4, 4],
    ("rounded", 1): [0] * 9,
    ("line_x", 1): [0] * 9,
    ("line_y", 1): [0] * 9,
}


def _raised_outside(shape: str, thickness: int, size: int) -> int:
    from descape import brush

    scenario = _load()
    mm = scenario.map_manager
    assert all(t.elevation == 0 for t in mm.terrain)
    footprint = set(brush.brush_tiles(40, 40, size, shape, mm.map_width, mm.map_height, thickness))
    set_tiles_elevation(mm, [(x, y, 1) for x, y in footprint])
    assert all(mm.get_tile(x, y).elevation == 1 for x, y in footprint)
    lo, hi = 40 - (size - 1) // 2, 40 + size // 2
    extra = [(i % mm.map_size, i // mm.map_size) for i, t in enumerate(mm.terrain) if t.elevation and
             (i % mm.map_size, i // mm.map_size) not in footprint]
    assert all(lo <= x <= hi and lo <= y <= hi for x, y in extra), "gap-fill stays inside the brush box"
    return len(extra)


@pytest.mark.parametrize(("shape", "thickness"), list(_GAP_FILL_EXTRA))
def test_raising_a_hollow_or_thin_brush_on_flat_ground_fills_these_gaps(shape: str, thickness: int) -> None:
    assert [_raised_outside(shape, thickness, size) for size in range(1, 10)] == _GAP_FILL_EXTRA[(shape, thickness)]


def test_lowering_a_square_outline_writes_exactly_the_ring() -> None:
    from descape import brush

    scenario = _load()
    mm = scenario.map_manager
    for tile in mm.terrain:
        tile.elevation = 2
    ring = set(brush.brush_tiles(40, 40, 5, brush.BRUSH_SHAPE_SQUARE_OUTLINE, mm.map_width, mm.map_height))
    assert len(ring) == 16
    set_tiles_elevation(mm, [(x, y, 1) for x, y in ring])
    lowered = {(i % mm.map_size, i // mm.map_size) for i, t in enumerate(mm.terrain) if t.elevation != 2}
    assert lowered == ring


def _flat(mm) -> list[int]:
    return [t.elevation for t in mm.terrain]


def _changed(before: list[int], mm) -> set[int]:
    return {i for i, (b, a) in enumerate(zip(before, _flat(mm), strict=True)) if b != a}


def _hill_targets() -> list[tuple[int, int, int]]:
    return conftest.illegal_step_hill_targets(*conftest.ILLEGAL_STEP_HILL_ANCHOR)


def test_a_pasted_hill_leaves_no_illegal_step_and_writes_only_the_repair() -> None:
    targets = _hill_targets()
    rx, ry = conftest.ILLEGAL_STEP_HILL_REPAIRED_TILE
    unrepaired = _load()
    umm = unrepaired.map_manager
    _propagate(umm, targets, set())
    assert not check_elevation(unrepaired).is_clean, "the repro no longer leaves a step, so this test is vacuous"

    scenario = _load()
    mm = scenario.map_manager
    start = _flat(mm)
    touched = set_tiles_elevation(mm, targets)

    assert check_elevation(scenario).is_clean
    assert all(mm.get_tile(x, y).elevation == e for x, y, e in targets)
    repair_index = ry * mm.map_size + rx
    assert umm.terrain[repair_index].elevation == 0 and mm.terrain[repair_index].elevation == 1
    assert {i for i, (a, b) in enumerate(zip(_flat(umm), _flat(mm), strict=True)) if a != b} == {repair_index}
    assert repair_index in touched
    assert _changed(start, mm) <= touched


def test_a_one_tile_edit_beside_a_hill_leaves_no_illegal_step() -> None:
    # set_tile_elevation() is batch_api's path: it must inherit the repair.
    hill = conftest.illegal_step_hill_targets(40, 40)
    unrepaired, scenario = _load(), _load()
    set_tiles_elevation(unrepaired.map_manager, hill)
    _propagate(unrepaired.map_manager, [(44, 40, 2)], set())
    assert not check_elevation(unrepaired).is_clean, "the repro no longer leaves a step, so this test is vacuous"

    mm = scenario.map_manager
    set_tiles_elevation(mm, hill)
    start = _flat(mm)
    touched = set_tile_elevation(mm, 44, 40, 2)

    assert check_elevation(scenario).is_clean
    assert mm.get_tile(44, 40).elevation == 2
    assert _changed(start, mm) <= touched


def _steep_grid(rng: random.Random, size: int, top: int = MAX_ELEVATION) -> np.ndarray:
    # max/min of Chebyshev cones, so every 8-neighbour step stays within 1: a steep legal map.
    yy, xx = np.mgrid[:size, :size]
    grid = np.full((size, size), rng.randint(0, top), dtype=np.int64)
    for _ in range(size):
        cone = np.maximum(np.abs(xx - rng.randrange(size)), np.abs(yy - rng.randrange(size)))
        level = rng.randint(0, top)
        grid = np.maximum(grid, level - cone) if rng.random() < 0.5 else np.minimum(grid, level + cone)
    return grid


def _write_grid(mm, grid: np.ndarray) -> None:
    for tile, elevation in zip(mm.terrain, grid.ravel().tolist(), strict=True):
        tile.elevation = elevation


def _call_targets(
    shape: str, rng: random.Random, grid: np.ndarray, top: int = MAX_ELEVATION
) -> list[tuple[int, int, int]]:
    size = len(grid)
    if shape == "set_one":
        return [(rng.randrange(size), rng.randrange(size), rng.randint(0, top))]
    if shape == "paste":
        side = rng.randint(3, 12)
        source = _steep_grid(rng, size, top)
        sx, sy = rng.randrange(size - side), rng.randrange(size - side)
        ax, ay = rng.randrange(-side // 2, size - side // 2), rng.randrange(-side // 2, size - side // 2)
        tiles = [(ax + dx, ay + dy, int(source[sy + dy, sx + dx])) for dy in range(side) for dx in range(side)]
        return [(x, y, e) for x, y, e in tiles if 0 <= x < size and 0 <= y < size]
    side = rng.randint(1, 5)
    ax, ay = rng.randrange(size - side), rng.randrange(size - side)
    tiles = [(ax + dx, ay + dy) for dy in range(side) for dx in range(side)]
    if shape == "set_many":
        level = rng.randint(0, MAX_ELEVATION)
        return [(x, y, level) for x, y in tiles]
    delta = rng.choice((-1, 1))
    return [(x, y, max(0, min(MAX_ELEVATION, int(grid[y, x]) + delta))) for x, y in tiles]


def _check_call_shape(shape: str, runs: int) -> None:
    scenario = _load()
    mm = scenario.map_manager
    left_a_step = 0
    for seed in range(runs):
        rng = random.Random(f"{shape} {seed}")
        grid = _steep_grid(rng, mm.map_size)
        targets = _call_targets(shape, rng, grid)
        _write_grid(mm, grid)
        _propagate(mm, targets, set())
        propagated, propagated_legal = _flat(mm), check_elevation(scenario).is_clean

        _write_grid(mm, grid)
        start = _flat(mm)
        touched = set_tiles_elevation(mm, targets)

        assert check_elevation(scenario).is_clean, seed
        assert all(mm.get_tile(x, y).elevation == e for x, y, e in targets), seed
        assert _changed(start, mm) <= touched, seed
        if propagated_legal:
            assert _flat(mm) == propagated, seed
        left_a_step += not propagated_legal
    assert left_a_step, "no run left a step, so the repair went unexercised"


_CALL_SHAPES = ("set_one", "set_many", "elevate", "paste")


@pytest.mark.parametrize("shape", _CALL_SHAPES)
def test_no_call_shape_leaves_a_new_step_and_a_legal_result_stands(shape: str) -> None:
    _check_call_shape(shape, 50)


@pytest.mark.slow
@pytest.mark.parametrize("shape", _CALL_SHAPES)
def test_no_call_shape_leaves_a_new_step_over_many_maps(shape: str) -> None:
    _check_call_shape(shape, 300)


def _brute_clamp(base: np.ndarray, fixed: dict[tuple[int, int], int], radius: int) -> np.ndarray:
    out = base.copy()
    for (y, x), value in np.ndenumerate(base):
        if (x, y) in fixed:
            out[y, x] = fixed[(x, y)]
            continue
        near = [(v, max(abs(x - kx), abs(y - ky))) for (kx, ky), v in fixed.items()]
        near = [(v, d) for v, d in near if d <= radius]
        floor = max((v - d for v, d in near), default=value)
        ceiling = min((v + d for v, d in near), default=value)
        out[y, x] = min(max(value, floor), ceiling)
    return out


@pytest.mark.parametrize("radius", [40, 3], ids=["whole_window", "short"])
@pytest.mark.parametrize("seed", range(20))
def test_cone_clamp_matches_a_brute_force_oracle(seed: int, radius: int) -> None:
    rng = random.Random(seed)
    height, width = rng.randint(1, 14), rng.randint(1, 14)
    base = np.array([[rng.randint(0, 7) for _ in range(width)] for _ in range(height)], dtype=np.int64)
    base[rng.randrange(height), rng.randrange(width)] = 20  # out of range, as some files hold
    fixed = {(rng.randrange(width), rng.randrange(height)): rng.randint(0, 15) for _ in range(seed % 7)}
    mask = np.zeros(base.shape, dtype=bool)
    value = np.zeros(base.shape, dtype=np.int64)
    for (x, y), v in fixed.items():
        mask[y, x], value[y, x] = True, v
    assert np.array_equal(cone_clamp(base, mask, value, radius), _brute_clamp(base, fixed, radius))


def test_conflicting_targets_clamp_to_the_upper_cone_and_keep_their_values() -> None:
    # A block holding a 3-level step: at (33, 32) the floor is 3 - 1 and the ceiling 0 + 1.
    targets = [(34, 32, 0), (34, 31, 3)]
    unrepaired = _load()
    _propagate(unrepaired.map_manager, targets, set(), gap_filled=(gaps := []))
    assert gaps and unrepaired.map_manager.get_tile(33, 32).elevation == 2, "the repair has nothing to do here"

    scenario = _load()
    mm = scenario.map_manager
    start = _flat(mm)
    touched = set_tiles_elevation(mm, targets)

    assert all(mm.get_tile(x, y).elevation == e for x, y, e in targets)
    assert mm.get_tile(33, 32).elevation == 1
    assert _changed(start, mm) <= touched


def test_a_cliffy_base_keeps_the_targets_and_reports_every_change() -> None:
    repaired = 0
    for seed in range(12):
        rng = random.Random(2000 + seed)
        scenario, unrepaired = _load(), _load()
        mm = scenario.map_manager
        cx, cy = rng.randint(20, 100), rng.randint(20, 100)
        _seed_elevations((mm, unrepaired.map_manager), rng, cx, cy, 9, False)
        start = _flat(mm)
        targets = [(cx + dx, cy + dy, rng.randint(0, 7)) for dx in range(3) for dy in range(3)]

        touched = set_tiles_elevation(mm, targets)
        _propagate(unrepaired.map_manager, targets, set())

        assert all(mm.get_tile(x, y).elevation == e for x, y, e in targets), seed
        assert _changed(start, mm) <= touched, seed
        repaired += _flat(mm) != _flat(unrepaired.map_manager)
    assert repaired, "the repair never ran on a cliffy base, so this test is vacuous"


def test_a_map_above_the_legal_range_gets_no_new_step_past_a_fifteen_tile_window() -> None:
    # With a spread over 15, a clamp stopped 15 tiles out would drop levels at its own edge.
    # check_elevation() caps its findings, and out-of-range tiles fill the cap, so read the grid.
    def has_step(mm) -> bool:
        e = np.array(_flat(mm)).reshape(mm.map_size, mm.map_size)
        pairs = ((e[:, 1:], e[:, :-1]), (e[1:], e[:-1]), (e[1:, 1:], e[:-1, :-1]), (e[1:, :-1], e[:-1, 1:]))
        return any(bool(np.any(np.abs(a - b) > 1)) for a, b in pairs)

    top = 2 * MAX_ELEVATION
    scenario = _load()
    mm = scenario.map_manager
    left_a_step = 0
    for seed in range(60):
        rng = random.Random(f"out of range {seed}")
        grid = _steep_grid(rng, mm.map_size, top)
        targets = _call_targets("paste", rng, grid, top)
        _write_grid(mm, grid)
        _propagate(mm, targets, set())
        left_a_step += has_step(mm)
        _write_grid(mm, grid)
        set_tiles_elevation(mm, targets)
        assert not has_step(mm), seed
    assert left_a_step, "no run left a step, so the repair went unexercised"


def test_before_write_sees_every_repair_index_before_its_write() -> None:
    targets = _hill_targets()
    rx, ry = conftest.ILLEGAL_STEP_HILL_REPAIRED_TILE
    unrepaired = _load()
    _propagate(unrepaired.map_manager, targets, propagated := set())
    scenario = _load()
    mm = scenario.map_manager
    start = _flat(mm)
    first_seen: dict[int, int] = {}

    def record(i: int) -> None:
        first_seen.setdefault(i, mm.terrain[i].elevation)

    set_tiles_elevation(mm, targets, before_write=record)
    repair_index = ry * mm.map_size + rx
    assert repair_index not in propagated, "the repair's tile is not its own, so this test is vacuous"
    assert repair_index in first_seen
    assert _changed(start, mm) - {y * mm.map_size + x for x, y, _ in targets} <= set(first_seen)
    assert all(start[i] == value for i, value in first_seen.items())
