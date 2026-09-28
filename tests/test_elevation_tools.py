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

import pytest

from descape.elevation_tools import set_tile_elevation, set_tiles_elevation
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

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
    touched = set_tiles_elevation(omm, targets)
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

    touched = set_tile_elevation(omm, cx, cy, level)
    tile = tmm.get_tile(cx, cy)
    tile.elevation = level
    tmm._elevation_tile_recursion(tile, {tile.xy})

    after = [t.elevation for t in omm.terrain]
    assert after == [t.elevation for t in tmm.terrain]
    assert {i for i, (b, a) in enumerate(zip(before, after, strict=True)) if b != a} <= touched
