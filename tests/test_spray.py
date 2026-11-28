"""descape/spray.py, the Spray Can's scatter kernel: the per-tile threshold
hash, the hardness weights and dab()'s coverage accumulation."""

from __future__ import annotations

import random

import pytest

from descape import brush, spray

_SEEDS = (1, 0xDEADBEEF, 2**63 + 12345)
_N = 240


def _grid(seed: int) -> list[list[float]]:
    return [[spray.tile_threshold(seed, x, y) for x in range(_N)] for y in range(_N)]


def _corr(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / (va * vb) ** 0.5


def test_the_threshold_is_deterministic_and_in_range() -> None:
    for seed in _SEEDS:
        for x, y in ((0, 0), (239, 0), (17, 203), (-1, 5)):
            t = spray.tile_threshold(seed, x, y)
            assert 0.0 <= t < 1.0
            assert spray.tile_threshold(seed, x, y) == t
    assert spray.tile_threshold(7, 3, 4) != spray.tile_threshold(7, 4, 3)


@pytest.mark.parametrize("seed", _SEEDS)
def test_the_threshold_is_uniform_and_uncorrelated_with_its_neighbours(seed: int) -> None:
    grid = _grid(seed)
    flat = [t for row in grid for t in row]
    for cut in (0.1, 0.3, 0.5):
        fraction = sum(t < cut for t in flat) / len(flat)
        assert abs(fraction - cut) < 0.01, (cut, fraction)
    across = _corr([row[x] for row in grid for x in range(_N - 1)], [row[x + 1] for row in grid for x in range(_N - 1)])
    down = _corr([grid[y][x] for y in range(_N - 1) for x in range(_N)], [grid[y + 1][x] for y in range(_N - 1) for x in range(_N)])
    assert abs(across) < 0.02 and abs(down) < 0.02, (across, down)


def test_adjacent_seeds_are_uncorrelated() -> None:
    for seed in (0, 41, 2**64 - 1):
        a = [t for row in _grid(seed) for t in row]
        b = [t for row in _grid((seed + 1) % 2**64) for t in row]
        assert abs(_corr(a, b)) < 0.02, seed


# The 2026-10-28 probe's weights, transcribed: centre row, then top row (None = not in the shape).
_GOLDEN = [
    ("square", 3, 0.0, [0.36, 1.00, 0.36], [0.10, 0.36, 0.10]),
    ("square", 5, 0.4, [0.54, 1.00, 1.00, 1.00, 0.54], [0.10, 0.41, 0.54, 0.41, 0.10]),
    (
        "square", 9, 0.0,
        [0.36, 0.52, 0.68, 0.84, 1.00, 0.84, 0.68, 0.52, 0.36],
        [0.10, 0.20, 0.29, 0.34, 0.36, 0.34, 0.29, 0.20, 0.10],
    ),
    ("circle", 5, 0.0, [0.20, 0.60, 1.00, 0.60, 0.20], [None, 0.10, 0.20, 0.10, None]),
    (
        "circle", 9, 0.4,
        [0.26, 0.59, 0.93, 1.00, 1.00, 1.00, 0.93, 0.59, 0.26],
        [None, None, 0.10, 0.22, 0.26, 0.22, 0.10, None, None],
    ),
]


@pytest.mark.parametrize(("shape", "size", "hardness", "centre", "top"), _GOLDEN)
def test_the_golden_weights(shape: str, size: int, hardness: float, centre: list, top: list) -> None:
    table = spray.weights(size, shape, 1, hardness)
    half = size // 2
    span = range(-half, half + 1)
    for dx, want in zip(span, centre, strict=True):
        assert table[(dx, 0)] == pytest.approx(want, abs=0.005), (dx, 0)
    for dx, want in zip(span, top, strict=True):
        if want is None:
            assert (dx, -half) not in table
        else:
            assert table[(dx, -half)] == pytest.approx(want, abs=0.005), (dx, -half)


def test_every_weight_is_between_the_floor_and_one() -> None:
    for shape in brush.BRUSH_SHAPES:
        for size in range(brush.BRUSH_SIZE_MIN, brush.BRUSH_SIZE_MAX + 1):
            for thickness in range(brush.BRUSH_THICKNESS_MIN, brush.BRUSH_THICKNESS_MAX + 1):
                offsets = set(brush.brush_offsets(size, shape, thickness))
                for hardness in (0.0, 0.4, 1.0):
                    table = spray.weights(size, shape, thickness, hardness)
                    assert set(table) == offsets
                    assert all(spray.WEIGHT_FLOOR - 1e-9 <= w <= 1.0 for w in table.values()), (shape, size, hardness)
                    if hardness == 1.0:
                        assert set(table.values()) == {1.0}
                if size >= 3 and shape in ("square", "circle"):
                    assert min(spray.weights(size, shape, thickness, 0.0).values()) == pytest.approx(spray.WEIGHT_FLOOR)


def _footprint(cx: int, cy: int, size: int, shape: str = "square", w: int = _N, h: int = _N) -> list[tuple[int, int]]:
    return brush.brush_tiles(cx, cy, size, shape, w, h)


def _spray_still(seed: int, flow: float, hardness: float, dabs: int, size: int = 9, shape: str = "square") -> set:
    """`dabs` dabs at one point, with the caller's painted-tile filter in front of each."""
    stroke = spray.SprayStroke(seed, flow, hardness)
    painted: set = set()
    for _ in range(dabs):
        fresh = [t for t in _footprint(100, 100, size, shape) if t not in painted]
        painted.update(spray.dab(stroke, fresh, 100, 100, size, shape, 1))
    return painted


def test_full_flow_and_hardness_paints_the_whole_footprint_in_one_dab() -> None:
    for shape in ("square", "circle", "plus"):
        stroke = spray.SprayStroke(5, 1.0, 1.0)
        tiles = _footprint(50, 50, 7, shape)
        assert spray.dab(stroke, tiles, 50, 50, 7, shape, 1) == tiles


@pytest.mark.parametrize("dabs", [1, 3, 5, 8, 10, 11])
def test_a_hard_core_fills_at_flow_per_dab(dabs: int) -> None:
    total = painted = 0
    for seed in range(50):
        painted += len(_spray_still(seed, 0.1, 1.0, dabs))
        total += 81
    assert abs(painted / total - min(1.0, 0.1 * dabs)) < 0.03


def test_painting_is_monotone_in_dabs() -> None:
    """Each dab adds to the set; the rim (weight 0.1) is full by dab 101."""
    for seed in range(5):
        stroke = spray.SprayStroke(seed, 0.1, 0.4)
        painted: set = set()
        for _ in range(101):
            fresh = [t for t in _footprint(100, 100, 9) if t not in painted]
            got = spray.dab(stroke, fresh, 100, 100, 9, "square", 1)
            assert not painted & set(got)
            painted.update(got)
        assert len(painted) == 81, "every tile fills eventually"
        assert _spray_still(seed, 0.1, 0.4, 101) == painted


def test_a_shuffled_footprint_paints_the_same_set() -> None:
    tiles = _footprint(60, 60, 9, "circle")
    shuffled = tiles[:]
    random.Random(3).shuffle(shuffled)
    a = spray.SprayStroke(99, 0.3, 0.2)
    b = spray.SprayStroke(99, 0.3, 0.2)
    for _ in range(4):
        got_a = set(spray.dab(a, tiles, 60, 60, 9, "circle", 1))
        got_b = set(spray.dab(b, shuffled, 60, 60, 9, "circle", 1))
        assert got_a == got_b
    assert a.coverage == b.coverage


def test_a_footprint_clipped_at_the_map_corner_keeps_its_weights() -> None:
    table = spray.weights(9, "square", 1, 0.0)
    stroke = spray.SprayStroke(11, 0.5, 0.0)
    clipped = _footprint(1, 1, 9)
    assert len(clipped) == 36
    spray.dab(stroke, clipped, 1, 1, 9, "square", 1)
    for x, y in clipped:
        assert stroke.coverage[(x, y)] == pytest.approx(0.5 * table[(x - 1, y - 1)])


def test_the_returned_tiles_keep_input_order_and_coverage_stays() -> None:
    stroke = spray.SprayStroke(4, 0.6, 1.0)
    tiles = _footprint(30, 30, 5)
    got = spray.dab(stroke, tiles, 30, 30, 5, "square", 1)
    assert got == [t for t in tiles if t in set(got)]
    assert all(stroke.coverage[t] == pytest.approx(0.6) for t in tiles)


def test_new_seed_ignores_the_global_random_state() -> None:
    random.seed(1)
    a = spray.new_seed()
    random.seed(1)
    b = spray.new_seed()
    assert a != b
    assert 0 <= a < 2**64
