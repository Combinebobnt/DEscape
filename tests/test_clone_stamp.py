"""descape/clone_stamp.py: the source and offset, the snapshot copy and the
elevation seam's cone clamp. Qt-free. The viewer's stroke is
tests/test_clone_stamp_viewer.py."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from descape import brush
from descape.clone_stamp import CloneSource, CloneStroke, apply_clone, clone_targets
from descape.edit_history import tile_state
from descape.mirror_tools import _elevation_violations


def _map(width: int, height: int, elevations=None, terrain_ids=None):
    tiles = [
        SimpleNamespace(
            terrain_id=0 if terrain_ids is None else int(terrain_ids[i]),
            layer=-1,
            elevation=0 if elevations is None else int(elevations[i]),
        )
        for i in range(width * height)
    ]
    return SimpleNamespace(map_width=width, map_height=height, terrain=tiles)


def _elevations(mm) -> list[int]:
    return [t.elevation for t in mm.terrain]


def _stroke(mm, dabs, offset, *, terrain: bool = True, elevation: bool = True, carry_layer: bool = True):
    """One stroke as the viewer runs it: a full snapshot at the press, a tile
    painted at most once, one apply_clone() per dab. Returns the stroke, the
    cloned tiles and every index written."""
    snapshot = [tile_state(t) for t in mm.terrain]
    width, height = mm.map_width, mm.map_height
    stroke = CloneStroke.from_tiles(mm.terrain, width, height) if elevation else None
    painted: set = set()
    cloned: list = []
    written: set = set()
    for dab in dabs:
        footprint = [t for t in dab if t not in painted]
        targets = clone_targets(footprint, offset, snapshot.__getitem__, width, height)
        got, wrote = apply_clone(mm, stroke, targets, terrain=terrain, elevation=elevation, carry_layer=carry_layer)
        painted.update(footprint)
        cloned += got
        written |= wrote
    return stroke, cloned, written


def _square(cx: int, cy: int, size: int, width: int, height: int):
    return brush.brush_tiles(cx, cy, size, brush.BRUSH_SHAPE_SQUARE, width, height)


# -- the source ---------------------------------------------------------------


def test_the_offset_is_fixed_once_from_the_first_press_and_reset_by_a_new_anchor() -> None:
    source = CloneSource()
    source.fix_offset((5, 5))
    assert source.offset is None, "an offset with no anchor"
    assert source.marker_origin((3, 3)) is None
    source.set_anchor((10, 4))
    assert source.marker_origin((30, 30)) == (10, 4), "before the press the marker is the anchor"
    source.fix_offset((13, 9))
    assert source.offset == (3, 5)
    source.fix_offset((40, 40))
    assert source.offset == (3, 5), "a later press moved the offset"
    assert source.source_of((20, 20)) == (17, 15)
    assert source.marker_origin((20, 20)) == (17, 15)
    assert source.marker_origin(None) is None
    source.set_anchor((1, 1))
    assert source.offset is None
    source.clear()
    assert (source.anchor, source.offset) == (None, None)


@pytest.mark.parametrize(
    "offset, dropped",
    [
        ((3, 0), lambda x, y: x < 3),  # west: sources at x - 3 < 0
        ((-3, 0), lambda x, y: x >= 13),  # east
        ((0, 3), lambda x, y: y < 3),  # north
        ((0, -3), lambda x, y: y >= 13),  # south
    ],
    ids=["west", "east", "north", "south"],
)
def test_a_dab_shrinks_where_its_source_leaves_the_map(offset, dropped) -> None:
    mm = _map(16, 16, elevations=[(i % 7) // 2 for i in range(256)], terrain_ids=range(256))
    before = [tile_state(t) for t in mm.terrain]
    everything = [(x, y) for y in range(16) for x in range(16)]
    stroke, cloned, _written = _stroke(mm, [everything], offset)
    assert set(cloned) == {t for t in everything if not dropped(*t)}
    for x, y in everything:
        if dropped(x, y):
            assert not stroke.cloned[y, x], (x, y)
            assert mm.terrain[y * 16 + x].terrain_id == before[y * 16 + x][0], "a skipped tile was written"
        else:
            sx, sy = x - offset[0], y - offset[1]
            assert mm.terrain[y * 16 + x].terrain_id == before[sy * 16 + sx][0]


def test_an_overlapping_stroke_copies_the_press_time_map_not_its_own_output() -> None:
    """Offset (1, 0) dragged east: each dab's source is the tile the last dab
    just wrote. A live read would smear tile 4's terrain along the row."""
    mm = _map(24, 4, terrain_ids=range(96), elevations=[(i % 24) % 4 for i in range(96)])
    before = [tile_state(t) for t in mm.terrain]
    _stroke(mm, [[(x, 1)] for x in range(5, 16)], (1, 0))
    row = [mm.terrain[24 + x].terrain_id for x in range(5, 16)]
    assert row == [before[24 + x - 1][0] for x in range(5, 16)]
    assert len(set(row)) == len(row), "smeared"


def test_terrain_and_layer_copy_as_a_pair_and_no_layer_without_carry_layer() -> None:
    mm = _map(10, 10)
    mm.terrain[11].terrain_id, mm.terrain[11].layer = 9, 4
    mm.terrain[12].terrain_id, mm.terrain[12].layer = 9, 9  # verbatim, even a layer equal to its base
    _stroke(mm, [[(5, 5), (6, 5)]], (4, 4))
    assert [(t.terrain_id, t.layer) for t in mm.terrain[55:57]] == [(9, 4), (9, 9)]
    mm2 = _map(10, 10)
    mm2.terrain[11].terrain_id, mm2.terrain[11].layer = 9, 4
    _stroke(mm2, [[(5, 5)]], (4, 4), carry_layer=False)
    assert (mm2.terrain[55].terrain_id, mm2.terrain[55].layer) == (9, -1)


def test_terrain_unticked_writes_elevation_only_and_elevation_unticked_writes_terrain_only() -> None:
    mm = _map(12, 12, elevations=[2] * 144, terrain_ids=[3] * 144)
    mm.terrain[0].terrain_id, mm.terrain[0].elevation = 7, 3
    _stroke(mm, [[(6, 6)]], (6, 6), terrain=False)
    assert (mm.terrain[78].terrain_id, mm.terrain[78].elevation) == (3, 3)
    mm2 = _map(12, 12, elevations=[2] * 144, terrain_ids=[3] * 144)
    mm2.terrain[0].terrain_id, mm2.terrain[0].elevation = 7, 3
    _stroke(mm2, [[(6, 6)]], (6, 6), elevation=False)
    assert (mm2.terrain[78].terrain_id, mm2.terrain[78].elevation) == (7, 2)
    assert _elevations(mm2) == [3] + [2] * 143


# -- the seam -----------------------------------------------------------------


def _oracle(e0: np.ndarray, cloned: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Decision 2's clamp, brute force over every cloned tile: no dilation, no cone_clamp()."""
    height, width = e0.shape
    ky, kx = np.nonzero(cloned)
    kv = value[ky, kx]
    ys, xs = np.mgrid[0:height, 0:width]
    d = np.maximum(np.abs(ys.reshape(-1, 1) - ky), np.abs(xs.reshape(-1, 1) - kx))
    low = (kv - d).max(axis=1).reshape(height, width)
    high = (kv + d).min(axis=1).reshape(height, width)
    return np.where(cloned, value, np.minimum(np.maximum(e0, low), high))


def _steep_legal(rng: random.Random, width: int, height: int) -> np.ndarray:
    """A legal map as a max of Chebyshev cones (hills) cut by a min of cones (pits), clipped to 0..15."""
    ys, xs = np.mgrid[0:height, 0:width]
    hills = np.full((height, width), -99)
    for _ in range(rng.randint(3, 8)):
        px, py, top = rng.randrange(width), rng.randrange(height), rng.randint(4, 24)
        hills = np.maximum(hills, top - np.maximum(np.abs(xs - px), np.abs(ys - py)))
    pits = np.full((height, width), 99)
    for _ in range(rng.randint(0, 4)):
        px, py, floor = rng.randrange(width), rng.randrange(height), rng.randint(-6, 6)
        pits = np.minimum(pits, floor + np.maximum(np.abs(xs - px), np.abs(ys - py)))
    return np.clip(np.minimum(hills, pits), 0, 15)


def _random_path(rng: random.Random, width: int, height: int, offset):
    """40 square dabs of size 1 to 5 on a random walk, starting where the source is on the map."""
    dabs = []
    x = rng.randrange(max(0, offset[0]), min(width, width + offset[0]))
    y = rng.randrange(max(0, offset[1]), min(height, height + offset[1]))
    for _ in range(40):
        size = rng.randint(1, 5)
        dabs.append(_square(x, y, size, width, height))
        x = min(width - 1, max(0, x + rng.randint(-2, 2)))
        y = min(height - 1, max(0, y + rng.randint(-2, 2)))
    return dabs


def _check_run(seed: int) -> None:
    rng = random.Random(seed)
    width = height = 48
    e0 = _steep_legal(rng, width, height)
    assert not _elevation_violations(e0.ravel().tolist(), width, height), "the fixture map is illegal"
    # A third of the runs overlap their own source.
    span = 3 if seed % 3 == 0 else 20
    offset = (rng.randint(-span, span), rng.randint(-span, span))
    mm = _map(width, height, elevations=e0.ravel(), terrain_ids=range(width * height))
    stroke, cloned, written = _stroke(mm, _random_path(rng, width, height, offset), offset)
    assert cloned, f"seed {seed}: nothing cloned -- vacuous"
    result = np.array(_elevations(mm)).reshape(height, width)
    mismatches = [
        (x, y) for x, y in cloned
        if result[y, x] != e0[y - offset[1], x - offset[0]]
        or mm.terrain[y * width + x].terrain_id != (y - offset[1]) * width + x - offset[0]
    ]
    assert not mismatches, f"seed {seed}: cloned tiles lost their source value: {mismatches[:5]}"
    assert not _elevation_violations(result.ravel().tolist(), width, height), f"seed {seed}: an illegal pair"
    assert np.array_equal(result, _oracle(e0, stroke.cloned, stroke.cloned_value)), f"seed {seed}: differs from the oracle"
    assert set(np.flatnonzero(result.ravel() != e0.ravel())) <= written, f"seed {seed}: a write went unreported"


@pytest.mark.parametrize("seed", range(50))
def test_per_dab_clamping_is_legal_and_equals_the_whole_stroke_oracle(seed: int) -> None:
    _check_run(seed)


@pytest.mark.slow
def test_the_seam_property_over_400_maps() -> None:
    for seed in range(400):
        _check_run(seed)


def test_one_dab_onto_uneven_ground_moves_its_ring_and_matches_the_oracle() -> None:
    """A 3x3 plateau at 10 cloned onto ground at 0: the ground ramps 9, 8, ... out from it."""
    width = height = 30
    ys, xs = np.mgrid[0:height, 0:width]
    e0 = (11 - np.maximum(np.abs(xs - 5), np.abs(ys - 5))).clip(0, 10)
    assert e0[4:7, 4:7].min() == 10 and e0[20, 20] == 0
    mm = _map(width, height, elevations=e0.ravel())
    stroke, _cloned, written = _stroke(mm, [_square(20, 20, 3, width, height)], (15, 15))
    result = np.array(_elevations(mm)).reshape(height, width)
    assert result[20, 20] == 10 and result[20, 22] == 9 and result[20, 28] == 3, result[20]
    assert np.array_equal(result, _oracle(e0, stroke.cloned, stroke.cloned_value))
    assert not _elevation_violations(result.ravel().tolist(), width, height)
    assert len(written) > 9, "the ring was not reported"


def test_an_out_of_range_value_still_clamps_exactly() -> None:
    """A stored 20 (past MAX_ELEVATION) cloned onto flat 0: its cone reaches 19
    tiles, past the legal range's 15, so a bare-15 radius leaves the tile 17
    away at 0 where the oracle wants 3."""
    width = height = 48
    e0 = np.zeros((height, width), dtype=np.int64)
    e0[2, 2] = 20
    mm = _map(width, height, elevations=e0.ravel())
    stroke, _cloned, _written = _stroke(mm, [[(10, 10)]], (8, 8))
    result = np.array(_elevations(mm)).reshape(height, width)
    assert stroke.radius == 20
    assert result[10, 27] == 3, result[10, 20:30]
    assert np.array_equal(result, _oracle(e0, stroke.cloned, stroke.cloned_value))


def test_an_illegal_source_pair_stays_near_its_copy_and_violations_reports_it() -> None:
    checked = 0
    for seed in range(20):
        rng = random.Random(1000 + seed)
        width = height = 40
        e0 = _steep_legal(rng, width, height)
        # A 3-level cliff inside the source area.
        cx, cy = rng.randrange(8, 30), rng.randrange(8, 30)
        e0[cy, cx] = min(15, e0[cy, cx] + 3) if e0[cy, cx] <= 12 else e0[cy, cx] - 3
        source_bad = _elevation_violations(e0.ravel().tolist(), width, height)
        assert source_bad
        offset = (rng.randint(-6, 6) or 7, rng.randint(-6, 6))
        dabs = [_square(cx + offset[0] + dx, cy + offset[1], 5, width, height) for dx in range(-3, 4)]
        mm = _map(width, height, elevations=e0.ravel())
        stroke, _cloned, written = _stroke(mm, dabs, offset)
        after = _elevation_violations(_elevations(mm), width, height)
        # Where the source's bad pairs landed: each endpoint's destination, where it was cloned.
        copied = []
        for i in {i for pair in source_bad for i in pair}:
            x, y = i % width + offset[0], i // width + offset[1]
            if 0 <= x < width and 0 <= y < height and stroke.cloned[y, x]:
                copied.append((x, y))
        reported = set(stroke.violations(written))
        new = [(a, b) for a, b in after if a in written or b in written]
        assert all(pair in reported for pair in new), (seed, set(new) - reported)
        for a, _b in new:
            near = [max(abs(a % width - x), abs(a // width - y)) <= stroke.radius + 1 for x, y in copied]
            assert any(near), f"seed {seed}: a new illegal pair far from any copied violation"
        checked += bool(new)
    assert checked, "no run copied the illegal pair -- vacuous"
