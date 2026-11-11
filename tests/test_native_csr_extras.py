"""native_composite._csr's vectorised extents and _i32's bound check
(TASK-031.85 D2 extras) against the per-op and array_equal implementations
they replaced, kept here. Pure numpy: no native kernel needed."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from descape import iso_geometry, native_composite, settings

LADDER = sorted({*iso_geometry.mip_tile_px_candidates(64, native_tier=True).values(), *iso_geometry.NATIVE_TIERS})


def _old_extent(dst_y, dst_x):
    if dst_y.size == 0:
        return native_composite._EMPTY_EXT
    return int(dst_y.min()), int(dst_y.max()), int(dst_x.min()), int(dst_x.max())


def _old_ext(ops) -> np.ndarray:
    return np.array([_old_extent(op[0], op[1]) for op in ops], dtype=np.int32).reshape(-1)


def _old_i32(parts):
    if not parts:
        return np.zeros(0, dtype=np.int32)
    wide = np.concatenate(parts)
    out = wide.astype(np.int32)
    if not np.array_equal(out, wide):
        raise OverflowError("an index table value does not fit in int32")
    return out


def _ops(rng, lens):
    return [tuple(rng.integers(-5000, 5000, n).astype(np.int64) for _ in range(2)) for n in lens]


@pytest.mark.parametrize(
    "lens",
    [[], [0], [0, 0, 0], [5], [3, 0, 4], [0, 7, 0, 0, 2, 0], [1, 1, 1], [0, 0, 9], [9, 0, 0], [200, 0, 1, 0, 50]],
)
def test_extents_match_per_op_extent(lens):
    ops = _ops(np.random.default_rng(len(lens) * 31 + sum(lens)), lens)
    got = native_composite._csr(ops, src=False)["ext"]
    want = _old_ext(ops)
    assert got.dtype == np.int32 and got.shape == want.shape
    assert np.array_equal(got, want), (lens, got, want)


def test_extents_match_on_random_op_lists():
    rng = np.random.default_rng(85)
    for _ in range(200):
        lens = rng.choice([0, 0, 1, 2, 17, 300], size=int(rng.integers(1, 40))).tolist()
        ops = _ops(rng, lens)
        assert np.array_equal(native_composite._csr(ops, src=False)["ext"], _old_ext(ops)), lens


@pytest.mark.parametrize("tile_px", LADDER)
def test_real_tables_extents_match(tile_px):
    _hw, half_h = iso_geometry.half_dims(tile_px)
    steps = {max(1, round(half_h * pct / 100)) for pct in settings.ELEV_STEP_PCT_STOPS}
    tables = [native_composite.SteppedTables(tile_px, s).cols for s in (min(steps), max(steps))]
    registry = native_composite.SlopedShapes(tile_px)
    s = max(steps)
    registry.shape_ids([(0, 0, 0, 0, False), (s, 0, 0, 0, True), (s, s, 0, 0, False), (0, s, s, 2 * s, True)])
    quads, edges, _samples = registry.packed()
    tables += [{"dy": t[0], "dx": t[1], "start": t[5], "ext": t[6]} for t in (quads, edges, registry.diamond)]
    for cols in tables:
        start = cols["start"]
        ops = [(cols["dy"][a:b], cols["dx"][a:b]) for a, b in pairwise(start.tolist())]
        assert np.array_equal(cols["ext"], _old_ext(ops)), tile_px


def _same_i32(parts) -> bool:
    """Asserts both implementations agree; returns whether they raised."""
    try:
        want = _old_i32(parts)
    except OverflowError:
        with pytest.raises(OverflowError):
            native_composite._i32(parts)
        return True
    got = native_composite._i32(parts)
    assert got.dtype == np.int32 and np.array_equal(got, want)
    return False


LO, HI = int(np.iinfo(np.int32).min), int(np.iinfo(np.int32).max)
EDGES = (LO - 1, LO, LO + 1, -1, 0, 1, HI - 1, HI, HI + 1)


@pytest.mark.parametrize(
    ("parts", "raises"),
    [
        ([], False),
        ([np.zeros(0, np.int64)], False),
        ([np.zeros(0, np.int64), np.array([LO, HI], np.int64)], False),
        ([np.array([HI], np.int64)], False),
        ([np.array([LO], np.int64)], False),
        ([np.array([0, HI + 1], np.int64)], True),
        ([np.array([LO - 1, 0], np.int64)], True),
        ([np.array([3], np.int64), np.array([HI + 1], np.int64)], True),
        ([np.array([HI], np.uint32)], False),
        ([np.array([HI + 1], np.uint32)], True),
        ([np.array([2**63 - 1], np.int64)], True),
        ([np.array([-(2**63)], np.int64)], True),
        ([np.array([LO, 0, HI], np.int32)], False),
        ([np.array([1.0, 2.0])], False),
        ([np.array([1.0, 2.5])], True),
    ],
)
def test_i32_bounds_match_array_equal_check(parts, raises):
    _same_i32(parts)
    if raises:
        with pytest.raises(OverflowError):
            native_composite._i32(parts)
    else:
        native_composite._i32(parts)


def test_i32_matches_on_random_parts():
    rng = np.random.default_rng(31)
    raised = 0
    for _ in range(300):
        parts = [rng.choice(np.array(EDGES, np.int64), int(rng.integers(0, 6))) if rng.random() < 0.2
                 else rng.integers(-(2**20), 2**20, int(rng.integers(0, 50)), dtype=np.int64)
                 for _ in range(int(rng.integers(0, 5)))]
        raised += _same_i32(parts)
    assert 0 < raised < 300, f"random parts raised {raised} of 300 times: not both outcomes"
