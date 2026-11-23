"""native_composite._csr's vectorised extents and _i32's bound check
(TASK-031.85 D2 extras) against the per-op and array_equal implementations
they replaced, kept here. Pure numpy: no native kernel needed."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from descape import iso_geometry, native_composite, render_paint, settings

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


# --- Sliced assembly (TASK-031.87): _CsrAssembly in ranges against one-shot _old_csr.


def _old_csr(ops, src):
    start = np.zeros(len(ops) + 1, dtype=np.int64)
    np.cumsum([op[0].size for op in ops], out=start[1:])
    cols = {"dy": _old_i32([op[0] for op in ops]), "dx": _old_i32([op[1] for op in ops]), "start": start}
    cols["ext"] = _old_ext(ops)
    if src:
        cols["sy"] = _old_i32([op[2] for op in ops])
        cols["sx"] = _old_i32([op[3] for op in ops])
    return cols


def _same_cols(got: dict, want: dict, what) -> None:
    assert got.keys() == want.keys(), what
    for k, w in want.items():
        g = got[k]
        assert g.dtype == w.dtype and g.flags.c_contiguous and g.shape == w.shape, (what, k)
        assert np.array_equal(g, w), (what, k)


def _sliced(ops, src, budget, **kw):
    asm = native_composite._CsrAssembly(ops, src, **kw)
    steps = 1
    while asm.step(budget):
        steps += 1
    return asm.finish(), steps


@pytest.fixture
def csr_compared(monkeypatch):
    """Every _csr call, checked against _old_csr."""
    real = native_composite._csr
    calls = []

    def spy(ops, src):
        got = real(ops, src)
        _same_cols(got, _old_csr(ops, src), len(ops))
        calls.append(len(ops))
        return got

    monkeypatch.setattr(native_composite, "_csr", spy)
    return calls


def test_zero_ops_layout():
    for src in (False, True):
        cols = native_composite._csr([], src=src)
        _same_cols(cols, _old_csr([], src), src)
        assert cols["start"].tolist() == [0]


def test_sliced_csr_matches_one_shot_on_random_op_lists():
    rng = np.random.default_rng(87)
    for trial in range(200):
        lens = rng.choice([0, 0, 1, 2, 17, 300], size=int(rng.integers(1, 30))).tolist()
        src = bool(trial % 2)
        ops = [tuple(rng.integers(-5000, 5000, n).astype(np.int64) for _ in range(4 if src else 2)) for n in lens]
        want = _old_csr(ops, src)
        for budget in (None, 1, int(rng.integers(1, 400)), 10**9):
            (got, _samples), steps = _sliced(ops, src, budget)
            _same_cols(got, want, (lens, budget))
            if budget == 1:
                # Empty ops may ride along in a range; an entry-holding op over the budget is alone.
                assert sum(1 for n in lens if n > 1) <= steps <= len(ops)


def _tables(tile_px, elev_step, budget):
    nc = native_composite
    if budget == "eager":
        return nc.SteppedTables(tile_px, elev_step)
    return render_paint._drain(nc.assemble_stepped(tile_px, elev_step, budget))


def _old_stepped(tile_px, elev_step):
    """The pre-TASK-031.87 SteppedTables.__init__'s cols, factor_keys, op_len and fstart."""
    nc = native_composite
    pairs = nc._fixed_ops(tile_px)
    for r in range(1, nc.MAX_RISE + 1):
        pairs += nc._rise_ops(tile_px, r * elev_step)
    cols = _old_csr([op if len(op) == 4 else (*op, np.zeros_like(op[0]), np.zeros_like(op[0])) for op, _key in pairs], True)
    keys = [key for _op, key in pairs]
    op_len = np.diff(cols["start"])
    fstart = np.full(len(pairs), -1, dtype=np.int64)
    total = 0
    for i, key in enumerate(keys):
        if key is not None:
            fstart[i] = total
            total += int(op_len[i])
    return cols, keys, op_len, fstart


def _same_array(a, b):
    assert (a is None and b is None) or (a.dtype == b.dtype and a.flags.c_contiguous and np.array_equal(a, b))


@pytest.mark.parametrize("tile_px", LADDER)
def test_stepped_tables_identical_eager_and_sliced(tile_px):
    _hw, half_h = iso_geometry.half_dims(tile_px)
    steps = {max(1, round(half_h * pct / 100)) for pct in settings.ELEV_STEP_PCT_STOPS}
    for elev_step in (min(steps), max(steps)):
        want, keys, op_len, fstart = _old_stepped(tile_px, elev_step)
        ref = _tables(tile_px, elev_step, "eager")
        # Factors and LUTs aren't assembled: the eager instance's stand in for the old code's.
        kernel = (want["dy"], want["dx"], want["sy"], want["sx"], None, want["start"], want["ext"], fstart,
                  *ref.kernel_tables()[8:])
        for budget in ("eager", native_composite.PRIME_ASSEMBLY_ELEMS, 1):
            got = ref if budget == "eager" else _tables(tile_px, elev_step, budget)
            _same_cols(got.cols, want, (tile_px, elev_step, budget))
            assert got.factor_keys == keys
            _same_array(got.op_len, op_len)
            _same_array(got.fstart, fstart)
            for a, b in zip(got.blend_samples, ref.blend_samples, strict=True):
                _same_array(a, b)
            for a, b in zip(got.kernel_tables(), kernel, strict=True):
                _same_array(a, b)


@pytest.mark.parametrize("tile_px", [32, 128, 192])
def test_other_csr_callers_match_one_shot(csr_compared, tile_px):
    from descape.grid_overlay import grid_bake

    rng = np.random.default_rng(tile_px)
    registry = native_composite.SlopedShapes(tile_px)
    s = max(1, tile_px // 8)
    for _batch in range(3):
        keys = []
        for _ in range(12):
            corners = rng.integers(0, 4, size=4) * s
            corners -= corners.min()
            keys.append((*(int(c) for c in corners), bool(rng.integers(0, 2))))
        registry.shape_ids(keys)
        registry.packed()
    _cand, stamps = native_composite._grid_stamps(tile_px, grid_bake(True, 60, 2), np.arange(81))
    assert stamps is not None
    # The diamond pack, three (quads, edges) repacks, and the grid stamps.
    assert len(csr_compared) == 1 + 3 * 2 + 1


@pytest.mark.parametrize("budget", ["eager", 1])
def test_overflow_still_raised(monkeypatch, budget):
    real = native_composite._rise_ops

    def wide(tile_px, rise):
        ops = real(tile_px, rise)
        (dy, *rest), key = ops[0]
        return [((dy + np.int64(2**31), *rest), key), *ops[1:]]

    monkeypatch.setattr(native_composite, "_rise_ops", wide)
    with pytest.raises(OverflowError, match="does not fit in int32"):
        _tables(32, 4, budget)
