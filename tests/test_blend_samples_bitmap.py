"""native_composite._blend_samples' presence bitmap (TASK-031.85 D2) against
the np.unique implementation it replaced, on the exact inputs SteppedTables
and SlopedShapes pass it. Pure numpy: no native kernel needed."""

from __future__ import annotations

import numpy as np
import pytest

from descape import iso_geometry, native_composite, render_paint, settings

# The default base-64/native-96 ladder plus the 192 native tier other bases reach.
LADDER = sorted({*iso_geometry.mip_tile_px_candidates(64, native_tier=True).values(), *iso_geometry.NATIVE_TIERS})


def _old_blend_samples(tile_px, sy, sx):
    flat = np.unique(np.concatenate([np.zeros(0, np.int64), *(y.astype(np.int64) * tile_px + x for y, x in zip(sy, sx, strict=True))]))
    return np.ascontiguousarray(flat // tile_px, dtype=np.int32), np.ascontiguousarray(flat % tile_px, dtype=np.int32)


def _elev_steps(tile_px: int) -> list[int]:
    _half_w, half_h = iso_geometry.half_dims(tile_px)
    return sorted({max(1, round(half_h * pct / 100)) for pct in settings.ELEV_STEP_PCT_STOPS})


@pytest.fixture
def compared(monkeypatch):
    """Every _blend_samples call, checked against the old implementation."""
    real = native_composite._blend_samples
    calls = []

    def spy(tile_px, sy, sx):
        new = real(tile_px, sy, sx)
        old = _old_blend_samples(tile_px, sy, sx)
        for got, want in zip(new, old, strict=True):
            assert got.dtype == np.int32 and got.flags.c_contiguous
            assert np.array_equal(got, want), f"tile_px {tile_px}: bitmap differs from np.unique"
        calls.append(tile_px)
        return new

    monkeypatch.setattr(native_composite, "_blend_samples", spy)
    return calls


def _stepped_reads(cols) -> tuple[list, list]:
    nc, start = native_composite, cols["start"]
    reads = [nc.OP_DIAMOND, *(nc.N_FIXED + (r - 1) * nc.PER_RISE + side for r in range(1, nc.MAX_RISE + 1) for side in (0, 1))]
    return [cols["sy"][start[i]:start[i + 1]] for i in reads], [cols["sx"][start[i]:start[i + 1]] for i in reads]


@pytest.mark.parametrize("tile_px", LADDER)
def test_stepped_tables_bitmap_matches_unique(tile_px):
    """Stepped marks its bitmap per assembly step (TASK-031.87), eager or sliced."""
    nc = native_composite
    for elev_step in _elev_steps(tile_px):
        for budget in (None, nc.PRIME_ASSEMBLY_ELEMS, 1):
            tables = render_paint._drain(nc.assemble_stepped(tile_px, elev_step, budget))
            want = _old_blend_samples(tile_px, *_stepped_reads(tables.cols))
            for got, ref in zip(tables.blend_samples, want, strict=True):
                assert got.dtype == np.int32 and got.flags.c_contiguous
                assert np.array_equal(got, ref), (tile_px, elev_step, budget)


@pytest.mark.parametrize("tile_px", LADDER)
def test_sloped_shapes_bitmap_matches_unique(compared, tile_px):
    rng = np.random.default_rng(tile_px)
    s = max(_elev_steps(tile_px))
    registry = native_composite.SlopedShapes(tile_px)
    for _batch in range(3):
        keys = []
        for _ in range(12):
            corners = rng.integers(0, 4, size=4) * s
            corners -= corners.min()
            keys.append((*(int(c) for c in corners), bool(rng.integers(0, 2))))
        registry.shape_ids(keys)
        registry.packed()
    assert len(registry.corners) > 3, "random shapes all collapsed"
    assert compared == [tile_px] * 3


def test_bitmap_on_duplicates_empty_and_unsorted_input():
    rng = np.random.default_rng(7)
    tile_px = 16
    sy = [rng.integers(0, tile_px, 200).astype(np.int32), np.zeros(0, np.int32), rng.integers(0, tile_px, 50).astype(np.int32)]
    sx = [rng.integers(0, tile_px, 200).astype(np.int32), np.zeros(0, np.int32), rng.integers(0, tile_px, 50).astype(np.int32)]
    for args in ((sy, sx), ([], [])):
        new = native_composite._blend_samples(tile_px, *args)
        old = _old_blend_samples(tile_px, *args)
        for got, want in zip(new, old, strict=True):
            assert got.dtype == np.int32 and got.flags.c_contiguous and np.array_equal(got, want)


@pytest.mark.parametrize(("y", "x"), [(-1, 15), (16, 0), (15, 16)], ids=["negative", "past the end", "last row overflow"])
def test_bitmap_bound_asserts(y, x):
    with pytest.raises(AssertionError):
        native_composite._blend_samples(16, [np.array([0, y], np.int32)], [np.array([0, x], np.int32)])
