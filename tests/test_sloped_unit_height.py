"""Verifies iso_geometry.unit_rise_px -- Track C5's Step 1, the surface
model a unit in Sloped will be placed on and picked against (Steps 2-3,
not yet wired in).

Checks:
  1. The load-bearing agreement check: at every painted pixel's own (fx,
     fy), unit_rise_px must agree with the rise sloped_quad_indices
     actually produced, to within 1px (measured 0.88px, an
     integer-rounding residual) -- across all 81 corner configurations
     (3 deltas x 4 corners) at every shipped tile_px.
  2. Exactness on a flat tile: equal corners must reduce to exactly that
     corner value at every (fx, fy), integer, no rounding -- the same
     byte-identity bar the rest of Sloped is held to.
  3. Tip agreement: at each of the diamond's four tips (fx, fy) in
     {0, 1}^2, unit_rise_px must equal that tip's own corner value
     exactly, not merely to within 1px.
  4. _corner_weights is gone. Its docstring called deleting it "a real
     decision", so pin that the decision stuck rather than a stale
     import quietly reappearing.

The two mutation checks the plan calls for (swap the triangle diagonal;
substitute a bilinear blend) were run by hand against check 1's oracle
during development, per this repo's convention of proving a check's
sensitivity once rather than shipping the wrong implementation alongside
the right one: both make the diff blow past 1px (16px and 8px respectively
at tile_px=64) on a non-flat fixture, which is what makes check 1 more
than a vacuous re-statement of the code it's checking.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from descape import iso_geometry as ig

# 3 deltas (0, 1, 2 elevation levels) x 4 corners = 81 configurations, the
# same corpus check 1 of test_sloped_geometry.py's docstring describes
# measuring this model against.
_DELTAS = (0, 1, 2)


def _measured_rise(tile_px: int, corners: tuple[int, int, int, int]):
    """(fx, fy, rise) per painted pixel of sloped_quad_indices(tile_px,
    *corners) -- fx/fy from tile_uv_fractions gathered through uv_idx (the
    same alignment tests/test_sloped_geometry.py's uv_idx checks pin), rise
    recovered as diamond_indices' own dst_y at that same pixel minus the
    sloped dst_y actually painted, plus d_min -- the identity that holds
    because both share the same base_y placement convention, just offset
    by the -d_min normalization sloped_quad_indices' own docstring
    requires callers to fold back in."""
    dst_y, _dst_x, _src_y, _src_x, uv_idx = ig.sloped_quad_indices(tile_px, *corners)
    d_dst_y, _d_dst_x, _d_src_y, _d_src_x = ig.diamond_indices(tile_px)
    d_min = min(corners)
    rise = d_dst_y[uv_idx] - dst_y + d_min
    fp, fq = ig.tile_uv_fractions(tile_px)
    fx = 1 - fq[uv_idx]
    fy = fp[uv_idx]
    return fx, fy, rise


@pytest.mark.parametrize("tile_px", [ig.MIP_MIN_TILE_PIXELS, 16, 32, 64, ig.MIP_MAX_TILE_PIXELS])
def test_unit_rise_px_matches_painted_surface_across_all_corner_configs(tile_px):
    elev_step = max(1, tile_px // 8)  # 8px per level at tile_px=64, per ELEV_STEP_DIVISOR
    worst = 0
    worst_combo = None
    for combo in itertools.product([d * elev_step for d in _DELTAS], repeat=4):
        fx, fy, rise = _measured_rise(tile_px, combo)
        corner_rise = np.array([[combo[0], combo[1]], [combo[2], combo[3]]], dtype=np.int64)
        preds = np.array(
            [ig.unit_rise_px(corner_rise, 0, 0, float(a), float(b)) for a, b in zip(fx.tolist(), fy.tolist())]
        )
        diff = int(np.abs(preds - rise).max())
        if diff > worst:
            worst, worst_combo = diff, combo
    assert worst <= 1, f"unit_rise_px disagreed with the painted surface by {worst}px at corners={worst_combo}"


def test_unit_rise_px_is_exact_on_a_flat_tile():
    # Equal corners: the two-triangle formula must collapse to exactly that
    # value everywhere, no rounding residue -- the same flat-map exactness
    # corner_rise_px's own "average" rule guarantees one level up.
    for value in (0, 5, 12, 30):
        corner_rise = np.full((2, 2), value, dtype=np.int64)
        for fx, fy in ((0.0, 0.0), (0.5, 0.5), (0.25, 0.75), (1.0, 0.0), (0.9, 0.9)):
            assert ig.unit_rise_px(corner_rise, 0, 0, fx, fy) == value


def test_unit_rise_px_is_exact_at_the_four_tips():
    corner_rise = np.array([[3, 11], [7, 21]], dtype=np.int64)  # nw, ne, sw, se
    nw, ne, sw, se = 3, 11, 7, 21
    assert ig.unit_rise_px(corner_rise, 0, 0, 0.0, 0.0) == nw
    assert ig.unit_rise_px(corner_rise, 0, 0, 1.0, 0.0) == ne
    assert ig.unit_rise_px(corner_rise, 0, 0, 0.0, 1.0) == sw
    assert ig.unit_rise_px(corner_rise, 0, 0, 1.0, 1.0) == se


def test_corner_weights_is_gone():
    assert not hasattr(ig, "_corner_weights"), (
        "_corner_weights was deleted as part of C5 Step 1 (unit_rise_px is tile_uv_fractions' "
        "replacement motivating example) -- it should not have come back"
    )
