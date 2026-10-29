"""Sloped's per-tile fold (TASK-008, GH #62): iso_geometry.tile_folds() and
the NW-SE branches of unit_rise_px() and the point inverse.

Checks:
  1. tile_folds() per shape class: flat and planar tiles keep NE-SW; a
     three-high, one-low tile folds through its low corner; an isolated
     one-high corner is chamfered unless three lower tiles sit behind the
     opposite corner; a border tile reads off-map tiles as the nearest edge
     tile (skipping them would flip the fixture below).
  2. The vectorized rule equals a per-tile scalar reading of the same rule
     on random terrain, borders included.
  3. unit_rise_px(nwse=True) is exact at the four tips and its two
     triangles meet on the fold line.
  4. screen_to_map_point() inverts the NW-SE surface: round trips land back
     on the point, both triangles and the fold line itself.

The rule's evidence is the corpus (tests/test_sloped_fold_corpus.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import iso_geometry as ig


def _fold(rows: list[list[int]], x: int, y: int) -> bool:
    return bool(ig.tile_folds(np.array(rows, dtype=np.int64))[y, x])


def test_flat_and_planar_tiles_keep_ne_sw():
    assert not ig.tile_folds(np.full((5, 5), 3, dtype=np.int64)).any()
    ramp = np.tile(np.arange(5, dtype=np.int64), (5, 1))
    assert not ig.tile_folds(ramp).any()
    assert not ig.tile_folds(ramp.T.copy()).any()


@pytest.mark.parametrize(
    ("rows", "nwse"),
    [
        ([[0, 0, 0], [0, 0, 1], [0, 1, 1]], True),  # low NW
        ([[1, 1, 0], [1, 0, 0], [0, 0, 0]], True),  # low SE
        ([[0, 0, 0], [1, 0, 0], [1, 1, 0]], False),  # low NE
        ([[0, 1, 1], [0, 0, 1], [0, 0, 0]], False),  # low SW
    ],
)
def test_three_high_one_low_folds_through_the_low_corner(rows, nwse):
    assert _fold(rows, 1, 1) is nwse


def test_isolated_high_corner_is_chamfered():
    se_high = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    assert _fold(se_high, 1, 1) is False
    ne_high = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert _fold(ne_high, 1, 2) is True


def test_high_corner_with_three_lower_tiles_behind_the_opposite_corner():
    rows = [[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 2, 1], [1, 1, 1, 1]]
    assert _fold(rows, 1, 1) is True
    rows[0][0] = rows[0][1] = rows[1][0] = 1
    assert _fold(rows, 1, 1) is False, "without the lower tiles the high corner is chamfered"


def test_border_tile_clamps_off_map_tiles_to_the_edge():
    # Clamped, the NW corner's touching tiles are (1, 1, 0, 0): second-highest 1.
    # Skipping the off-map pair would make it 0 and fold this tile NE-SW.
    assert _fold([[1, 0, 0], [0, 0, 0], [0, 1, 0]], 0, 1) is True


def _scalar_fold(e: np.ndarray, x: int, y: int) -> bool:
    h, w = e.shape
    offs = {
        "nw": ((-1, -1), (0, -1), (-1, 0), (0, 0)),
        "ne": ((0, -1), (1, -1), (0, 0), (1, 0)),
        "sw": ((-1, 0), (0, 0), (-1, 1), (0, 1)),
        "se": ((0, 0), (1, 0), (0, 1), (1, 1)),
    }
    top, c2 = {}, {}
    for k, ds in offs.items():
        vals = sorted((int(e[min(max(y + dy, 0), h - 1), min(max(x + dx, 0), w - 1)]) for dx, dy in ds), reverse=True)
        top[k], c2[k] = vals[0], vals[1]
    if top["nw"] + top["se"] == top["ne"] + top["sw"]:
        return False
    a, b = abs(c2["ne"] - c2["sw"]), abs(c2["nw"] - c2["se"])
    if a != b:
        return b > a
    return top["nw"] + top["se"] < top["ne"] + top["sw"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_vectorized_rule_matches_a_scalar_reading(seed):
    e = np.random.default_rng(seed).integers(0, 4, size=(9, 11)).astype(np.int64)
    got = ig.tile_folds(e)
    assert got.shape == e.shape and got.dtype == bool
    want = np.array([[_scalar_fold(e, x, y) for x in range(e.shape[1])] for y in range(e.shape[0])])
    assert np.array_equal(got, want)
    assert got.any() and not got.all(), "the fixture should exercise both folds"


def test_nwse_rise_is_exact_at_the_tips_and_continuous_on_the_fold():
    nw, ne, sw, se = 3, 11, 7, 21
    cr = np.array([[nw, ne], [sw, se]], dtype=np.int64)
    for (fx, fy), want in (((0.0, 0.0), nw), ((1.0, 0.0), ne), ((0.0, 1.0), sw), ((1.0, 1.0), se)):
        assert ig.unit_rise_px(cr, 0, 0, fx, fy, True) == want
    # On fx = fy both triangles read nw + t*(se - nw).
    for t in (0.25, 0.5, 0.75):
        assert ig.unit_rise_px(cr, 0, 0, t, t, True) == ig._round_div(4 * nw + int(4 * t) * (se - nw), 4)
    # The folds differ off the diagonals: (0.5, 0.5) is the NE-SW centre (ne+sw)/2.
    assert ig.unit_rise_px(cr, 0, 0, 0.5, 0.5, False) == 9
    assert ig.unit_rise_px(cr, 0, 0, 0.5, 0.5, True) == 12


# Corners in levels (nw, ne, sw, se), all non-planar. At pct <= 100 each
# triangle's screen slope in fx is >= 16px per tile, so a rounded pixel recovers fx to 0.05.
@pytest.mark.parametrize("levels", [(0, 1, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0), (1, 1, 1, 0)])
@pytest.mark.parametrize("pct", [50, 100])
def test_nwse_point_inverse_round_trips(levels, pct):
    proj = ig.canvas_size_and_origin(3, 3, 64, 0, ig.MAX_ELEVATION, elev_step_pct=pct, corner_headroom_steps=1)
    corners = tuple(level * proj.elev_step for level in levels)
    cr = np.zeros((4, 4), dtype=np.int64)
    cr[1, 1], cr[1, 2], cr[2, 1], cr[2, 2] = corners
    fold = np.zeros((3, 3), dtype=bool)
    fold[1, 1] = True
    for fx, fy in ((0.7, 0.2), (0.2, 0.7), (0.5, 0.5), (0.3, 0.3), (0.9, 0.6), (0.4, 0.95)):
        rise = ig.unit_rise_px(cr, 1, 1, fx, fy, True)
        sx, sy = ig.map_point_to_screen(1 + fx, 1 + fy, rise, proj)
        got = ig.screen_to_map_point(sx, sy, "sloped", proj=proj, corner_rise=cr, tile=(1, 1), fold=fold)
        assert got is not None, f"no inverse at ({fx}, {fy}), corners={corners}"
        gx, gy = got
        back = ig.map_point_to_screen(gx, gy, ig.unit_rise_px(cr, 1, 1, gx - 1, gy - 1, True), proj)
        assert abs(back[0] - sx) <= 1 and abs(back[1] - sy) <= 1
        assert abs(gx - (1 + fx)) < 0.05 and abs(gy - (1 + fy)) < 0.05
