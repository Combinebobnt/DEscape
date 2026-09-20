"""Verifies iso_geometry.screen_to_map_point -- Stage 2 of free (non-snapped)
unit placement, the inverse of map_point_to_screen.

The load-bearing check is the Sloped sweep, and it is built the way
tests/test_sloped_unit_height.py builds its own: all 81 corner configurations
(3 elevation deltas x 4 corners), swept over the whole unit square, at three
elev_step_pct stops. For each painted point it maps FORWARD through
unit_rise_px + map_point_to_screen and then back through the inverse, and
requires the recovered point to reproduce the same pixel.

**Same pixel, not the same fractions, and that is the honest bar.** The
forward map lands on integer pixels, so it is many-to-one: several (fx, fy)
inside one tile paint the same pixel and no inverse can tell them apart.
Asserting the fractions came back identical would be asserting something
false about the geometry; asserting the pixel does is exactly what the caller
needs, since the caller is placing a unit under a cursor.

The degenerate denominator is checked by construction rather than hoped
absent. 2*half_h + ne - sw == 0 is reachable at elev_step_pct = 200
(half_w = 2*half_h, and one level of drop from sw to ne then equals
-2*half_h), which is the same stop at which _sloped_column_runs' max(raw_len,
1) clamp becomes reachable. The inverse must return None there, not a wrong
answer, and a test below pins that it really does hit that branch.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from descape import iso_geometry as ig

MAP_W = MAP_H = 4
TILE_PX = 64
_DELTAS = (0, 1, 2)
# Coarse enough to stay fast at 81 configs x 3 stops, fine enough to sample
# both triangles and every edge and tip.
_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def _proj(elev_step_pct: int) -> ig.IsoProjection:
    return ig.canvas_size_and_origin(
        MAP_W, MAP_H, TILE_PX,
        ig.MIN_ELEVATION, ig.MAX_ELEVATION,
        elev_step_pct=elev_step_pct,
        corner_headroom_steps=1,
    )


def _corner_rise(combo: tuple[int, int, int, int], tx: int, ty: int) -> np.ndarray:
    """A (MAP_H+1, MAP_W+1) corner field that is flat everywhere except the
    four corners of tile (tx, ty), which carry `combo` as (nw, ne, sw, se)."""
    field = np.zeros((MAP_H + 1, MAP_W + 1), dtype=np.int64)
    field[ty, tx], field[ty, tx + 1] = combo[0], combo[1]
    field[ty + 1, tx], field[ty + 1, tx + 1] = combo[2], combo[3]
    return field


# --- Flat and Stepped -------------------------------------------------


def test_flat_is_a_division():
    assert ig.screen_to_map_point(128, 64, "flat", tile_px=32) == (4.0, 2.0)
    assert ig.screen_to_map_point(140, 64, "flat", tile_px=32) == (140 / 32, 2.0)


def test_flat_without_a_tile_px_refuses_rather_than_guessing():
    assert ig.screen_to_map_point(128, 64, "flat") is None


def test_stepped_round_trips_every_pixel_of_a_tile():
    proj = _proj(50)
    elevations = np.zeros((MAP_H, MAP_W), dtype=np.int64)
    elevations[2, 1] = 3
    checked = same_tile = 0
    for ty in range(MAP_H):
        for tx in range(MAP_W):
            e = int(elevations[ty, tx])
            for fx, fy in itertools.product(_FRACTIONS, repeat=2):
                sx, sy = ig.map_point_to_screen(tx + fx, ty + fy, e * proj.elev_step, proj)
                got = ig.screen_to_map_point(sx, sy, "stepped", proj, elevations=elevations)
                resolved = ig.screen_to_tile(sx, sy, elevations, proj)
                if got is None:
                    # Only legitimate where screen_to_tile itself has no
                    # answer; a diamond's tip can fall on a neighbour, and a
                    # raised tile's diamond genuinely covers a lower one.
                    assert resolved is None
                    continue
                # Round-tripped at the elevation the INVERSE itself used, not
                # at the one this loop started from: where a raised tile's
                # diamond covers a lower neighbour, the visible point really
                # is the raised tile's, and that is the answer a click wants.
                back = ig.map_point_to_screen(
                    got[0], got[1],
                    int(elevations[resolved[1], resolved[0]]) * proj.elev_step, proj,
                )
                assert abs(back[0] - sx) <= 1 and abs(back[1] - sy) <= 1
                checked += 1
                same_tile += resolved == (tx, ty)
    assert checked > 100, f"only {checked} stepped points round-tripped; the sweep went vacuous"
    assert same_tile > 100, (
        f"only {same_tile} points resolved back to their own tile -- the sweep is "
        "measuring occlusion, not the inverse"
    )


def test_stepped_off_map_is_none():
    proj = _proj(50)
    elevations = np.zeros((MAP_H, MAP_W), dtype=np.int64)
    assert ig.screen_to_map_point(0, 0, "stepped", proj, elevations=elevations) is None


def test_an_unknown_style_raises_rather_than_returning_none():
    with pytest.raises(ValueError):
        ig.screen_to_map_point(0, 0, "isometric", _proj(50))


# --- Sloped: the 81-configuration sweep -------------------------------


def _sweep(elev_step_pct: int) -> tuple[int, int, int]:
    """(round_tripped, refused, degenerate) counts over all 81 corner
    configurations at this stop."""
    proj = _proj(elev_step_pct)
    elev_step = proj.elev_step
    tx, ty = 1, 1
    ok = refused = degenerate = 0
    for combo in itertools.product([d * elev_step for d in _DELTAS], repeat=4):
        field = _corner_rise(combo, tx, ty)
        ne, sw = combo[1], combo[2]
        is_degenerate = 2 * proj.half_h + ne - sw <= 0
        for fx, fy in itertools.product(_FRACTIONS, repeat=2):
            rise = ig.unit_rise_px(field, tx, ty, fx, fy)
            sx, sy = ig.map_point_to_screen(tx + fx, ty + fy, rise, proj)
            got = ig.screen_to_map_point(
                sx, sy, "sloped", proj, corner_rise=field, tile=(tx, ty)
            )
            if got is None:
                if is_degenerate:
                    degenerate += 1
                else:
                    refused += 1
                continue
            gx, gy = got
            back_rise = ig.unit_rise_px(field, int(gx), int(gy), gx - int(gx), gy - int(gy))
            bx, by = ig.map_point_to_screen(gx, gy, back_rise, proj)
            assert abs(bx - sx) <= 1 and abs(by - sy) <= 1, (
                f"pct={elev_step_pct} corners={combo} (fx, fy)=({fx}, {fy}): "
                f"recovered {got} paints ({bx}, {by}), not ({sx}, {sy})"
            )
            ok += 1
    return ok, refused, degenerate


@pytest.mark.parametrize("elev_step_pct", [25, 50, 200])
def test_sloped_inverse_recovers_the_painting_point(elev_step_pct: int):
    ok, refused, _degenerate = _sweep(elev_step_pct)
    assert ok > 500, f"only {ok} points inverted at pct={elev_step_pct}; the sweep went vacuous"
    # A non-degenerate configuration must never be refused: an inverse that
    # bails reads to the caller as "fall back to the snapped position", which
    # is a silent feature regression rather than a visible failure.
    assert refused == 0, f"{refused} non-degenerate points were refused at pct={elev_step_pct}"


def test_the_degenerate_denominator_is_reached_and_refused_at_pct_200():
    """Not a hypothetical branch: half_w = 2*half_h and elev_step scales with
    the stop, so at 200 one level of drop from sw to ne makes the denominator
    exactly zero. Without this, the guard could be dead code and the sweep
    above would pass just as happily."""
    proj = _proj(200)
    assert 2 * proj.half_h - proj.elev_step <= 0, (
        "pct=200 no longer produces a degenerate configuration; this test needs re-deriving"
    )
    _ok, _refused, degenerate = _sweep(200)
    assert degenerate > 0, "the degenerate branch was never taken, so the guard is untested"


def test_a_degenerate_tile_returns_none_rather_than_a_wrong_point():
    proj = _proj(200)
    tx, ty = 1, 1
    field = _corner_rise((0, 0, proj.elev_step, 0), tx, ty)  # sw above ne
    assert 2 * proj.half_h + 0 - proj.elev_step <= 0
    assert ig.screen_to_map_point(
        proj.origin_x + 2 * proj.half_w, proj.origin_y, "sloped", proj,
        corner_rise=field, tile=(tx, ty),
    ) is None


def test_sloped_without_a_resolved_tile_is_none():
    proj = _proj(50)
    field = np.zeros((MAP_H + 1, MAP_W + 1), dtype=np.int64)
    assert ig.screen_to_map_point(100, 100, "sloped", proj, corner_rise=field, tile=None) is None
