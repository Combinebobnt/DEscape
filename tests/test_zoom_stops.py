"""TASK-600: the wheel zoom's level lattice (descape/zoom_stops.py) on every
ladder shape a cache is built with. Pure, no Qt."""

from __future__ import annotations

import itertools
import math

import pytest

from descape import iso_geometry, zoom_stops

# (name, base tile_px, the ladder's tile_px values)
LADDERS = [
    ("base 64 Default", 64, iso_geometry.mip_tile_px_candidates(64, 128, native_tier=True)),
    ("base 32 Default", 32, iso_geometry.mip_tile_px_candidates(32, 128, native_tier=True)),
    ("base 32 Potato", 32, iso_geometry.mip_tile_px_candidates(32, 64, native_tier=True)),
    ("base 16 Potatest", 16, iso_geometry.mip_tile_px_candidates(16, 32, native_tier=True)),
    ("top-down Flat", 64, iso_geometry.mip_tile_px_candidates(64, 128)),
    ("single level", 16, {0: 16}),
]
LO, HI = 0.01, 40.0


def _lattice(base, ladder):
    stops = zoom_stops.level_stops(px / base for px in ladder.values())
    return stops, zoom_stops.lattice(stops, LO, HI)


@pytest.mark.parametrize("name, base, ladder", LADDERS, ids=[row[0] for row in LADDERS])
def test_every_level_stop_is_a_point_and_every_step_is_a_notch(name, base, ladder) -> None:
    stops, points = _lattice(base, ladder)
    assert len(stops) == len(ladder)
    for stop in stops:
        assert stop in points, f"{name}: the {stop * base:g} px stop is no point"
    assert points == sorted(points) and len(set(points)) == len(points)
    pairs = list(itertools.pairwise(points))
    on_ladder = [b / a for a, b in pairs if stops[0] <= a < stops[-1]]
    assert all(1.2 <= s <= 1.34 for s in on_ladder), (name, on_ladder)
    # Off the ladder: octaves split in three, out to the bounds.
    assert all(b / a == pytest.approx(2 ** (1 / 3), rel=1e-12) for a, b in pairs if not stops[0] <= a < stops[-1])
    assert LO <= points[0] < LO * 2 ** (1 / 3) and HI / 2 ** (1 / 3) < points[-1] <= HI


def test_the_base_64_default_split() -> None:
    """The plan's Decision 1 numbers: 16 -> 32 -> 64 in three steps each, 64 -> 96 in two, 96 -> 128 in one."""
    stops, points = _lattice(64, LADDERS[0][2])
    index = [points.index(s) for s in stops]
    assert [b - a for a, b in itertools.pairwise(index)] == [3, 3, 2, 1]


@pytest.mark.parametrize("name, base, ladder", LADDERS, ids=[row[0] for row in LADDERS])
def test_in_then_out_retraces_the_same_scales(name, base, ladder) -> None:
    _stops, points = _lattice(base, ladder)
    for start in points[1:-1]:
        for n in (1, 2, 3):
            there = zoom_stops.notch_target(start, n, points)
            if there is None:
                continue
            assert zoom_stops.notch_target(there, -n, points) == start, (name, start, n)
            back = zoom_stops.notch_target(start, -n, points)
            if back is not None:
                assert zoom_stops.notch_target(back, n, points) == start, (name, start, -n)


def test_a_point_moves_to_its_neighbour_and_the_ends_return_none() -> None:
    _stops, points = _lattice(64, LADDERS[0][2])
    for i, p in enumerate(points):
        assert zoom_stops.notch_target(p, 1, points) == (points[i + 1] if i + 1 < len(points) else None)
        assert zoom_stops.notch_target(p, -1, points) == (points[i - 1] if i else None)
        # A float multiply's drift still reads as on the point.
        assert zoom_stops.notch_target(p * (1 + 1e-14), 1, points) == (points[i + 1] if i + 1 < len(points) else None)
    assert zoom_stops.notch_target(points[3], 0, points) is None
    assert zoom_stops.notch_target(1.0, 1, []) is None


@pytest.mark.parametrize("name, base, ladder", LADDERS, ids=[row[0] for row in LADDERS])
def test_an_off_lattice_start_moves_at_least_half_a_notch(name, base, ladder) -> None:
    """Decision 2: at least NOTCH_STEP ** 0.5, at most that times the widest lattice step."""
    _stops, points = _lattice(base, ladder)
    widest = max(b / a for a, b in itertools.pairwise(points))
    lo_move, hi_move = zoom_stops.MIN_OFF_LATTICE_STEP, zoom_stops.MIN_OFF_LATTICE_STEP * widest
    checked = 0
    for start in [points[0] * 1.6 ** (k / 37) for k in range(1, 37 * int(math.log(points[-1] / points[0], 1.6)))]:
        if zoom_stops.on_point(start, points) is not None:
            continue
        up, down = zoom_stops.notch_target(start, 1, points), zoom_stops.notch_target(start, -1, points)
        if up is not None:
            assert lo_move * (1 - 1e-12) <= up / start <= hi_move * (1 + 1e-12), (name, start, up)
            checked += 1
        if down is not None:
            assert lo_move * (1 - 1e-12) <= start / down <= hi_move * (1 + 1e-12), (name, start, down)
            checked += 1
        if up is not None and down is not None:
            assert zoom_stops.on_point(up, points) is not None and zoom_stops.on_point(down, points) is not None
    assert checked > 100, f"{name}: only {checked} off-lattice moves checked"


def test_level_stops_merge_duplicates_and_sort() -> None:
    assert zoom_stops.level_stops([2.0, 0.5, 1.0, 1.0 * (1 + 1e-13), 0.0]) == [0.5, 1.0, 2.0]
    assert zoom_stops.lattice([], LO, HI) == []
