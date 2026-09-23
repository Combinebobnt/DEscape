"""View > Range Rings' Qt-free geometry (GH #49, slice 1).

The projection property the whole feature rests on is that a TILE-space
circle lands on screen as a 2:1 ellipse, so that is asserted directly here
rather than eyeballed: nothing in descape/range_overlay.py draws an ellipse.
The Qt half -- which entry gets a ring, when the item is built and cleared --
is tests/test_unit_selection_viewer.py's job.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from descape import iso_geometry, range_overlay

_CASTLE_CONST = 82  # 4x4, range 8.0
_HOUSE_CONST = 70  # 2x2, range 0.0
_ARCHER_CONST = 4  # range 4.0, but not a building
_WATCH_TOWER_CONST = 79  # 1x1, range 8.0

_TILE_PX = 64
_MAP = 64


def _proj(max_elev: int = 7) -> iso_geometry.IsoProjection:
    return iso_geometry.canvas_size_and_origin(_MAP, _MAP, _TILE_PX, 0, max_elev)


# --- the radius rule ---------------------------------------------------------


def test_the_radius_is_measured_from_the_footprint_edge() -> None:
    assert range_overlay.ring_radius_tiles(8.0, (4, 4)) == 10.0
    assert range_overlay.ring_radius_tiles(4.0, (1, 1)) == 4.5


def test_only_a_building_with_a_range_gets_a_ring() -> None:
    assert range_overlay.ring_radius_for_const(_CASTLE_CONST) == 10.0
    assert range_overlay.ring_radius_for_const(_WATCH_TOWER_CONST) == 8.5
    assert range_overlay.ring_radius_for_const(_HOUSE_CONST) is None
    assert range_overlay.ring_radius_for_const(_ARCHER_CONST) is None


# --- sampling ----------------------------------------------------------------


@pytest.mark.parametrize("radius", [0.5, 4.5, 10.0, 13.0, 40.0])
def test_the_sample_count_is_a_multiple_of_eight(radius: float) -> None:
    """Both the tile-space axis extremes (theta 0 + k*90) and the screen
    extremes of the projected ellipse (45 + k*90) have to be sampled."""
    segments = range_overlay.ring_segments(radius)
    assert segments % range_overlay.SEGMENT_MULTIPLE == 0
    assert segments >= range_overlay.MIN_SEGMENTS


@pytest.mark.parametrize("style", ["flat", "stepped", "sloped"])
def test_the_polygon_closes_and_carries_the_requested_sample_count(style: str) -> None:
    """Implicitly closed, unit_pick.unit_polygons' own convention: no
    duplicated first point, and the last-to-first edge is the same chord as
    every other, so the subpath the caller closes has no seam."""
    points = range_overlay.ring_points(
        20.5, 20.5, 10.0, style, tile_px=_TILE_PX, proj=_proj(), rise_px=0
    )
    assert len(points) == range_overlay.ring_segments(10.0)
    assert points[0] != points[-1]
    # The closing edge is one more sample step, not a long seam back across
    # the ring. Compared against its own NEIGHBOUR rather than against every
    # chord: the projected ellipse's chords genuinely vary around it.
    closing = math.dist(points[-1], points[0])
    assert closing == pytest.approx(math.dist(points[0], points[1]), abs=1.5)


def test_an_unknown_style_raises() -> None:
    with pytest.raises(ValueError):
        range_overlay.ring_points(4.5, 4.5, 3.0, "isometric", tile_px=_TILE_PX)


# --- flat --------------------------------------------------------------------


def test_flat_is_a_true_circle_in_scene_space() -> None:
    """Flat's scene space is (mx * tile_px, my * tile_px), so the ring is a
    circle there and the view's own rotate-and-squash makes the ellipse."""
    cx, cy, radius = 20.5, 30.5, 10.0
    points = range_overlay.ring_points(cx, cy, radius, "flat", tile_px=_TILE_PX)
    centre = (cx * _TILE_PX, cy * _TILE_PX)
    for point in points:
        assert math.dist(point, centre) == pytest.approx(radius * _TILE_PX, abs=1e-6)


def test_flat_without_tile_px_raises() -> None:
    with pytest.raises(ValueError):
        range_overlay.ring_points(4.5, 4.5, 3.0, "flat", proj=_proj())


def test_stepped_without_proj_raises() -> None:
    with pytest.raises(ValueError):
        range_overlay.ring_points(4.5, 4.5, 3.0, "stepped", tile_px=_TILE_PX)


# --- the isometric styles ----------------------------------------------------


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_the_projected_ring_has_a_two_to_one_bounding_box(style: str) -> None:
    """iso_geometry.half_dims() derives half_w as exactly 2 * half_h, so this
    ratio is exact up to the per-sample rounding."""
    proj = _proj()
    points = range_overlay.ring_points(30.5, 30.5, 10.0, style, proj=proj, rise_px=0)
    xs = [x for x, _y in points]
    ys = [y for _x, y in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    assert width == pytest.approx(2 * height, abs=2.0)


def test_a_stepped_sample_round_trips_back_to_the_requested_tile_radius() -> None:
    """The ring's distance is in TILES, on the ground plane, and survives the
    projection: every sample inverts to a point exactly `radius` tiles from
    the centre."""
    proj = _proj(max_elev=0)
    elevations = np.zeros((_MAP, _MAP), dtype=np.int16)
    cx, cy, radius = 30.5, 30.5, 10.0
    for sx, sy in range_overlay.ring_points(cx, cy, radius, "stepped", proj=proj, rise_px=0):
        solved = iso_geometry.screen_to_map_point(
            int(sx), int(sy), "stepped", proj=proj, elevations=elevations
        )
        assert solved is not None
        assert math.dist(solved, (cx, cy)) == pytest.approx(radius, abs=0.1)


def test_a_sloped_sample_round_trips_back_to_the_requested_tile_radius() -> None:
    proj = _proj(max_elev=0)
    corner_rise = np.zeros((_MAP + 1, _MAP + 1), dtype=np.int32)
    cx, cy, radius = 30.5, 30.5, 10.0
    segments = range_overlay.ring_segments(radius)
    for i, (sx, sy) in enumerate(range_overlay.ring_points(cx, cy, radius, "sloped", proj=proj)):
        theta = 2.0 * math.pi * i / segments
        tile = (int(cx + radius * math.cos(theta)), int(cy + radius * math.sin(theta)))
        solved = iso_geometry.screen_to_map_point(
            int(sx), int(sy), "sloped", proj=proj, corner_rise=corner_rise, tile=tile
        )
        assert solved is not None
        assert math.dist(solved, (cx, cy)) == pytest.approx(radius, abs=0.1)


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_rise_lifts_the_whole_ring_by_exactly_that_many_pixels(style: str) -> None:
    """One rise for the whole ring (the building's own), not one per sample:
    a multi-tile building is a flat slab at one height."""
    proj = _proj()
    ground = range_overlay.ring_points(30.5, 30.5, 10.0, style, proj=proj, rise_px=0)
    raised = range_overlay.ring_points(30.5, 30.5, 10.0, style, proj=proj, rise_px=20)
    assert len(ground) == len(raised)
    for (gx, gy), (rx, ry) in zip(ground, raised, strict=True):
        assert rx == gx
        assert gy - ry == 20
