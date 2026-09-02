"""draw-perf seed-dilation plan Step 4: the test today's suite does not have.

Per that plan's fact 8, no default-tier test pins the constraint the Step 3
narrowing has to satisfy -- every existing sprite/unit fixture edits a unit's
OWN tile, and test_sloped_edit.py's own ring check runs with_units=False. A
narrowing that is correct for Stepped but under-covers Sloped's corner
sharing would ship green on the default tier without this file.

Worked geometry, using the Mill BUILDING_TILE_SPANS/tests already share
(MILL_CONST, span 2x2), anchored at (60.0, 60.0):
  _span_start(60.0, 2) == (round(120) - 2 + 1) // 2 == 59, so the Mill's
  footprint is tiles {59, 60} x {59, 60}, own tile (60, 60).

Edit elevation at (61, 61) -- strictly outside the footprint, but inside the
anchor's 3x3 ring. It shares exactly one corner_rise entry, [61, 61], with
tile (60, 60), so under Sloped it moves the Mill's rise and therefore all
four of its diamonds; three of those four -- (59, 59), (59, 60), (60, 59) --
lie outside dilate({(61, 61)}, 1), so a floor-1-only seed under-covers them.
unit_band_radius=1 on the Sloped wrapper is what makes the bbox reach them.

Stepped is the CONTROL, not a failure case: (61, 61) is not the Mill's own
tile, so under Stepped the footprint genuinely does not move and the bbox
must NOT widen to include (59, 59) -- the direction that stops this test
from passing for the wrong reason (i.e. old UNIT_FOOTPRINT_MAX_RADIUS-style
over-widening still in effect)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from descape import iso_geometry
from descape.elevation_tools import set_tile_elevation
from descape.render import (
    dirty_screen_bbox_iso,
    dirty_screen_bbox_sloped,
    elevations_and_proj,
    render_terrain_iso_with_proj,
    render_terrain_sloped,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import IsoChunkCache, SlopedChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.terrain_palette import BUILDING_TILE_SPANS

MILL_CONST = 68  # BUILDING_TILE_SPANS[68] == (2, 2)
assert BUILDING_TILE_SPANS[MILL_CONST] == (2, 2)

BASE_ELEVATION = 5
MILL_TILE = (60.0, 60.0)
EDIT_TILE = (61, 61)
MILL_FOOTPRINT = {(59, 59), (59, 60), (60, 59), (60, 60)}
# The three footprint tiles the floor-1 ring around EDIT_TILE alone (radius
# 1, i.e. {60,61,62} x {60,61,62}) does NOT already cover -- these are the
# ones that discriminate a correct unit_band_radius=1 from a missing one.
UNCOVERED_BY_FLOOR_1 = {(59, 59), (59, 60), (60, 59)}
assert UNCOVERED_BY_FLOOR_1 < MILL_FOOTPRINT


@dataclass
class Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


def _scenario():
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for t in scenario.map_manager.terrain:
        t.elevation = BASE_ELEVATION
    scenario.unit_manager.units[1].append(Unit(*MILL_TILE, MILL_CONST))
    return scenario


def _edit_and_dirty(mm):
    """A single-tile elevation raise at EDIT_TILE. Uniform starting
    elevation keeps _elevation_tile_recursion from cascading, so the dirty
    set stays exactly {EDIT_TILE} -- load-bearing for this fixture: a wider
    dirty set would let the floor-1 ring alone reach the footprint tiles by
    accident and the test would pass without unit_band_radius doing anything."""
    ex, ey = EDIT_TILE
    set_tile_elevation(mm, ex, ey, BASE_ELEVATION + 1)
    dirty = [i for i, t in enumerate(mm.terrain) if (t.x, t.y) == EDIT_TILE]
    assert len(dirty) == 1, "fixture must produce exactly one dirty tile"
    return dirty


def _swept_bounds_clamped(x, y, proj, canvas_w, canvas_h):
    tx0, ty0, tx1, ty1 = iso_geometry.tile_screen_bounds_swept(x, y, proj)
    tx0, ty0 = max(0, tx0), max(0, ty0)
    tx1, ty1 = min(canvas_w, tx1), min(canvas_h, ty1)
    return tx0, ty0, tx1, ty1


def _contains(bbox, inner) -> bool:
    x0, y0, x1, y1 = bbox
    ix0, iy0, ix1, iy1 = inner
    if ix1 <= ix0 or iy1 <= iy0:
        return True  # a degenerate (off-canvas) inner rect imposes no requirement
    return x0 <= ix0 and y0 <= iy0 and x1 >= ix1 and y1 >= iy1


def test_sloped_bbox_reaches_the_mills_far_corner():
    scenario = _scenario()
    mm = scenario.map_manager
    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
    canvas_w, canvas_h = proj.canvas_w, proj.canvas_h

    dirty = _edit_and_dirty(mm)
    bbox = dirty_screen_bbox_sloped(scenario, dirty, elevations, proj, with_units=True)
    assert bbox is not None

    for fx, fy in UNCOVERED_BY_FLOOR_1:
        inner = _swept_bounds_clamped(fx, fy, proj, canvas_w, canvas_h)
        assert _contains(bbox, inner), (
            f"Sloped bbox {bbox} does not reach footprint tile ({fx}, {fy}) -- "
            f"unit_band_radius=1 should have pulled the Mill's own tile into the seed"
        )


def test_stepped_bbox_does_not_widen_for_a_non_own_tile_edit():
    """The control: EDIT_TILE is not the Mill's own tile, so Stepped's
    footprint union must not trigger at all. A bbox that reaches (59, 59)
    here would mean the narrowing regressed to something as wide as the old
    blanket dilation, passing the sibling test for the wrong reason."""
    scenario = _scenario()
    mm = scenario.map_manager
    elevations, proj = elevations_and_proj(scenario)
    canvas_w, canvas_h = proj.canvas_w, proj.canvas_h

    dirty = _edit_and_dirty(mm)
    bbox = dirty_screen_bbox_iso(scenario, dirty, elevations, proj, with_units=True)
    assert bbox is not None

    far_corner = _swept_bounds_clamped(59, 59, proj, canvas_w, canvas_h)
    assert not _contains(bbox, far_corner), (
        f"Stepped bbox {bbox} widened to cover ({59, 59}) for an edit that "
        f"never touched the Mill's own tile -- the narrowing is not narrowing"
    )


def test_sloped_patch_matches_a_fresh_full_render():
    scenario = _scenario()
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
    cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px)
    canvas_w, canvas_h = cache.canvas_dims()
    cache.render_rect(0, 0, canvas_w, canvas_h)

    dirty = _edit_and_dirty(mm)
    elevation_changed: set = set()
    bbox = dirty_screen_bbox_sloped(
        scenario, dirty, elevations, proj, with_units=True, elevation_changed=elevation_changed
    )
    assert bbox is not None
    cache.patch(bbox, elevation_changed=elevation_changed)

    stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
    full = render_terrain_sloped(scenario, with_units=True)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])


def test_stepped_patch_matches_a_fresh_full_render():
    scenario = _scenario()
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, proj = elevations_and_proj(scenario)
    cache = IsoChunkCache(scenario, elevations, proj, tile_px)
    canvas_w, canvas_h = cache.canvas_dims(0)
    cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)

    dirty = _edit_and_dirty(mm)
    elevation_changed: set = set()
    bbox = dirty_screen_bbox_iso(
        scenario, dirty, elevations, proj, with_units=True, elevation_changed=elevation_changed
    )
    assert bbox is not None
    cache.patch(bbox, elevation_changed=elevation_changed)

    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    full, _full_elev, full_proj = render_terrain_iso_with_proj(scenario, with_units=True)
    assert np.array_equal(stitched, full[:canvas_h, :canvas_w])
