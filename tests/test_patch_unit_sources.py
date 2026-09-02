"""A draw-perf invariant: an edit patch() must never rebuild per-edit
unit-derived state (units_by_tile/building_bboxes/
sprites) for a terrain-only edit, and must always rebuild it for an elevation
edit under a unit's own tile -- both directions, since only the second one is
a correctness requirement and it's trivially satisfied by never optimizing
anything at all.

Built on BLANK_TEMPLATE_PATH (tracked, 120x120) with duck-typed units
appended, the tests/test_sprite_edit_bbox.py / tests/test_sprite_chunks.py
pattern -- NOT examples/blank_map.aoe2scenario, which is untracked corpus
and would push this into the corpus tier for no reason.

Two traps this file exists to catch:
- `_building_bboxes_iso` skips any unit with `span_x <= 1 and span_y <= 1`,
  so a single-tile duck-typed unit never enters building_bboxes -- the "must
  rebuild" direction would pass for the wrong reason with only that fixture.
  `MILL_CONST` (BUILDING_TILE_SPANS[68] == (2, 2), see
  tests/test_unit_footprints.py) is a real multi-tile entry.
- A buildings-only gate would pass every test above it and fail only
  `test_elevation_under_1x1_unit_rebuilds_with_sprites`, since
  `sprite_draws_by_anchor` reads elevation for EVERY unit, 1x1 included.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from descape.elevation_tools import set_tile_elevation
from descape.render import (
    dirty_screen_bbox_iso,
    dirty_screen_bbox_sloped,
    elevations_and_proj,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import IsoChunkCache, SlopedChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.terrain_palette import BUILDING_TILE_SPANS

MILL_CONST = 68  # BUILDING_TILE_SPANS[68] == (2, 2) -- a real multi-tile building
UNIT_1X1_CONST = 999999  # synthetic: resolves to no graphic, no foundation terrain -- see
# sprite_draws_by_anchor's own docstring ("a unit that resolves nowhere is
# simply absent from every field here"), so this never needs a real install.
assert MILL_CONST in BUILDING_TILE_SPANS and BUILDING_TILE_SPANS[MILL_CONST] != (1, 1)
assert UNIT_1X1_CONST not in BUILDING_TILE_SPANS

BASE_ELEVATION = 5
BUILDING_TILE = (60, 60)
FAR_TILE = (10, 10)


@dataclass
class Unit:
    """The attributes _units_by_tile/_building_bboxes_iso/sprite_draws_by_anchor
    actually read -- duck-typed, matching tests/test_sprite_edit_bbox.py's own
    Unit fixture."""

    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


def _scenario(unit_const: int, tile: tuple[int, int] = BUILDING_TILE):
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for t in scenario.map_manager.terrain:
        t.elevation = BASE_ELEVATION
    scenario.unit_manager.units[1].append(Unit(float(tile[0]), float(tile[1]), unit_const))
    return scenario


def _make_cache(style: str, scenario, sprites: bool = False):
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=sprites)
    else:
        elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
        cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, sprites=sprites)
    cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0)
    return cache, elevations, proj


def _rebuild_snapshot(style: str, cache):
    """A proxy for "did source state actually get rebuilt", different per
    class because they expose the fact differently: IsoChunkCache gates a
    LAZY per-level rebuild behind _source_gen (see its own _level()); Sloped
    has no laziness and just reassigns the three objects directly -- see
    render_cache.py's own docstrings for both."""
    if style == "stepped":
        return cache._source_gen
    return (id(cache.corner_rise), id(cache.building_bboxes), id(cache.sprites))


def _bbox_fn(style):
    return dirty_screen_bbox_iso if style == "stepped" else dirty_screen_bbox_sloped


def _paint_and_bbox(style, scenario, cache, elevations, proj, tx, ty, terrain_id):
    """A pure terrain-paint edit: no elevation change at all."""
    mm = scenario.map_manager
    tile = mm.get_tile(tx, ty)
    tile.terrain_id = terrain_id
    tile.layer = -1
    dirty_indices = [i for i, t in enumerate(mm.terrain) if t.x == tx and t.y == ty]
    elevation_changed = set()
    bbox = _bbox_fn(style)(
        scenario, dirty_indices, elevations, proj, with_units=True,
        with_sprites=cache.sprites_enabled, elevation_changed=elevation_changed,
    )
    return bbox, elevation_changed


def _elevation_edit_and_bbox(style, scenario, cache, elevations, proj, ex, ey, new_elev):
    mm = scenario.map_manager
    before = {(t.x, t.y): t.elevation for t in mm.terrain}
    set_tile_elevation(mm, ex, ey, new_elev)
    dirty_indices = [i for i, t in enumerate(mm.terrain) if before[(t.x, t.y)] != t.elevation]
    assert dirty_indices, "set_tile_elevation produced no change -- fixture is broken"
    elevation_changed = set()
    bbox = _bbox_fn(style)(
        scenario, dirty_indices, elevations, proj, with_units=True,
        with_sprites=cache.sprites_enabled, elevation_changed=elevation_changed,
    )
    return bbox, elevation_changed


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_terrain_only_patch_does_not_rebuild_unit_sources(style):
    scenario = _scenario(MILL_CONST)
    cache, elevations, proj = _make_cache(style, scenario)
    before = _rebuild_snapshot(style, cache)

    bbox, elevation_changed = _paint_and_bbox(style, scenario, cache, elevations, proj, *FAR_TILE, terrain_id=5)
    assert not elevation_changed, "fixture edit was not terrain-only"
    assert bbox is not None
    cache.patch(bbox, elevation_changed=elevation_changed)

    assert _rebuild_snapshot(style, cache) == before, "a terrain-only patch rebuilt unit sources"


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_elevation_under_building_rebuilds(style):
    scenario = _scenario(MILL_CONST)
    cache, elevations, proj = _make_cache(style, scenario)
    before = _rebuild_snapshot(style, cache)

    bbox, elevation_changed = _elevation_edit_and_bbox(
        style, scenario, cache, elevations, proj, *BUILDING_TILE, new_elev=BASE_ELEVATION + 1
    )
    assert elevation_changed
    cache.patch(bbox, elevation_changed=elevation_changed)

    assert _rebuild_snapshot(style, cache) != before, (
        "an elevation edit under a building did not rebuild unit sources"
    )


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_elevation_far_from_any_unit_does_not_rebuild(style):
    """Stepped-only: IsoChunkCache's gate (3b) is "any unit's OWN tile", so an
    elevation edit nowhere near one must skip the bump. SlopedChunkCache
    deliberately uses one COARSER gate for corner_rise/building_bboxes/
    sprites together -- "any elevation changed at all" -- since corner_rise
    feeds every tile's own shading, not just tiles under units (see
    SlopedChunkCache._refresh_source_caches's own docstring); the sibling
    test below pins that a far-from-any-unit edit still rebuilds there."""
    scenario = _scenario(MILL_CONST)
    cache, elevations, proj = _make_cache(style, scenario)
    before = _rebuild_snapshot(style, cache)

    bbox, elevation_changed = _elevation_edit_and_bbox(
        style, scenario, cache, elevations, proj, *FAR_TILE, new_elev=BASE_ELEVATION + 1
    )
    assert elevation_changed
    assert not (elevation_changed & cache.units_by_tile.keys()), (
        "fixture edit reached the building's own tile -- pick a tile further away"
    )
    cache.patch(bbox, elevation_changed=elevation_changed)

    after = _rebuild_snapshot(style, cache)
    if style == "stepped":
        assert after == before, "an elevation edit nowhere near a unit rebuilt unit sources anyway"
    else:
        assert after != before, (
            "SlopedChunkCache's single gate should still rebuild on ANY elevation change"
        )


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_elevation_under_1x1_unit_rebuilds_with_sprites(style):
    """The trap a buildings-only gate would miss (see this module's own
    docstring): with sprites OFF this would pass under a buildings-only gate
    too, since building_bboxes correctly skips a 1x1 unit either way --
    sprites must be on to distinguish the two gates at all."""
    scenario = _scenario(UNIT_1X1_CONST)
    cache, elevations, proj = _make_cache(style, scenario, sprites=True)
    before = _rebuild_snapshot(style, cache)

    bbox, elevation_changed = _elevation_edit_and_bbox(
        style, scenario, cache, elevations, proj, *BUILDING_TILE, new_elev=BASE_ELEVATION + 1
    )
    assert elevation_changed
    cache.patch(bbox, elevation_changed=elevation_changed)

    assert _rebuild_snapshot(style, cache) != before, (
        "an elevation edit under a 1x1 unit did not rebuild sources with sprites on"
    )
